#include "saber/core/tensor_op.h"
#ifdef USE_ARM_PLACE
#include "saber/core/tensor_op.h"
#include "saber/funcs/timer.h"
#include "test/saber/test_saber_func.h"
#include "saber/funcs/conv.h"
#include "saber/funcs/impl/arm/neon/impl/conv_arm_impl.h"
#include "saber/funcs/type_trans.h"
using namespace anakin::saber;



int g_cluster = 0;
int g_threads = 1;
int g_test_iter = 1;

bool g_basic_test = false;
bool g_compare_result = true;
bool g_flag_relu = false;
bool g_flag_bias = false;

int g_num = 1;
int g_chin = 4;
int g_h_in = 10;
int g_w_in = 10;

int g_ch_out = 4;
int g_group = 1;
int g_kw = 1;
int g_pad_w = 0;
int g_stride_w = 1;
int g_dila_w = 1;
int g_kh = 1;
int g_pad_h = 0;
int g_stride_h = 1;
int g_dila_h = 1;

typedef Tensor<ARM> TensorH;

/**
 * \brief basic direct convolution function
 */
//! for float, dtype1 and type2 is float
//! for int8, dytpe1 is char, dtype2 is int
template <typename Dtype1, typename Dtype2>
static void conv_basic(const Dtype1* din, Dtype2* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const Dtype1* weights, const Dtype2* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu) {

    Dtype2 beta = 0;
    auto src_data = din;
    auto dst_data_ref = dout;
    auto weights_data = weights;
    auto with_bias = flag_bias;
    auto bias_data = bias;

    int in_num = num;
    int out_channels = chout;
    int out_h = hout;
    int out_w = wout;

    int in_channel = chin;
    int in_h = hin;
    int in_w = win;
    int out_c_group = out_channels / group;
    int in_c_group = in_channel / group;

    for (int n = 0; n < in_num; ++n) {
#pragma omp parallel for collapse(4)
        for (int g = 0; g < group; ++g) {
            for (int oc = 0; oc < out_c_group; ++oc) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        int out_idx = n * group * out_c_group * out_h * out_w + g * out_c_group * out_h * out_w
                                      + oc * out_h * out_w + oh * out_w + ow;
                        Dtype2 bias_d = with_bias ? (bias_data[g * out_c_group + oc]) : (Dtype2)0;
                        dst_data_ref[out_idx] = bias_d;// + dst_data_ref[out_idx] * beta;
                        for (int ic = 0; ic < in_c_group; ++ic) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int iw = ow * stride_w - pad_w + kw * (dila_w);
                                    int ih = oh * stride_h - pad_h + kh * (dila_h);
                                    if (iw < 0 || iw >= in_w) continue;
                                    if (ih < 0 || ih >= in_h) continue;

                                    int iidx = n * in_channel * in_h * in_w
                                               + g * in_c_group * in_h * in_w
                                               + ic * in_h * in_w
                                               + ih * in_w
                                               + iw;
                                    int widx = g * out_c_group * in_c_group * kernel_h * kernel_w
                                               + oc * in_c_group * kernel_h * kernel_w
                                               + ic * kernel_h * kernel_w
                                               + kh * kernel_w
                                               + kw;

                                    dst_data_ref[out_idx]
                                            += src_data[iidx]
                                               * weights_data[widx];
                                }
                            }
                        }
                        if (flag_relu) {
                            dst_data_ref[out_idx] = dst_data_ref[out_idx] > (Dtype2)0 ? dst_data_ref[out_idx] : (Dtype2)0;
                        }
                    }
                }
            }
        }
    }
}

template <typename dtype>
static int count_diff(const dtype* src1, const dtype* src2, int size, double max_ratio, float tensor_scale) {
    double sum_abs1 = 0.0;
    double sum_abs2 = 0.0;
    for (int i = 0; i < size; ++i) {
        sum_abs1 += fabs(src1[i]);
        sum_abs2 += fabs(src2[i]);
    }
    double mean_abs1 = sum_abs1 / size;
    double mean_abs2 = sum_abs2 / size;
    double mean_val = (mean_abs1 + mean_abs2) / 2.0;
    if (max_ratio <= 0) {
        max_ratio = 0.1;
    }
    int count = 0;
    for (int i = 0; i < size; ++i) {
        double abs_diff = fabs(src1[i] - src2[i]);
        double ratio =  abs_diff / (fabs(src1[i] + src2[i]) + 1e-12);
        if (ratio > max_ratio && abs_diff > (tensor_scale + 1e-5f) && abs_diff > mean_val * 0.1f) {
            ++count;
        }
    }
    return count;
}

SaberStatus test_arm_conv_int8(int n, int c, int h, int w, \
    int ch_out, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h, \
    int dila_w, int dila_h, int group, bool is_bias, bool is_relu, int thread_num, int cluster_id) {

    double to = 0;
    double min_time = 1000000;
    SaberTimer<ARM> t1;

    Context<ARM> ctx1;
    PowerMode mode = static_cast<PowerMode>(cluster_id);
    ctx1.set_run_mode(mode, thread_num);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    TensorH tout_basic_int32;
    TensorH tout_basic_int8;
    TensorH tout_saber_int32;
    TensorH tout_saber_int8;
    TensorH tout_basic_fp32;
    TensorH tout_saber_fp32;

    TensorH thinf;
    TensorH thinc;
    Shape shin({n, c, h, w});
    thinf.re_alloc(shin, AK_FLOAT);
    thinc.re_alloc(shin, AK_INT8);

    int num = n;
    int chin = c;
    int hin = h;
    int win = w;

    LOG(INFO) << "conv param: ";
    LOG(INFO) << " img_num = " << num << " in_channels = " << chin << " img_h = " << hin << " img_w = " << win;
    LOG(INFO) << " ch_out = " << ch_out << " group = " << group
              << " kernel_w = " << kernel_w << " kernel_h = " << kernel_h;
    LOG(INFO) << " pad_width = " << pad_w << " pad_height = " << pad_h << \
        " stride_width = " << stride_w << " stride_height = " << stride_h << \
         " dilation_w = " << dila_w << " dilation_h = " << dila_h << \
         " bias flag = " << (is_bias? "true" : "false") << ", relu flag = " << (is_relu? "true" : "false");

    int kernel_exten = dila_h * (kernel_h - 1) + 1;
    int hout = (h + 2 * pad_h - kernel_exten) / stride_h + 1;

    kernel_exten = dila_w * (kernel_w - 1) + 1;
    int wout = (w + 2 * pad_w - kernel_exten) / stride_w + 1;

    if (hout <= 0 || wout <= 0) {
        return SaberSuccess;
    }

    Shape shape_out({num, ch_out, hout, wout});

    Shape shw({ch_out, chin / group, kernel_h, kernel_w});
    Shape shb({1, ch_out, 1, 1});

    TensorH pweihtf;
    TensorH pbiasf;

    TensorH pweihtc;
    TensorH pbiasi;

    pweihtf.re_alloc(shw, AK_FLOAT);
    //pbiasf.re_alloc(shb, AK_FLOAT);

    pweihtc.re_alloc(shw, AK_FLOAT);
    //pbiasi.re_alloc(shb, AK_INT32);

    fill_tensor_rand(thinf, -1.f, 1.f);
    fill_tensor_rand(pweihtf, -1.f, 1.f);
    // fill_tensor_const(thinf, 1.f);
    // fill_tensor_const(pweihtf, 1.f);

    LOG(INFO) << "get input scale";
    pweihtc.copy_from(pweihtf);
    //! convert input data type
    std::vector<float> scale;
    get_tensor_scale(thinf, scale, -1, 127.f);
    thinf.set_scale(scale);
    LOG(INFO) << "input tesnor scale at factor 127.f is " << thinf.get_scale()[0] << ", max_val: " << 127.f * thinf.get_scale()[0];

    trans_tensor_dtype<ARM, AK_FLOAT, AK_INT8>(thinf, thinc, scale[0], 1.f, {1.f});
    thinc.set_scale(scale);
//    print_tensor(thinf);
//    print_tensor(thinc);

    LOG(INFO) << "get weights scale";
    //! convert weight data type

    trans_weights_dtype<ARM>(pweihtc, AK_INT8, 127.f, CONV_TYPE, group);
    std::vector<float> w_scale = pweihtc.get_scale();
   // LOG(INFO) << "input tesnor scale at factor 127.f is ";
   // for (int j = 0; j < w_scale.size(); ++j) {
   //     LOG(INFO) << "|-- " << j << ": " << w_scale[j] << ", max_val: " << 127.f * w_scale[j];
   // }
    if (is_bias){
        pbiasf.re_alloc(shb, AK_FLOAT);
        pbiasi.re_alloc(shb, AK_INT32);
        fill_tensor_rand(pbiasf, -1.f, 1.f);
        trans_fp32_bias_to_int32(pbiasf, pbiasi, thinf.get_scale()[0], w_scale);
    }

//    print_tensor(pweihtf);
//    print_tensor(pweihtc);

    std::vector<float> scale_out = {1.f};
    tout_saber_int8.set_scale(scale_out);
    tout_basic_int8.set_scale(scale_out);

    //! get int8 and fp32 basic result
    if (g_compare_result) {
        LOG(INFO) << "run basic conv for precision comparation";
        const int8_t* dinc = static_cast<const int8_t*>(thinc.data());
        const int8_t* weightc = static_cast<const int8_t*>(pweihtc.data());
        const int* biasi = static_cast<const int*>(pbiasi.data());
        const float* dinf = static_cast<const float*>(thinf.data());
        const float* weightf = static_cast<const float*>(pweihtf.data());
        const float* biasf = static_cast<const float*>(pbiasf.data());
        tout_basic_fp32.re_alloc(shape_out, AK_FLOAT);
        tout_basic_int32.re_alloc(shape_out, AK_INT32);
        tout_basic_int8.re_alloc(shape_out, AK_INT8);

        float* dout_basic_fp32 = static_cast<float*>(tout_basic_fp32.mutable_data());
        int* dout_basic_int32 = static_cast<int*>(tout_basic_int32.mutable_data());

        memset(dout_basic_fp32, 0, sizeof(float) * tout_basic_fp32.valid_size());
        memset(dout_basic_int32, 0, sizeof(float) * tout_basic_int32.valid_size());

//        LOG(INFO) << "do basic fp32 conv";
//        conv_basic<float, float>(dinf, dout_basic_fp32, num, ch_out, hout, wout, chin, hin, win, \
//            weightf, biasf, group, kernel_w, kernel_h, stride_w, stride_h, \
//            dila_w, dila_h, pad_w, pad_h, is_bias, is_relu);

        LOG(INFO) << "do basic int8 conv, trans basic int32 to fp32";
        conv_basic<int8_t, int>(dinc, dout_basic_int32, num, ch_out, hout, wout, chin, hin, win, \
            weightc, biasi, group, kernel_w, kernel_h, stride_w, stride_h, \
            dila_w, dila_h, pad_w, pad_h, is_bias, is_relu);

        LOG(INFO) << "trans basic int32 to int8";
        trans_tensor_dtype<ARM, AK_INT32, AK_INT8>(tout_basic_int32, tout_basic_int8, thinf.get_scale()[0], tout_basic_int8.get_scale()[0], w_scale);
        LOG(INFO) << "trans basic int32 to fp32";
        trans_tensor_dtype<ARM, AK_INT32, AK_FLOAT>(tout_basic_int32, tout_basic_fp32, thinf.get_scale()[0], 1.f, w_scale);

//        print_tensor(tout_basic_fp32);
        // LOG(INFO) << "basic in32 result";
        // print_tensor(tout_basic_int32);
    }

    Conv<ARM, AK_INT8> conv_int8;
    Conv<ARM, AK_INT8> conv_int8_fp32;
    Conv<ARM, AK_INT8> conv_int8_int32;

    ConvParam<ARM> param(group, pad_h, pad_w, stride_h, stride_w, dila_h, dila_w, &pweihtc, &pbiasf);
    if (is_relu) {
        ActivationParam<ARM> act_param(Active_relu);
        param.activation_param = act_param;
    }
    std::vector<TensorH*> tvin_fp32;
    std::vector<TensorH*> tvin_int8;
    std::vector<TensorH*> tvout_saber_fp32;
    std::vector<TensorH*> tvout_saber_int32;
    std::vector<TensorH*> tvout_saber_int8;

    tvin_fp32.push_back(&thinf);
    tvin_int8.push_back(&thinc);
    tvout_saber_fp32.push_back(&tout_saber_fp32);
    tvout_saber_int32.push_back(&tout_saber_int32);
    tvout_saber_int8.push_back(&tout_saber_int8);

    //! fp32
    conv_int8_fp32.compute_output_shape(tvin_int8, tvout_saber_fp32, param);
    Shape sh_out_saber_fp32 = tvout_saber_fp32[0]->valid_shape();
    //! int32
    conv_int8_int32.compute_output_shape(tvin_int8, tvout_saber_int32, param);
    Shape sh_out_saber_int32 = tvout_saber_int32[0]->valid_shape();
    //! int8
    conv_int8.compute_output_shape(tvin_int8, tvout_saber_int8, param);
    Shape sh_out_saber = tvout_saber_int8[0]->valid_shape();

    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
//    LOG(INFO) << "re-alloc output memory";
    tvout_saber_int32[0]->re_alloc(shape_out, AK_INT32);
    tvout_saber_fp32[0]->re_alloc(shape_out, AK_FLOAT);
    tvout_saber_int8[0]->re_alloc(shape_out, AK_INT8);

    //! init the op
    LOG(INFO) << "saber conv impl init";
    //! fp32
    auto states = conv_int8_fp32.init(tvin_int8, tvout_saber_fp32, param, SPECIFY, SABER_IMPL, ctx1);
    // states = conv_int8.init(tvin_int8, tvout_saber_fp32, ctx1);
    //! int32
    states = conv_int8_int32.init(tvin_int8, tvout_saber_int32, param, SPECIFY, SABER_IMPL, ctx1);
    //! int8
    states = conv_int8.init(tvin_int8, tvout_saber_int8, param, SPECIFY, SABER_IMPL, ctx1);
    CHECK_EQ(states, SaberSuccess) << "Saber conv init failed";

    //! compute
    LOG(INFO) << "saber conv compute";
    to = 0;
    min_time = 1000000;
    for (int i = 0; i < g_test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        //! fp32
        //states = conv_int8.dispatch(tvin_int8, tvout_saber_fp32);
        //! int32
        //states = conv_int8.dispatch(tvin_int8, tvout_saber_int32);
        //! int8
        states = conv_int8(tvin_int8, tvout_saber_int8, param, ctx1);
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
        CHECK_EQ(states, SaberSuccess) << "Saber conv compute failed";
    }
    double gops = 2.0 * n * ch_out * wout * hout * (chin / group) * kernel_w * kernel_h;
    LOG(INFO) << "saber int8 conv running time, ave: " << to / g_test_iter << ", min time: " << min_time << \
        ", GOPS: " << 0.000001 * gops / min_time;
    to = 0;
    min_time = 1000000;
    for (int i = 0; i < g_test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        //! int32
        states = conv_int8_int32(tvin_int8, tvout_saber_int32, param, ctx1);
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
        CHECK_EQ(states, SaberSuccess) << "Saber conv compute failed";
    }

    LOG(INFO) << "saber int32 conv running time, ave: " << to / g_test_iter << ", min time: " << min_time << \
        ", GOPS: " << 0.000001 * gops / min_time;
    to = 0;
    min_time = 1000000;
    for (int i = 0; i < g_test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        //! fp32
        states = conv_int8_fp32(tvin_int8, tvout_saber_fp32, param, ctx1);
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
        CHECK_EQ(states, SaberSuccess) << "Saber conv compute failed";
    }
    LOG(INFO) << "saber fp32 conv running time, ave: " << to / g_test_iter << ", min time: " << min_time << \
        ", GOPS: " << 0.000001 * gops / min_time;

//    print_tensor(tout_saber_fp32);
#if 0
    if (g_compare_result) {
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic_fp32, tout_saber_fp32, max_ratio, max_diff);
                LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabsf(max_ratio) > 1e-3f) {
            if (max_diff > 5e-4f) {
                        LOG(WARNING) << "basic result";
                print_tensor(tout_basic_fp32);
                        LOG(WARNING) << "saber result";
                print_tensor(tout_saber_fp32);
                TensorH tdiff(tout_basic_fp32.valid_shape(), AK_FLOAT);
                tensor_diff(tout_basic_fp32, tout_saber_fp32, tdiff);
                print_tensor(tdiff);
                return SaberInvalidValue;
            }
        }
    }
#endif
#if 1
     if (g_compare_result) {
        LOG(INFO) << "int32 result: ";
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host((const int*)tout_basic_int32.data(), (const int*)tout_saber_int32.data(), tout_basic_int32.valid_size(), max_ratio, max_diff);
        LOG(INFO) << "int32 compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;

        //! int32
       double mean_basic = tensor_mean_value<ARM>(tout_basic_int32, nullptr);
       double mean_saber = tensor_mean_value<ARM>(tout_saber_int32, nullptr);

        LOG(INFO) << "int32 mean_basic: " << mean_basic << ", mean_saber: " << mean_saber;
        double max_ratio_thresh = 2e-1f;
        //! int32
       long long diff_num = count_diff<int>(static_cast<const int*>(tout_basic_int32.data()), \
           static_cast<const int*>(tout_saber_int32.data()), tout_saber_int32.valid_size(), max_ratio_thresh, thinf.get_scale()[0]);
       LOG(INFO) << "int32 number of diff ratio > " << max_ratio_thresh << " is: " << diff_num << ", %" \
           << 100.f * diff_num / tout_basic_int32.valid_size();

        if ((float)diff_num / tout_saber_int32.valid_size() > 0.05/* || mean_diff_ratio > 0.1*/) {
            //!int32
           print_tensor(thinc);
           print_tensor(pweihtc);
           LOG(INFO) << "int32 basic result:";
           print_tensor(tout_basic_int32);
           LOG(INFO) << "int32 saber result:";
           print_tensor(tout_saber_int32);
            return SaberInvalidValue;
        }
        LOG(INFO) << "int32 passed";
    }
    if (g_compare_result) {
        LOG(INFO) << "fp32 result: ";
        double max_ratio = 0;
        double max_diff = 0;
        // ! fp32
        tensor_cmp_host((const float*)tout_basic_fp32.data(), (const float*)tout_saber_fp32.data(), tout_basic_fp32.valid_size(), max_ratio, max_diff);
        // ! int8
        LOG(INFO) << "fp32 compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;

        //! fp32
        double mean_basic = tensor_mean_value<ARM>(tout_basic_fp32, nullptr);
        double mean_saber = tensor_mean_value<ARM>(tout_saber_fp32, nullptr);

        LOG(INFO) << "fp32 mean_basic: " << mean_basic << ", mean_saber: " << mean_saber;
        double max_ratio_thresh = 2e-1f;
        //! fp32
        long long diff_num = count_diff<float>(static_cast<const float*>(tout_basic_fp32.data()), \
            static_cast<const float*>(tout_saber_fp32.data()), tout_saber_fp32.valid_size(), max_ratio_thresh, thinf.get_scale()[0]);
        LOG(INFO) << "fp32 number of diff ratio > " << max_ratio_thresh << " is: " << diff_num << ", %" \
            << 100.f * diff_num / tout_basic_fp32.valid_size();

        if ((float)diff_num / tout_saber_fp32.valid_size() > 0.05/* || mean_diff_ratio > 0.1*/) {
            //! fp32
            print_tensor(thinc);
            print_tensor(pweihtc);

            LOG(INFO) << "fp32 basic result-int32:";
            print_tensor(tout_basic_int32);
            LOG(INFO) << "fp32 basic result-fp32:";
            print_tensor(tout_basic_fp32);
            LOG(INFO) << "fp32 saber result-fp32:";
            print_tensor(tout_saber_fp32);

            return SaberInvalidValue;
        }
        LOG(INFO) << "fp32 passed";
    }
    if (g_compare_result) {
        LOG(INFO) << "int8 result: ";
        double max_ratio = 0;
        double max_diff = 0;
        // ! int8
        tensor_cmp_host((const int8_t*)tout_basic_int8.data(), (const int8_t*)tout_saber_int8.data(), \
            tout_basic_int8.valid_size(), max_ratio, max_diff);
        LOG(INFO) << "int8 compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
         //! int8
        double mean_basic = tensor_mean_value(tout_basic_int8, nullptr);
        double mean_saber = tensor_mean_value(tout_saber_int8, nullptr);

        LOG(INFO) << "int8 mean_basic: " << mean_basic << ", mean_saber: " << mean_saber;
        double max_ratio_thresh = 2e-1f;
        //! int8
        long long diff_num = count_diff<int8_t>(static_cast<const int8_t*>(tout_basic_int8.data()), \
            static_cast<const int8_t*>(tout_saber_int8.data()), tout_saber_int8.valid_size(), max_ratio_thresh, thinf.get_scale()[0]);
        LOG(INFO) << "int8 number of diff ratio > " << max_ratio_thresh << " is: " << diff_num << ", %" \
            << 100.f * diff_num / tout_saber_int8.valid_size();
        if ((float)diff_num / tout_saber_int8.valid_size() > 0.05/* || mean_diff_ratio > 0.1*/) {
            //! int8
            print_tensor(thinc);
            print_tensor(pweihtc);
            LOG(INFO) << "int8 basic result int32:";
            print_tensor(tout_basic_int32);
            LOG(INFO) << "int8 basic result int8:";
            print_tensor(tout_basic_int8);
            LOG(INFO) << "int8 saber result:";
            print_tensor(tout_saber_int8);
            return SaberInvalidValue;
        }
        LOG(INFO) << "int8 passed";
//        CHECK_EQ(fabsf(max_ratio) < 1e-4f, true) << "compute result error";
    }
#endif
    return SaberSuccess;
}

#if 1
TEST(TestSaberFunc, test_func_conv_depthwise_3x3_int8) {
    if (g_basic_test) {
        for (auto& batch : {1, 2}) {
            for (auto& c : {1, 3, 8, 16, 24}) {
                    for (auto& h : {4, 8, 9, 15, 28, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 112, 128, 256}) {
                        for (auto &flag_bias : {false, true}) {
                            for (auto &flag_relu : {false, true}) {
                                for (auto &th : {1, 2, 4}) {
                                    for (auto & stride : {1, 2}){
                                        int stride_w = stride;
                                        int stride_h = stride;
                                        int group = c;
                                        int pad_w = 1;
                                        int pad_h = 1;
                                        int dila_w = 1;
                                        int dila_h = 1;
                                        int kw = 3;
                                        int kh = 3;
                                        int w = h;
                                        int chout = c;
                                        LOG(INFO) << "conv_depthwise_3x3_int8 OP";
                                        auto flag = test_arm_conv_int8(batch, c, h, w, chout, kw, kh, stride_w, stride_h, \
                                            pad_w, pad_h, dila_w, dila_h, group, flag_bias, flag_relu, \
                                            th, g_cluster);
                                        if (flag == SaberSuccess) {
                                            LOG(INFO) << "test int8 3x3s2_dw conv: batchsize: " << batch << ", channel: "
                                                << c << ", h & w: " << h << ", ch_out: " << chout << ", group: " << group << \
                                                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                                                << (flag_relu ? "true" : "false") << ", threads: " << \
                                                th << ", cluster: " << g_cluster << " passed!!\n";
                                        } else {
                                            LOG(FATAL) << "test int8 3x3s2_dw conv: batchsize: " << batch << ", channel: "
                                                << c << ", h & w: " << h << ", ch_out: " << chout << ", group: " << group << \
                                                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                                                << (flag_relu ? "true" : "false") << ", threads: " << \
                                                th << ", cluster: " << g_cluster << " failed!!\n";
                                        }
                                    }
                                }
                            }
                        }
                    }
            }
        }
    }
}
#endif

#ifdef __aarch64__
#if 0
TEST(TestSaberFunc, test_func_conv_depthwise_5x5_int8) {
    if (g_basic_test) {
        for (auto& batch : {1, 2}) {
            for (auto& c : { 1, 3, 8, 16, 24}) {
                    for (auto& h : {1, 2, 4, 8, 9, 15, 28, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,/* 112, 128, 256*/}) {
                        for (auto &flag_bias : {false, /*true*/}) {
                            for (auto &flag_relu : {false, /*true*/}) {
                                for (auto &th : {2 /*1, 2, 4*/}) {
                                    for (auto & stride : {1/*, 2*/}){
                                        int stride_w = stride;
                                        int stride_h = stride;
                                        int group = c;
                                        int pad_w = 2;
                                        int pad_h = 2;
                                        int dila_w = 1;
                                        int dila_h = 1;
                                        int kw = 5;
                                        int kh = 5;
                                        int w = h;
                                        int chout = c;
                                        LOG(INFO) << "conv_depthwise_5x5_int8 OP";
                                        auto flag = test_arm_conv_int8(batch, c, h, w, chout, kw, kh, stride_w, stride_h, \
                                            pad_w, pad_h, dila_w, dila_h, group, flag_bias, flag_relu, \
                                            th, g_cluster);
                                        if (flag == SaberSuccess) {
                                            LOG(INFO) << "test int8 5x5s1_dw conv: batchsize: " << batch << ", channel: "
                                                << c << ", h & w: " << h << ", ch_out: " << chout << ", group: " << group << \
                                                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                                                << (flag_relu ? "true" : "false") << ", threads: " << \
                                                th << ", cluster: " << g_cluster << " passed!!\n";
                                        } else {
                                            LOG(FATAL) << "test int8 5x5s1_dw conv: batchsize: " << batch << ", channel: "
                                                << c << ", h & w: " << h << ", ch_out: " << chout << ", group: " << group << \
                                                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                                                << (flag_relu ? "true" : "false") << ", threads: " << \
                                                th << ", cluster: " << g_cluster << " failed!!\n";
                                        }
                                    }
                                }
                            }
                        }
                    }
            }
        }
    }
}
#endif
#endif // __aarch64__

#if 1
TEST(TestSaberFunc, test_func_conv_3x3s1_direct_int8) {
    if (g_basic_test) {
        for (auto& batch : {1, 2}) {
            for (auto& c : {1, 3, 8, 16, 32, 64}) {
                for (auto& h : {5, 15, 16, 28, 56, 112, 128, 256}) {
                    for (auto& w : {6, 15, 28, 29, 30, 31, 32, 56, 112, 128, 255, 256}) {
                        for (auto &flag_bias : {false, true}) {
                            for (auto &flag_relu : {false, true}) {
                                for (auto &th : {1, 2, 4}) {
                                    for (auto & chout : {3, 8, 9, 10, 11, 12}){
                                        int stride_w = 1;
                                        int stride_h = 1;
                                        int group = 1;
                                        int pad_w = 1;
                                        int pad_h = 1;
                                        int dila_w = 1;
                                        int dila_h = 1;
                                        int kw = 3;
                                        int kh = 3;
                                        LOG(INFO) << "conv_3x3s1_direct_int8 OP";
                                        auto flag = test_arm_conv_int8(batch, c, h, w, chout, kw, kh, stride_w, stride_h, \
                                            pad_w, pad_h, dila_w, dila_h, group, flag_bias, flag_relu, \
                                            th, g_cluster);
                                        if (flag == SaberSuccess) {
                                            LOG(INFO) << "test int8 3x3s1_direct conv: batchsize: " << batch << ", channel: "
                                                << c << ", h & w: " << h << ", ch_out: " << chout << ", group: " << group << \
                                                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                                                << (flag_relu ? "true" : "false") << ", threads: " << \
                                                th << ", cluster: " << g_cluster << " passed!!\n";
                                        } else {
                                            LOG(FATAL) << "test int8 3x3s1_direct conv: batchsize: " << batch << ", channel: "
                                                << c << ", h & w: " << h << ", ch_out: " << chout << ", group: " << group << \
                                                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                                                << (flag_relu ? "true" : "false") << ", threads: " << \
                                                th << ", cluster: " << g_cluster << " failed!!\n";
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif

#if 1
TEST(TestSaberFunc, test_func_conv_3x3s2_direct_int8) {

    if (g_basic_test) {
        for (auto& batch : {1, 2}) {
        for (auto& ci : {2, 3, 8}) {
        for (auto& co : {1, 5, 16}) {
        for (auto& h : {1, 3, 8, 15, 16, 28, 32, 75}) {
        for (auto &flag_bias : {false, true}) {
        for (auto &flag_relu : {false, true}) {
        for (auto &th : {1, 2, 4}) {
            int stride_w = 2;
            int stride_h = 2;
            int group = 1;
            int pad_w = 1;
            int pad_h = 1;
            int dila_w = 1;
            int dila_h = 1;
            int kw = 3;
            int kh = 3;
            LOG(INFO) << "conv_3x3s2_direct_int8 OP";
            auto flag = test_arm_conv_int8(batch, ci, h, h, co, kw, kh, stride_w, stride_h, \
                pad_w, pad_h, dila_w, dila_h, group, flag_bias, flag_relu, \
                th, g_cluster);
            if (flag == SaberSuccess) {
                LOG(INFO) << "test int8 3x3s2_direct conv: batchsize: " << batch << ", channel: "
                    << ci << ", h & w: " << h << ", ch_out: " << co << ", group: " << group << \
                    ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                    << (flag_relu ? "true" : "false") << ", threads: " << \
                    th << ", cluster: " << g_cluster << " passed!!\n";
            } else {
                LOG(FATAL) << "test int8 3x3s2_direct conv: batchsize: " << batch << ", channel: "
                    << ci << ", h & w: " << h << ", ch_out: " << co << ", group: " << group << \
                    ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                    << (flag_relu ? "true" : "false") << ", threads: " << \
                    th << ", cluster: " << g_cluster << " failed!!\n";
            }
        }
        }
        }
        }
        }
        }
        }
    }
}
#endif

#if 1
TEST(TestSaberFunc, test_func_conv_1x1s1_int8) {

    if (g_basic_test) {
    for (auto& batch : {1, 2}) {
    for (auto& c : {1, 3, 8}) {
    for (auto& cout : {1, 5, 16}) {
    for (auto& g_div : {1, 2}) {
    for (auto& h : {1, 3, 8, 15, 28, 32, 38, 75}) {
    for (auto &flag_bias : {false, true}) {
    for (auto &flag_relu : {false, true}) {
    for (auto &th : {1, 2, 4}) {
        int w = h;
        int g = g_div;
        if ((c % g_div != 0) || (cout % g_div != 0)) {
            g = 1;
        }
        auto flag = test_arm_conv_int8(batch, c, h, w, cout, 1, 1, 1, 1, \
            0, 0, 1, 1, g, flag_bias, flag_relu, th, g_cluster);
        if (flag == SaberSuccess) {
            LOG(INFO) << "test int8 1x1s1 conv: batchsize: " << batch << ", channel: "
                << c << ", h & w: " << h << ", ch_out: " << cout << ", group: " << g << \
                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                << (flag_relu ? "true" : "false") << ", threads: " << \
                th << ", cluster: " << g_cluster << " passed!!\n";
        } else {
            LOG(FATAL) << "test int8 1x1s1 conv: batchsize: " << batch << ", channel: "
                << c << ", h & w: " << h << ", ch_out: " << cout << ", group: " << g << \
                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                << (flag_relu ? "true" : "false") << ", threads: " << \
                th << ", cluster: " << g_cluster << " failed!!\n";
        }
    }
    }
    }
    }
    }
    }
    }
    }
    }
}
#endif

#if 1
TEST(TestSaberFunc, test_func_conv_gemm_int8) {
    if (g_basic_test) {
    for (auto& batch : {1, 2}) {
    for (auto& c : {1, 3, 8}) {
    for (auto& cout : {1, 5, 16}) {
    for (auto& g_div : {1, 2}) {
    for (auto& h : {1, 3, 8, 15, 28, 32, 38, 75}) {
    for (auto& kw : {1, 2, 3, 5}) {
    for (auto& kh : {1, 2, 3, 5}) {
    for (auto& pad : {1, 2}) {
    for (auto& stride : {1, 2}) {
    for (auto& dila : {1, 2}) {
    for (auto &flag_bias : {false, true}) {
    for (auto &flag_relu : {false, true}) {
    for (auto &th : {1, 2, 4}) {
        int w = h;
        int g = g_div;
        if ((c % g_div != 0) || (cout % g_div != 0)) {
            g = 1;
        }
        //! 3x3s1/s2 direct
        if (kw == 3 && kh == 3 && (stride == 1 || stride == 2) && dila == 1) {
            continue;
        }
        //! 3x3 dw
        if (kw == 3 && kh == 3 && dila == 1 && pad == 1 && g == cout && g == c) {
            continue;
        }
        //! 5x5 dw
        if (kw == 5 && kh == 5 && dila == 1 && pad == 2 && g == cout && g == c) {
            continue;
        }
        auto flag = test_arm_conv_int8(batch, c, h, w, cout, kw, kh, stride, stride, \
            pad, pad, dila, dila, g, flag_bias, flag_relu, th, g_cluster);
        if (flag == SaberSuccess) {
            LOG(INFO) << "test int8 conv: batchsize: " << batch << ", channel: "
                << c << ", h & w: " << h << ", ch_out: " << cout << ", group: " << g << \
                ", kernel_h: " << kh << ", kernel_w: " << kw << \
                ", pad: " << pad << ", stride: " << stride << ", dila: " << dila << \
                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                << (flag_relu ? "true" : "false") << ", threads: " << \
                th << ", cluster: " << g_cluster << " passed!!\n";
        } else {
            LOG(FATAL) << "test int8 conv: batchsize: " << batch << ", channel: "
                << c << ", h & w: " << h << ", ch_out: " << cout << ", group: " << g << \
                ", kernel_h: " << kh << ", kernel_w: " << kw << \
                ", pad: " << pad << ", stride: " << stride << ", dila: " << dila << \
                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                << (flag_relu ? "true" : "false") << ", threads: " << \
                th << ", cluster: " << g_cluster << " failed!!\n";
        }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
}
#endif

#if 1
TEST(TestSaberFunc, test_conv_int8_custom_size) {
    for (int i = 0; i < 1; i++) {
    auto flag = test_arm_conv_int8(g_num, g_chin, g_h_in, g_w_in, g_ch_out, g_kw, g_kh, g_stride_w, g_stride_h, \
            g_pad_w, g_pad_h, g_dila_w, g_dila_h, g_group, g_flag_bias, g_flag_relu, g_threads, g_cluster);
    if (flag == SaberSuccess) {
        LOG(INFO) << "test int8 conv: batchsize: " << g_num << ", channel: " \
            << g_chin << ", h & w: " << g_h_in << \
            ", pad: " << g_pad_h << ", stride: " << g_stride_h << ", dila: " << g_dila_h << \
            ", bias: " << (g_flag_bias ? "true" : "false") << ", relu: "
                          << (g_flag_relu ? "true" : "false") << ", threads: " << \
            g_threads << ", cluster: " << g_cluster << " passed!!";
    } else {
        LOG(FATAL) << "test int8 conv: batchsize: " << g_num << ", channel: "
            << g_chin << ", h & w: " << g_h_in << \
            ", pad: " << g_pad_h << ", stride: " << g_stride_h << ", dila: " << g_dila_h << \
            ", bias: " << (g_flag_bias ? "true" : "false") << ", relu: "
                          << (g_flag_relu ? "true" : "false") << ", threads: " << \
            g_threads << ", cluster: " << g_cluster << " failed!!";
    }
    }
}
#endif

int main(int argc, const char** argv){
    Env<ARM>::env_init();
            LOG(ERROR) << "usage: ./" << argv[0] << " basic_test cluster  threads  test_iter " << \
                " compare_result flag_bias flag_relu num ch_in h_in w_in ch_out group" << \
                " kernel pad stride dila [kernel_h] [pad_h] [stride_h] [dila_h]";

    if (argc >= 2) {
        g_basic_test = atoi(argv[1]) > 0;
    }

    if (argc >= 3) {
        g_cluster = atoi(argv[2]);
    }
    if (argc >= 4) {
        g_threads = atoi(argv[3]);
    }
    if (argc >= 5) {
        g_test_iter = atoi(argv[4]);
    }
    if (argc >= 6) {
        g_compare_result = atoi(argv[5]) > 0;
    }
    if (argc >= 7) {
        g_flag_bias = atoi(argv[6]) > 0;
    }
    if (argc >= 8) {
        g_flag_relu = atoi(argv[7]) > 0;
    }
    if (argc >= 9) {
        if (argc < 18) {
            LOG(FATAL) << "usage: ./" << argv[0] << "basic_test cluster  threads  test_iter " << \
                " compare_result flag_bias flag_relu num ch_in h_in w_in ch_out group" << \
                " kernel pad stride dila [kernel_h] [pad_h] [stride_h] [dila_h]";
            return -1;
        }
        g_num = atoi(argv[8]);
        g_chin = atoi(argv[9]);
        g_h_in = atoi(argv[10]);
        g_w_in = atoi(argv[11]);
        g_ch_out = atoi(argv[12]);
        g_group = atoi(argv[13]);
        g_kw = atoi(argv[14]);
        g_kh = g_kw;
        g_pad_w = atoi(argv[15]);
        g_pad_h = g_pad_w;
        g_stride_w = atoi(argv[16]);
        g_stride_h = g_stride_w;
        g_dila_w = atoi(argv[17]);
        g_dila_h = g_dila_w;
    }
    if (argc > 18) {
        g_kh = atoi(argv[18]);
    }
    if (argc > 19) {
        g_pad_h = atoi(argv[19]);
    }
    if (argc > 20) {
        g_stride_h = atoi(argv[20]);
    }
    if (argc > 21) {
        g_dila_h = atoi(argv[21]);
    }

    // initial logger
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

#else

int main(int argc, const char** argv){
    LOG(INFO) << "this unit test only be used in TargetType is ARM";
    return 0;
}

#endif

