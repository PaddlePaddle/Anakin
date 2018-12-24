#include "test_lite.h"
#include "saber/lite/funcs/saber_conv.h"
#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

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

typedef Tensor<CPU> TensorH;

SaberStatus test_arm_conv_int8(int n, int c, int h, int w, \
    int ch_out, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h, \
    int dila_w, int dila_h, int group, bool is_bias, bool is_relu, int thread_num, int cluster_id) {

    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;

    Context ctx1;
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
    Shape shin = {n, c, h, w};
    thinf.re_alloc(shin, AK_FLOAT);
    thinc.re_alloc(shin, AK_INT8);

    int num = n;
    int chin = c;
    int hin = h;
    int win = w;

    LOG(INFO) << "conv param: ";
    LOG(INFO) << " img_num = " << num << " in_channels = " << chin << " img_h = " << hin << " img_w = " << win;
    LOG(INFO) << " num_out = " << ch_out << " group = " << group << " kernel_w = " << kernel_w << " kernel_h = " << kernel_h;
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

    Shape shape_out{num, ch_out, hout, wout};

    Shape shw{ch_out, chin / group, kernel_h, kernel_w};
    Shape shb{1, ch_out, 1, 1};

    TensorH pweihtf;
    TensorH pbiasf;

    TensorH pweihtc;
    TensorH pbiasi;

    pweihtf.re_alloc(shw, AK_FLOAT);
    pbiasf.re_alloc(shb, AK_FLOAT);

    pweihtc.re_alloc(shw, AK_FLOAT);
    pbiasi.re_alloc(shb, AK_INT32);

    fill_tensor_rand(thinf, -1.f, 1.f);
    fill_tensor_rand(pweihtf, -1.f, 1.f);
    fill_tensor_rand(pbiasf, -1.f, 1.f);
//    fill_tensor_const(thinf, 1.f);
//    fill_tensor_const(pweihtf, 1.f);
//    fill_tensor_const(pbiasf, 1.f);

    LOG(INFO) << "get input scale";
    pweihtc.copy_from(pweihtf);
    //! convert input data type
    std::vector<float> scale;
    get_tensor_scale(thinf, scale, -1, 127.f);
    thinf.set_scale(scale);
    LOG(INFO) << "input tesnor scale at factor 127.f is " << thinf.get_scale()[0] << ", max_val: " << 127.f * thinf.get_scale()[0];

    trans_tensor_fp32_to_int8(thinf, thinc, scale[0]);
    thinc.set_scale(scale);
//    print_tensor(thinf);
//    print_tensor(thinc);

    LOG(INFO) << "get weights scale";
    //! convert weight data type

    trans_weights_dtype(pweihtc, AK_INT8, 127.f, false);
    std::vector<float> w_scale = pweihtc.get_scale();
//    LOG(INFO) << "input tesnor scale at factor 127.f is ";
//    for (int j = 0; j < w_scale.size(); ++j) {
//        LOG(INFO) << "|-- " << j << ": " << w_scale[j] << ", max_val: " << 127.f * w_scale[j];
//    }

    trans_fp32_bias_to_int32(pbiasf, pbiasi, thinf.get_scale()[0], w_scale);

//    print_tensor(pweihtf);
//    print_tensor(pweihtc);

    std::vector<float> scale_out = {1.f};
    tout_saber_int8.set_scale(scale_out);
    tout_basic_int8.set_scale(scale_out);

    //! get int8 and fp32 basic result
    if (g_compare_result) {
        LOG(INFO) << "run basic conv for precision comparation";
        const char* dinc = static_cast<const char*>(thinc.data());
        const char* weightc = static_cast<const char*>(pweihtc.data());
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
        conv_basic<char, int>(dinc, dout_basic_int32, num, ch_out, hout, wout, chin, hin, win, \
            weightc, biasi, group, kernel_w, kernel_h, stride_w, stride_h, \
            dila_w, dila_h, pad_w, pad_h, is_bias, is_relu);

        LOG(INFO) << "trans basic int32 to int8";
        trans_tensor_int32_to_int8(tout_basic_int32, tout_basic_int8, thinf.get_scale()[0], tout_basic_int8.get_scale()[0], w_scale);
        LOG(INFO) << "trans basic int32 to fp32";
        trans_tensor_int32_to_fp32(tout_basic_int32, tout_basic_fp32, thinf.get_scale()[0], w_scale);

//        print_tensor(tout_basic_fp32);
        // LOG(INFO) << "basic in32 result";
        // print_tensor(tout_basic_int32);
    }

    SaberConv2D conv_int8;

    Conv2DParam param(pweihtf.valid_size(), ch_out, group, kernel_w, kernel_h, \
        stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, is_bias, AK_INT8,\
        pweihtc.data(), w_scale.data(), pbiasf.data(), \
        false, is_relu, Active_relu, 0.f, 0.f, false, nullptr);


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

    conv_int8.load_param(&param);

    //! fp32
    conv_int8.compute_output_shape(tvin_int8, tvout_saber_fp32);
    Shape sh_out_saber = tvout_saber_fp32[0]->valid_shape();
    //! int32
//    conv_int8.compute_output_shape(tvin_int8, tvout_saber_int32);
//    Shape sh_out_saber = tvout_saber_int32[0]->valid_shape();


    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
//    LOG(INFO) << "re-alloc output memory";
    tvout_saber_int32[0]->re_alloc(shape_out, AK_INT32);
    tvout_saber_fp32[0]->re_alloc(shape_out, AK_FLOAT);
    tvout_saber_int8[0]->re_alloc(shape_out, AK_INT8);

    //! set compute precision
//    LOG(INFO) << "set compute precision";
    auto states = conv_int8.set_op_precision(AK_INT8);
    CHECK_EQ(states, SaberSuccess) << "Saber conv op precision to int8 failed";

    //! init the op
//    LOG(INFO) << "saber conv impl init";
    //! fp32
    states = conv_int8.init(tvin_int8, tvout_saber_fp32, ctx1);
    //! int32
//    states = conv_int8.init(tvin_int8, tvout_saber_int32, ctx1);
    CHECK_EQ(states, SaberSuccess) << "Saber conv init failed";

    //! compute
//    LOG(INFO) << "saber conv compute";
    to = 0;
    for (int i = 0; i < g_test_iter; ++i) {
        t1.clear();
        t1.start();
        //! fp32
        states = conv_int8.dispatch(tvin_int8, tvout_saber_fp32);
        //! int32
//        states = conv_int8.dispatch(tvin_int8, tvout_saber_int32);
        t1.end();
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
        CHECK_EQ(states, SaberSuccess) << "Saber conv compute failed";
    }

    long long gops = n * ch_out * wout * hout * (chin / group) * kernel_w * kernel_h;
    LOG(INFO) << "saber conv running time, ave: " << to / g_test_iter << ", min time: " << min_time << \
        ", GOPS: " << 0.000001 * gops / min_time;

//    print_tensor(tout_saber_fp32);

    if (g_compare_result) {
        double max_ratio = 0;
        double max_diff = 0;
        //! fp32
        tensor_cmp_host(tout_basic_fp32, tout_saber_fp32, max_ratio, max_diff);
        //! int32
//        tensor_cmp_host(tout_basic_int32, tout_saber_int32, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        //! fp32
        double mean_basic = tensor_mean(tout_basic_fp32);
        double mean_saber = tensor_mean(tout_saber_fp32);
        //! int32
//        double mean_basic = tensor_mean(tout_basic_int32);
//        double mean_saber = tensor_mean(tout_saber_int32);
        LOG(INFO) << "mean_basic: " << mean_basic << ", mean_saber: " << mean_saber;
        double max_ratio_thresh = 2e-1f;
        //! fp32
        long long diff_num = count_diff<float>(static_cast<const float*>(tout_basic_fp32.data()), \
            static_cast<const float*>(tout_saber_fp32.data()), tout_saber_fp32.valid_size(), max_ratio_thresh, thinf.get_scale()[0]);
        LOG(INFO) << "number of diff ratio > " << max_ratio_thresh << " is: " << diff_num << ", %" \
            << 100.f * diff_num / tout_basic_fp32.valid_size();
        //! int32
//        long long diff_num = count_diff<float>(static_cast<const float*>(tout_basic_int32.data()), \
//            static_cast<const float*>(tout_saber_int32.data()), tout_saber_int32.valid_size(), max_ratio_thresh, thinf.get_scale()[0]);
//        LOG(INFO) << "number of diff ratio > " << max_ratio_thresh << " is: " << diff_num << ", %" \
//            << 100.f * diff_num / tout_basic_int32.valid_size();

//        double mean_diff_ratio = fabs(mean_basic - mean_saber) / (fabs(mean_basic) + fabs(mean_saber));
//        LOG(INFO) << "mean val diff ratio: " << mean_diff_ratio;
        if ((float)diff_num / tout_saber_fp32.valid_size() > 0.05/* || mean_diff_ratio > 0.1*/) {
            //! fp32
            TensorH tdiff;
            tdiff.re_alloc(shape_out, AK_FLOAT);
            tensor_diff(tout_basic_fp32, tout_saber_fp32, tdiff);
            print_tensor(thinc);
            print_tensor(pweihtc);
            LOG(INFO) << "basic result int32:";
            print_tensor(tout_basic_int32);
            LOG(INFO) << "basic result fp32:";
            print_tensor(tout_basic_fp32);
            LOG(INFO) << "saber result:";
            print_tensor(tout_saber_fp32);
            LOG(INFO) << "diff result:";
            print_tensor(tdiff);

            //!int32
//            TensorH tdiff;
//            tdiff.re_alloc(shape_out, AK_INT32);
//            tensor_diff(tout_basic_int32, tout_saber_int32, tdiff);
//            LOG(INFO) << "basic result:";
//            print_tensor(tout_basic_int32);
//            LOG(INFO) << "saber result:";
//            print_tensor(tout_saber_int32);
//            LOG(INFO) << "diff result:";
//            print_tensor(tdiff);
            return SaberInvalidValue;
        }
//        CHECK_EQ(fabsf(max_ratio) < 1e-4f, true) << "compute result error";
    }
    return SaberSuccess;
}

#if 1
TEST(TestSaberLite, test_func_conv_depthwise_3x3_int8) {
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
                                                << c << ", h & w: " << h << ", num_out: " << chout << ", group: " << group << \
                                                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                                                << (flag_relu ? "true" : "false") << ", threads: " << \
                                                th << ", cluster: " << g_cluster << " passed!!\n";
                                        } else {
                                            LOG(FATAL) << "test int8 3x3s2_dw conv: batchsize: " << batch << ", channel: "
                                                << c << ", h & w: " << h << ", num_out: " << chout << ", group: " << group << \
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
TEST(TestSaberLite, test_func_conv_depthwise_5x5_int8) {
    if (g_basic_test) {
        for (auto& batch : {1, 2}) {
            for (auto& c : {1, 3, 8, 16, 24}) {
                    for (auto& h : {4, 8, 9, 15, 28, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 112, 128, 256}) {
                        for (auto &flag_bias : {false, true}) {
                            for (auto &flag_relu : {false/*, true*/}) {
                                for (auto &th : {1, 2, 4}) {
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
                                        LOG(INFO) << "conv_depthwise_3x3_int8 OP";
                                        auto flag = test_arm_conv_int8(batch, c, h, w, chout, kw, kh, stride_w, stride_h, \
                                            pad_w, pad_h, dila_w, dila_h, group, flag_bias, flag_relu, \
                                            th, g_cluster);
                                        if (flag == SaberSuccess) {
                                            LOG(INFO) << "test int8 3x3s2_dw conv: batchsize: " << batch << ", channel: "
                                                << c << ", h & w: " << h << ", num_out: " << chout << ", group: " << group << \
                                                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                                                << (flag_relu ? "true" : "false") << ", threads: " << \
                                                th << ", cluster: " << g_cluster << " passed!!\n";
                                        } else {
                                            LOG(FATAL) << "test int8 3x3s2_dw conv: batchsize: " << batch << ", channel: "
                                                << c << ", h & w: " << h << ", num_out: " << chout << ", group: " << group << \
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
// fixme
#if 0
TEST(TestSaberLite, test_func_conv_3x3s1_direct_int8) {
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
                                                << c << ", h & w: " << h << ", num_out: " << chout << ", group: " << group << \
                                                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                                                << (flag_relu ? "true" : "false") << ", threads: " << \
                                                th << ", cluster: " << g_cluster << " passed!!\n";
                                        } else {
                                            LOG(FATAL) << "test int8 3x3s1_direct conv: batchsize: " << batch << ", channel: "
                                                << c << ", h & w: " << h << ", num_out: " << chout << ", group: " << group << \
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

// fixme
#if 0
TEST(TestSaberLite, test_func_conv_3x3s2_direct_int8) {

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
                    << ci << ", h & w: " << h << ", num_out: " << co << ", group: " << group << \
                    ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                    << (flag_relu ? "true" : "false") << ", threads: " << \
                    th << ", cluster: " << g_cluster << " passed!!\n";
            } else {
                LOG(FATAL) << "test int8 3x3s2_direct conv: batchsize: " << batch << ", channel: "
                    << ci << ", h & w: " << h << ", num_out: " << co << ", group: " << group << \
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

#if 0
TEST(TestSaberLite, test_func_conv_1x1s1_int8) {

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
                << c << ", h & w: " << h << ", num_out: " << cout << ", group: " << g << \
                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                << (flag_relu ? "true" : "false") << ", threads: " << \
                th << ", cluster: " << g_cluster << " passed!!\n";
        } else {
            LOG(FATAL) << "test int8 1x1s1 conv: batchsize: " << batch << ", channel: "
                << c << ", h & w: " << h << ", num_out: " << cout << ", group: " << g << \
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

#if 0
TEST(TestSaberLite, test_func_conv_gemm_int8) {
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
        auto flag = test_arm_conv_int8(batch, c, h, w, cout, kw, kh, stride, stride, \
            pad, pad, dila, dila, g, flag_bias, flag_relu, th, g_cluster);
        if (flag == SaberSuccess) {
            LOG(INFO) << "test int8 conv: batchsize: " << batch << ", channel: "
                << c << ", h & w: " << h << ", num_out: " << cout << ", group: " << g << \
                ", kernel_h: " << kh << ", kernel_w: " << kw << \
                ", pad: " << pad << ", stride: " << stride << ", dila: " << dila << \
                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                << (flag_relu ? "true" : "false") << ", threads: " << \
                th << ", cluster: " << g_cluster << " passed!!\n";
        } else {
            LOG(FATAL) << "test int8 conv: batchsize: " << batch << ", channel: "
                << c << ", h & w: " << h << ", num_out: " << cout << ", group: " << g << \
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
TEST(TestSaberLite, test_conv_int8_custom_size) {
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
    Env::env_init();
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
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

