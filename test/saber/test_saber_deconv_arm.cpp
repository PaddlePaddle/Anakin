#include "saber/funcs/deconv.h"
#include "saber/funcs/timer.h"
#include "test/saber/test_saber_func.h"
#include "saber/core/tensor_op.h"
using namespace anakin::saber;

#ifdef USE_ARM_PLACE

int g_cluster = 0;
int g_threads = 1;
int g_test_iter = 1;

bool g_basic_test = false;

bool g_compare_result = true;
bool g_flag_bias = true;
bool g_flag_relu = false;

int g_num = 1;
int g_ch_in = 128;
int g_h_in = 10;
int g_w_in = 10;

int g_ch_out = 128;
int g_group = 128;
int g_kernel = 4;
int g_pad = 1;
int g_stride = 2;
int g_dila = 1;

typedef Tensor<ARM> TensorHf4;

template <typename Dtype>
static void fill_bias_relu(Dtype* tensor, const Dtype* bias, int channel, int channel_size, \
    bool flag_bias, bool flag_relu) {
    Dtype* data = tensor;
    for (int j = 0; j < channel; ++j) {
        Dtype bias_c = flag_bias? bias[j] : 0;
        for (int i = 0; i < channel_size; i++) {
            data[i] += bias_c;
            if (flag_relu) {
                data[i] = data[i] > 0 ? data[i] : 0.f;
            }
        }
        data += channel_size;
    }
}

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void col2im(const Dtype* data_col, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w,
                Dtype* data_im) {
    memset(data_im, 0, height * width * channels * sizeof(Dtype));
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        data_col += output_w;
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                data_im[input_row * width + input_col] += *data_col;
                            }
                            data_col++;
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}

template  <typename type,  typename type2>
static void basic_gemm(int m, int n, int k, const type* a, const type* b, const type2* bias, type2* c, \
    type2 alpha, type2 beta, \
    bool trans_a = false, bool trans_b = false, bool flag_bias = false, bool flag_relu = false) {
#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        type2 bias_data = (type2)0;
        if (flag_bias) {
            bias_data = bias[i];
        }
        for (int j = 0; j < n; ++j) {
            type2 sum = static_cast<type2>(0);
            for (int l = 0; l < k; ++l) {
                type av;
                type bv;
                if (trans_a) {
                    av = a[l * m + i];
                } else{
                    av = a[i * k + l];
                }
                if (trans_b) {
                    bv = b[j * k + l];
                } else {
                    bv = b[l * n + j];
                }
                sum += av * bv;
            }
            type2 tmp = alpha * sum + beta * c[i * n + j] + bias_data;
            if (flag_relu) {
                c[i * n + j] = tmp > (type2)0? tmp : (type2)0;
            } else {
                c[i * n + j] = tmp;
            }
        }
    }
}

//! for float, dtype1 and type2 is float
//! for int8, dytpe1 is char, dtype2 is int
template <typename Dtype1, typename Dtype2>
void deconv_basic(const Dtype1* din, Dtype2* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const Dtype1* weights, const Dtype2* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, \
                          int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu) {


    int m = chout * kernel_w * kernel_h / group;
    int n = hin * win;
    int k = chin / group;

    if (chin != chout || group != chin) {
        CHECK_EQ(chin % group, 0) << "input channel or group size error";
        CHECK_EQ(chout % group, 0) << "output channel or group size error";
    }

    Tensor<ARM> workspace_tensor;
    Shape workspace_shape({1, 1, 1, group * m * n});
    workspace_tensor.re_alloc(workspace_shape, anakin::saber::AK_FLOAT);

    int group_size_in = win * hin * chin / group;
    int group_size_out = wout * hout * chout / group;
    int group_size_coldata = m * n;
    int group_size_weights = chin * chout * kernel_w * kernel_h / (group * group);
    bool flag_1x1s1p1 = (kernel_w == 1) && (kernel_h == 1) && (stride_h == 1) && \
                        (stride_w == 1) && (pad_w == 1) && (pad_h == 1) && \
                        (dila_w == 1) && (dila_h == 1);

    Dtype2* workspace_ptr = static_cast<Dtype2*>(workspace_tensor.mutable_data());

    for (int i = 0; i < num; ++i) {
        const Dtype1* din_batch = din + i * chin * hin * win;
        Dtype2* dout_batch = dout + i * chout * hout * wout;

        Dtype2* col_data = workspace_ptr;
        if (flag_1x1s1p1) {
            col_data = dout_batch;
        }
        memset(col_data, 0, sizeof(Dtype2) * group_size_coldata);
        for (int g = 0; g < group; ++g) {
            const Dtype1* din_group = din_batch + g * group_size_in;
            const Dtype1* weights_group = weights + g * group_size_weights;
            Dtype2* coldata_group = col_data + g * group_size_coldata;
            basic_gemm<Dtype1, Dtype2>(m, n, k, weights_group, din_group, nullptr, coldata_group, \
                (Dtype2)1, (Dtype2)0, true, false, false, (!flag_bias && flag_relu));
        }
        if (!flag_1x1s1p1) {
            col2im(col_data, chout, hout, wout, kernel_h, kernel_w, pad_h, pad_w, \
                stride_h, stride_w, dila_h, dila_w, dout_batch);
        }
        //! add bias
        if (flag_bias) {
            fill_bias_relu(dout_batch, bias, chout, wout * hout, flag_bias, flag_relu);
        }
    }
}

SaberStatus test_arm_deconv(int n, int c, int h, int w, \
    int ch_out, int kernel, int stride, int pad, \
    int dila, int group, bool flag_bias, bool flag_relu, \
    int thread_num, int cluster_id) {

    double to = 0;
    double min_time = 1000000;
    SaberTimer<ARM> t1;

    Context<ARM> ctx1;
    ctx1.set_run_mode(PowerMode(cluster_id), thread_num);
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    TensorHf4 tout_basic;
    TensorHf4 tout_saber;

    TensorHf4 thin;
    thin.re_alloc(Shape({n, c, h, w}), AK_FLOAT);

    std::vector<TensorHf4*> tin;
    std::vector<TensorHf4*> tvout_saber;

    tin.push_back(&thin);
    tvout_saber.push_back(&tout_saber);

    int num = n;
    int chin = c;
    int hin = h;
    int win = w;

    LOG(INFO) << "deconv param: " << " img_num = " << num << " in_channels = " << chin \
        << " img_h = " << hin << " img_w = " << win << " group = " << group << " pad = " \
        << pad << " stride = " << stride << " dilation = " << dila << " kernel = " \
        << kernel << " out_channels = " << ch_out << " bias flag = " << (flag_bias? "true" : "false ") \
        << " relu flag = " << (flag_relu ? "true" : "false");

    int kernel_exten = dila * (kernel - 1) + 1;
    int hout = (h - 1) * stride + kernel_exten - 2 * pad;

    kernel_exten = dila * (kernel - 1) + 1;
    int wout = (w - 1) * stride + kernel_exten - 2 * pad;

    if (hout <=0 || wout <= 0) {
        return SaberSuccess;
    }

    Shape shape_out({num, ch_out, hout, wout});

    Shape shw({ch_out/group, chin, kernel, kernel});
    Shape shb({1, ch_out, 1, 1});
    TensorHf4 pweiht(shw);
    TensorHf4 pweihtb(shw);
    TensorHf4 pbias;

    fill_tensor_rand(thin, -1.f, 1.f);
    fill_tensor_rand(pweiht, -1.f, 1.f);

//    fill_tensor_const(thin, 1.f);
//    fill_tensor_const(pweiht, 1.f);
//    fill_tensor_const(pbias, 1.f);

    TensorHf4* bias_ptr = nullptr;
    if (flag_bias) {
        pbias.re_alloc(shb);
        fill_tensor_rand(pbias, -1.f, 1.f);
    }
    std::vector<float> scale(ch_out, 1.f);
    const float* din = static_cast<const float*>(thin.data());

    if (g_compare_result) {
        LOG(INFO) << "run basic deconv for precision comparation";
        tout_basic.re_alloc(shape_out);
        float* dout = static_cast<float*>(tout_basic.mutable_data());
        deconv_basic(din, dout, num, ch_out, hout, wout, chin, hin, win, \
            static_cast<const float*>(pweiht.data()), static_cast<const float*>(pbias.data()), \
            group, kernel, kernel, stride, stride, \
            dila, dila, pad, pad, flag_bias, flag_relu);
//        print_tensor(tout_basic);
    }

    Deconv<ARM, AK_FLOAT> deconv;

    ConvParam<ARM> param(group, pad, pad, stride, stride, dila, dila, &pweiht, &pbias);
    if (flag_relu){
        ActivationParam<ARM> act_param(Active_relu);
        param.activation_param = act_param;
    }

    deconv.compute_output_shape(tin, tvout_saber, param);

    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    LOG(INFO) << "saber output shape: " << sh_out_saber[0] << ", " << sh_out_saber[1] << ", " \
        << sh_out_saber[2] << ", " << shape_out[3];
    //CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

//    LOG(INFO) << "saber deconv impl init";
    CHECK_EQ(deconv.init(tin, tvout_saber, param, SPECIFY, SABER_IMPL, ctx1), SaberSuccess) << "Saber deconv init failed";

    //! compute
//    LOG(INFO) << "saber conv compute";
    to = 0;

    for (int i = 0; i < g_test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        deconv(tin, tvout_saber, param, ctx1);
        //tvout_saber[0]->record_event(ctx1.get_compute_stream());
        //tvout_saber[0]->sync();
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "saber deconv running time, ave: " << to / g_test_iter << ", min time: " << min_time;
//    print_tensor(tout_saber);

    if (g_compare_result) {
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host((const float*)tout_basic.data(), (const float*)tout_saber.data(),
                    tout_basic.valid_size(), max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabsf(max_ratio) > 1e-4f) {
            LOG(INFO) << "basic result:";
            print_tensor(tout_basic);
            LOG(INFO) << "saber result:";
            print_tensor(tout_saber);
            return SaberInvalidValue;
        }
//        CHECK_EQ(fabsf(max_ratio) < 1e-4f, true) << "compute result error";
    }
    return SaberSuccess;
}

TEST(TestSaberFunc, test_deconv_custom_size) {

    int num = g_num;
    int chin = g_ch_in;
    int hin = g_h_in;
    int win = g_w_in;

    int dilation = g_dila;
    int chout = g_ch_out;

    test_arm_deconv(num, chin, hin, win, chout, g_kernel, g_stride, g_pad, \
        dilation, g_group, g_flag_bias, g_flag_relu, g_threads, g_cluster);
}

TEST(TestSaberFunc, fp32_deconv_basic_test) {

    if (g_basic_test) {
    for (auto& n : {1, 2}) {
    for (auto& c : {1, 3, 8, 15}) {
    for (auto& cout : {1, 3, 8, 16}) {
    for (auto& h : {8, 15, 28, 32, 38, 75}) {
    for (auto& kh : {2, 3, 4}) {
    for (auto& stride : {1, 2}) {
    for (auto &dila : {1, 2}) {
    for (auto &g : {1, 2}) {
    for (auto &bias : {false, true}) {
    for (auto &relu : {false, true}) {
    for (auto &threads : {1, 2, 4}) {
        int w = h;
        int group = g;
        if (c % g != 0 || cout % g != 0) {
            group = 1;
        }
        int pad = kh / 2;
        auto flag = test_arm_deconv(n, c, h, w, cout, kh, stride, pad, dila, group, bias, relu, threads, 0);
        if (flag == SaberSuccess) {
            LOG(INFO) << "test fp32 depthwise conv: batchsize: " << n << ", channel: " << c << ", h & w: " << h << \
                "num_out: " << cout << ", group:" << group << ", kernel: " << kh << ", stride: " << stride << \
                ", pad: " << pad << ", dila: " << dila << \
                ", bias: " << (bias? "true" : "false") << ", relu: " << (relu? "true" : "false") << ", threads: " << \
                threads << ", cluster: " << g_cluster << " passed!!";
        } else {
            LOG(FATAL) << "test fp32 depthwise conv: batchsize: " << n << ", channel: " << c << ", h & w: " << h << \
                "num_out: " << cout << ", group:" << group << ", kernel: " << kh << ", stride: " << stride << \
                ", pad: " << pad << ", dila: " << dila << \
                ", bias: " << (bias? "true" : "false") << ", relu: " << (relu? "true" : "false") << ", threads: " << \
                threads << ", cluster: " << g_cluster << " failed!!";
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


int main(int argc, const char** argv){
    Env<ARM>::env_init();
    LOG(INFO) << "usage: ./" << argv[0] << " basic_test cluster  threads  test_iter " << \
                " compare_result flag_bias flag_relu num ch_in h_in w_in ch_out group" << \
                " kernel pad stride dila";
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
            LOG(ERROR) << "usage: ./" << argv[0] << " basic_test cluster  threads  test_iter " << \
                " compare_result flag_bias flag_relu num ch_in h_in w_in ch_out group" << \
                " kernel pad stride dila";
            return 0;
        }
        g_num = atoi(argv[8]);
        g_ch_in = atoi(argv[9]);
        g_h_in = atoi(argv[10]);
        g_w_in = atoi(argv[11]);
        g_ch_out = atoi(argv[12]);
        g_group = atoi(argv[13]);
        g_kernel = atoi(argv[14]);
        g_pad = atoi(argv[15]);
        g_stride = atoi(argv[16]);
        g_dila = atoi(argv[17]);
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

