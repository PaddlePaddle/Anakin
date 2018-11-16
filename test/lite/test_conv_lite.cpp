#include "test_lite.h"
#include "saber/lite/funcs/saber_conv.h"
#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

int g_cluster = 0;
int g_threads = 1;
int g_test_iter = 2;
bool g_basic_test = false;
bool g_compare_result = true;
bool g_flag_relu = true;
bool g_flag_bias = true;

int g_num = 1;
int g_ch_in = 32;
int g_h_in = 112;
int g_w_in = 112;

int g_ch_out = 32;
int g_group = 32;
int g_kw = 3;
int g_pad_w = 1;
int g_stride_w = 1;
int g_dila_w = 1;
int g_kh = 3;
int g_pad_h = 1;
int g_stride_h = 1;
int g_dila_h = 1;

typedef Tensor<CPU> TensorHf4;

SaberStatus test_arm_conv(int n, int c, int h, int w, \
    int ch_out, int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h, \
    int dila_w, int dila_h, int group, bool is_bias, bool is_relu, int thread_num, int cluster_id) {

    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;

    Context ctx1;
    PowerMode mode = (PowerMode)cluster_id;
    ctx1.set_run_mode(mode, thread_num);
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

    Shape shin = {n, c, h, w};
    TensorHf4 thin;

    thin.re_alloc(shin, AK_FLOAT);

    std::vector<TensorHf4*> tvin;
    std::vector<TensorHf4*> tvout_saber;

    tvin.push_back(&thin);
    tvout_saber.push_back(&tout_saber);

    LOG(INFO) << "conv param: ";
    LOG(INFO) << " img_num = " << n;
    LOG(INFO) << " in_channels = " << c;
    LOG(INFO) << " img_h = " << h;
    LOG(INFO) << " img_w = " << w;
    LOG(INFO) << " group = " << group;
    LOG(INFO) << " pad_width = " << pad_w;
    LOG(INFO) << " pad_height = " << pad_h;
    LOG(INFO) << " stride_width = " << stride_w;
    LOG(INFO) << " stride_height = " << stride_h;
    LOG(INFO) << " dilation_w = " << dila_w;
    LOG(INFO) << " dilation_h = " << dila_h;
    LOG(INFO) << " kernel_w = " << kernel_w;
    LOG(INFO) << " kernel_h = " << kernel_h;
    LOG(INFO) << " out_channels = " << ch_out;
    LOG(INFO) << " bias flag = " << (is_bias? "true" : "false");
    LOG(INFO) << " relu flag = " << (is_relu? "true" : "false");

    int kernel_exten = dila_h * (kernel_h - 1) + 1;
    int hout = (h + 2 * pad_h - kernel_exten) / stride_h + 1;

    kernel_exten = dila_w * (kernel_w - 1) + 1;
    int wout = (w + 2 * pad_w - kernel_exten) / stride_w + 1;

    Shape shape_out{n, ch_out, hout, wout};

    Shape shw{ch_out, c / group, kernel_h, kernel_w};
    Shape shb{1, ch_out, 1, 1};
    TensorHf4 pweiht(shw);
    TensorHf4 pbias(shb);

    fill_tensor_rand(thin, -1.f, 1.f);
    fill_tensor_rand(pweiht, -1.f, 1.f);
    fill_tensor_rand(pbias, -1.f, 1.f);

//    fill_tensor_const(thin, 1.f);
//    fill_tensor_const(pweiht, 1.f);
//    fill_tensor_const(pbias, 1.f);
//    print_tensor(pweiht);
//    print_tensor(pbias);
    TensorHf4* bias_ptr = nullptr;
    if (is_bias) {
        bias_ptr = &pbias;
    }
    const float* din = static_cast<const float*>(thin.data());

    if (g_compare_result) {
        LOG(INFO) << "run basic conv for precision comparation";
        tout_basic.re_alloc(shape_out);
        fill_tensor_const(tout_basic, 0.f);
        float* dout = static_cast<float*>(tout_basic.mutable_data());
        const float* wptr = static_cast<const float*>(pweiht.data());
        const float* bptr = nullptr;
        if (is_bias) {
            bptr = static_cast<const float*>(pbias.data());
        }
        conv_basic<float, float>(din, dout, n, ch_out, hout, wout, c, h, w, \
            wptr, bptr, group, kernel_w, kernel_h, stride_w, stride_h, \
            dila_w, dila_h, pad_w, pad_h, is_bias, is_relu);
//        print_tensor(tout_basic);
    }

    SaberConv2D conv_lite;

    Conv2DParam param(pweiht.valid_size(), ch_out, group, kernel_w, kernel_h, \
        stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, is_bias, pweiht.data(), pbias.data(), \
        false, is_relu, Active_relu, 0.f, 1.f, false, nullptr);

    conv_lite.load_param(&param);

    conv_lite.compute_output_shape(tvin, tvout_saber);

    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

    LOG(INFO) << "saber conv impl init";
    auto states = conv_lite.init(tvin, tvout_saber, ctx1);
    CHECK_EQ(states, SaberSuccess) << "Saber conv init failed";

    //! compute
    LOG(INFO) << "saber conv compute";
    to = 0;
    for (int i = 0; i < g_test_iter; ++i) {
        t1.clear();
        t1.start();
        conv_lite.dispatch(tvin, tvout_saber);
        t1.end();
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "saber conv running time, ave: " << to / g_test_iter << ", min time: " << min_time;
//    print_tensor(*tvout_saber[0]);

    if (g_compare_result) {
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabsf(max_ratio) > 1e-3f) {
            if (max_diff > 1e-4f) {
                TensorHf4 tdiff(tout_basic.valid_shape());
                tensor_diff(tout_basic, tout_saber, tdiff);
                print_tensor(tdiff);
                return SaberInvalidValue;
            }

        }
//        CHECK_EQ(fabsf(max_ratio) < 5e-4f, true) << "compute result error";
    }
    return SaberSuccess;

}

#if 1
TEST(TestSaberLite, test_conv_depthwise) {
    if (g_basic_test) {
    for (auto& batch : {1, 2, 4, 8}) {
    for (auto& c : {1, 8, 16, 32, 64}) {
    for (auto& h : {2, 3, 15, 28, 56, 112, 128, 150, 224, 300}) {
    int w = h;
    for (auto& stride : {1, 2}) {
    for (auto& flag_bias : {false, true}) {
    for (auto& flag_relu : {false, true}) {
    for (auto& th : {1, 2, 4}) {
        auto flag = test_arm_conv(batch, c, h, w, c, 3, 3, stride, stride, 1, 1, 1, 1, c, flag_bias, flag_relu, th, g_cluster);
        if (flag == SaberSuccess) {
            LOG(INFO) << "test fp32 depthwise conv: batchsize: " << batch << ", channel: " << c << ", h & w: " << h << \
                ", stride: " << stride << \
                ", bias: " << (flag_bias? "true" : "false") << ", relu: " << (flag_relu? "true" : "false") << ", threads: " << \
                th << ", cluster: " << g_cluster << " passed!!";
        } else {
            LOG(FATAL) << "test fp32 depthwise conv: batchsize: " << batch << ", channel: " << c << ", h & w: " << h << \
                ", stride: " << stride << \
                ", bias: " << (flag_bias? "true" : "false") << ", relu: " << (flag_relu? "true" : "false") << ", threads: " << \
                th << ", cluster: " << g_cluster << " failed!!";
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
TEST(TestSaberLite, test_conv_1x1s1) {
    if (g_basic_test) {
    for (auto& batch : {1, 2, 4, 8}) {
    for (auto &c : {1, 8, 16, 32, 64}) {
    for (auto& cout : {1, 16, 32, 64, 128}) {
    for (auto &g_div : {1, 2, 4}) {
    for (auto &h : {2, 3, 15, 28, 56, 112, 128, 150, 224, 300}) {
    for (auto &flag_bias : {false, true}) {
    for (auto &flag_relu : {false, true}) {
    for (auto &th : {1, 2, 4}) {

        int w = h;
        int g = g_div;
        if (g % g_div != 0) {
            g = 1;
        }
        auto flag = test_arm_conv(batch, c, h, w, cout, 1, 1, 1, 1, \
            0, 0, 1, 1, g, flag_bias, flag_relu, th, g_cluster);
        if (flag == SaberSuccess) {
            LOG(INFO) << "test fp32 1x1s1 conv: batchsize: " << batch << ", channel: "
                << c << ", h & w: " << h << ", num_out: " << cout << ", group: " << g << \
                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                << (flag_relu ? "true" : "false") << ", threads: " << \
                th << ", cluster: " << g_cluster << " passed!!";
        } else {
            LOG(FATAL) << "test fp32 1x1s1 conv: batchsize: " << batch << ", channel: "
                << c << ", h & w: " << h << ", num_out: " << cout << ", group: " << g << \
                ", bias: " << (flag_bias ? "true" : "false") << ", relu: "
                << (flag_relu ? "true" : "false") << ", threads: " << \
                th << ", cluster: " << g_cluster << " failed!!";
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
TEST(TestSaberLite, test_conv_fp32_costom_size) {
    auto flag = test_arm_conv(g_num, g_ch_in, g_h_in, g_w_in, g_ch_out, g_kw, g_kh, g_stride_w, g_stride_h, \
            g_pad_w, g_pad_h, g_dila_w, g_dila_h, g_group, g_flag_bias, g_flag_relu, g_threads, g_cluster);
    if (flag == SaberSuccess) {
        LOG(INFO) << "test fp32 conv: batchsize: " << g_num << ", channel: "
            << g_ch_in << ", h & w: " << g_h_in << \
            ", bias: " << (g_flag_bias ? "true" : "false") << ", relu: "
            << (g_flag_relu ? "true" : "false") << ", threads: " << \
            g_threads << ", cluster: " << g_cluster << " passed!!";
    } else {
        LOG(INFO) << "test fp32 1x1s1 conv: batchsize: " << g_num << ", channel: "
            << g_ch_in << ", h & w: " << g_h_in << \
            ", bias: " << (g_flag_bias ? "true" : "false") << ", relu: "
            << (g_flag_relu ? "true" : "false") << ", threads: " << \
            g_threads << ", cluster: " << g_cluster << " failed!!";
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
            LOG(FATAL) << "usage: ./" << argv[0] << " cluster  threads  test_iter " << \
                " compare_result flag_bias flag_relu num ch_in h_in w_in ch_out group" << \
                " kernel pad stride dila [kernel_h] [pad_h] [stride_h] [dila_h]";
            return -1;
        }
        g_num = atoi(argv[8]);
        g_ch_in = atoi(argv[9]);
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

