#include "test_lite.h"
#include "saber/lite/funcs/saber_deconv.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

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

typedef Tensor<CPU> TensorHf4;

SaberStatus test_arm_deconv(int n, int c, int h, int w, \
    int ch_out, int kernel, int stride, int pad, \
    int dila, int group, bool flag_bias, bool flag_relu, \
    int thread_num, int cluster_id) {

    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;

    Context ctx1;
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
    thin.re_alloc(Shape(n, c, h, w), AK_FLOAT);

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
        << kernel << " out_channels = " << ch_out << " bias flag = " << (flag_bias? "true" : "false");

    int kernel_exten = dila * (kernel - 1) + 1;
    int hout = (h - 1) * stride + kernel_exten - 2 * pad;

    kernel_exten = dila * (kernel - 1) + 1;
    int wout = (w - 1) * stride + kernel_exten - 2 * pad;

    if (hout <=0 || wout <= 0) {
        return SaberSuccess;
    }

    Shape shape_out{num, ch_out, hout, wout};

    Shape shw{ch_out, chin / group, kernel, kernel};
    Shape shb{1, ch_out, 1, 1};
    TensorHf4 pweiht(shw);
    TensorHf4 pbias(shb);

    fill_tensor_rand(thin, -1.f, 1.f);
    fill_tensor_rand(pweiht, -1.f, 1.f);
    fill_tensor_rand(pbias, -1.f, 1.f);

//    fill_tensor_const(thin, 1.f);
//    fill_tensor_const(pweiht, 1.f);
//    fill_tensor_const(pbias, 1.f);

    TensorHf4* bias_ptr = nullptr;
    if (flag_bias) {
        bias_ptr = &pbias;
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

    SaberDeconv2D deconv_lite;

    Conv2DParam param(pweiht.valid_size(), ch_out, group, kernel, kernel, \
        stride, stride, pad, pad, dila, dila, flag_bias, AK_FLOAT, pweiht.data(), scale.data(), pbias.data(), false, flag_relu, Active_relu);

    deconv_lite.load_param(&param);
    deconv_lite.compute_output_shape(tin, tvout_saber);

    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

//    LOG(INFO) << "saber deconv impl init";
    CHECK_EQ(deconv_lite.init(tin, tvout_saber, ctx1), SaberSuccess) << "Saber deconv init failed";

    //! compute
//    LOG(INFO) << "saber conv compute";
    to = 0;

    for (int i = 0; i < g_test_iter; ++i) {
        t1.clear();
        t1.start();
        deconv_lite.dispatch(tin, tvout_saber);
        //tvout_saber[0]->record_event(ctx1.get_compute_stream());
        //tvout_saber[0]->sync();
        t1.end();
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
        tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabsf(max_ratio) > 1e-4f) {
            TensorHf4 tdiff(tout_basic.valid_shape());
            tensor_diff(tout_basic, tout_saber, tdiff);
            LOG(INFO) << "basic result:";
            print_tensor(tout_basic);
            LOG(INFO) << "saber result:";
            print_tensor(tout_saber);
            LOG(INFO) << "diff:";
            print_tensor(tdiff);
            return SaberInvalidValue;
        }
//        CHECK_EQ(fabsf(max_ratio) < 1e-4f, true) << "compute result error";
    }
//    printf("out mean: %.5f\n", tensor_mean(tout_saber));
    return SaberSuccess;
}

TEST(TestSaberLite, test_deconv_custom_size) {

    int num = g_num;
    int chin = g_ch_in;
    int hin = g_h_in;
    int win = g_w_in;

    int dilation = g_dila;
    int chout = g_ch_out;

    test_arm_deconv(num, chin, hin, win, chout, g_kernel, g_stride, g_pad, \
        dilation, g_group, g_flag_bias, g_flag_relu, g_threads, g_cluster);
}

TEST(TestSaberLite, fp32_deconv_basic_test) {

    if (g_basic_test) {
    for (auto& n : {1, 2}) {
    for (auto& c : {1, 3, 8, 15}) {
    for (auto& cout : {1, 3, 8, 16}) {
    for (auto& h : {1, 3, 8, 15, 28, 32, 38, 75}) {
    for (auto& kh : {1, 2, 3, 4}) {
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
    Env::env_init();
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
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

