#include "test_lite.h"
#include "saber/lite/funcs/saber_conv_pooling.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

int cluster = 0;
int threads = 4;

#define USE_COMPARE
const bool FLAG_RELU = true;

typedef Tensor<CPU> TensorHf4;
template <typename Tensor_t>
void tensor_diff(Tensor_t& t1, Tensor_t& t2, Tensor_t& tdiff) {

    typedef typename Tensor_t::Dtype dtype;
    int size1 = t1.valid_size();
    int size2 = t2.valid_size();
    int size_out = tdiff.valid_size();
    CHECK_EQ(size1, size2) << "wrong shape";
    CHECK_EQ(size1, size_out) << "wrong shape";
    const dtype* ptr1 = t1.data();
    const dtype* ptr2 = t2.data();
    dtype* ptr_out = tdiff.mutable_data();
    for (int i = 0; i < size1; ++i) {
        ptr_out[i] = ptr1[i] - ptr2[i];
    }
}

void test_arm_conv(std::vector<TensorHf4*>& tin, \
    int ch_out, int kernel, int stride, int pad, \
    int dila, int group, bool bias, int thread_num, int cluster_id) {

    int test_iter = 100;
    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;

    SaberConvPooling2D conv;

    Context ctx1;
    PowerMode mode = cluster_id == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
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

    TensorHf4* thin = tin[0];

    std::vector<TensorHf4*> tvout_saber;

    tvout_saber.push_back(&tout_saber);

    int num = tin[0]->num();
    int chin = tin[0]->channel();
    int hin = tin[0]->height();
    int win = tin[0]->width();

    LOG(INFO) << "conv param: ";
    LOG(INFO) << " img_num = " << num;
    LOG(INFO) << " in_channels = " << chin;
    LOG(INFO) << " img_h = " << hin;
    LOG(INFO) << " img_w = " << win;
    LOG(INFO) << " group = " << group;
    LOG(INFO) << " pad = " << pad;
    LOG(INFO) << " stride = " << stride;
    LOG(INFO) << " dilation = " << dila;
    LOG(INFO) << " kernel = " << kernel;
    LOG(INFO) << " out_channels = " << ch_out;

    int input_dim = tin[0]->height(); // P
    int kernel_exten = dila * (kernel - 1) + 1;
    int hout = (input_dim + 2 * pad - kernel_exten) / stride + 1;

    input_dim = tin[0]->width(); // Q
    kernel_exten = dila * (kernel - 1) + 1;
    int wout = (input_dim + 2 * pad - kernel_exten) / stride + 1;

    Shape shape_out{num, ch_out, 1, 1};

    Shape shw{ch_out, chin / group, kernel, kernel};
    Shape shb{1, ch_out, 1, 1};
    TensorHf4 pweiht(shw);
    TensorHf4 pbias(shb);

    fill_tensor_rand(pweiht, -1.f, 1.f);
    fill_tensor_rand(pbias, -1.f, 1.f);

    //fill_tensor_host_const(pweiht, 1.f);
    //fill_tensor_host_const(pbias, 1.f);

    TensorHf4* bias_ptr = nullptr;
    if (bias) {
        bias_ptr = &pbias;
    }
    std::vector<float> scale(ch_out, 1.f);
    SaberConvPooling2D conv_lite;
    ConvPool2DParam param(pweiht.valid_size(), ch_out, group, \
        kernel, kernel, stride, stride, pad, pad, dila, dila, bias, AK_FLOAT, pweiht.data(), scale.data(), pbias.data(), \
        false, true, Active_relu, 0.f, 1.f, false, nullptr, \
        Pooling_average_include_padding, true, 1, 1, 1, 1, 1, 1, 1);
//    conv_lite.load_param(pweiht.valid_size(), ch_out, group, \
//        kernel, kernel, stride, stride, pad, pad, dila, dila, bias, Active_relu, true, \
//        Pooling_average_include_padding, true, 1, 1, 1, 1, 1, 1, pweiht.data(), pbias.data());
    LITE_CHECK(conv_lite.load_param(&param));

    conv_lite.compute_output_shape(tin, tvout_saber);
    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

    LOG(INFO) << "saber conv impl init";
    CHECK_EQ(conv_lite.init(tin, tvout_saber, ctx1), SaberSuccess) << "init error";

    //! compute
    LOG(INFO) << "saber conv compute";
    to = 0;

    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        conv_lite.dispatch(tin, tvout_saber);
        t1.end();
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "saber conv running time, ave: " << to / test_iter << ", min time: " << min_time;
    //print_tensor_host(*tvout_saber[0]);
}

#if 1
TEST(TestSaberLite, test_conv_act_pooling) {

    int num = 1;
    int chin = 32;
    int hin = 112;
    int win = 112;

    int group = chin;
    int pad = 1;
    int stride = 2;
    int dilation = 1;
    int kernel = 3;
    int chout = chin;

    bool bias_term = true;

    Shape shape_in(num, chin, hin, win);

    TensorHf4 tdin;

    tdin.re_alloc(shape_in);
    fill_tensor_const(tdin, 1.f);

    std::vector<TensorHf4*> tin;
    tin.push_back(&tdin);

    test_arm_conv(tin, chout, kernel, stride, pad, dilation, group, bias_term, threads, cluster);
}
#endif
int main(int argc, const char** argv){
    Env::env_init();

    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

