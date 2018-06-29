#include "funcs/conv.h"
#include "test_saber_func_test_arm.h"
#include "tensor_op.h"
#include "saber/funcs/impl/arm/impl/conv_arm_impl.h"

using namespace anakin::saber;

int cluster = 0;
int threads = 4;
int test_iter = 10;

bool compare_result = false;
bool flag_relu = false;
bool flag_bias = true;

int num = 1;
int ch_in = 32;
int h_in = 112;
int w_in = 112;

int ch_out = 64;
int group = 1;
int kernel = 3;
int pad = 1;
int stride = 1;
int dila = 1;

typedef TargetWrapper<ARM> ARM_API;
typedef Tensor<ARM, AK_FLOAT, NCHW> TensorHf4;
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
    int dila, int group, bool bias_flag, int thread_num, int cluster_id) {

    double to = 0;
    double min_time = 1000000;
    SaberTimer<ARM> t1;

    Context<ARM> ctx1;
    std::vector<int> act_ids;
    //! set runtime context
    LOG(INFO) << "set runtine context";
    std::vector<int> big_cores;
    std::vector<int> small_cores;
    for (int i = 0; i < ctx1.devs[0]._info._cluster_ids.size(); ++i) {
        if (ctx1.devs[0]._info._cluster_ids[i] == 0) {
            big_cores.push_back(ctx1.devs[0]._info._core_ids[i]);
        } else {
            small_cores.push_back(ctx1.devs[0]._info._core_ids[i]);
        }
    }

    if (cluster_id == 0) {
        if (big_cores.size() == 0) {
                    LOG(FATAL) << "big cores are not supported";
        }
        if (thread_num > big_cores.size()) {
                    LOG(WARNING) << "not enough big cores for inference";
            act_ids = big_cores;
        } else {
            for (int i = 0; i < thread_num; ++i) {
                act_ids.push_back(big_cores[i]);
            }
        }
    } else {
        if (small_cores.size() == 0) {
                    LOG(FATAL) << "small cores are not supported";
        }
        if (thread_num > small_cores.size()) {
                    LOG(WARNING) << "not enough small cores for inference";
            act_ids = small_cores;
        } else {
            for (int i = 0; i < thread_num; ++i) {
                act_ids.push_back(small_cores[i]);
            }
        }
    }
    ctx1.set_act_cores(act_ids);

    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
                LOG(INFO) << "number of threads: " << thread;
#endif
    }
    int th_id;
#pragma omp parallel private(th_id)
    {
#ifdef USE_OPENMP
        th_id = omp_get_thread_num();
#pragma omp parallel
                LOG(INFO) << "thread core ID: " << act_ids[th_id];
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
    LOG(INFO) << "bias flag = " << (bias_flag? "true" : "false");

    int input_dim = tin[0]->height(); // P
    int kernel_exten = dila * (kernel - 1) + 1;
    int hout = (input_dim + 2 * pad - kernel_exten) / stride + 1;

    input_dim = tin[0]->width(); // Q
    kernel_exten = dila * (kernel - 1) + 1;
    int wout = (input_dim + 2 * pad - kernel_exten) / stride + 1;

    Shape shape_out{num, ch_out, hout, wout};

    Shape shw{ch_out, chin / group, kernel, kernel};
    Shape shb{1, ch_out, 1, 1};
    TensorHf4 pweiht(shw);
    TensorHf4 pbias(shb);

    fill_tensor_host_rand(pweiht, -1.f, 1.f);
    fill_tensor_host_rand(pbias, -1.f, 1.f);

    //fill_tensor_host_const(pweiht, 1.f);
    //fill_tensor_host_const(pbias, 1.f);

    TensorHf4* bias_ptr = nullptr;
    if (bias_flag) {
        bias_ptr = &pbias;
    }

    ConvParam<TensorHf4> conv_param(group, pad, pad,
                                    stride, stride,
                                    dila, dila,
                                    &pweiht, bias_ptr);

    if (compare_result) {
        LOG(INFO) << "run basic conv for precision comparation";
        tout_basic.re_alloc(shape_out);
        //size_t workspace_size = sizeof(float) * num * chin * (hin + 2 * pad) * (win + 2 * pad);
        //void* work_space_data = fast_malloc(workspace_size);
        Sgemm gemmer;
        conv_arm_basic(tout_basic, *thin, pweiht.data(), pbias.data(), group, kernel, \
        kernel, stride, stride, dila, dila, pad, pad, bias_flag, flag_relu, gemmer, nullptr);
        //conv_direct_basic1(tout_basic, *thin, pweiht.data(), pbias.data(), group, kernel, \
        kernel, stride, stride, dila, dila, pad, pad, bias_flag, flag_relu, &gemmer, work_space_data);
        //fast_free(work_space_data);
        //print_tensor_host(tout_basic);
    }

    Conv<ARM, AK_FLOAT> conv_saber;

    conv_saber.compute_output_shape(tin, tvout_saber, conv_param);
    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

    LOG(INFO) << "saber conv impl init";
    SABER_CHECK(conv_saber.init(tin, tvout_saber, conv_param, SPECIFY, SABER_IMPL, ctx1));

    //! compute
    LOG(INFO) << "saber conv compute";
    to = 0;

    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        conv_saber(tin, tvout_saber, conv_param, ctx1);
        //tvout_saber[0]->record_event(ctx1.get_compute_stream());
        //tvout_saber[0]->sync();
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "saber conv running time, ave: " << to / test_iter << ", min time: " << min_time;
    //print_tensor_host(*tvout_saber[0]);

    if (compare_result) {
        double max_ratio = 0;
        double max_diff = 0;
        //TensorHf4 tdiff(tout_basic.valid_shape());
        //tensor_diff(tout_basic, tout_saber, tdiff);
        //print_tensor_host(tdiff);
        tensor_cmp_host(tout_basic.data(), tout_saber.data(), tout_basic.valid_size(), max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
    }

}

#if 0
TEST(TestSaberFuncTest, test_func_conv3x3s1_arm) {

    int num = 1;
    int chin = 3;
    int hin = 224;
    int win = 224;

    int group = 1;
    int pad = 1;
    int stride = 1;
    int dilation = 1;
    int kernel = 3;
    int chout = 64;

    bool bias_term = true;

    Shape shape_in(num, chin, hin, win);

    TensorHf4 tdin;

    tdin.re_alloc(shape_in);
    //fill_tensor_host_rand(tdin, -1.f, 1.f);
    fill_tensor_host_const(tdin, 1.f);

    std::vector<TensorHf4*> tin;
    tin.push_back(&tdin);

    test_arm_conv(tin, chout, kernel, stride, pad, dilation, group, bias_term, 4, 0);
    //LOG(WARNING) << "conv3x3s1 not support yet";
}
#endif
#if 1
TEST(TestSaberFuncTest, test_func_conv3x3s2_arm) {

    int num = 1;
    int chin = 32;
    int hin = 112;
    int win = 112;

    int group = 1;
    int pad = 1;
    int stride = 2;
    int dilation = 1;
    int kernel = 3;
    int chout = 64;

    bool bias_term = true;

    Shape shape_in(num, chin, hin, win);

    TensorHf4 tdin;

    tdin.re_alloc(shape_in);
    fill_tensor_host_rand(tdin, -1.f, 1.f);
    //fill_tensor_host_const(tdin, -1.f);
    std::vector<TensorHf4*> tin;
    tin.push_back(&tdin);

    test_arm_conv(tin, chout, kernel, stride, pad, dilation, group, bias_term, threads, cluster);
}
#endif
#if 1
TEST(TestSaberFuncTest, test_func_conv1x1s1_arm) {

    int num = 1;
    int chin = 32;
    int hin = 112;
    int win = 112;

    int group = 1;
    int pad = 0;
    int stride = 1;
    int dilation = 1;
    int kernel = 1;
    int chout = 64;

    bool bias_term = true;

    Shape shape_in(num, chin, hin, win);

    TensorHf4 tdin;

    tdin.re_alloc(shape_in);
    fill_tensor_host_rand(tdin, -1.f, 1.f);
    //fill_tensor_host_const(tdin, -1.f);
    std::vector<TensorHf4*> tin;
    tin.push_back(&tdin);

    test_arm_conv(tin, chout, kernel, stride, pad, dilation, group, bias_term, threads, cluster);
}
#endif
#if 1
TEST(TestSaberFuncTest, test_func_conv1x1s2_arm) {

    int num = 1;
    int chin = 32;
    int hin = 112;
    int win = 112;

    int group = 1;
    int pad = 0;
    int stride = 2;
    int dilation = 1;
    int kernel = 1;
    int chout = 64;

    bool bias_term = true;

    Shape shape_in(num, chin, hin, win);

    TensorHf4 tdin;

    tdin.re_alloc(shape_in);
    fill_tensor_host_rand(tdin, -1.f, 1.f);
    //fill_tensor_host_const(tdin, -1.f);
    std::vector<TensorHf4*> tin;
    tin.push_back(&tdin);

    test_arm_conv(tin, chout, kernel, stride, pad, dilation, group, bias_term, threads, cluster);
}
#endif
#if 1
TEST(TestSaberFuncTest, test_func_depthwise_conv3x3s1_arm) {

    int num = 1;
    int chin = 32;
    int hin = 112;
    int win = 112;

    int group = chin;
    int pad = 1;
    int stride = 1;
    int dilation = 1;
    int kernel = 3;
    int chout = chin;

    bool bias_term = true;

    Shape shape_in(num, chin, hin, win);

    TensorHf4 tdin;

    tdin.re_alloc(shape_in);
    fill_tensor_host_rand(tdin, -1.f, 1.f);
    //fill_tensor_host_const(tdin, 1.f);
    std::vector<TensorHf4*> tin;
    tin.push_back(&tdin);

    test_arm_conv(tin, chout, kernel, stride, pad, dilation, group, bias_term, threads, cluster);
}
#endif
#if 1
TEST(TestSaberFuncTest, test_func_depthwise_conv3x3s2_arm) {

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
    fill_tensor_host_rand(tdin, -1.f, 1.f);
    //fill_tensor_host_const(tdin, 1.f);

    std::vector<TensorHf4*> tin;
    tin.push_back(&tdin);

    test_arm_conv(tin, chout, kernel, stride, pad, dilation, group, bias_term, threads, cluster);
}
#endif

#if 1
TEST(TestSaberFuncTest, test_conv_custom_size) {

    int chin = ch_in;
    int hin = h_in;
    int win = w_in;

    int dilation = dila;
    int chout = ch_out;

    bool bias_term = true;

    Shape shape_in(num, chin, hin, win);

    TensorHf4 tdin;

    tdin.re_alloc(shape_in);
    fill_tensor_host_rand(tdin, -1.f, 1.f);
    //fill_tensor_host_const(tdin, 1.f);

    std::vector<TensorHf4*> tin;
    tin.push_back(&tdin);

    test_arm_conv(tin, chout, kernel, stride, pad, dilation, group, bias_term, threads, cluster);
}
#endif

int main(int argc, const char** argv){
    anakin::saber::Env<ARM>::env_init();

    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }
    if (argc >= 4) {
        test_iter = atoi(argv[3]);
    }
    if (argc >= 5) {
        compare_result = atoi(argv[4]) > 0;
    }
    if (argc >= 6) {
        flag_bias = atoi(argv[5]) > 0;
    }
    if (argc >= 7) {
        flag_relu = atoi(argv[6]) > 0;
    }
    if(argc >= 8) {
        if (argc < 17) {
            LOG(ERROR) << "usage: ./" << argv[0] << " cluster  threads  test_iter " << \
                " compare_result flag_bias flag_relu num ch_in h_in w_in ch_out group" << \
                " kernel pad stride dila";
            return 0;
        }
        num = atoi(argv[7]);
        ch_in = atoi(argv[8]);
        h_in = atoi(argv[9]);
        w_in = atoi(argv[10]);
        ch_out = atoi(argv[11]);
        group = atoi(argv[12]);
        kernel = atoi(argv[13]);
        pad = atoi(argv[14]);
        stride = atoi(argv[15]);
        dila = atoi(argv[16]);
    }

    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

