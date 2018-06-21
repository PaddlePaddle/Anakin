#include "saber/funcs/pooling.h"
#include "test_saber_func_test_arm.h"
#include "tensor_op.h"
#include "saber/funcs/impl/arm/impl/pooling_arm_impl.h"

using namespace anakin::saber;

int cluster = 0;
int threads = 4;
int test_iter = 10;

bool compare_result = false;
bool global_pool = false;

int num = 1;
int ch_in = 32;
int h_in = 112;
int w_in = 112;

int kernel = 2;
int pad = 0;
int stride = 2;

PoolingType type = Pooling_max;

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

void test_arm_pooling(std::vector<TensorHf4*>& tin, \
    int kernel, int stride, int pad, \
    PoolingType type, bool global, int threads, int cluster_id) {

    //int test_iter = 1000;
    double to = 0;
    double min_time = 1000000;
    SaberTimer<ARM> t1;
    SaberTimer<ARM> t2;

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
            small_cores.push_back(ctx1.devs[0]._info._cluster_ids[i]);
        }
    }

    if (cluster_id == 0) {
        if (big_cores.size() == 0) {
            LOG(FATAL) << "big cores are not supported";
        }
        if (threads > big_cores.size()) {
            LOG(WARNING) << "not enough big cores for inference";
            act_ids = big_cores;
        } else {
            for (int i = 0; i < threads; ++i) {
                act_ids.push_back(big_cores[i]);
            }
        }
    } else {
        if (small_cores.size() == 0) {
            LOG(FATAL) << "small cores are not supported";
        }
        if (threads > small_cores.size()) {
            LOG(WARNING) << "not enough small cores for inference";
            act_ids = small_cores;
        } else {
            for (int i = 0; i < threads; ++i) {
                act_ids.push_back(small_cores[i]);
            }
        }
    }
    ctx1.set_act_cores(act_ids);

    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int threads = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << threads;
#endif
    }
    int th_id;
#pragma omp parallel private(th_id)
    {
#ifdef USE_OPENMP
        th_id = omp_get_thread_num();
#pragma omp parallel
        LOG(INFO) << "thread core ID: " << big_cores[th_id];
#endif
    }

    TensorHf4 tout_basic;
    TensorHf4 tout_saber;

    TensorHf4* thin = tin[0];

    std::vector<TensorHf4*> tvout_saber;
    std::vector<TensorHf4*> tvout_basic;

    tvout_saber.push_back(&tout_saber);
    tvout_basic.push_back(&tout_basic);

    int num = tin[0]->num();
    int chin = tin[0]->channel();
    int hin = tin[0]->height();
    int win = tin[0]->width();

    LOG(INFO) << "pooling param: ";
    LOG(INFO) << " img_num = " << num;
    LOG(INFO) << " in_channels = " << chin;
    LOG(INFO) << " img_h = " << hin;
    LOG(INFO) << " img_w = " << win;
    LOG(INFO) << "kernel size = " << kernel;
    LOG(INFO) << "stride = " << stride;
    LOG(INFO) << "pad = " << pad;
    LOG(INFO) << "type = " << type;
    int wout = 1;
    int hout = 1;
    if(!global) {
        int hin = tin[0]->height(); // P
        hout = static_cast<int>(ceilf(static_cast<float>(
                             hin + 2 * pad - kernel) / stride)) + 1;
        int win = tin[0]->width(); // Q
        wout = static_cast<int>(ceilf(static_cast<float>(
                             win + 2 * pad - kernel) / stride)) + 1;
  }
   Shape shape_out{num, chin, hout, wout};
   PoolingParam<TensorHf4> pooling_param(kernel,kernel, pad, pad,
                                    stride,stride,type,global);
   //LOG(INFO) << "input tensor";
   //print_tensor_host(*tin[0]);

    if (compare_result) {
        LOG(INFO) << "run basic pooling for precision comparation";
        tout_basic.re_alloc(shape_out);
        //pooling_basic(tout_basic, *thin, type,global, kernel, \
                kernel, stride, stride, pad, pad);
        //print_tensor_host(tout_basic);
         LOG(INFO) << "basic pooling compute";
        to = 0;
        min_time = 1000000;
        for (int i = 0; i < test_iter; ++i) {
           t1.clear();
           t1.start(ctx1);
           pooling_basic(tout_basic, *thin, type,global, kernel, \
                kernel, stride, stride, pad, pad);
           tvout_basic[0]->record_event(ctx1.get_compute_stream());
           tvout_basic[0]->sync();
           t1.end(ctx1);
           to += t1.get_average_ms();
           if (t1.get_average_ms() < min_time) {
               min_time = t1.get_average_ms();
             }
        }
        LOG(INFO) << "basic pooling running time, ave: " << to / test_iter << ", min time: " << min_time;
       // print_tensor_host(tout_basic);

    }

    Pooling<ARM, AK_FLOAT> pooling_saber;

    pooling_saber.compute_output_shape(tin, tvout_saber, pooling_param);
    Shape sh_out_saber = tvout_saber[0]->valid_shape();
    LOG(INFO) << "output shape_1: " << sh_out_saber[0] << ", " << sh_out_saber[1] << ", " \
        << sh_out_saber[2] << ", " << sh_out_saber[3];
    LOG(INFO) << "output shape: " << shape_out[0] << ", " << shape_out[1] << ", " \
        << shape_out[2] << ", " << shape_out[3];
    CHECK_EQ(shape_out == sh_out_saber, true) << "compute output shape error";

    //! re_alloc mem for output tensor
    tvout_saber[0]->re_alloc(shape_out);

    LOG(INFO) << "saber pooling impl init";
    SABER_CHECK(pooling_saber.init(tin, tvout_saber, pooling_param, SPECIFY, SABER_IMPL, ctx1));

    //print_tensor_host(*thin);

    //! compute
    LOG(INFO) << "saber pooling compute";
    to = 0;
    min_time = 1000000;
    for (int i = 0; i < test_iter; ++i) {
        t2.clear();
        t2.start(ctx1);
        pooling_saber(tin, tvout_saber, pooling_param, ctx1);
        //pooling3x3s2_max(tout_saber,*thin,type,global,kernel, \
            kernel, stride, stride, pad, pad);
        tvout_saber[0]->record_event(ctx1.get_compute_stream());
        tvout_saber[0]->sync();
        t2.end(ctx1);
        to += t2.get_average_ms();
        if (t2.get_average_ms() < min_time) {
            min_time = t2.get_average_ms();
        }
    }
    LOG(INFO) << "saber pooling running time, ave: " << to / test_iter << ", min time: " << min_time;
    //print_tensor_host(tout_saber);

    if (compare_result) {
        double max_ratio = 0;
        double max_diff = 0;
        TensorHf4 tdiff(tout_basic.valid_shape());
        tensor_diff(tout_basic, tout_saber, tdiff);
        //print_tensor_host(tdiff);
        tensor_cmp_host(tout_basic.data(), tout_saber.data(), tout_basic.valid_size(), max_ratio, max_diff);

        // LOG(INFO) << "tout_basic";
        // print_tensor_host(tout_basic);
        // LOG(INFO) << "tout_saber";
        // print_tensor_host(tout_saber);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
    }
}

#if 1
TEST(TestSaberFuncTest, test_func_pooling_global_arm) {

    Shape shape_in(num, ch_in, h_in, w_in);

    TensorHf4 tdin;

    tdin.re_alloc(shape_in);
    fill_tensor_host_rand(tdin, -1.f, 1.f);
    //fill_tensor_host_const(tdin, 1.f);

    std::vector<TensorHf4*> tin;
    tin.push_back(&tdin);

    test_arm_pooling(tin, kernel, stride, pad, type, global_pool, threads, cluster);
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
        global_pool = atoi(argv[5]) > 0;
    }
    if(argc >= 7) {
        if (argc < 14) {
            LOG(ERROR) << "usage: ./" << argv[0] << " cluster  threads  test_iter " << \
                " compare_result global_pool num ch_in h_in w_in kernel pad stride pool_type";
            return 0;
        }
        num = atoi(argv[6]);
        ch_in = atoi(argv[7]);
        h_in = atoi(argv[8]);
        w_in = atoi(argv[9]);
        kernel = atoi(argv[10]);
        pad = atoi(argv[11]);
        stride = atoi(argv[12]);
        type = (PoolingType)atoi(argv[13]);
    }

    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

