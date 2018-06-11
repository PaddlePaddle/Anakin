#include "test_saber_func_test_arm.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/impl/arm/impl/sgemv_arm.h"
#include "saber/funcs/timer.h"

using namespace anakin::saber;

int cluster = 0;
int threads = 4;

int batch = 1;
int M = 100;
int K = 100;

int test_iter = 10;

bool flag_bias = false;
bool flag_relu = false;
bool COMPARE_RESULT = false;

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

template  <typename type>
void basic_gemv(int m, int k, const type* a, const type* b, type* c, const type* bias, bool flag_bias, \
    type alpha, type beta, bool flag_relu = false, bool trans_a = false) {
//#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        type sum = 0;
        for (int j = 0; j < k; ++j) {
            type av;
            if (trans_a) {
                av = a[j * m + i];
            } else {
                av = a[i * k + j];
            }
            sum += av * b[j];
        }
        //printf("sum: %0.2f, alpha: %.2f, beta: %.2f, c: %.2f, flag: %d\n", sum, alpha, beta, c[i], flag_bias);
        c[i] = alpha * sum + beta * c[i] + (flag_bias? bias[i] : 0);
        if (flag_relu) {
            c[i] = c[i] > type(0)? c[i] : type(0);
        }
    }
}

void test_arm_sgemv(const int m, const int k, bool flag_bias, bool flag_relu, int thread_num, int cluster_id) {

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

    LOG(INFO) << "sgemv M: " << M << ", K: " << K;
    //LOG(INFO) << "transA: " << (traA? "true" : "false") << ", transB: " << (traB? "true" : "false");
    LOG(INFO) << "test iter: " << test_iter;
    LOG(INFO) << "compare result with basic sgemv: " << (COMPARE_RESULT? "true" : "false");

    TensorHf4 tin;
    TensorHf4 tw, tb;
    TensorHf4 tout_basic;
    TensorHf4 tout_saber;

    tin.reshape(Shape(1, 1, 1, k));
    tw.reshape(Shape(1, 1, m, k));
    tb.reshape(Shape(1, 1, 1, m));
    tout_basic.reshape(Shape(1, 1, 1, m));
    tout_saber.reshape(Shape(1, 1, 1, m));

    fill_tensor_host_rand(tin, -1.f, 1.f);
    fill_tensor_host_rand(tw, -1.f, 1.f);
    fill_tensor_host_rand(tb, -1.f, 1.f);

    //fill_tensor_host_const(tin, 1.f);
    //fill_tensor_host_const(tw, 1.f);
    //fill_tensor_host_const(tb, 1.f);

    const float* da = tw.data();
    const float* db = tin.data();
    const float* dbias = tb.data();
    float* dc_basic = tout_basic.mutable_data();
    float* dc_saber = tout_saber.mutable_data();

    if(COMPARE_RESULT) {
        LOG(INFO) << "run basic conv for precision comparation";
        basic_gemv(m, k, da, db, dc_basic, dbias, flag_bias, 1.f, 0.f, flag_relu);
        //print_tensor_host(tout_basic);
    }
    for (int i = 0; i < 20; ++i) {
        sgemv(false, m, k, da, db, dc_saber);
    }
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        if (flag_bias) {
            if (flag_relu) {
                sgemv_bias_relu(false, m, k, da, db, dc_saber, dbias);
            } else {
                sgemv_bias(false, m, k, da, db, dc_saber, dbias);
            }

        } else {
            if (flag_relu) {
                sgemv_relu(false, m, k, da, db, dc_saber);
            } else {
                sgemv(false, m, k, da, db, dc_saber);
            }
        }
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "saber conv running time, ave: " << to / test_iter << ", min time: " << min_time;
    //print_tensor_host(tc);

    if (COMPARE_RESULT) {
        double max_ratio = 0;
        double max_diff = 0;
        //TensorHf4 tdiff(tout_basic.valid_shape());
        //tensor_diff(tout_basic, tout_saber, tdiff);
        //print_tensor_host(tdiff);
        tensor_cmp_host(dc_basic, dc_saber, tout_basic.valid_size(), max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
    }
}

TEST(TestSaberFuncTest, test_func_sgemv_arm) {

    test_arm_sgemv(M, K, flag_bias, flag_relu, threads, cluster);
    //LOG(WARNING) << "conv3x3s1 not support yet";
}

int main(int argc, const char** argv){
    anakin::saber::Env<ARM>::env_init();

    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }
    if(argc >= 4) {
        if (argc < 5) {
            LOG(ERROR) << "usage: " << argv[0] << " cluster  threads  m  k [iters] [flag_compare] [flag_bias] [flag_relu]";
            return 0;
        }
        M = atoi(argv[3]);
        K = atoi(argv[4]);
    }
    if (argc > 5) {
        test_iter = atoi(argv[5]);
    }
    if (argc > 6) {
        COMPARE_RESULT = atoi(argv[6]) > 0;
    }
    if (argc > 7) {
        flag_bias = atoi(argv[7]) > 0;
    }
    if (argc > 8) {
        flag_relu = atoi(argv[8]) > 0;
    }
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

