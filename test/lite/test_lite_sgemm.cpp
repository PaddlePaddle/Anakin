#include "test_lite.h"
#include "saber/lite/funcs/neon/impl/sgemm_arm.h"
#include "saber/lite/funcs/neon/impl/sgemm_prepacked.h"
using namespace anakin::saber;
using namespace anakin::saber::lite;

int cluster = 0;
int threads = 4;

bool Basic_test = false;

int M = 512;
int N = 512;
int K = 512;
bool traA = false;
bool traB = false;
bool flag_relu = false;
bool flag_bias = false;

int test_iter = 1;

bool COMPARE_RESULT = false;

typedef Tensor<CPU> TensorHf4;

SaberStatus test_arm_sgemm(int M, int N, int K, bool tra, bool trb, bool flag_bias, bool flag_relu) {

    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;

    Context ctx1;
    PowerMode mode = (PowerMode)cluster;
    ctx1.set_run_mode(mode, threads);
        LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    Shape sha(1, 1, M, K);
    Shape shb(1, 1, N, K);
    Shape shc(1, 1, M, N);

    TensorHf4 ta;
    TensorHf4 tb;

    TensorHf4 tbias;

    ta.reshape(sha);
    tb.reshape(shb);
    tbias.reshape(Shape(M));

    fill_tensor_rand(ta, -1.f, 1.f);
    fill_tensor_rand(tb, -1.f, 1.f);

    TensorHf4 tout_basic;
    TensorHf4 tout_saber;

    tout_saber.reshape(shc);

    int m = M;
    int n = N;
    int k = K;

    LOG(INFO) << "sgemm M: " << m << ", N: " << n << ", K: " << k;
    LOG(INFO) << "transA: " << (tra? "true" : "false") << ", transB: " << (trb? "true" : "false");
    LOG(INFO) << "relu: " << (flag_relu? "true" : "false") << ", bias: " << (flag_bias? "true" : "false");
    LOG(INFO) << "test iter: " << test_iter;
    LOG(INFO) << "compare result with basic sgemm: " << (COMPARE_RESULT? "true" : "false");

    const float* da = static_cast<const float*>(ta.data());
    const float* db = static_cast<const float*>(tb.data());

    if(COMPARE_RESULT) {
        LOG(INFO) << "run basic conv for precision comparation";
        tout_basic.reshape(shc);
        float* dc_basic = static_cast<float*>(tout_basic.mutable_data());
        basic_gemm(m, n, k, da, db, static_cast<const float*>(tbias.data()), dc_basic, 1.f, 0.f, traA, traB, flag_relu, flag_bias);
        //print_tensor(tout_basic);
    }
    //! sgemm init
    int l1_cache = ctx1.l1_cache_size();
    int l2_cache = ctx1.l2_cache_size();
    //! if L1 cache size is not provided, set to 32K
    l1_cache = l1_cache > 0? l1_cache : 32 * 1024;
    //! if L2 cache size is not provided, set to 2M
    l2_cache = l2_cache > 0? l2_cache : 512 * 1024;
    Sgemm gemmer;
    gemmer.init(l1_cache, l2_cache, m, n, k, traA, traB, threads);
    //! compute
    LOG(INFO) << "saber sgemm compute";
    to = 0;
    int lda, ldb, ldc;
    if (traA) {
        lda = m;
    } else {
        lda = k;
    }
    if (traB) {
        ldb = k;
    } else {
        ldb = n;
    }
    ldc = n;

    long long ops = m * n * k;

    float* dc_saber = static_cast<float*>(tout_saber.mutable_data());
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        gemmer(da, lda, db, ldb, dc_saber, ldc, 1.f, 0.f);
        t1.end();
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "saber gemm running time, ave: " << to / test_iter << ", min time: " << min_time;
    LOG(WARNING) << "mean gops: " << 0.000001f * ops * test_iter / to << " GFLOPS, max gops: " \
        << 0.000001f * ops / min_time << " GFLOPS";
    //print_tensor(tout_saber);

    if (COMPARE_RESULT) {
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        if (fabs(max_ratio) > 1e-4f) {
            TensorHf4 tdiff(tout_basic.valid_shape());
            tensor_diff(tout_basic, tout_saber, tdiff);
            print_tensor(tdiff);
        }
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabs(max_ratio) > 1e-4f) {
            return SaberInvalidValue;
        }
    }
    return SaberSuccess;
}

TEST(TestSaberLite, test_func_sgemm_arm) {
    if (Basic_test) {
        LOG(INFO) << "run basic sgemm test";
        for (auto& m : {1, 8, 16, 111, 256, 397, 512, 777, 1024}) {
            for (auto& n : {1, 3, 13, 141, 256, 345, 512, 789, 1024}) {
                for (auto& k : {1, 4, 15, 59, 128, 234, 512, 678, 1024}) {
                    for (auto& tra : {false, true}) {
                        for (auto& trb : {false, true}) {
                            for (auto& flag_bias : {false, true}) {
                                for (auto& flag_relu : {false, true}) {
                                    SaberStatus flag = test_arm_sgemm(m, n, k, traA, traB, flag_bias, flag_relu);
                                    if (flag == SaberSuccess) {
                                        LOG(INFO) << "test m = " << m << ", n=" << n << ", k=" << k << \
                                            ", bias: " << (flag_bias? "true" : "false") << ", relu: " << \
                                            (flag_relu? "true" : "false") << ", trans A: " << (tra? "true" : "false") << \
                                            ", trans B: " << (trb? "true" : "false") << " passed";
                                    } else {
                                        LOG(FATAL) << "test m = " << m << ", n=" << n << ", k=" << k << \
                                            ", bias: " << (flag_bias? "true" : "false") << ", relu: " << \
                                            (flag_relu? "true" : "false") << ", trans A: " << (tra? "true" : "false") << \
                                            ", trans B: " << (trb? "true" : "false") << " failed";
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

TEST(TestSaberLite, test_func_sgemm_arm_custom) {

    test_arm_sgemm(M, N, K, traA, traB, flag_bias, flag_relu);
    LOG(INFO) << "test m = " << M << ", n=" << N << ", k=" << K << "passed";

}

int main(int argc, const char** argv){
    anakin::saber::lite::Env::env_init();

    LOG(ERROR) << "usage: ./" << argv[0] << " [do_basic_test] [cluster]  [threads]  [m] [n]  [k] [transA] [transB] [relu] [bias] [test iter] [compare result]";

    if (argc > 1) {
        Basic_test = atoi(argv[1]) > 0;
    }

    if (argc > 2) {
        cluster = atoi(argv[2]);
    }
    if (argc > 3) {
        threads = atoi(argv[3]);
    }
    if(argc > 4) {
        if (argc < 10) {
            LOG(ERROR) << "usage: ./" << argv[0] << " [do_basic_test] [cluster]  [threads]  [m] [n]  [k] [transA] [transB] [relu] [bias] [test iter] [compare result]";
            return 0;
        }
        M = atoi(argv[4]);
        N = atoi(argv[5]);
        K = atoi(argv[6]);
        traA = atoi(argv[7]) > 0;
        traB = atoi(argv[8]) > 0;
        flag_relu = atoi(argv[9]) > 0;
        flag_bias = atoi(argv[10]) > 0;
    }
    if (argc > 11) {
        test_iter = atoi(argv[11]);
    }
    if (argc > 12) {
        COMPARE_RESULT = atoi(argv[12]) > 0;
    }
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

