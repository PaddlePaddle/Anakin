#include "test_lite.h"
#include "saber/lite/funcs/neon/impl/gemm_prepacked_int8.h"
using namespace anakin::saber;
using namespace anakin::saber::lite;
int cluster = 0;
int threads = 1;

bool Basic_test = false;

int M = 1024;
int N = 1024;
int K = 1024;
bool traA = false;
bool traB = false;
bool flag_relu = false;
bool flag_bias = false;
ARMArch flag_arch = A73;
int test_iter = 1;
bool COMPARE_RESULT = false;
typedef Tensor<CPU> TensorHf4;

SaberStatus test_arm_sgemm(int M, int N, int K, bool tra, bool trb, bool flag_bias, bool flag_relu, int in_th) {
    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;
    Context ctx1;
    PowerMode mode = (PowerMode)cluster;
    ctx1.set_run_mode(mode, in_th);
    //ctx1.set_arch(flag_arch);
    //LOG(INFO) << "CPU ARCH: A" << flag_arch;
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << in_th;
#endif
    }
    Shape sha(M, K);
    Shape shb(K, N);
    Shape shc(M, N);
    TensorHf4 ta;
    TensorHf4 tb;
    TensorHf4 tbias;
    ta.re_alloc(sha, AK_INT8);
    tb.re_alloc(shb, AK_INT8);
    tbias.re_alloc(Shape(M), AK_INT32);
    fill_tensor_rand(ta, -127, 127);
//    fill_tensor_const(ta, 1);
    fill_tensor_rand(tb, -127, 127);
//    fill_tensor_const(tb, 1);
    fill_tensor_rand(tbias, -65536, 65535);
//    fill_tensor_const(tbias, 1);
//    print_tensor(ta);
//    print_tensor(tb);
//    print_tensor(tbias);
    TensorHf4 tout_basic;
    TensorHf4 tout_saber;
    tout_saber.re_alloc(shc, AK_INT32);
    int m = M;
    int n = N;
    int k = K;
    LOG(INFO) << "sgemm M: " << m << ", N: " << n << ", K: " << k;
    LOG(INFO) << "transA: " << (tra? "true" : "false") << ", transB: " << (trb? "true" : "false");
    LOG(INFO) << "relu: " << (flag_relu? "true" : "false") << ", bias: " << (flag_bias? "true" : "false");
    LOG(INFO) << "test iter: " << test_iter;
    LOG(INFO) << "compare result with basic sgemm: " << (COMPARE_RESULT? "true" : "false");
    const char* da = static_cast<const char*>(ta.data());
    const char* db = static_cast<const char*>(tb.data());
    if (COMPARE_RESULT) {
        LOG(INFO) << "run basic conv for precision comparation";
        tout_basic.re_alloc(shc, AK_INT32);
        int* dc_basic = static_cast<int*>(tout_basic.mutable_data());
        basic_gemm(m, n, k, da, db, static_cast<const int*>(tbias.data()), \
            dc_basic, 1, 0, tra, trb, flag_bias, flag_relu);
//        LOG(WARNING) << "basic result";
//        print_tensor(tout_basic);
    }
    long long ops = m * n * k;
    int* dc_saber = static_cast<int*>(tout_saber.mutable_data());
    to = 0;
    min_time = 1000000;
    int hblock = get_hblock_int8(ctx1.get_arch());
    int round_up_a = ((hblock + m - 1) / hblock) * hblock;
    TensorHf4 tpackedA(Shape(K, round_up_a), AK_INT8);
    //fill_tensor_const(tpackedA, 1);
    int lda = k;
    if (tra) {
        lda = m;
    }
    prepackA_int8(static_cast<char*>(tpackedA.mutable_data()), da, lda, 0, m, 0, k, tra, &ctx1);
    //! compute
    LOG(INFO) << "saber sgemm compute";
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        gemm_prepack_int8(static_cast<const char*>(tpackedA.data()), db, \
            static_cast<const int*>(tbias.data()), dc_saber, m, n, k, flag_bias, flag_relu, trb, &ctx1);
        t1.end();
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "saber packed gemm running time, ave: " << to / test_iter << ", min time: " << min_time;
    LOG(WARNING) << "mean gops: " << 0.000001f * ops * test_iter / to \
        << " GFLOPS, max gops: " << 0.000001f * ops / min_time << " GFLOPS";
    if (COMPARE_RESULT) {
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        if (fabs(max_ratio) > 1e-4f) {
            TensorHf4 tdiff(tout_basic.valid_shape(), AK_INT32);
            tensor_diff(tout_basic, tout_saber, tdiff);
            LOG(WARNING) << "basic result";
            print_tensor(tout_basic);
            LOG(WARNING) << "saber result";
            print_tensor(tout_saber);
            LOG(WARNING) << "diff tensor";
            print_tensor(tdiff);
        }
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabs(max_ratio) > 1e-4f) {
            return SaberInvalidValue;
        }
    }
    return SaberSuccess;
}
TEST(TestSaberLite, test_func_sgemm_prepacked) {
    if (Basic_test) {
        LOG(INFO) << "run basic sgemm test";
        for (auto& m : {1, 8, 16, 111, 256, 397, 512, 777, 1024}) {
        for (auto& n : {1, 3, 13, 141, 256, 345, 512, 789, 1024}) {
        for (auto& k : {1, 4, 15, 59, 128, 234, 512, 678, 1024}) {
        for (auto& tra : {false, true}) {
        for (auto& trb : {false, true}) {
        for (auto& flag_bias : {false, true}) {
        for (auto& flag_relu : {false, true}) {
        for (auto& th : {1, 2, 4}) {
            SaberStatus flag = test_arm_sgemm(m, n, k, tra, trb, flag_bias, flag_relu, th);
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
}
TEST(TestSaberLite, test_func_sgemm_prepacked_custom) {
    if (test_arm_sgemm(M, N, K, traA, traB, flag_bias, flag_relu, threads) == SaberSuccess) {
        LOG (INFO) << "test m = " << M << ", n=" << N << ", k=" << K << \
            ", bias: " << (flag_bias ? "true" : "false") << ", relu: " << \
            (flag_relu ? "true" : "false") << ", trans A: " << (traA ? "true" : "false") << \
            ", trans B: " << (traB ? "true" : "false") << " passed";
    } else {
        LOG (FATAL) << "test m = " << M << ", n=" << N << ", k=" << K << \
            ", bias: " << (flag_bias ? "true" : "false") << ", relu: " << \
            (flag_relu ? "true" : "false") << ", trans A: " << (traA ? "true" : "false") << \
            ", trans B: " << (traB ? "true" : "false") << " failed";
    }
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
    if (argc > 4) {
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
    if (argc > 13) {
        if (atoi(argv[13]) > 0) {
            flag_arch = A72;
        } else {
            flag_arch = A73;
        }
    }
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
