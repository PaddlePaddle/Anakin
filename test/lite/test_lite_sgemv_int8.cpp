#include "test_lite.h"
#include "saber/lite/funcs/neon/impl/gemv_arm_int8.h"
using namespace anakin::saber;
using namespace anakin::saber::lite;
int cluster = 0;
int threads = 1;

bool Basic_test = false;

int M = 512;
int K = 512;
bool flag_relu = false;
bool flag_bias = false;
ARMArch flag_arch = A73;
int test_iter = 1;
bool COMPARE_RESULT = true;
typedef Tensor<CPU> TensorHf4;

SaberStatus test_arm_gemv_int8(int M, int K, bool tra, bool flag_bias, bool flag_relu, int in_threads) {
    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;
    Context ctx1;
    PowerMode mode = (PowerMode)cluster;
    ctx1.set_run_mode(mode, in_threads);
    ctx1.set_arch(flag_arch);
    LOG(INFO) << "CPU ARCH: A" << flag_arch;
    LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << in_threads;
#endif
    }
    Shape sha(1, 1, M, K);
    Shape shb(1, 1, 1, K);
    Shape shc(1, 1, 1, M);
    TensorHf4 ta;
    TensorHf4 tb;
    TensorHf4 tbias;
    ta.re_alloc(sha, AK_INT8);
    tb.re_alloc(shb, AK_INT8);
    tbias.re_alloc(Shape(M), AK_INT32);
    fill_tensor_rand(ta, -127, 127);
    fill_tensor_rand(tb, -127, 127);
    fill_tensor_rand(tbias, -65536, 65536);
//    fill_tensor_const(ta, 1.f);
//    fill_tensor_const(tb, 1.f);
//    fill_tensor_const(tbias, 1.f);
//    print_tensor(ta);
//    print_tensor(tb);
//    print_tensor(tbias);
    TensorHf4 tout_basic;
    TensorHf4 tout_saber;
    tout_saber.re_alloc(shc, AK_INT32);
    tout_basic.re_alloc(shc ,AK_INT32);
    int m = M;
    int k = K;
    LOG(INFO) << "gemv M: " << m << ", K: " << k << ", transA: " << (tra? "true" : "false") \
        << ", relu: " << (flag_relu? "true" : "false") << ", bias: " << (flag_bias? "true" : "false");
//    LOG(INFO) << "compare result with basic sgemm: " << (COMPARE_RESULT? "true" : "false");
    const char* da = static_cast<const char*>(ta.data());
    const char* db = static_cast<const char*>(tb.data());
    const int* dbias = static_cast<const int*>(tbias.data());
    int* dc_basic = static_cast<int*>(tout_basic.mutable_data());
    memset(dc_basic, 0, sizeof(int) * tout_basic.valid_size());
    if (COMPARE_RESULT) {
        LOG(INFO) << "run basic conv for precision comparation";
        basic_gemv(m, k, da, db, dbias, dc_basic, 1, 0, tra, flag_bias, flag_relu);
        //print_tensor(tout_basic);
    }

    //! compute
    LOG(INFO) << "saber sgemm compute";

    long long ops = m * k;
    int* dc_saber = static_cast<int*>(tout_saber.mutable_data());

    to = 0;
    min_time = 1000000;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start();
        gemv_int8(da, db, dc_saber, false, m, k, flag_bias, dbias, flag_relu);
        t1.end();
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "saber sgemv running time, ave: " << to / test_iter << ", min time: " << min_time;
    LOG(WARNING) << "mean gops: " << 0.000001f * ops * test_iter / to << " GFLOPS, max gops: " << 0.000001f * ops / min_time << " GFLOPS";
    //print_tensor(tout_saber);
    if (COMPARE_RESULT) {
        double max_ratio = 0;
        double max_diff = 0;
        tensor_cmp_host(tout_basic, tout_saber, max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabs(max_ratio) > 1e-4f && fabsf(max_diff) > 5e-5f) {
            TensorHf4 tdiff;
            tdiff.re_alloc(tout_basic.valid_shape(), tout_saber.get_dtype());
            tensor_diff(tout_basic, tout_saber, tdiff);
            LOG(INFO) << "basic result: ";
            print_tensor(tout_basic);
            LOG(INFO) << "saber result: ";
            print_tensor(tout_saber);
            LOG(INFO) << "diff result: ";
            print_tensor(tdiff);
            return SaberInvalidValue;
        }
    }
    return SaberSuccess;
}
TEST(TestSaberLite, test_func_sgemm_prepacked) {
    if (Basic_test) {
        LOG(INFO) << "run basic sgemm test";
        for (auto& m : {1, 3, 7, 16, 111, 256, 397, 512, 777, 1024}) {
        for (auto& k : {1, 5, 8, 15, 59, 128, 234, 512, 678, 1024}) {
        for (auto& flag_bias : {false, true}) {
        for (auto& flag_relu : {false, true}) {
        for (auto& th : {1, 2, 4}) {
            SaberStatus flag = test_arm_gemv_int8(m, k, false, flag_bias, flag_relu, th);
            if (flag == SaberSuccess) {
                LOG(INFO) << "test m = " << m << ", k=" << k << \
                    ", bias: " << (flag_bias? "true" : "false") << ", relu: " << \
                    (flag_relu? "true" : "false") << ", trans A: false" << " passed\n";
            } else {
                LOG(FATAL) << "test m = " << m << ", k=" << k << \
                    ", bias: " << (flag_bias? "true" : "false") << ", relu: " << \
                    (flag_relu? "true" : "false") << ", trans A: false" << " failed\n";
                                        }
        }
        }
        }
        }
        }
    }
}
TEST(TestSaberLite, test_func_sgemm_prepacked_custom) {
    auto flag = test_arm_gemv_int8(M, K, false, flag_bias, flag_relu, threads);
    if (flag != SaberSuccess) {
        LOG(FATAL) << "test m = " << M << ", k=" << K << \
            ", trans A: false" << ", bias: " << flag_bias << \
            ", relu: " << flag_relu  << " failed!!\n";
    }
    LOG(INFO) << "test m = " << M << ", k=" << K << \
            ", trans A: false" << ", bias: " << flag_bias << \
            ", relu: " << flag_relu  << " passed!!\n";
}
int main(int argc, const char** argv){
    anakin::saber::lite::Env::env_init();
    LOG(ERROR) << "usage: ./" << argv[0] << " [do_basic_test] [cluster]  [threads] [m] [k] [bias] [relu] [test iter] [compare result] [arch]";
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
        if (argc < 8) {
            LOG(ERROR) << "usage: ./" << argv[0] << " [do_basic_test] [cluster]  [threads] [m] [k] [bias] [relu] [test iter] [compare result] [arch]";
            return 0;
        }
        M = atoi(argv[4]);
        K = atoi(argv[5]);
        flag_bias = atoi(argv[6]) > 0;
        flag_relu = atoi(argv[7]) > 0;
    }
    if (argc > 8) {
        test_iter = atoi(argv[8]);
    }
    if (argc > 9) {
        COMPARE_RESULT = atoi(argv[9]) > 0;
    }
    if (argc > 10) {
        if (atoi(argv[10]) > 0) {
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
