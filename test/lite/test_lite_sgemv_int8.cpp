#include "test_lite.h"
#include "saber/lite/funcs/neon/impl/sgemv_arm_int8.h"
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
int test_iter = 2;
bool COMPARE_RESULT = false;
typedef Tensor<CPU> TensorHf4;
void basic_sgemv(int m, int n, const signed char* a, const signed char* b, const int* bias, int* c, \
    bool trans_b = false, bool flag_bias = false, bool flag_relu = false) {
//#pragma omp parallel for
    for (int i = 0; i < m; i++){
        int sum = 0;
        if (flag_bias)sum = bias[i];
        const signed char* ptr_din = b;
        const signed char* ptr_wei = a + i * n;
        for (int j = 0; j < n; j++){
            sum += (int)(ptr_din[j] * ptr_wei[j]);
        }
        if (flag_relu) sum = sum > 0 ? sum : 0;
        *c++ = sum;
    }
}
SaberStatus test_arm_sgemv(int M, int N, bool flag_bias, bool flag_relu, int in_th) {
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
    Shape sha(M, N);
    Shape shin(N);
    Shape shout(M);
    TensorHf4 ta;
    TensorHf4 tb;
    TensorHf4 tbias;
    ta.re_alloc(sha, AK_INT8); //weights
    tb.re_alloc(shin, AK_INT8); //x
    tbias.re_alloc(shout, AK_INT32);//y
    fill_tensor_rand(ta, -64, 63);
    // fill_tensor_const(ta, 1);
    fill_tensor_rand(tb, -64, 63);
    // fill_tensor_const(tb, 1);
    fill_tensor_rand(tbias, -65536, 65535);
    // print_tensor(ta);
    // print_tensor(tb);
    //print_tensor(tbias);
    TensorHf4 tout_basic;
    TensorHf4 tout_saber;
    tout_saber.re_alloc(shout, AK_INT32);
    int m = M;
    int n = N;
    LOG(INFO) << "sgemv M: " << m << ", N: " << n;
    LOG(INFO) << "relu: " << (flag_relu? "true" : "false") << ", bias: " << (flag_bias? "true" : "false");
    LOG(INFO) << "test iter: " << test_iter;
    LOG(INFO) << "compare result with basic sgemv: " << (COMPARE_RESULT? "true" : "false");
    const signed char* da = static_cast<const signed char*>(ta.data());
    const signed char* db = static_cast<const signed char*>(tb.data());
    if (COMPARE_RESULT) {
        LOG(INFO) << "run basic conv for precision comparation";
        tout_basic.re_alloc(shout, AK_INT32);
        int* dc_basic = static_cast<int*>(tout_basic.mutable_data());
        basic_sgemv(m, n, da, db, static_cast<const int*>(tbias.data()), dc_basic, \
            false, flag_bias, flag_relu);
       // LOG(WARNING) << "basic result";
       // print_tensor(tout_basic);
    }
    long long ops = m * n;
    //! compute
    int* dc_saber = static_cast<int*>(tout_saber.mutable_data());
    LOG(INFO) << "saber sgemm compute";
    for (int i = 0; i < test_iter; ++i) {
        // t1.clear();
        // t1.start();
        if (flag_bias){
            if (flag_relu){
                t1.clear();
                t1.start();
                sgemv_bias_relu_int8(false, m, n, da, db, dc_saber, static_cast<const int*>(tbias.data()));
                t1.end();
            }else{
                t1.clear();
                t1.start();
                sgemv_bias_int8(false, m, n, da, db, dc_saber, static_cast<const int*>(tbias.data()));
                t1.end();
            }
        }else{
            if (flag_relu){
                t1.clear();
                t1.start();
                sgemv_relu_int8(false, m, n, da, db, dc_saber);
                t1.end();
            }else{
                t1.clear();
                t1.start();
                sgemv_int8(false, m, n, da, db, dc_saber);
                t1.end();
            }
        }
        // sgemv_bias_relu_int8(false, m, n, da, db, dc_saber, static_cast<const char*>(tbias.data()));
        // sgemv_relu_int8(false, m, n, da, db, dc_saber);
        // t1.end();
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    // LOG(WARNING) << "saber result";
    // print_tensor(tout_saber);

    LOG(INFO) << "saber sgemv running time, ave: " << to / test_iter << ", min time: " << min_time;
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
                for (auto& flag_bias : {false, true}) {
                    for (auto& flag_relu : {false, true}) {
                        for (auto& th : {1, 2, 4}) {
                            SaberStatus flag = test_arm_sgemv(m, n, flag_bias, flag_relu, th);
                            if (flag == SaberSuccess) {
                                LOG(INFO) << "test m = " << m << ", n=" << n << \
                                    ", bias: " << (flag_bias? "true" : "false") << ", relu: " << \
                                    (flag_relu? "true" : "false") << " passed";
                            } else {
                                LOG(FATAL) << "test m = " << m << ", n=" << n << \
                                    ", bias: " << (flag_bias? "true" : "false") << ", relu: " << \
                                    (flag_relu? "true" : "false") << " failed";
                            }
                        }
                    }
                }
            }
        }
    }
}
TEST(TestSaberLite, test_func_sgemm_prepacked_custom) {
    if (test_arm_sgemv(M, N, flag_bias, flag_relu, threads) == SaberSuccess) {
        LOG (INFO) << "test m = " << M << ", n=" << N << \
            ", bias: " << (flag_bias ? "true" : "false") << ", relu: " << \
            (flag_relu ? "true" : "false") << " passed";
    } else {
        LOG (FATAL) << "test m = " << M << ", n=" << N << \
            ", bias: " << (flag_bias ? "true" : "false") << ", relu: " << \
            (flag_relu ? "true" : "false") << " failed";
    }
}
int main(int argc, const char** argv){
    anakin::saber::lite::Env::env_init();
    LOG(ERROR) << "usage: ./" << argv[0] << " [do_basic_test] [cluster]  [threads]  [m] [n] [relu] [bias] [test iter] [compare result]";
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
        if (argc < 7) {
            LOG(ERROR) << "usage: ./" << argv[0] << " [do_basic_test] [cluster]  [threads]  [m] [n] [relu] [bias] [test iter] [compare result]";
            return 0;
        }
        M = atoi(argv[4]);
        N = atoi(argv[5]);
        flag_relu = atoi(argv[6]) > 0;
        flag_bias = atoi(argv[7]) > 0;
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
