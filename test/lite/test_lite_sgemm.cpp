#include "test_lite.h"
#include "saber/lite/funcs/neon/impl/sgemm_arm.h"

using namespace anakin::saber;
using namespace anakin::saber::lite;

int cluster = 0;
int threads = 4;

int batch = 1;
int M = 100;
int N = 100;
int K = 100;
bool traA = false;
bool traB = false;
bool flag_relu = false;

int test_iter = 10;

bool COMPARE_RESULT = false;

typedef Tensor<CPU, AK_FLOAT> TensorHf4;

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
void basic_gemm(int m, int n, int k, const type* a, const type* b, type* c, type alpha, type beta, \
    bool trans_a = false, bool trans_b = false, bool flag_relu = false) {
//#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            type sum = static_cast<type>(0);
            for (int l = 0; l < k; ++l) {
                type av;
                type bv;
                if (trans_a) {
                    av = a[l * m + i];
                } else{
                    av = a[i * k + l];
                }
                if (trans_b) {
                    bv = b[j * k + l];
                } else {
                    bv = b[l * n + j];
                }
                sum += av * bv;
            }
            type tmp = alpha * sum + beta * c[i * n + j];
            if (flag_relu) {
                c[i * n + j] = tmp > (type)0? tmp : (type)0;
            } else {
                c[i * n + j] = tmp;
            }

        }
    }
}

void test_arm_sgemm(TensorHf4& ta, TensorHf4& tb, TensorHf4& tc, \
    bool flag_bias, bool flag_relu, int thread_num, int cluster_id) {

    double to = 0;
    double min_time = 1000000;
    SaberTimer t1;

    Context ctx1;
    PowerMode mode = cluster_id == 0? SABER_POWER_HIGH : SABER_POWER_LOW;
    ctx1.set_run_mode(mode, threads);
        LOG(INFO) << "test threads activated";
#pragma omp parallel
    {
#ifdef USE_OPENMP
        int thread = omp_get_num_threads();
        LOG(INFO) << "number of threads: " << thread;
#endif
    }

    LOG(INFO) << "sgemm M: " << M << ", N: " << N << ", K: " << K;
    LOG(INFO) << "transA: " << (traA? "true" : "false") << ", transB: " << (traB? "true" : "false");
    LOG(INFO) << "test iter: " << test_iter;
    LOG(INFO) << "compare result with basic sgemm: " << (COMPARE_RESULT? "true" : "false");

    TensorHf4 tout_basic;
    TensorHf4 tout_saber;
    Shape shape_out = tc.valid_shape();
    int m = ta.height();
    int n = tb.width();
    int k = ta.width();

    const float* da = ta.data();
    const float* db = tb.data();

    if(COMPARE_RESULT) {
        LOG(INFO) << "run basic conv for precision comparation";
        tout_basic.re_alloc(shape_out);
        float* dc_basic = tout_basic.mutable_data();
        basic_gemm(m, n, k, da, db, dc_basic, 1.f, 0.f, traA, traB);
        //print_tensor(tout_basic);
    }

    //! sgemm init
    int l1_cache = Env::cur_env()._L1_cache;
    int l2_cache = Env::cur_env()._L2_cache;
    //! if L1 cache size is not provided, set to 31K
    l1_cache = l1_cache > 0? l1_cache : 31000;
    //! if L2 cache size is not provided, set to 2M
    l2_cache = l2_cache > 0? l2_cache : 2000000;
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

    float* dc_saber = tc.mutable_data();
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
    LOG(INFO) << "saber conv running time, ave: " << to / test_iter << ", min time: " << min_time;
    //print_tensor(tc);

    if (COMPARE_RESULT) {
        double max_ratio = 0;
        double max_diff = 0;
//        TensorHf4 tdiff(tout_basic.valid_shape());
//        tensor_diff(tout_basic, tc, tdiff);
//        print_tensor(tdiff);
        tensor_cmp_host(tout_basic.data(), tc.data(), tout_basic.valid_size(), max_ratio, max_diff);
        LOG(INFO) << "compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        CHECK_EQ(fabsf(max_ratio) < 1e-5f, true) << "compute result error";
    }
}

TEST(TestSaberLite, test_func_sgemm_arm) {

    int num = batch;
    int ch = 1;
    int h_a = M;
    int w_a = K;

    int h_b = K;
    int w_b = N;

    int h_c = M;
    int w_c = N;

    bool flag_relu = true;
    bool flag_bias = true;

    Shape sha(num, ch, h_a, w_a);
    Shape shb(num, ch, h_b, w_b);
    Shape shc(num, ch, h_c, w_c);

    TensorHf4 ta, tb, tc;

    ta.re_alloc(sha);
    tb.re_alloc(shb);
    tc.re_alloc(shc);
#if 0
    float* ptr_a = ta.mutable_data();
    for (int i = 0; i < ta.valid_size(); ++i) {
        ptr_a[i] = i;
    }
    float* ptr_b = tb.mutable_data();
    for (int i = 0; i < tb.valid_size(); ++i) {
        ptr_b[i] = i;
    }
#else
//    fill_tensor_const(ta, 1.f);
//    fill_tensor_const(tb, 1.f);
    fill_tensor_rand(ta, -1.f, 1.f);
    fill_tensor_rand(tb, -1.f, 1.f);
#endif
    test_arm_sgemm(ta, tb, tc, flag_bias, flag_relu, threads, cluster);
    //LOG(WARNING) << "conv3x3s1 not support yet";
}

int main(int argc, const char** argv){
    anakin::saber::lite::Env::env_init();

    LOG(ERROR) << "usage: ./" << argv[0] << " [cluster]  [threads]  [m] [n]  [k] [transA] [transB] [relu] [test iter] [compare result]";

    if (argc >= 2) {
        cluster = atoi(argv[1]);
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }
    if(argc >= 4) {
        if (argc < 9) {
            LOG(ERROR) << "usage: ./" << argv[0] << " [cluster]  [threads]  [m] [n]  [k] [transA] [transB] [relu] [test iter] [compare result]";
            return 0;
        }
        M = atoi(argv[3]);
        N = atoi(argv[4]);
        K = atoi(argv[5]);
        traA = atoi(argv[6]) > 0;
        traB = atoi(argv[7]) > 0;
        flag_relu = atoi(argv[8]) > 0;
    }
    if (argc > 9) {
        test_iter = atoi(argv[9]);
    }
    if (argc > 10) {
        COMPARE_RESULT = atoi(argv[10]) > 0;
    }
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}

