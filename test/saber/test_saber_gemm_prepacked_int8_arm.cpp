#include "saber/core/tensor_op.h"
#include "saber/funcs/timer.h"
#include "test/saber/test_saber_func.h"
#include "saber/funcs/type_trans.h"

#include "saber/funcs/impl/arm/neon/impl/gemm_prepacked_int8.h"

using namespace anakin::saber;

#ifdef USE_ARM_PLACE

int cluster = 0;
int threads = 1;

bool Basic_test = false;

int M = 16;
int N = 16;
int K = 16;
bool traA = false;
bool traB = false;
bool flag_relu = false;
bool flag_bias = false;
ARMArch flag_arch = A73;
int test_iter = 1;
bool COMPARE_RESULT = true;

typedef Tensor<ARM> TensorH;


template  <typename type,  typename type2>
static void basic_gemm(int m, int n, int k, const type* a, const type* b, const type2* bias, type2* c, \
    type2 alpha, type2 beta, \
    bool trans_a = false, bool trans_b = false, bool flag_bias = false, bool flag_relu = false) {
#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        type2 bias_data = (type2)0;
        if (flag_bias) {
            bias_data = bias[i];
        }
        for (int j = 0; j < n; ++j) {
            type2 sum = static_cast<type2>(0);
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
            type2 tmp = alpha * sum + beta * c[i * n + j] + bias_data;
            if (flag_relu) {
                c[i * n + j] = tmp > (type2)0? tmp : (type2)0;
            } else {
                c[i * n + j] = tmp;
            }
        }
    }
}

SaberStatus test_arm_gemm(int M, int N, int K, bool tra, bool trb, bool is_bias, bool is_relu, int in_th) {
    double to = 0;
    double min_time = 1000000;
    SaberTimer<ARM> t1;
    Context<ARM> ctx1;
    PowerMode mode = (PowerMode)cluster;
    ctx1.set_run_mode(mode, in_th);
    //ctx1.set_arch(flag_arch);
    //LOG(INFO) << "CPU ARCH: A" << flag_arch;
    Shape sha({1, 1, M, K});
    Shape shb({1, 1, K, N});
    if (tra) {
        sha = Shape({1, 1, K, M});
    }
    if (trb) {
        shb = Shape({1, 1, N, K});
    }
    Shape shc({1, 1, M, N});

    TensorH ta(sha, AK_INT8);
    TensorH tb(shb, AK_INT8);

    TensorH tbias(Shape({1, 1, 1, M}), AK_INT32);

    std::vector<float> scale_a(M);
    std::vector<float> scale_b = {1.f / 63};
    std::vector<float> scale_c = {K / 63.f};
    std::vector<float> scale_merge_fp32(M);
    std::vector<float> scale_merge_int8(M);

    for (int j = 0; j < M; ++j) {
        scale_a[j] = 1.f / 63;
        scale_merge_fp32[j] = scale_a[j] * scale_b[0];
        scale_merge_int8[j] = scale_merge_fp32[j] / scale_c[0];
    }
    fill_tensor_rand(ta, -63, 63);
    //fill_tensor_const(ta, 1);
    fill_tensor_rand(tb, -63, 63);
    //fill_tensor_const(tb, 1);
    fill_tensor_rand(tbias, -65536, 65535);
    //fill_tensor_const(tbias, 1);
//    LOG(INFO) << "tensor A: ";
//    print_tensor(ta);
//    LOG(INFO) << "tensor B: ";
//    print_tensor(tb);
//    print_tensor(tbias);
    TensorH tout_basic_int32;
    TensorH tout_basic_fp32;
    TensorH tout_basic_int8;

    TensorH tout_saber_int32;
    TensorH tout_saber_fp32;
    TensorH tout_saber_int8;
    tout_saber_int32.re_alloc(shc, AK_INT32);
    tout_saber_fp32.re_alloc(shc, AK_FLOAT);
    tout_saber_int8.re_alloc(shc, AK_INT8);
    int m = M;
    int n = N;
    int k = K;

    LOG(INFO) << "gemm M: " << m << ", N: " << n << ", K: " << k
        << ", transA: " << (tra? "true" : "false") << ", transB: " << (trb? "true" : "false");
    LOG(INFO) << "relu: " << (is_relu? "true" : "false") << ", bias: " << (is_bias? "true" : "false")
        << ", test iter: " << test_iter << ", threads: " << in_th;
    auto da = static_cast<const int8_t*>(ta.data());
    auto db = static_cast<const int8_t*>(tb.data());
    if (COMPARE_RESULT) {
        tout_basic_int32.re_alloc(shc, AK_INT32);
        tout_basic_fp32.re_alloc(shc, AK_FLOAT);
        tout_basic_int8.re_alloc(shc, AK_INT8);
        int* dc_basic = static_cast<int*>(tout_basic_int32.mutable_data());
        basic_gemm(m, n, k, da, db, static_cast<const int*>(tbias.data()), \
            dc_basic, 1, 0, tra, trb, is_bias, is_relu);
        //! convert to fp32 and int8
        trans_tensor_dtype<ARM, AK_INT32, AK_FLOAT>(tout_basic_int32, tout_basic_fp32, scale_b[0], 1.f, scale_a);
        trans_tensor_dtype<ARM, AK_INT32, AK_INT8>(tout_basic_int32, tout_basic_int8, scale_b[0], scale_c[0], scale_a);
//        print_tensor(tout_basic_int32);
//        print_tensor(tout_basic_fp32);
//        print_tensor(tout_basic_int8);
    }
    double ops = 2.0 * m * n * k;
    int* dc_saber_int32 = static_cast<int*>(tout_saber_int32.mutable_data());
    float* dc_saber_fp32 = static_cast<float*>(tout_saber_fp32.mutable_data());
    int8_t* dc_saber_int8 = static_cast<int8_t*>(tout_saber_int8.mutable_data());
    to = 0;
    min_time = 1000000;
    int hblock = get_hblock_int8(ctx1.get_arch());
    int round_up_a = ((hblock + m - 1) / hblock) * hblock;
    TensorH tpackedA(Shape({1, 1, round_up_a, ROUNDUP(k, KBLOCK_INT8)}), AK_INT8);
    int lda = k;
    if (tra) {
        lda = m;
    }

    prepackA_int8(tpackedA, ta, m, k, 1, tra, &ctx1);
    //! compute int32 out
#if 1
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        gemm_prepack_int8(static_cast<const int8_t*>(tpackedA.data()), db, \
            static_cast<const int*>(tbias.data()), dc_saber_int32, m, n, k, \
            is_bias, is_relu, trb, nullptr, &ctx1);
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "int8->int32 packed gemm running time, ave: " << to / test_iter << ", min time: " << min_time;
    LOG(INFO) << "mean gops: " << 0.000001f * ops * test_iter / to \
        << " GFLOPS, max gops: " << 0.000001f * ops / min_time << " GFLOPS";
#endif
    //! compute fp32 out
#if 1
    min_time = 100000;
    to = 0;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        gemm_prepack_int8(static_cast<const int8_t*>(tpackedA.data()), db, \
            static_cast<const int*>(tbias.data()), dc_saber_fp32, m, n, k, \
            is_bias, is_relu, trb, scale_merge_fp32.data(), &ctx1);
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "int8->fp32 packed gemm running time, ave: " << to / test_iter << ", min time: " << min_time;
    LOG(INFO) << "mean gops: " << 0.000001f * ops * test_iter / to \
        << " GFLOPS, max gops: " << 0.000001f * ops / min_time << " GFLOPS";
#endif
    //! compute int8 out
#if 1
    min_time = 100000;
    to = 0;
    for (int i = 0; i < test_iter; ++i) {
        t1.clear();
        t1.start(ctx1);
        gemm_prepack_int8(static_cast<const int8_t*>(tpackedA.data()), db, \
            static_cast<const int*>(tbias.data()), dc_saber_int8, m, n, k, \
            is_bias, is_relu, trb, scale_merge_int8.data(), &ctx1);
        t1.end(ctx1);
        to += t1.get_average_ms();
        if (t1.get_average_ms() < min_time) {
            min_time = t1.get_average_ms();
        }
    }
    LOG(INFO) << "int8->int8 packed gemm running time, ave: " << to / test_iter << ", min time: " << min_time;
    LOG(INFO) << "mean gops: " << 0.000001f * ops * test_iter / to \
        << " GFLOPS, max gops: " << 0.000001f * ops / min_time << " GFLOPS";
#endif
    if (COMPARE_RESULT) {
        double max_ratio = 0;
        double max_diff = 0;
        //! int32
#if 1
        tensor_cmp_host((const int32_t*)tout_basic_int32.data(), (const int32_t*)tout_saber_int32.data(), \
            tout_basic_int32.valid_size(), max_ratio, max_diff);
        if (fabs(max_ratio) > 1e-4f) {
            LOG(WARNING) << "int32 basic result";
            print_tensor(tout_basic_int32);
            LOG(WARNING) << "int32 saber result";
            print_tensor(tout_saber_int32);
        }
        LOG(INFO) << "int32 compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabs(max_ratio) > 1e-4f) {
            return SaberInvalidValue;
        }
#endif
        //! fp32
#if 1
        max_ratio = 0;
        max_diff = 0;
        tensor_cmp_host((const float*)tout_basic_fp32.data(), (const float*)tout_saber_fp32.data(), \
            tout_basic_fp32.valid_size(), max_ratio, max_diff);
        if (fabs(max_ratio) > 1e-4f) {
            LOG(WARNING) << "fp32 basic result";
            print_tensor(tout_basic_fp32);
            LOG(WARNING) << "fp32 saber result";
            print_tensor(tout_saber_fp32);
        }
        LOG(INFO) << "fp32 compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabs(max_ratio) > 1e-4f) {
            return SaberInvalidValue;
        }
#endif
        //! int8
#if 1
        max_ratio = 0;
        max_diff = 0;
        tensor_cmp_host((const int8_t*)tout_basic_int8.data(), (const int8_t*)tout_saber_int8.data(), \
            tout_basic_int8.valid_size(), max_ratio, max_diff);
        if (fabs(max_ratio) > 1e-4f) {
            LOG(WARNING) << "int8 basic result";
            print_tensor(tout_basic_int8);
            LOG(WARNING) << "int8 saber result";
            print_tensor(tout_saber_int8);
        }
        LOG(INFO) << "int8 compare result, max diff: " << max_diff << ", max ratio: " << max_ratio;
        if (fabs(max_ratio) > 1e-4f) {
            return SaberInvalidValue;
        }
#endif
    }
    return SaberSuccess;
}

TEST(TestSaberFunc, test_func_gemm_prepacked) {
    if (Basic_test) {
        LOG(INFO) << "run basic gemm test";
        for (auto& m : {1, 8, 16, 111, 256}) {
        for (auto& n : {1, 3, 13, 141, 345, 512, 1024}) {
        for (auto& k : {1, 4, 15, 59, 128, 512}) {
        for (auto& tra : {false, true}) {
        for (auto& trb : {false, true}) {
        for (auto& is_bias : {false, true}) {
        for (auto& is_relu : {false, true}) {
        for (auto& th : {1, 2, 4}) {
            SaberStatus flag = test_arm_gemm(m, n, k, tra, trb, is_bias, is_relu, th);
            if (flag == SaberSuccess) {
                LOG(INFO) << "test m = " << m << ", n=" << n << ", k=" << k << \
                    ", bias: " << (is_bias? "true" : "false") << ", relu: " << \
                    (is_relu? "true" : "false") << ", trans A: " << (tra? "true" : "false") << \
                    ", trans B: " << (trb? "true" : "false") << " passed\n";
                                        } else {
                LOG(FATAL) << "test m = " << m << ", n=" << n << ", k=" << k << \
                    ", bias: " << (is_bias? "true" : "false") << ", relu: " << \
                    (is_relu? "true" : "false") << ", trans A: " << (tra? "true" : "false") << \
                    ", trans B: " << (trb? "true" : "false") << " failed\n";
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

TEST(TestSaberFunc, test_func_gemm_prepacked_custom) {
    if (test_arm_gemm(M, N, K, traA, traB, flag_bias, flag_relu, threads) == SaberSuccess) {
        LOG (INFO) << "test m = " << M << ", n=" << N << ", k=" << K << \
            ", bias: " << (flag_bias ? "true" : "false") << ", relu: " << \
            (flag_relu ? "true" : "false") << ", trans A: " << (traA ? "true" : "false") << \
            ", trans B: " << (traB ? "true" : "false") << " passed\n";
    } else {
        LOG (FATAL) << "test m = " << M << ", n=" << N << ", k=" << K << \
            ", bias: " << (flag_bias ? "true" : "false") << ", relu: " << \
            (flag_relu ? "true" : "false") << ", trans A: " << (traA ? "true" : "false") << \
            ", trans B: " << (traB ? "true" : "false") << " failed\n";
    }
}

int main(int argc, const char** argv) {
    Env<ARM>::env_init();
    LOG(INFO) << "usage: ./" << argv[0] << " [do_basic_test] [cluster]  [threads]  [m] [n]  [k] [transA] [transB] [bias] [relu] [test iter] [compare result]";
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
            LOG(ERROR) << "usage: ./" << argv[0] << " [do_basic_test] [cluster]  [threads]  [m] [n]  [k] [transA] [transB] [bias] [relu] [test iter] [compare result]";
            return 0;
        }
        M = atoi(argv[4]);
        N = atoi(argv[5]);
        K = atoi(argv[6]);
        traA = atoi(argv[7]) > 0;
        traB = atoi(argv[8]) > 0;
        flag_bias = atoi(argv[9]) > 0;
        flag_relu = atoi(argv[10]) > 0;
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
    logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
#else

int main(int argc, const char** argv){
    LOG(INFO) << "this unit test only be used in TargetType is ARM";
    return 0;
}

#endif
