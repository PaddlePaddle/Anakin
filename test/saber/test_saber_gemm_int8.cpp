#include "saber/core/context.h"
#include "saber/funcs/gemm.h"
#include "saber/funcs/timer.h"
#include "saber/core/tensor.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/calibrate.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "conv_func_helper.h"
#include <vector>

using namespace anakin::saber;

void gemm_check(const int m, const int n, const int k,
                const float* a, const float* b, float* c,
                const float alpha, const float beta,
                const bool trans_a, const bool trans_b) {
    if (!trans_a && !trans_b) {
        int lda = k;
        int ldb = n;
        int ldc = n;
        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;
                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += alpha * a[m_i * lda + k_i] * b[k_i * ldb + n_i];
                }
            }
        }
    } else if (!trans_a && trans_b) {
        int lda = k;
        int ldb = k;
        int ldc = n;
        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;
                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += alpha * a[m_i * lda + k_i] * b[n_i * ldb + k_i];
                }
            }
        }
    } else if (trans_a && !trans_b) {
        int lda = m;
        int ldb = n;
        int ldc = n;
        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;
                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += alpha * a[k_i * lda + m_i] * b[k_i * ldb + n_i];
                }
            }
        }
    } else {
        int lda = m;
        int ldb = k;
        int ldc = n;
        for (int m_i = 0; m_i < m; ++m_i) {
            for (int n_i = 0; n_i < n; ++n_i) {
                c[m_i * ldc + n_i] *= beta;
                for (int k_i = 0; k_i < k; ++k_i) {
                    c[m_i * ldc + n_i] += alpha * a[k_i * lda + m_i] * b[n_i * ldb + k_i];
                }
            }
        }
    }
}

template <typename dtype>
int count_diff(const dtype* src1, const dtype* src2, int size, double max_ratio) {
    if (max_ratio <= 0) {
        max_ratio = 0.1;
    }
    int count = 0;
    for (int i = 0; i < size; ++i) {
        double ratio = fabs(src1[i] - src2[i]) / fabs(src1[i] + src2[i] + 1e-12);
        if (ratio > max_ratio) {
            ++count;
        }
    }
    return count;
}

template <typename TargetType, typename TargetType_H>
void test_gemm_int8_result (int m, int n, int k, bool trans_a, bool trans_b) {

    Tensor<TargetType> a_dev, b_dev, c_dev;
    Tensor<TargetType_H> a_host, b_host, c_host, c_check;

    Context<TargetType> ctx1(0, 1, 0);
    int generate_arch = Env<NV>::cur_env()[ctx1.get_device_id()]._info._generate_arch;
    // only support 61 arch for now.
    bool arch_check = (generate_arch == 61);
    if (!arch_check) {
                LOG(INFO) << "device not support int8 op!!";
        return;
    }
    Gemm<TargetType, VENDER_IMPL, char, float> gemm_vender;
    Gemm<TargetType, SABER_IMPL, char, float> gemm_saber;
    SaberStatus vender_status = gemm_vender.init(trans_a, trans_b, m, n, k, ctx1);
    SaberStatus saber_status = gemm_saber.init(trans_a, trans_b, m, n, k, ctx1);

    float alpha = 1.f;
    float beta = 0.f;

    Shape a_shape({m, k}, Layout_HW);
    Shape b_shape({k, n}, Layout_HW);
    Shape c_shape({m, n}, Layout_HW);

    Tensor<TargetType> a_dev_int8, b_dev_int8;
    Tensor<TargetType> a_scale, b_scale;
    Shape a_scale_shape({m}, Layout_W);
    Shape b_scale_shape({n}, Layout_W);
    a_dev_int8.re_alloc(a_shape, AK_INT8);
    b_dev_int8.re_alloc(b_shape, AK_INT8);
    a_scale.re_alloc(a_scale_shape, AK_FLOAT);
    b_scale.re_alloc(b_scale_shape, AK_FLOAT);

    a_dev.re_alloc(a_shape, AK_FLOAT);
    b_dev.re_alloc(b_shape, AK_FLOAT);
    c_dev.re_alloc(c_shape, AK_FLOAT);

    a_host.re_alloc(a_shape, AK_FLOAT);
    b_host.re_alloc(b_shape, AK_FLOAT);
    c_host.re_alloc(c_shape, AK_FLOAT);
    c_check.re_alloc(c_shape, AK_FLOAT);

    fill_tensor_rand(a_dev, -10.f, 10.f);
    fill_tensor_rand(b_dev, -10.f, 10.f);

    a_host.copy_from(a_dev);
    b_host.copy_from(b_dev);
    SaberTimer<TargetType> vender_time, saber_time;

    int ts = 100;
    if (vender_status == SaberSuccess) {

        /// step 1: calibrate matrix a into int8 matrix
        ///         using row direction.
        /// input : const float* src
        /// output: char* dst_int8
        ///         float* scale
        float2char(trans_a, (signed char*)a_dev_int8.mutable_data(),
                       (const float*)a_dev.data(),
                       (float*)a_scale.mutable_data(),
                       trans_a?k:m, trans_a?m:k, ctx1);

        /// step 2: calibrate matrix a into int8 matrix
        ///         using col direction.
        /// input : const float* src
        /// output: char* dst_int8
        ///         float* scale
        float2char(!trans_b, (signed char*)b_dev_int8.mutable_data(),
                       (const float*)b_dev.data(),
                       (float*)b_scale.mutable_data(),
                       trans_b?n:k, trans_b?k:n, ctx1);

        /// step 3: dispatch matrix multiply using int8 gemm
        gemm_vender.dispatch(alpha, beta,
                             (const char *) a_dev_int8.data(),
                             (const char *) b_dev_int8.data(),
                             (float *) c_dev.mutable_data());

        /// step 4: convert int32 into float32 using fix2float.
        /// input : float* dst
        ///         const scaleA(row scale)
        ///         const scaleB(col scale)
        /// output: float* dst
        fix2float((float*)c_dev.mutable_data(),
                  (const float*)a_scale.data(),
                  (const float*)b_scale.data(),
                  alpha, beta, m, n, ctx1);

        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();

        c_host.copy_from(c_dev);
        gemm_check(m, n, k, (const float *) a_host.data(), (const float *) b_host.data(),
                   (float *) c_check.mutable_data(),
                   alpha, beta, trans_a, trans_b);

        int counts = count_diff((const float*)c_check.data(), (const float*)c_host.data(),
                                c_check.valid_size(), 1e-1);

        if (((double)counts / (double)c_host.valid_size()) > 0.05) {
            print_tensor_valid(c_check);
            print_tensor_valid(c_host);
            LOG(FATAL) << "VENDER: FAIL!!!! counts = " <<counts
                       << "m = "<< m << " n = "<< n << " k = "<< k;
        }
        for (int t = 0; t < ts; ++t) {
            vender_time.start(ctx1);
            gemm_vender.dispatch(alpha, beta,
                                 (const char *) a_dev_int8.data(),
                                 (const char *) b_dev_int8.data(),
                                 (float *) c_dev.mutable_data());
            typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
            c_dev.record_event(stream);
            c_dev.sync();
            vender_time.end(ctx1);
        }
    }
    if (saber_status == SaberSuccess) {

        /// step 1: calibrate matrix a into int8 matrix
        ///         using row direction.
        /// input : const float* src
        /// output: char* dst_int8
        ///         float* scale
        float2char(trans_a, (signed char*)a_dev_int8.mutable_data(),
                   (const float*)a_dev.data(),
                   (float*)a_scale.mutable_data(),
                   trans_a?k:m, trans_a?m:k, ctx1);

        /// step 2: calibrate matrix a into int8 matrix
        ///         using col direction.
        /// input : const float* src
        /// output: char* dst_int8
        ///         float* scale
        float2char(!trans_b, (signed char*)b_dev_int8.mutable_data(),
                   (const float*)b_dev.data(),
                   (float*)b_scale.mutable_data(),
                   trans_b?n:k, trans_b?k:n, ctx1);

        /// step 3: dispatch matrix multiply using int8 gemm
        gemm_saber.dispatch(alpha, beta,
                            (const char *) a_dev_int8.data(),
                            (const char *) b_dev_int8.data(),
                            (float *) c_dev.mutable_data());

        /// step 4: convert int32 into float32 using fix2float.
        /// input : float* dst
        ///         const scaleA(row scale)
        ///         const scaleB(col scale)
        /// output: float* dst
        fix2float((float*)c_dev.mutable_data(),
                  (const float*)a_scale.data(),
                  (const float*)b_scale.data(),
                  alpha, beta, m, n, ctx1);

        typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
        c_dev.record_event(stream);
        c_dev.sync();

        c_host.copy_from(c_dev);
        gemm_check(m, n, k, (const float *) a_host.data(), (const float *) b_host.data(),
                   (float *) c_check.mutable_data(),
                   alpha, beta, trans_a, trans_b);

        int counts = count_diff((const float*)c_check.data(), (const float*)c_host.data(),
                                c_check.valid_size(), 1e-1);

        if (((double)counts / (double)c_host.valid_size()) > 0.05) {
            print_tensor_valid(c_check);
            print_tensor_valid(c_host);
                    LOG(FATAL) << "VENDER: FAIL!!!! counts = " <<counts
                               << "m = "<< m << " n = "<< n << " k = "<< k;
        }
        for (int t = 0; t < ts; ++t) {
            saber_time.start(ctx1);
            gemm_saber.dispatch(alpha, beta,
                                 (const char *) a_dev_int8.data(),
                                 (const char *) b_dev_int8.data(),
                                 (float *) c_dev.mutable_data());
            typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
            c_dev.record_event(stream);
            c_dev.sync();
            saber_time.end(ctx1);
        }
    }
    LOG(INFO) << "Vender time: " << (vender_status == SaberSuccess ? vender_time.get_average_ms() : 0)
              << " ms Saber time: " << (saber_status == SaberSuccess ? saber_time.get_average_ms() : 0)
              << " ms";
}

TEST(TestSaberFunc, test_vender_gemm_float) {

    std::vector<int> m_v = {40, 20, 140, 200, 300};
    std::vector<int> n_v = {10, 20, 140, 200, 300};
    std::vector<int> k_v = {40, 20, 140, 200, 300};
    std::vector<int> trans_a_v{false, true};
    std::vector<int> trans_b_v{false, true};

    for (auto m : m_v)
    for (auto n : n_v)
    for (auto k : k_v)
    for (auto trans_a : trans_a_v)
    for (auto trans_b : trans_b_v) {

#ifdef USE_CUDA
        test_gemm_int8_result<NV, NVHX86>(m, n, k, trans_a, trans_b);
#endif

#ifdef USE_X86_PLACE
//        test_gemm_int8_result<X86, X86>(m, n, k, trans_a, trans_b);
#endif
    }
}

int main(int argc, char* argv[]) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    Env<NVHX86>::env_init();
#endif
#ifdef USE_X86_PLACE
    Env<X86>::env_init();
#endif
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}