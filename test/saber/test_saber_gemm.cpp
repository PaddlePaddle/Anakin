#include "saber/core/context.h"
#include "saber/funcs/gemm.h"
#include "saber/core/tensor.h"
#include "saber/core/tensor_op.h"
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

template <typename TargetType, typename TargetType_H, ImplEnum impl>
void test_result (int m, int n, int k, bool trans_a, bool trans_b) {

    Tensor<TargetType> a_dev, b_dev, c_dev;
    Tensor<TargetType_H> a_host, b_host, c_host, c_check;

    Context<TargetType> ctx1(0, 1, 0);
    Gemm<TargetType, impl, float> gemm_op;
    gemm_op.init(trans_a, trans_b, m, n, k, ctx1);
    float alpha = 1.f;
    float beta = 0.f;

    Shape a_shape({m, k}, Layout_HW);
    Shape b_shape({k, n}, Layout_HW);
    Shape c_shape({m, n}, Layout_HW);

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

    gemm_op.dispatch(alpha, beta,
                     (const float*)a_dev.data(),
                     (const float*)b_dev.data(),
                     (float*)c_dev.mutable_data());
    typename Tensor<TargetType>::API::stream_t stream = ctx1.get_compute_stream();
    c_dev.record_event(stream);
    c_dev.sync();
    c_host.copy_from(c_dev);
    gemm_check(m, n, k, (const float*)a_host.data(), (const float*)b_host.data(),
               (float*)c_check.mutable_data(),
               alpha, beta, trans_a, trans_b);
    double max_ratio = 0.f, max_diff = 0.f;
    tensor_cmp_host((const float*)c_check.data(), (const float*)c_host.data(),
                    c_check.valid_size(), max_ratio, max_diff);
    if (max_ratio < 1e-3){
        LOG(INFO) << "PASS!!!! max_ratio = " <<max_ratio << " max_diff: "<< max_diff;
    }else {
        print_tensor_valid(c_check);
        print_tensor_valid(c_host);
        LOG(FATAL) << "FAIL!!!! max_ratio = " <<max_ratio << " max_diff: "<< max_diff
                   << "m = "<< m<< " n = "<< n << " k = "<< k;
    }
}

TEST(TestSaberFunc, test_vender_gemm_float) {

    std::vector<int> m_v = {5, 10, 15, 20, 25, 30};
    std::vector<int> n_v = {5, 10, 15, 20, 25, 30};
    std::vector<int> k_v = {5, 10, 15, 20, 25, 30};
    std::vector<int> trans_a_v{false, true};
    std::vector<int> trans_b_v{false, true};

    for (auto m : m_v)
    for (auto n : n_v)
    for (auto k : k_v)
    for (auto trans_a : trans_a_v)
    for (auto trans_b : trans_b_v) {

#ifdef USE_CUDA
        test_result<NV, NVHX86, VENDER_IMPL>(m, n, k, trans_a, trans_b);
        test_result<NV, NVHX86, SABER_IMPL>(m, n, k, trans_a, trans_b);
#endif

#ifdef USE_X86_PLACE
        test_result<X86, X86, VENDER_IMPL>(m, n, k, trans_a, trans_b);
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
