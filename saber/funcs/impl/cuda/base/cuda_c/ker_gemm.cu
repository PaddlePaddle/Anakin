
#include "sass_funcs.h"

namespace anakin {
namespace saber {

template <>
void ker_sgemm_sass<false, false, 32>(const int M, const int N, const int K,
                        const float alpha, const float* A,
                        const float beta, const float* B,
                        float* C, cudaStream_t cuda_stream) {

    int lda = 0, ldb = 0, ldc = 0;
    lda = K;
    ldb = N; ldc = N;

    if (ifVec(M, N, K, lda, ldb, ldc)) {
        ker_sgemm_nn_vec<32>(M, N, K, lda, ldb, ldc, alpha, A,
                                 beta, B, C, cuda_stream);
    } else {
        ker_sgemm_nn<32>(M, N, K, lda, ldb, ldc, alpha, A,
                                 beta, B, C, cuda_stream);
    }

}

template <>
void ker_sgemm_sass<false, true, 32>(const int M, const int N, const int K,
                                    const float alpha, const float* A,
                                    const float beta, const float* B,
                                    float* C, cudaStream_t cuda_stream) {
    int lda = 0, ldb = 0, ldc = 0;
    lda = K;
    ldb = K; ldc = N;
    if (ifVec(M, N, K, lda, ldb, ldc)) {
        ker_sgemm_nt_vec<32>(M, N, K, lda, ldb, ldc, alpha, A,
                                 beta, B, C, cuda_stream);
    } else {
        ker_sgemm_nt<32>(M, N, K, lda, ldb, ldc, alpha, A,
                                 beta, B, C, cuda_stream);
    }
}

template <>
void ker_sgemm_sass<true, false, 32>(const int M, const int N, const int K,
                                    const float alpha, const float* A,
                                    const float beta, const float* B,
                                    float* C, cudaStream_t cuda_stream) {
    int lda = 0, ldb = 0, ldc = 0;
    lda = M;
    ldb = N; ldc = N;
    if (ifVec(M, N, K, lda, ldb, ldc)) {
        ker_sgemm_tn_vec<32>(M, N, K, lda, ldb, ldc, alpha, A,
                                 beta, B, C, cuda_stream);
    } else {
        ker_sgemm_tn<32>(M, N, K, lda, ldb, ldc, alpha, A,
                             beta, B, C, cuda_stream);
    }

}

template <>
void ker_sgemm_sass<true, true, 32>(const int M, const int N, const int K,
                                    const float alpha, const float* A,
                                    const float beta, const float* B,
                                    float* C, cudaStream_t cuda_stream) {
    int lda = 0, ldb = 0, ldc = 0;
    lda = M;
    ldb = K; ldc = N;
    if (ifVec(M, N, K, lda, ldb, ldc)) {
        ker_sgemm_tt_vec<32>(M, N, K, lda, ldb, ldc,
                                 alpha, A, beta, B,
                                 C, cuda_stream);
    } else {
        ker_sgemm_tt<32>(M, N, K, lda, ldb, ldc,
                             alpha, A, beta, B,
                             C, cuda_stream);
    }
}

template <>
void ker_sgemm_sass<false, false, 128>(const int M, const int N, const int K,
                                const float alpha, const float* A,
                                const float beta, const float* B,
                                float* C, cudaStream_t cuda_stream) {

    int lda = 0, ldb = 0, ldc = 0;
    lda = K;
    ldb = N; ldc = N;

    if (ifVec(M, N, K, lda, ldb, ldc)) {
        ker_sgemm_nn_vec<128>(M, N, K, lda, ldb, ldc,
                             alpha, A, beta, B,
                             C, cuda_stream);
    } else {
        ker_sgemm_nn<128>(M, N, K, lda, ldb, ldc,
                         alpha, A, beta, B,
                         C, cuda_stream);
    }
}

template <>
void ker_sgemm_sass<false, true, 128>(const int M, const int N, const int K,
                                 const float alpha, const float* A,
                                 const float beta, const float* B,
                                 float* C, cudaStream_t cuda_stream) {
    int lda = 0, ldb = 0, ldc = 0;
    lda = K;
    ldb = K; ldc = N;
    if (ifVec(M, N, K, lda, ldb, ldc)) {
        ker_sgemm_nt_vec<128>(M, N, K, lda, ldb, ldc, alpha, A,
                                 beta, B, C, cuda_stream);
    } else {
        ker_sgemm_nt<128>(M, N, K, lda, ldb, ldc, alpha, A,
                                 beta, B, C, cuda_stream);
    }
}

template <>
void ker_sgemm_sass<true, false, 128>(const int M, const int N, const int K,
                                 const float alpha, const float* A,
                                 const float beta, const float* B,
                                 float* C, cudaStream_t cuda_stream) {
    int lda = 0, ldb = 0, ldc = 0;
    lda = M;
    ldb = N; ldc = N;
    if (ifVec(M, N, K, lda, ldb, ldc)) {
        ker_sgemm_tn_vec<128>(M, N, K, lda, ldb, ldc, alpha, A,
                                 beta, B, C, cuda_stream);
    } else {
        ker_sgemm_tn<128>(M, N, K, lda, ldb, ldc, alpha, A,
                                 beta, B, C, cuda_stream);
    }
}

template <>
void ker_sgemm_sass<true, true, 128>(const int M, const int N, const int K,
                                  const float alpha, const float* A,
                                  const float beta, const float* B,
                                  float* C, cudaStream_t cuda_stream) {
    int lda = 0, ldb = 0, ldc = 0;
    lda = M;
    ldb = K; ldc = N;
    if (ifVec(M, N, K, lda, ldb, ldc)) {
        ker_sgemm_tt_vec<128>(M, N, K, lda, ldb, ldc, alpha, A,
                                 beta, B, C, cuda_stream);
    } else {
        ker_sgemm_tt<128>(M, N, K, lda, ldb, ldc, alpha, A,
                                 beta, B, C, cuda_stream);
    }
}

std::function<void(const int, const int, const int,
                   const float, const float*, const float,
                   const float*, float*, cudaStream_t)>
saber_find_fast_sass_gemm (const bool TransA, const bool TransB, const int M, const int N, const int K) {

    bool choose_128_gemm = false;

    if (choose_128_gemm) {
        if (TransA && TransB) {
            return ker_sgemm_sass<true, true, 128>;
        }
        if (TransA && !TransB) {
            return ker_sgemm_sass<true, false, 128>;
        }
        if (!TransA && TransB) {
            return ker_sgemm_sass<false, true, 128>;
        }
        if (!TransA && !TransB) {
            return ker_sgemm_sass<false, false, 128>;
        }
    } else {
        if (TransA && TransB) {
            return ker_sgemm_sass<true, true, 32>;
        }
        if (TransA && !TransB) {
            return ker_sgemm_sass<true, false, 32>;
        }
        if (!TransA && TransB) {
            return ker_sgemm_sass<false, true, 32>;
        }
        if (!TransA && !TransB) {
            return ker_sgemm_sass<false, false, 32>;
        }
    }
}

}
}
