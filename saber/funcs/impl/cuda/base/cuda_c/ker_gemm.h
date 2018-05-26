
#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_BASE_CUDA_C_KER_GEMM_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_BASE_CUDA_C_KER_GEMM_H

#include "saber/core/tensor.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin {
namespace saber {

__inline__
bool ifVec(int m, int n, int k,
           int lda, int ldb, int ldc)
{
    bool vec_a = false;
    bool vec_b = false;
    bool vec_c = false;

    vec_a = ((lda & 3) == 0) && ((k & 3) == 0);
    vec_b = ((ldb & 3) == 0) && ((n & 3) == 0);
    vec_c = ((ldc & 3) == 0) && ((n & 3) == 0);

    return vec_a && vec_b && vec_c;
}

void ker_gemm_32x32x32_NN_bias_relu(const int M, const int N, const int K,
                                    const float alpha, const float* A,
                                    const float beta, const float* B,
                                    float* C, const float* bias, cudaStream_t cuda_stream);

void ker_gemm_32x32x32_NN_vec_bias_relu(const int M, const int N, const int K,
                                        const float alpha, const float* A,
                                        const float beta, const float* B,
                                        float* C, const float* bias, cudaStream_t cuda_stream);
}

}

#endif