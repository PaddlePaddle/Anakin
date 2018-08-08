
#ifndef SABER_FUNCS_IMPL_CUDA_VENDER_GEMM_H
#define SABER_FUNCS_IMPL_CUDA_VENDER_GEMM_H

#include "saber/core/tensor.h"
#include "saber/funcs/gemm.h"

namespace anakin {
namespace saber {

template<typename Dtype>
class Gemm<NV, VENDER_IMPL, Dtype> {

public:
    Gemm() = default;
    ~Gemm() {}

    SaberStatus init(const bool trans_a, const bool trans_b,
                     const int m, const int n, const int k,
                     Context<NV> ctx);

    SaberStatus dispatch(const Dtype alpha, const Dtype* ptr_a,
                         const Dtype beta, const Dtype* ptr_b, Dtype* ptr_c);

private:
    Context<NV> _ctx;
    cublasHandle_t _handle{nullptr};
    cublasOperation_t cu_trans_a;
    cublasOperation_t cu_trans_b;
    int _m{-1};
    int _n{-1};
    int _k{-1};
    int _lda{-1};
    int _ldb{-1};
    int _ldc{-1};
};

}
}

#endif