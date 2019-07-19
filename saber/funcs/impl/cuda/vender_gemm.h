
#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_GEMM_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_VENDER_GEMM_H

#include "saber/core/tensor.h"
#include "saber/funcs/gemm.h"

namespace anakin {
namespace saber {

template<typename inDtype,
        typename outDtype>
class Gemm<NV, VENDER_IMPL, inDtype, outDtype>
        : public MatrixFunc<NV, inDtype, outDtype>{

public:
    Gemm() = default;
    ~Gemm() {}

    SaberStatus init(const bool trans_a, const bool trans_b,
                     const int m, const int n, const int k,
                     Context<NV> ctx);

    SaberStatus dispatch(const outDtype alpha, const outDtype beta,
                         const inDtype* a, const inDtype* b,
                         outDtype* c);

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

template<typename inDtype,
        typename outDtype>
class Gemv<NV, VENDER_IMPL, inDtype, outDtype> {

public:
    Gemv() = default;
    ~Gemv() {}

    SaberStatus init(const bool trans, const int m, const int n,
                     const int incx, const int incy,
                     Context<NV> ctx);

    SaberStatus dispatch(const outDtype alpha, const outDtype beta,
                         const inDtype* a, const inDtype* b,
                         outDtype* c);

private:
    Context<NV> _ctx;
    cublasHandle_t _handle{nullptr};
    cublasOperation_t _cu_trans;
    int _incx{-1};
    int _incy{-1};
    int _m{-1};
    int _n{-1};
    int _lda{-1};
};

}
}

#endif