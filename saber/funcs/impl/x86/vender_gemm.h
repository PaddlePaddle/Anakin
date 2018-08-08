
#ifndef SABER_FUNCS_IMPL_X86_VENDER_GEMM_H
#define SABER_FUNCS_IMPL_X86_VENDER_GEMM_H

#include "saber/core/tensor.h"
#include "mkl.h"

namespace anakin {
namespace saber {

template<typename Dtype>
class Gemm<X86, VENDER_IMPL, Dtype> {

public:
    Gemm() = default;
    ~Gemm() {}

    SaberStatus init(const bool trans_a, const bool trans_b,
                     const int m, const int n, const int k,
                     Context<X86> ctx){

        _lda = (!trans_a) ? k : m;
        _ldb = (!trans_b) ? n : k;
        _ldc = n;
        _m = m;
        _n = n;
        _k = k;
        c_trans_a = trans_a ? CblasTrans: CblasNoTrans;
        c_trans_b = trans_b ? CblasTrans: CblasNoTrans;
        return SaberSuccess;
    }

    SaberStatus dispatch(const float alpha, const float* ptr_a,
                         const float beta, const float* ptr_b, float* ptr_c) {

        cblas_sgemm(_layout, c_trans_a, c_trans_b, _m, _n, _k,
                    alpha, ptr_a, _lda, ptr_b, _ldb, beta, ptr_c, _ldc);
        return SaberSuccess;
    }

private:
    Context<X86> _ctx;
    CBLAS_LAYOUT _layout = CblasRowMajor;
    CBLAS_TRANSPOSE c_trans_a;
    CBLAS_TRANSPOSE c_trans_b;
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