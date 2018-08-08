
#include "saber/funcs/impl/x86/vender_gemm.h"

namespace anakin {

namespace saber {

template <>
SaberStatus Gemm<X86, VENDER_IMPL, float, float>::init(
        const bool trans_a, const bool trans_b,
        const int m, const int n, const int k,
        Context<X86> ctx){

    if (!(ctx == this->_ctx)) {
        _ctx = ctx;
    }
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

template <>
SaberStatus Gemm<X86, VENDER_IMPL, float, float>::dispatch(
        const float alpha, const float beta,
        const float* ptr_a, const float* ptr_b, float* ptr_c) {
    CHECK(ptr_a != nullptr);
    CHECK(ptr_b != nullptr);
    CHECK(ptr_c != nullptr);
    cblas_sgemm(_layout, c_trans_a, c_trans_b, _m, _n, _k,
                alpha, ptr_a, _lda, ptr_b, _ldb, beta, ptr_c, _ldc);
    return SaberSuccess;
}

}

}