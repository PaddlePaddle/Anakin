
#include "saber/funcs/impl/cuda/saber_gemm.h"
#include "sass_funcs.h"
namespace anakin {
namespace saber {

template <>
SaberStatus Gemm<NV, SABER_IMPL, float, float>::init(const bool trans_a, const bool trans_b,
                 const int m, const int n, const int k,
                 Context<NV> ctx) {

    if (!(ctx == this->_ctx)) {
        this->_ctx = ctx;
    }
    _m = m;
    _n = n;
    _k = k;
    _kernel =saber_find_fast_sass_gemm(trans_a, trans_b, _m, _n, _k);
    return SaberSuccess;
}
template <>
SaberStatus Gemm<NV, SABER_IMPL, float, float>::dispatch(const float alpha, const float beta,
                                      const float* ptr_a, const float* ptr_b, float* ptr_c) {
    CHECK(ptr_a != nullptr);
    CHECK(ptr_b != nullptr);
    CHECK(ptr_c != nullptr);
    cudaStream_t cuda_stream = _ctx.get_compute_stream();
    _kernel(_m, _n, _k, alpha, ptr_a, beta, ptr_b, ptr_c, cuda_stream);
    return SaberSuccess;
}
template <>
SaberStatus Gemm<NV, SABER_IMPL, char, float>::init(const bool trans_a, const bool trans_b,
                                                     const int m, const int n, const int k,
                                                     Context<NV> ctx) {
    return SaberUnImplError;
}
template <>
SaberStatus Gemm<NV, SABER_IMPL, char, float>::dispatch(const float alpha, const float beta,
                                                         const char* ptr_a, const char* ptr_b, float* ptr_c) {

    return SaberUnImplError;
}

}
}