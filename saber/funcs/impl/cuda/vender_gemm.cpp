
#include "saber/funcs/impl/cuda/vender_gemm.h"

namespace anakin {
namespace saber {

template<>
SaberStatus Gemm<NV, VENDER_IMPL, float, float>::init(const bool trans_a, const bool trans_b,
                 const int m, const int n, const int k,
                 Context<NV> ctx) {

    if (!(ctx == this->_ctx)) {
        if (_handle != NULL) {
            CUBLAS_CHECK(cublasDestroy(_handle));
        }
        this->_ctx = ctx;
        cudaStream_t cuda_stream = ctx.get_compute_stream();
        CUBLAS_CHECK(cublasCreate(&_handle));
        CUBLAS_CHECK(cublasSetStream(_handle, cuda_stream));
    }

    _lda = (!trans_a) ? k : m;
    _ldb = (!trans_b) ? n : k;
    _ldc = n;
    _m = m;
    _n = n;
    _k = k;
    cu_trans_a = trans_a ? CUBLAS_OP_T: CUBLAS_OP_N;
    cu_trans_b = trans_b ? CUBLAS_OP_T: CUBLAS_OP_N;

    return SaberSuccess;
}

template<>
SaberStatus Gemm<NV, VENDER_IMPL, float, float>::dispatch(
                     const float alpha, const float beta,
                     const float* ptr_a, const float* ptr_b, float* ptr_c) {

    CHECK(ptr_a != nullptr);
    CHECK(ptr_b != nullptr);
    CHECK(ptr_c != nullptr);
    CUBLAS_CHECK(cublasSgemm(_handle, cu_trans_b, cu_trans_a,
                             _n, _m, _k, &alpha, ptr_b, _ldb, ptr_a,
                             _lda, &beta, ptr_c, _ldc));

    return SaberSuccess;
}

}
}