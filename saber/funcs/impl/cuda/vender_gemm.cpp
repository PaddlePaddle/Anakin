
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

template<>
SaberStatus Gemm<NV, VENDER_IMPL, char, float>::init(const bool trans_a, const bool trans_b,
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
SaberStatus Gemm<NV, VENDER_IMPL, char, float>::dispatch(
        const float alpha, const float beta,
        const char* ptr_a, const char* ptr_b, float* ptr_c) {

    CHECK(ptr_a != nullptr);
    CHECK(ptr_b != nullptr);
    CHECK(ptr_c != nullptr);

//    CUBLAS_CHECK(cublasGemmEx(_handle, cu_trans_b, cu_trans_a,
//            _n, _m, _k, &alpha, ptr_b, CUDA_R_8I, _ldb, ptr_a,
//            CUDA_R_8I, _lda, &beta, ptr_c, CUDA_R_32F, _ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

    CUBLAS_CHECK(cublasSgemmEx(_handle, cu_trans_b, cu_trans_a,
                               _n, _m, _k, &alpha, ptr_b, CUDA_R_8I, _ldb, ptr_a,
                               CUDA_R_8I, _lda, &beta, ptr_c, CUDA_R_32F, _ldc));
    return SaberSuccess;
}

template<>
SaberStatus Gemv<NV, VENDER_IMPL, float, float>::init(const bool trans, const int m, const int n,
                                                      const int incx, const int incy,
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

    _lda = n;
    CHECK_GT(m, 0);
    CHECK_GT(n, 0);
    CHECK_GT(incx, 0);
    CHECK_GT(incy, 0);
    _m = m;
    _n = n;
    _incx = incx;
    _incy = incy;
    _cu_trans = trans ? CUBLAS_OP_N: CUBLAS_OP_T;

    return SaberSuccess;
}

template<>
SaberStatus Gemv<NV, VENDER_IMPL, float, float>::dispatch(
        const float alpha, const float beta,
        const float* a, const float* b,
        float* c) {

    CHECK(a != nullptr);
    CHECK(b != nullptr);
    CHECK(c != nullptr);
    CUBLAS_CHECK(cublasSgemv(_handle, _cu_trans, _n, _m,
                             &alpha, a, _lda, b, _incx, &beta, c, _incy));
    return SaberSuccess;
}

}
}