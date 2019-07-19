#include "saber/funcs/impl/cuda/vender_batch_gemm.h"

namespace anakin {
namespace saber {

template<>
SaberStatus BatchGemm<NV, VENDER_IMPL, float, float>::init(const bool trans_a, const bool trans_b, const int max_batch,
                 Context<NV> ctx) {

    if ((!(ctx == this->_ctx)) || (_handle == nullptr)) {
        if (_handle != NULL) {
            CUBLAS_CHECK(cublasDestroy(_handle));
        }
        this->_ctx = ctx;
        cudaStream_t cuda_stream = ctx.get_compute_stream();
        CUBLAS_CHECK(cublasCreate(&_handle));
        CUBLAS_CHECK(cublasSetStream(_handle, cuda_stream));
    }

    _trans_a = trans_a;
    _trans_b = trans_b;
    _cu_trans_a = trans_a ? CUBLAS_OP_T: CUBLAS_OP_N;
    _cu_trans_b = trans_b ? CUBLAS_OP_T: CUBLAS_OP_N;
    cudaMalloc((void**)&_A, 3 * max_batch * sizeof(float*));
    return SaberSuccess;
}

template<>
SaberStatus BatchGemm<NV, VENDER_IMPL, float, float>::dispatch(
                     const float alpha, const float beta,
                     const float* ptr_a[], const float* ptr_b[], 
                     const int m, const int n, const int k,
                     float* ptr_c[], const int batch) {

    CHECK(ptr_a != nullptr);
    CHECK(ptr_b != nullptr);
    CHECK(ptr_c != nullptr);
    _lda = (!_trans_a) ? k : m;
    _ldb = (!_trans_b) ? n : k;
    _ldc = n;
    _m = m;
    _n = n;
    _k = k;
    cudaStream_t cuda_stream = this->_ctx.get_compute_stream();
    _B = _A + batch;
    _C = _B + batch;
    cudaMemcpyAsync(_A, ptr_a, batch * sizeof(const float*), cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(_B, ptr_b, batch * sizeof(const float*), cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(_C, ptr_c, batch * sizeof(const float*), cudaMemcpyHostToDevice, cuda_stream);
    CUBLAS_CHECK(cublasSgemmBatched(_handle, _cu_trans_b,
                             _cu_trans_a,
                             _n, _m, _k, &alpha,
                              _B, _ldb,
                              _A, _lda, 
                              &beta, _C, _ldc, batch));
    return SaberSuccess;
}

template<>
SaberStatus BatchGemm<NV, VENDER_IMPL, float, float>::dispatch(
                     const float alpha, const float beta,
                     float* ptr_a[], 
                     const int m, const int n, const int k,
                     const int batch) {

    CHECK(ptr_a != nullptr);
    _lda = (!_trans_a) ? k : m;
    _ldb = (!_trans_b) ? n : k;
    _ldc = n;
    _m = m;
    _n = n;
    _k = k;
    cudaStream_t cuda_stream = this->_ctx.get_compute_stream();
    cudaMemcpyAsync(_A, ptr_a, 3 * batch * sizeof(const float*), cudaMemcpyDefault, cuda_stream);
    CUBLAS_CHECK(cublasSgemmBatched(_handle, _cu_trans_b,
                             _cu_trans_a,
                             _n, _m, _k, &alpha,
                              _A + batch, _ldb,
                              _A, _lda, 
                              &beta, _A + 2*batch, _ldc, batch));
    return SaberSuccess;
}


}
}
