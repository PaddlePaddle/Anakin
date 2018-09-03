#include "saber/funcs/impl/cuda/vender_fc.h"


namespace anakin{

namespace saber{

template <typename dtype>
void anakin_NV_gemv(cublasHandle_t handle, const bool TransA, \
					 const int M, const int N, \
					 const dtype alpha, const dtype* A,\
					 const dtype* x, const dtype beta,\
					 dtype* y);

template <>
void anakin_NV_gemv<float>(cublasHandle_t handle, const bool TransA, \
    const int M, const int N, const float alpha, const float* A, const float* x, \
    const float beta, float* y) {
    cublasOperation_t cuTransA = (TransA == false) ? CUBLAS_OP_T : CUBLAS_OP_N;
        CUBLAS_CHECK(cublasSgemv(handle, cuTransA, N, M, &alpha,
                                 A, N, x, 1, &beta, y, 1));
    }

template <>
void anakin_NV_gemv<char>(cublasHandle_t handle, const bool TransA, \
					 const int M, const int N, \
					 const char alpha, const char* A,\
					 const char* x, const char beta,\
					 char* y) {
    LOG(FATAL) << "int8 gemv is not implemented";
}

template <typename dtype>
void anakin_NV_gemm(cublasHandle_t handle, const bool TransA,
                    const bool TransB, const int M, const int N, const int K,
                    const dtype alpha, const dtype* A, const dtype* B, const dtype beta,
                    dtype* C);

template <>
void anakin_NV_gemm<float>(cublasHandle_t handle, const bool TransA,
                           const bool TransB, const int M, const int N, const int K,
                           const float alpha, const float* A, const float* B, const float beta,
                           float* C) {
    // Note that cublas follows fortran order.
    int lda = (!TransA/* == CblasNoTrans*/) ? K : M;
    int ldb = (!TransB/* == CblasNoTrans*/) ? N : K;
    cublasOperation_t cuTransA =
            (!TransA/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
            (!TransB/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA,
                             N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void anakin_NV_gemm<char>(cublasHandle_t handle, const bool TransA,
                           const bool TransB, const int M, const int N, const int K,
                           const char alpha, const char* A, const char* B, const char beta,
                           char* C) {
    LOG(FATAL) << "int8 gemm is not implemented";
}

template <typename dtype>
__global__ void add_bias(int n, int output_size, const dtype* bias, dtype* dout) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int bias_index = index % output_size;
    if (index < n) {
        //printf("index: %d, bias_index: %d, val_in: %.2f\n", index, bias_index, bias[bias_index]);
        dout[index] = dout[index] + bias[bias_index];
    }
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus VenderFc<NV, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
            const std::vector<DataTensor_in *>& inputs,
            std::vector<DataTensor_out *>& outputs,
            FcParam<OpTensor>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();

    const InDataType* din = inputs[0]->data();
    OutDataType* dout = outputs[0]->mutable_data();
    const OpDataType* weight = param.weights->data();
    const InDataType* bias = nullptr;
    bool bias_term = param.bias != nullptr;
    //dim3 grid(CUDA_GET_BLOCKS(param.num_output), _M);
    if (bias_term) {
        bias = param.bias->data();
    }

    if (_M == 1 && _K > 50000) {
        anakin_NV_gemv<InDataType>(_handle, false, _N, _K, (InDataType)1, weight, din, \
            (InDataType)0, dout);
    } else {
        anakin_NV_gemm<InDataType>(_handle, false, !_flag_trans_weights, \
            _M, _N, _K, (InDataType)1, din, weight, (InDataType)0, dout);
    }
    if (bias_term) {
        int total_size = _M * _N;
        add_bias<InDataType><<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>\
            (total_size, _N, bias, dout);
    }
    return SaberSuccess;
}

/*
#define INSTANCE_FC(datatype, layouttype) \
    template SaberStatus CublasFc<datatype, layouttype>::dispatch( \
        const std::vector<CublasFc<datatype, layouttype>::ioTensor *> inputs, \
        std::vector<CublasFc<datatype, layouttype>::ioTensor *> outputs, \
        FcParam<CublasFc<datatype, layouttype>::ioTensor> &param);

INSTANCE_FC(AK_FLOAT, NCHW);
INSTANCE_FC(AK_INT8, NCHW);
INSTANCE_FC(AK_FLOAT, NHWC);
INSTANCE_FC(AK_INT8, NHWC);
*/
} //namespace anakin

} //namespace anakin
