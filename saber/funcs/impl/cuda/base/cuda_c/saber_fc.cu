#include "saber/funcs/impl/cuda/saber_fc.h"
#include "sass_funcs.h"

namespace anakin{

namespace saber{
template <typename dtype>
__global__ void add_bias(int n, int output_size, const dtype* bias, dtype* dout) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int bias_index = index % output_size;
    if (index < n) {
        dout[index] = dout[index] + bias[bias_index];
    }
}

template <DataType OpDtype>
SaberStatus SaberFc<NV, OpDtype>::dispatch(
            const std::vector<Tensor<NV> *>& inputs,
            std::vector<Tensor<NV> *>& outputs,
            FcParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();

    const OpDataType *din = (const OpDataType *)inputs[0]->data();
    OpDataType *dout = (float *)outputs[0]->mutable_data();
    const OpDataType *weight = (OpDataType *)param.weights->data();
    const OpDataType *bias = nullptr;


    bool bias_term = param.bias != nullptr;

    if (bias_term) {
        bias = (const OpDataType *)param.bias->data();
    }
    
    float alpha = 1.f;
    float beta = 0.f;

    _kernel(_M, _N, _K, alpha, din, beta, weight, dout, stream);

    if (bias_term) {
        int total_size = _M * _N;
        add_bias<OpDataType><<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>\
            (total_size, _N, bias, dout);
    }
    return SaberSuccess;
}

template class SaberFc<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberFc, FcParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberFc, FcParam, NV, AK_INT8);
} //namespace anakin

} //namespace anakin
