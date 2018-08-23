#include "saber/funcs/impl/cuda/saber_fc.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"

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

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus SaberFc<NV, OpDtype, inDtype, outDtype, \
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

    if (bias_term) {
        bias = param.bias->data();
    }

    float alpha = 1.f;
    float beta = 0.f;
    _kernel(_M, _N, _K, alpha, din, beta, weight, dout, stream);
    if (bias_term) {
        int total_size = _M * _N;
        add_bias<InDataType><<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>\
            (total_size, _N, bias, dout);
    }
    return SaberSuccess;
}

} //namespace anakin

} //namespace anakin
