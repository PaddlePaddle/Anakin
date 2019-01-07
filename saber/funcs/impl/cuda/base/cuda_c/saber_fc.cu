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

template <>
SaberStatus SaberFc<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        FcParam<NV>& param, Context<NV>& ctx){

    if (!(&ctx == this->_ctx)) {
        this->_ctx = &ctx;
    }

    Shape shape_out = inputs[0]->valid_shape();
    _M = inputs[0]->count_valid(0, param.axis);
    _K = inputs[0]->count_valid(param.axis, inputs[0]->dims());
    _N = param.num_output;
    if (_N <= 0) {
        int weight_size = param.weights->valid_size();
        _N = weight_size / _K;
    }
    //! weights dims must be in h and w
    _gemm.init(false, !_flag_trans_weights, _M, _N, _K, *_ctx);

    return SaberSuccess;
}

template <>
SaberStatus SaberFc<NV, AK_FLOAT>::init(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        FcParam<NV>& param, Context<NV> &ctx) {
    // get context
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberFc<NV, AK_FLOAT>::dispatch(
            const std::vector<Tensor<NV> *>& inputs,
            std::vector<Tensor<NV> *>& outputs,
            FcParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();

    const float *din = (const float *)inputs[0]->data();
    float *dout = (float *)outputs[0]->mutable_data();
    const float *weight = (float *)param.weights->data();
    const float *bias = nullptr;

    bool bias_term = param.bias != nullptr;

    if (bias_term) {
        bias = (const float *)param.bias->data();
    }
    
    float alpha = 1.f;
    float beta = 0.f;

    _gemm.dispatch(alpha, beta, din, weight, dout);

    if (bias_term) {
        int total_size = _M * _N;
        add_bias<float><<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>\
            (total_size, _N, bias, dout);
    }
    return SaberSuccess;
}

template class SaberFc<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberFc, FcParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberFc, FcParam, NV, AK_INT8);
} //namespace anakin

} //namespace anakin
