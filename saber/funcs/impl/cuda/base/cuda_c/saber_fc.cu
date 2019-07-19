#include "saber/funcs/impl/cuda/saber_fc.h"
#include "saber/funcs/calibrate.h"
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
    _flag_trans_weights = param.is_transpose_weights;
    if (_N <= 0) {
        int weight_size = param.weights->valid_size();
        _N = weight_size / _K;
    }
    //! weights dims must be in h and w
    _gemm->init(false, !_flag_trans_weights, _M, _N, _K, *_ctx);

    return SaberSuccess;
}

template <>
SaberStatus SaberFc<NV, AK_FLOAT>::init(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        FcParam<NV>& param, Context<NV> &ctx) {
    // get context
    this->_ctx = &ctx;
    int generate_arch = Env<NV>::cur_env()[_ctx->get_device_id()]._info._generate_arch;
    bool arch_check = (generate_arch == 50) || (generate_arch == 61);
    if (arch_check) {
        _gemm = new Gemm<NV, SABER_IMPL, float, float>;
    } else {
        _gemm = new Gemm<NV, VENDER_IMPL, float, float>;
    }
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberFc<NV, AK_INT8>::create(
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
    _flag_trans_weights = param.is_transpose_weights;
    if (_N <= 0) {
        int weight_size = param.weights->valid_size();
        _N = weight_size / _K;
    }
    //! weights dims must be in h and w
    _gemm_s8f32->init(false, !_flag_trans_weights, _M, _N, _K, *_ctx);

    return SaberSuccess;
}

template <>
SaberStatus SaberFc<NV, AK_INT8>::init(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        FcParam<NV>& param, Context<NV> &ctx) {
    // get context
    this->_ctx = &ctx;
    int generate_arch = Env<NV>::cur_env()[_ctx->get_device_id()]._info._generate_arch;
    bool arch_check = generate_arch == 61;

    if (arch_check) {
        _gemm_s8f32 = new Gemm<NV, VENDER_IMPL, char, float>;
        if (param.weights->get_dtype() == AK_FLOAT) {
            Tensor<NVHX86> _host_weight;
            _trans_weight.re_alloc(param.weights->valid_shape(), AK_INT8);
            _host_weight.re_alloc(param.weights->valid_shape(), AK_FLOAT);
            _host_weight.copy_from(*param.weights);
            std::vector<float> scale;
            get_tensor_scale(scale, _host_weight, 0, false);
            param.weights->set_scale(scale);
            _trans_weight.set_scale(scale);
            flatten_calibrate<NV, char, float>(_trans_weight, *param.weights, ctx);
            _trans_weight.record_event(ctx.get_compute_stream());
            _trans_weight.sync();
        }
    } else {
        LOG(FATAL) << "not support this arch!! ";
    }

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberFc<NV, AK_INT8>::dispatch(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs, FcParam<NV>& param) {
    cudaStream_t stream = this->_ctx->get_compute_stream();

    const char *din = (const char *)inputs[0]->data();
    float *dout = (float *)outputs[0]->mutable_data();
    const char *weight = nullptr;
    if (param.weights->get_dtype() == AK_INT8) {
        weight = (const char *)param.weights->data();
    } else {
        weight = (const char *)_trans_weight.data();
    }

    const float *bias = nullptr;

    bool bias_term = param.bias != nullptr;

    if (bias_term) {
        bias = (const float *)param.bias->data();
    }
    _inner_tensor.re_alloc(inputs[0]->valid_shape(), AK_INT8);

    layout_trans_nchwc4_2_nchw(_inner_tensor, *inputs[0],
            inputs[0]->get_scale()[0], *_ctx);
    din = (const char*)_inner_tensor.data();

    float beta = 0.f;
    float alpha = 1.f;
    if (param.weights->get_scale().size() == 1) {
        CHECK_GE(inputs[0]->get_scale().size(), 1);
        alpha = inputs[0]->get_scale()[0] * param.weights->get_scale()[0];
    }
    if (outputs[0]->get_dtype() == AK_INT8) {
        LOG(FATAL) << " this is not right!";
//        CHECK_GE(outputs[0]->get_scale().size(), 1);
//        alpha /= outputs[0]->get_scale()[0];
    }
    _gemm_s8f32->dispatch(alpha, beta, din, weight, dout);

    if (bias_term) {
        int total_size = _M * _N;
        add_bias<float><<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>\
            (total_size, _N, bias, dout);
    }
    return SaberSuccess;
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

    _gemm->dispatch(alpha, beta, din, weight, dout);

    if (bias_term) {
        int total_size = _M * _N;
        add_bias<float><<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>\
            (total_size, _N, bias, dout);
    }
    return SaberSuccess;
}

template class SaberFc<NV, AK_INT8>;
template class SaberFc<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberFc, FcParam, NV, AK_HALF);
} //namespace anakin

} //namespace anakin
