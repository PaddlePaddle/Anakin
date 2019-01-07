
#include "saber/funcs/impl/cuda/saber_conv_direct.h"
#include "saber/funcs/calibrate.h"
#include "saber_conv.h"
#include "saber/core/tensor_op.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberDirectConv<AK_FLOAT>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV> &ctx) {
    if (_use_saber_act) {
        _saber_act->create(inputs, outputs, param.activation_param, ctx);
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberDirectConv<AK_FLOAT>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV> &ctx) {

    this->_ctx = &ctx;
    _use_saber_act = param.activation_param.has_active
            && !(param.activation_param.active == Active_relu
            && fabsf(param.activation_param.negative_slope) < 1e-6f);
    _use_saber_act = _use_saber_act ||
            (param.bias()->valid_size() == 0 && param.activation_param.has_active);
    if (param.activation_param.has_active) {
        if (_use_saber_act) {
            _saber_act = new SaberActivation<NV, AK_FLOAT>;
            _saber_act->init(inputs, outputs, param.activation_param, ctx);
        }
    }  

    const int K = param.weight()->num();
    if (K % 4 == 0) {
        if (param.bias()->size() > 0 && (!param.activation_param.has_active || _use_saber_act)) {
            dispatch_func = direct_conv_bias_Kdivis4<float, float>;
        } else if (param.bias()->valid_size() > 0 && !_use_saber_act) {
            dispatch_func = direct_conv_bias_relu_Kdivis4<float, float>;
        } else {
            dispatch_func = direct_conv_Kdivis4<float, float>;
        }
    } else {
        if (param.bias()->size() > 0 && (!param.activation_param.has_active || _use_saber_act)) {
            dispatch_func = direct_conv_bias_Kindiv4<float, float>;
        } else if (param.bias()->valid_size() > 0 && !_use_saber_act) {
            dispatch_func = direct_conv_bias_relu_Kindiv4<float, float>;
        } else {
            dispatch_func = direct_conv_Kindiv4<float, float>;
        }
    }
    return create(inputs, outputs, param, ctx);

}

template <>
SaberStatus SaberDirectConv<AK_FLOAT>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {

    CHECK_EQ(inputs[0]->get_dtype(), AK_FLOAT);
    CHECK_EQ(outputs[0]->get_dtype(), AK_FLOAT);

    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();    
    const float* weight_ptr = (const float *) param.weight()->data();
    const float* bias_data = nullptr;
    if (param.bias()->size() > 0) {
        bias_data = (const float*)param.bias()->data();
    }

    dispatch_func((const float *) inputs[0]->data(),
                (float *) outputs[0]->mutable_data(),
                weight_ptr,
                bias_data,
                inputs[0]->num(),
                inputs[0]->channel(),
                inputs[0]->height(),
                inputs[0]->width(),
                outputs[0]->channel(),
                outputs[0]->height(),
                outputs[0]->width(),
                shape_in[1],
                shape_in[2],
                shape_in[3],
                shape_out[1],
                shape_out[2],
                shape_out[3],
                param.weight()->height(),
                param.weight()->width(),
                param.pad_h,
                param.pad_w,
                param.stride_h,
                param.stride_w,
                param.dilation_h,
                param.dilation_w,
                param.group,
                param.alpha,
                param.beta,
                this->_ctx->get_compute_stream());

    if (this->_saber_act != nullptr) {
        this->_saber_act->dispatch(outputs, outputs, param.activation_param);
    }
    CUDA_CHECK(cudaGetLastError());
    return SaberSuccess;
}
template <>
SaberStatus SaberDirectConv<AK_INT8>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param, Context<NV> &ctx) {

    if (&ctx != this->_ctx) {
        this->_ctx = &ctx;
    }

    int input_num = inputs[0]->num();
    int input_channel = inputs[0]->channel();
    int input_height = inputs[0]->height();
    int input_width = inputs[0]->width();
    int output_channel = outputs[0]->channel();
    int output_height = outputs[0]->height();
    int output_width = outputs[0]->width();
    int in_size = inputs[0]->valid_size();
    int out_size = outputs[0]->valid_size();

    // ====== int8 conv, the input channel must be a multiple of 4
    CHECK_EQ(input_channel % 4, 0);

    int kernel_h = param.weight()->height();
    int kernel_w = param.weight()->width();

    int filter_dim_a[] = {output_channel,
                          input_channel,
                          kernel_h, kernel_w};

    int pad_a[] = {param.pad_h, param.pad_w};
    int filter_stride_a[] = {param.stride_h, param.stride_w};
    int dilation_a[] = {param.dilation_h, param.dilation_w};

    return SaberSuccess;
}

template <>
SaberStatus SaberDirectConv<AK_INT8>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV> &ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}
template <>
SaberStatus SaberDirectConv<AK_INT8>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {

    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    const void* weight_ptr = param.weight()->data();
    const float* bias_data = nullptr;
    if (param.bias()->size() > 0) {
        bias_data = (const float*)param.bias()->data();
    }
    int in_channel_4 = inputs[0]->channel() / 4;

    if ((inputs[0]->channel() & 3) != 0) {
        LOG(FATAL) << "input channel is not a multipler of 4 in_channel = "
                   << inputs[0]->channel();
    }

    const void* in_data = nullptr;
    void* out_data = nullptr;

    in_data = (const void*)inputs[0]->data();
    out_data = (void*)outputs[0]->mutable_data();
    cudaStream_t stream = _ctx->get_compute_stream();
    float alpha = 1.f;
    if (param.weight()->get_scale().size() == 1) {
        alpha = inputs[0]->get_scale()[0] * param.weight()->get_scale()[0];
    }
    if (param.bias()->size() > 0 && (!param.activation_param.has_active || _use_saber_act)) {
        direct_conv_Kdivis4_s8_to_f32<true, false>(
                in_data,
                out_data,
                weight_ptr,
                bias_data,
                inputs[0]->num(),
                in_channel_4,
                inputs[0]->height(),
                inputs[0]->width(),
                outputs[0]->channel(),
                outputs[0]->height(),
                outputs[0]->width(),
                param.weight()->height(),
                param.weight()->width(),
                param.pad_h,
                param.pad_w,
                param.stride_h,
                param.stride_w,
                param.dilation_h,
                param.dilation_w,
                param.group,
                alpha,
                param.beta,
                this->_ctx->get_compute_stream());
    } else if (param.bias()->valid_size() > 0 &&
    (param.activation_param.has_active && !_use_saber_act)) {
        direct_conv_Kdivis4_s8_to_f32<true, true>(
                in_data,
                out_data,
                weight_ptr,
                bias_data,
                inputs[0]->num(),
                in_channel_4,
                inputs[0]->height(),
                inputs[0]->width(),
                outputs[0]->channel(),
                outputs[0]->height(),
                outputs[0]->width(),
                param.weight()->height(),
                param.weight()->width(),
                param.pad_h,
                param.pad_w,
                param.stride_h,
                param.stride_w,
                param.dilation_h,
                param.dilation_w,
                param.group,
                alpha,
                param.beta,
                this->_ctx->get_compute_stream());
    } else {
        direct_conv_Kdivis4_s8_to_f32<false, false>(
                in_data,
                out_data,
                weight_ptr,
                bias_data,
                inputs[0]->num(),
                in_channel_4,
                inputs[0]->height(),
                inputs[0]->width(),
                outputs[0]->channel(),
                outputs[0]->height(),
                outputs[0]->width(),
                param.weight()->height(),
                param.weight()->width(),
                param.pad_h,
                param.pad_w,
                param.stride_h,
                param.stride_w,
                param.dilation_h,
                param.dilation_w,
                param.group,
                alpha,
                param.beta,
                this->_ctx->get_compute_stream());
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberDirectConv<AK_HALF>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV> &ctx){
    return SaberUnImplError;
}

template <>
SaberStatus SaberDirectConv<AK_HALF>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {
    return SaberUnImplError;
}

} // namespace saber
} // namespace anakin
