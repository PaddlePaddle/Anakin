
#include "saber/funcs/impl/cuda/saber_conv_direct.h"
#include "saber/funcs/calibrate.h"
#include "saber_conv.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberDirectConv<AK_FLOAT>::create(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {
    if (_use_saber_act) {
        _saber_act->create(inputs, outputs, param.activation_param, ctx);
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberDirectConv<AK_FLOAT>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    _use_saber_act = param.activation_param.has_active
                     && !(param.activation_param.active == Active_relu
                          && param.activation_param.negative_slope == 0.f);
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
    const float* weight_ptr = (const float*) param.weight()->data();
    const float* bias_data = nullptr;

    if (param.bias()->size() > 0) {
        bias_data = (const float*)param.bias()->data();
    }

    dispatch_func((const float*) inputs[0]->data(),
                  (float*) outputs[0]->mutable_data(),
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
    ConvParam<NV>& param, Context<NV>& ctx) {
    LOG(INFO) << "conv int8 create"
              << " input tensor dtype: " << (inputs[0]->get_dtype() == AK_FLOAT ? "AK_FLOAT" : "AK_INT8")
              << " output tensor dtype: " << (outputs[0]->get_dtype() == AK_FLOAT ? "AK_FLOAT" : "AK_INT8");
    return SaberSuccess;
}

template <>
SaberStatus SaberDirectConv<AK_INT8>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {
    LOG(INFO) << "conv int8 init"
              << " input tensor dtype: " << (inputs[0]->get_dtype() == AK_FLOAT ? "AK_FLOAT" : "AK_INT8")
              << " output tensor dtype: " << (outputs[0]->get_dtype() == AK_FLOAT ? "AK_FLOAT" : "AK_INT8");
    return SaberSuccess;
}

template <>
SaberStatus SaberDirectConv<AK_INT8>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {

    LOG(INFO) << "conv int8 dispatch"
              << " input tensor dtype: " << (inputs[0]->get_dtype() == AK_FLOAT ? "AK_FLOAT" : "AK_INT8")
              << " output tensor dtype: " << (outputs[0]->get_dtype() == AK_FLOAT ? "AK_FLOAT" : "AK_INT8");
    return SaberSuccess;
}


template <>
SaberStatus SaberDirectConv<AK_HALF>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {
    return SaberUnImplError;
}

template <>
SaberStatus SaberDirectConv<AK_HALF>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {
    return SaberUnImplError;
}

}
}
