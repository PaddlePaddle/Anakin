
#include "saber/funcs/impl/cuda/saber_conv.h"
#include "saber/funcs/calibrate.h"
namespace anakin {
namespace saber {

template <>
SaberStatus SaberConv2D<NV, AK_FLOAT>::\
    init(const std::vector<Tensor<NV> *>& inputs,
           std::vector<Tensor<NV> *>& outputs,
           ConvParam<NV>& param, Context<NV>& ctx) {
    this->_ctx = &ctx;

    if (param.stride_h == 1 &&
        param.stride_w == 1 &&
        param.weight()->height() == 3 &&
        param.weight()->width() == 3 && param.group == 1) {
        if (param.activation_param.has_active) {
            if (param.activation_param.active == Active_relu) {
                dispatch_func = winograd_conv_relu<float, OpDataType>;
            } else {
                _with_saber_act = true;
                dispatch_func = winograd_conv<float, OpDataType>;
            }
        } else {
            dispatch_func = winograd_conv<float, OpDataType>;
        }
    } else if (param.group == 1) {
        const int K = param.weight()->num();
        if (K % 4 == 0) {
            if (param.bias()->size() > 0 && !param.activation_param.has_active) {
                dispatch_func = direct_conv_bias_Kdivis4<float, OpDataType>;
            } else if (param.bias()->valid_size() > 0 && param.activation_param.active == Active_relu) {
                dispatch_func = direct_conv_bias_relu_Kdivis4<float, OpDataType>;
            } else {
                if (param.activation_param.has_active) {
                    // NOT SUPPORT conv relu fusion
                    _with_saber_act = true;
                }
                dispatch_func = direct_conv_Kdivis4<float, OpDataType>;
            }
        } else {
            if (param.bias()->size() > 0 && !param.activation_param.has_active) {
                dispatch_func = direct_conv_bias_Kindiv4<float, OpDataType>;
            } else if (param.bias()->valid_size() > 0 && param.activation_param.active == Active_relu) {
                dispatch_func = direct_conv_bias_relu_Kindiv4<float, OpDataType>;
            } else {
                if (param.activation_param.has_active) {
                    // NOT SUPPORT conv relu fusion
                    _with_saber_act = true;
                }
                dispatch_func = direct_conv_Kindiv4<float, OpDataType>;
            }
        }
    } else {
        return SaberUnImplError;
    }

    _kernel_height = param.weight()->height();
    _kernel_width = param.weight()->width();
    trans_weights(inputs, outputs, param, ctx);
    cudaDeviceSynchronize();
    if (_with_saber_act) {
        _saber_act = new SaberActivation<NV, AK_FLOAT>;
        _saber_act->init(outputs, outputs, param.activation_param, ctx);
    }
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConv2D<NV, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<NV> *>& inputs,
             std::vector<Tensor<NV> *>& outputs,
             ConvParam<NV>& param) {
    //err code?
    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    const float* bias_data = nullptr;
    if (param.bias()->size() > 0) {
        bias_data = param.bias()->data();
    }

    dispatch_func((const float*)inputs[0]->data(),
                  (float*)outputs[0]->mutable_data(),
                  (const float*)param.weight()->data(),
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
                  _kernel_height,
                  _kernel_width,
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

    if (_with_saber_act) {
        _saber_act->dispatch(outputs, outputs, param.activation_param);
    }
    CUDA_CHECK(cudaGetLastError());
    return SaberSuccess;
}

template <>
SaberStatus SaberConv2D<NV, AK_INT8>::\
    init(const std::vector<Tensor<NV> *>& inputs,
         std::vector<Tensor<NV> *>& outputs,
         ConvParam<NV>& param, Context<NV>& ctx) {
    return SaberInvalidValue;
}

template <>
SaberStatus SaberConv2D<NV, AK_INT8>::\
    dispatch(const std::vector<Tensor<NV> *>& inputs,
             std::vector<Tensor<NV> *>& outputs,
             ConvParam<NV>& param) {

    return SaberInvalidValue;
}

}
}