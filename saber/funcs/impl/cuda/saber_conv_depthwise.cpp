#include "saber/funcs/impl/cuda/saber_conv_depthwise.h"
#include "saber/funcs/calibrate.h"
#include "saber/funcs/impl/cuda/saber_activation.h"

namespace anakin {
namespace saber {

template <typename dtype, bool bias_flag, bool relu_flag>
SaberStatus saber_depthwise_conv_act(const dtype* input, dtype* output, \
                                     int num, int cin, int hin, int win, int hout, int wout, \
                                     int kw, int kh, int stride_w, int stride_h, \
                                     int pad_h, int pad_w, const dtype* weights, const dtype* bias, \
                                     cudaStream_t stream);

template <>
SaberStatus SaberDepthWiseConv<AK_FLOAT>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;

    if (param.activation_param.has_active) {
        if (param.activation_param.active != Active_relu) {
            _saber_act = new SaberActivation<NV, AK_FLOAT>;
            _saber_act->init(inputs, outputs, param.activation_param, ctx);
        }
    }

    if (param.activation_param.has_active) {
        if (param.activation_param.active == Active_relu) {
            if (param.bias()->size() > 0) {
                dispatch_func = saber_depthwise_conv_act<float, true, true>;
            } else {
                dispatch_func = saber_depthwise_conv_act<float, false, true>;
            }
        } else {
            if (param.bias()->size() > 0) {
                dispatch_func = saber_depthwise_conv_act<float, true, false>;
            } else {
                dispatch_func = saber_depthwise_conv_act<float, false, false>;
            }
        }
    } else if (param.bias()->size() > 0) {
        dispatch_func = saber_depthwise_conv_act<float, true, false>;
    } else {
        dispatch_func = saber_depthwise_conv_act<float, false, false>;
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberDepthWiseConv<AK_FLOAT>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {

    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    const float* weight_ptr = (const float*) param.weight()->data();
    const float* bias_data = nullptr;

    if (param.bias()->size() > 0) {
        bias_data = (const float*)param.bias()->data();
    }

    dispatch_func((const float*)inputs[0]->data(),
                  (float*)outputs[0]->mutable_data(),
                  inputs[0]->num(), inputs[0]->channel(),
                  inputs[0]->height(), inputs[0]->width(), outputs[0]->height(),
                  outputs[0]->width(), param.weight()->width(), param.weight()->height(), param.stride_w,
                  param.stride_h, param.pad_w, param.pad_h,
                  (const OpDataType*)param.weight()->data(), (const float*)bias_data,
                  this->_ctx->get_compute_stream());

    if (this->_saber_act != nullptr) {
        this->_saber_act->dispatch(outputs, outputs, param.activation_param);
    }

    CUDA_CHECK(cudaGetLastError());
    return SaberSuccess;
}

template <>
SaberStatus SaberDepthWiseConv<AK_INT8>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {
    return SaberUnImplError;
}

template <>
SaberStatus SaberDepthWiseConv<AK_INT8>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {
    return SaberUnImplError;
}


template <>
SaberStatus SaberDepthWiseConv<AK_HALF>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {
    return SaberUnImplError;
}

template <>
SaberStatus SaberDepthWiseConv<AK_HALF>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param) {
    return SaberUnImplError;
}

}
}
