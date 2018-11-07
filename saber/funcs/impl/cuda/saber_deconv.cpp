
#include "saber/funcs/impl/cuda/saber_deconv.h"
#include "saber/funcs/impl/cuda/depthwise_deconv.h"
#include "saber/funcs/impl/cuda/sass_deconv.h"
#include "saber/funcs/impl/cuda/vender_deconv.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberDeconv2D<NV, AK_FLOAT>::create(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {

    _impl->create(inputs, outputs, param, ctx);
    return SaberSuccess;
}

template <>
SaberStatus SaberDeconv2D<NV, AK_FLOAT>::init(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;

    _use_k4_s2_p1 = true;
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.weight()->width() == 4);
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.weight()->height() == 4);
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.stride_h == 2);
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.stride_w == 2);
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.pad_h == 1);
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.pad_w == 1);
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.group == 1);
    _use_k4_s2_p1 = _use_k4_s2_p1 && ((inputs[0]->width() % 64 == 0)
                                      || (inputs[0]->width() % 32 == 0 && param.activation_param.has_active));

    int in_channel = inputs[0]->channel();
    int out_channel = outputs[0]->channel();

    if (_use_k4_s2_p1) {
        _impl = new SassDeconv<NV, AK_FLOAT>;
    } else if (param.group == in_channel && param.group == out_channel) {
        _impl = new DepthwiseDeconv<NV, AK_FLOAT>;
    } else {
        _impl = new VenderDeconv2D<NV, AK_FLOAT>;
    }

    _impl->init(inputs, outputs, param, ctx);
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberDeconv2D<NV, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        ConvParam<NV>& param) {
    _impl->dispatch(inputs, outputs, param);
}

template <>
SaberStatus SaberDeconv2D<NV, AK_FLOAT>::trans_weights(Tensor<NV>& target_weights,
        Tensor<NV>& target_bias,
        int in_channel, int out_channel,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dilation_h, int dilation_w,
        int group) {

    if (_use_k4_s2_p1 && _impl != nullptr) {
        (static_cast<SaberDeconv2D<NV, AK_FLOAT>* >(_impl))->trans_weights(
            target_weights, target_bias, in_channel, out_channel,
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group);
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberDeconv2D<NV, AK_HALF>::trans_weights(Tensor<NV>& target_weights,
        Tensor<NV>& target_bias,
        int in_channel, int out_channel,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dilation_h, int dilation_w,
        int group) {

    return SaberUnImplError;
}

template <>
SaberStatus SaberDeconv2D<NV, AK_INT8>::trans_weights(Tensor<NV>& target_weights,
        Tensor<NV>& target_bias,
        int in_channel, int out_channel,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dilation_h, int dilation_w,
        int group) {
    return SaberUnImplError;
}

template class SaberDeconv2D<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberDeconv2D, ConvParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberDeconv2D, ConvParam, NV, AK_INT8);

}
}