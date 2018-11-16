
#include "saber/funcs/impl/x86/saber_conv.h"
#include "saber/funcs/impl/x86/saber_eltwise.h"
#include "saber/funcs/impl/x86/saber_conv_eltwise.h"
#include "saber/funcs/calibrate.h"
#include "saber_conv_eltwise.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberConvEltwise<X86, AK_FLOAT>::\
        create(const std::vector<Tensor<X86> *>& inputs,
            std::vector<Tensor<X86> *>& outputs,
            ConvEltwiseParam<X86>& param, Context<X86>& ctx) {

    _ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);

    return SaberSuccess;
}

template <>
SaberStatus SaberConvEltwise<X86, AK_FLOAT>::
    init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param, Context<X86>& ctx) {

    _ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);
    _kernel_height = param.conv_param.weight()->height();
    _kernel_width = param.conv_param.weight()->width();

    {
        _inner_tensor.re_alloc(_inner_shape, AK_FLOAT);
        _inner_tensor_v.resize(2);
        _inner_tensor_v[0] = &_inner_tensor;
        _conv.init(inputs, _inner_tensor_v, param.conv_param, ctx);
        _eltwise.init(_inner_tensor_v, outputs, param.eltwise_param, ctx);
    }
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConvEltwise<X86, AK_FLOAT>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param) {

    const float* bias_data;
    if (param.conv_param.bias()->size() > 0) {
        bias_data = (const float*)param.conv_param.bias()->data();
    } else {
        bias_data = nullptr;
    }
    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chout = outputs[0]->channel();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int in_stride = chin * win * hin;
    int out_stride = chout * wout * hout;
    {
        _conv.dispatch(inputs, _inner_tensor_v, param.conv_param);
        _inner_tensor_v[1] = outputs[0];
        _eltwise.dispatch(_inner_tensor_v, outputs, param.eltwise_param);
    }
    return SaberSuccess;
}
template <>
SaberStatus SaberConvEltwise<X86, AK_FLOAT>::trans_weights(
        Tensor<X86> &target_weights, Tensor<X86> &target_bias,
        int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    return SaberSuccess;
}
template <>
SaberStatus SaberConvEltwise<X86, AK_INT8>::trans_weights(
        Tensor<X86> &target_weights, Tensor<X86> &target_bias,
        int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    return SaberSuccess;
}
template <>
SaberStatus SaberConvEltwise<X86, AK_HALF>::trans_weights(
        Tensor<X86> &target_weights, Tensor<X86> &target_bias,
        int pad_h, int pad_w, int dilation_h, int dilation_w,
        int stride_h, int stride_w, int group) {
    return SaberSuccess;
}

template class SaberConvEltwise<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConvEltwise, ConvEltwiseParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConvEltwise, ConvEltwiseParam, X86, AK_INT8);
}
}
