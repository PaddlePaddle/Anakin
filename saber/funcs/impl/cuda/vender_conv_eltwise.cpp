
#include "saber/funcs/impl/cuda/vender_conv_eltwise.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
#include "saber/funcs/calibrate.h"
#include "saber/funcs/impl/cuda/vender_conv.h"
#include "saber/core/tensor_op.h"
namespace anakin {
namespace saber {

// FP32 part
template <>
SaberStatus VenderConvEltwise<NV, AK_FLOAT>::\
create(const std::vector<Tensor<NV> *>& inputs,
       std::vector<Tensor<NV> *>& outputs,
       ConvEltwiseParam<NV>& param, Context<NV>& ctx) {

    _ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);

    if (_use_vender_eltwise) {
        SABER_CHECK(_conv.create(inputs, outputs, param.conv_param, ctx));
    } else {
        _inner_tensor.reshape(_inner_shape);
        _inner_tensor_v.resize(2);
        _inner_tensor_v[0] = &_inner_tensor;
        _conv.create(inputs, _inner_tensor_v, param.conv_param, ctx);
        _eltwise.create(_inner_tensor_v, outputs, param.eltwise_param, ctx);
    }

    return SaberSuccess;
}

template <>
SaberStatus VenderConvEltwise<NV, AK_FLOAT>::
init(const std::vector<Tensor<NV> *>& inputs,
     std::vector<Tensor<NV> *>& outputs,
     ConvEltwiseParam<NV>& param, Context<NV>& ctx) {

    _ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);
    _use_vender_eltwise = true;
    _use_vender_eltwise = _use_vender_eltwise
                          && (!param.conv_param.activation_param.has_active);
    _use_vender_eltwise = _use_vender_eltwise
                          && (!param.eltwise_param.activation_param.has_active);
    _use_vender_eltwise = _use_vender_eltwise
                          && param.eltwise_param.operation == Eltwise_sum;
    _use_vender_eltwise = _use_vender_eltwise &&
                          _inner_shape == outputs[0]->valid_shape();

    if (_use_vender_eltwise) {
        SABER_CHECK(_conv.init(inputs, outputs, param.conv_param, ctx));
        _conv.set_beta(1.f);
    } else {
        _inner_tensor.re_alloc(_inner_shape, AK_FLOAT);
        _inner_tensor_v.resize(2);
        _inner_tensor_v[0] = &_inner_tensor;
        _conv.init(inputs, _inner_tensor_v, param.conv_param, ctx);
        _eltwise.init(_inner_tensor_v, outputs, param.eltwise_param, ctx);
    }

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderConvEltwise<NV, AK_FLOAT>::dispatch(
    const std::vector<Tensor<NV>*>& inputs,
    std::vector<Tensor<NV>*>& outputs,
    ConvEltwiseParam<NV>& param) {
    if (_use_vender_eltwise) {
        SABER_CHECK(_conv.dispatch(inputs, outputs, param.conv_param));
    } else {
        _conv.dispatch(inputs, _inner_tensor_v, param.conv_param);
        _inner_tensor_v[1] = outputs[0];
        _eltwise.dispatch(_inner_tensor_v, outputs, param.eltwise_param);
    }

    return SaberSuccess;
}

template class VenderConvEltwise<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderConvEltwise, ConvEltwiseParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(VenderConvEltwise, ConvEltwiseParam, NV, AK_INT8);
}
}
