
#include "saber/funcs/impl/cuda/vender_conv_pooling.h"
#include "saber/funcs/impl/cuda/cudnn_helper.h"
#include "saber/funcs/calibrate.h"
#include "saber/funcs/impl/cuda/vender_conv.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin {
namespace saber {

// FP32 part
template <>
SaberStatus VenderConv2DPooling<NV, AK_FLOAT>::\
        create(const std::vector<Tensor<NV> *>& inputs,
                std::vector<Tensor<NV> *>& outputs,
                ConvPoolingParam<NV>& param, Context<NV>& ctx) {
    _ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);
    _inner_tensor.reshape(_inner_shape);
    _inner_tensor_v.resize(1);
    _inner_tensor_v[0] = &_inner_tensor;

    _vender_conv.create(inputs, _inner_tensor_v, param.conv_param, ctx);
    _vender_pool.create(_inner_tensor_v, outputs, param.pooling_param, ctx);
    return SaberSuccess;
}

template <>
SaberStatus VenderConv2DPooling<NV, AK_FLOAT>::
        init(const std::vector<Tensor<NV> *>& inputs,
                std::vector<Tensor<NV> *>& outputs,
                ConvPoolingParam<NV>& param, Context<NV>& ctx) {

    _ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);
    _inner_tensor.re_alloc(_inner_shape, AK_FLOAT);

    _inner_tensor_v.resize(1);
    _inner_tensor_v[0] = &_inner_tensor;
    _vender_conv.init(inputs, _inner_tensor_v, param.conv_param, ctx);
    _vender_pool.init(_inner_tensor_v, outputs, param.pooling_param, ctx);
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderConv2DPooling<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ConvPoolingParam<NV>& param) {
    _vender_conv.dispatch(inputs, _inner_tensor_v, param.conv_param);
    _vender_pool.dispatch(_inner_tensor_v, outputs, param.pooling_param);
    return SaberSuccess;
}

template class VenderConv2DPooling<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderConv2DPooling, ConvPoolingParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(VenderConv2DPooling, ConvPoolingParam, NV, AK_INT8);
}
}
