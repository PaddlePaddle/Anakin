
#include "saber/funcs/impl/x86/saber_conv_pooling.h"
#include "saber/funcs/calibrate.h"
#include "saber/funcs/impl/x86/saber_conv.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/funcs_utils.h"
#include "saber/funcs/impl/x86/kernel/jit_conv_pooling_normal.h"

namespace anakin {
namespace saber {

// FP32 part
template <>
SaberStatus SaberConv2DPooling<X86, AK_FLOAT>::\
        create(const std::vector<Tensor<X86> *>& inputs,
            std::vector<Tensor<X86> *>& outputs,
            ConvPoolingParam<X86>& param, Context<X86>& ctx) {

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
SaberStatus SaberConv2DPooling<X86, AK_FLOAT>::
        init(const std::vector<Tensor<X86> *>& inputs,
            std::vector<Tensor<X86> *>& outputs,
            ConvPoolingParam<X86>& param, Context<X86>& ctx) {

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
SaberStatus SaberConv2DPooling<X86, AK_FLOAT>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvPoolingParam<X86>& param) {
    _vender_conv.dispatch(inputs, _inner_tensor_v, param.conv_param);
    _vender_pool.dispatch(_inner_tensor_v, outputs, param.pooling_param);
    return SaberSuccess;
}

template class SaberConv2DPooling<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConv2DPooling, ConvPoolingParam, X86, AK_HALF);

template <>
SaberStatus SaberConv2DPooling<X86, AK_INT8>::\
create(const std::vector<Tensor<X86> *>& inputs,
       std::vector<Tensor<X86> *>& outputs,
       ConvPoolingParam<X86>& param, Context<X86>& ctx) {
    SaberStatus ret = SaberUnImplError;

    return ret;
}

template <>
SaberStatus SaberConv2DPooling<X86, AK_INT8>::\
init(const std::vector<Tensor<X86> *>& inputs,
     std::vector<Tensor<X86> *>& outputs,
     ConvPoolingParam<X86>& param, Context<X86>& ctx) {
    SaberStatus ret = SaberSuccess;
    return ret;
}

template <>
SaberStatus SaberConv2DPooling<X86, AK_INT8>::\
dispatch(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvPoolingParam<X86>& param) {
    SaberStatus ret = SaberSuccess;

    return ret;
}


}
}
