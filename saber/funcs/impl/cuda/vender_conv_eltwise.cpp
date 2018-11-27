
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

    _vender_conv.create(inputs, outputs, param.conv_param, ctx);

    return SaberSuccess;
}

template <>
SaberStatus VenderConvEltwise<NV, AK_FLOAT>::
        init(const std::vector<Tensor<NV> *>& inputs,
                std::vector<Tensor<NV> *>& outputs,
                ConvEltwiseParam<NV>& param, Context<NV>& ctx) {

    if (param.eltwise_param.activation_param.has_active) {
        return SaberUnImplError;
    }
    if (param.conv_param.activation_param.has_active) {
        return SaberUnImplError;
    }

    _vender_conv.init(inputs, outputs, param.conv_param, ctx);
    _vender_conv.set_beta(1.f);
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus VenderConvEltwise<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ConvEltwiseParam<NV>& param) {

    _vender_conv.dispatch(inputs, outputs, param.conv_param);
    return SaberSuccess;
}

template class VenderConvEltwise<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderConvEltwise, ConvEltwiseParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(VenderConvEltwise, ConvEltwiseParam, NV, AK_INT8);
}
}
