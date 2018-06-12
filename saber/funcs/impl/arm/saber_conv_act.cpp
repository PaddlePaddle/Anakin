#include "saber/funcs/impl/arm/saber_conv_act.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

template <>
SaberConv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::SaberConv2DAct() {
    _conv_op = new SaberConv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}

template <>
SaberConv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::~SaberConv2DAct() {
    delete _conv_op;
}

template <>
SaberStatus SaberConv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::create(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvActiveParam<OpTensor> &param, Context<ARM> &ctx) {
    return _conv_op->create(inputs, outputs, param.conv_param, ctx);
}

template <>
SaberStatus SaberConv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::init(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvActiveParam<OpTensor> &param, Context<ARM> &ctx) {
    if (param.has_active) {
        SABER_CHECK(_conv_op->set_activation(true));
    } else {
        SABER_CHECK(_conv_op->set_activation(false));
    }
    _conv_op->init(inputs, outputs, param.conv_param, ctx);
    return SaberSuccess;
}

template <>
SaberStatus SaberConv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvActiveParam<OpTensor> &param) {
    return _conv_op->dispatch(inputs, outputs, param.conv_param);
}

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE


