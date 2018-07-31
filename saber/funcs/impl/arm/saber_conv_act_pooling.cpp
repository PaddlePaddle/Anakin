#include "saber/funcs/impl/arm/saber_conv_act_pooling.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

template <>
SaberConv2DActPooling<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::SaberConv2DActPooling() {
    _conv_act_op = new SaberConv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
    _pool_op = new SaberPooling<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
    _vtensor_tmp.resize(1);
}

template <>
SaberConv2DActPooling<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::~SaberConv2DActPooling() {
    delete _conv_act_op;
    delete _pool_op;
    _vtensor_tmp.clear();
}

template <>
SaberStatus SaberConv2DActPooling<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::create(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvActivePoolingParam<OpTensor> &param, Context<ARM> &ctx) {

    get_conv_out_tensor(inputs, param);
    if (param.has_activation) {
        ConvActiveParam<OpTensor> conv_act_param(param.conv_param, param.activation_param);
        _conv_act_param = conv_act_param;
    } else {
        ConvActiveParam<OpTensor> conv_act_param(param.conv_param);
        _conv_act_param = conv_act_param;
    }
    SaberStatus state = _conv_act_op->create(inputs, _vtensor_tmp, _conv_act_param, ctx);

    return (SaberStatus)(state & _pool_op->create(_vtensor_tmp, outputs, param.pooling_param, ctx));
}

template <>
SaberStatus SaberConv2DActPooling<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::init(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvActivePoolingParam<OpTensor> &param, Context<ARM> &ctx) {

    if (param.has_activation) {
        ConvActiveParam<OpTensor> conv_act_param(param.conv_param, param.activation_param);
        _conv_act_param = conv_act_param;
    } else {
        ConvActiveParam<OpTensor> conv_act_param(param.conv_param);
        _conv_act_param = conv_act_param;
    }
    get_conv_out_tensor(inputs, param);
    SaberStatus state = _conv_act_op->init(inputs, _vtensor_tmp, _conv_act_param, ctx);
    return (SaberStatus)(state & _pool_op->init(_vtensor_tmp, outputs, param.pooling_param, ctx));
}

template <>
SaberStatus SaberConv2DActPooling<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvActivePoolingParam<OpTensor> &param) {
    SaberStatus state = _conv_act_op->dispatch(inputs, _vtensor_tmp, _conv_act_param);
    return (SaberStatus)(state & _pool_op->dispatch(_vtensor_tmp, outputs, param.pooling_param));
}

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE


