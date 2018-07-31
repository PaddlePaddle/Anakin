#include "saber/funcs/impl/arm/saber_conv_act.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

template <>
SaberConv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::SaberConv2DAct() {
    _conv_op = new SaberConv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
    _act_op = nullptr;
}

template <>
SaberConv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::~SaberConv2DAct() {

    //printf("flag_relu: %d \n", this->_param->has_active);
  //  printf("~SaberConvAct2D start\n");
    delete _conv_op;
    if(_act_op) {
        delete _act_op;
    }
  //  printf("~SaberConvAct2D end\n");
}

template <>
SaberStatus SaberConv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::create(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvActiveParam<OpTensor> &param, Context<ARM> &ctx) {
    SaberStatus state = _conv_op->create(inputs, outputs, param.conv_param, ctx);
    if (_act_op) {
        state &= _act_op->create(outputs, outputs, param.activation_param, ctx);
    }
    return state;
}

template <>
SaberStatus SaberConv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::init(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvActiveParam<OpTensor> &param, Context<ARM> &ctx) {

    _conv_op->init(inputs, outputs, param.conv_param, ctx);
    if (param.has_active) {
        if (param.activation_param.active == Active_relu) {
            _conv_op->set_activation(param.has_active);
        } else {
            if (_act_op == nullptr) {
                _act_op = new SaberActivation<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
            }
            _act_op->init(outputs, outputs, param.activation_param, ctx);
        }
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberConv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvActiveParam<OpTensor> &param) {
    SaberStatus state = _conv_op->dispatch(inputs, outputs, param.conv_param);
    if (_act_op) {
        state &= _act_op->dispatch(outputs, outputs, param.activation_param);
    }
    return state;
}

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE


