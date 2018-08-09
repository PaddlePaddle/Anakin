#include "saber/funcs/impl/arm/saber_deconv_act.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

template <>
SaberDeconv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::SaberDeconv2DAct() {
    _conv_op = new SaberDeconv2D<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}

template <>
SaberDeconv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::~SaberDeconv2DAct() {
    delete _conv_op;
}

template <>
SaberStatus SaberDeconv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::create(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvActiveParam<OpTensor> &param, Context<ARM> &ctx) {
    //LOG(INFO) << "deconv create";
    return _conv_op->create(inputs, outputs, param.conv_param, ctx);
}

template <>
SaberStatus SaberDeconv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::init(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvActiveParam<OpTensor> &param, Context<ARM> &ctx) {
    //LOG(INFO) << "deconv init";
    if (param.has_active) {
        SABER_CHECK(_conv_op->set_activation(true));
    } else {
        SABER_CHECK(_conv_op->set_activation(false));
    }
   // LOG(INFO) << "Deconv act";
    _conv_op->init(inputs, outputs, param.conv_param, ctx);
    return SaberSuccess;
}

template <>
SaberStatus SaberDeconv2DAct<ARM, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    ConvActiveParam<OpTensor> &param) {
    //LOG(INFO) << "Deconv dispatch";
    return _conv_op->dispatch(inputs, outputs, param.conv_param);
}

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE


