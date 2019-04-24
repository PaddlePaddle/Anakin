#include "saber/funcs/impl/arm/saber_conv_pooling.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin{

namespace saber{
template <>
SaberStatus SaberConv2DPooling<ARM, AK_FLOAT>::create(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        ConvPoolingParam<ARM> &param, Context<ARM>& ctx) {
    this->_ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);
    _tensor_tmp.reshape(_inner_shape);

    _vtensor_tmp.resize(1);
    _vtensor_tmp[0] = &_tensor_tmp;
    SaberStatus state = _conv_func.create(inputs, _vtensor_tmp, param.conv_param, ctx);

    if (state != SaberSuccess) {
        return state;
    }
    return _pool_func.create(_vtensor_tmp, outputs, param.pooling_param, ctx);
}
template <>
SaberStatus SaberConv2DPooling<ARM, AK_FLOAT>::init(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        ConvPoolingParam<ARM> &param, Context<ARM>& ctx) {
    this->_ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);
    _tensor_tmp.re_alloc(_inner_shape, AK_FLOAT);

    _vtensor_tmp.resize(1);
    _vtensor_tmp[0] = &_tensor_tmp;
    SaberStatus state = _conv_func.init(inputs, _vtensor_tmp, param.conv_param, ctx);

    if (state != SaberSuccess) {
        return state;
    }
    return _pool_func.init(_vtensor_tmp, outputs, param.pooling_param, ctx);
}
template <>
SaberStatus SaberConv2DPooling<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        ConvPoolingParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    SaberStatus state = _conv_func.dispatch(inputs, _vtensor_tmp, param.conv_param);
    if (state != SaberSuccess) {
        return state;
    }
    _pool_func.dispatch(_vtensor_tmp, outputs, param.pooling_param);

#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "convPooling: " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("convPooling", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberConv2DPooling, ConvPoolingParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConv2DPooling, ConvPoolingParam, ARM, AK_INT8);

} //namespace anakin

} //namespace anakin
