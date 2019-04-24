#include "saber/funcs/type_trans.h"
#include "saber/funcs/impl/arm/saber_conv.h"
#include "saber/funcs/impl/arm/saber_conv_dw.h"
#include "saber/funcs/impl/arm/saber_conv_gemmlike.h"
#include "saber/funcs/impl/arm/saber_conv_direct.h"
#include "saber/funcs/impl/arm/saber_conv_winograd.h"

namespace anakin{
namespace saber {

/****************************************** Conv Precision Is Float ******************************************/
template<>
SaberStatus SaberConv2D<ARM, AK_FLOAT>::create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            ConvParam<ARM>& param, Context<ARM> &ctx){
    this->_ctx = &ctx;
    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chin = inputs[0]->channel();
    int num = inputs[0]->num();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();
    int kw = param.weight()->width();
    int kh = param.weight()->height();
    int pad = param.pad_w;
    int stride = param.stride_w;

#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
    LOG(INFO) << "conv fp32 param: " << " img_num = " << num << " in_channels = " << chin \
    << " img_h = " << hin << " img_w = " << win << " group = " << param.group \
    << " pad_width = " <<  param.pad_w << " pad_height = " <<  param.pad_h << " stride_width = " \
    << param.stride_w << " stride_height = " << param.stride_h << " dilation_w = " << param.dilation_w \
    << " dilation_h = " << param.dilation_h << " kernel_w = " << kw << " kernel_h = " \
    << kh << " out_channels = " << chout;
#endif

    CHECK_EQ(chin % param.group, 0) << "ERROR: input channel or group size error";
    CHECK_EQ(chout % param.group, 0) << "ERROR: output channel or group size error";

    bool kps_equal = (param.pad_w == param.pad_h) && (param.stride_w == param.stride_h) && (kw == kh);
    bool no_dilation = (param.dilation_h == 1) && (param.dilation_w == 1);
    bool flag_dw_3x3 = (kw == 3 && (pad == 0 || pad == 1) && (stride == 1 || stride == 2));
    bool flag_dw_5x5 = (kw == 5 && stride == 1) || (kw == 5 && stride == 2 && pad == 2);
    bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

    //! select conv impl
    if (param.group == chin && chin == chout && kps_equal && no_dilation && flag_dw){
            //! dw conv impl
            _impl = new SaberDepthWiseConv<AK_FLOAT>;
    } else if (param.group == 1 && kw == 3 && stride == 1 && kps_equal && no_dilation){
        if (chin >= 32 && chout >= 32 && hout > 16 && wout > 16){
            //! winograd conv impl
            _impl = new SaberWinogradConv<AK_FLOAT>;
        } else {
            //! direct conv impl
            _impl = new SaberDirectConv<AK_FLOAT>;
        }
    } else if (param.group == 1 && kw == 3 && stride == 2 && kps_equal && no_dilation){
        //! direct conv impl
        _impl = new SaberDirectConv<AK_FLOAT>;
    } else {
        //! anything else fall to IM2COL + GEMM
        _impl = new SaberGemmLikeConv<AK_FLOAT>;
    }

    return this->_impl->create(inputs, outputs, param, ctx);
}

template<>
SaberStatus SaberConv2D<ARM, AK_FLOAT>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    if (param.activation_param.has_active)
    {
        if (param.activation_param.active != Active_relu || fabs(param.activation_param.negative_slope) > 1e-6f)
        {
            _saber_act = new SaberActivation<ARM, AK_FLOAT>;
            _saber_act->init(inputs, outputs, param.activation_param, ctx);
        }
    }
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConv2D<ARM, AK_FLOAT>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        ConvParam<ARM> &param) {

#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    if (_impl != nullptr) {
        _impl->dispatch(inputs, outputs, param);
    } else {
        return SaberUnImplError;
    }

    if (this->_saber_act != nullptr){
        this->_saber_act->dispatch(outputs, outputs, param.activation_param);
    }
#ifdef ENABLE_OP_TIMER
    int num = inputs[0]->num();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();
    int chin = inputs[0]->channel();
    int kw = param.weight()->width();
    int kh = param.weight()->height();
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    float op_macs = 2.f * kw * kh * num * chout * wout * hout * chin / param.group;
    LOG(INFO) << "Convlution fp32: " << this->_op_name.c_str() << ", time: " << ts << \
        ", GOPs: " << 1e-9f * op_macs << ", GOPS: " << 0.000001 * op_macs / ts;
    GOPS ops;
    //fixme
    ops.ops = op_macs;
    ops.ts = ts;
    OpTimer::add_timer("Conv", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}

/****************************************** Conv Precision Is Int8 ******************************************/
template<>
SaberStatus SaberConv2D<ARM, AK_INT8>::create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            ConvParam<ARM>& param, Context<ARM> &ctx){

    this->_ctx = &ctx;
    Shape shape_in = inputs[0]->valid_shape();
    Shape shape_out = outputs[0]->valid_shape();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chin = inputs[0]->channel();
    int num = inputs[0]->num();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();
    int kw = param.weight()->width();
    int kh = param.weight()->height();
    int pad = param.pad_w;
    int stride = param.stride_w;

#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
    LOG(INFO) << "conv int8 param: " << " img_num = " << num << " in_channels = " << chin \
    << " img_h = " << hin << " img_w = " << win << " group = " << param.group \
    << " pad_width = " <<  param.pad_w << " pad_height = " <<  param.pad_h << " stride_width = " \
    << param.stride_w << " stride_height = " << param.stride_h << " dilation_w = " << param.dilation_w \
    << " dilation_h = " << param.dilation_h << " kernel_w = " << kw << " kernel_h = " \
    << kh << " out_channels = " << chout;
#endif

    CHECK_EQ(chin % param.group, 0) << "ERROR: input channel or group size error";
    CHECK_EQ(chout % param.group, 0) << "ERROR: output channel or group size error";

    bool kps_equal = (param.pad_w == param.pad_h) && (param.stride_w == param.stride_h) && (kw == kh);
    bool no_dilation = (param.dilation_h == 1) && (param.dilation_w == 1);
    bool flag_dw_3x3 = (kw == 3) && (pad == 1) && (stride == 1 || stride == 2);
    bool flag_dw_5x5 = (kw == 5 && stride == 1 && pad == 2);
    bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

    //! checkout inputs
    if (inputs[0]->get_scale().size() < 1) {
        LOG(ERROR) << "ERROR: CONV INT8, inputs must have scale";
        return SaberInvalidValue;
    }

    //! select conv impl
    if (param.group == chin && chin == chout && kps_equal && no_dilation && flag_dw){
        //! dw conv impl
        _impl = new SaberDepthWiseConv<AK_INT8>;
    } else if (param.group == 1 && kw == 3 && (stride == 1 || stride == 2 ) && kps_equal && no_dilation){
        _impl = new SaberDirectConv<AK_INT8>;
    } else {
        //! anything else fall to IM2COL + GEMM
        _impl = new SaberGemmLikeConv<AK_INT8>;
    }

    return this->_impl->create(inputs, outputs, param, ctx);
}

template<>
SaberStatus SaberConv2D<ARM, AK_INT8>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;

    //! trans to int8 weights
    if (trans_weights_dtype(*param.mutable_weight(), AK_INT8, 127.f, CONV_TYPE, param.group) != SaberSuccess) {
        LOG(ERROR) << "ERROR: Conv trans weights to int8 failed";
        return SaberInvalidValue;
    }

    //! trans bias to int32 for all implementation
    if (param.bias()->size() > 0) {
        trans_fp32_bias_to_int32(*param.mutable_bias(), *param.mutable_bias(), inputs[0]->get_scale()[0], param.weight()->get_scale());
    }

    if (param.activation_param.has_active)
    {
        if (param.activation_param.active != Active_relu || param.activation_param.negative_slope > 1e-6f)
        {
            outputs[0]->set_dtype(AK_FLOAT);
            _saber_act = new SaberActivation<ARM, AK_FLOAT>;
            _saber_act->init(inputs, outputs, param.activation_param, ctx);
        }
    }
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConv2D<ARM, AK_INT8>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        ConvParam<ARM> &param) {

#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    if (inputs[0]->get_dtype() != AK_INT8) {
        _tmp_in.set_dtype(AK_INT8);
        _tmp_in.reshape(inputs[0]->valid_shape());
        trans_tensor_dtype<ARM, AK_FLOAT, AK_INT8>(*inputs[0], _tmp_in, inputs[0]->get_scale()[0], 1.f, {1.f});
        _vin_tensor.push_back(&_tmp_in);
        _tmp_in.set_scale(inputs[0]->get_scale());
    } else {
        _vin_tensor.push_back(inputs[0]);
    }

    if (_impl != nullptr) {
        _impl->dispatch(_vin_tensor, outputs, param);
    } else {
        return SaberUnImplError;
    }

    if (this->_saber_act != nullptr){
        this->_saber_act->dispatch(outputs, outputs, param.activation_param);
    }
    return SaberSuccess;
#ifdef ENABLE_OP_TIMER
    int num = inputs[0]->num();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();
    int chin = inputs[0]->channel();
    int kw = param.weight()->width();
    int kh = param.weight()->height();
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    float op_macs = 2.f * kw * kh * num * chout * wout * hout * chin / param.group;
    LOG(INFO) << "Convlution int8: " << this->_op_name.c_str() << ", time: " << ts << \
        ", GOPs: " << 1e-9f * op_macs << ", GOPS: " << 0.000001 * op_macs / ts;
    GOPS ops;
    //fixme
    ops.ops = op_macs;
    ops.ts = ts;
    OpTimer::add_timer("Conv", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberConv2D, ConvParam, ARM, AK_HALF);
}
} // namespace anakin
