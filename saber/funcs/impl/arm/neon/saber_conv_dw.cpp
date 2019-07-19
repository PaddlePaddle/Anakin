#include "saber/funcs/impl/arm/saber_conv_dw.h"
#include "saber/funcs/impl/arm/neon/impl/conv_arm_impl.h"
#include "saber/funcs/type_trans.h"
namespace anakin{
namespace saber {


/****************************************** Dw Conv Precision Is Float ******************************************/
template<>
SaberStatus SaberDepthWiseConv<AK_FLOAT>::create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    int ks = param.weight()->width();
    int win = inputs[0]->width();
    int wout = outputs[0]->width();
    //! select dw conv kernel
    if (ks == 3){
        //! 3x3 dw conv
        //printf("invoke 3x3dw\n");
        _impl = conv_depthwise_3x3;
    } else if (ks == 5){
        //! 5x5 dw conv
        this->_ctx->workspace_extend(Shape({1, 1, 1, win + wout}));
        //printf("invoke 5x5dw\n");
        _impl = conv_depthwise_5x5;
    } else {
        LOG(ERROR) << "this type dw conv not impl";
        return SaberUnImplError;
    }
    return SaberSuccess;
}

template<>
SaberStatus SaberDepthWiseConv<AK_FLOAT>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberDepthWiseConv<AK_FLOAT>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        ConvParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    const float* din = static_cast<const float*>(inputs[0]->data());
    float* dout = static_cast<float*>(outputs[0]->mutable_data());
    const float* weights = static_cast<const float*>(param.weight()->data());
    const float* bias = static_cast<const float*>(param.bias()->data());

    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();
    _impl(din, dout, num, chin, hout, wout, chin, hin, win, weights, bias, param, this->_ctx);
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "DepthWiseConv fp32: " << this->_op_name.c_str() << " : time: " << ts;
#endif
    return SaberSuccess;
}

/****************************************** Dw Conv Precision Is INT8 ******************************************/
template<>
SaberStatus SaberDepthWiseConv<AK_INT8>::create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    int ks = param.weight()->width();
    int stride = param.stride_w;
    int win = inputs[0]->width();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();
    _w_scale = param.weight()->get_scale();

    //! select dw conv kernel
    if (ks == 3){
        //! init int32 tmp out
        _tmp_out.set_dtype(AK_INT32);
        _tmp_out.reshape(outputs[0]->valid_shape());
        //! 3x3 dw int8 conv
        _impl_int8 = conv_depthwise_3x3_int8;
    } else if (ks == 5){
        //! update weights scale
        DataType out_type = outputs[0]->get_dtype();
        if (out_type == AK_FLOAT || out_type == AK_INT8) {
            CHECK_EQ(_w_scale.size(), chout) << "weights scale size must be chout";
            float input_scale = inputs[0]->get_scale()[0];
            for (auto& ws : _w_scale)
            {
                ws *= input_scale;
                if (out_type == AK_INT8) {
                   ws /= outputs[0]->get_scale()[0];
                }
            }
        }

        const int wout_round = ((wout + 7) / 8) * 8;
        const int win_round = wout_round * stride + 5 - 1;
        const int hout_round = ((hout + 2) / 3) * 3;
        const int hin_round = hout_round * stride + 5 - 1;

        const int tmp_size_out = wout_round * hout_round;
        const int tmp_size_in = win_round * hin_round;
        const int tmp_size_io_bytes = tmp_size_in + tmp_size_out * sizeof(int);
        const int tmp_row_io_bytes =  win_round + wout_round * sizeof(int);
        const int tmp_size_io_float = (tmp_size_io_bytes + sizeof(float)-1) / sizeof(float);
        const int tmp_row_io_float = (tmp_row_io_bytes + sizeof(float)-1) / sizeof(float);
        _ctx->workspace_extend(Shape({1, 1, 1, _ctx->get_threads() * tmp_size_io_float + tmp_row_io_float}));
        //! 5x5 dw int8 conv
        _impl_int8 = conv_depthwise_5x5;
    } else {
        LOG(ERROR) << "this type dw int8 conv not impl";
        return SaberUnImplError;
    }
    return SaberSuccess;
}

template<>
SaberStatus SaberDepthWiseConv<AK_INT8>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberDepthWiseConv<AK_INT8>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        ConvParam<ARM> &param) {
#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    const int8_t* din = static_cast<const int8_t*>(inputs[0]->data());
    int32_t* dout = nullptr;
    const int8_t* weights = static_cast<const int8_t*>(param.weight()->data());
    const int32_t* bias = static_cast<const int32_t*>(param.bias()->data());

    int ks = param.weight()->width();
    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();
    int chout = outputs[0]->channel();

    if (ks == 3 && outputs[0]->get_dtype() != AK_INT32){
        dout = _tmp_out.mutable_data();
    } else if (ks == 5 || (ks == 3 && outputs[0]->get_dtype() == AK_INT32)){
        dout = outputs[0]->mutable_data();
    } else {
        LOG(ERROR) <<  "this type dw int8 conv not impl";
        return SaberUnImplError;
    }
    _impl_int8(din, dout, num, chout, hout, wout, chin, hin, win, weights, bias, param, \
        this->_ctx, outputs[0]->get_dtype(), _w_scale.data());

    if (ks == 3){
        if (outputs[0]->get_dtype() == AK_INT8) {
            trans_tensor_dtype<ARM, AK_INT32, AK_INT8>(_tmp_out, *outputs[0], inputs[0]->get_scale()[0], \
                outputs[0]->get_scale()[0], _w_scale);
        } else if (outputs[0]->get_dtype() == AK_FLOAT) {
            trans_tensor_dtype<ARM, AK_INT32, AK_FLOAT>(_tmp_out, *outputs[0], inputs[0]->get_scale()[0], 1.f, _w_scale);
        } else if (outputs[0]->get_dtype() != AK_INT32) {
            LOG(ERROR) << "unsupported precision type!!";
            return SaberInvalidValue;
        }
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "DepthWiseConv int8: " << this->_op_name.c_str() << " : time: " << ts;
#endif
    return SaberSuccess;
}

}
} // namespace anakin
