#include "saber/funcs/impl/arm/saber_conv_direct.h"
#include "saber/funcs/impl/arm/neon/impl/conv_arm_impl.h"
#include "saber/funcs/impl/arm/neon/impl/conv_block_utils.h"
namespace anakin{
namespace saber {

/****************************************** Direct Conv Precision Is Float ******************************************/
template<>
SaberStatus SaberDirectConv<AK_FLOAT>::create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    int ks = param.weight()->width();
    int stride = param.stride_w;
    int win = inputs[0]->width();
    int chin = inputs[0]->channel();
    int wout = outputs[0]->width();
    int chout = outputs[0]->channel();
    //! select dw conv kernel
    if (ks == 3 && stride == 1){
        //! 3x3s1 direct conv
        //printf("invoke 3x3s1 direct conv\n");
        _impl = conv_3x3s1_direct_fp32;
         //! transform weights
        const int cblock = 4;
        int cround = (chout + cblock - 1) / cblock * cblock;
        _weights_trans.reshape(Shape({cround, chin, ks, ks}));
        float *dwout = static_cast<float *>(_weights_trans.mutable_data());
        const float *dwin = static_cast<const float *>(param.weight()->data());
        conv_trans_weights_numc(dwin, dwout, chout, chin, cblock, ks * ks);
         _is_trans_weights = true;
    } else if (ks == 3 && stride == 2){
        //! 3x3s2 direct conv
        //printf("invoke 3x3s2 direct conv\n");
        _impl = conv_3x3s2_direct_fp32;
        //! transform weights
        const int cblock = 4;
        int cround = (chout + cblock - 1) / cblock * cblock;
        _weights_trans.reshape(Shape({cround, chin, ks, ks}));
        float *dwout = static_cast<float *>(_weights_trans.mutable_data());
        const float *dwin = static_cast<const float *>(param.weight()->data());
        conv_trans_weights_numc(dwin, dwout, chout, chin, cblock, ks * ks);
         _is_trans_weights = true;
    } else {
        LOG(ERROR) << "this type direct conv not impl";
        return SaberUnImplError;

    }
    return SaberSuccess;
}

template<>
SaberStatus SaberDirectConv<AK_FLOAT>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberDirectConv<AK_FLOAT>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        ConvParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    const float* din = static_cast<const float*>(inputs[0]->data());
    float* dout = static_cast<float*>(outputs[0]->mutable_data());
    const float* weights = nullptr;
    if (_is_trans_weights == true){
        weights = static_cast<const float*>(_weights_trans.data());
    } else {
        weights = static_cast<const float*>(param.weight()->data());
    }
    const float* bias = static_cast<const float*>(param.bias()->data());

    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();
    int chout = outputs[0]->channel();
    _impl(din, dout, num, chout, hout, wout, chin, hin, win, weights, bias, param, this->_ctx);

#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "DirectConv fp32: " << this->_op_name.c_str() << " : time: " << ts;
#endif
    return SaberSuccess;
}

/****************************************** Direct Conv Precision Is INT8 ******************************************/
template<>
SaberStatus SaberDirectConv<AK_INT8>::create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    int ks = param.weight()->width();
    int stride = param.stride_w;
    int win = inputs[0]->width();
    int chin = inputs[0]->channel();
    int wout = outputs[0]->width();
    int chout = outputs[0]->channel();
    _w_scale = param.weight()->get_scale();
    //! update weights scale
    const DataType out_type = outputs[0]->get_dtype();
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
    //! select dw conv kernel
    if (ks == 3 && stride == 1){
        //! 3x3s1 direct int8 conv
        _impl_int8 = conv_3x3s1_direct_int8;
        //! transform weights
        int hout_c_block = 4;
        int inpad = 4;
        Shape shape_out({((chout + hout_c_block - 1 ) / hout_c_block) * hout_c_block, chin, 3, 3});
        _weights_trans.re_alloc(shape_out, AK_INT8);
        conv_trans_weights_numc(static_cast<const signed char*>(param.weight()->data()), \
            static_cast<signed char*>(_weights_trans.mutable_data()), chout, chin, hout_c_block, 9);
        int wout_round = ((wout + 3) / 4) * 4;
        int win_round = wout_round * param.stride_w + inpad;
        int row_out = 2;
        int row_in = 4;
        int tmp_size_out = wout_round * row_out * hout_c_block;
        int in_len = win_round * chin;
        int tmp_size_in = row_in * in_len;
        _ctx->workspace_extend(Shape({1, 1, 1, _ctx->get_threads() * tmp_size_out + \
            (tmp_size_in + 3) / 4 * 4 + wout_round + win_round}));
        _is_trans_weights = true;

    } else if (ks == 3 && stride == 2){
        //! 3x3s2 direct int8 conv
        _impl_int8 = conv_3x3s2_direct_int8;
        //! transform weights
        int cblock = conv_3x3s2_direct_int8_c_num();
        int cround = (chout + cblock - 1) / cblock * cblock;
        _weights_trans.re_alloc(Shape({cround, chin, ks, ks}), AK_INT8);
        conv_trans_weights_numc(static_cast<const int8_t*>(param.weight()->data()), \
            static_cast<int8_t*>(_weights_trans.mutable_data()), chout, chin, cblock, 9);
         _is_trans_weights = true;
    } else {
        LOG(ERROR) << "this type direct int8 conv not impl";
        return SaberUnImplError;

    }
    return SaberSuccess;
}

template<>
SaberStatus SaberDirectConv<AK_INT8>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberDirectConv<AK_INT8>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        ConvParam<ARM> &param) {
#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    const int8_t* din = static_cast<const int8_t*>(inputs[0]->data());
    int32_t* dout = static_cast<int32_t*>(outputs[0]->mutable_data());
    const int8_t* weights = nullptr;
    if (_is_trans_weights == true){
        weights = static_cast<const int8_t*>(_weights_trans.data());
    } else {
        weights = static_cast<const int8_t*>(param.weight()->data());
    }
    const int32_t* bias = static_cast<const int32_t*>(param.bias()->data());

    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();
    int chout = outputs[0]->channel();
    _impl_int8(din, dout, num, chout, hout, wout, chin, hin, win, weights, bias, param, \
        this->_ctx, outputs[0]->get_dtype(), _w_scale.data());
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "DirectConv int8: " << this->_op_name.c_str() << " : time: " << ts;
#endif
    return SaberSuccess;
}
}
} // namespace anakin
