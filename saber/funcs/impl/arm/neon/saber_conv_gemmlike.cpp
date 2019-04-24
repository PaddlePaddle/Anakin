#include "saber/funcs/impl/arm/saber_conv_gemmlike.h"
#include "saber/funcs/impl/arm/neon/impl/conv_arm_impl.h"
#include "saber/funcs/impl/arm/neon/impl/sgemm_prepacked.h"
#include "saber/funcs/impl/arm/neon/impl/gemm_prepacked_int8.h"
namespace anakin{
namespace saber {

/****************************************** Gemmlike Conv Precision Is Float ******************************************/
template<>
SaberStatus SaberGemmLikeConv<AK_FLOAT>::create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    int kw = param.weight()->width();
    int kh = param.weight()->height();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();
    int chin = inputs[0]->channel();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int m = chout / param.group;
    int k = chin * kh * kw / param.group;
    int n = hout * wout;
    bool kps_equal = (param.pad_w == param.pad_h) && (param.stride_w == param.stride_h) && (kw == kh);
    bool ks_equal = (param.stride_w == param.stride_h) && (kw == kh);
    //! select conv gemmlike kernel
    if (kw == 1 && param.stride_w == 1 && param.pad_w == 0 && kps_equal){
        //! 1x1s1p0 gemmlike conv
        _impl = conv1x1s1_gemm;
    } else {
        //! otherwise case
        if (kw == 3 && param.stride_w == 1 && n > 1 && ks_equal){
            _idx_data.reshape(Shape({1, 1, 1, n * kh * kw}));
            int* idx_out = (int*)_idx_data.mutable_data();
            for (int i = 0; i < hout; ++i) {
                for (int j = 0; j < wout; ++j) {
                    compute_offset(idx_out, i, j, kh, kw, hin, win, param.pad_h, \
                        param.pad_w, param.dilation_h, param.dilation_w);
                    idx_out += kh * kw;
                }
            }
        }
        //! im2col gemmlike conv
         _impl = conv_im2col_gemm;
        this->_ctx->workspace_extend(Shape({1, 1, 1, k * n}));
    }

    if (n > 1) {
        int hblock = get_hblock(this->_ctx->get_arch());
        int m_roundup = hblock * ((m + hblock - 1) / hblock);
        int group_size_round_up = ((m_roundup * k + 15) / 16) * 16;
        float* w_trans_ptr = nullptr;
        _weights_trans.reshape(Shape({1, 1, 1, group_size_round_up * param.group}));
        w_trans_ptr = static_cast<float*>(_weights_trans.mutable_data());

        for (int g = 0; g < param.group; ++g) {
            const float* weights_group = static_cast<const float*>(param.weight()->data()) + g * m * k;
            float* weights_trans_ptr = w_trans_ptr + g * group_size_round_up;
            prepackA(weights_trans_ptr, weights_group, k, 0, m, 0, k, false, this->_ctx);
        }
         _is_trans_weights = true;
    }
    return SaberSuccess;
}

template<>
SaberStatus SaberGemmLikeConv<AK_FLOAT>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberGemmLikeConv<AK_FLOAT>::dispatch(
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
    const int* idx_data = static_cast<const int*>(_idx_data.data());

    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int chout = outputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();
    // printf("invoke gemm conv\n");
    _impl(din, dout, num, chout, hout, wout, chin, hin, win, weights, bias, param, this->_ctx, idx_data);
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "GemmLikeConv fp32: " << this->_op_name.c_str() << " : time: " << ts;
#endif
    return SaberSuccess;
}

/****************************************** Gemmlike Conv Precision Is Int8 ******************************************/

template<>
SaberStatus SaberGemmLikeConv<AK_INT8>::create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    int kw = param.weight()->width();
    int kh = param.weight()->height();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();
    int chin = inputs[0]->channel();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int m = chout / param.group;
    int k = chin * kh * kw / param.group;
    int n = hout * wout;
    _w_scale = param.weight()->get_scale();
    //! update weights scale
    const DataType out_type = outputs[0]->get_dtype();
    if (out_type == AK_FLOAT || out_type == AK_INT8) {
         CHECK_EQ(_w_scale.size(), chout) << "weights scale size must be chout";
        float input_scale = inputs[0]->get_scale()[0];
        for (auto& ws : this->_w_scale)
        {
            ws *= input_scale;
            if (out_type == AK_INT8) {
               ws /= outputs[0]->get_scale()[0];
            }
        }
    }
    bool kps_equal = (param.pad_w == param.pad_h) && (param.stride_w == param.stride_h) && (kw == kh);
    bool ks_equal = (param.stride_w == param.stride_h) && (kw == kh);
    //! select conv gemmlike kernel
    if (kw == 1 && param.stride_w == 1 && param.pad_w == 0 && kps_equal){
        //! 1x1s1p0 gemmlike conv
        _impl_int8 = conv1x1s1_gemm_int8;
    } else {
        //! otherwise case
        if (kw == 3 && param.stride_w == 1 && n > 1 && ks_equal){
            _idx_data.reshape(Shape({1, 1, 1, n * kh * kw}));
            int* idx_out = (int*)_idx_data.mutable_data();
            for (int i = 0; i < hout; ++i) {
                for (int j = 0; j < wout; ++j) {
                    compute_offset(idx_out, i, j, kh, kw, hin, win, param.pad_h, \
                        param.pad_w, param.dilation_h, param.dilation_w);
                    idx_out += kh * kw;
                }
            }
        }
        //! im2col gemmlike conv
         _impl_int8 = conv_im2col_gemm_int8;
        this->_ctx->workspace_extend(Shape({1, 1, 1, (k * n + 3) / 4}));
    }

    if (n > 1) {
        prepackA_int8(_weights_trans, *param.weight(), m, k, param.group, false, this->_ctx);
        _is_trans_weights = true;
    }
    return SaberSuccess;
}

template<>
SaberStatus SaberGemmLikeConv<AK_INT8>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberGemmLikeConv<AK_INT8>::dispatch(
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
    const int32_t* idx_data = static_cast<const int32_t*>(_idx_data.data());

    int num = inputs[0]->num();
    int chin = inputs[0]->channel();
    int chout = outputs[0]->channel();
    int hin = inputs[0]->height();
    int win = inputs[0]->width();
    int hout = outputs[0]->height();
    int wout = outputs[0]->width();
    // printf("invoke gemm int8 conv\n");
    _impl_int8(din, dout, num, chout, hout, wout, chin, hin, win, weights, bias, param, \
        this->_ctx, outputs[0]->get_dtype(), _w_scale.data(), idx_data);
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "GemmLikeConv int8: " << this->_op_name.c_str() << " : time: " << ts;
#endif
    return SaberSuccess;
}
}
} // namespace anakin
