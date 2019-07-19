#include "saber/funcs/impl/arm/saber_conv_winograd.h"
#include "saber/funcs/impl/arm/neon/impl/conv_arm_impl.h"
#include "saber/funcs/impl/arm/neon/impl/sgemm_prepacked.h"

namespace anakin{
namespace saber {

template<>
SaberStatus SaberWinogradConv<AK_FLOAT>::create(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    int ks = param.weight()->width();
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chin = inputs[0]->channel();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();

    if (ks == 3){
        //! 3x3 winograd conv
        _is_trans_weights = true;
        int tile_w = (wout + 5) / 6;
        int tile_h = (hout + 5) / 6;
        int size_tile = tile_h * tile_w;
        int size_trans_channel = 8 * 8 * size_tile;
        int max_ch = chin > chout? chin : chout;

        const int m_wino = chout;
        const int n_wino = size_tile;
        int hblock = get_hblock(this->_ctx->get_arch());
        int m_round = hblock * ((m_wino + hblock - 1) / hblock);

        Shape shape_w_out({1, 1, 1, 8 * 8 * m_round * chin});
        _weights_trans.reshape(shape_w_out);
        auto flag = _ctx->workspace_extend(Shape({1, 1, 1, size_trans_channel * max_ch * 2 + n_wino}));
        float* weights_wino = static_cast<float*>(fast_malloc(sizeof(float) * 8 * 8 * chout * chin));
        void* trans_tmp_ptr = fast_malloc(sizeof(float) * 8 * 8 * chout * chin);

        if (flag == SaberSuccess && weights_wino && trans_tmp_ptr) {
            winograd_transform_weights(weights_wino, param.weight()->data(), chout, chin, trans_tmp_ptr);
            float* weights_trans = static_cast<float*>(_weights_trans.mutable_data());
            for (int i = 0; i < 64; ++i) {
                float* packed_weights = weights_trans + i * m_round * chin;
                const float* weights_wino_ptr = weights_wino + i * chout * chin;
                prepackA(packed_weights, weights_wino_ptr, chin, 0, m_wino, 0, chin, false, this->_ctx);
            }
            _impl = conv_arm_winograd3x3;
            fast_free(trans_tmp_ptr);
            fast_free(weights_wino);
            return SaberSuccess;
        }
        fast_free(trans_tmp_ptr);
        fast_free(weights_wino);
    } else {
        LOG(ERROR) << "this type winograd conv not impl";
        return SaberUnImplError;
    }
    return SaberSuccess;
}

template<>
SaberStatus SaberWinogradConv<AK_FLOAT>::init(const std::vector<Tensor<ARM> *>& inputs,
                        std::vector<Tensor<ARM> *>& outputs,
                        ConvParam<ARM>& param, Context<ARM>& ctx){
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberWinogradConv<AK_FLOAT>::dispatch(
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
    LOG(INFO) << "WinogradConv fp32: " << this->_op_name.c_str() << " : time: " << ts;
#endif
    return SaberSuccess;
}

}
} // namespace anakin
