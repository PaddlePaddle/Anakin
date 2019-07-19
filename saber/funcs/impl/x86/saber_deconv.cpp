
#include "saber/funcs/impl/x86/saber_deconv.h"
#include "saber/funcs/impl/x86/saber_col2im_deconv.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_deconv.h"
#ifndef USE_SGX
#include "saber/funcs/impl/x86/vender_deconv.h"
#endif
namespace anakin {
namespace saber {

template <>
SaberStatus SaberDeconv2D<X86, AK_FLOAT>::create(
    const std::vector<Tensor<X86> *>& inputs,
    std::vector<Tensor<X86> *>& outputs,
    ConvParam<X86>& param, Context<X86>& ctx) {

    _impl->create(inputs, outputs, param, ctx);
    return SaberSuccess;
}

template <>
SaberStatus SaberDeconv2D<X86, AK_FLOAT>::init(
    const std::vector<Tensor<X86> *>& inputs,
    std::vector<Tensor<X86> *>& outputs,
    ConvParam<X86>& param, Context<X86>& ctx) {

    this->_ctx = &ctx;

    if (inputs[0]->get_layout() == Layout_NCHW_C8R) {
        _impl = new JitAvx2Deconv<AK_FLOAT>;
    } else if (inputs[0]->get_layout() == Layout_NCHW && outputs[0]->get_layout() == Layout_NCHW) {
        _impl = new SaberCol2ImDeconv<AK_FLOAT>;
    } else {
        LOG(FATAL) << "not support this layout";
    }

    return _impl->init(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberDeconv2D<X86, AK_FLOAT>::dispatch(
    const std::vector<Tensor<X86> *>& inputs,
    std::vector<Tensor<X86> *>& outputs,
    ConvParam<X86>& param) {

    return _impl->dispatch(inputs, outputs, param);
}

template <>
SaberStatus SaberDeconv2D<X86, AK_FLOAT>::trans_weights(Tensor<X86>& target_weights,
        Tensor<X86>& target_bias, int in_channel, int out_channel,
        int stride_h, int stride_w, int pad_h, int pad_w,
        int dilation_h, int dilation_w, int group) {
    return SaberUnImplError;
}

template <>
SaberStatus SaberDeconv2D<X86, AK_HALF>::trans_weights(Tensor<X86>& target_weights,
        Tensor<X86>& target_bias, int in_channel, int out_channel,
        int stride_h, int stride_w, int pad_h, int pad_w,
        int dilation_h, int dilation_w, int group) {

    return SaberUnImplError;
}

template <>
SaberStatus SaberDeconv2D<X86, AK_INT8>::trans_weights(Tensor<X86>& target_weights,
        Tensor<X86>& target_bias, int in_channel, int out_channel,
        int stride_h, int stride_w, int pad_h, int pad_w,
        int dilation_h, int dilation_w, int group) {
    return SaberUnImplError;
}

template class SaberDeconv2D<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberDeconv2D, ConvParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberDeconv2D, ConvParam, X86, AK_INT8);

}
}
