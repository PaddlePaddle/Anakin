
#include "saber/funcs/impl/x86/saber_conv.h"
#include "saber/funcs/impl/x86/saber_im2col_conv.h"
#include "saber/funcs/impl/x86/jit_avx2_conv.h"
#include "saber/funcs/impl/x86/jit_uni_dwconv.h"
#include "saber/funcs/impl/x86/jit_avx512_conv1x1.h"
#include "saber/funcs/impl/x86/jit_avx512_conv.h"

namespace anakin {
namespace saber {

using namespace jit;
template <>
SaberStatus SaberConv2D<X86, AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
                                             std::vector<Tensor<X86> *>& outputs,
                                             ConvParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    return this->impl->create(inputs, outputs, param, ctx);
}
template <>
SaberStatus SaberConv2D<X86, AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvParam<X86>& param, Context<X86>& ctx) {

    this->_ctx = &ctx;
    bool use_avx512 = mayiuse(avx512_common);
    bool use_avx2 = mayiuse(avx2);
    if (use_avx512 && param.group == inputs[0]->channel() && param.group == outputs[0]->channel()) {
        this->impl = new JitUniDWConv<AK_FLOAT>;
    } else if (use_avx512 && param.weight()->height() == 1 && param.weight()->width() == 1) {
        this->impl = new JitAvx512Conv1x1<AK_FLOAT>;
    } else if (use_avx512 && outputs[0]->get_layout() == Layout_NCHW_C16) {
        this->impl = new JitAvx512Conv<AK_FLOAT>;
    } else if (use_avx2 && (outputs[0]->get_layout() == Layout_NCHW_C8)) {
        this->impl = new JitAvx2Conv<AK_FLOAT>;
    } else {
        this->impl = new SaberIm2colConv<AK_FLOAT>;
    }
    this->impl->init(inputs, outputs, param, ctx);
    return create(inputs, outputs, param, ctx);

}

template <>
SaberStatus SaberConv2D<X86, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<X86> *>& inputs,
             std::vector<Tensor<X86> *>& outputs,
             ConvParam<X86>& param) {
    return this->impl->dispatch(inputs, outputs, param);
}

template <>
SaberStatus SaberConv2D<X86, AK_INT8>::\
    create(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvParam<X86>& param, Context<X86>& ctx) {
    return SaberInvalidValue;
}

template <>
SaberStatus SaberConv2D<X86, AK_INT8>::\
    init(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvParam<X86>& param, Context<X86>& ctx) {
    return SaberInvalidValue;
}

template <>
SaberStatus SaberConv2D<X86, AK_INT8>::\
    dispatch(const std::vector<Tensor<X86> *>& inputs,
             std::vector<Tensor<X86> *>& outputs,
             ConvParam<X86>& param) {

    return SaberInvalidValue;
}
DEFINE_OP_TEMPLATE(SaberConv2D, ConvParam, X86, AK_INT16);
}
}
