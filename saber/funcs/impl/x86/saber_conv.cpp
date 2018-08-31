
#include "saber/funcs/impl/x86/saber_conv.h"
#include "saber/funcs/impl/x86/saber_im2col_conv.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberConv2D<X86, AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
                                             std::vector<Tensor<X86> *>& outputs,
                                             ConvParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    this->impl->create(inputs, outputs, param, ctx);
    return SaberSuccess;
}
template <>
SaberStatus SaberConv2D<X86, AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvParam<X86>& param, Context<X86>& ctx) {

    this->_ctx = &ctx;
    this->impl = new SaberIm2colConv<AK_FLOAT>;
    this->impl->init(inputs, outputs, param, ctx);
    return create(inputs, outputs, param, ctx);

}

template <>
SaberStatus SaberConv2D<X86, AK_FLOAT>::\
    dispatch(const std::vector<Tensor<X86> *>& inputs,
             std::vector<Tensor<X86> *>& outputs,
             ConvParam<X86>& param) {
    this->impl->dispatch(inputs, outputs, param);
    return SaberSuccess;
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