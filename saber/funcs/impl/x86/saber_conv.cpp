
#include "saber/funcs/impl/x86/saber_conv.h"
#include "saber/funcs/impl/x86/saber_im2col_conv.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_conv.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_dwconv.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_conv1x1.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_conv.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_core_u8s8s32x_conv.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_core_u8s8s32x_1x1_conv.h"
#include "saber/funcs/impl/x86/gemm_u8s8s32x_conv.h"

namespace anakin {
namespace saber {

using namespace jit;

template <>
SaberStatus SaberConv2D<X86, AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    EltwiseParam<X86> elt_param(Eltwise_sum);
    elt_param.has_eltwise = false;
    ConvEltwiseParam<X86> conv_elt_param(param, elt_param);

    return this->impl->create(inputs, outputs, conv_elt_param, ctx);
}

template <>
SaberStatus SaberConv2D<X86, AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    EltwiseParam<X86> elt_param(Eltwise_sum);
    elt_param.has_eltwise = false;
    ConvEltwiseParam<X86> conv_elt_param(param, elt_param);
    bool use_avx512 = false;//mayiuse(avx512_common);
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

    this->impl->init(inputs, outputs, conv_elt_param, ctx);
    return create(inputs, outputs, param, ctx);

}

template <>
SaberStatus SaberConv2D<X86, AK_FLOAT>::\
dispatch(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvParam<X86>& param) {
    EltwiseParam<X86> elt_param(Eltwise_sum);
    elt_param.has_eltwise = false;
    ConvEltwiseParam<X86> conv_elt_param(param, elt_param);
    return this->impl->dispatch(inputs, outputs, conv_elt_param);
}


template <>
SaberStatus SaberConv2D<X86, AK_INT8>::\
create(const std::vector<Tensor<X86> *>& inputs,
       std::vector<Tensor<X86> *>& outputs,
       ConvParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    EltwiseParam<X86> elt_param(Eltwise_sum);
    elt_param.has_eltwise = false;
    ConvEltwiseParam<X86> conv_elt_param(param, elt_param);

    return this->impl->create(inputs, outputs, conv_elt_param, ctx);
}

template <>
SaberStatus SaberConv2D<X86, AK_INT8>::\
init(const std::vector<Tensor<X86> *>& inputs,
     std::vector<Tensor<X86> *>& outputs,
     ConvParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    EltwiseParam<X86> elt_param(Eltwise_sum);
    elt_param.has_eltwise = false;
    ConvEltwiseParam<X86> conv_elt_param(param, elt_param);
    ConvParam<X86>* conv_param = &(param);

    int kernel_h = param.weight()->height();
    int kernel_w = param.weight()->width();
    Shape src_shape(inputs[0]->shape());
    Shape dst_shape(outputs[0]->shape());
    int ic = src_shape[3], oc = dst_shape[3];

    if (ic & 0xf || oc & 0xf) {
        this->impl = new GemmU8S8S32XConv();
    } else if (kernel_h == 1 && kernel_w == 1 && conv_param->pad_h == 0 && conv_param->pad_w == 0
               && conv_param->stride_h == 1 && conv_param->stride_w == 1 && conv_param->group == 1) {
        this->impl = new JitAvx512u8s8s32xConv1x1();
    } else {
        this->impl = new JitAvx512U8S8S32XConv();
    }

    return this->impl->init(inputs, outputs, conv_elt_param, ctx);
}

template <>
SaberStatus SaberConv2D<X86, AK_INT8>::\
dispatch(const std::vector<Tensor<X86> *>& inputs,
         std::vector<Tensor<X86> *>& outputs,
         ConvParam<X86>& param) {
    EltwiseParam<X86> elt_param(Eltwise_sum);
    elt_param.has_eltwise = false;
    ConvEltwiseParam<X86> conv_elt_param(param, elt_param);
    return this->impl->dispatch(inputs, outputs, conv_elt_param);
}


DEFINE_OP_TEMPLATE(SaberConv2D, ConvParam, X86, AK_HALF);
}
}
