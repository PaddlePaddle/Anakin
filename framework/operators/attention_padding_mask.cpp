#include "framework/operators/attention_padding_mask.h"

namespace anakin {

namespace ops {

#define INSTANCE_ATTENTION_PADDING_MASK(Ttype, Ptype) \
template<> \
void AttentionPaddingMask<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<AttentionPaddingMaskHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<AttentionPaddingMaskHelper<Ttype, Ptype>*>(this->_helper)->_param_attention_padding_mask; \
    impl->_funcs_attention_padding_mask(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
AttentionPaddingMaskHelper<Ttype, Ptype>::~AttentionPaddingMaskHelper() {
}

template<typename Ttype, Precision Ptype>
Status AttentionPaddingMaskHelper<Ttype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing AttentionPaddingMask op parameter.";
    auto mask = GET_PARAMETER(float, mask);
    AttentionPaddingMaskParam<Ttype> param_attention_padding_mask(mask, 12800001);
    _param_attention_padding_mask = param_attention_padding_mask;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status AttentionPaddingMaskHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_attention_padding_mask.init(ins, outs, _param_attention_padding_mask, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status AttentionPaddingMaskHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_attention_padding_mask.compute_output_shape(ins, outs, _param_attention_padding_mask));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ATTENTION_PADDING_MASK(NV, Precision::FP32);
template class AttentionPaddingMaskHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(AttentionPaddingMask, AttentionPaddingMaskHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_ATTENTION_PADDING_MASK(X86, Precision::FP32);
INSTANCE_ATTENTION_PADDING_MASK(X86, Precision::FP16);
INSTANCE_ATTENTION_PADDING_MASK(X86, Precision::INT8);
template class AttentionPaddingMaskHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(AttentionPaddingMask, AttentionPaddingMaskHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ATTENTION_PADDING_MASK(ARM, Precision::FP32);
template class AttentionPaddingMaskHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(AttentionPaddingMask, AttentionPaddingMaskHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_ATTENTION_PADDING_MASK(AMD, Precision::FP32);
template class AttentionPaddingMaskHelper<AMD, Precision::FP32>;
template class AttentionPaddingMaskHelper<AMD, Precision::FP16>;
template class AttentionPaddingMaskHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(AttentionPaddingMask, AttentionPaddingMaskHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(AttentionPaddingMask)
.Doc("AttentionPaddingMask operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("attention_padding_mask")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("attention_padding_mask")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("attention_padding_mask")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("attention_padding_mask")
#endif
.num_in(2)
.num_out(1)
.Args<float>("mask", "padding data need to be set to mask");

} /* namespace ops */

} /* namespace anakin */

