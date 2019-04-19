#include "framework/operators/sequence_padding.h"

namespace anakin {

namespace ops {

#define INSTANCE_SEQUENCE_PADDING(Ttype, Ptype) \
template<> \
void SequencePadding<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<SequencePaddingHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SequencePaddingHelper<Ttype, Ptype>*>(this->_helper)->_param_sequence_padding; \
    impl->_funcs_sequence_padding(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
SequencePaddingHelper<Ttype, Ptype>::~SequencePaddingHelper() {
}

template<typename Ttype, Precision Ptype>
Status SequencePaddingHelper<Ttype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing SequencePadding op parameter.";
    SequencePaddingParam<Ttype> param_sequence_padding;
    _param_sequence_padding = param_sequence_padding;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SequencePaddingHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_sequence_padding.init(ins, outs, _param_sequence_padding, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SequencePaddingHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_sequence_padding.compute_output_shape(ins, outs, _param_sequence_padding));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SEQUENCE_PADDING(NV, Precision::FP32);
template class SequencePaddingHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequencePadding, SequencePaddingHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_SEQUENCE_PADDING(X86, Precision::FP32);
INSTANCE_SEQUENCE_PADDING(X86, Precision::FP16);
INSTANCE_SEQUENCE_PADDING(X86, Precision::INT8);
template class SequencePaddingHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequencePadding, SequencePaddingHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SEQUENCE_PADDING(ARM, Precision::FP32);
template class SequencePaddingHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequencePadding, SequencePaddingHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_SEQUENCE_PADDING(AMD, Precision::FP32);
template class SequencePaddingHelper<AMD, Precision::FP32>;
template class SequencePaddingHelper<AMD, Precision::FP16>;
template class SequencePaddingHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(SequencePadding, SequencePaddingHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(SequencePadding)
.Doc("SequencePadding operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("sequence_padding")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("sequence_padding")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("sequence_padding")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("sequence_padding")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */

