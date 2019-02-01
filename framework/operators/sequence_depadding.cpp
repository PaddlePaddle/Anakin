#include "framework/operators/sequence_depadding.h"

namespace anakin {

namespace ops {

#define INSTANCE_SEQUENCE_DEPADDING(Ttype, Ptype) \
template<> \
void SequenceDePadding<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<SequenceDePaddingHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SequenceDePaddingHelper<Ttype, Ptype>*>(this->_helper)->_param_sequence_depadding; \
    impl->_funcs_sequence_depadding(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
SequenceDePaddingHelper<Ttype, Ptype>::~SequenceDePaddingHelper() {
}

template<typename Ttype, Precision Ptype>
Status SequenceDePaddingHelper<Ttype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing SequenceDePadding op parameter.";
    SequenceDePaddingParam<Ttype> param_sequence_depadding;
    _param_sequence_depadding = param_sequence_depadding;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SequenceDePaddingHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_sequence_depadding.init(ins, outs, _param_sequence_depadding, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SequenceDePaddingHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_sequence_depadding.compute_output_shape(ins, outs, _param_sequence_depadding));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SEQUENCE_DEPADDING(NV, Precision::FP32);
template class SequenceDePaddingHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequenceDePadding, SequenceDePaddingHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_SEQUENCE_DEPADDING(X86, Precision::FP32);
INSTANCE_SEQUENCE_DEPADDING(X86, Precision::FP16);
INSTANCE_SEQUENCE_DEPADDING(X86, Precision::INT8);
template class SequenceDePaddingHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequenceDePadding, SequenceDePaddingHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SEQUENCE_DEPADDING(ARM, Precision::FP32);
template class SequenceDePaddingHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequenceDePadding, SequenceDePaddingHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_SEQUENCE_DEPADDING(AMD, Precision::FP32);
template class SequenceDePaddingHelper<AMD, Precision::FP32>;
template class SequenceDePaddingHelper<AMD, Precision::FP16>;
template class SequenceDePaddingHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(SequenceDePadding, SequenceDePaddingHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(SequenceDePadding)
.Doc("SequenceDePadding operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("sequence_depadding")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("sequence_depadding")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("sequence_depadding")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("sequence_depadding")
#endif
.num_in(2)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */

