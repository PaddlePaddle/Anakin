#include "framework/operators/sequence_concat.h"

namespace anakin {

namespace ops {

#define INSTANCE_SEQUENCE_CONCAT(Ttype, Ptype) \
template<> \
void SequenceConcat<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<SequenceConcatHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SequenceConcatHelper<Ttype, Ptype>*>(this->_helper)->_param_sequence_concat; \
    impl->_funcs_sequence_concat(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
SequenceConcatHelper<Ttype, Ptype>::~SequenceConcatHelper() {
}

template<typename Ttype, Precision Ptype>
Status SequenceConcatHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing SequenceConcat op parameter.";
    SequenceConcatParam<Ttype> param_sequence_concat;
    _param_sequence_concat = param_sequence_concat;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SequenceConcatHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_sequence_concat.init(ins, outs, _param_sequence_concat, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SequenceConcatHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_sequence_concat.compute_output_shape(ins, outs, _param_sequence_concat));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SEQUENCE_CONCAT(NV, Precision::FP32);
template class SequenceConcatHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequenceConcat, SequenceConcatHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_SEQUENCE_CONCAT(X86, Precision::FP32);
template class SequenceConcatHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequenceConcat, SequenceConcatHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SEQUENCE_CONCAT(ARM, Precision::FP32);
template class SequenceConcatHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequenceConcat, SequenceConcatHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_SEQUENCE_CONCAT(AMD, Precision::FP32);
template class SequenceConcatHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequenceConcat, SequenceConcatHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(SequenceConcat)
.Doc("SequenceConcat operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("sequence_concat")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("sequence_concat")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("sequence_concat")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("sequence_concat")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */

