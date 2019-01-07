#include "framework/operators/reverse_sequence.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
Status ReverseSequenceHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing ReverseSequence op parameter.";
    _param_reverse_sequence = EmptyParam<Ttype>();
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ReverseSequenceHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_reverse_sequence.init(ins, outs, _param_reverse_sequence, SPECIFY,
                                          SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ReverseSequenceHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>>&
        ins,
        std::vector<Tensor4dPtr<Ttype>>& outs) {
    SABER_CHECK(_funcs_reverse_sequence.compute_output_shape(ins, outs,
                _param_reverse_sequence));
    return Status::OK();
}


#define INSTANCE_CONCAT(Ttype, Ptype) \
template<> \
void ReverseSequence<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
                std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<ReverseSequenceHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ReverseSequenceHelper<Ttype, Ptype>*>(this->_helper)->_param_reverse_sequence; \
    impl->_funcs_reverse_sequence(ins, outs, param, ctx); \
}

#ifdef USE_CUDA
INSTANCE_CONCAT(NV, Precision::FP32);
template class ReverseSequenceHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReverseSequence, ReverseSequenceHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_CONCAT(ARM, Precision::FP32);
template class ReverseSequenceHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReverseSequence, ReverseSequenceHelper, ARM, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_CONCAT(X86, Precision::FP32);
template class ReverseSequenceHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReverseSequence, ReverseSequenceHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(ReverseSequence)
.Doc("ReverseSequence operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("reverse_sequence")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("reverse_sequence")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("reverse_sequence")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


