#include "framework/operators/gather.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Gather<NV, Precision::FP32>::operator()(OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV>>& ins,
        std::vector<Tensor4dPtr<NV>>& outs) {
}
#endif
#ifdef USE_X86_PLACE
template<>
void Gather<X86, Precision::FP32>::operator()(OpContext<X86>& ctx,
      const std::vector<Tensor4dPtr<X86>>& ins,
      std::vector<Tensor4dPtr<X86>>& outs) {
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
GatherHelper<Ttype, Ptype>::~GatherHelper() {
}

template<typename Ttype, Precision Ptype>
Status GatherHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Gather op parameter.";
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status GatherHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype>>& ins,
        std::vector<Tensor4dPtr<Ttype>>& outs) {
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status GatherHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    outs[0]->set_shape(ins[0]->valid_shape());
    outs[0]->set_seq_offset(ins[0]->get_seq_offset());
    return Status::OK();
}

#ifdef USE_CUDA
template class GatherHelper<NV, Precision::FP32>;
template class GatherHelper<NV, Precision::FP16>;
template class GatherHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class GatherHelper<ARM, Precision::FP32>;
template class GatherHelper<ARM, Precision::FP16>;
template class GatherHelper<ARM, Precision::INT8>;
#endif
#ifdef USE_X86_PLACE
template class GatherHelper<X86, Precision::FP32>;
template class GatherHelper<X86, Precision::FP16>;
template class GatherHelper<X86, Precision::INT8>;
#endif

// register help
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, NV, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, NV, Precision::FP16);
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, NV, Precision::INT8);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, ARM, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, ARM, Precision::FP16);
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, ARM, Precision::INT8);
#endif

#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, X86, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, X86, Precision::FP16);
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, X86, Precision::INT8);
#endif

//! register op
ANAKIN_REGISTER_OP(Gather) 
#ifdef USE_CUDA
    .__alias__<NV, Precision::FP32>("gather")
#endif
#ifdef USE_ARM_PLACE
    .__alias__<ARM, Precision::FP32>("gather")
#endif
#ifdef USE_X86_PLACE
    .__alias__<X86, Precision::FP32>("gather")
#endif
	.Doc("Gather operator [ only a middle data holder and reshape ] ");

} /* namespace ops */

} /* namespace anakin */


