#include "framework/operators/gather.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Gather<NV, AK_FLOAT, Precision::FP32>::operator()(OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV, AK_FLOAT>>& ins,
        std::vector<Tensor4dPtr<NV, AK_FLOAT>>& outs) {
}
#endif
#ifdef USE_X86_PLACE
template<>
void Gather<X86, AK_FLOAT, Precision::FP32>::operator()(OpContext<X86>& ctx,
      const std::vector<Tensor4dPtr<X86, AK_FLOAT>>& ins,
      std::vector<Tensor4dPtr<X86, AK_FLOAT>>& outs) {
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
GatherHelper<Ttype, Dtype, Ptype>::~GatherHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status GatherHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Gather op parameter.";
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status GatherHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype>>& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype>>& outs) {
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status GatherHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    outs[0]->set_shape(ins[0]->valid_shape());
    outs[0]->set_seq_offset(ins[0]->get_seq_offset());
    return Status::OK();
}

#ifdef USE_CUDA
template class GatherHelper<NV, AK_FLOAT, Precision::FP32>;
template class GatherHelper<NV, AK_FLOAT, Precision::FP16>;
template class GatherHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class GatherHelper<ARM, AK_FLOAT, Precision::FP32>;
template class GatherHelper<ARM, AK_FLOAT, Precision::FP16>;
template class GatherHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif
#ifdef USE_X86_PLACE
template class GatherHelper<X86, AK_FLOAT, Precision::FP32>;
template class GatherHelper<X86, AK_FLOAT, Precision::FP16>;
template class GatherHelper<X86, AK_FLOAT, Precision::INT8>;
#endif

// register help
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, NV, AK_FLOAT, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, NV, AK_FLOAT, Precision::FP16);
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, NV, AK_FLOAT, Precision::INT8);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, ARM, AK_FLOAT, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, ARM, AK_FLOAT, Precision::FP16);
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, ARM, AK_FLOAT, Precision::INT8);
#endif

#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, X86, AK_FLOAT, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, X86, AK_FLOAT, Precision::FP16);
ANAKIN_REGISTER_OP_HELPER(Gather, GatherHelper, X86, AK_FLOAT, Precision::INT8);
#endif

//! register op
ANAKIN_REGISTER_OP(Gather) 
#ifdef USE_CUDA
    .__alias__<NV, AK_FLOAT, Precision::FP32>("gather")
#endif
#ifdef USE_ARM_PLACE
    .__alias__<ARM, AK_FLOAT, Precision::FP32>("gather")
#endif
#ifdef USE_X86_PLACE
    .__alias__<X86, AK_FLOAT, Precision::FP32>("gather")
#endif
	.Doc("Gather operator [ only a middle data holder and reshape ] ");

} /* namespace ops */

} /* namespace anakin */


