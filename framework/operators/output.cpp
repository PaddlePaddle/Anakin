#include "framework/operators/output.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Output<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT>>& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT>>& outs) {
}
#endif

/// TODO ... specialization other type of operator
#ifdef USE_ARM_PLACE
template<>
void Output<ARM, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<ARM>& ctx,
    const std::vector<Tensor4dPtr<ARM, AK_FLOAT>>& ins,
    std::vector<Tensor4dPtr<ARM, AK_FLOAT>>& outs) {
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
OutputHelper<Ttype, Dtype, Ptype>::~OutputHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status OutputHelper<Ttype, Dtype, Ptype>::InitParam() {
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status OutputHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype>>& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype>>& outs) {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status OutputHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    return Status::OK();
}

#ifdef USE_CUDA
template class OutputHelper<NV, AK_FLOAT, Precision::FP32>;
template class OutputHelper<NV, AK_FLOAT, Precision::FP16>;
template class OutputHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
template class OutputHelper<ARM, AK_FLOAT, Precision::FP32>;
#endif
#ifdef ANAKIN_TYPE_FP16
template class OutputHelper<ARM, AK_FLOAT, Precision::FP16>;
#endif
#ifdef ANAKIN_TYPE_INT8
template class OutputHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif
#endif

#ifdef USE_CUDA
// register help
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, NV, AK_FLOAT, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, NV, AK_FLOAT, Precision::FP16);
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, NV, AK_FLOAT, Precision::INT8);
#endif
#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, ARM, AK_FLOAT, Precision::FP32);
#endif
#ifdef ANAKIN_TYPE_FP16
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, ARM, AK_FLOAT, Precision::FP16);
#endif
#ifdef ANAKIN_TYPE_INT8
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, ARM, AK_FLOAT, Precision::INT8);
#endif
#endif
//! register op
ANAKIN_REGISTER_OP(Output)
.Doc("Output operator [ only a input data holder and reshape ] ")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("output")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("output");
#endif

} /* namespace ops */

} /* namespace anakin */


