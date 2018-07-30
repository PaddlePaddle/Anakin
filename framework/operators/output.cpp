#include "framework/operators/output.h"

namespace anakin {

namespace ops {

#define INSTANCE_OUTPUT(Ttype, Dtype, Ptype) \
template<> \
void Output<Ttype, Dtype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype>>& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype>>& outs) {}


template<typename Ttype, DataType Dtype, Precision Ptype>
Status OutputHelper<Ttype, Dtype, Ptype>::InitParam() {
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status OutputHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype> &ctx,
                                               const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                                               std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status OutputHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                                std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_OUTPUT(NV, AK_FLOAT, Precision::FP32);
template class OutputHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
INSTANCE_OUTPUT(X86, AK_FLOAT, Precision::FP32);
template class OutputHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_OUTPUT(ARM, AK_FLOAT, Precision::FP32);
template class OutputHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, ARM, AK_FLOAT, Precision::FP32);
#endif //arm

//! register op
ANAKIN_REGISTER_OP(Output)
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("output")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("output")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, AK_FLOAT, Precision::FP32>("output")
.__alias__<X86, AK_FLOAT, Precision::FP32>("Output")
#endif
.Doc("Output operator [ only a input data holder and reshape ] ");

} /* namespace ops */

} /* namespace anakin */


