#include "framework/operators/output.h"

namespace anakin {

namespace ops {

#define INSTANCE_OUTPUT(Ttype, Ptype) \
template<> \
void Output<Ttype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype>>& ins, \
    std::vector<Tensor4dPtr<Ttype>>& outs) {}

template<typename Ttype, Precision Ptype>
Status OutputHelper<Ttype, Ptype>::InitParam() {
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status OutputHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
                                        const std::vector<Tensor4dPtr<Ttype>>& ins,
                                        std::vector<Tensor4dPtr<Ttype>>& outs) {
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status OutputHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>>& ins,
        std::vector<Tensor4dPtr<Ttype>>& outs) {
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_OUTPUT(NV, Precision::FP32);
template class OutputHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_OUTPUT(X86, Precision::FP32);
template class OutputHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_OUTPUT(ARM, Precision::FP32);
template class OutputHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, ARM, Precision::FP32);
#endif //arm

#ifdef AMD_GPU
INSTANCE_OUTPUT(AMD, Precision::FP32);
template class OutputHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Output, OutputHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Output)
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("output")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("output")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("output")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("output")
#endif
.Doc("Output operator [ only a input data holder and reshape ] ");

} /* namespace ops */

} /* namespace anakin */


