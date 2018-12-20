#include "framework/operators/anchor_generator.h"

namespace anakin {

namespace ops {

#define INSTANCE_ACTIVATION(Ttype, Ptype) \
template<> \
void AnchorGenerator<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<AnchorGeneratorHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<AnchorGeneratorHelper<Ttype, Ptype>*>(this->_helper)->_param_anchor_generator; \
    impl->_funcs_anchor_generator(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
AnchorGeneratorHelper<Ttype, Ptype>::~AnchorGeneratorHelper() {
}

template<typename Ttype, Precision Ptype>
Status AnchorGeneratorHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing AnchorGenerator op parameter.";
    auto offset = GET_PARAMETER(float, offset);
    auto anchor_sizes = GET_PARAMETER(PTuple<float>, anchor_sizes);
    auto aspect_ratios = GET_PARAMETER(PTuple<float>, aspect_ratios);
    auto variances = GET_PARAMETER(PTuple<float>, variances);
    auto stride = GET_PARAMETER(PTuple<float>, stride);
    AnchorGeneratorParam<Ttype> param_anchor_generator(anchor_sizes.vector(),
            aspect_ratios.vector(),
            variances.vector(),
            stride.vector(),
            offset);
    _param_anchor_generator = param_anchor_generator;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status AnchorGeneratorHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_anchor_generator.init(ins, outs, _param_anchor_generator, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status AnchorGeneratorHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_anchor_generator.compute_output_shape(ins, outs, _param_anchor_generator));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ACTIVATION(NV, Precision::FP32);

template<>
Status AnchorGeneratorHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx, 
                                                   const std::vector< Tensor4dPtr<NV> > & ins, 
                                                   std::vector< Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_anchor_generator.init(ins, outs, _param_anchor_generator, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(AnchorGenerator, AnchorGeneratorHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_ACTIVATION(X86, Precision::FP32);
INSTANCE_ACTIVATION(X86, Precision::FP16);
INSTANCE_ACTIVATION(X86, Precision::INT8);
template class AnchorGeneratorHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(AnchorGenerator, AnchorGeneratorHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ACTIVATION(ARM, Precision::FP32);
template class AnchorGeneratorHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(AnchorGenerator, AnchorGeneratorHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_ACTIVATION(AMD, Precision::FP32);
template class AnchorGeneratorHelper<AMD, Precision::FP32>;
template class AnchorGeneratorHelper<AMD, Precision::FP16>;
template class AnchorGeneratorHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(AnchorGenerator, AnchorGeneratorHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(AnchorGenerator)
.Doc("AnchorGenerator operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("anchor_generator")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("anchor_generator")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("anchor_generator")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("anchor_generator")
#endif
.num_in(1)
.num_out(2)
.Args<PTuple<float>>("anchor_sizes", " box size in image ")
.Args<PTuple<float>>("aspect_ratios", " box height and width ratio in image ")
.Args<PTuple<float>>("variances", " variances ")
.Args<PTuple<float>>("stride", " stride ")
.Args<float>("offset", " offset ");

} /* namespace ops */

} /* namespace anakin */

