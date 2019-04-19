#include "framework/operators/affine_channel.h"

namespace anakin {

namespace ops {

#define INSTANCE_ACTIVATION(Ttype, Ptype) \
template<> \
void AffineChannel<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<AffineChannelHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<AffineChannelHelper<Ttype, Ptype>*>(this->_helper)->_param_affine_channel; \
    impl->_funcs_affine_channel(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
AffineChannelHelper<Ttype, Ptype>::~AffineChannelHelper() {
}

template<typename Ttype, Precision Ptype>
Status AffineChannelHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing AffineChannel op parameter.";
    using pblock_type = PBlock<Ttype>;
    auto weights = GET_PARAMETER(pblock_type, weight_1);
    auto bias = GET_PARAMETER(pblock_type, weight_2);
    AffineChannelParam<Ttype> param_affine_channel(&(weights.d_tensor()), &(bias.d_tensor()));
    _param_affine_channel = param_affine_channel;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status AffineChannelHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_affine_channel.init(ins, outs, _param_affine_channel, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status AffineChannelHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_affine_channel.compute_output_shape(ins, outs, _param_affine_channel));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ACTIVATION(NV, Precision::FP32);

template<>
Status AffineChannelHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx, 
                                                   const std::vector< Tensor4dPtr<NV> > & ins, 
                                                   std::vector< Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_affine_channel.init(ins, outs, _param_affine_channel, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(AffineChannel, AffineChannelHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_ACTIVATION(X86, Precision::FP32);
INSTANCE_ACTIVATION(X86, Precision::FP16);
INSTANCE_ACTIVATION(X86, Precision::INT8);
template class AffineChannelHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(AffineChannel, AffineChannelHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ACTIVATION(ARM, Precision::FP32);
template class AffineChannelHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(AffineChannel, AffineChannelHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_ACTIVATION(AMD, Precision::FP32);
template class AffineChannelHelper<AMD, Precision::FP32>;
template class AffineChannelHelper<AMD, Precision::FP16>;
template class AffineChannelHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(AffineChannel, AffineChannelHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(AffineChannel)
.Doc("AffineChannel operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("affine_channel")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("affine_channel")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("affine_channel")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("affine_channel")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */

