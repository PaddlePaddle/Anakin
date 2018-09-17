#include "framework/operators/conv_unpadding_padding.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
Status ConvUnpaddingPaddingHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing ConvUnpaddingPadding op parameter.";
    _param_conv_upadding_padding = ConvUnpaddingPaddingParam<Ttype>();
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConvUnpaddingPaddingHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
                                        const std::vector<Tensor4dPtr<Ttype> >& ins,
                                        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_conv_upadding_padding.init(ins, outs, _param_conv_upadding_padding, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConvUnpaddingPaddingHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>>& ins,
        std::vector<Tensor4dPtr<Ttype>>& outs) {
    SABER_CHECK(_funcs_conv_upadding_padding.compute_output_shape(ins, outs, _param_conv_upadding_padding));
    return Status::OK();
}


#define INSTANCE_CONCAT(Ttype, Ptype) \
template<> \
void ConvUnpaddingPadding<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
                std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<ConvUnpaddingPaddingHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ConvUnpaddingPaddingHelper<Ttype, Ptype>*>(this->_helper)->_param_conv_upadding_padding; \
    impl->_funcs_conv_upadding_padding(ins, outs, param, ctx); \
}

#ifdef USE_CUDA
INSTANCE_CONCAT(NV, Precision::FP32);
template class ConvUnpaddingPaddingHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvUnpaddingPadding, ConvUnpaddingPaddingHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_CONCAT(ARM, Precision::FP32);
template class ConvUnpaddingPaddingHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvUnpaddingPadding, ConvUnpaddingPaddingHelper, ARM, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_CONCAT(X86, Precision::FP32);
template class ConvUnpaddingPaddingHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvUnpaddingPadding, ConvUnpaddingPaddingHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(ConvUnpaddingPadding)
.Doc("ConvUnpaddingPadding operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("conv_unpadding_padding")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("conv_unpadding_padding")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("conv_unpadding_padding")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


