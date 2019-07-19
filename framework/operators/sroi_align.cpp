#include "framework/operators/sroi_align.h"

namespace anakin {

namespace ops {

#define INSTANCE_SROI_ALIGN(Ttype, Ptype) \
template<> \
void SRoiAlign<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<SRoiAlignHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SRoiAlignHelper<Ttype, Ptype>*>(this->_helper)->_param_sroi_align; \
    impl->_funcs_sroi_align(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
SRoiAlignHelper<Ttype, Ptype>::~SRoiAlignHelper() {}

template<typename Ttype, Precision Ptype>
Status SRoiAlignHelper<Ttype, Ptype>::InitParam() {
            DLOG(WARNING) << "Parsing SRoiAlign op parameter.";
    auto pooled_h = GET_PARAMETER(int, pooled_h);
    auto pooled_w = GET_PARAMETER(int, pooled_w);
    auto spatial_scale = GET_PARAMETER(float, spatial_scale);
    SRoiAlignParam<Ttype> param_sroi_align(pooled_h, pooled_w, spatial_scale);
    _param_sroi_align = param_sroi_align;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SRoiAlignHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    saber::ImplEnum impl_e = SABER_IMPL;
    if (std::is_same<Ttype, X86>::value) {
        impl_e = SABER_IMPL;
    }
    SABER_CHECK(_funcs_sroi_align.init(ins, outs, _param_sroi_align, SPECIFY, impl_e, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SRoiAlignHelper<Ttype, Ptype>::InferShape(
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_sroi_align.compute_output_shape(ins, outs, _param_sroi_align));
    return Status::OK();
}

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_SROI_ALIGN(X86, Precision::FP32);
INSTANCE_SROI_ALIGN(X86, Precision::FP16);
INSTANCE_SROI_ALIGN(X86, Precision::INT8);
template class SRoiAlignHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SRoiAlign, SRoiAlignHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SROI_ALIGN(ARM, Precision::FP32);
template class SRoiAlignHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SRoiAlign, SRoiAlignHelper, ARM, Precision::FP32);
#endif//arm

//! register op
ANAKIN_REGISTER_OP(SRoiAlign)
.Doc("SRoiAlign operator")
#if defined USE_X86_PLACE || defined(BUILD_LITE)
.__alias__<X86, Precision::FP32>("sroi_align")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("sroi_align")
#endif
.num_in(1)
.num_out(1)
.Args<int>("pooled_h", "pooled_h of SRoiAlign")
.Args<int>("pooled_w", "pooled_w of SRoiAlign")
.Args<float>("spatial_scale", "spatial_scale of SRoiAlign");

} /* namespace ops */

} /* namespace anakin */

