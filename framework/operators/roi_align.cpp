#include "framework/operators/roi_align.h"

namespace anakin {

namespace ops {

#define INSTANCE_ROI_ALIGN(Ttype, Ptype) \
template<> \
void RoiAlign<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<RoiAlignHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<RoiAlignHelper<Ttype, Ptype>*>(this->_helper)->_param_roi_align; \
    impl->_funcs_roi_align(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
RoiAlignHelper<Ttype, Ptype>::~RoiAlignHelper() {
}

template<typename Ttype, Precision Ptype>
Status RoiAlignHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing RoiAlign op parameter.";
    auto pooled_height = GET_PARAMETER(int, pooled_height);
    auto pooled_width = GET_PARAMETER(int, pooled_width);
    auto spatial_scale = GET_PARAMETER(float, spatial_scale);
    auto sampling_ratio = GET_PARAMETER(int, sampling_ratio);
    RoiAlignParam<Ttype> param_roi_align(pooled_height, pooled_width, spatial_scale, sampling_ratio);
    _param_roi_align = param_roi_align;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status RoiAlignHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_roi_align.init(ins, outs, _param_roi_align, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status RoiAlignHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_roi_align.compute_output_shape(ins, outs, _param_roi_align));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ROI_ALIGN(NV, Precision::FP32);

template<>
Status RoiAlignHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx, 
                                                   const std::vector< Tensor4dPtr<NV> > & ins, 
                                                   std::vector< Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_roi_align.init(ins, outs, _param_roi_align, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(RoiAlign, RoiAlignHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_ROI_ALIGN(X86, Precision::FP32);
INSTANCE_ROI_ALIGN(X86, Precision::FP16);
INSTANCE_ROI_ALIGN(X86, Precision::INT8);
template class RoiAlignHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(RoiAlign, RoiAlignHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ROI_ALIGN(ARM, Precision::FP32);
template class RoiAlignHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(RoiAlign, RoiAlignHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_ROI_ALIGN(AMD, Precision::FP32);
template class RoiAlignHelper<AMD, Precision::FP32>;
template class RoiAlignHelper<AMD, Precision::FP16>;
template class RoiAlignHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(RoiAlign, RoiAlignHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(RoiAlign)
.Doc("RoiAlign operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("roi_align")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("roi_align")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("roi_align")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("roi_align")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " type of RoiAlign ")
.Args<bool>("channel_shared", "prelu channel is shared or not ");

} /* namespace ops */

} /* namespace anakin */

