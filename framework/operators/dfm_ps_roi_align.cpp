#include "framework/operators/dfm_ps_roi_align.h"
namespace anakin {
namespace ops {
#ifdef USE_CUDA
template<>
void DFMBPSROIAlign<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<DFMBPSROIAlignHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<DFMBPSROIAlignHelper<NV, Precision::FP32>*>
                  (this->_helper)->_param_dfm_ps_roi_align;
    impl->_funcs_dfm_ps_roi_align(ins, outs, param, ctx);
}
#endif
/// TODO ... specialization other type of operator
/// set helper
template<typename Ttype, Precision Ptype>
DFMBPSROIAlignHelper<Ttype, Ptype>::~DFMBPSROIAlignHelper() {
}
template<typename Ttype, Precision Ptype>
Status DFMBPSROIAlignHelper<Ttype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing DFMBPSROIAlign op parameter.";
    // get dfmb_psroi_pooling_param
    auto heat_map_a = GET_PARAMETER(float, heat_map_a);
    auto heat_map_b = GET_PARAMETER(float, heat_map_b);
    auto pad_ratio = GET_PARAMETER(float, pad_ratio);
    auto output_dim = GET_PARAMETER(int, output_dim);
    auto trans_std = GET_PARAMETER(float, trans_std);
    auto sample_per_part = GET_PARAMETER(int, sample_per_part);
    auto group_height = GET_PARAMETER(int, group_height);
    auto group_width = GET_PARAMETER(int, group_width);
    auto pooled_height = GET_PARAMETER(int, pooled_height);
    auto pooled_width = GET_PARAMETER(int, pooled_width);
    auto part_height = GET_PARAMETER(int, part_height);
    auto part_width = GET_PARAMETER(int, part_width);
    CHECK_EQ(part_height, 7);
    CHECK_EQ(part_width, 7);
    LOG(INFO) << "part_height " << part_height;
    LOG(INFO) << "part_width " << part_width;
    saber::DFMBPSROIAlignParam<Ttype> dfmb_psroi_align_param(
                heat_map_a,
                output_dim,
                heat_map_b,
                pad_ratio,
                trans_std,
                sample_per_part,
                group_height,
                group_width,
                pooled_height,
                pooled_width,
                part_height,
                part_width);
    _param_dfm_ps_roi_align = dfmb_psroi_align_param;
    return Status::OK();
}
template<typename Ttype, Precision Ptype>
Status DFMBPSROIAlignHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_dfm_ps_roi_align.init(ins, outs, _param_dfm_ps_roi_align, SPECIFY, SABER_IMPL, ctx);
    return Status::OK();
}
template<typename Ttype,  Precision Ptype>
Status DFMBPSROIAlignHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_dfm_ps_roi_align.compute_output_shape(ins, outs, _param_dfm_ps_roi_align);
    return Status::OK();
}
#ifdef USE_CUDA
template class DFMBPSROIAlignHelper<NV, Precision::FP32>;
template class DFMBPSROIAlignHelper<NV, Precision::FP16>;
template class DFMBPSROIAlignHelper<NV, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class DFMBPSROIAlignHelper<ARM, Precision::FP32>;
template class DFMBPSROIAlignHelper<ARM, Precision::FP16>;
template class DFMBPSROIAlignHelper<ARM, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(DFMBPSROIAlign, DFMBPSROIAlignHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(DFMBPSROIAlign, DFMBPSROIAlignHelper, ARM, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(DFMBPSROIAlign)
.Doc("DFMBPSROIAlign operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("rpn_proposal_ssd")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("rpn_proposal_ssd")
#endif
.num_in(1)
.num_out(1)
.Args<float>("heat_map_a", "heat_map_a of dfmb_psroi_pooling_param")
.Args<float>("heat_map_b", " of dfmb_psroi_pooling_param")
.Args<float>("pad_ratio", " of dfmb_psroi_pooling_param")
.Args<int>("output_dim", " of dfmb_psroi_pooling_param")
.Args<float>("trans_std", " of dfmb_psroi_pooling_param")
.Args<int>("sample_per_part", " of dfmb_psroi_pooling_param")
.Args<int>("group_height", " of dfmb_psroi_pooling_param")
.Args<int>("group_width", " of dfmb_psroi_pooling_param")
.Args<int>("pooled_height", " of dfmb_psroi_pooling_param")
.Args<int>("pooled_width", " of dfmb_psroi_pooling_param")
.Args<int>("part_height", " of dfmb_psroi_pooling_param")
.Args<int>("part_width", " of dfmb_psroi_pooling_param");
} /* namespace ops */
} /* namespace anakin */