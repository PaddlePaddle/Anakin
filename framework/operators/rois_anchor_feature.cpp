#include "framework/operators/rois_anchor_feature.h"

namespace anakin {
namespace ops {

#ifdef USE_CUDA
template<>
void RoisAnchorFeature<NV, Precision::FP32>::operator()(
        OpContext<NV>& ctx, const std::vector<Tensor4dPtr<NV> >& ins,
        std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<RoisAnchorFeatureHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<RoisAnchorFeatureHelper<NV, Precision::FP32>*>
    (this->_helper)->_param_rois_anchor_feature;
    impl->_funcs_rois_anchor_feature(ins, outs, param, ctx);
}
#endif
/// TODO ... specialization other type of operator
/// set helper
template<typename Ttype, Precision Ptype>
RoisAnchorFeatureHelper<Ttype, Ptype>::~RoisAnchorFeatureHelper() {
}
template<typename Ttype, Precision Ptype>
Status RoisAnchorFeatureHelper<Ttype, Ptype>::InitParam() {
            LOG(WARNING) << "Parsing RoisAnchorFeature op parameter.";
    auto min_anchor_size_in = GET_PARAMETER(float, min_anchor_size);
    auto num_anchor_scales_in = GET_PARAMETER(int, num_anchor_scales);
    auto anchor_scale_pow_base_in = GET_PARAMETER(float, anchor_scale_pow_base);
    auto anchor_wph_ratios_in = GET_PARAMETER(PTuple<float>, anchor_wph_ratios);
    auto num_top_iou_anchor_in = GET_PARAMETER(int, num_top_iou_anchor);
    auto min_num_top_iou_anchor_in = GET_PARAMETER(int, min_num_top_iou_anchor);
    auto iou_thr_in = GET_PARAMETER(float, iou_thr);
    auto ft_ratio_h_in = GET_PARAMETER(bool, ft_ratio_h);
    auto ft_ratio_w_in = GET_PARAMETER(bool, ft_ratio_w);
    auto ft_log_ratio_h_in = GET_PARAMETER(bool, ft_log_ratio_h);
    auto ft_log_ratio_w_in = GET_PARAMETER(bool, ft_log_ratio_w);
    auto bbox_size_add_one_in = GET_PARAMETER(bool, bbox_size_add_one);

    saber::RoisAnchorFeatureParam<Ttype> rois_anchor_feature_param(
            anchor_wph_ratios_in.vector(), min_anchor_size_in, num_anchor_scales_in,
            anchor_scale_pow_base_in, num_top_iou_anchor_in, min_num_top_iou_anchor_in,
            iou_thr_in, ft_ratio_h_in, ft_ratio_w_in, ft_log_ratio_h_in, ft_log_ratio_w_in,
            bbox_size_add_one_in);

    _param_rois_anchor_feature = rois_anchor_feature_param;
    return Status::OK();
}
template<typename Ttype, Precision Ptype>
Status RoisAnchorFeatureHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    _funcs_rois_anchor_feature.init(ins, outs, _param_rois_anchor_feature,
            SPECIFY, SABER_IMPL, ctx);

    return Status::OK();
}
template<typename Ttype,  Precision Ptype>
Status RoisAnchorFeatureHelper<Ttype, Ptype>::InferShape(
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    _funcs_rois_anchor_feature.compute_output_shape(ins, outs, _param_rois_anchor_feature);
    return Status::OK();
}
#ifdef USE_CUDA
template class RoisAnchorFeatureHelper<NV, Precision::FP32>;
template class RoisAnchorFeatureHelper<NV, Precision::FP16>;
template class RoisAnchorFeatureHelper<NV, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class RoisAnchorFeatureHelper<ARM, Precision::FP32>;
template class RoisAnchorFeatureHelper<ARM, Precision::FP16>;
template class RoisAnchorFeatureHelper<ARM, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(RoisAnchorFeature, RoisAnchorFeatureHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(RoisAnchorFeature, RoisAnchorFeatureHelper, ARM, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(RoisAnchorFeature)
.Doc("RoisAnchorFeature operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("rpn_proposal_ssd")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("rpn_proposal_ssd")
#endif
.num_in(1)
.num_out(1)
.Args<float>("min_anchor_size", " param of rois_anchor_feature_param")
.Args<int>("num_anchor_scales", " param of rois_anchor_feature_param")
.Args<float>("anchor_scale_pow_base", " param of rois_anchor_feature_param")
.Args<PTuple<float> >("anchor_wph_ratios", "param of rois_anchor_feature_param")
.Args<int>("num_top_iou_anchor", " param of rois_anchor_feature_param")
.Args<int>("min_num_top_iou_anchor", " param of rois_anchor_feature_param")
.Args<float>("iou_thr", " param of rois_anchor_feature_param")
.Args<bool>("ft_ratio_h", " param of rois_anchor_feature_param")
.Args<bool>("ft_ratio_w", " param of rois_anchor_feature_param")
.Args<bool>("ft_log_ratio_h", " param of rois_anchor_feature_param")
.Args<bool>("ft_log_ratio_w", " param of rois_anchor_feature_param")
.Args<bool>("bbox_size_add_one", " param of rois_anchor_feature_param");
} /* namespace ops */
} /* namespace anakin */