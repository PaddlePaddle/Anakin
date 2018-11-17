#include "framework/operators/rpn_proposal_ssd.h"
namespace anakin {
namespace ops {
#ifdef USE_CUDA
template<>
void RPNProposalSSD<NV,  Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<RPNProposalSSDHelper<NV,  Precision::FP32>*>(this->_helper);
    auto& param = static_cast<RPNProposalSSDHelper<NV,  Precision::FP32>*>
                  (this->_helper)->_param_rpn_prop_ssd;
    impl->_funcs_rpn_prop_ssd(ins, outs, param, ctx);
}
#endif
/// TODO ... specialization other type of operator
/// set helper
template<typename Ttype, Precision Ptype>
RPNProposalSSDHelper<Ttype, Ptype>::~RPNProposalSSDHelper() {
}
template<typename Ttype, Precision Ptype>
Status RPNProposalSSDHelper<Ttype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing RPNProposalSSD op parameter.";
    // get nms_param
    auto need_nms = GET_PARAMETER(bool, need_nms);
    auto overlap_ratio = GET_PARAMETER(PTuple<float>, overlap_ratio);
    auto top_n = GET_PARAMETER(PTuple<int>, top_n);
    auto add_score = GET_PARAMETER(bool, add_score);
    auto max_candidate_n = GET_PARAMETER(PTuple<int>, max_candidate_n);
    auto use_soft_nms = GET_PARAMETER(PTuple<bool>, use_soft_nms);
    auto nms_among_classes = GET_PARAMETER(bool, nms_among_classes);
    auto voting = GET_PARAMETER(PTuple<bool>, voting);
    auto vote_iou = GET_PARAMETER(PTuple<float>, vote_iou);
    auto nms_gpu_max_n_per_time = GET_PARAMETER(int, nms_gpu_max_n_per_time);
    // get detect_output_ssd
    auto threshold = GET_PARAMETER(PTuple<float>, threshold);
    auto channel_per_scale = GET_PARAMETER(int, channel_per_scale);
    auto class_name_list = GET_PARAMETER(std::string, class_name_list);
    auto num_class = GET_PARAMETER(int, num_class);
    auto refine_out_of_map_bbox = GET_PARAMETER(bool, refine_out_of_map_bbox);
    auto class_indexes = GET_PARAMETER(PTuple<int>, class_indexes);
    auto heat_map_a = GET_PARAMETER(PTuple<float>, heat_map_a);
    auto heat_map_b = GET_PARAMETER(PTuple<float>, heat_map_b);
    auto threshold_objectness = GET_PARAMETER(float, threshold_objectness);
    auto proposal_min_sqrt_area = GET_PARAMETER(PTuple<float>, proposal_min_sqrt_area);
    auto proposal_max_sqrt_area = GET_PARAMETER(PTuple<float>, proposal_max_sqrt_area);
    auto bg_as_one_of_softmax = GET_PARAMETER(bool, bg_as_one_of_softmax);
    auto use_target_type_rcnn = GET_PARAMETER(bool, use_target_type_rcnn);
    auto im_width = GET_PARAMETER(float, im_width);
    auto im_height = GET_PARAMETER(float, im_height);
    auto rpn_proposal_output_score = GET_PARAMETER(bool, rpn_proposal_output_score);
    auto regress_agnostic = GET_PARAMETER(bool, regress_agnostic);
    auto allow_border = GET_PARAMETER(float, allow_border);
    auto allow_border_ratio = GET_PARAMETER(float, allow_border_ratio);
    auto bbox_size_add_one = GET_PARAMETER(bool, bbox_size_add_one);
    auto read_width_scale = GET_PARAMETER(float, read_width_scale);
    auto read_height_scale = GET_PARAMETER(float, read_height_scale);
    auto read_height_offset = GET_PARAMETER(float, read_height_offset);
    auto min_size_h = GET_PARAMETER(float, min_size_h);
    auto min_size_w = GET_PARAMETER(float, min_size_w);
    auto min_size_mode = GET_PARAMETER(std::string, min_size_mode);
    // get gen_anchor_param
    auto base_size = GET_PARAMETER(float, base_size);
    auto ratios = GET_PARAMETER(PTuple<float>, ratios);
    auto scales = GET_PARAMETER(PTuple<float>, scales);
    auto anchor_width = GET_PARAMETER(PTuple<float>, anchor_width);
    auto anchor_height = GET_PARAMETER(PTuple<float>, anchor_height);
    auto anchor_x1 = GET_PARAMETER(PTuple<float>, anchor_x1);
    auto anchor_y1 = GET_PARAMETER(PTuple<float>, anchor_y1);
    auto anchor_x2 = GET_PARAMETER(PTuple<float>, anchor_x2);
    auto anchor_y2 = GET_PARAMETER(PTuple<float>, anchor_y2);
    auto zero_anchor_center = GET_PARAMETER(bool, zero_anchor_center);
    // get kpts_param
    //    auto kpts_exist_bottom_idx = GET_PARAMETER(int, kpts_exist_bottom_idx);
    //    auto kpts_reg_bottom_idx = GET_PARAMETER(int, kpts_reg_bottom_idx);
    //    auto kpts_reg_as_classify = GET_PARAMETER(bool, kpts_reg_as_classify);
    //    auto kpts_classify_width = GET_PARAMETER(int, kpts_classify_width);
    //    auto kpts_classify_height = GET_PARAMETER(int, kpts_classify_height);
    //    auto kpts_reg_norm_idx_st = GET_PARAMETER(int, kpts_reg_norm_idx_st);
    //    auto kpts_st_for_each_class = GET_PARAMETER(PTuple<int>, kpts_st_for_each_class);
    //    auto kpts_ed_for_each_class = GET_PARAMETER(PTuple<int>, kpts_ed_for_each_class);
    //    auto kpts_classify_pad_ratio = GET_PARAMETER(float, kpts_classify_pad_ratio);
    // get atrs_param
    //    auto atrs_reg_bottom_idx = GET_PARAMETER(int, atrs_reg_bottom_idx);
    //    auto atrs_reg_norm_idx_st = GET_PARAMETER(int, atrs_reg_norm_idx_st);
    //    auto atrs_norm_type = GET_PARAMETER(std::string, atrs_norm_type);
    // get ftrs_bottom_idx
    //    auto ftrs_bottom_idx = GET_PARAMETER(int, ftrs_bottom_idx);
    // get spmp_param
    //    auto spmp_bottom_idx = GET_PARAMETER(int, spmp_bottom_idx);
    //    auto spmp_class_aware = GET_PARAMETER(PTuple<bool>, spmp_class_aware);
    //    auto spmp_label_width = GET_PARAMETER(PTuple<int>, spmp_label_width);
    //    auto spmp_label_height = GET_PARAMETER(PTuple<int>, spmp_label_height);
    //    auto spmp_pad_ratio = GET_PARAMETER(PTuple<float>, spmp_pad_ratio);
    // get cam3d_param
    //    auto cam3d_bottom_idx = GET_PARAMETER(int, cam3d_bottom_idx);
    // get bbox_reg_param
    auto bbox_mean = GET_PARAMETER(PTuple<float>, bbox_mean);
    auto bbox_std = GET_PARAMETER(PTuple<float>, bbox_std);
    saber::NMSSSDParam<Ttype> nms_param(
            overlap_ratio.vector(),
            top_n.vector(),
            max_candidate_n.vector(),
            use_soft_nms.vector(),
            voting.vector(),
            vote_iou.vector(),
            need_nms,
            add_score,
            nms_among_classes,
            nms_gpu_max_n_per_time);
    saber::GenerateAnchorParam<Ttype> gen_anchor_param(
                ratios.vector(),
                scales.vector(),
                anchor_width.vector(),
                anchor_height.vector(),
                anchor_x1.vector(),
                anchor_y1.vector(),
                anchor_x2.vector(),
                anchor_y2.vector(),
                base_size,
                zero_anchor_center
            );
    //TODO!!!!!! enum
    saber::DetectionOutputSSD_MIN_SIZE_MODE min_size_mode_in;

    if (min_size_mode == "HEIGHT_AND_WIDTH") {
        min_size_mode_in = DetectionOutputSSD_HEIGHT_AND_WIDTH;
    } else {
        min_size_mode_in = DetectionOutputSSD_HEIGHT_OR_WIDTH;
    }

    saber::DetectionOutputSSDParam<Ttype> detection_output_ssd_param(
                threshold.vector(),
                class_indexes.vector(),
                heat_map_a.vector(),
                heat_map_b.vector(),
                proposal_min_sqrt_area.vector(),
                proposal_max_sqrt_area.vector(),
                refine_out_of_map_bbox,
                channel_per_scale,
                class_name_list,
                num_class,
                threshold_objectness,
                bg_as_one_of_softmax,
                use_target_type_rcnn,
                im_width,
                im_height,
                rpn_proposal_output_score,
                regress_agnostic,
                allow_border,
                allow_border_ratio,
                bbox_size_add_one,
                read_width_scale,
                read_height_scale,
                read_height_offset,
                min_size_h,
                min_size_w,
                min_size_mode_in
            );
    detection_output_ssd_param.nms_param = nms_param;
    detection_output_ssd_param.nms_param.has_param = true;
    detection_output_ssd_param.gen_anchor_param = gen_anchor_param;
    detection_output_ssd_param.gen_anchor_param.has_param = true;
    saber::BBoxRegParam<Ttype> bbox_reg_param(bbox_mean.vector(), bbox_std.vector());
    bbox_reg_param.has_param = true;
    saber::ProposalParam<Ttype> proposal_param(bbox_reg_param,
            detection_output_ssd_param);
    _param_rpn_prop_ssd = proposal_param;
    return Status::OK();
}
template<typename Ttype, Precision Ptype>
Status RPNProposalSSDHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_rpn_prop_ssd.init(ins, outs, _param_rpn_prop_ssd, SPECIFY, SABER_IMPL, ctx);
    return Status::OK();
}
template<typename Ttype, Precision Ptype>
Status RPNProposalSSDHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_rpn_prop_ssd.compute_output_shape(ins, outs, _param_rpn_prop_ssd);
    return Status::OK();
}
#ifdef USE_CUDA
template class RPNProposalSSDHelper<NV,  Precision::FP32>;
template class RPNProposalSSDHelper<NV,  Precision::FP16>;
template class RPNProposalSSDHelper<NV,  Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class RPNProposalSSDHelper<ARM,  Precision::FP32>;
template class RPNProposalSSDHelper<ARM,  Precision::FP16>;
template class RPNProposalSSDHelper<ARM,  Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(RPNProposalSSD, RPNProposalSSDHelper, NV,  Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(RPNProposalSSD, RPNProposalSSDHelper, ARM,  Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(RPNProposalSSD)
.Doc("RPNProposalSSD operator")
#ifdef USE_CUDA
.__alias__<NV,  Precision::FP32>("rpn_proposal_ssd")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM,  Precision::FP32>("rpn_proposal_ssd")
#endif
.num_in(1)
.num_out(1)
.Args<bool>("need_nms", "need_nms of nms_param")
.Args<PTuple<float>>("overlap_ratio", "overlap_ratio of nms_param")
.Args<PTuple<int>>("top_n", "top_n of nms_param")
.Args<bool>("add_score", "add_score of nms_param")
.Args<PTuple<int>>("max_candidate_n", "max_candidate_n of nms_param")
.Args<PTuple<bool>>("use_soft_nms", "use_soft_nms of nms_param")
.Args<bool>("nms_among_classes", "nms_among_classes of nms_param")
.Args<PTuple<bool>>("voting", "voting of nms_param")
.Args<PTuple<float>>("vote_iou", "vote_iou of nms_param")
.Args<int>("nms_gpu_max_n_per_time", "nms_gpu_max_n_per_time of nms_param")
.Args<PTuple<float>>("threshold", "threshold of detect_output_ssd ")
.Args<int>("channel_per_scale", "channel_per_scale of detect_output_ssd ")
.Args<std::string>("class_name_list", "class_name_list of detect_output_ssd ")
.Args<int>("num_class", "num_class of detect_output_ssd ")
.Args<bool>("refine_out_of_map_bbox", "refine_out_of_map_bbox of detect_output_ssd ")
.Args<PTuple<int>>("class_indexes", "class_indexes of detect_output_ssd ")
.Args<PTuple<float>>("heat_map_a", "heat_map_a of detect_output_ssd ")
.Args<PTuple<float>>("heat_map_b", "heat_map_b of detect_output_ssd ")
.Args<float>("threshold_objectness", "threshold_objectness of detect_output_ssd ")
.Args<PTuple<float>>("proposal_min_sqrt_area", "proposal_min_sqrt_area of detect_output_ssd ")
.Args<PTuple<float>>("proposal_max_sqrt_area", "proposal_max_sqrt_area of detect_output_ssd ")
.Args<bool>("bg_as_one_of_softmax", "bg_as_one_of_softmax of detect_output_ssd ")
.Args<bool>("use_target_type_rcnn", "use_target_type_rcnn of detect_output_ssd ")
.Args<float>("im_width", "im_width of detect_output_ssd ")
.Args<float>("im_height", "im_height of detect_output_ssd ")
.Args<bool>("rpn_proposal_output_score", "rpn_proposal_output_score of detect_output_ssd ")
.Args<bool>("regress_agnostic", "regress_agnostic of detect_output_ssd ")
.Args<float>("allow_border", "allow_border of detect_output_ssd ")
.Args<float>("allow_border_ratio", "allow_border_ratio of detect_output_ssd ")
.Args<bool>("bbox_size_add_one", "bbox_size_add_one of detect_output_ssd ")
.Args<float>("read_width_scale", "read_width_scale of detect_output_ssd ")
.Args<float>("read_height_scale", "read_height_scale of detect_output_ssd ")
.Args<float>("read_height_offset", "read_height_offset of detect_output_ssd ")
.Args<float>("min_size_h", "min_size_h of detect_output_ssd ")
.Args<float>("min_size_w", "min_size_w of detect_output_ssd ")
.Args<std::string>("min_size_mode", "min_size_mode of detect_output_ssd")
.Args<float>("base_size", "base_size of gen_anchor_param")
.Args<PTuple<float>>("ratios", "ratios of gen_anchor_param")
.Args<PTuple<float>>("scales", "scales of gen_anchor_param ")
.Args<PTuple<float>>("anchor_width", "anchor_width of gen_anchor_param ")
.Args<PTuple<float>>("anchor_height", "anchor_height of gen_anchor_param ")
.Args<PTuple<float>>("anchor_x1", "anchor_x1 of gen_anchor_param ")
.Args<PTuple<float>>("anchor_y1", "anchor_y1 of gen_anchor_param ")
.Args<PTuple<float>>("anchor_x2", "anchor_x2 of gen_anchor_param ")
.Args<PTuple<float>>("anchor_y2", "anchor_y2 of gen_anchor_param ")
.Args<bool>("zero_anchor_center", "zero_anchor_center of gen_anchor_param ")
.Args<int>("kpts_exist_bottom_idx", "kpts_exist_bottom_idx of kpts_param ")
.Args<int>("kpts_reg_bottom_idx", "kpts_reg_bottom_idx of kpts_param ")
.Args<bool>("kpts_reg_as_classify", "kpts_reg_as_classify of kpts_param ")
.Args<int>("kpts_classify_width", "kpts_classify_width of kpts_param ")
.Args<int>("kpts_classify_height", "kpts_classify_height of kpts_param ")
.Args<int>("kpts_reg_norm_idx_st", "kpts_reg_norm_idx_st of kpts_param ")
.Args<PTuple<int>>("kpts_st_for_each_class", "kpts_st_for_each_class of kpts_param ")
.Args<PTuple<int>>("kpts_ed_for_each_class", "kpts_ed_for_each_class of kpts_param ")
.Args<float>("kpts_classify_pad_ratio", "kpts_classify_pad_ratio of kpts_param ")
.Args<int>("atrs_reg_bottom_idx", "atrs_reg_bottom_idx of atrs_param")
.Args<int>("atrs_reg_norm_idx_st", "atrs_reg_norm_idx_st of atrs_param")
.Args<PTuple<std::string>>("atrs_norm_type", "atrs_norm_type of atrs_param")
.Args<int>("ftrs_bottom_idx", "ftrs_bottom_idx of ftrs_param")
.Args<int>("spmp_bottom_idx", "spmp_bottom_idx of spmp_param ")
.Args<PTuple<bool>>("spmp_class_aware", "spmp_class_aware of spmp_param ")
.Args<PTuple<int>>("spmp_label_width", "spmp_label_width of spmp_param ")
.Args<PTuple<int>>("spmp_label_height", "spmp_label_height of spmp_param ")
.Args<PTuple<float>>("spmp_pad_ratio", "spmp_pad_ratio of spmp_param ")
.Args<int>("cam3d_bottom_idx", "cam3d_bottom_idx of cam3d_param")
.Args<PTuple<float>>("bbox_mean", "bbox_mean of bbox_reg_param")
.Args<PTuple<float>>("bbox_std", "bbox_std of bbox_reg_param");
} /* namespace ops */
} /* namespace anakin */
