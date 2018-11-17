#include "framework/operators/proposal_img_scale_to_cam_coords.h"
namespace anakin {
namespace ops {
#ifdef USE_CUDA
template<>
void ProposalImgScaleToCamCoords<NV, Precision::FP32>::operator()(
        OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV> >& ins,
        std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<ProposalImgScaleToCamCoordsHelper<NV, Precision::FP32>*>
        (this->_helper);

    auto& param = static_cast<ProposalImgScaleToCamCoordsHelper<NV, Precision::FP32>*>
        (this->_helper)->_param_proposal_img_scale_to_cam_coords;
    impl->_funcs_proposal_img_scale_to_cam_coords(ins, outs, param, ctx);
}
#endif
/// TODO ... specialization other type of operator
/// set helper
template<typename Ttype, Precision Ptype>
ProposalImgScaleToCamCoordsHelper<Ttype, Ptype>::~ProposalImgScaleToCamCoordsHelper() {}

template<typename Ttype, Precision Ptype>
Status ProposalImgScaleToCamCoordsHelper<Ttype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing ProposalImgScaleToCamCoords op parameter.";
    // get proposal_img_scale_to_cam_coords_param
    auto num_class = GET_PARAMETER(int, num_class);
    auto sub_class_num_class = GET_PARAMETER(PTuple<int>, sub_class_num_class);
    auto sub_class_bottom_idx = GET_PARAMETER(PTuple<int>, sub_class_bottom_idx);
    auto prj_h_norm_type = GET_PARAMETER(std::string, prj_h_norm_type);
    auto has_size3d_and_orien3d = GET_PARAMETER(bool, has_size3d_and_orien3d);
    auto orien_type = GET_PARAMETER(std::string, orien_type);
    auto cls_ids_zero_size3d_w = GET_PARAMETER(PTuple<int>, cls_ids_zero_size3d_w);
    auto cls_ids_zero_size3d_l = GET_PARAMETER(PTuple<int>, cls_ids_zero_size3d_l);
    auto cls_ids_zero_orien3d = GET_PARAMETER(PTuple<int>, cls_ids_zero_orien3d);
    auto cmp_pts_corner_3d = GET_PARAMETER(bool, cmp_pts_corner_3d);
    auto cmp_pts_corner_2d = GET_PARAMETER(bool, cmp_pts_corner_2d);
    auto ctr_2d_means = GET_PARAMETER(PTuple<float>, ctr_2d_means);
    auto ctr_2d_stds = GET_PARAMETER(PTuple<float>, ctr_2d_stds);
    auto prj_h_means = GET_PARAMETER(PTuple<float>, prj_h_means);
    auto prj_h_stds = GET_PARAMETER(PTuple<float>, prj_h_stds);
    auto real_h_means = GET_PARAMETER(PTuple<float>, real_h_means);
    auto real_h_stds = GET_PARAMETER(PTuple<float>, real_h_stds);
    auto real_w_means = GET_PARAMETER(PTuple<float>, real_w_means);
    auto real_w_stds = GET_PARAMETER(PTuple<float>, real_w_stds);
    auto real_l_means = GET_PARAMETER(PTuple<float>, real_l_means);
    auto real_l_stds = GET_PARAMETER(PTuple<float>, real_l_stds);
    auto sin_means = GET_PARAMETER(PTuple<float>, sin_means);
    auto sin_stds = GET_PARAMETER(PTuple<float>, sin_stds);
    auto cos_means = GET_PARAMETER(PTuple<float>, cos_means);
    auto cos_stds = GET_PARAMETER(PTuple<float>, cos_stds);
    auto real_h_means_as_whole = GET_PARAMETER(PTuple<float>, real_h_means_as_whole);
    auto real_h_stds_as_whole = GET_PARAMETER(PTuple<float>, real_h_stds_as_whole);
    auto cam_info_idx_st_in_im_info = GET_PARAMETER(int, cam_info_idx_st_in_im_info);
    auto im_width_scale = GET_PARAMETER(float, im_width_scale);
    auto im_height_scale = GET_PARAMETER(float, im_height_scale);
    auto cords_offset_x = GET_PARAMETER(float, cords_offset_x);
    auto cords_offset_y = GET_PARAMETER(float, cords_offset_y);
    auto bbox_size_add_one = GET_PARAMETER(bool, bbox_size_add_one);
    auto rotate_coords_by_pitch = GET_PARAMETER(bool, rotate_coords_by_pitch);
    auto regress_ph_rh_as_whole = GET_PARAMETER(bool, regress_ph_rh_as_whole);
//    auto refine_coords_by_bbox = GET_PARAMETER(bool, refine_coords_by_bbox);
    // update new
//    auto refine_min_dist = GET_PARAMETER(float, refine_min_dist);
//    auto refine_dist_for_height_ratio_one = GET_PARAMETER(float, refine_dist_for_height_ratio_one);
//    auto max_3d2d_height_ratio_for_min_dist = GET_PARAMETER(float, max_3d2d_height_ratio_for_min_dist);
    auto with_trunc_ratio = GET_PARAMETER(bool, with_trunc_ratio);
    //TODO!!!! enum!!
    ProposalImgScaleToCamCoords_NormType prj_h_norm_type_in;

    if (prj_h_norm_type == "HEIGHT_LOG") {
        prj_h_norm_type_in = ProposalImgScaleToCamCoords_NormType_HEIGHT_LOG;
    } else {
        prj_h_norm_type_in = ProposalImgScaleToCamCoords_NormType_HEIGHT;
    }

    ProposalImgScaleToCamCoords_OrienType orien_type_in;

    if (orien_type == "PI2") {
        orien_type_in = ProposalImgScaleToCamCoords_OrienType_PI2;
    } else {
        orien_type_in = ProposalImgScaleToCamCoords_OrienType_PI;
    }

    saber::ProposalImgScaleToCamCoordsParam<Ttype> proposal_img_param(
            num_class,
            sub_class_num_class.vector(),
            sub_class_bottom_idx.vector(),
            cls_ids_zero_size3d_w.vector(),
            cls_ids_zero_size3d_l.vector(),
            cls_ids_zero_orien3d.vector(),
            ctr_2d_means.vector(),
            ctr_2d_stds.vector(),
            prj_h_means.vector(),
            prj_h_stds.vector(),
            real_h_means.vector(),
            real_h_stds.vector(),
            real_w_means.vector(),
            real_w_stds.vector(),
            real_l_means.vector(),
            real_l_stds.vector(),
            sin_means.vector(),
            sin_stds.vector(),
            cos_means.vector(),
            cos_stds.vector(),
            real_h_means_as_whole.vector(),
            real_h_stds_as_whole.vector(),
            prj_h_norm_type_in,
            has_size3d_and_orien3d,
            orien_type_in,
            cmp_pts_corner_3d,
            cmp_pts_corner_2d,
            cam_info_idx_st_in_im_info,
            im_width_scale,
            im_height_scale,
            cords_offset_x,
            cords_offset_y,
            bbox_size_add_one,
            rotate_coords_by_pitch,
            regress_ph_rh_as_whole,
            with_trunc_ratio
//            refine_coords_by_bbox,
//            refine_min_dist,
//            refine_dist_for_height_ratio_one,
//            max_3d2d_height_ratio_for_min_dist,

    );
    _param_proposal_img_scale_to_cam_coords = proposal_img_param;
    return Status::OK();
}
template<typename Ttype, Precision Ptype>
Status ProposalImgScaleToCamCoordsHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    _funcs_proposal_img_scale_to_cam_coords.init(ins, outs,
            _param_proposal_img_scale_to_cam_coords, SPECIFY, SABER_IMPL, ctx);

    return Status::OK();
}
template<typename Ttype,  Precision Ptype>
Status ProposalImgScaleToCamCoordsHelper<Ttype, Ptype>::InferShape(
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    _funcs_proposal_img_scale_to_cam_coords.compute_output_shape(ins, outs,
            _param_proposal_img_scale_to_cam_coords);
    return Status::OK();
}
#ifdef USE_CUDA
template class ProposalImgScaleToCamCoordsHelper<NV, Precision::FP32>;
template class ProposalImgScaleToCamCoordsHelper<NV, Precision::FP16>;
template class ProposalImgScaleToCamCoordsHelper<NV, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class ProposalImgScaleToCamCoordsHelper<ARM, Precision::FP32>;
template class ProposalImgScaleToCamCoordsHelper<ARM, Precision::FP16>;
template class ProposalImgScaleToCamCoordsHelper<ARM, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(ProposalImgScaleToCamCoords,
        ProposalImgScaleToCamCoordsHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(ProposalImgScaleToCamCoords,
        ProposalImgScaleToCamCoordsHelper, ARM, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(ProposalImgScaleToCamCoords)
.Doc("ProposalImgScaleToCamCoords operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("proposal_img_scal_to_cam_coords")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("proposal_img_scal_to_cam_coords")
#endif
.num_in(1)
.num_out(1)
.Args<int>("num_class", "num_class of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<int>>("sub_class_num_class",
        "sub_class_num_class of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<int>>("sub_class_bottom_idx",
        "sub_class_bottom_idx of proposal_img_scale_to_cam_coords_param")
.Args<std::string>("prj_h_norm_type", "prj_h_norm_type of proposal_img_scale_to_cam_coords_param")
.Args<bool>("has_size3d_and_orien3d",
        "has_size3d_and_orien3d of proposal_img_scale_to_cam_coords_param")
.Args<std::string>("orien_type", "orien_type of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<int>>("cls_ids_zero_size3d_w",
        "cls_ids_zero_size3d_w of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<int>>("cls_ids_zero_size3d_l",
        "cls_ids_zero_size3d_l of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<int>>("cls_ids_zero_orien3d",
        "cls_ids_zero_orien3d of proposal_img_scale_to_cam_coords_param")
.Args<bool>("cmp_pts_corner_3d", "cmp_pts_corner_3d of proposal_img_scale_to_cam_coords_param")
.Args<bool>("cmp_pts_corner_2d", "cmp_pts_corner_2d of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("ctr_2d_means", "ctr_2d_means of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("ctr_2d_stds", "ctr_2d_stds of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("prj_h_means", "prj_h_means of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("prj_h_stds", "prj_h_stds of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("real_h_means", "real_h_means of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("real_h_stds", "real_h_stds of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("real_w_means", "real_w_means of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("real_w_stds", "real_w_stds of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("real_l_means", "real_l_means of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("real_l_stds", "real_l_stds of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("sin_means", "sin_means of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("sin_stds", "sin_stds of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("cos_means", "cos_means of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("cos_stds", "cos_stds of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("real_h_means_as_whole",
        "real_h_means_as_whole of proposal_img_scale_to_cam_coords_param")
.Args<PTuple<float>>("real_h_stds_as_whole",
        "real_h_stds_as_whole of proposal_img_scale_to_cam_coords_param")
.Args<int>("cam_info_idx_st_in_im_info",
        "cam_info_idx_st_in_im_info of proposal_img_scale_to_cam_coords_param")
.Args<float>("im_width_scale", "im_width_scale of proposal_img_scale_to_cam_coords_param")
.Args<float>("im_height_scale", "im_height_scale of proposal_img_scale_to_cam_coords_param")
.Args<float>("cords_offset_x", "cords_offset_x of proposal_img_scale_to_cam_coords_param")
.Args<float>("cords_offset_y", "cords_offset_y of proposal_img_scale_to_cam_coords_param")
.Args<bool>("bbox_size_add_one", "bbox_size_add_one of proposal_img_scale_to_cam_coords_param")
.Args<bool>("rotate_coords_by_pitch",
        "rotate_coords_by_pitch of proposal_img_scale_to_cam_coords_param")
.Args<bool>("with_trunc_ratio", "with_trunc_ratio of proposal_img_scale_to_cam_coords_param")
.Args<bool>("regress_ph_rh_as_whole", "regress_ph_rh_as_whole of proposal_img_scale_to_cam_coords_param");
} /* namespace ops */
} /* namespace anakin */