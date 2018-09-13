#! /usr/bin/env python
# Copyright (c) 2017, Cuichaowen. All rights reserved.
# -*- coding: utf-8 -*-

# ops helper dictionary


class Dictionary(object):
    """
    Dictionary for op param which needs to be combined
    """
    def __init__(self):
        self.__dict__ = {}

    def set_attr(self, **kwargs):
        """
        set dict from kwargs
        """
        for key in kwargs.keys():
            if type(kwargs[key]) == type(dict()):
                for key_inner in kwargs[key].keys():
                    self.__dict__[key_inner] = kwargs[key][key_inner]
            else:
                self.__dict__[key] = kwargs[key]
        return self

    def __call__(self):
        """
        call class function to generate dictionary param
        """
        ret = {key: self.__dict__[key] for key in self.__dict__.keys()}
        return ret

########### Object track and detection helper (for adu(caffe layer type)) Op io define #############
# NMSSSDParameter
nms_param = Dictionary().set_attr(need_nms=bool(), 
                                  overlap_ratio=list(), 
                                  top_n=list(), 
                                  add_score=bool(), 
                                  max_candidate_n=list(), 
                                  use_soft_nms=list(), 
                                  nms_among_classes=bool(), 
                                  voting=list(), 
                                  vote_iou=list(),
                                  nms_gpu_max_n_per_time=int())

# BBoxRegParameter
bbox_reg_param = Dictionary().set_attr(bbox_mean=list(), 
                                       bbox_std=list())


# GenerateAnchorParameter
gen_anchor_param = Dictionary().set_attr(base_size=float(), 
                                         ratios=list(), 
                                         scales=list(), 
                                         anchor_width=list(), 
                                         anchor_height=list(), 
                                         anchor_x1=list(), 
                                         anchor_y1=list(), 
                                         anchor_x2=list(), 
                                         anchor_y2=list(), 
                                         zero_anchor_center=bool())

# KPTSParameter
kpts_param = Dictionary().set_attr(kpts_exist_bottom_idx=int(), 
                                   kpts_reg_bottom_idx=int(), 
                                   kpts_reg_as_classify=bool(), 
                                   kpts_classify_width=int(), 
                                   kpts_classify_height=int(), 
                                   kpts_reg_norm_idx_st=int(), 
                                   kpts_st_for_each_class=list(), 
                                   kpts_ed_for_each_class=list(), 
                                   kpts_classify_pad_ratio=float()) 
# ATRSParameter
# enum NormType { 
# 	NONE, 
#	WIDTH,
#	HEIGHT,
# 	WIDTH_LOG,
#	HEIGHT_LOG
# }
atrs_param = Dictionary().set_attr(atrs_reg_bottom_idx=int(), 
                                   atrs_reg_norm_idx_st=int(), 
                                   atrs_norm_type=str())

# FTRSParameter
ftrs_param = Dictionary().set_attr(ftrs_bottom_idx=int())

# SPMPParameter
spmp_param = Dictionary().set_attr(spmp_bottom_idx=int(), 
                                   spmp_class_aware=list(), 
                                   spmp_label_width=list(), 
                                   spmp_label_height=list(), 
                                   spmp_pad_ratio=list())

# Cam3dParameter
cam3d_param = Dictionary().set_attr(cam3d_bottom_idx=int())

# DetectionOutputSSDParameter
# enum MIN_SIZE_MODE {
# 		HEIGHT_AND_WIDTH,
#		HEIGHT_OR_WIDTH
# }
detection_output_ssd_param = Dictionary().set_attr(nms=nms_param(), 
                                                   threshold=list(), 
                                                   channel_per_scale=int(), 
                                                   class_name_list=str(), 
                                                   num_class=int(), 
                                                   refine_out_of_map_bbox=bool(), 
                                                   class_indexes=list(), 
                                                   heat_map_a=list(), 
                                                   heat_map_b=list(), 
                                                   threshold_objectness=float(), 
                                                   proposal_min_sqrt_area=list(), 
                                                   proposal_max_sqrt_area=list(), 
                                                   bg_as_one_of_softmax=bool(), 
                                                   use_target_type_rcnn=bool(), 
                                                   im_width=float(), 
                                                   im_height=float(), 
                                                   rpn_proposal_output_score=bool(), 
                                                   regress_agnostic=bool(), 
                                                   gen_anchor=gen_anchor_param(), 
                                                   allow_border=float(), 
                                                   allow_border_ratio=float(), 
                                                   bbox_size_add_one=bool(), 
                                                   read_width_scale=float(), 
                                                   read_height_scale=float(), 
                                                   read_height_offset=int(), 
                                                   min_size_h=float(), 
                                                   min_size_w=float(), 
                                                   min_size_mode="HEIGHT_AND_WIDTH",
                                                   kpts=kpts_param(), 
                                                   atrs=atrs_param(), 
                                                   ftrs=ftrs_param(), 
                                                   spmp=spmp_param(), 
                                                   cam3d=cam3d_param()) 
# DFMBPSROIPoolingParameter
dfmb_psroi_pooling_param = Dictionary().set_attr(heat_map_a=float(), 
                                                 heat_map_b=float(), 
                                                 pad_ratio=float(), 
                                                 output_dim=int(), 
                                                 trans_std=float(), 
                                                 sample_per_part=int(), 
                                                 group_height=int(), 
                                                 group_width=int(), 
                                                 pooled_height=int(), 
                                                 pooled_width=int(), 
                                                 part_height=int(), 
                                                 part_width=int()) 
# ProposalImgScaleToCamCoordsParameter 
#
# enum NormType { 
#	HEIGHT, 
#	HEIGHT_LOG 
# }
#
# enum OrienType { 
#	PI, 
#	PI2 
# }
proposal_img_scale_to_cam_coords_param = Dictionary().set_attr(num_class=int(), 
                                                               sub_class_num_class=list(), 
                                                               sub_class_bottom_idx=list(), 
                                                               prj_h_norm_type=str(), 
                                                               has_size3d_and_orien3d=bool(), 
                                                               orien_type=str(), 
                                                               cls_ids_zero_size3d_w=list(), 
                                                               cls_ids_zero_size3d_l=list(), 
                                                               cls_ids_zero_orien3d=list(), 
                                                               cmp_pts_corner_3d=bool(), 
                                                               cmp_pts_corner_2d=bool(), 
                                                               ctr_2d_means=list(), 
                                                               ctr_2d_stds=list(), 
                                                               prj_h_means=list(), 
                                                               prj_h_stds=list(), 
                                                               real_h_means=list(), 
                                                               real_h_stds=list(), 
                                                               real_w_means=list(), 
                                                               real_w_stds=list(), 
                                                               real_l_means=list(), 
                                                               real_l_stds=list(), 
                                                               sin_means=list(), 
                                                               sin_stds=list(), 
                                                               cos_means=list(), 
                                                               cos_stds=list(), 
                                                               cam_info_idx_st_in_im_info=int(), 
                                                               im_width_scale=float(), 
                                                               im_height_scale=float(), 
                                                               cords_offset_x=float(), 
                                                               cords_offset_y=float(), 
                                                               bbox_size_add_one=bool(), 
                                                               rotate_coords_by_pitch=bool(), 
                                                               #refine_coords_by_bbox=bool(), 
                                                               #refine_min_dist=float(), 
                                                               #refine_dist_for_height_ratio_one=float(), 
                                                               #max_3d2d_height_ratio_for_min_dist=float(), 
                                                               with_trunc_ratio=bool(),
                                                               regress_ph_rh_as_whole=bool(),
                                                               real_h_means_as_whole=list(),
                                                               real_h_stds_as_whole=list()) 
# RPNProposalSSD parameter
RPNProposalSSD_param = Dictionary().set_attr(detection_output_ssd=detection_output_ssd_param(), 
                                             bbox_reg=bbox_reg_param())
