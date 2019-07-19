/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef ANAKIN_SABER_FUNCS_IMPL_IMPL_ROI_OUTPUT_SSD_H
#define ANAKIN_SABER_FUNCS_IMPL_IMPL_ROI_OUTPUT_SSD_H

#include "anakin_config.h"
#include "saber/core/tensor.h"
#include "saber_funcs_param.h"
#include "saber/utils.h"
#include "saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include <vector>
#include <cfloat>
#include <limits.h>

namespace anakin {

namespace saber {

template<typename TargetType, \
         DataType OpDtype = AK_FLOAT>
class ImplROIOutputSSD : public ImplBase <
    TargetType, OpDtype, \
    ProposalParam <TargetType> >  {
public:
    typedef Tensor <TargetType> OpTensor;
    typedef typename DataTrait<TargetType, OpDtype>::Dtype OpDataType;

    ImplROIOutputSSD()
        : _has_inited(false), refine_out_of_map_bbox_(false), bg_as_one_of_softmax_(false),
          rpn_proposal_output_score_(false), regress_agnostic_(false), atrs_reg_bottom_idx_(-1),
          ftrs_bottom_idx_(-1), spmp_bottom_idx_(-1), num_spmp_(-1), nms_need_nms_(false),
          nms_add_score_(false), num_class_(-1), do_bbox_norm_(false), read_height_offset_(0),
          atrs_reg_norm_idx_st_(-1), has_cam3d_(false), bbox_size_add_one_(false), zero_anchor_center_(false),
          kpts_classify_width_(-1), kpts_do_norm_(false), has_spmp_(false), spmp_dim_sum_(-1),
          cam3d_bottom_idx_(-1), use_target_type_rcnn_(false),
          kpts_reg_as_classify_(false),
          kpts_classify_height_(-1), atrs_do_norm_(false), has_ftrs_(false), nms_among_classes_(false),
          channel_per_scale_(false), has_kpts_(false), kpts_exist_bottom_idx_(-1), kpts_reg_bottom_idx_(-1),
          kpts_reg_norm_idx_st_(-1), has_atrs_(false) {}

    ~ImplROIOutputSSD() {}

    virtual SaberStatus init(const std::vector<OpTensor*> inputs,
                             std::vector<OpTensor*> outputs,
                             ProposalParam <TargetType>& param, Context <TargetType>& ctx) {
        this->_ctx = &ctx;
        create(inputs, outputs, param, ctx);
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<OpTensor*> inputs,
                               std::vector<OpTensor*> outputs,
                               ProposalParam <TargetType>& param, Context <TargetType>& ctx) {
        this->_ctx = &ctx;

        if (_has_inited) {
            return SaberSuccess;
        }

        _has_inited = true;
        DetectionOutputSSDParam<TargetType>& detection_output_ssd_param =
            param.detection_output_ssd_param;
        std::copy(detection_output_ssd_param.threshold.begin(),
                  detection_output_ssd_param.threshold.end(),
                  std::back_inserter(threshold_));
        threshold_objectness_ = detection_output_ssd_param.threshold_objectness;
        nms_need_nms_ = detection_output_ssd_param.nms_param.need_nms;
        nms_among_classes_ = detection_output_ssd_param.nms_param.nms_among_classes;
        std::copy(detection_output_ssd_param.nms_param.overlap_ratio.begin(),
                  detection_output_ssd_param.nms_param.overlap_ratio.end(),
                  std::back_inserter(nms_overlap_ratio_));
        std::copy(detection_output_ssd_param.nms_param.top_n.begin(),
                  detection_output_ssd_param.nms_param.top_n.end(),
                  std::back_inserter(nms_top_n_));
        std::copy(detection_output_ssd_param.nms_param.max_candidate_n.begin(),
                  detection_output_ssd_param.nms_param.max_candidate_n.end(),
                  std::back_inserter(nms_max_candidate_n_));
        std::copy(detection_output_ssd_param.nms_param.use_soft_nms.begin(),
                  detection_output_ssd_param.nms_param.use_soft_nms.end(),
                  std::back_inserter(nms_use_soft_nms_));
        std::copy(detection_output_ssd_param.nms_param.voting.begin(),
                  detection_output_ssd_param.nms_param.voting.end(),
                  std::back_inserter(nms_voting_));
        std::copy(detection_output_ssd_param.nms_param.vote_iou.begin(),
                  detection_output_ssd_param.nms_param.vote_iou.end(),
                  std::back_inserter(nms_vote_iou_));
        nms_add_score_ = detection_output_ssd_param.nms_param.add_score;
        channel_per_scale_ = detection_output_ssd_param.channel_per_scale;
        CHECK_GE(channel_per_scale_, 5) <<
                                        "channel_per_scale_ should be at least 5: (score,dx1,dy1,dx2,dy2)";
        num_class_ = detection_output_ssd_param.num_class;
        CHECK_GT(num_class_, 0);

        if (threshold_.size() == 0) {
            threshold_.resize(num_class_, 0.0);
        } else if (threshold_.size() == 1) {
            threshold_.resize(num_class_, threshold_[0]);
        } else {
            CHECK_EQ(num_class_, threshold_.size());
        }

        if (nms_overlap_ratio_.size() == 0) {
            nms_overlap_ratio_.resize(num_class_, 0.4);
        } else if (nms_overlap_ratio_.size() == 1) {
            nms_overlap_ratio_.resize(num_class_, nms_overlap_ratio_[0]);
        } else {
            CHECK_EQ(num_class_, nms_overlap_ratio_.size());
        }

        if (nms_top_n_.size() == 0) {
            nms_top_n_.resize(num_class_, 200);
        } else if (nms_top_n_.size() == 1) {
            nms_top_n_.resize(num_class_, nms_top_n_[0]);
        } else {
            CHECK_EQ(num_class_, nms_top_n_.size());
        }

        if (nms_use_soft_nms_.size() == 0) {
            nms_use_soft_nms_.resize(num_class_, false);
        } else if (nms_use_soft_nms_.size() == 1) {
            nms_use_soft_nms_.resize(num_class_, nms_use_soft_nms_[0]);
        } else {
            CHECK_EQ(num_class_, nms_use_soft_nms_.size());
        }

        if (nms_voting_.size() == 0) {
            nms_voting_.resize(num_class_, false);
        } else if (nms_voting_.size() == 1) {
            nms_voting_.resize(num_class_, nms_voting_[0]);
        } else {
            CHECK_EQ(num_class_, nms_voting_.size());
        }

        if (nms_vote_iou_.size() == 0) {
            nms_vote_iou_.resize(num_class_, 0.5);
        } else if (nms_vote_iou_.size() == 1) {
            nms_vote_iou_.resize(num_class_, nms_vote_iou_[0]);
        } else {
            CHECK_EQ(num_class_, nms_vote_iou_.size());
        }

        if (nms_max_candidate_n_.size() == 0) {
            nms_max_candidate_n_.resize(num_class_, 3000);
        } else if (nms_max_candidate_n_.size() == 1) {
            nms_max_candidate_n_.resize(num_class_, nms_max_candidate_n_[0]);
        } else {
            CHECK_EQ(num_class_, nms_max_candidate_n_.size());
        }

        nms_gpu_max_n_per_time_ = param.detection_output_ssd_param.nms_param.nms_gpu_max_n_per_time;
        int nms_max_max_candidate_n = 0;
        for (int k = 0; k < nms_max_candidate_n_.size(); k++) {
            if (nms_max_max_candidate_n < nms_max_candidate_n_[k]) {
                nms_max_max_candidate_n = nms_max_candidate_n_[k];
            }
        }
        if (nms_gpu_max_n_per_time_ > nms_max_max_candidate_n ||
            nms_gpu_max_n_per_time_ <= 0) {
            nms_gpu_max_n_per_time_ = nms_max_max_candidate_n;
        }
        for (int k = 0; k < nms_top_n_.size(); k++) {
            CHECK_GE(nms_gpu_max_n_per_time_, nms_top_n_[k]);
        }

        class_names_.clear();
        //    if (num_class_ > 1) {
        //        if (detection_output_ssd_param.class_name_list.size()>0) {
        //            std::string file_list_name = detection_output_ssd_param.class_name_list;
        //            LOG(INFO) << "Opening class list file " << file_list_name;
        //            std::ifstream infile(file_list_name.c_str());
        //            CHECK(infile.good());
        //            std::string class_name;
        //            while (std::getline(infile, class_name)) {
        //                if (class_name.empty()) {
        //                    continue;
        //                }
        //                class_names_.push_back(class_name);
        //            }
        //            infile.close();
        //                    CHECK_EQ(class_names_.size(), num_class_);
        //        } else {
        //            for (int k = 0; k < num_class_; ++k) {
        //                std::ostringstream ss;
        //                ss << "class_" << k << "_";
        //                class_names_.push_back(ss.str());
        //            }
        //        }
        //    } else {
        //        class_names_.push_back("main");
        //    }
        class_names_.push_back("main");
        all_candidate_bboxes_.clear();
        is_candidate_bbox_selected_.clear();
        output_bboxes_.clear();

        for (int class_id = 0; class_id < num_class_; ++class_id) {
            all_candidate_bboxes_.push_back(std::vector<BBox<OpDataType> >());
            output_bboxes_.push_back(std::vector<BBox<OpDataType> >());
        }

        time_get_bbox_ = time_total_ = time_nms_ = 0;
        refine_out_of_map_bbox_ = detection_output_ssd_param.refine_out_of_map_bbox;
        std::copy(detection_output_ssd_param.class_indexes.begin(),
                  detection_output_ssd_param.class_indexes.end(),
                  std::back_inserter(class_indexes_));

        for (int i = 0; i < class_indexes_.size(); i++) {
            CHECK_LT(class_indexes_[i], num_class_);
        }

        std::copy(detection_output_ssd_param.heat_map_a.begin(),
                  detection_output_ssd_param.heat_map_a.end(),
                  std::back_inserter(heat_map_a_vec_));
        std::copy(detection_output_ssd_param.heat_map_b.begin(),
                  detection_output_ssd_param.heat_map_b.end(),
                  std::back_inserter(heat_map_b_vec_));

        if (heat_map_b_vec_.size() == 0) {
            heat_map_b_vec_.resize(heat_map_a_vec_.size(), 0);
        }

        CHECK_EQ(heat_map_a_vec_.size(), heat_map_b_vec_.size());

        for (int i = 0; i < heat_map_a_vec_.size(); i++) {
            CHECK_GT(heat_map_a_vec_[i], 0);
            CHECK_GE(heat_map_b_vec_[i], 0);
        }

        bg_as_one_of_softmax_ = detection_output_ssd_param.bg_as_one_of_softmax;
        allow_border_ = detection_output_ssd_param.allow_border;
        allow_border_ratio_ = detection_output_ssd_param.allow_border_ratio;
        CHECK_GE(OpDataType(1.0), allow_border_ratio_);
        bbox_size_add_one_ = detection_output_ssd_param.bbox_size_add_one;
        OpDataType bsz01 = bbox_size_add_one_ ? OpDataType(1.0) : OpDataType(0.0);
        // gen anchors like faster rcnn
        zero_anchor_center_ = false;

        if (detection_output_ssd_param.gen_anchor_param.has_param) {
            GenerateAnchorParam<TargetType>& gen_anchor_param
                = detection_output_ssd_param.gen_anchor_param;
            OpDataType base_size = gen_anchor_param.base_size;
            std::vector<OpDataType> ratios, scales;
            std::copy(gen_anchor_param.ratios.begin(),
                      gen_anchor_param.ratios.end(), std::back_inserter(ratios));
            std::copy(gen_anchor_param.scales.begin(),
                      gen_anchor_param.scales.end(), std::back_inserter(scales));
            OpDataType w = base_size;
            OpDataType h = base_size;
            OpDataType size = w * h;
            OpDataType x_ctr = zero_anchor_center_ ? 0 : ((w - bsz01) / 2);
            OpDataType y_ctr = zero_anchor_center_ ? 0 : ((h - bsz01) / 2);
            std::vector<std::vector<OpDataType> > ratio_anchors;

            for (int r = 0; r < ratios.size(); r++) {
                OpDataType w_r = round(sqrt(size / ratios[r]));
                OpDataType h_r = round(w_r * ratios[r]);

                for (int s = 0; s < scales.size(); s++) {
                    OpDataType w_r_s = w_r * scales[s];
                    OpDataType h_r_s = h_r * scales[s];
                    OpDataType x1_r_s = x_ctr - (w_r_s - bsz01) / 2.0;
                    OpDataType y1_r_s = y_ctr - (h_r_s - bsz01) / 2.0;
                    OpDataType x2_r_s = x_ctr + (w_r_s - bsz01) / 2.0;
                    OpDataType y2_r_s = y_ctr + (h_r_s - bsz01) / 2.0;
                    anchor_x1_vec_.push_back(x1_r_s);
                    anchor_y1_vec_.push_back(y1_r_s);
                    anchor_x2_vec_.push_back(x2_r_s);
                    anchor_y2_vec_.push_back(y2_r_s);
                }
            }

            // assign pre-computed anchors: [[x1, y1, x2, y2], ...]
            std::copy(gen_anchor_param.anchor_x1.begin(),
                      gen_anchor_param.anchor_x1.end(),
                      std::back_inserter(anchor_x1_vec_));
            std::copy(gen_anchor_param.anchor_y1.begin(),
                      gen_anchor_param.anchor_y1.end(),
                      std::back_inserter(anchor_y1_vec_));
            std::copy(gen_anchor_param.anchor_x2.begin(),
                      gen_anchor_param.anchor_x2.end(),
                      std::back_inserter(anchor_x2_vec_));
            std::copy(gen_anchor_param.anchor_y2.begin(),
                      gen_anchor_param.anchor_y2.end(),
                      std::back_inserter(anchor_y2_vec_));
            CHECK_EQ(anchor_x1_vec_.size(), anchor_y1_vec_.size());
            CHECK_EQ(anchor_x1_vec_.size(), anchor_x2_vec_.size());
            CHECK_EQ(anchor_x1_vec_.size(), anchor_y2_vec_.size());
            // assign pre-computed centered-zero anchors: [[w, h], ...]
            std::vector<OpDataType> anchor_width_vec_cur, anchor_height_vec_cur;
            std::copy(gen_anchor_param.anchor_width.begin(),
                      gen_anchor_param.anchor_width.end(),
                      std::back_inserter(anchor_width_vec_cur));
            std::copy(gen_anchor_param.anchor_height.begin(),
                      gen_anchor_param.anchor_height.end(),
                      std::back_inserter(anchor_height_vec_cur));
            CHECK_EQ(anchor_height_vec_cur.size(), anchor_width_vec_cur.size());
            zero_anchor_center_ = gen_anchor_param.zero_anchor_center;

            for (int i = 0; i < anchor_height_vec_cur.size(); i++) {
                if (zero_anchor_center_) {
                    anchor_x1_vec_.push_back((anchor_width_vec_cur[i] - bsz01) / OpDataType(-2.0));
                    anchor_y1_vec_.push_back((anchor_height_vec_cur[i] - bsz01) / OpDataType(-2.0));
                    anchor_x2_vec_.push_back((anchor_width_vec_cur[i] - bsz01) / OpDataType(2.0));
                    anchor_y2_vec_.push_back((anchor_height_vec_cur[i] - bsz01) / OpDataType(2.0));
                } else {
                    anchor_x1_vec_.push_back(0.0);
                    anchor_y1_vec_.push_back(0.0);
                    anchor_x2_vec_.push_back(anchor_width_vec_cur[i] - bsz01);
                    anchor_y2_vec_.push_back(anchor_height_vec_cur[i] - bsz01);
                }
            }

            //cmp anchor_height and anchor_width
            for (int i = 0; i < anchor_x1_vec_.size(); i++) {
                anchor_height_vec_.push_back(anchor_y2_vec_[i] - anchor_y1_vec_[i] + bsz01);
                anchor_width_vec_.push_back(anchor_x2_vec_[i] - anchor_x1_vec_[i] + bsz01);
                CHECK_GT(anchor_height_vec_.back(), 0);
                CHECK_GT(anchor_width_vec_.back(), 0);
            }
        }

        use_target_type_rcnn_ = detection_output_ssd_param.use_target_type_rcnn;
        // bbox mean and std
        do_bbox_norm_ = false;
        bbox_means_.clear();
        bbox_stds_.clear();

        if (param.bbox_reg_param.bbox_mean.size() > 0
                && param.bbox_reg_param.bbox_std.size() > 0) {
            do_bbox_norm_ = true;
            int num_bbox_means = param.bbox_reg_param.bbox_mean.size();
            int num_bbox_stds = param.bbox_reg_param.bbox_std.size();
            CHECK_EQ(num_bbox_means, 4);
            CHECK_EQ(num_bbox_stds, 4);

            for (int i = 0; i < 4; i++) {
                bbox_means_.push_back(param.bbox_reg_param.bbox_mean[i]);
                bbox_stds_.push_back(param.bbox_reg_param.bbox_std[i]);
            }
        }

        std::copy(detection_output_ssd_param.proposal_min_sqrt_area.begin(),
                  detection_output_ssd_param.proposal_min_sqrt_area.end(),
                  std::back_inserter(proposal_min_area_vec_));
        std::copy(detection_output_ssd_param.proposal_max_sqrt_area.begin(),
                  detection_output_ssd_param.proposal_max_sqrt_area.end(),
                  std::back_inserter(proposal_max_area_vec_));
        int proposal_area_vec_size = std::max<int>(1, outputs.size());

        if (proposal_min_area_vec_.size() == 0) {
            proposal_min_area_vec_.resize(proposal_area_vec_size, 0);
        } else if (proposal_min_area_vec_.size() == 1) {
            proposal_min_area_vec_[0] = proposal_min_area_vec_[0] * proposal_min_area_vec_[0];
            proposal_min_area_vec_.resize(proposal_area_vec_size, proposal_min_area_vec_[0]);
        } else {
            CHECK_EQ(proposal_min_area_vec_.size(), proposal_area_vec_size);

            for (int i = 0; i < proposal_area_vec_size; i++) {
                proposal_min_area_vec_[i] = proposal_min_area_vec_[i] * proposal_min_area_vec_[i];
            }
        }

        if (proposal_max_area_vec_.size() == 0) {
            proposal_max_area_vec_.resize(proposal_area_vec_size, FLT_MAX);
        } else if (proposal_max_area_vec_.size() == 1) {
            proposal_max_area_vec_[0] = proposal_max_area_vec_[0] * proposal_max_area_vec_[0];
            proposal_max_area_vec_.resize(proposal_area_vec_size, proposal_max_area_vec_[0]);
        } else {
            CHECK_EQ(proposal_max_area_vec_.size(), proposal_area_vec_size);

            for (int i = 0; i < proposal_area_vec_size; i++) {
                proposal_max_area_vec_[i] = proposal_max_area_vec_[i] * proposal_max_area_vec_[i];
            }
        }

        im_width_ = detection_output_ssd_param.im_width;
        im_height_ = detection_output_ssd_param.im_height;
        rpn_proposal_output_score_ = detection_output_ssd_param.rpn_proposal_output_score;
        regress_agnostic_ = detection_output_ssd_param.regress_agnostic;
        read_width_scale_ = detection_output_ssd_param.read_width_scale;
        read_height_scale_ = detection_output_ssd_param.read_height_scale;
        read_height_offset_ = detection_output_ssd_param.read_height_offset;
        CHECK_GT(read_width_scale_, 0);
        CHECK_GT(read_height_scale_, 0);
        CHECK_GE(read_height_offset_, 0);
        min_size_h_ = detection_output_ssd_param.min_size_h;
        min_size_w_ = detection_output_ssd_param.min_size_w;
        min_size_mode_ = detection_output_ssd_param.min_size_mode;
        reg_means_.clear();
        reg_stds_.clear();

        if (param.reg_param.mean.size() > 0
                && param.reg_param.std.size() > 0) {
            std::copy(param.reg_param.mean.begin(),
                      param.reg_param.mean.end(),
                      std::back_inserter(reg_means_));
            std::copy(param.reg_param.std.begin(),
                      param.reg_param.std.end(),
                      std::back_inserter(reg_stds_));
        }

        CHECK_EQ(reg_means_.size(), reg_stds_.size());
        // kpts param
        has_kpts_ = false;
        kpts_do_norm_ = false;

        if (detection_output_ssd_param.kpts_param.has_param) {
            has_kpts_ = true;
            kpts_exist_bottom_idx_ = detection_output_ssd_param.kpts_param.kpts_exist_bottom_idx;
            kpts_reg_bottom_idx_ = detection_output_ssd_param.kpts_param.kpts_reg_bottom_idx;
            CHECK_LT(kpts_exist_bottom_idx_, inputs.size());
            CHECK_LT(kpts_reg_bottom_idx_, inputs.size());
            kpts_reg_as_classify_ = detection_output_ssd_param.kpts_param.kpts_reg_as_classify;

            if (kpts_reg_as_classify_) {
                CHECK(detection_output_ssd_param.kpts_param.kpts_classify_width > 0);
                CHECK(detection_output_ssd_param.kpts_param.kpts_classify_height > 0);
                kpts_classify_width_ = detection_output_ssd_param.kpts_param.kpts_classify_width;
                kpts_classify_height_ = detection_output_ssd_param.kpts_param.kpts_classify_height;
                CHECK_GE(kpts_classify_width_, 2);
                CHECK_GE(kpts_classify_height_, 2);
                kpts_classify_pad_ratio_ = detection_output_ssd_param.kpts_param.kpts_classify_pad_ratio;
                CHECK_GE(kpts_classify_pad_ratio_, 0.0);
            } else {
                if (detection_output_ssd_param.kpts_param.kpts_reg_norm_idx_st > 0) {
                    kpts_reg_norm_idx_st_ = detection_output_ssd_param.kpts_param.kpts_reg_norm_idx_st;

                    if (kpts_reg_norm_idx_st_ >= 0) {
                        kpts_do_norm_ = true;
                    }
                }
            }

            std::copy(detection_output_ssd_param.kpts_param.kpts_st_for_each_class.begin(),
                      detection_output_ssd_param.kpts_param.kpts_st_for_each_class.end(),
                      std::back_inserter(kpts_st_for_each_class_));
            std::copy(detection_output_ssd_param.kpts_param.kpts_ed_for_each_class.begin(),
                      detection_output_ssd_param.kpts_param.kpts_ed_for_each_class.end(),
                      std::back_inserter(kpts_ed_for_each_class_));

            if (kpts_st_for_each_class_.size() == 0) {
                kpts_st_for_each_class_.resize(num_class_, 0);
            } else {
                CHECK_EQ(num_class_, kpts_st_for_each_class_.size());
            }

            if (kpts_ed_for_each_class_.size() == 0) {
                kpts_ed_for_each_class_.resize(num_class_, INT_MAX);
            } else {
                CHECK_EQ(num_class_, kpts_ed_for_each_class_.size());
            }
        }

        has_atrs_ = false;
        atrs_do_norm_ = false;

        if (detection_output_ssd_param.atrs_param.has_param) {
            has_atrs_ = true;
            atrs_reg_bottom_idx_ = detection_output_ssd_param.atrs_param.atrs_reg_bottom_idx;
            CHECK_LT(atrs_reg_bottom_idx_, inputs.size());

            if (detection_output_ssd_param.atrs_param.atrs_reg_norm_idx_st > 0) {
                atrs_reg_norm_idx_st_ = detection_output_ssd_param.atrs_param.atrs_reg_norm_idx_st;

                if (atrs_reg_norm_idx_st_ >= 0) {
                    atrs_do_norm_ = true;
                }
            }

            for (int i = 0; i < detection_output_ssd_param.atrs_param.atrs_norm_type.size(); i++) {
                atrs_norm_type_.push_back(detection_output_ssd_param.atrs_param.atrs_norm_type[i]);
            }
        }

        has_ftrs_ = false;

        if (detection_output_ssd_param.ftrs_param.has_param) {
            has_ftrs_ = true;
            ftrs_bottom_idx_ = detection_output_ssd_param.ftrs_param.ftrs_bottom_idx;
            CHECK_LT(ftrs_bottom_idx_, inputs.size());
        }

        has_spmp_ = false;

        if (detection_output_ssd_param.spmp_param.has_param) {
            has_spmp_ = true;
            spmp_bottom_idx_ = detection_output_ssd_param.spmp_param.spmp_bottom_idx;
            CHECK_LT(spmp_bottom_idx_, inputs.size());
            std::copy(detection_output_ssd_param.spmp_param.spmp_class_aware.begin(),
                      detection_output_ssd_param.spmp_param.spmp_class_aware.end(),
                      std::back_inserter(spmp_class_aware_));
            std::copy(detection_output_ssd_param.spmp_param.spmp_label_width.begin(),
                      detection_output_ssd_param.spmp_param.spmp_label_width.end(),
                      std::back_inserter(spmp_label_width_));
            std::copy(detection_output_ssd_param.spmp_param.spmp_label_height.begin(),
                      detection_output_ssd_param.spmp_param.spmp_label_height.end(),
                      std::back_inserter(spmp_label_height_));
            std::copy(detection_output_ssd_param.spmp_param.spmp_pad_ratio.begin(),
                      detection_output_ssd_param.spmp_param.spmp_pad_ratio.end(),
                      std::back_inserter(spmp_pad_ratio_));
            num_spmp_ = spmp_class_aware_.size();
            CHECK_GT(num_spmp_, 0);
            CHECK_EQ(num_spmp_, spmp_label_width_.size());
            CHECK_EQ(num_spmp_, spmp_label_height_.size());
            CHECK_EQ(num_spmp_, spmp_pad_ratio_.size());
            spmp_dim_st_.push_back(0);
            spmp_dim_.push_back((spmp_class_aware_[0] ? num_class_ : 1) * spmp_label_width_[0] *
                                spmp_label_height_[0]);
            CHECK_GT(spmp_label_width_[0], 0);
            CHECK_GT(spmp_label_height_[0], 0);

            for (int p = 1; p < num_spmp_; p++) {
                spmp_dim_st_.push_back(spmp_dim_st_.back() + spmp_dim_.back());
                spmp_dim_.push_back((spmp_class_aware_[p] ? num_class_ : 1) * spmp_label_width_[p] *
                                    spmp_label_height_[p]);
                CHECK_GT(spmp_label_width_[p], 0);
                CHECK_GT(spmp_label_height_[p], 0);
            }

            spmp_dim_sum_ = spmp_dim_st_.back() + spmp_dim_.back();
        }

        has_cam3d_ = false;

        if (detection_output_ssd_param.cam3d_param.has_param) {
            has_cam3d_ = true;
            cam3d_bottom_idx_ = detection_output_ssd_param.cam3d_param.cam3d_bottom_idx;
            CHECK_LT(cam3d_bottom_idx_, inputs.size());
        }

        return SaberSuccess;
    };

    //call cudnnConvolutionForward here
    virtual SaberStatus dispatch(const std::vector<OpTensor*> inputs,
                                 std::vector<OpTensor*> outputs,
                                 ProposalParam <TargetType>& param) {
        return SaberUnImplError;
    }

    inline std::vector <BBox<OpDataType>>& GetFilteredBBox(int class_id) {
        return output_bboxes_[class_id];
    }

    inline std::vector <std::vector<BBox < OpDataType>> >&

    GetFilteredBBox() {
        return output_bboxes_;
    }

    inline int GetNumClass() {
        return num_class_;
    }

    inline std::vector <std::string> GetClassNames() {
        return class_names_;
    }

    inline int tensor_offset(Shape valid_shape,
                             const int n,
                             const int c = 0,
                             const int h = 0,
                             const int w = 0) const {
        return (((n * valid_shape[1] + c) * valid_shape[2] + h) * valid_shape[3] + w);
    }

protected:
    std::vector <std::vector<BBox < OpDataType>> > all_candidate_bboxes_;
    std::vector<bool> is_candidate_bbox_selected_;
    std::vector <std::vector<BBox < OpDataType>> > output_bboxes_;
    //int bbox_data_size, bbox_info_size;
    std::vector <OpDataType> threshold_;
    bool nms_need_nms_;
    std::vector <OpDataType> nms_overlap_ratio_;
    std::vector<int> nms_top_n_;
    // added by liming
    int nms_gpu_max_n_per_time_;
    std::vector<int> nms_max_candidate_n_;
    std::vector<bool> nms_use_soft_nms_;
    OpDataType threshold_objectness_;
    bool nms_among_classes_;
    std::vector<bool> nms_voting_;
    std::vector <OpDataType> nms_vote_iou_;
    bool nms_add_score_;
    bool refine_out_of_map_bbox_;
    int channel_per_scale_;
    int num_class_;
    std::vector <std::string> class_names_;
    std::vector<int> class_indexes_;
    std::vector <OpDataType> heat_map_a_vec_;
    std::vector <OpDataType> heat_map_b_vec_;
    std::vector <OpDataType> anchor_width_vec_;
    std::vector <OpDataType> anchor_height_vec_;
    std::vector <OpDataType> anchor_x1_vec_;
    std::vector <OpDataType> anchor_y1_vec_;
    std::vector <OpDataType> anchor_x2_vec_;
    std::vector <OpDataType> anchor_y2_vec_;
    std::vector <OpDataType> proposal_min_area_vec_;
    std::vector <OpDataType> proposal_max_area_vec_;
    bool bg_as_one_of_softmax_;
    bool use_target_type_rcnn_;
    bool do_bbox_norm_;
    std::vector <OpDataType> bbox_means_;
    std::vector <OpDataType> bbox_stds_;
    OpDataType im_width_;
    OpDataType im_height_;
    bool rpn_proposal_output_score_;
    bool regress_agnostic_;
    OpDataType time_get_bbox_, time_total_, time_nms_, time_bbox_to_blob_;
    OpDataType allow_border_;
    OpDataType allow_border_ratio_;
    bool bbox_size_add_one_;
    OpDataType read_width_scale_;
    OpDataType read_height_scale_;
    unsigned int read_height_offset_;
    bool zero_anchor_center_;
    OpDataType min_size_h_;
    OpDataType min_size_w_;
    DetectionOutputSSD_MIN_SIZE_MODE min_size_mode_;
    std::vector <OpDataType> reg_means_;
    std::vector <OpDataType> reg_stds_;
    //kpts params
    bool has_kpts_;
    bool kpts_reg_as_classify_;
    int kpts_exist_bottom_idx_;
    int kpts_reg_bottom_idx_;
    int kpts_classify_width_;
    int kpts_classify_height_;
    bool kpts_do_norm_;
    int kpts_reg_norm_idx_st_;
    std::vector<int> kpts_st_for_each_class_;
    std::vector<int> kpts_ed_for_each_class_;
    OpDataType kpts_classify_pad_ratio_;
    //atrs params
    bool has_atrs_;
    int atrs_reg_bottom_idx_;
    bool atrs_do_norm_;
    int atrs_reg_norm_idx_st_;
    std::vector <ATRS_NormType> atrs_norm_type_;
    //ftrs params
    bool has_ftrs_;
    int ftrs_bottom_idx_;
    //spmp params
    bool has_spmp_;
    int spmp_bottom_idx_;
    int num_spmp_;
    std::vector<bool> spmp_class_aware_;
    std::vector<int> spmp_label_width_;
    std::vector<int> spmp_label_height_;
    std::vector <OpDataType> spmp_pad_ratio_;
    std::vector<int> spmp_dim_st_;
    std::vector<int> spmp_dim_;
    int spmp_dim_sum_;
    //cam3d params
    bool has_cam3d_;
    int cam3d_bottom_idx_;
    bool _has_inited;
};

}
}
#endif //SABER_FUNCS_IMPL_IMPL_ROI_OUTPUT_SSD_H
