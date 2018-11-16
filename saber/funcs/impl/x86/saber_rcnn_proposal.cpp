#include "saber_rcnn_proposal.h"
#include "saber/core/tensor_op.h"
#include <cfloat>

namespace anakin {
namespace saber {

template <>
SaberStatus SaberRCNNProposal<X86, AK_FLOAT>::create(const std::vector<Tensor<X86>*> &inputs,
                                                    std::vector<Tensor<X86>*> &outputs,
                                                    ProposalParam<X86>& param,
                                                    Context<X86>& ctx) {
    ImplROIOutputSSD<X86, AK_FLOAT>::create(inputs, outputs, param, ctx);

    this->_ctx = &ctx;
    if (inputs.size() == 4) {
        has_img_info_ = true;
    } else {
        has_img_info_ = false;
    }
            CHECK_GT(this->num_class_, 0);
    rois_dim_ = this->rpn_proposal_output_score_ ? (5 + this->num_class_ + 1) : 5;
    return SaberSuccess;
}

template <>
SaberStatus SaberRCNNProposal<X86, AK_FLOAT>::init(const std::vector<Tensor<X86>*> &inputs,
                                                  std::vector<Tensor<X86>*> &outputs,
                                                  ProposalParam<X86>& param,
                                                  Context<X86>& ctx) {
    this->_ctx = &ctx;
    _img_info_data_host_tensor = new Tensor<X86>();
    _probs_st_host_tensor = new Tensor<X86>();
    _cords_st_host_tensor = new Tensor<X86>();
    _rois_st_host_tensor = new Tensor<X86>();
    _outputs_boxes_scores_host_tensor = new Tensor<X86>();
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberRCNNProposal<X86, AK_FLOAT>::dispatch(
        const std::vector<Tensor<X86>*> &inputs,
        std::vector<Tensor<X86>*> &outputs,
        ProposalParam<X86>& param) {

    _img_info_data_host_tensor->reshape(inputs.back()->valid_shape());
    // TODO !!!!
    _probs_st_host_tensor->reshape(inputs[0]->valid_shape());
    _cords_st_host_tensor->reshape(inputs[1]->valid_shape());
    _rois_st_host_tensor->reshape(inputs[2]->valid_shape());
    float im_height = this->im_height_, im_width = this->im_width_;
    float input_height = im_height, input_width = im_width;
    bool is_input_paramid = false;
    std::vector<float> im_width_scale = std::vector<float>(1, this->read_width_scale_);
    std::vector<float> im_height_scale = std::vector<float>(1, this->read_height_scale_);
    std::vector<float> cords_offset_x = std::vector<float>(1, float(0));
    std::vector<float> cords_offset_y = std::vector<float>(1, this->read_height_offset_);
    float min_size_w_cur = this->min_size_w_;
    float min_size_h_cur = this->min_size_h_;

    if (has_img_info_) {
        if (inputs.back()->count(1, inputs.back()->dims()) == 6) {
            // copy gpu data to cpu.
            _img_info_data_host_tensor->copy_from(*inputs.back());
//            _img_info_data_host_tensor->async_copy_from(*inputs.back(), cuda_stream);
//            inputs.back()->record_event(cuda_stream);
//            inputs.back()->sync();
            const float* img_info_data = (const float*)_img_info_data_host_tensor->data();
            im_width = img_info_data[0];
            im_height = img_info_data[1];
            input_width = im_width;
            input_height = im_height;
            im_width_scale.clear();
            im_height_scale.clear();
            cords_offset_x.clear();
            cords_offset_y.clear();

            for (int n = 0; n < inputs.back()->num(); n++) {
                im_width_scale.push_back(img_info_data[n * 6 + 2]);
                im_height_scale.push_back(img_info_data[n * 6 + 3]);
                        CHECK_GT(im_width_scale[n], 0);
                        CHECK_GT(im_height_scale[n], 0);
                cords_offset_x.push_back(img_info_data[n * 6 + 4]);
                cords_offset_y.push_back(img_info_data[n * 6 + 5]);
            }
        } else {
            //            CHECK_GT(inputs.back()->count(), 7);
            //            this->pyramid_image_data_param_.ReadFromSerialized(*(inputs.back()), 0);
            //            im_height = pyramid_image_data_param_.img_h_;
            //            im_width = pyramid_image_data_param_.img_w_;
            //            RectBlockPacking<float>&  block_packer =
            //                    this->pyramid_image_data_param_.rect_block_packer_;
            //            input_width = block_packer.block_width();
            //            input_height = block_packer.block_height();
            //            is_input_paramid = true;
        }
    }

    if (this->refine_out_of_map_bbox_) {
        if (outputs.size() == 0) {
                    CHECK_GT(im_width, 0);
                    CHECK_GT(im_height, 0);
        } else {
                    CHECK_GT(input_width, 0);
                    CHECK_GT(input_height, 0);
        }
    }

    if (this->allow_border_ >= float(0.0)
        || this->allow_border_ratio_ >= float(0.0)) {
                CHECK_GT(input_width, 0);
                CHECK_GT(input_height, 0);
    }

    float bsz01 = this->bbox_size_add_one_ ? float(1.0) : float(0.0);
    const int num_rois = inputs[0]->num();
    const int probs_dim = inputs[0]->channel();
    const int cords_dim = inputs[1]->channel();
    const int pre_rois_dim = inputs[2]->channel();
            CHECK_EQ(num_rois, inputs[1]->num());
            CHECK_EQ(num_rois, inputs[2]->num());
            CHECK_EQ(probs_dim, this->num_class_ + 1);

    if (this->regress_agnostic_) {
                CHECK_EQ(cords_dim, 2 * 4);
    } else {
                CHECK_EQ(cords_dim, (this->num_class_ + 1) * 4);
    }

            CHECK_EQ(pre_rois_dim, 5); // imid, x1, y1, x2, y2
    // copy gpu data to cpu.
//    _probs_st_host_tensor->async_copy_from(*inputs[0], cuda_stream);
//    inputs[0]->record_event(cuda_stream);
//    inputs[0]->sync();
//    _cords_st_host_tensor->async_copy_from(*inputs[1], cuda_stream);
//    inputs[1]->record_event(cuda_stream);
//    inputs[1]->sync();
//    _rois_st_host_tensor->async_copy_from(*inputs[2], cuda_stream);
//    inputs[2]->record_event(cuda_stream);
//    inputs[2]->sync();

    _probs_st_host_tensor->copy_from(*inputs[0]);
    _cords_st_host_tensor->copy_from(*inputs[1]);
    _rois_st_host_tensor->copy_from(*inputs[2]);
    const float* probs_st = (const float*)_probs_st_host_tensor->data();
    const float* cords_st = (const float*)_cords_st_host_tensor->data();
    const float* rois_st = (const float*)_rois_st_host_tensor->data();
    std::vector<std::vector<BBox<float> > > proposal_per_img_vec;

    for (int i = 0; i < num_rois; ++i) {
        const float* probs = probs_st + i * probs_dim;
        const float* cords = cords_st + i * cords_dim;
        const float* rois = rois_st + i * pre_rois_dim;

        // filter those width low probs
        if ((1.0 - probs[0]) < this->threshold_objectness_) {
            continue;
        }

        float score_max = -FLT_MAX;
        int cls_max = -1;

        for (int c = 0; c < this->num_class_; c++) {
            float score_c = probs[c + 1] - this->threshold_[c];

            if (score_c > score_max) {
                score_max = score_c;
                cls_max = c;
            }
        }

        if (score_max < 0) {
            continue;
        }

                CHECK_GE(cls_max, 0);
        int imid = int(rois[0]);
        int cdst = this->regress_agnostic_ ? 4 : (cls_max + 1) * 4;
        BBox<float> bbox;
        bbox.id = imid;
        bbox.score = probs[cls_max + 1];
        float ltx, lty, rbx, rby;
        float rois_w = rois[3] - rois[1] + bsz01;
        float rois_h = rois[4] - rois[2] + bsz01;
        float rois_ctr_x = rois[1] + 0.5 * (rois_w - bsz01);
        float rois_ctr_y = rois[2] + 0.5 * (rois_h - bsz01);

        if (this->allow_border_ >= float(0.0)
            || this->allow_border_ratio_ >= float(0.0)) {
            float x1 = rois[1];
            float y1 = rois[2];
            float x2 = rois[3];
            float y2 = rois[4];

            if (this->allow_border_ >= float(0.0) && (
                    x1 < -this->allow_border_ || y1 < -this->allow_border_
                    || x2 > input_width - 1 + this->allow_border_ ||
                    y2 > input_height - 1 + this->allow_border_)) {
                continue;
            } else if (this->allow_border_ratio_ >= float(0.0)) {
                float x11 = std::max<float>(0, x1);
                float y11 = std::max<float>(0, y1);
                float x22 = std::min<float>(input_width - 1, x2);
                float y22 = std::min<float>(input_height - 1, y2);

                if ((y22 - y11 + bsz01) * (x22 - x11 + bsz01)
                    / ((y2 - y1 + bsz01) * (x2 - x1 + bsz01))
                    < (1.0 - this->allow_border_ratio_)) {
                    continue;
                }
            }
        }

        targets2coords<float>(cords[cdst], cords[cdst + 1],
                              cords[cdst + 2], cords[cdst + 3],
                              rois_ctr_x, rois_ctr_y, rois_w, rois_h,
                              this->use_target_type_rcnn_, this->do_bbox_norm_,
                              this->bbox_means_, this->bbox_stds_,
                              ltx, lty, rbx, rby, this->bbox_size_add_one_);

        if (outputs.size() == 0 && is_input_paramid) {
            //            RectBlockPacking<float>&  block_packer =
            //                    this->pyramid_image_data_param_.rect_block_packer_;
            //            int block_id = this->pyramid_image_data_param_.GetBlockIdBy(imid);
            //            const float heat_map_a = 1;
            //            float cur_map_start_x = 0, cur_map_start_y = 0;
            //            block_packer.GetFeatureMapStartCoordsByBlockId(block_id,
            //                                                           heat_map_a, cur_map_start_y, cur_map_start_x);
            //            float center_buffered_img_w = cur_map_start_x + rois_ctr_x;
            //            float center_buffered_img_h = cur_map_start_y + rois_ctr_y;
            //            int roi_id = block_packer.GetRoiIdByBufferedImgCoords(
            //                    int(center_buffered_img_h), int(center_buffered_img_w));
            //            if (roi_id <= -1) {
            //                continue;
            //            }
            //            block_packer.GetInputImgCoords(roi_id, cur_map_start_y + lty,
            //                                           cur_map_start_x + ltx, bbox.y1, bbox.x1);
            //            block_packer.GetInputImgCoords(roi_id, cur_map_start_y + rby,
            //                                           cur_map_start_x + rbx, bbox.y2, bbox.x2);
        } else {
            bbox.x1 = ltx;
            bbox.y1 = lty;
            bbox.x2 = rbx;
            bbox.y2 = rby;
        }

        if (this->refine_out_of_map_bbox_) {
            if (outputs.size() == 0 && is_input_paramid) {
                //                bbox.x1 = MIN(MAX(bbox.x1, 0), im_width-1);
                //                bbox.y1 = MIN(MAX(bbox.y1, 0), im_height-1);
                //                bbox.x2 = MIN(MAX(bbox.x2, 0), im_width-1);
                //                bbox.y2 = MIN(MAX(bbox.y2, 0), im_height-1);
            } else {
                bbox.x1 = std::min(std::max(bbox.x1, 0.f), input_width - 1);
                bbox.y1 = std::min(std::max(bbox.y1, 0.f), input_height - 1);
                bbox.x2 = std::min(std::max(bbox.x2, 0.f), input_width - 1);
                bbox.y2 = std::min(std::max(bbox.y2, 0.f), input_height - 1);
            }
        }

        float bw = bbox.x2 - bbox.x1 + bsz01;
        float bh = bbox.y2 - bbox.y1 + bsz01;

        if (this->min_size_mode_
            == DetectionOutputSSD_HEIGHT_AND_WIDTH) {
            if (bw < min_size_w_cur || bh < min_size_h_cur) {
                continue;
            }
        } else if (this->min_size_mode_
                   == DetectionOutputSSD_HEIGHT_OR_WIDTH) {
            if (bw < min_size_w_cur && bh < min_size_h_cur) {
                continue;
            }
        } else {
            CHECK(false);
        }

        if (outputs.size() != 0) {
            if (imid + 1 > proposal_per_img_vec.size()) {
                proposal_per_img_vec.resize(imid + 1);
            }

            for (int c = 0; c < this->num_class_ + 1; ++c) {
                bbox.prbs.push_back(probs[c]);
            }

            proposal_per_img_vec[imid].push_back(bbox);
        } else {
            for (int c = 0; c < this->num_class_; ++c) {
                if (probs[c + 1] < this->threshold_[c]) {
                    continue;
                }

                bbox.score = probs[c + 1];
                this->all_candidate_bboxes_[c].push_back(bbox);
            }
        }
    }

    if (outputs.size() != 0) {
        std::vector<std::vector<BBox<float> > > proposal_batch_vec(outputs.size());

        for (int i = 0; i < proposal_per_img_vec.size(); ++i) {
            std::vector<BBox<float> >& proposal_cur = proposal_per_img_vec[i];
            //do nms
            std::vector<bool> sel;

            if (this->nms_use_soft_nms_[0]) {
                sel = soft_nms_lm(proposal_cur, this->nms_overlap_ratio_[0],
                                  this->nms_top_n_[0], this->nms_max_candidate_n_[0],
                                  this->bbox_size_add_one_, this->nms_voting_[0],
                                  this->nms_vote_iou_[0]);
            } else {
                sel = nms_lm(proposal_cur, this->nms_overlap_ratio_[0],
                             this->nms_top_n_[0], false, this->nms_max_candidate_n_[0],
                             this->bbox_size_add_one_, this->nms_voting_[0],
                             this->nms_vote_iou_[0]);
            }

            for (int k = 0; k < sel.size(); k++) {
                if (sel[k]) {
                    float bw = proposal_cur[k].x2 - proposal_cur[k].x1 + bsz01;
                    float bh = proposal_cur[k].y2 - proposal_cur[k].y1 + bsz01;

                    if (bw <= 0 || bh <= 0) {
                        continue;
                    }

                    float bwxh = bw * bh;

                    for (int t = 0; t < outputs.size(); t++) {
                        if (bwxh > this->proposal_min_area_vec_[t]
                            && bwxh < this->proposal_max_area_vec_[t]) {
                            proposal_batch_vec[t].push_back(proposal_cur[k]);
                        }
                    }
                }
            }
        }

        for (int t = 0; t < outputs.size(); t++) {
            if (proposal_batch_vec[t].empty()) {
                // for special case when there is no box
                //                outputs[t]->Reshape(1, rois_dim_, 1, 1);
                //                float* outputs_boxes_scores = outputs[t]->mutable_cpu_data();
                //                caffe_set(outputs[t]->count(), float(0), outputs_boxes_scores);
                Shape output_shape({1, rois_dim_, 1, 1}, Layout_NCHW);
                outputs[t]->reshape(output_shape);
                fill_tensor_const(*outputs[t], 0);
            } else {
                //                const int outputs_num = proposal_batch_vec[t].size();
                //                outputs[t]->Reshape(outputs_num, rois_dim_, 1, 1);
                //                float* outputs_boxes_scores = outputs[t]->mutable_cpu_data();
                const int outputs_num = proposal_batch_vec[t].size();
                Shape output_shape({outputs_num, rois_dim_, 1, 1}, Layout_NCHW);
                outputs[t]->reshape(output_shape);
                // IN THIS CONDITION, THESE reshape IS NECESSARY.
                _outputs_boxes_scores_host_tensor->reshape(outputs[t]->valid_shape());
                //                _outputs_boxes_scores_host_tensor->async_copy_from(*outputs[t], cuda_stream);
                float* outputs_boxes_scores = (float*)_outputs_boxes_scores_host_tensor->mutable_data();

                for (int k = 0; k < outputs_num; k++) {
                    outputs_boxes_scores[k * rois_dim_] = proposal_batch_vec[t][k].id;
                    outputs_boxes_scores[k * rois_dim_ + 1] = proposal_batch_vec[t][k].x1;
                    outputs_boxes_scores[k * rois_dim_ + 2] = proposal_batch_vec[t][k].y1;
                    outputs_boxes_scores[k * rois_dim_ + 3] = proposal_batch_vec[t][k].x2;
                    outputs_boxes_scores[k * rois_dim_ + 4] = proposal_batch_vec[t][k].y2;

                    if (this->rpn_proposal_output_score_) {
                        for (int c = 0; c < this->num_class_ + 1; c++) {
                            outputs_boxes_scores[k * rois_dim_ + 5 + c]
                                    = proposal_batch_vec[t][k].prbs[c];
                        }
                    }
                }

//                outputs[t]->async_copy_from(*_outputs_boxes_scores_host_tensor, cuda_stream);
                outputs[t]->copy_from(*_outputs_boxes_scores_host_tensor);
            }
        }
    } else {
        /* If this is the last forward for thi image, do nms */
        //        if (!is_input_paramid || this->pyramid_image_data_param_.forward_iter_id_ ==
        //                                 (this->pyramid_image_data_param_.forward_times_for_cur_sample_ - 1))
        {
            for (int class_id = 0; class_id < this->num_class_; ++class_id) {
                std::vector<BBox<float> >& cur_box_list = this->all_candidate_bboxes_[class_id];
                std::vector<BBox<float> >& cur_outbox_list = this->output_bboxes_[class_id];

                if (this->nms_use_soft_nms_[class_id]) {
                    this->is_candidate_bbox_selected_ = soft_nms_lm(cur_box_list,
                                                                    this->nms_overlap_ratio_[class_id],
                                                                    this->nms_top_n_[class_id],
                                                                    this->nms_max_candidate_n_[class_id],
                                                                    this->bbox_size_add_one_,
                                                                    this->nms_voting_[class_id],
                                                                    this->nms_vote_iou_[class_id]);
                } else {
                    this->is_candidate_bbox_selected_ = nms_lm(cur_box_list,
                                                               this->nms_overlap_ratio_[class_id],
                                                               this->nms_top_n_[class_id],
                                                               false, this->nms_max_candidate_n_[class_id],
                                                               this->bbox_size_add_one_,
                                                               this->nms_voting_[class_id],
                                                               this->nms_vote_iou_[class_id]);
                }

                cur_outbox_list.clear();

                for (int i = 0; i < this->is_candidate_bbox_selected_.size(); ++i) {
                    if (this->is_candidate_bbox_selected_[i]) {
                        int id = im_width_scale.size() > 1 ? cur_box_list[i].id : 0;
                                CHECK_LT(id, im_width_scale.size());
                        cur_box_list[i].x1 = cur_box_list[i].x1
                                             / im_width_scale[id] + cords_offset_x[id];
                        cur_box_list[i].y1 = cur_box_list[i].y1
                                             / im_height_scale[id] + cords_offset_y[id];
                        cur_box_list[i].x2 = cur_box_list[i].x2
                                             / im_width_scale[id] + cords_offset_x[id];
                        cur_box_list[i].y2 = cur_box_list[i].y2
                                             / im_height_scale[id] + cords_offset_y[id];
                        cur_outbox_list.push_back(cur_box_list[i]);
                    }
                }

                cur_box_list.clear();
            }
        }
    }

    return SaberSuccess;
}
template class SaberRCNNProposal<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberRCNNProposal, ProposalParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberRCNNProposal, ProposalParam, X86, AK_INT8);
}
}