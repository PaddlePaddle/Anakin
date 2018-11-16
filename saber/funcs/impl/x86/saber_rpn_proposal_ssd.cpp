#include "saber_rpn_proposal_ssd.h"
#include "saber/core/tensor_op.h"
#include "saber/core/common.h"
#include "saber/core/tensor.h"
#include "saber/saber_funcs_param.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberRPNProposalSSD<X86, AK_FLOAT>::create(
        const std::vector<Tensor<X86>*> &inputs,
        std::vector<Tensor<X86>*> &outputs,
        ProposalParam<X86> &param, Context<X86> &ctx) {

    ImplROIOutputSSD<X86, AK_FLOAT>::create(inputs, outputs, param, ctx);
    this->_ctx = &ctx;
    if (inputs.size() % 2 == 1) {
        has_img_info_ = true;
    } else {
        has_img_info_ = false;
    }
    num_rpns_ = inputs.size() / 2;
    CHECK_EQ(num_rpns_, this->heat_map_a_vec_.size());
    CHECK_EQ(num_rpns_, this->heat_map_b_vec_.size());
    if (outputs.size() == 0) {
        CHECK_GT(this->num_class_, 0);
    }
    num_anchors_ = this->anchor_x1_vec_.size();
    CHECK_GE(num_anchors_, 1);
    rois_dim_ = this->rpn_proposal_output_score_ ? 6 : 5;
    return SaberSuccess;
}

template <>
SaberStatus SaberRPNProposalSSD<X86, AK_FLOAT>::init(
        const std::vector<Tensor<X86>*> &inputs,
        std::vector<Tensor<X86>*> &outputs,
        ProposalParam<X86> &param, Context<X86> &ctx) {

    // ADD CPU TENSORS
    _img_info_data_host_tensor = new Tensor<X86>();
    _prob_data_host_tensor = new Tensor<X86>();
    _tgt_data_host_tensor = new Tensor<X86>();
    _outputs_boxes_scores_host_tensor = new Tensor<X86>();

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberRPNProposalSSD<X86, AK_FLOAT>::dispatch(
        const std::vector<Tensor<X86>*> &inputs,
        std::vector<Tensor<X86>*> &outputs,
        ProposalParam<X86>& param) {

    float im_height = this->im_height_, im_width = this->im_width_;
    float input_height = im_height, input_width = im_width;
    bool is_input_paramid = false;
//    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    std::vector<float> im_width_scale = std::vector<float>(1, this->read_width_scale_);
    std::vector<float> im_height_scale = std::vector<float>(1, this->read_height_scale_);
    std::vector<float> cords_offset_x = std::vector<float>(1, float(0));
    std::vector<float> cords_offset_y = std::vector<float>(1, this->read_height_offset_);
    float min_size_w_cur = this->min_size_w_;
    float min_size_h_cur = this->min_size_h_;

    if (has_img_info_) {
        if (inputs.back()->count(1, inputs.back()->dims()) == 6) {
            _img_info_data_host_tensor->reshape(inputs.back()->valid_shape());
            _img_info_data_host_tensor->copy_from(*inputs.back());
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
            //    CHECK_GT(inputs.back()->count(), 7);
            //            this->pyramid_image_data_param_.ReadFromSerialized(*(inputs.back()), 0);
            //            im_width = pyramid_image_data_param_.img_w_;
            //            im_height = pyramid_image_data_param_.img_h_;
            //            RectBlockPacking<float>&  block_packer =
            //                this->pyramid_image_data_param_.rect_block_packer_;
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
    const int num = inputs[0]->num();
    std::vector<std::vector<BBox<float> > > proposal_batch_vec(outputs.size());

    for (int i = 0; i < num; ++i) {
        std::vector<BBox<float> > proposal_cur;

        for (int r = 0; r < num_rpns_; ++r) {
            int prob_idx = 2 * r;
            int tgt_idx = prob_idx + 1;

            CHECK_EQ(inputs[prob_idx]->num(), num);
            CHECK_EQ(inputs[tgt_idx]->num(), num);
            CHECK_EQ(inputs[prob_idx]->channel(), num_anchors_ * 2);
            CHECK_EQ(inputs[tgt_idx]->channel(), num_anchors_ * 4);
            CHECK_EQ(inputs[prob_idx]->height(), inputs[tgt_idx]->height());
            CHECK_EQ(inputs[prob_idx]->width(), inputs[tgt_idx]->width());

            const int map_height = inputs[prob_idx]->height();
            const int map_width  = inputs[prob_idx]->width();
            const int map_size   = map_height * map_width;
            const float heat_map_a = this->heat_map_a_vec_[r];
            const float heat_map_b = this->heat_map_b_vec_[r];
            float pad_h_map = 0;
            float pad_w_map = 0;
            float cur_map_start_x = 0, cur_map_start_y = 0;

            // These are needed in this situation.
            _prob_data_host_tensor->reshape(inputs[prob_idx]->valid_shape());
            _tgt_data_host_tensor->reshape(inputs[tgt_idx]->valid_shape());

            _prob_data_host_tensor->copy_from(*inputs[prob_idx]);
            _tgt_data_host_tensor->copy_from(*inputs[tgt_idx]);

            const float* prob_data = (const float*)_prob_data_host_tensor->data();
            const float* tgt_data = (const float*)_tgt_data_host_tensor->data();

            for (int a = 0; a < num_anchors_; ++a) {
                int score_channel  = num_anchors_ + a;
                int offset_channel = 4 * a;
                const float* scores = prob_data
                                      + tensor_offset(inputs[prob_idx]->valid_shape(),
                                                      i, score_channel, 0, 0);
                const float* dx1 = tgt_data
                                   + tensor_offset(inputs[tgt_idx]->valid_shape(),
                                                   i, offset_channel + 0, 0, 0);
                const float* dy1 = tgt_data
                                   + tensor_offset(inputs[tgt_idx]->valid_shape(),
                                                   i, offset_channel + 1, 0, 0);
                const float* dx2 = tgt_data
                                   + tensor_offset(inputs[tgt_idx]->valid_shape(),
                                                   i, offset_channel + 2, 0, 0);
                const float* dy2 = tgt_data
                                   + tensor_offset(inputs[tgt_idx]->valid_shape(),
                                                   i, offset_channel + 3, 0, 0);
                float anchor_width = this->anchor_x2_vec_[a]
                                     - this->anchor_x1_vec_[a] + bsz01;
                float anchor_height = this->anchor_y2_vec_[a]
                                      - this->anchor_y1_vec_[a] + bsz01;
                float anchor_ctr_x = this->anchor_x1_vec_[a]
                                     + 0.5f * (anchor_width - bsz01);
                float anchor_ctr_y = this->anchor_y1_vec_[a]
                                     + 0.5f * (anchor_height - bsz01);

                for (int off = 0; off < map_size; ++off) {
                    float score_cur = 0.0f;

                    if (scores[off] < this->threshold_objectness_) {
                        continue;
                    }

                    score_cur = scores[off];

                    int h = off / map_width;
                    int w = off % map_width ;
                    float center_buffered_img_h = 0;
                    float center_buffered_img_w = 0;
                    int roi_id = 0;
                    float input_ctr_x = w * heat_map_a + heat_map_b + anchor_ctr_x;
                    float input_ctr_y = h * heat_map_a + heat_map_b + anchor_ctr_y;

                    if (outputs.size() == 0 && is_input_paramid) {
//                        if (h < pad_h_map || h >= map_height - pad_h_map ||
//                                w < pad_w_map || w >= map_width - pad_w_map) {
//                            continue;
//                        }
//                        RectBlockPacking<float>&  block_packer =
//                            this->pyramid_image_data_param_.rect_block_packer_;
//                        center_buffered_img_h = input_ctr_y + cur_map_start_y * heat_map_a;
//                        center_buffered_img_w = input_ctr_x + cur_map_start_x * heat_map_a;
//                        roi_id = block_packer.GetRoiIdByBufferedImgCoords(
//                                int(center_buffered_img_h), int(center_buffered_img_w));
//                        if (roi_id <= -1) {
//                            continue;
//                        }
                    } else {
                        center_buffered_img_h = input_ctr_y;
                        center_buffered_img_w = input_ctr_x;
                    }

                    if (this->allow_border_ >= float(0.0)
                        || this->allow_border_ratio_ >= float(0.0)) {
                        float x1 = input_ctr_x - 0.5 * (anchor_width - bsz01);
                        float y1 = input_ctr_y - 0.5 * (anchor_height - bsz01);
                        float x2 = x1 + anchor_width - bsz01;
                        float y2 = y1 + anchor_height - bsz01;

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

                    BBox<float> bbox;
                    bbox.id = i;
                    bbox.score = score_cur;
                    float ltx, lty, rbx, rby;

                    targets2coords<float>(dx1[off], dy1[off], dx2[off], dy2[off],
                                          center_buffered_img_w, center_buffered_img_h,
                                          anchor_width, anchor_height, this->use_target_type_rcnn_,
                                          this->do_bbox_norm_, this->bbox_means_, this->bbox_stds_,
                                          ltx, lty, rbx, rby, this->bbox_size_add_one_);

                    if (outputs.size() == 0 && is_input_paramid) {
                        //                        RectBlockPacking<float>&  block_packer =
                        //                            this->pyramid_image_data_param_.rect_block_packer_;
                        //                        block_packer.GetInputImgCoords(roi_id, lty, ltx, bbox.y1, bbox.x1);
                        //                        block_packer.GetInputImgCoords(roi_id, rby, rbx, bbox.y2, bbox.x2);
                    } else {
                        bbox.x1 = ltx;
                        bbox.y1 = lty;
                        bbox.x2 = rbx;
                        bbox.y2 = rby;
                    }

                    if (this->refine_out_of_map_bbox_) {
                        if (outputs.size() == 0 && is_input_paramid) {
                            //                            bbox.x1 = MIN(MAX(bbox.x1, 0), im_width-1);
                            //                            bbox.y1 = MIN(MAX(bbox.y1, 0), im_height-1);
                            //                            bbox.x2 = MIN(MAX(bbox.x2, 0), im_width-1);
                            //                            bbox.y2 = MIN(MAX(bbox.y2, 0), im_height-1);
                        } else {
                            bbox.x1 = std::min(std::max(bbox.x1, 0.f), input_width - 1.f);
                            bbox.y1 = std::min(std::max(bbox.y1, 0.f), input_height - 1.f);
                            bbox.x2 = std::min(std::max(bbox.x2, 0.f), input_width - 1.f);
                            bbox.y2 = std::min(std::max(bbox.y2, 0.f), input_height - 1.f);
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
                        proposal_cur.push_back(bbox);
                    } else {
                        bbox.id = 0;

                        for (int c = 0; c < this->num_class_; ++c) {
                            this->all_candidate_bboxes_[c].push_back(bbox);
                        }
                    }
                }
            }
        }

        if (outputs.size() != 0) {
            // caffe: rpn_proposal_ssd_gpu.cu 317
            std::vector<NmsBox> simplebox(proposal_cur.size());
            //do nms
            std::vector<bool> sel;
            if (this->nms_use_soft_nms_[0]) {
                //Timer tm;
                //tm.Start();
                sel = soft_nms_lm(proposal_cur, this->nms_overlap_ratio_[0],
                                  this->nms_top_n_[0], this->nms_max_candidate_n_[0],
                                  this->bbox_size_add_one_, this->nms_voting_[0],
                                  this->nms_vote_iou_[0]);
                //LOG(INFO)<<"soft-nms time: "<<tm.MilliSeconds();
            } else {
                sel = nms_lm(proposal_cur, this->nms_overlap_ratio_[0],
                             this->nms_top_n_[0], false, this->nms_max_candidate_n_[0],
                             this->bbox_size_add_one_, this->nms_voting_[0],
                             this->nms_vote_iou_[0]);
       // caffe: rpn_proposal_ssd_gpu.cu 371
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
    }

    for (int t = 0; t < outputs.size(); t++) {
        if (proposal_batch_vec[t].empty()) {
            // for special case when there is no box
            Shape output_shape({1, rois_dim_, 1, 1}, Layout_NCHW);
            outputs[t]->reshape(output_shape);

            fill_tensor_const(*outputs[t], 0);
            //            float* outputs_boxes_scores = _outputs_boxes_scores_host_tensor->mutable_data();
            //            caffe_set(outputs[t]->count(), float(0), outputs_boxes_scores);
        } else {
            const int outputs_num = proposal_batch_vec[t].size();
            Shape output_shape({outputs_num, rois_dim_, 1, 1}, Layout_NCHW);
            outputs[t]->reshape(output_shape);

            // copy and reshape output_host_tensor.
            _outputs_boxes_scores_host_tensor->reshape(output_shape);
            //            _outputs_boxes_scores_host_tensor->copy_from(*outputs[t]);
            float* outputs_boxes_scores = (float*)_outputs_boxes_scores_host_tensor->mutable_data();

            for (int k = 0; k < outputs_num; k++) {
                outputs_boxes_scores[k * rois_dim_] = proposal_batch_vec[t][k].id;
                outputs_boxes_scores[k * rois_dim_ + 1] = proposal_batch_vec[t][k].x1;
                outputs_boxes_scores[k * rois_dim_ + 2] = proposal_batch_vec[t][k].y1;
                outputs_boxes_scores[k * rois_dim_ + 3] = proposal_batch_vec[t][k].x2;
                outputs_boxes_scores[k * rois_dim_ + 4] = proposal_batch_vec[t][k].y2;

                if (this->rpn_proposal_output_score_) {
                    outputs_boxes_scores[k * rois_dim_ + 5] = proposal_batch_vec[t][k].score;
                }
            }
            outputs[t]->copy_from(*_outputs_boxes_scores_host_tensor);
            // copy back to gpu.
        }
    }

    if (outputs.size() == 0) {
        /* If this is the last forward for this image, do nms */
        //        if (!is_input_paramid || this->pyramid_image_data_param_.forward_iter_id_ ==
        //                (this->pyramid_image_data_param_.forward_times_for_cur_sample_ - 1))
        for (int class_id = 0; class_id < this->num_class_; ++class_id) {
            std::vector<BBox<float> >& cur_box_list = this->all_candidate_bboxes_[class_id];
            std::vector<BBox<float> >& cur_outbox_list = this->output_bboxes_[class_id];

            if (this->nms_use_soft_nms_[class_id]) {
                this->is_candidate_bbox_selected_ = soft_nms_lm(cur_box_list,
                        this->nms_overlap_ratio_[class_id], this->nms_top_n_[class_id],
                        this->nms_max_candidate_n_[class_id], this->bbox_size_add_one_,
                        this->nms_voting_[class_id], this->nms_vote_iou_[class_id]);
            } else {
                this->is_candidate_bbox_selected_ = nms_lm(cur_box_list,
                        this->nms_overlap_ratio_[class_id], this->nms_top_n_[class_id],
                        false, this->nms_max_candidate_n_[class_id], this->bbox_size_add_one_,
                        this->nms_voting_[class_id], this->nms_vote_iou_[class_id]);
            }

            cur_outbox_list.clear();

            for (int i = 0; i < this->is_candidate_bbox_selected_.size(); ++i) {
                if (this->is_candidate_bbox_selected_[i]) {
                    int id = im_width_scale.size() > 1 ? cur_box_list[i].id : 0;
                            CHECK_LT(id, im_width_scale.size());
                    cur_box_list[i].x1 = cur_box_list[i].x1
                                         / (im_width_scale[id] == 0 ? 1 : im_width_scale[id])
                                         + cords_offset_x[id];
                    cur_box_list[i].y1 = cur_box_list[i].y1
                                         / im_height_scale[id] + cords_offset_y[id];
                    cur_box_list[i].x2 = cur_box_list[i].x2
                                         / (im_width_scale[id] == 0 ? 1 : im_width_scale[id])
                                         + cords_offset_x[id];
                    cur_box_list[i].y2 = cur_box_list[i].y2
                                         / im_height_scale[id] + cords_offset_y[id];
                    cur_outbox_list.push_back(cur_box_list[i]);
                }
            }

            cur_box_list.clear();
        }
    }

    return SaberSuccess;
}
template class SaberRPNProposalSSD<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberRPNProposalSSD, ProposalParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberRPNProposalSSD, ProposalParam, X86, AK_INT8);
}
}
