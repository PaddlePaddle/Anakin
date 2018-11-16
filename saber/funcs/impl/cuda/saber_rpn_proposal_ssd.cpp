
#include "saber_rpn_proposal_ssd.h"
#include "saber/core/tensor_op.h"
#include "saber/core/common.h"
#include "saber/core/tensor.h"
#include "saber/saber_funcs_param.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberRPNProposalSSD<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV>*> &inputs,
        std::vector<Tensor<NV>*> &outputs,
        ProposalParam<NV> &param, Context<NV> &ctx) {

    ImplROIOutputSSD<NV, AK_FLOAT>::create(inputs, outputs, param, ctx);
//    CHECK_EQ(this->nms_gpu_max_n_per_time_, 600);
    this->_ctx = &ctx;
    CHECK_EQ(1, this->heat_map_a_vec_.size());
    CHECK_EQ(1, this->heat_map_b_vec_.size());

    if (outputs.size() == 0) {
                CHECK_GT(this->num_class_, 0);
    }

    num_anchors_ = this->anchor_x1_vec_.size();
            CHECK_GE(num_anchors_, 1);

    rois_dim_ = this->rpn_proposal_output_score_ ? 6 : 5;
    Shape anc_shape({num_anchors_, 4, 1, 1}, Layout_NCHW);
    anc_.re_alloc(anc_shape, AK_FLOAT);

    float* anc_data = (float*)anc_.host_mutable_data(_ctx);

    float bsz01 = this->bbox_size_add_one_ ? float(1.0) : float(0.0);
    for (int a = 0; a < num_anchors_; ++a) {
        float anchor_width = this->anchor_x2_vec_[a]
                             - this->anchor_x1_vec_[a] + bsz01;
        float anchor_height = this->anchor_y2_vec_[a]
                              - this->anchor_y1_vec_[a] + bsz01;
        float anchor_ctr_x = this->anchor_x1_vec_[a]
                             + 0.5f * (anchor_width - bsz01);
        float anchor_ctr_y = this->anchor_y1_vec_[a]
                             + 0.5f * (anchor_height - bsz01);
        anc_data[a * 4] = anchor_ctr_x;
        anc_data[a * 4 + 1] = anchor_ctr_y;
        anc_data[a * 4 + 2] = anchor_width;
        anc_data[a * 4 + 3] = anchor_height;
    }
    Shape overlapped_shape({this->nms_gpu_max_n_per_time_ * this->nms_gpu_max_n_per_time_}, Layout_W);
    overlapped_.re_alloc(overlapped_shape, AK_BOOL);
    Shape idx_sm_shape({this->nms_gpu_max_n_per_time_}, Layout_W);
    idx_sm_.re_alloc(idx_sm_shape, AK_INT32);
    return SaberSuccess;
}

template <>
SaberStatus SaberRPNProposalSSD<NV, AK_FLOAT>::init(
        const std::vector<Tensor<NV>*> &inputs,
        std::vector<Tensor<NV>*> &outputs,
        ProposalParam<NV> &param, Context<NV> &ctx) {

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberRPNProposalSSD<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV>*> &inputs,
        std::vector<Tensor<NV>*> &outputs,
        ProposalParam<NV>& param) {

    float input_height = this->im_height_, input_width = this->im_width_;
    float min_size_w_cur = this->min_size_w_;
    float min_size_h_cur = this->min_size_h_;
    std::vector<float> im_width_scale = std::vector<float>(1, this->read_width_scale_);
    std::vector<float> im_height_scale = std::vector<float>(1, this->read_height_scale_);
    std::vector<float> cords_offset_x = std::vector<float>(1, float(0));
    std::vector<float> cords_offset_y = std::vector<float>(1, this->read_height_offset_);
    CHECK_EQ(inputs.back()->count(1, inputs.back()->dims()), 6);
    _img_info_glue.set_extern_tensor(inputs.back());
    const float* img_info_data = (const float*)_img_info_glue.host_data(_ctx);
    input_width = img_info_data[0];
    input_height = img_info_data[1];
    CHECK_GT(input_width, 0);
    CHECK_GT(input_height, 0);
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

    float bsz01 = this->bbox_size_add_one_ ? float(1.0) : float(0.0);

    float min_size_mode_and_else_or = true;
    if (this->min_size_mode_ == DetectionOutputSSD_HEIGHT_OR_WIDTH) {
        min_size_mode_and_else_or = false;
    } else {
        CHECK(this->min_size_mode_ == DetectionOutputSSD_HEIGHT_AND_WIDTH);
    }

    const int num = inputs[0]->num();
    const int map_height = inputs[0]->height();
    const int map_width  = inputs[0]->width();
    const float heat_map_a = this->heat_map_a_vec_[0];
    const float heat_map_b = this->heat_map_b_vec_[0];
            CHECK_EQ(inputs[0]->channel(), num_anchors_ * 2);
            CHECK_EQ(inputs[1]->num(), num);
            CHECK_EQ(inputs[1]->channel(), num_anchors_ * 4);
            CHECK_EQ(inputs[1]->height(), map_height);
            CHECK_EQ(inputs[1]->width(), map_width);

    const float* prob_gpu_data = (const float*)inputs[0]->data();
    const float* tgt_gpu_data = (const float*)inputs[1]->data();

    int num_bboxes = num_anchors_ * map_height * map_width;
    Shape dt_conf_ahw_shape({num_bboxes, 1, 1, 1}, Layout_NCHW);
    Shape dt_bbox_ahw_shape({num_bboxes, 4, 1, 1}, Layout_NCHW);

    dt_conf_ahw_.re_alloc(dt_conf_ahw_shape, AK_FLOAT);
    dt_bbox_ahw_.re_alloc(dt_bbox_ahw_shape, AK_FLOAT);

    std::vector<BBox<float> > proposal_all;
    std::vector<std::vector<std::vector<float> > > proposal_batch_vec(outputs.size());
    for (int i = 0; i < num; ++i) {
        //Timer tm;
        //tm.Start();
        int stride = inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();
//        cudaDeviceSynchronize();
        rpn_cmp_conf_bbox_gpu(num_anchors_,
                              map_height, map_width,
                              input_height, input_width,
                              heat_map_a, heat_map_b,
                              this->allow_border_, this->allow_border_ratio_,
                              min_size_w_cur, min_size_h_cur,
                              min_size_mode_and_else_or, this->threshold_objectness_,
                              bsz01, this->do_bbox_norm_,
                              this->bbox_means_[0], this->bbox_means_[1],
                              this->bbox_means_[2], this->bbox_means_[3],
                              this->bbox_stds_[0], this->bbox_stds_[1],
                              this->bbox_stds_[2], this->bbox_stds_[3],
                              this->refine_out_of_map_bbox_, (const float*)anc_.device_data(_ctx),
                              prob_gpu_data + i * stride,
                              tgt_gpu_data + i * stride,
                              (float*)dt_conf_ahw_.device_mutable_data(_ctx),
                              (float*)dt_bbox_ahw_.device_mutable_data(_ctx),
                              _ctx);
//        cudaDeviceSynchronize();
        //LOG(INFO)<<"nms rpn_cmp_conf_bbox time: "<<tm.MilliSeconds();
        //tm.Start();

        //do nms by gpu
        const float* conf_data = (const float*)dt_conf_ahw_.host_data(_ctx);
        const float* bbox_gpu_data = (const float*)dt_bbox_ahw_.device_data(_ctx);
        std::vector<int> indices;
        apply_nms_gpu(bbox_gpu_data, conf_data, num_bboxes, 4,
                      float(0.0), this->nms_max_candidate_n_[0],
                      this->nms_top_n_[0], this->nms_overlap_ratio_[0],
                      bsz01, &indices, &overlapped_, &idx_sm_, _ctx,
                      NULL, 1, 0, this->nms_gpu_max_n_per_time_);
//        cudaDeviceSynchronize();
        //LOG(INFO)<<"nms apply_nms_gpu time: "<<tm.MilliSeconds();

        const float* bbox_data = (const float*)dt_bbox_ahw_.host_data(_ctx);
        if (outputs.size() == 0) {
            for (int k = 0; k < indices.size(); k++) {
                BBox<float> bbox;
                bbox.id = i;
                int idk = indices[k];
                int idkx4 = idk * 4;
                bbox.score = conf_data[idk];
                int imid_cur = im_width_scale.size() > 1 ? i : 0;
                        CHECK_LT(imid_cur, im_width_scale.size());
                bbox.x1 = bbox_data[idkx4] / im_width_scale[imid_cur]
                          + cords_offset_x[imid_cur];
                bbox.y1 = bbox_data[idkx4 + 1] / im_height_scale[imid_cur]
                          + cords_offset_y[imid_cur];
                bbox.x2 = bbox_data[idkx4 + 2] / im_width_scale[imid_cur]
                          + cords_offset_x[imid_cur];
                bbox.y2 = bbox_data[idkx4 + 3] / im_height_scale[imid_cur]
                          + cords_offset_y[imid_cur];
                proposal_all.push_back(bbox);
            }
        } else if (outputs.size() == 1) {
            for (int k = 0; k < indices.size(); k++) {
                std::vector<float> bbox(6, 0);
                bbox[0] = i;
                int idk = indices[k];
                int idkx4 = idk * 4;
                bbox[1] = conf_data[idk];
                bbox[2] = bbox_data[idkx4];
                bbox[3] = bbox_data[idkx4 + 1];
                bbox[4] = bbox_data[idkx4 + 2];
                bbox[5] = bbox_data[idkx4 + 3];
                proposal_batch_vec[0].push_back(bbox);
            }
        } else {
            for (int k = 0; k < indices.size(); k++) {
                std::vector<float> bbox(6, 0);
                bbox[0] = i;
                int idk = indices[k];
                int idkx4 = idk * 4;
                bbox[1] = conf_data[idk];
                bbox[2] = bbox_data[idkx4];
                bbox[3] = bbox_data[idkx4 + 1];
                bbox[4] = bbox_data[idkx4 + 2];
                bbox[5] = bbox_data[idkx4 + 3];
                float bw = bbox[4] - bbox[2] + bsz01;
                float bh = bbox[5] - bbox[3] + bsz01;
                float bwxh = bw * bh;
                for (int t = 0; t < outputs.size(); t++) {
                    if (bwxh > this->proposal_min_area_vec_[t]
                       && bwxh < this->proposal_max_area_vec_[t]) {
                        proposal_batch_vec[t].push_back(bbox);
                    }
                }
            }
        }
    }

    for (int t = 0; t < outputs.size(); t++) {
        _outputs_boxes_scores_glue.set_extern_tensor(outputs[t]);
        if (proposal_batch_vec[t].empty()) {
            // for special case when there is no box
            Shape output_shape({1, rois_dim_, 1, 1}, Layout_NCHW);
            _outputs_boxes_scores_glue.reshape(output_shape);
            fill_tensor_const(*outputs[t], 0.f);
//            float* top_boxes_scores = (float*)_outputs_boxes_scores_glue->host_mutable_data(_ctx);
//            caffe_set(outputs[t]->count(), float(0), top_boxes_scores);
        } else {
            const int top_num = proposal_batch_vec[t].size();
            Shape output_shape({top_num, rois_dim_, 1, 1}, Layout_NCHW);
            _outputs_boxes_scores_glue.reshape(output_shape);
            float* top_boxes_scores = (float*)_outputs_boxes_scores_glue.host_mutable_data(_ctx);
//            outputs[t]->Reshape(top_num, rois_dim_, 1, 1);
//            float* top_boxes_scores = outputs[t]->mutable_cpu_data();
            for (int k = 0; k < top_num; k++) {
                top_boxes_scores[k*rois_dim_] = proposal_batch_vec[t][k][0];
                top_boxes_scores[k*rois_dim_+1] = proposal_batch_vec[t][k][2];
                top_boxes_scores[k*rois_dim_+2] = proposal_batch_vec[t][k][3];
                top_boxes_scores[k*rois_dim_+3] = proposal_batch_vec[t][k][4];
                top_boxes_scores[k*rois_dim_+4] = proposal_batch_vec[t][k][5];
                if (this->rpn_proposal_output_score_) {
                    top_boxes_scores[k*rois_dim_+5] = proposal_batch_vec[t][k][1];
                }
            }
        }
        _outputs_boxes_scores_glue.to_device(_ctx);
    }

    if (outputs.size() == 0) {
        for (int class_id = 0; class_id < this->num_class_; ++class_id) {
            this->output_bboxes_[class_id] = proposal_all;
        }
    }
    return SaberSuccess;
}
template class SaberRPNProposalSSD<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberRPNProposalSSD, ProposalParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberRPNProposalSSD, ProposalParam, NV, AK_INT8);
}
}
