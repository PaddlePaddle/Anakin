#include "saber_rcnn_proposal.h"
#include "saber/core/tensor_op.h"
#include <cfloat>

namespace anakin {
namespace saber {

template <>
SaberStatus SaberRCNNProposal<NV, AK_FLOAT>::create(const std::vector<Tensor<NV>*> &inputs,
                                                    std::vector<Tensor<NV>*> &outputs,
                                                    ProposalParam<NV>& param,
                                                    Context<NV>& ctx) {
    ImplROIOutputSSD<NV, AK_FLOAT>::create(inputs, outputs, param, ctx);

    CHECK_GT(this->num_class_, 0);

    rois_dim_ = this->rpn_proposal_output_score_ ? (5 + this->num_class_ + 1) : 5;

    Shape thr_cls_shape({1, this->num_class_, 1, 1}, Layout_NCHW);
    thr_cls_.re_alloc(thr_cls_shape, AK_FLOAT);
    float* thr_cls_data = (float*)thr_cls_.host_mutable_data(_ctx);
    for (int c = 0; c < this->num_class_; c++) {
        thr_cls_data[c] = this->threshold_[c];
    }
//    overlapped_.reset(new caffe::SyncedMemory(
//            this->nms_gpu_max_n_per_time_ *
//            this->nms_gpu_max_n_per_time_ * sizeof(bool)));
//    idx_sm_.reset(new caffe::SyncedMemory(
//            this->nms_gpu_max_n_per_time_ * sizeof(int)));

    Shape overlapped_shape({this->nms_gpu_max_n_per_time_ * this->nms_gpu_max_n_per_time_}, Layout_W);
    overlapped_.re_alloc(overlapped_shape, AK_BOOL);
    Shape idx_sm_shape({this->nms_gpu_max_n_per_time_}, Layout_W);
    idx_sm_.re_alloc(idx_sm_shape, AK_INT32);
    return SaberSuccess;
}

template <>
SaberStatus SaberRCNNProposal<NV, AK_FLOAT>::init(const std::vector<Tensor<NV>*> &inputs,
                         std::vector<Tensor<NV>*> &outputs,
                         ProposalParam<NV>& param,
                         Context<NV>& ctx) {
    this->_ctx = &ctx;
//    _img_info_glue = new PGlue<Tensor<NV>, Tensor<NVHX86> >(inputs.back());
//    _probs_st_glue = new PGlue<Tensor<NV>, Tensor<NVHX86> >(inputs[0]);
//    _rois_st_glue = new PGlue<Tensor<NV>, Tensor<NVHX86> >(inputs[2]);
//    _outputs_boxes_scores_glue = new PGlue<Tensor<NV>, Tensor<NVHX86> >;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberRCNNProposal<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV>*> &inputs,
        std::vector<Tensor<NV>*> &outputs,
        ProposalParam<NV>& param) {

    cudaStream_t  cuda_stream = this->_ctx->get_compute_stream();
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

    const float* prob_gpu_data = (const float*)inputs[0]->data();
    const float* tgt_gpu_data = (const float*)inputs[1]->data();
    const float* rois_gpu_data = (const float*)inputs[2]->data();
    Shape dt_conf_shape({num_rois, 1, 1, 1}, Layout_NCHW);
    Shape dt_bbox_shape({num_rois, 4, 1, 1}, Layout_NCHW);
    dt_conf_.re_alloc(dt_conf_shape, AK_FLOAT);
    dt_bbox_.re_alloc(dt_bbox_shape, AK_FLOAT);
    float* conf_gpu_data = (float*)dt_conf_.device_mutable_data(_ctx);
    float* bbox_gpu_data = (float*)dt_bbox_.device_mutable_data(_ctx);

    rcnn_cmp_conf_bbox_gpu(num_rois, input_height, input_width,
            this->allow_border_, this->allow_border_ratio_,
            min_size_w_cur, min_size_h_cur,
            min_size_mode_and_else_or, this->threshold_objectness_,
            bsz01, this->do_bbox_norm_,
            this->bbox_means_[0], this->bbox_means_[1],
            this->bbox_means_[2], this->bbox_means_[3],
            this->bbox_stds_[0], this->bbox_stds_[1],
            this->bbox_stds_[2], this->bbox_stds_[3],
            this->refine_out_of_map_bbox_, this->regress_agnostic_,
            this->num_class_, (const float*)thr_cls_.device_data(_ctx),
            rois_gpu_data, prob_gpu_data, tgt_gpu_data,
            conf_gpu_data, bbox_gpu_data, _ctx);
//    const float* bbox_host_data = (const float*)dt_bbox_.host_data(_ctx);
//    printf("%f,%f,%f,%f,%f\n", bbox_host_data[0],
//           bbox_host_data[1],
//           bbox_host_data[2],
//           bbox_host_data[3],
//           bbox_host_data[4]);
    cudaDeviceSynchronize();
    _probs_st_glue.set_extern_tensor(inputs[0]);
    _rois_st_glue.set_extern_tensor(inputs[2]);
    const float* prob_data = (const float*)_probs_st_glue.host_data(_ctx);
    const float* rois_data = (const float*)_rois_st_glue.host_data(_ctx);
    const float* conf_data = (const float*)dt_conf_.host_data(_ctx);
    const float* bbox_data = (const float*)dt_bbox_.host_data(_ctx);

    // cmp valid idxes per img
    std::vector<std::vector<int> > idx_per_img_vec;
    for (int i = 0; i < num_rois; i++) {
        if (conf_data[i] == float(0.0)) {
            continue;
        }
        int imid = int(rois_data[i * 5]);
        if (imid + 1 > idx_per_img_vec.size()) {
            idx_per_img_vec.resize(imid + 1);
        }
        idx_per_img_vec[imid].push_back(i);
    }

    std::vector<std::vector<BBox<float> > > proposal_per_class(this->num_class_);
    std::vector<std::vector<std::vector<float> > > proposal_batch_vec(outputs.size());
    if (outputs.size() != 0 || this->nms_among_classes_) {
        for (int imid = 0; imid < idx_per_img_vec.size(); imid++) {
            if (idx_per_img_vec[imid].size() == 0) {
                continue;
            }
            std::vector<int> indices;
//            cudaDeviceSynchronize();
            apply_nms_gpu(bbox_gpu_data, conf_data, num_rois, 4,
                          float(0.0), this->nms_max_candidate_n_[0],
                          this->nms_top_n_[0], this->nms_overlap_ratio_[0],
                          bsz01, &indices, &overlapped_, &idx_sm_, _ctx,
                          &idx_per_img_vec[imid], 1, 0,
                          this->nms_gpu_max_n_per_time_);
//            cudaDeviceSynchronize();
            if (outputs.size() == 0) {
                for (int k = 0; k < indices.size(); k++) {
                    BBox<float> bbox;
                    bbox.id = imid;
                    int idk = indices[k];
                    int idkx4 = idk * 4;
                    int imid_cur = im_width_scale.size() > 1 ? imid : 0;
                            CHECK_LT(imid_cur, im_width_scale.size());
                    bbox.x1 = bbox_data[idkx4] / im_width_scale[imid_cur]
                              + cords_offset_x[imid_cur];
                    bbox.y1 = bbox_data[idkx4 + 1] / im_height_scale[imid_cur]
                              + cords_offset_y[imid_cur];
                    bbox.x2 = bbox_data[idkx4 + 2] / im_width_scale[imid_cur]
                              + cords_offset_x[imid_cur];
                    bbox.y2 = bbox_data[idkx4 + 3] / im_height_scale[imid_cur]
                              + cords_offset_y[imid_cur];
                    const float* probs = prob_data + idk * probs_dim;
                    for (int c = 0; c < this->num_class_; ++c) {
                        if (probs[c + 1] < this->threshold_[c]) {
                            continue;
                        }
                        bbox.score = probs[c + 1];
                        proposal_per_class[c].push_back(bbox);
                    }
                }
            } else if (outputs.size() == 1) {
                for (int k = 0; k < indices.size(); k++) {
                    std::vector<float> bbox(6 + probs_dim, 0);
                    bbox[0] = imid;
                    int idk = indices[k];
                    int idkx4 = idk * 4;
                    bbox[1] = conf_data[idk];
                    bbox[2] = bbox_data[idkx4];
                    bbox[3] = bbox_data[idkx4 + 1];
                    bbox[4] = bbox_data[idkx4 + 2];
                    bbox[5] = bbox_data[idkx4 + 3];
                    const float* probs = prob_data + idk * probs_dim;
                    for (int c = 0; c < probs_dim; ++c) {
                        bbox[6 + c] = probs[c];
                    }
                    proposal_batch_vec[0].push_back(bbox);
                }
            } else {
                for (int k = 0; k < indices.size(); k++) {
                    std::vector<float> bbox(6 + probs_dim, 0);
                    bbox[0] = imid;
                    int idk = indices[k];
                    int idkx4 = idk * 4;
                    bbox[1] = conf_data[idk];
                    bbox[2] = bbox_data[idkx4];
                    bbox[3] = bbox_data[idkx4 + 1];
                    bbox[4] = bbox_data[idkx4 + 2];
                    bbox[5] = bbox_data[idkx4 + 3];
                    const float* probs = prob_data + idk * probs_dim;
                    for (int c = 0; c < probs_dim; ++c) {
                        bbox[6 + c] = probs[c];
                    }
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
    } else {
        for (int imid = 0; imid < idx_per_img_vec.size(); imid++) {
            if (idx_per_img_vec[imid].size() == 0) {
                continue;
            }
            for (int c = 0; c < this->num_class_; ++c) {
                std::vector<int> indices;
//                cudaDeviceSynchronize();
                apply_nms_gpu(bbox_gpu_data, prob_data, num_rois, 4,
                              this->threshold_[c], this->nms_max_candidate_n_[c],
                              this->nms_top_n_[c], this->nms_overlap_ratio_[c],
                              bsz01, &indices, &overlapped_, &idx_sm_, _ctx,
                              &idx_per_img_vec[imid], probs_dim, c + 1,
                              this->nms_gpu_max_n_per_time_);
//                cudaDeviceSynchronize();
                for (int k = 0; k < indices.size(); k++) {
                    BBox<float> bbox;
                    bbox.id = imid;
                    int idk = indices[k];
                    int idkx4 = idk * 4;
                    int imid_cur = im_width_scale.size() > 1 ? imid : 0;
                    CHECK_LT(imid_cur, im_width_scale.size());
                    bbox.x1 = bbox_data[idkx4] / im_width_scale[imid_cur]
                              + cords_offset_x[imid_cur];
                    bbox.y1 = bbox_data[idkx4 + 1] / im_height_scale[imid_cur]
                              + cords_offset_y[imid_cur];
                    bbox.x2 = bbox_data[idkx4 + 2] / im_width_scale[imid_cur]
                              + cords_offset_x[imid_cur];
                    bbox.y2 = bbox_data[idkx4 + 3] / im_height_scale[imid_cur]
                              + cords_offset_y[imid_cur];
                    const float* probs = prob_data + idk * probs_dim;
                    bbox.score = probs[c + 1];
                    proposal_per_class[c].push_back(bbox);
                }
            }
        }
    }

    if (outputs.size() != 0) {
        for (int t = 0; t < outputs.size(); t++) {
            _outputs_boxes_scores_glue.set_extern_tensor(outputs[t]);
            if (proposal_batch_vec[t].empty()) {
                // for special case when there is no box
//                outputs[t]->Reshape(1, rois_dim_, 1, 1);
//                float* top_boxes_scores = outputs[t]->mutable_cpu_data();
//                caffe_set(outputs[t]->count(), float(0), top_boxes_scores);
                Shape output_shape({1, rois_dim_, 1, 1}, Layout_NCHW);
                _outputs_boxes_scores_glue.reshape(output_shape);
                fill_tensor_const(*outputs[t], 0.f);
            } else {
//                const int top_num = proposal_batch_vec[t].size();
//                outputs[t]->Reshape(top_num, rois_dim_, 1, 1);
//                float* top_boxes_scores = outputs[t]->mutable_cpu_data();
                const int top_num = proposal_batch_vec[t].size();
                Shape output_shape({top_num, rois_dim_, 1, 1}, Layout_NCHW);
                _outputs_boxes_scores_glue.reshape(output_shape);
                float* top_boxes_scores = (float*)_outputs_boxes_scores_glue.host_mutable_data(_ctx);
                for (int k = 0; k < top_num; k++) {
                    top_boxes_scores[k * rois_dim_] = proposal_batch_vec[t][k][0];
                    top_boxes_scores[k * rois_dim_ + 1] = proposal_batch_vec[t][k][2];
                    top_boxes_scores[k * rois_dim_ + 2] = proposal_batch_vec[t][k][3];
                    top_boxes_scores[k * rois_dim_ + 3] = proposal_batch_vec[t][k][4];
                    top_boxes_scores[k * rois_dim_ + 4] = proposal_batch_vec[t][k][5];
                    if (this->rpn_proposal_output_score_) {
                        for (int c = 0; c < probs_dim; c++) {
                            top_boxes_scores[k * rois_dim_ + 5 + c] =
                                    proposal_batch_vec[t][k][6 + c];
                        }
                    }
                }
            }
            _outputs_boxes_scores_glue.to_device(_ctx);
        }
    } else {
        for (int class_id = 0; class_id < this->num_class_; ++class_id) {
            this->output_bboxes_[class_id] = proposal_per_class[class_id];
        }
    }

    return SaberSuccess;
}
template class SaberRCNNProposal<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberRCNNProposal, ProposalParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberRCNNProposal, ProposalParam, NV, AK_INT8);
}
}