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

#ifndef ANAKIN_SABER_UTILS_H
#define ANAKIN_SABER_UTILS_H

#include "saber/core/common.h"
#include "saber/core/shape.h"
#include "saber/core/context.h"
#include <algorithm>
#include <vector>
#include <cmath>
namespace anakin{

namespace saber{

template <typename Tensor_device, typename Tensor_host>
struct PGlue {
    enum SyncHead {UNINITIALIZED, HEAD_AT_HOST, HEAD_AT_DEVICE, SYNCED };

    PGlue()
            : head(UNINITIALIZED)
            , extern_h_tensor(false)
            , extern_d_tensor(false)
            , d_tensor(nullptr)
            , h_tensor(nullptr)
    {}
    PGlue(Tensor_device *tensor_d, Tensor_host * tensor_h)
            : head(SYNCED)
            , extern_d_tensor(true)
            , extern_h_tensor(true)
            , d_tensor(tensor_d)
            , h_tensor(tensor_h)
    {}
    PGlue(Tensor_device *tensor_d)
            : head(HEAD_AT_DEVICE)
            , extern_d_tensor(true)
            , extern_h_tensor(false)
            , d_tensor(tensor_d)
            , h_tensor(nullptr)
    {}
    PGlue(Tensor_host * tensor_h)
            : head(HEAD_AT_HOST)
            , extern_d_tensor(false)
            , extern_h_tensor(true)
            , d_tensor(nullptr)
            , h_tensor(tensor_h)
    {}
    ~PGlue() {
        if (!extern_h_tensor) {
            delete h_tensor;
        }
        if (!extern_d_tensor) {
            delete d_tensor;
        }
    }
    void set_extern_tensor(Tensor_device *tensor_d) {
        if (!extern_d_tensor) {
            delete d_tensor;
        }
        extern_d_tensor = true;
        d_tensor = tensor_d;
        head = HEAD_AT_DEVICE;
    }
    void set_extern_tensor(Tensor_host *tensor_h) {
        if (!extern_h_tensor) {
            delete h_tensor;
        }
        extern_h_tensor = true;
        h_tensor = tensor_h;
        head = HEAD_AT_HOST;
    }
    void re_alloc(Shape valid_shape, DataType data_type) {
        if (d_tensor == nullptr) {
            d_tensor = new Tensor_device;
        }
        if (h_tensor == nullptr) {
            h_tensor = new Tensor_host;
        }
        d_tensor->re_alloc(valid_shape, data_type);
        h_tensor->re_alloc(valid_shape, data_type);
        head = SYNCED;
    }
    void reshape(Shape valid_shape) {
        if (d_tensor == nullptr) {
            d_tensor = new Tensor_device;
        }
        if (h_tensor == nullptr) {
            h_tensor = new Tensor_host;
        }
        d_tensor->reshape(valid_shape);
        h_tensor->reshape(valid_shape);
    }
    template <typename Context_d>
    const void* host_data(Context_d *ctx) {
        to_host(ctx);
        return (const void*)h_tensor->data();
    }
    template <typename Context_d>
    void* host_mutable_data(Context_d *ctx) {
        to_host(ctx);
        head = HEAD_AT_HOST;
        return (void*)h_tensor->mutable_data();
    }
    template <typename Context_d>
    const void* device_data(Context_d *ctx) {
        to_device(ctx);
        return (const void*)d_tensor->data();
    }
    template <typename Context_d>
    void* device_mutable_data(Context_d *ctx) {
        to_device(ctx);
        head = HEAD_AT_DEVICE;
        return (void*)d_tensor->mutable_data();
    }
    template <typename Context_d>
    void to_host(Context_d *ctx) {
        switch (head) {
            case UNINITIALIZED:
                LOG(FATAL) << "uninitialized glue";
            case HEAD_AT_DEVICE:
                if (h_tensor == nullptr) {
                    h_tensor = new Tensor_host;
                    h_tensor->re_alloc(d_tensor->valid_shape(), d_tensor->get_dtype());
                }
                if (h_tensor->valid_shape() != d_tensor->valid_shape()) {
                    h_tensor->reshape(d_tensor->valid_shape());
                }
                h_tensor->async_copy_from(*d_tensor, ctx->get_compute_stream());
                h_tensor->record_event(ctx->get_compute_stream());
                h_tensor->sync();
//                LOG(INFO) << "copy to host";
                head = SYNCED;
                break;
            case HEAD_AT_HOST:
            case SYNCED:
                break;
        }
    }
    template <typename Context_d>
    void to_device(Context_d *ctx) {
        switch (head) {
            case UNINITIALIZED:
                LOG(FATAL) << "uninitialized glue";
            case HEAD_AT_HOST:
                if (d_tensor == nullptr) {
                    d_tensor = new Tensor_device;
                    d_tensor->re_alloc(h_tensor->valid_shape(), h_tensor->get_dtype());
                }
                if (d_tensor->valid_shape() != h_tensor->valid_shape()) {
                    d_tensor->reshape(h_tensor->valid_shape());
                }
                d_tensor->async_copy_from(*h_tensor, ctx->get_compute_stream());
                d_tensor->record_event(ctx->get_compute_stream());
                d_tensor->sync();
//                        LOG(INFO) << "copy to device";
                head = SYNCED;
                break;
            case HEAD_AT_DEVICE:
            case SYNCED:
                break;
        }
    }
    bool extern_d_tensor{false};
    bool extern_h_tensor{false};
    Tensor_device *d_tensor{nullptr};
    Tensor_host *h_tensor{nullptr};
    SyncHead head;
};

template <typename Dtype>
struct InfoCam3d
{
    InfoCam3d() {
        x = y = z = h = w = l = o = 0;
    }
    Dtype x;
    Dtype y;
    Dtype z;
    Dtype h;
    Dtype w;
    Dtype l;
    Dtype o;
    std::vector<std::vector<Dtype> > pts3d;
    std::vector<std::vector<Dtype> > pts2d;
};

// 0410Rui //
template <typename Dtype>
struct BBox
{
    BBox()
    {
        id = center_h = center_w = score = x1 = x2 = y1 = y2 = 0;
    }
    Dtype score, x1, y1, x2, y2, center_h, center_w, id;

    //0406 New Add By Rui Star//
    Dtype fdlx, fdly, fdrx, fdry, bdrx, bdry, bdlx, bdly;
    Dtype fulx, fuly, furx, fury, burx, bury, bulx, buly;
    Dtype l_3d, w_3d, h_3d;
    Dtype thl, yaw;

    // tracking fea extra //
    int scale_id;
    int heat_map_y;
    int heat_map_x;

    std::vector<Dtype> data;
//    std::vector<cv::Point3f> pts3;

    // added by mingli
    std::vector<Dtype> prbs;
    std::vector<Dtype> ftrs;
    std::vector<Dtype> atrs;
    std::vector<std::pair<Dtype, Dtype> > kpts;
    std::vector<Dtype> kpts_prbs;
    //spatial maps for each instance
    std::vector<std::vector<Dtype> > spmp;
    InfoCam3d<Dtype> cam3d;
    // end mingli

    static bool greater(const BBox<Dtype>& a, const BBox<Dtype>& b){return a.score > b.score;}
};

template <typename Dtype>
void targets2coords(const Dtype tg0, const Dtype tg1, const Dtype tg2, const Dtype tg3,
                    const Dtype acx, const Dtype acy, const Dtype acw, const Dtype ach,
                    const bool use_target_type_rcnn, const bool do_bbox_norm,
                    const std::vector<Dtype>& bbox_means, const std::vector<Dtype>& bbox_stds,
                    Dtype& ltx, Dtype& lty, Dtype& rbx, Dtype& rby, bool bbox_size_add_one) {

    Dtype ntg0 = tg0, ntg1 = tg1, ntg2 = tg2, ntg3 = tg3;
    if (do_bbox_norm) {
        ntg0 *= bbox_stds[0];
        ntg0 += bbox_means[0];
        ntg1 *= bbox_stds[1];
        ntg1 += bbox_means[1];
        ntg2 *= bbox_stds[2];
        ntg2 += bbox_means[2];
        ntg3 *= bbox_stds[3];
        ntg3 += bbox_means[3];
    }
    if (use_target_type_rcnn) {
        Dtype bsz01 = bbox_size_add_one ? Dtype(1.0) : Dtype(0.0);
        Dtype ctx = ntg0 * acw + acx;
        Dtype cty = ntg1 * ach + acy;
        Dtype tw = Dtype(acw * exp(ntg2));
        Dtype th = Dtype(ach * exp(ntg3));
        ltx = Dtype(ctx - 0.5 * (tw - bsz01));
        lty = Dtype(cty - 0.5 * (th - bsz01));
        rbx = Dtype(ltx + tw - bsz01);
        rby = Dtype(lty + th - bsz01);
    } else {
        ltx = ntg0 + acx;
        lty = ntg1 + acy;
        rbx = ntg2 + acx;
        rby = ntg3 + acy;
    }
}
// soft nms, added by mingli
template <typename Dtype>
const std::vector<bool> soft_nms_lm(std::vector< BBox<Dtype> >& candidates,
                               const Dtype iou_std, const int top_N, const int max_candidate_N,
                               bool bbox_size_add_one, bool voting, Dtype vote_iou) {

    Dtype bsz01 = bbox_size_add_one?Dtype(1.0):Dtype(0.0);
    std::stable_sort(candidates.begin(), candidates.end(), BBox<Dtype>::greater);
    std::vector<bool> mask(candidates.size(), false);
    if (mask.size() == 0) {
        return mask;
    }
    int consider_size = candidates.size();
    if (max_candidate_N > 0) {
        consider_size = std::min<int>(consider_size, max_candidate_N);
    }
    std::vector<float> areas(consider_size, 0);
    for (int i = 0; i < consider_size; ++i) {
        areas[i] = (candidates[i].x2 - candidates[i].x1 + bsz01)
                   * (candidates[i].y2- candidates[i].y1 + bsz01);
    }
    int top_n_real = std::min<int>(consider_size, top_N);
    for (int count = 0; count < top_n_real; ++count) {
        int max_box_idx = -1;
        for (int i = 0; i < consider_size; ++i) {
            if (mask[i]) {
                continue;
            }
            if (max_box_idx == -1 || candidates[i].score > candidates[max_box_idx].score) {
                max_box_idx = i;
            }
        }
        CHECK(max_box_idx != -1);
        mask[max_box_idx] = true;
        Dtype s_vt = candidates[max_box_idx].score;
        Dtype x1_vt = 0.0;
        Dtype y1_vt = 0.0;
        Dtype x2_vt = 0.0;
        Dtype y2_vt = 0.0;
        if (voting) {
                    CHECK_GE(s_vt, 0);
            x1_vt = candidates[max_box_idx].x1 * s_vt;
            y1_vt = candidates[max_box_idx].y1 * s_vt;
            x2_vt = candidates[max_box_idx].x2 * s_vt;
            y2_vt = candidates[max_box_idx].y2 * s_vt;
        }
        // suppress the significantly covered bbox
        for (int j = 0; j < consider_size; ++j) {
            if (mask[j]) {
                continue;
            }
            // get intersections
            float xx1 = std::max(candidates[max_box_idx].x1, candidates[j].x1);
            float yy1 = std::max(candidates[max_box_idx].y1, candidates[j].y1);
            float xx2 = std::min(candidates[max_box_idx].x2, candidates[j].x2);
            float yy2 = std::min(candidates[max_box_idx].y2, candidates[j].y2);
            float w = xx2 - xx1 + bsz01;
            float h = yy2 - yy1 + bsz01;
            if (w > 0 && h > 0) {
                // compute overlap
                float o = w * h;
                o = o / (areas[max_box_idx] + areas[j] - o);
                candidates[j].score *= std::exp(-1.0 * o * o / iou_std);
                if (voting && o > vote_iou) {
                    Dtype s_vt_cur = candidates[j].score;
                            CHECK_GE(s_vt_cur, 0);
                    s_vt += s_vt_cur;
                    x1_vt += candidates[j].x1 * s_vt_cur;
                    y1_vt += candidates[j].y1 * s_vt_cur;
                    x2_vt += candidates[j].x2 * s_vt_cur;
                    y2_vt += candidates[j].y2 * s_vt_cur;
                }
            }
        }
        if (voting && s_vt > 0.0001) {
            candidates[max_box_idx].x1 = x1_vt / s_vt;
            candidates[max_box_idx].y1 = y1_vt / s_vt;
            candidates[max_box_idx].x2 = x2_vt / s_vt;
            candidates[max_box_idx].y2 = y2_vt / s_vt;
        }
    }
    std::stable_sort(candidates.begin(),
                     candidates.begin() + consider_size, BBox<Dtype>::greater);
    mask.clear();
    mask.resize(top_n_real, true);
    mask.resize(candidates.size(), false);
    return mask;
}
template <typename Dtype>
const std::vector<bool> nms_lm(std::vector< BBox<Dtype> >& candidates,
                          const Dtype overlap, const int top_N, const bool addScore,
                          const int max_candidate_N, bool bbox_size_add_one, bool voting,
                          Dtype vote_iou) {

    Dtype bsz01 = bbox_size_add_one ? Dtype(1.0) : Dtype(0.0);
    std::stable_sort(candidates.begin(), candidates.end(), BBox<Dtype>::greater);
    std::vector<bool> mask(candidates.size(), false);
    if (mask.size() == 0) {
        return mask;
    }
    int consider_size = candidates.size();
    if (max_candidate_N > 0) {
        consider_size = std::min<int>(consider_size, max_candidate_N);
    }
    std::vector<bool> skip(consider_size, false);
    std::vector<float> areas(consider_size, 0);
    for (int i = 0; i < consider_size; ++i) {
        areas[i] = (candidates[i].x2 - candidates[i].x1 + bsz01)
                   * (candidates[i].y2- candidates[i].y1 + bsz01);
    }
    for (int count = 0, i = 0; count < top_N && i < consider_size; ++i) {
        if (skip[i]) {
            continue;
        }
        mask[i] = true;
        ++count;
        Dtype s_vt = candidates[i].score;
        Dtype x1_vt = 0.0;
        Dtype y1_vt = 0.0;
        Dtype x2_vt = 0.0;
        Dtype y2_vt = 0.0;
        if (voting) {
                    CHECK_GE(s_vt, 0);
            x1_vt = candidates[i].x1 * s_vt;
            y1_vt = candidates[i].y1 * s_vt;
            x2_vt = candidates[i].x2 * s_vt;
            y2_vt = candidates[i].y2 * s_vt;
        }
        // suppress the significantly covered bbox
        for (int j = i + 1; j < consider_size; ++j) {
            if (skip[j]) {
                continue;
            }
            // get intersections
            float xx1 = std::max(candidates[i].x1, candidates[j].x1);
            float yy1 = std::max(candidates[i].y1, candidates[j].y1);
            float xx2 = std::min(candidates[i].x2, candidates[j].x2);
            float yy2 = std::min(candidates[i].y2, candidates[j].y2);
            float w = xx2 - xx1 + bsz01;
            float h = yy2 - yy1 + bsz01;
            if (w > 0 && h > 0) {
                // compute overlap
                //float o = w * h / areas[j];
                float o = w * h;
                o = o / (areas[i] + areas[j] - o);
                if (o > overlap) {
                    skip[j] = true;
                    if (addScore) {
                        candidates[i].score += candidates[j].score;
                    }
                }
                if (voting && o > vote_iou) {
                    Dtype s_vt_cur = candidates[j].score;
                            CHECK_GE(s_vt_cur, 0);
                    s_vt += s_vt_cur;
                    x1_vt += candidates[j].x1 * s_vt_cur;
                    y1_vt += candidates[j].y1 * s_vt_cur;
                    x2_vt += candidates[j].x2 * s_vt_cur;
                    y2_vt += candidates[j].y2 * s_vt_cur;
                }
            }
        }
        if (voting && s_vt > 0.0001) {
            candidates[i].x1 = x1_vt / s_vt;
            candidates[i].y1 = y1_vt / s_vt;
            candidates[i].x2 = x2_vt / s_vt;
            candidates[i].y2 = y2_vt / s_vt;
        }
    }
    return mask;
}

template <typename Dtype>
void coef2dTo3d(Dtype cam_xpz, Dtype cam_xct, Dtype cam_ypz,
                Dtype cam_yct, Dtype cam_pitch, Dtype px, Dtype py,
                Dtype & k1, Dtype & k2, Dtype & u, Dtype & v) {
    k1 = (px - cam_xct) / cam_xpz;
    k2 = (py - cam_yct) / cam_ypz;
    Dtype sin_ = sin(cam_pitch);
    Dtype cos_ = cos(cam_pitch);
    Dtype tmp1 = cam_xpz * k1 * sin_;
    Dtype tmp2 = cam_ypz * (k2 * sin_ + cos_);
    u = sqrt(tmp1 * tmp1 + tmp2 * tmp2);
    v = sin_ * sin_;
}
template <typename Dtype>
void cord2dTo3d(Dtype k1, Dtype k2, Dtype u,
                Dtype v, Dtype ph, Dtype rh,
                Dtype & x, Dtype & y, Dtype & z) {
    Dtype uph = u / ph;
    z = 0.5 * rh * (uph + sqrt(uph * uph + v));
    x = k1 * z;
    y = k2 * z;
}

// caffe util_others.hpp: 185
struct NmsBox{

    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    static bool greater(const NmsBox& a, const NmsBox& b){return a.score > b.score;}
};
const std::vector<bool> nms_voting0(const float *boxes_dev, unsigned long long * mask_dev,
                               int boxes_num, float nms_overlap_thresh,
                               const int max_candidates,
                               const int top_n);
// caffe util_others.hpp:197

#ifdef NVIDIA_GPU
template <typename Dtype>
void rpn_cmp_conf_bbox_gpu(const int num_anchors,
                           const int map_height, const int map_width,
                           const Dtype input_height, const Dtype input_width,
                           const Dtype heat_map_a, const Dtype heat_map_b,
                           const Dtype allow_border, const Dtype allow_border_ratio,
                           const Dtype min_size_w, const Dtype min_size_h,
                           const bool min_size_mode_and_else_or, const Dtype thr_obj,
                           const Dtype bsz01, const bool do_bbox_norm,
                           const Dtype mean0, const Dtype mean1,
                           const Dtype mean2, const Dtype mean3,
                           const Dtype std0, const Dtype std1,
                           const Dtype std2, const Dtype std3,
                           const bool refine_out_of_map_bbox, const Dtype* anc_data,
                           const Dtype* prob_data, const Dtype* tgt_data,
                           Dtype* conf_data, Dtype* bbox_data, Context<NV> *ctx);

template <typename Dtype>
void rcnn_cmp_conf_bbox_gpu(const int num_rois,
                            const Dtype input_height, const Dtype input_width,
                            const Dtype allow_border, const Dtype allow_border_ratio,
                            const Dtype min_size_w, const Dtype min_size_h,
                            const bool min_size_mode_and_else_or, const Dtype thr_obj,
                            const Dtype bsz01, const bool do_bbox_norm,
                            const Dtype mean0, const Dtype mean1,
                            const Dtype mean2, const Dtype mean3,
                            const Dtype std0, const Dtype std1,
                            const Dtype std2, const Dtype std3,
                            const bool refine_out_of_map_bbox, const bool regress_agnostic,
                            const int num_class, const Dtype* thr_cls,
                            const Dtype* rois_data, const Dtype* prob_data,
                            const Dtype* tgt_data, Dtype* conf_data, Dtype* bbox_data,
                            Context<NV> *ctx);

template <typename Dtype, typename PGlue_nv>
void apply_nms_gpu(const Dtype *bbox_data, const Dtype *conf_data,
                   const int num_bboxes, const int bbox_step, const Dtype confidence_threshold,
                   const int max_canditate_n, const int top_k, const Dtype nms_threshold,
                   const Dtype bsz01, std::vector<int> *indices,
                   PGlue_nv *overlapped, PGlue_nv *idx_sm,
                   Context<NV> *ctx, std::vector<int> *idx_ptr = NULL,
                   const int conf_step = 1, const int conf_idx = 0,
                   const int nms_gpu_max_n_per_time = 1000000);
template <typename Dtype>
void GenGrdFt_gpu(unsigned int im_width, unsigned int im_height,
        unsigned int blob_width, unsigned int blob_height,
        Dtype std_height, const std::vector<Dtype> & cam_params,
        Dtype* grd_ft, Dtype read_width_scale = 1.0,
        Dtype read_height_scale = 1.0, unsigned read_height_offset = 0,
        unsigned int valid_param_idx_st = 0, bool trans_cam_pitch_to_zero = false,
        bool normalize_grd_ft = false, unsigned int normalize_grd_ft_dim = 11);

#endif

template <typename Dtype>
void GenGrdFt_cpu(unsigned int im_width, unsigned int im_height,
        unsigned int blob_width, unsigned int blob_height,
        Dtype std_height, const std::vector<Dtype> & cam_params,
        Dtype* grd_ft, Dtype read_width_scale = 1.0,
        Dtype read_height_scale = 1.0, unsigned read_height_offset = 0,
        unsigned int valid_param_idx_st = 0, bool trans_cam_pitch_to_zero = false,
        bool normalize_grd_ft = false, unsigned int normalize_grd_ft_dim = 11);

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_CORE_TYPES_H
