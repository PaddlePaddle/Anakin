#include "saber_rois_anchor_feature.h"
#include "saber/core/tensor_op.h"

#include <queue>
#include <cmath>
namespace anakin {
namespace saber {

template <>
SaberStatus SaberRoisAnchorFeature<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV>*> &inputs,
        std::vector<Tensor<NV>*> &outputs,
        RoisAnchorFeatureParam<NV>& param,
        Context<NV>& ctx) {
    this->_ctx = &ctx;

    if (_has_inited) {
        return SaberSuccess;
    }

    _has_inited = true;

    const RoisAnchorFeatureParam<NV>& rois_anchor_param = param;

    float min_anchor_size = rois_anchor_param.min_anchor_size;
    CHECK_GT(min_anchor_size, float(0));
    int num_anchor_scales = rois_anchor_param.num_anchor_scales;
    CHECK_GT(num_anchor_scales, 0);
    float anchor_scale_pow_base = rois_anchor_param.anchor_scale_pow_base;
    CHECK_GT(anchor_scale_pow_base, 1.0);
    std::vector<float> anchor_wph_ratios;
    std::copy(rois_anchor_param.anchor_wph_ratios.begin(),
              rois_anchor_param.anchor_wph_ratios.end(),
              std::back_inserter(anchor_wph_ratios));
    CHECK_GT(anchor_wph_ratios.size(), 0);
    num_top_iou_anchor_ = rois_anchor_param.num_top_iou_anchor;
    CHECK_GT(num_top_iou_anchor_, 0);
    min_num_top_iou_anchor_ = rois_anchor_param.min_num_top_iou_anchor;
    CHECK_GE(num_top_iou_anchor_, min_num_top_iou_anchor_);
    iou_thr_ = rois_anchor_param.iou_thr;
    CHECK(iou_thr_ >= 0.0 && iou_thr_ <= 1.0);
    num_anchors_ = num_anchor_scales * anchor_wph_ratios.size();
    for (int s = 0; s < num_anchor_scales; s++) {
        float sz = min_anchor_size * std::pow<float>(anchor_scale_pow_base, s);
        for (int r = 0; r < anchor_wph_ratios.size(); r++) {
            float rt = anchor_wph_ratios[r];
            CHECK_GT(rt, float(0));
            float sqrt_rt = std::sqrt(rt);
            float anc_h = sz / sqrt_rt;
            float anc_w = sz * sqrt_rt;
            anchor_height_.push_back(anc_h);
            anchor_width_.push_back(anc_w);
            anchor_area_.push_back(anc_h * anc_w);
        }
    }
    ft_ratio_h_ = rois_anchor_param.ft_ratio_h;
    ft_ratio_w_ = rois_anchor_param.ft_ratio_w;
    ft_log_ratio_h_ = rois_anchor_param.ft_log_ratio_h;
    ft_log_ratio_w_ = rois_anchor_param.ft_log_ratio_w;
    num_ft_per_anchor_ = ft_ratio_h_ + ft_ratio_w_ + ft_log_ratio_h_ + ft_log_ratio_w_;
    CHECK_GT(num_ft_per_anchor_, 0);

    bbox_size_add_one_ = rois_anchor_param.bbox_size_add_one;

    return SaberSuccess;
}

template <>
SaberStatus SaberRoisAnchorFeature<NV, AK_FLOAT>::init(const std::vector<Tensor<NV>*> &inputs,
                 std::vector<Tensor<NV>*> &outputs,
                 RoisAnchorFeatureParam<NV>& param,
                 Context<NV>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberRoisAnchorFeature<NV, AK_FLOAT>::dispatch(const std::vector<Tensor<NV>*> &inputs,
                 std::vector<Tensor<NV>*> &outputs,
                 RoisAnchorFeatureParam<NV>& param) {

    bottom.reshape(inputs[0]->valid_shape());
    bottom.async_copy_from(*inputs[0], _ctx->get_compute_stream());
    bottom.record_event(_ctx->get_compute_stream());
    top.reshape(outputs[0]->valid_shape());
    fill_tensor_const(top, 0);
    bottom.sync();

    const float* bottom_data = (const float*)bottom.data();
    float* top_data = (float*)top.mutable_data();

    int num_rois = inputs[0]->num();
    int rois_dim = inputs[0]->count_valid(1, inputs[0]->dims());
    int ft_dim = num_anchors_ * num_ft_per_anchor_;
    float bsz01 = bbox_size_add_one_ ? float(1.0) : float(0.0);
    for (int n = 0; n < num_rois; n++) {
        float roi_x1 = bottom_data[n * rois_dim + 1];
        float roi_y1 = bottom_data[n * rois_dim + 2];
        float roi_x2 = bottom_data[n * rois_dim + 3];
        float roi_y2 = bottom_data[n * rois_dim + 4];
        float roi_w = roi_x2 - roi_x1 + bsz01;
        float roi_h = roi_y2 - roi_y1 + bsz01;
        if (roi_w < 1.0 || roi_h < 1.0) {
            continue;
        }
        float roi_s = roi_w * roi_h;
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
                std::greater<std::pair<float, int> > > anc_with_iou_top_n;
        for (int a = 0; a < num_anchors_; a++) {
            float anc_w = anchor_width_[a];
            float anc_h = anchor_height_[a];
            float anc_s  = anchor_area_[a];
            float o = std::min(roi_w, anc_w) * std::min(roi_h, anc_h);
            float iou = o / (roi_s + anc_s - o);
            if (iou < 0.3) {
                continue;
            }
            anc_with_iou_top_n.push(std::pair<float, int>(iou, a));
            if (anc_with_iou_top_n.size() > num_top_iou_anchor_) {
                anc_with_iou_top_n.pop();
            }
        }
        while (!anc_with_iou_top_n.empty()) {
            std::pair<float, int> anc_with_iou = anc_with_iou_top_n.top();
            anc_with_iou_top_n.pop();
            if (anc_with_iou_top_n.size() >= min_num_top_iou_anchor_
                && anc_with_iou.first < iou_thr_) {
                continue;
            }
            int a = anc_with_iou.second;
            float anc_w = anchor_width_[a];
            float anc_h = anchor_height_[a];
            int st = n * ft_dim + a * num_ft_per_anchor_;
            if (ft_ratio_h_) {
                float ratio_h = anc_h / roi_h;
                top_data[st++] = ratio_h;
            }
            if (ft_ratio_w_) {
                float ratio_w = anc_w / roi_w;
                top_data[st++] = ratio_w;
            }
            if (ft_log_ratio_h_) {
                float log_ratio_h = log(anc_h / roi_h);
                top_data[st++] = log_ratio_h;
            }
            if (ft_log_ratio_w_) {
                float log_ratio_w = log(anc_w / roi_w);
                top_data[st++] = log_ratio_w;
            }
        }
    }
    outputs[0]->async_copy_from(top, _ctx->get_compute_stream());
    return SaberSuccess;
}

template class SaberRoisAnchorFeature<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberRoisAnchorFeature, RoisAnchorFeatureParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberRoisAnchorFeature, RoisAnchorFeatureParam, NV, AK_INT8);

} // namespace saber
} // namespace anakin