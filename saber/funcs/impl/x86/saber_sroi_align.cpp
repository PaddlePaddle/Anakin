
#include "saber/funcs/impl/x86/saber_sroi_align.h"
#include <limits>
#include <cmath>

namespace anakin {

namespace saber {

template <>
SaberStatus SaberSRoiAlign<X86, AK_FLOAT>::create(\
        const std::vector<Tensor<X86> *>& inputs, \
        std::vector<Tensor<X86> *>& outputs, \
        SRoiAlignParam<X86>& param,
        Context<X86> &ctx) {
    return SaberSuccess;
}

template <>
SaberStatus SaberSRoiAlign<X86, AK_FLOAT>::init(\
        const std::vector<Tensor<X86> *>& inputs, \
        std::vector<Tensor<X86> *>& outputs, \
        SRoiAlignParam<X86>& param,
        Context<X86> &ctx) {

    this->_ctx = &ctx;

    CHECK_GT(param.pooled_h, 0)
        << "pooled_h must be > 0";
    CHECK_GT(param.pooled_w, 0)
        << "pooled_w must be > 0";
    _pooled_height = param.pooled_h;
    _pooled_width = param.pooled_w;
    _spatial_scale = param.spatial_scale;
    LOG(INFO) << "Spatial scale: " << _spatial_scale;
    _channels = inputs[0]->channel();
    _height = inputs[0]->height();
    _width = inputs[0]->width();
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberSRoiAlign<X86, AK_FLOAT>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs, \
    SRoiAlignParam<X86>& param) {

    const float* bottom_data = (const float*)inputs[0]->data();
    const float* bottom_rois = (const float*)inputs[1]->data();
    // Number of ROIs
    int num_rois = inputs[1]->num();
    int batch_size = inputs[0]->num();
    float* top_data = (float*)outputs[0]->mutable_data();

    int in_0_c = inputs[0]->channel();
    int in_0_h = inputs[0]->height();
    int in_0_w = inputs[0]->width();
    int in_1_c = inputs[1]->channel();
    int in_1_h = inputs[1]->height();
    int in_1_w = inputs[1]->width();
    int out_0_h = outputs[0]->height();
    int out_0_w = outputs[0]->width();
    // For each ROI R = [batch_index x1 y1 x2 y2]: roi align over R
    for (int n = 0; n < num_rois; ++n) {
        int roi_batch_ind = (int)bottom_rois[0];
        float roi_start_w = bottom_rois[1] * _spatial_scale;
        float roi_start_h = bottom_rois[2] * _spatial_scale;
        float roi_end_w = bottom_rois[3] * _spatial_scale;
        float roi_end_h = bottom_rois[4] * _spatial_scale;
        CHECK_GE(roi_batch_ind, 0);
        CHECK_LT(roi_batch_ind, batch_size);

        float roi_height = std::max(roi_end_h - roi_start_h + 1, static_cast<float>(0.));
        float roi_width = std::max(roi_end_w - roi_start_w + 1, static_cast<float>(0.));
        const float bin_size_h = static_cast<float>(roi_height)
                                 / static_cast<float>(_pooled_height - 1.);
        const float bin_size_w = static_cast<float>(roi_width)
                                 / static_cast<float>(_pooled_width - 1.);

        int offset_roi_batch_ind = roi_batch_ind * in_0_c * in_0_h * in_0_w;
        const float* batch_data = bottom_data + offset_roi_batch_ind;

        for (int c = 0; c < _channels; ++c) {
            for (int ph = 0; ph < _pooled_height; ++ph) {
                for (int pw = 0; pw < _pooled_width; ++pw) {
                    float h = static_cast<float>(ph) * bin_size_h + roi_start_h;
                    float w = static_cast<float>(pw) * bin_size_w + roi_start_w;

                    int hstart = std::min(static_cast<int>(floor(h)), _height - 2);
                    int wstart = std::min(static_cast<int>(floor(w)), _width - 2);

                    bool is_empty(h < 0 || h >= _height || w < 0 || w >= _width);
                    const int pool_index = ph * _pooled_width + pw;
                    if (is_empty) {
                        top_data[pool_index] = 0;
                    }
                    else {
                        float h_ratio = h - static_cast<float>(hstart);
                        float w_ratio = w - static_cast<float>(wstart);
                        int upleft = hstart * _width + wstart;
                        int upright = upleft + 1;
                        int downleft = upleft + _width;
                        int downright = downleft + 1;

                        top_data[pool_index] = batch_data[upleft] * (1.f - h_ratio) * (1.f - w_ratio)
                                               + batch_data[upright] * (1.f - h_ratio) * w_ratio
                                               + batch_data[downleft] * h_ratio * (1.f - w_ratio)
                                               + batch_data[downright] * h_ratio * w_ratio;
                    }
                }
            }
            // Increment all data pointers by one channel
//            batch_data += inputs[0]->offset(0, 1);
//            top_data += outputs[0]->offset(0, 1);
            batch_data += in_0_h * in_0_w;
            top_data += out_0_h * out_0_w;
        }
        // Increment ROI data pointer
//        bottom_rois += inputs[1]->offset(1);
        bottom_rois += in_1_c * in_1_h * in_1_w;
    }

    return SaberSuccess;
}

template class SaberSRoiAlign<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSRoiAlign, SRoiAlignParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSRoiAlign, SRoiAlignParam, X86, AK_INT8);

} //namespace saber.
} //namespace anakin.
