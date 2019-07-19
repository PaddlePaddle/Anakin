#include "saber/funcs/impl/arm/saber_sroi_align.h"

namespace anakin{

namespace saber{
template <>
SaberStatus SaberSRoiAlign<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        SRoiAlignParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    // Number of ROIs
    int num_rois = inputs[1]->num();
    int batch_size = inputs[0]->num();
    int top_count = outputs[0]->valid_size();

    int in_0_c = inputs[0]->channel();
    int in_0_h = inputs[0]->height();
    int in_0_w = inputs[0]->width();
    int in_1_c = inputs[1]->channel();
    int in_1_h = inputs[1]->height();
    int in_1_w = inputs[1]->width();
    int out_0_h = outputs[0]->height();
    int out_0_w = outputs[0]->width();

    const float* bottom_data = (const float*)inputs[0]->data();
    const float* bottom_rois = (const float*)inputs[1]->data();
    float* top_data = (float*)outputs[0]->mutable_data();

    // For each ROI R = [batch_index x1 y1 x2 y2]: roi align over R
#pragma omp parallel for
    for (int n = 0; n < num_rois; ++n) {
        int roi_batch_ind = (int)bottom_rois[0];
        float roi_start_w = bottom_rois[1] * _spatial_scale;
        float roi_start_h = bottom_rois[2] * _spatial_scale;
        float roi_end_w = bottom_rois[3] * _spatial_scale;
        float roi_end_h = bottom_rois[4] * _spatial_scale;
        CHECK_GE(roi_batch_ind, 0) << "roi_batch_ind must be >= 0 \n";
        CHECK_LT(roi_batch_ind, batch_size) << "roi_batch_ind must be < batch_size \n";
        float roi_height = std::max(roi_end_h - roi_start_h + 1, static_cast<float>(0.));
        float roi_width = std::max(roi_end_w - roi_start_w + 1, static_cast<float>(0.));
        const float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(_pooled_height - 1.);
        const float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(_pooled_width - 1.);

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
                    }else {
                        float h_ratio = h - static_cast<float>(hstart);
                        float w_ratio = w - static_cast<float>(wstart);
                        int upleft = hstart * _width + wstart;
                        int upright = upleft + 1;
                        int downleft = upleft + _width;
                        int downright = downleft + 1;
                        top_data[pool_index] = batch_data[upleft] * (1. - h_ratio) * (1. - w_ratio) \
                                                + batch_data[upright] * (1. - h_ratio) * w_ratio \
                                                + batch_data[downleft] * h_ratio * (1. - w_ratio) \
                                                + batch_data[downright] * h_ratio * w_ratio;

                    }
                }
            }
            batch_data += in_0_h * in_0_w;
            top_data += out_0_h * out_0_w;
        }
        bottom_rois += in_1_c * in_1_h * in_1_w;
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "SRoiAlign : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("SRoiAlign", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberSRoiAlign, SRoiAlignParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSRoiAlign, SRoiAlignParam, ARM, AK_INT8);

} //namespace anakin

} //namespace anakin
