#include "saber/funcs/impl/cuda/saber_dfmb_psroi_align.h"
#include "saber/saber_funcs_param.h"
namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void DFMBPSROIAlignForward(
        const int nthreads, const Dtype* bottom_data,
        const Dtype heat_map_a, const Dtype heat_map_b,
        const Dtype pad_ratio, const int channels,
        const int height, const int width,
        const int pooled_height, const int pooled_width,
        const Dtype* bottom_rois, const Dtype* bottom_trans,
        const bool no_trans, const Dtype trans_std,
        const int sample_per_part, const int output_dim,
        const int group_height, const int group_width,
        const int part_height, const int part_width,
        const int num_classes, const int channels_each_class,
        Dtype* top_data, Dtype* top_count) {

    CUDA_KERNEL_LOOP(index, nthreads) {
        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        const Dtype* offset_bottom_rois = bottom_rois + n * 5;
        int roi_batch_ind = offset_bottom_rois[0];

        Dtype pad_w = (offset_bottom_rois[3] - offset_bottom_rois[1] + 1) * pad_ratio;
        Dtype pad_h = (offset_bottom_rois[4] - offset_bottom_rois[2] + 1) * pad_ratio;
        Dtype roi_start_w = (offset_bottom_rois[1] - pad_w - heat_map_b) / heat_map_a;
        Dtype roi_start_h = (offset_bottom_rois[2] - pad_h - heat_map_b) / heat_map_a;
        Dtype roi_end_w = (offset_bottom_rois[3] + pad_w - heat_map_b) / heat_map_a;
        Dtype roi_end_h = (offset_bottom_rois[4] + pad_h - heat_map_b) / heat_map_a;
        // Force too small ROIs to be 1x1
        Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);
        Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0

        // Compute w and h at bottom
        Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
        Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

        Dtype sub_bin_size_h = bin_size_h / static_cast<Dtype>(sample_per_part);
        Dtype sub_bin_size_w = bin_size_w / static_cast<Dtype>(sample_per_part);

        int part_h = floor(static_cast<Dtype>(ph) / pooled_height * part_height);
        int part_w = floor(static_cast<Dtype>(pw) / pooled_width * part_width);
        int class_id = ctop / channels_each_class;
        Dtype trans_x = no_trans ? static_cast<Dtype>(0) :
                        bottom_trans[(((n * num_classes + class_id) * 2) *
                                      part_height + part_h) * part_width + part_w] * trans_std;
        Dtype trans_y = no_trans ? static_cast<Dtype>(0) :
                        bottom_trans[(((n * num_classes + class_id) * 2 + 1) *
                                      part_height + part_h) * part_width + part_w] * trans_std;

        int hstart = static_cast<Dtype>(ph) * bin_size_h +
                     roi_start_h + trans_y * roi_height;
        int wstart =  static_cast<Dtype>(pw)* bin_size_w +
                      roi_start_w + trans_x * roi_width;

        Dtype sum = 0;
        int count = 0;
        int gh = floor(static_cast<Dtype>(ph)* group_height / pooled_height);
        int gw = floor(static_cast<Dtype>(pw) * group_width / pooled_width);
        gh = min(max(gh, 0), group_height - 1);
        gw = min(max(gw, 0), group_width - 1);

        const Dtype* offset_bottom_data = bottom_data +
                                          (roi_batch_ind * channels) * height * width;
        for (int ih = 0; ih < sample_per_part; ih++) {
            for (int iw = 0; iw < sample_per_part; iw++) {
                Dtype w = wstart + (iw + 0.5) * sub_bin_size_w;
                Dtype h = hstart + (ih + 0.5) * sub_bin_size_h;
                // bilinear interpolation
                if (w <= -1 || w >= width || h <= -1 || h >= height) {
                    continue;
                }
                int c = (ctop * group_height + gh) * group_width + gw;
                int x1 = floor(w);
                int x2 = ceil(w);
                int y1 = floor(h);
                int y2 = ceil(h);
                Dtype dist_x = static_cast<Dtype>(w - x1);
                Dtype dist_y = static_cast<Dtype>(h - y1);
                const Dtype* data = offset_bottom_data + c * height * width;
                Dtype value11 = (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) ? data[y1 * width + x1] : Dtype(0.0);
                Dtype value12 = (x1 >= 0 && x1 < width && y2 >= 0 && y2 < height) ? data[y2 * width + x1] : Dtype(0.0);
                Dtype value21 = (x2 >= 0 && x2 < width && y1 >= 0 && y1 < height) ? data[y1 * width + x2] : Dtype(0.0);
                Dtype value22 = (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height) ? data[y2 * width + x2] : Dtype(0.0);
                Dtype value = (1 - dist_x) * (1 - dist_y) * value11
                              + (1 - dist_x) * dist_y * value12
                              + dist_x * (1 - dist_y) * value21
                              + dist_x * dist_y * value22;
                sum += value;
                count++;
            }
        }
        top_data[index] = count == 0 ? static_cast<Dtype>(0) : sum / count;
//        top_count[index] = count;
    }
}

template <>
SaberStatus SaberDFMBPSROIAlign<NV, AK_FLOAT>::create(
        const std::vector<OpTensor*>& inputs,
        std::vector<OpTensor*>& outputs,
        DFMBPSROIAlignParam<NV>& param, Context<NV> &ctx) {
    this->_ctx = &ctx;
    channels_ = inputs[0]->channel();
    height_ = inputs[0]->height();
    width_ = inputs[0]->width();

    CHECK_EQ(channels_, output_dim_ * group_height_ * group_width_);
    CHECK_EQ(inputs[1]->channel(), 5);
    if (!no_trans_) {
        CHECK_EQ(inputs[2]->channel() % 2, 0);
        int num_classes = inputs[2]->channel() / 2;
        CHECK_EQ(output_dim_ % num_classes, 0);
        CHECK_EQ(part_height_, inputs[2]->height());
        CHECK_EQ(part_width_, inputs[2]->width());
    }

    Shape out_shape({inputs[1]->num(), param.output_dim, param.pooled_height, param.pooled_width});
//    top_count_.re_alloc(out_shape, AK_FLOAT);
    return SaberSuccess;
}

template <>
SaberStatus SaberDFMBPSROIAlign<NV, AK_FLOAT>::init(
        const std::vector<OpTensor*>& inputs,
        std::vector<OpTensor*>& outputs,
        DFMBPSROIAlignParam<NV>& param, Context<NV> &ctx) {
    this->_ctx = &ctx;
    heat_map_a_ = param.heat_map_a;
    heat_map_b_ = param.heat_map_b;
    pad_ratio_ = param.pad_ratio;
    CHECK_GT(heat_map_a_, 0);
    CHECK_GE(heat_map_b_, 0);
    CHECK_GE(pad_ratio_, 0);
    output_dim_ = param.output_dim;
    trans_std_ = param.trans_std;
    sample_per_part_ = param.sample_per_part;
    group_height_ = param.group_height;
    group_width_ = param.group_width;
    pooled_height_ = param.pooled_height;
    pooled_width_ = param.pooled_width;
    part_height_ = param.part_height;
    LOG(INFO) << param.part_height;
    part_width_ = param.part_width;
    no_trans_ = (inputs.size() < 3);

    CHECK_GT(output_dim_, 0);
    CHECK_GT(sample_per_part_, 0);
    CHECK_GT(group_height_, 0);
    CHECK_GT(group_width_, 0);
    CHECK_GT(pooled_height_, 0);
    CHECK_GT(pooled_width_, 0);
    CHECK_GT(part_height_, 0);
    CHECK_GT(part_width_, 0);
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberDFMBPSROIAlign<NV, AK_FLOAT>::dispatch(
    const std::vector<OpTensor*>& inputs,
    std::vector<OpTensor*>& outputs,
    DFMBPSROIAlignParam<NV>& param) {

    const float* bottom_data = (const float*)inputs[0]->data();
    const float* bottom_rois = (const float*)inputs[1]->data();
    const float *bottom_trans = no_trans_ ? NULL : (const float*)inputs[2]->data();
    float* top_data = (float*)outputs[0]->mutable_data();
//    float* top_count_data = (float*)top_count_.mutable_data();
    int count = outputs[0]->valid_size();
    int num_classes = no_trans_ ? 1 : inputs[2]->channel() / 2;
    int channels_each_class = no_trans_ ? output_dim_ : output_dim_ / num_classes;
    cudaMemsetAsync(top_data, 0, outputs[0]->valid_size() * sizeof(float), _ctx->get_compute_stream());
//    caffe_gpu_set(count, float(0), top_data);
//    caffe_gpu_set(count, float(0), top_count_data);
    cudaDeviceSynchronize();
#undef CUDA_NUM_THREADS
#define CUDA_NUM_THREADS 256
    // NOLINT_NEXT_LINE(whitespace/operators)
    int blocks = CUDA_GET_BLOCKS(count);
    int threads = CUDA_NUM_THREADS;
    DFMBPSROIAlignForward<float> <<< blocks,
            threads, 0, _ctx->get_compute_stream()>>>(count, bottom_data,
                    heat_map_a_, heat_map_b_, pad_ratio_, channels_, height_, width_,
                    pooled_height_, pooled_width_, bottom_rois, bottom_trans,
                    no_trans_, trans_std_, sample_per_part_, output_dim_,
                    group_height_, group_width_, part_height_, part_width_,
                    num_classes, channels_each_class, top_data, nullptr);
    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberDFMBPSROIAlign<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberDFMBPSROIAlign, DFMBPSROIAlignParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberDFMBPSROIAlign, DFMBPSROIAlignParam, NV, AK_INT8);

}
}