#include "saber/funcs/impl/cuda/saber_roi_align.h"
#include "saber/core/tensor_op.h"
// #include "cuda_fp16.h"
// #include <cfloat>

namespace anakin {

namespace saber {

//The Bilinear interpolation
template <typename dtype>
__device__ dtype BilinearInterpolate(const dtype* input_data, const int height,
                                    const int width, dtype y, dtype x) {
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
      return 0;
    }
    y = y <= 0 ? 0 : y;
    x = x <= 0 ? 0 : x;
    int y_low = static_cast<int>(y);
    int x_low = static_cast<int>(x);
    int y_high;
    int x_high;
    if (y_low >= height - 1) {
      y_high = y_low = height - 1;
      y = static_cast<dtype>(y_low);
    } else {
      y_high = y_low + 1;
    }
    if (x_low >= width - 1) {
      x_high = x_low = width - 1;
      x = static_cast<dtype>(x_low);
    } else {
      x_high = x_low + 1;
    }
    dtype ly = y - y_low, lx = x - x_low;
    dtype hy = 1. - ly, hx = 1. - lx;

    dtype v1 = input_data[y_low * width + x_low];
    dtype v2 = input_data[y_low * width + x_high];
    dtype v3 = input_data[y_high * width + x_low];
    dtype v4 = input_data[y_high * width + x_high];
    dtype w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}


template <typename dtype>
__global__ void kernel_roi_align(const dtype* src,
                                const dtype* input_rois,
                                dtype* dst,
                                const int in_n_stride,
                                const int in_c_stride,
                                const int in_h_stride,
                                const int in_w_stride,
                                const int out_n_stride,
                                const int out_c_stride,
                                const int out_h_stride,
                                const int out_w_stride,
                                const int in_c,
                                const int in_h,
                                const int in_w,
                                const int pooled_height,
                                const int pooled_width,
                                const int sampling_ratio,
                                const int kROISize,
                                const int num_threads,
                                const dtype spatial_scale) {
    CUDA_KERNEL_LOOP(tid, num_threads) {
        int n = tid / out_n_stride;
        int c = (tid / out_c_stride) % in_c;
        int ph = (tid / pooled_width) % pooled_height;
        int pw = tid % pooled_width;

        const dtype* offset_input_rois = input_rois + n * kROISize;
        int roi_batch_id = offset_input_rois[0];
        dtype roi_xmin = offset_input_rois[1] * spatial_scale;
        dtype roi_ymin = offset_input_rois[2] * spatial_scale;
        dtype roi_xmax = offset_input_rois[3] * spatial_scale;
        dtype roi_ymax = offset_input_rois[4] * spatial_scale;

        dtype roi_width = fmaxf(roi_xmax - roi_xmin, 1.0f);
        dtype roi_height = fmaxf(roi_ymax - roi_ymin, 1.0f);
        dtype bin_size_h = static_cast<dtype>(roi_height) / static_cast<dtype>(pooled_height);
        dtype bin_size_w = static_cast<dtype>(roi_width) / static_cast<dtype>(pooled_width);

        const dtype* offset_src = src + roi_batch_id * in_n_stride + c * in_c_stride;
        int roi_bin_grid_h = sampling_ratio > 0? sampling_ratio : ceil(roi_height / pooled_height);
        int roi_bin_grid_w = sampling_ratio > 0? sampling_ratio : ceil(roi_width / pooled_width);
        const int sample_count = roi_bin_grid_h * roi_bin_grid_w;
        dtype val = 0;
        for (int iy = 0; iy < roi_bin_grid_h; ++iy) {
            dtype y = roi_ymin + ph * bin_size_h + 
                    static_cast<dtype>(iy + 0.5f) * bin_size_h / static_cast<dtype>(roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ++ix) {
                dtype x = roi_xmin + pw * bin_size_w + 
                static_cast<dtype>(ix + 0.5f) * bin_size_w / static_cast<dtype>(roi_bin_grid_w);
                dtype tmp = BilinearInterpolate<dtype>(offset_src, in_h, in_w, y, x);
                val += tmp;
            }
        }
        val /= sample_count;
        dst[tid] = val;
    }
}

template <DataType OpDtype>
SaberStatus SaberRoiAlign<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    RoiAlignParam<NV>& param) {

    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    const OpDataType* in_rois = (const OpDataType*)inputs[1]->data();
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = outputs[0]->valid_size();
    int out_n = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    int in_n = inputs[0]->num();
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        kernel_roi_align<OpDataType>\
                 <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                    in_data, in_rois, out_data, \
                 _in_n_stride, _in_c_stride, _in_h_stride, _in_w_stride,\
                 _out_n_stride, _out_c_stride, _out_h_stride, _out_w_stride,\
                 in_c, in_h, in_w,
                 param.pooled_height, param.pooled_width, param.sampling_ratio, \
                 _kROISize, count, param.spatial_scale);
    }
    return SaberSuccess;
}

template class SaberRoiAlign<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberRoiAlign, RoiAlignParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberRoiAlign, RoiAlignParam, NV, AK_INT8);
}
}
