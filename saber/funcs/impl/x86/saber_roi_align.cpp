#include "saber/funcs/impl/x86/saber_roi_align.h"
#include <limits>
#include <cmath>
namespace anakin {

namespace saber {

// we calculate the src coordinary and weights previsiously.
template <typename dtype, typename TargetType>
void bilinear_interpolate(
    const int height, const int width,
    const int pooled_height, const int pooled_width, const int iy_upper,
    const int ix_upper, dtype roi_ymin, dtype roi_xmin, dtype bin_size_h, dtype bin_size_w,
    int roi_bin_grid_h, int roi_bin_grid_w, const int kROISize, 
    const int prePosROISize, Tensor<TargetType>* pre_pos, Tensor<TargetType>* pre_w) {
  int pre_calc_index = 0;
  int* pre_pos_data = (int*)pre_pos->mutable_data();
  dtype* pre_w_data = (dtype*)pre_w->mutable_data();
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        // calculate y of sample points
        dtype y = roi_ymin + ph * bin_size_h +
              static_cast<dtype>(iy + .5f) * bin_size_h /
                  static_cast<dtype>(roi_bin_grid_h);
        // calculate x of samle points
        for (int ix = 0; ix < ix_upper; ix++) {
          dtype x = roi_xmin + pw * bin_size_w +
                static_cast<dtype>(ix + .5f) * bin_size_w /
                    static_cast<dtype>(roi_bin_grid_w);
          // deal with elements out of map
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            for (int i = 0; i < prePosROISize; ++i) {
              pre_pos_data[i + pre_calc_index * prePosROISize] = 0;
              pre_w_data[i + pre_calc_index * prePosROISize] = 0;
            }
            pre_calc_index += 1;
            continue;
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
          pre_pos_data[pre_calc_index * prePosROISize] = y_low * width + x_low;
          pre_pos_data[pre_calc_index * prePosROISize + 1] = y_low * width + x_high;
          pre_pos_data[pre_calc_index * prePosROISize + 2] = y_high * width + x_low;
          pre_pos_data[pre_calc_index * prePosROISize + 3] = y_high * width + x_high;
          pre_w_data[pre_calc_index * prePosROISize] = hy * hx;
          pre_w_data[pre_calc_index * prePosROISize + 1] = hy * lx;
          pre_w_data[pre_calc_index * prePosROISize + 2] = ly * hx;
          pre_w_data[pre_calc_index * prePosROISize + 3] = ly * lx;
          pre_calc_index += 1;
        }
      }
    }
  }
}

template <DataType OpDtype>
SaberStatus SaberRoiAlign<X86, OpDtype>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs, \
    RoiAlignParam<X86>& param) {

    const OpDataType* input_data = (const OpDataType*)inputs[0]->data();
    const OpDataType* rois = (const OpDataType*)inputs[1]->data();
    OpDataType* output_data = (OpDataType*)outputs[0]->mutable_data();
    
    int batch_size = inputs[0]->num();
    int channels = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int rois_num = inputs[1]->num();
    // int count = input[0]->valid_size();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
    // For each ROIs, do fix-sized align.
        for (int n = 0; n < rois_num; ++n) {
            const OpDataType* cur_rois = rois + n * _kROISize;
            int rois_id = cur_rois[0];
            OpDataType roi_xmin = cur_rois[1] * param.spatial_scale;
            OpDataType roi_ymin = cur_rois[2] * param.spatial_scale;
            OpDataType roi_xmax = cur_rois[3] * param.spatial_scale;
            OpDataType roi_ymax = cur_rois[4] * param.spatial_scale;
            
            OpDataType roi_width = std::max(roi_xmax - roi_xmin, static_cast<OpDataType>(1.));
            OpDataType roi_height = std::max(roi_ymax - roi_ymin, static_cast<OpDataType>(1.));
            OpDataType bin_size_h = static_cast<OpDataType>(roi_height) / static_cast<OpDataType>(param.pooled_height);
            OpDataType bin_size_w = static_cast<OpDataType>(roi_width) / static_cast<OpDataType>(param.pooled_width);
            const OpDataType* batch_data = input_data + rois_id * _in_n_stride;
            int roi_bin_grid_h = (param.sampling_ratio > 0)? param.sampling_ratio : ceil(roi_height / param.pooled_height);
            int roi_bin_grid_w = (param.sampling_ratio > 0)? param.sampling_ratio : ceil(roi_width / param.pooled_width);
            int count = roi_bin_grid_h * roi_bin_grid_w;
            int pre_size = count * _out_c_stride;
            _pre_pos.reshape(Shape({pre_size, _prePosROISize, 1, 1})); //pre ROI
            _pre_w.reshape(Shape({pre_size, _prePosROISize, 1, 1})); // pre ROI weights.

            bilinear_interpolate<OpDataType, X86>(height, width, 
                                        param.pooled_height, param.pooled_width, 
                                        roi_bin_grid_h,roi_bin_grid_w, 
                                        roi_ymin, roi_xmin, 
                                        bin_size_h, bin_size_w,
                                        roi_bin_grid_h, roi_bin_grid_w,
                                        _kROISize, _prePosROISize,
                                        &_pre_pos, &_pre_w);
            const int* pre_pos_data = (const int*)_pre_pos.data();
            const OpDataType* pre_w_data = (const OpDataType*)_pre_w.data();
            for (int c = 0; c < channels; c++) {
                int pre_calc_index = 0;
                for (int ph = 0; ph < param.pooled_height; ph++) {
                    for (int pw = 0; pw < param.pooled_width; pw++) {
                        const int pool_index = ph * param.pooled_width + pw;
                        OpDataType output_val = 0;
                        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                                for (int i = 0; i < _prePosROISize; i++) {
                                    int pos = pre_pos_data[pre_calc_index * _prePosROISize + i];
                                    OpDataType w = pre_w_data[pre_calc_index * _prePosROISize + i];
                                    output_val += w * batch_data[pos];
                                }
                                pre_calc_index += 1;
                            }
                        }
                        output_val /= count;
                        output_data[pool_index] = output_val;
                    }
                }
                batch_data += _in_c_stride;
                output_data += _out_c_stride;
            }
        }
    }
    return SaberSuccess;
}
template class SaberRoiAlign<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberRoiAlign, RoiAlignParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberRoiAlign, RoiAlignParam, X86, AK_INT8);
} //namespace saber.
} //namespace anakin.
