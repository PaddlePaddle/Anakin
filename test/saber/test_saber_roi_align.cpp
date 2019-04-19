#include "saber/core/context.h"
#include "saber/funcs/roi_align.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include <vector>

using namespace anakin::saber;


/**
 * @brief This operator is Region of Interest(ROIAlign) Align.
 *   The main steps of RoiAlign are as follows:
 *      For each ROI, extract fixed-size map ([pooled_height, pooled_width]something like 3*3):
 *       1. chose a sampling_ratio[the number of sampling points] for each pixel of fixed-size map
 *       2. then, for each smapling point, compute the src coordinate, and 
 *           suppose that we get the src's coordinate (x, y).
 *            using the fomula to calculate coordinate (x, y).
 *       3. for each (x, y) , do bilinear interpolate and suppose we get val.
 *       4. sum up val and calculate the mean of them.
 * 
 * 
 * @tparam dtype 
 * @tparam TargetType_D 
 * @tparam TargetType_H 
 * @param input 
 * @param output 
 * @param param 
 */

template <typename dtype, typename TargetType>
void PreCalcForBilinearInterpolate(
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
          int y_high = 0;
          int x_high = 0;
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
          dtype ly = y - y_low;
          dtype lx = x - x_low;
          dtype hy = 1. - ly;
          dtype hx = 1. - lx;
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


template <typename dtype, typename TargetType_D, typename TargetType_H>
void roi_align_cpu_base(const std::vector<Tensor<TargetType_H>* >& input,
                      std::vector<Tensor<TargetType_H>* >& output, RoiAlignParam<TargetType_D>& param) {
    
    CHECK_EQ(input.size(), 2) << "input size must be 2!!!";
    int batch_size = input[0]->num();
    int channels = input[0]->channel();
    int height = input[0]->height();
    int width = input[0]->width();
    int rois_num = input[1]->num();
    // int count = input[0]->valid_size();
    const int kROISize = 5;
    const int prePosROISize = 4;

    Shape in_stride = input[0]->get_stride();
    Shape roi_stride = input[1]->get_stride();
    Shape out_stride = output[0]->get_stride();

    const dtype* input_data = (const dtype*)input[0]->data();
    const dtype* rois = (const dtype*)input[1]->data();
    dtype* output_data = (dtype*)output[0]->mutable_data();
    // For each ROIs, do fix-sized align.
    for (int n = 0; n < rois_num; ++n) {
        const dtype* cur_rois = rois + n * kROISize;
        int rois_id = cur_rois[0];
        dtype roi_xmin = cur_rois[1] * param.spatial_scale;
        dtype roi_ymin = cur_rois[2] * param.spatial_scale;
        dtype roi_xmax = cur_rois[3] * param.spatial_scale;
        dtype roi_ymax = cur_rois[4] * param.spatial_scale;
        
        dtype roi_width = std::max(roi_xmax - roi_xmin, static_cast<dtype>(1.));
        dtype roi_height = std::max(roi_ymax - roi_ymin, static_cast<dtype>(1.));
        dtype bin_size_h = static_cast<dtype>(roi_height) / static_cast<dtype>(param.pooled_height);
        dtype bin_size_w = static_cast<dtype>(roi_width) / static_cast<dtype>(param.pooled_width);
        const dtype* batch_data = input_data + rois_id * in_stride[0];
        int roi_bin_grid_h = (param.sampling_ratio > 0)? param.sampling_ratio : ceil(roi_height / param.pooled_height);
        int roi_bin_grid_w = (param.sampling_ratio > 0)? param.sampling_ratio : ceil(roi_width / param.pooled_width);
        int count = roi_bin_grid_h * roi_bin_grid_w;
        Tensor<TargetType_H> pre_pos;
        Tensor<TargetType_H> pre_w;
        int pre_size = count * out_stride[1];
        pre_pos.reshape(Shape({pre_size, prePosROISize, 1, 1})); //pre ROI
        pre_w.reshape(Shape({pre_size, prePosROISize, 1, 1})); // pre ROI weights.

        PreCalcForBilinearInterpolate<dtype, TargetType_H>(height, width, 
                                     param.pooled_height, param.pooled_width, 
                                     roi_bin_grid_h,roi_bin_grid_w, 
                                     roi_ymin, roi_xmin, 
                                     bin_size_h, bin_size_w,
                                     roi_bin_grid_h, roi_bin_grid_w,
                                     kROISize, prePosROISize,
                                     &pre_pos, &pre_w);
        const int* pre_pos_data = (const int*)pre_pos.data();
        const dtype* pre_w_data = (const dtype*)pre_w.data();
        for (int c = 0; c < channels; c++) {
            int pre_calc_index = 0;
            for (int ph = 0; ph < param.pooled_height; ph++) {
                for (int pw = 0; pw < param.pooled_width; pw++) {
                    const int pool_index = ph * param.pooled_width + pw;
                    dtype output_val = 0;
                    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                            for (int i = 0; i < prePosROISize; i++) {
                                int pos = pre_pos_data[pre_calc_index * prePosROISize + i];
                                dtype w = pre_w_data[pre_calc_index * prePosROISize + i];
                                output_val += w * batch_data[pos];
                            }
                            pre_calc_index += 1;
                        }
                    }
                    output_val /= count;
                    output_data[pool_index] = output_val;
                }
            }
            batch_data += in_stride[1];
            output_data += out_stride[1];
        }
    }
}

template <DataType Dtype, typename TargetType_D, typename TargetType_H>
void test_roi_align(){
    TestSaberBase<TargetType_D, TargetType_H, Dtype, RoiAlign, RoiAlignParam> testbase(2);
    float spatial_scale = 1.0f;
    int sampling_ratio = -1.0;
    // RoiAlignParam<TargetType_D> param;


    for (int num_in : {2, 8, 16, 32}) {
        for (int c_in : {2, 8, 16, 32}) {
            for (int h_in : {2, 7, 8, 16}) {
                for (int w_in:{2, 21, 16, 32}) {
                    for (auto roi_num:{1, 3, 6}){
                        for (auto pooled_height:{1, 2, 4}){
                            for (auto pooled_width:{1, 2, 4}){
                                Shape in_shape({num_in, c_in, h_in, w_in});
                                Shape roi_shape({roi_num, 5, 1, 1});
                                RoiAlignParam<TargetType_D> param(pooled_height, 
                                pooled_width, spatial_scale, sampling_ratio);
                                Tensor<TargetType_H> th_in, th_roi;
                                Tensor<TargetType_D> td_in, td_roi;
                                th_in.re_alloc(in_shape, AK_FLOAT);
                                th_roi.re_alloc(roi_shape, AK_FLOAT);
                                td_in.re_alloc(in_shape, AK_FLOAT);
                                td_roi.re_alloc(roi_shape, AK_FLOAT);
                                // prepare host data
                                fill_tensor_rand(th_in, 0.0, 1.0);
                                // prepare roi data
                                float* roi_data = (float*)th_roi.mutable_data();
                                srand(time(0));
                                for (int i = 0; i < roi_num; ++i) {
                                    roi_data[i * 5] = rand() % num_in;
                                    roi_data[i * 5 + 1] = floor(rand() % (w_in/2) / spatial_scale);
                                    roi_data[i * 5 + 2] = floor(rand() % (h_in/2) / spatial_scale);
                                    roi_data[i * 5 + 3] = floor((rand() % (w_in/2) + w_in/2) / spatial_scale);
                                    roi_data[i * 5 + 4] = floor((rand() % (h_in/2) + h_in/2) / spatial_scale);
                                }
                                td_in.copy_from(th_in);
                                td_roi.copy_from(th_roi);
                                std::vector<Tensor<TargetType_D>*> input;
                                input.push_back(&td_in);
                                input.push_back(&td_roi);
                                testbase.add_custom_input(input);
                                testbase.set_param(param);
                                testbase.run_test(roi_align_cpu_base<float, TargetType_D, TargetType_H>);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_op_RoiAlign) {

#ifdef USE_CUDA
   //Init the test_base
    test_roi_align<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef USE_X86_PLACE
//    test_roi_align<AK_FLOAT, X86, X86>();
#endif
#ifdef USE_ARM_PLACE
    //test_RoiAlign<AK_FLOAT, ARM, ARM>();
#endif
#ifdef USE_BM
   // Env<BM>::env_init();
    //test_accuracy<BM, X86>(num, channel, height, width,VENDER_IMPL);
#endif
}

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
