#include "saber/funcs/impl/cuda/saber_ps_roi_pooling.h"
#include "saber/core/tensor_op.h"
#include <cfloat>
#include <cmath>

namespace anakin {

namespace saber {

/* 
 * crop rois and resize to [crop_height, crop_width] from in_data
 * in_data shape: [pooled_h * pooled_w * c, im_h, im_w]
 * rois shape: [num_rois, 4]
 * out_data: [pooled_h * pooled_w * c, num_rois, crop_height, crop_width]
 */
template <typename Dtype>
__global__ void crop_and_resize_kernel(
    const Dtype* in_data, 
    const Dtype* rois, 
    Dtype* out_data, 
    int num_rois, 
    int im_h, int im_w, 
    int crop_height, int crop_width,
    int count,
    int method,
    float extra_value){

    CUDA_KERNEL_LOOP(index, count){
        int temp_ind = index;
        int cur_w = temp_ind % crop_width;
        temp_ind /= crop_width;
        int cur_h = temp_ind % crop_height;
        temp_ind /= crop_height;
        int cur_n = temp_ind % num_rois;
        int cur_c = temp_ind / num_rois;

        const Dtype* rois_data = rois + cur_n * 4;
        
        float y1 = rois_data[0] * (im_h - 1);
        float x1 = rois_data[1] * (im_w - 1);
        float y2 = rois_data[2] * (im_h - 1);
        float x2 = rois_data[3] * (im_w - 1);

        float height_scale = crop_height > 1 ? (y2 - y1)/(crop_height - 1) : 0;
        float width_scale = crop_width > 1 ? (x2 - x1)/(crop_width - 1) : 0;

        float in_y = crop_height > 1 ? y1 + cur_h * height_scale : (y1 + y2)/2;

        if ( in_y < 0 || in_y > im_h - 1){
            out_data[index] = extra_value;
            continue;
        }

        float in_x = crop_width > 1 ? x1 + cur_w * width_scale : (x1 + x2)/2;
        if ( in_x < 0 || in_x > im_w - 1){
            out_data[index] = extra_value;
            continue;
        }

        const Dtype* im_data = in_data + cur_c * im_h * im_w;

        //resize method 0 means bilinear
        if (method == 0){
            int top_y = floor(in_y);
            int bot_y = ceil(in_y);
            float y_lerp = in_y - top_y;

            int left_x = floor(in_x);
            int right_x = ceil(in_x);
            float x_lerp = in_x - left_x;

            Dtype top_left = im_data[top_y*im_w + left_x];
            Dtype top_right = im_data[top_y*im_w + right_x];
            Dtype bot_left = im_data[bot_y*im_w + left_x];
            Dtype bot_right = im_data[bot_y*im_w + right_x];
            float top = top_left + (top_right - top_left) * y_lerp;
            float bot = bot_left + (bot_right - bot_left) * y_lerp;
            out_data[index] = top + (bot - top) * x_lerp; 
        } else {
          //else method means nearest 
          int closest_x = round(in_x);
          int closest_y = round(in_y);
          out_data[index] = im_data[closest_y*im_w + closest_x];
        }
    }

}

template <typename Dtype>
__global__ void crop_global_pooling_kernel(const Dtype* in_data, Dtype* out_data, 
    int pooled_size, int channel, int num_rois, int crop_height, int crop_width, 
    int count){
    CUDA_KERNEL_LOOP(index, count){
        int cur_n = index / channel;
        int cur_c = index % channel;
        int crop_size = crop_height * crop_width;
        Dtype sum = 0;
        for (int i = 0; i < crop_size; ++i){
            Dtype tmp_sum = 0;
            for (int j = 0; j < pooled_size; ++j){
                tmp_sum += in_data[(j * num_rois + cur_n) * crop_size + i];
            }
            sum += tmp_sum / pooled_size;
        }
        out_data[index] = sum /crop_size;
    }
}

template <typename Dtype>
__global__ void crop_no_global_pooling_kernel(const Dtype* in_data, Dtype* out_data, 
    int pooled_height, int pooled_width, int channel, int num_rois, int crop_height, int crop_width, 
    int count){
    CUDA_KERNEL_LOOP(index, count){
        int cur_pw = index % pooled_width;
        index /= pooled_width;
        int cur_cw = index % crop_width;
        index /= crop_width;
        int cur_ph = index % pooled_height;
        index /= pooled_height;
        int cur_ch = index % crop_height;
        index /= crop_height;
        int cur_c = index % channel;
        int cur_n = index / channel;

        int in_index = ((((cur_ph * pooled_width + cur_pw) * channel + 
            cur_c) * num_rois + cur_n) * crop_height + cur_ch) * crop_width + cur_cw;
        out_data[index] = in_data[in_index];
    }
}

//for tf, it has no batch_ind
template <typename Dtype>
__global__ void psroi_pool_kernel_no_batchind(const Dtype* in_data, const Dtype* rois, Dtype* out_data, 
    int in_n, int in_c, int in_h, int in_w, int o_c, int o_h, int o_w, 
    int pooled_h, int pooled_w, float spatial_scale, int count){

    CUDA_KERNEL_LOOP(index, count){
        int temp_ind = index;
        int cur_w = temp_ind % o_w;
        temp_ind /= o_w;
        int cur_h = temp_ind % o_h;
        temp_ind /= o_h;
        int cur_c = temp_ind % o_c;
        int cur_n = temp_ind / o_c;

        const Dtype* rois_data = rois + cur_n * 4;
        
        int roi_x0 = fminf(fmaxf(rois_data[0] * spatial_scale, 0), in_w-1);
        int roi_y0 = fminf(fmaxf(rois_data[1] * spatial_scale, 0), in_h-1);
        int roi_x1 = fminf(fmaxf(rois_data[2] * spatial_scale, 0), in_w-1);
        int roi_y1 = fminf(fmaxf(rois_data[3] * spatial_scale, 0), in_h-1);

        int roi_h = roi_y1 - roi_y0 + 1;
        int roi_w = roi_x1 - roi_x0 + 1;

        Dtype bin_w = static_cast<Dtype>(roi_w) / pooled_w;
        Dtype bin_h = static_cast<Dtype>(roi_h) / pooled_h;

        int ws = roi_x0 + bin_w * cur_w;
        int we = ceil(roi_x0 + bin_w * (cur_w + 1));
        int ys = roi_y0 + bin_h * cur_h;
        int ye = ceil(roi_y0 + bin_h * (cur_h + 1));

        int c_index = (cur_h * pooled_w + cur_w) * o_c + cur_c;

        const Dtype* offset_in_data = in_data + c_index * in_w * in_h;

        Dtype sum = 0;

        for (int y = ys; y < ye; ++y){
            for (int w = ws; w < we; ++w){
                sum += offset_in_data[y * in_w + w];
            }
        }
        sum /= (ye - ys) * (we - ws);

        //tf is set to `hwc` format, here we set `chw` format
        out_data[index] = sum;  
        
    }

}

//for caffe, it has batchind
template <typename Dtype>
__global__ void psroi_pool_kernel_with_batchind(const Dtype* in_data, const Dtype* rois, Dtype* out_data,
    int in_n, int in_c, int in_h, int in_w, int o_c, int o_h, int o_w, 
    int pooled_h, int pooled_w, float spatial_scale, int count){

    CUDA_KERNEL_LOOP(index, count){
        int temp_ind = index;
        int cur_w = temp_ind % o_w;
        temp_ind /= o_w;
        int cur_h = temp_ind % o_h;
        temp_ind /= o_h;
        int cur_c = temp_ind % o_c;
        int cur_n = temp_ind / o_c;

        const Dtype* rois_data = rois + cur_n * 5;
        
        int batch = rois_data[0]; 
        Dtype roi_x0 = rois_data[1] * spatial_scale;
        Dtype roi_y0 = rois_data[2] * spatial_scale;
        Dtype roi_x1 = (rois_data[3] + 1) * spatial_scale;
        Dtype roi_y1 = (rois_data[4] + 1) * spatial_scale;

        Dtype roi_h = roi_y1 - roi_y0;
        Dtype roi_w = roi_x1 - roi_x0;

        Dtype bin_w = roi_w / pooled_w;
        Dtype bin_h = roi_h / pooled_h;

        int ws = roi_x0 + bin_w * cur_w;
        int we = ceil(roi_x0 + bin_w * (cur_w + 1));
        int ys = roi_y0 + bin_h * cur_h;
        int ye = ceil(roi_y0 + bin_h * (cur_h + 1));

        ws = fminf(fmaxf(ws, 0), in_w);
        we = fminf(fmaxf(we, 0), in_w);
        ys = fminf(fmaxf(ys, 0), in_h);
        ye = fminf(fmaxf(ye, 0), in_h);

        int c_index = (cur_h * pooled_w + cur_w) * o_c + cur_c;

        const Dtype* offset_in_data = in_data + (batch * in_c + c_index) * in_w * in_h;

        Dtype sum = 0.f;

        for (int y = ys; y < ye; ++y){
            for (int w = ws; w < we; ++w){
                sum += offset_in_data[y * in_w + w];
            }
        }
        sum /= (ye - ys) * (we - ws);

        out_data[index] = sum;  
        
    }

}

template <DataType OpDtype>
SaberStatus SaberPsRoiPool<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    PsRoiPoolParam<NV>& param) {

    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    const OpDataType* in_rois = (const OpDataType*)inputs[1]->data();
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* inter_data = (OpDataType*)_crop_data.mutable_data();
    
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();


    int num_rois = inputs[1] -> num();
    int out_n = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    int in_n = inputs[0]->num();
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();

    int crop_width = param.crop_width / param.pooled_width;
    int crop_height = param.crop_height / param.pooled_height;

    int crop_count = _crop_data.valid_size();
    int pool_count = outputs[0]->valid_size();
    int pooled_size = param.pooled_height * param.pooled_width;

    crop_and_resize_kernel<OpDataType>\
        <<<CUDA_GET_BLOCKS(crop_count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
            in_data, in_rois, inter_data, num_rois, in_h, in_w,
            crop_height, crop_width, crop_count, param.method,
            param.extra_value);
    if (param.global_pooling){
        crop_global_pooling_kernel<OpDataType>\
        <<<CUDA_GET_BLOCKS(pool_count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
        inter_data, out_data, pooled_size, out_c,
        num_rois, crop_height, crop_width, pool_count);
    } else {
        crop_no_global_pooling_kernel<OpDataType>\
        <<<CUDA_GET_BLOCKS(crop_count), CUDA_NUM_THREADS, 0, cuda_stream>>>\
        (inter_data, out_data, param.pooled_height, param.pooled_width,
        out_c, num_rois, crop_height, crop_width, pool_count);
    }

    return SaberSuccess;
    
}

}
}
