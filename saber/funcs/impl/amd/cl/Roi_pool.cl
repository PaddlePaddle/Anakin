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

#pragma OPENCL EXTENSION cl_amd_printf : enable
#define Dtype float
#define fmaxf(a, b) (((a) > (b)) ? (a) : (b))
#define fminf(a, b) (((a) < (b)) ? (a) : (b))

__kernel void Roi_pool(
        __global float* out_data,
        __global float* out_index,
        __global const float* in_data,
        __global const float* in_rois,
        const int in_n_stride,
        const int in_c_stride,
        const int in_h_stride,
        const int in_w_stride,
        const int out_n_stride,
        const int out_c_stride,
        const int out_h_stride,
        const int out_w_stride,
        const float spatial_scale,
        const int in_n,
        const int in_c,
        const int in_h,
        const int in_w,
        const int roi_num,
        const int roi_size,
        const int out_h,
        const int out_w,
        const int num_threads) {
    int tid = get_global_id(0);
    if (tid < num_threads) {
        int n                         = (tid / out_n_stride);
        int c                         = (tid / out_c_stride) % in_c;
        int h                         = (tid / out_h_stride) % out_h;
        int w                         = (tid / out_w_stride) % out_w;
        __global const float* cur_roi = in_rois + n * roi_size;
        int roi_batch_id              = cur_roi[0];
        int roi_start_w               = round(cur_roi[1] * spatial_scale);
        int roi_start_h               = round(cur_roi[2] * spatial_scale);
        int roi_end_w                 = round(cur_roi[3] * spatial_scale);
        int roi_end_h                 = round(cur_roi[4] * spatial_scale);
        int roi_width                 = roi_end_w - roi_start_w + 1;
        int roi_height                = roi_end_h - roi_start_h + 1;
        Dtype pool_w_rate             = (Dtype)roi_width / out_w;
        Dtype pool_h_rate             = (Dtype)roi_height / out_h;

        int h_start                  = (int)(floor((float)(h)*pool_h_rate));
        int w_start                  = (int)(floor((float)(w)*pool_w_rate));
        int h_end                    = (int)(ceil((float)(h + 1) * pool_h_rate));
        int w_end                    = (int)(ceil((float)(w + 1) * pool_w_rate));
        h_start                      = fminf(fmaxf(h_start + roi_start_h, 0), in_h);
        h_end                        = fminf(fmaxf(h_end + roi_start_h, 0), in_h);
        w_start                      = fminf(fmaxf(w_start + roi_start_w, 0), in_w);
        w_end                        = fminf(fmaxf(w_end + roi_start_w, 0), in_w);
        bool is_empty                = (h_end <= h_start) || (w_end <= w_start);
        Dtype max_val                = is_empty ? 0 : -FLT_MAX;
        int max_idx                  = -1;
        __global const float* in_tmp = in_data + roi_batch_id * in_n_stride + c * in_c_stride;
        for (int h_id = h_start; h_id < h_end; ++h_id) {
            for (int w_id = w_start; w_id < w_end; ++w_id) {
                int input_data_index = h_id * in_h_stride + w_id * in_w_stride;
                Dtype data           = in_tmp[input_data_index];
                if (data > max_val) {
                    max_val = data;
                    max_idx = input_data_index;
                }
            }
        }
        out_data[tid] = max_val;
        if (out_index) {
            out_index[tid] = max_idx;
        }
    }
}
