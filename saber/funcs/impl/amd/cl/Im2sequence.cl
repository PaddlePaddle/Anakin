/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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
#define WIN_SIZE (WIN_H * WIN_W)
#define KERNEL_EXTEN (KERNEL_EXTEN_H * KERNEL_EXTEN_W)

__kernel void ker_im2sequence_fwd(
    global float* out_data,
    global const float* in_data,
    const int in_n,
    const int in_c,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int col_height,
    const int col_width,
    const int window_h,
    const int window_w,
    const int pad_up,
    const int pad_left,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int kernel_exten_h,
    const int kernel_exten_w,
    const int num_threads) {

    int global_idx       = get_global_id(0);
    int out_size_per_img = out_h * out_w;
    int w                = global_idx % out_w;
    int h                = (global_idx / out_w) % out_h;
    int n                = (global_idx / out_size_per_img) % in_n;
    int c                = global_idx / (out_size_per_img * in_n);

    int in_start_w                  = w * stride_w - pad_left;
    int in_start_h                  = h * stride_h - pad_up;
    int in_end_w                    = in_start_w + kernel_exten_w;
    int in_end_h                    = in_start_h + kernel_exten_h;
    int in_offset                   = (n * in_c + c) * in_h * in_w;
    global const float* in_data_tmp = in_data + in_offset;
    int out_offset                  = (global_idx % col_height * in_c + c) * window_h * window_w;
    global float* out_data_tmp      = out_data;

    for (int i = in_start_h; i < in_end_h; i += dilation_h) {
        for (int j = in_start_w; j < in_end_w; j += dilation_w) {
            if (i < 0 || i >= in_h || j < 0 || j >= in_w) {
                out_data_tmp[out_offset++] = 0;
            } else {
                out_data_tmp[out_offset++] = in_data_tmp[i * in_w + j];
            }
        }
    }
}

__kernel void ker_im2sequence_fwd_shared(
    global float* out_data,
    global const float* in_data,
    const int in_n,
    const int in_c,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int col_height,
    const int col_width,
    const int window_h,
    const int window_w,
    const int pad_up,
    const int pad_left,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int kernel_exten_h,
    const int kernel_exten_w,
    const int num_threads) {
    int local_idx  = get_local_id(0);
    int global_idx = get_global_id(0);
    local float share_data[256 * WIN_SIZE + KERNEL_EXTEN];
    int out_size_per_img = out_h * out_w;
    int w                = global_idx % out_w;
    int h                = (global_idx / out_w) % out_h;
    int n                = (global_idx / out_size_per_img) % in_n;
    int c                = global_idx / col_height;

    int in_start_w                  = w * stride_w - pad_left;
    int in_start_h                  = h * stride_h - pad_up;
    int in_end_w                    = in_start_w + kernel_exten_w;
    int in_end_h                    = in_start_h + kernel_exten_h;
    int in_offset                   = (n * in_c + c) * in_h * in_w;
    global const float* in_data_tmp = in_data + in_offset;
    int window_size                 = window_h * window_w;

    int id = 0;

    for (int i = in_start_h; i < in_end_h; i += dilation_h) {
        for (int j = in_start_w; j < in_end_w; j += dilation_w) {
            float data = 0;

            if (i < 0 || i >= in_h || j < 0 || j >= in_w) {
                data = 0.0f;
                // share_data[id * blockDim.x + thread_id] = 0;
                // out_data_tmp[out_offset++] = 0;
            } else {
                data = in_data_tmp[i * in_w + j];
                // out_data_tmp[out_offset++] = in_data_tmp[i * in_w + j]
            }

            share_data[local_idx * window_size + id] = data;
            id++;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    int valid_height = min(num_threads - get_group_id(0) * get_local_size(0), get_local_size(0));
    // if (threadIdx.x == 0) {
    //     printf("share memory\n");
    //     for (int i = 0; i < valid_height; i++) {
    //          for (int j = 0; j < window_h * window_w; j++) {
    //              printf("%f, ", share_data[i *  window_h * window_w + j]);
    //          }
    //          printf("\n");
    //     }
    //}

    for (int i = local_idx; i < valid_height * window_h * window_w; i += get_local_size(0)) {
        int h_id                              = i / window_size;
        int w_id                              = i % window_size;
        int id                                = get_group_id(0) * get_local_size(0) + h_id;
        int row_id                            = id % col_height;
        int col_id                            = id / col_height * window_size + w_id;
        out_data[row_id * col_width + col_id] = share_data[i];
    }
}
