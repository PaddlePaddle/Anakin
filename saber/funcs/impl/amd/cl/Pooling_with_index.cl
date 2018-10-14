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

#define fmaxf(a, b) (((a) > (b)) ? (a) : (b))
#define fminf(a, b) (((a) < (b)) ? (a) : (b))

__kernel void Pooling_with_index(
        __global float* out_data,
        __global float* out_index,
        __global const float* in_data,
        const int in_n_stride,
        const int in_c_stride,
        const int in_h_stride,
        const int in_w_stride,
        const int in_h,
        const int in_w,
        const int out_n_stride,
        const int out_c_stride,
        const int out_h_stride,
        const int out_w_stride,
        const int out_h,
        const int out_w,
        const int in_n,
        const int in_c,
        const int pad_h,
        const int pad_w,
        const int stride_h,
        const int stride_w,
        const int window_h,
        const int window_w,
        const int num_threads) {
    int tid = get_global_id(0);
    if (tid < num_threads) {
        int n           = (tid / out_n_stride) % in_n;
        int c           = (tid / out_c_stride) % in_c;
        int h           = (tid / out_h_stride) % out_h;
        int w           = (tid / out_w_stride) % out_w;
        float max_data  = -FLT_MAX;
        float max_index = 0;
        int start_h     = h * stride_h - pad_h;
        int end_h       = start_h + window_h;
        start_h         = start_h < 0 ? 0 : start_h;
        end_h           = end_h > in_h ? in_h : end_h;

        int start_w = w * stride_w - pad_w;
        int end_w   = start_w + window_w;
        start_w     = start_w < 0 ? 0 : start_w;
        end_w       = end_w > in_w ? in_w : end_w;

        int in_offset = n * in_n_stride + c * in_c_stride;
        for (int i = start_h; i < end_h; i++) {
            for (int j = start_w; j < end_w; j++) {
                float data = in_data[in_offset + i * in_h_stride + j * in_w_stride];
                if (data > max_data) {
                    max_data = data;
                    ;
                    max_index = i * in_w + j;
                }
            }
        }
        out_data[tid]  = max_data;
        out_index[tid] = max_index;
    }
}
