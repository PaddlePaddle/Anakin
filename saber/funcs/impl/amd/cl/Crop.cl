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

__kernel void
Crop(__global float* __restrict out_data,
     __global const float* __restrict in_data,
     int in_n_stride,
     int in_c_stride,
     int in_h_stride,
     int in_w_stride,
     int out_n_stride,
     int out_c_stride,
     int out_h_stride,
     int out_w_stride,
     int out_n,
     int out_c,
     int out_h,
     int out_w,
     int img_offset) {

    // img_offset
    in_data += img_offset;
    int tid         = get_global_id(0);
    int global_size = get_global_size(0);
    int count       = out_n * out_c * out_h * out_w;

    for (; tid < count; tid += global_size) {
        int n = (tid / out_n_stride) % out_n;
        int c = (tid / out_c_stride) % out_c;
        int h = (tid / out_h_stride) % out_h;
        int w = (tid / out_w_stride) % out_w;

        int in_offset = n * in_n_stride + c * in_c_stride + h * in_h_stride + w * in_w_stride;
        out_data[tid] = in_data[in_offset];
    }
}