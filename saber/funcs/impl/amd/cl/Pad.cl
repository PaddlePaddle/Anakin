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
#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8

__kernel void
Pad(__global const _FLOAT* __restrict in_data,
    __global _FLOAT* __restrict out_data,
    int in_n_stride,
    int in_c_stride,
    int in_h_stride,
    int in_w_stride,
    int out_n_stride,
    int out_c_stride,
    int out_h_stride,
    int out_w_stride,
    int in_n,
    int in_c,
    int in_h,
    int in_w,
    int img_offset) {
    out_data += img_offset;
    const int count       = in_n * in_c * in_h * in_w;
    const int global_size = get_global_size(0);
    int tid               = get_global_id(0);

    for (; tid < count; tid += global_size) {
        int n          = (tid / in_n_stride) % in_n;
        int c          = (tid / in_c_stride) % in_c;
        int h          = (tid / in_h_stride) % in_h;
        int w          = (tid / in_w_stride) % in_w;
        int out_offset = n * out_n_stride + c * out_c_stride + h * out_h_stride + w * out_w_stride;
        out_data[out_offset] = in_data[tid];
    }
}
