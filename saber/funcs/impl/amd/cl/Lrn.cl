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
Lrn(__global const _FLOAT* __restrict in_data,
    __global _FLOAT* __restrict out_data,
    int in_n_stride,
    int in_c_stride,
    int in_h_stride,
    int in_w_stride,
    int in_n,
    int in_c,
    int in_h,
    int in_w,
    _FLOAT alpha,
    _FLOAT beta,
    _FLOAT k,
    int size) {

    const int count       = in_n * in_h * in_w;
    const int global_size = get_global_size(0);
    int tid               = get_global_id(0);

    for (; tid < count; tid += global_size) {
        const int n        = tid / (in_h * in_w);
        const int h        = (tid / in_w) % in_h;
        const int w        = tid % in_w;
        const int offset   = n * in_n_stride + h * in_h_stride + w * in_w_stride;
        in_data            = in_data + offset;
        out_data           = out_data + offset;
        const int pre_pad  = (size - 1) / 2;
        const int post_pad = size - pre_pad - 1;

        _FLOAT accum = 0;
        int index    = 0;
        while (index < in_c + post_pad) {
            if (index < in_c) {
                _FLOAT val = in_data[index * in_c_stride];
                accum += val * val;
            }
            if (index >= size) {
                _FLOAT val = in_data[(index - size) * in_c_stride];
                accum -= val * val;
            }
            if (index >= post_pad) {
                _FLOAT mid    = k + accum * alpha;
                int off       = (index - post_pad) * in_c_stride;
                out_data[off] = in_data[off] * pow(mid, -beta);
            }
            ++index;
        }
    }
}