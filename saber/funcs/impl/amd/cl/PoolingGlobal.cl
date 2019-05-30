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

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F /* max value */
#endif

#define Pooling_max 1
#define Pooling_average_include_padding 2
#define Pooling_average_exclude_padding 3

#ifndef MLO_CONV_BIAS
#define MLO_CONV_BIAS 0
#endif

#ifndef MLO_CONV_PRELU
#define MLO_CONV_PRELU 0
#endif

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void PoolingGlobal(
    const __global float* src,
    __global float* dst,
#if MLO_CONV_BIAS
    const __global float* bias,
#endif
#if MLO_CONV_PRELU
    float negSlope,
#endif
    int N,
    int C,
    int H,
    int W,
    int pad_h,
    int pad_w) {
    __local float lcl_buffer[GROUP_SIZE];
    float self;
    float temp;

    int n   = get_global_id(2);
    int c   = get_global_id(1);
    int idx = get_local_id(0);
    int lds = get_local_size(0);

    int window = H * W;
    int dst_offset = n * C + c;
    int src_offset = dst_offset * window;

#if POOLING_TYPE == Pooling_max
    self = -FLT_MAX;
#else
    self = 0;
#endif

    for (int i = idx; i < window; i += lds) {
        temp = src[src_offset + i];
#if MLO_CONV_BIAS
        temp += bias[c];
#endif
#if MLO_CONV_PRELU
        temp = (temp > 0) ? temp : temp * negSlope;
#endif

#if POOLING_TYPE == Pooling_max
        self = (self > temp) ? self : temp;
#else
        self += temp;
#endif
    }

    lcl_buffer[idx] = self;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (lds = lds >> 1; lds >= 1; lds = lds >> 1) {
        if (idx < lds) {
#if POOLING_TYPE == Pooling_max
            self = lcl_buffer[idx];
            temp = lcl_buffer[idx + lds];
            lcl_buffer[idx] = (self > temp) ? self : temp;
#else
            lcl_buffer[idx] += lcl_buffer[idx + lds];
#endif
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (idx == 0) {
#if POOLING_TYPE == Pooling_max
        dst[dst_offset] = lcl_buffer[0];
#elif POOLING_TYPE == Pooling_average_include_padding
        dst[dst_offset] = lcl_buffer[0] / ((H + pad_h * 2) * (W + pad_w * 2));
#else
        dst[dst_offset] = lcl_buffer[0] / window;
#endif
    }
}
