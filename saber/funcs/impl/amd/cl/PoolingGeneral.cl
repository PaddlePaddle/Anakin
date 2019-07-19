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

static inline float OP_init() {
#if POOLING_TYPE == Pooling_max
    return -FLT_MAX;
#else
    return 0;
#endif
}

static inline float OP(float accum, float temp) {
#if POOLING_TYPE == Pooling_max
    return (accum > temp) ? accum : temp;
#else
    return accum + temp;
#endif
}

static inline float OP_post_process(float value, int H_in, int W_in, int h, int w, int win_h,
                                    int win_w, int str_h, int str_w, int pad_h, int pad_w) {
#if POOLING_TYPE == Pooling_max
    return value;
#elif POOLING_TYPE == Pooling_average_include_padding
    int h_start = h * str_h;
    int w_start = w * str_w;
    int h_end   = h_start + win_h;
    int w_end   = w_start + win_w;

    h_start = (h_start >= 0) ? h_start : 0;
    w_start = (w_start >= 0) ? w_start : 0;

    h_end   = (h_end <= (H_in + pad_h * 2)) ? h_end : (H_in + pad_h * 2);
    w_end   = (w_end <= (W_in + pad_w * 2)) ? w_end : (W_in + pad_w * 2);

    return value / ((h_end - h_start) * (w_end - w_start));
#else
    int h_start = h * str_h - pad_h;
    int w_start = w * str_w - pad_w;
    int h_end   = h_start + win_h;
    int w_end   = w_start + win_w;

    h_start = (h_start >= 0) ? h_start : 0;
    w_start = (w_start >= 0) ? w_start : 0;

    h_end   = (h_end <= H_in) ? h_end : H_in;
    w_end   = (w_end <= W_in) ? w_end : W_in;

    return value / ((h_end - h_start) * (w_end - w_start));
#endif
}

__attribute__((reqd_work_group_size(GROUP_SIZE_0, GROUP_SIZE_1, 1)))
__kernel void PoolingGeneral(
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
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int win_h,
    int win_w,
    int str_h,
    int str_w,
    int pad_h,
    int pad_w) {
    __local float lcl_aggregate[GROUP_SIZE];

    int l_idx = get_local_id(1);
    int g_idx = get_group_id(1);
    int idx   = l_idx + g_idx * GROUP_SIZE_1;

    int NC = N * C;

    int stride_NC = H_out * W_out;
    int stride_H  = W_out;

    int nc = idx / stride_NC;
    int hw = idx - nc * stride_NC;
    int h  = hw / stride_H;
    int w  = hw - h * stride_H;

#if MLO_CONV_BIAS
    int n = nc / C;
    int c = nc - n * C;
#endif

    int a  = get_local_id(0);
    int adder = ADDER;

    float accum;
    float temp;

    int window = win_h * win_w;

    accum = OP_init();
    int src_off = nc * H_in * W_in;

    for (int i = a; i < window; i += ADDER) {
        int src_h = i / win_w;
        int src_w = i - src_h * win_w;

        src_h = src_h + h * str_h - pad_h;
        src_w = src_w + w * str_w - pad_w;

        bool vis = (nc < NC) && (src_h < H_in) && (src_h >= 0) && (src_w < W_in) && (src_w >= 0);
        int src_off2 = src_off + src_h * W_in + src_w;

        temp = vis ? src[src_off2] : OP_init();
#if MLO_CONV_BIAS
        temp = vis ? (temp + bias[c]) : temp;
#endif
#if MLO_CONV_PRELU
        temp = ((temp < 0) && vis) ? (temp * negSlope) : temp;
#endif
        accum = OP(accum, temp);
    }

    int agg_idx = l_idx * GROUP_SIZE_0 + a;
    lcl_aggregate[agg_idx] = accum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (adder = adder >> 1; adder >= 1; adder = adder >> 1) {
        if (a < adder) {
            lcl_aggregate[agg_idx] = OP(lcl_aggregate[agg_idx], lcl_aggregate[agg_idx + adder]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if (a == 0 && nc < NC) {
        int dst_offset  = (nc * H_out + h) * W_out + w;
        dst[dst_offset] = OP_post_process(lcl_aggregate[agg_idx], H_in, W_in, h, w, win_h, win_w, str_h,
                                          str_w, pad_h, pad_w);
    }
}
