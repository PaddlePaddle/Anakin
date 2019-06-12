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

#define CACHE_SIZE (CACHE_SIZE_1 * CACHE_SIZE_0)

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

__kernel void PoolingWithShare(
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
    int pad_w)
{
    __local float lcl_input[CACHE_SIZE];

    int ox = get_local_id(0);
    int oy = get_local_id(1);

    int ow = get_group_id(0) * get_local_size(0);
    int oh = get_group_id(1) * get_local_size(1);
    int nc = get_group_id(2);

#if MLO_CONV_BIAS
    int n = nc / C;
    int c = nc - n * C;
#endif

    int iw = ow * str_w;
    int ih = oh * str_h;

    int lcl_idx = oy * get_local_size(0) + ox;

    int in_channel_stride = H_in * W_in;

    // read input data to lcl
    int group_size = get_local_size(0) * get_local_size(1);
    int src_offset = nc * in_channel_stride;

    for(int i=lcl_idx; i<CACHE_SIZE; i+=group_size)
    {
        int src_y = i / CACHE_SIZE_0;
        int src_x = i - src_y * CACHE_SIZE_0;

        src_y = src_y + ih - pad_h;
        src_x = src_x + iw - pad_w;

        bool  vis  = ((src_y >= 0) && (src_y < H_in) && (src_x >= 0) && (src_x < W_in));
        float temp = vis ? src[src_offset + src_y * W_in + src_x] : OP_init();
#if MLO_CONV_BIAS
        temp = vis ? (temp + bias[c]) : temp;
#endif
#if MLO_CONV_PRELU
        temp = ((temp < 0) && vis) ? (temp * negSlope) : temp;
#endif
        lcl_input[i] = temp;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    oh += oy;
    ow += ox;

    float result = OP_init();
    int lcl_off = oy * str_h * CACHE_SIZE_0 + ox * str_w;

    for(int j=0; j<win_h; j++, lcl_off += CACHE_SIZE_0)
    {
        for(int i=0; i<win_w; i++)
        {
            result = OP(result, lcl_input[lcl_off + i]);
        }
    }

    if ((oh < H_out) && (ow < W_out))
    {
        int dst_off = (nc * H_out + oh) * W_out + ow;
        dst[dst_off] = OP_post_process(result, H_in, W_in, oh, ow, win_h, win_w, str_h, str_w, pad_h, pad_w);
    }
}

