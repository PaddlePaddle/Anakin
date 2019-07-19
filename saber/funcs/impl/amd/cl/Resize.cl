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

#ifndef RESIZE_TYPE
#define RESIZE_TYPE RESIZE_CUSTOM
#endif

#define BILINEAR_ALIGN 0
#define BILINEAR_NO_ALIGN 1
#define RESIZE_CUSTOM 2

__kernel void resize_2D(
            global const float* src,
            global float* dst,
            const int num,
            const int channel,
            const int height_in,
            const int width_in,
            const int height_out,
            const int width_out,
            const int src_stride_batch,
            const int src_stride_c,
            const int src_stride_h,
            const int src_stride_w,
            const int dst_stride_batch,
            const int dst_stride_c,
            const int dst_stride_h,
            const int dst_stride_w,
            const float scale_h,
            const float scale_w)
{
    int index = get_global_id(0);

    if (index >= num * channel * height_out * width_out)
        return;

    int n = index / width_out / height_out / channel;
    int c = (index / width_out / height_out) % channel;
    int h = (index / width_out) % height_out;
    int w = index % width_out;

#if RESIZE_TYPE == BILINEAR_ALIGN
    float fh = h * scale_h;
    float fw = w * scale_w;

    int h_start = (int) fh;
    int h_id    = (h_start < height_in - 1) ? 1 : 0;
    int h_end   = h_start + h_id;

    int w_start = (int) fw;
    int w_in    = (w_start < width_in - 1) ? 1 : 0;
    int w_end   = w_start + w_in;
#elif RESIZE_TYPE == BILINEAR_NO_ALIGN
    float fh = scale_h * (h + 0.5f) - 0.5f;
    fh = (fh < 0.0f) ? 0.0f : fh;
    float fw = scale_w * (w + 0.5f) - 0.5f;
    fw = (fw < 0.0f) ? 0.0f : fw;

    int h_start = (int) fh;
    int h_id    = (h_start < height_in - 1) ? 1 : 0;
    int h_end   = h_start + h_id;

    int w_start = (int) fw;
    int w_in    = (w_start < width_in - 1) ? 1 : 0;
    int w_end   = w_start + w_in;
#elif RESIZE_TYPE == RESIZE_CUSTOM
    float fh = h * scale_h;
    float fw = w * scale_w;

    int h_start = (int) fh;
    int h_end   = h_start + 1;

    int w_start = (int) fw;
    int w_end   = w_start + 1;
#endif

    int nc_offset = n * src_stride_batch + c * src_stride_c;
    float tl = src[nc_offset + h_start * src_stride_h + w_start * src_stride_w];
    float tr = (w_end >= width_in ) ? 0 : src[nc_offset + h_start * src_stride_h + w_end * src_stride_w];
    float bl = (h_end >= height_in) ? 0 : src[nc_offset + h_end   * src_stride_h + w_start * src_stride_w];
    float br = ((w_end >= width_in ) || (h_end >= height_in)) ? 0 : src[nc_offset + h_end * src_stride_h + w_end * src_stride_w];

    fh = fh - h_start;
    fw = fw - w_start;

    const float w00 = (1.0f - fh) * (1.0f - fw);
    const float w01 = fw * (1.0f - fh);
    const float w10 = fh * (1.0f - fw);
    const float w11 = fw * fh;

    int dst_index = n * dst_stride_batch + c * dst_stride_c + h * dst_stride_h + w * dst_stride_w;
    dst[dst_index] = w00 * tl + w01 * tr + w10 * bl + w11 * br;
}

