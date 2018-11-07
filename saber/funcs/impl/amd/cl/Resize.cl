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
__kernel void resize_2d_kernel(
        const int wout,
        const int hout,
        const int num,
        const int channels,
        const int dst_stride_w,
        const int dst_stride_h,
        const int dst_stride_c,
        const int dst_stride_batch,
        const int win,
        const int hin,
        const int src_stride_w,
        const int src_stride_h,
        const int src_stride_c,
        const int src_stride_batch,
        const float scale_w,
        const float scale_h,
        global const float* src,
        global float* dst) {
    int local_idx   = get_local_id(0);
    int local_idy   = get_local_id(1);
    int group_idx   = get_group_id(0);
    int group_idy   = get_group_id(1);
    int local_sizex = get_local_size(0);
    int local_sizey = get_local_size(1);
    int dst_w       = group_idx * local_sizex + local_idx;
    int dst_h       = group_idy * local_sizey + local_idy;

    if (dst_w < wout && dst_h < hout) {
#if 0 //! more precise method
        float fw = scale_w * (dst_w + 0.5f) - 0.5f;
        float fh = scale_h * (dst_h + 0.5f) - 0.5f;
        int src_w = int(floor(fw));
        int w = src_w + 1;
        int src_h = int(floor(fh));
        int h = src_h + 1;
#else
        float fh        = scale_h * dst_h;
        float fw        = scale_w * dst_w;
        const int src_h = (int)fh;
        const int src_w = (int)fw;
        int w           = src_w + 1;
        int h           = src_h + 1;
#endif
        fh -= src_h;
        fw -= src_w;
        const float w_h0 = 1.0f - fh;
        const float w_w0 = 1.0f - fw;
        const float w_h1 = fh;
        const float w_w1 = fw;

        float w_00 = w_h0 * w_w0;
        float w_01 = w_h0 * w_w1;
        float w_10 = w_h1 * w_w0;
        float w_11 = w_h1 * w_w1;

        int hl = src_h * src_stride_h;
        int hh = h * src_stride_h;
        int wl = src_w * src_stride_w;
        int wh = w * src_stride_w;

        int src_indexTL = hl + wl;
        int src_indexTR = hl + wh;
        int src_indexBL = hh + wl;
        int src_indexBR = hh + wh;

        int dst_index = dst_w * dst_stride_w + dst_h * dst_stride_h;
#if 1
        // for (int i = 0; i < num; ++i) {
        for (int j = 0; j < channels; ++j) {
#if 0
				float tl = src[src_indexTL];
				float tr = w > win ? 0 : src[1];
				float bl = h > hin ? 0 : src[2];
				float br = (w > win || h > hin) ? 0 : src[3];
#else
#if 0
                float tl = (src_w < 0 || src_h < 0)? 0 : src[src_indexTL];
                float tr = (w > win || src_h < 0)? 0 : src[src_indexTR];
                float bl = (src_w < 0 || h > hin)? 0 : src[src_indexBL];
                float br = (w > win || h > hin)? 0 : src[src_indexBR];
#else
            float tl = src[src_indexTL];
            float tr = w >= win ? 0 : src[src_indexTR];               // w > win? 0 :
            float bl = h >= hin ? 0 : src[src_indexBL];               // h > hin? 0 :
            float br = (w >= win || h >= hin) ? 0 : src[src_indexBR]; //(w > win || h > hin)? 0 :
#endif
#endif
            dst[dst_index] = (float)(w_00 * tl + w_01 * tr + w_10 * bl + w_11 * br);
            src_indexBR += src_stride_c;
            src_indexBL += src_stride_c;
            src_indexTR += src_stride_c;
            src_indexTL += src_stride_c;
            dst_index += dst_stride_c;
        }
        //}
#endif
    }
}

// local = 8, 8, 1
// global = (w_out + 8 - 1), (h_out + 8 - 1), 1
