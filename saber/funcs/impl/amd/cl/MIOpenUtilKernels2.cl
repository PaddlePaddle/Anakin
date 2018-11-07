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
Col2Im(global float* col,
       const int col_h,
       const int col_w,
       const int wei_h,
       const int wei_w,
       const int pad_h,
       const int pad_w,
       const int stride_h,
       const int stride_w,
       const int dilation_h,
       const int dilation_w,
       const int height,
       const int width,
       global float* im,
       int im_offset) {
    global float* im_off = im + im_offset;
    int gid              = (int)get_global_id(0);

    int im_ch  = gid / (width * height);
    int im_pix = gid % (width * height);
    int im_h   = (im_pix / width) + pad_h;
    int im_w   = (im_pix % width) + pad_w;

    int start_h = (im_h < dilation_h * (wei_h - 1) + 1)
                          ? 0
                          : (im_h - (dilation_h * (wei_h - 1) + 1)) / stride_h + 1;
    int end_h   = min(col_h, im_h / stride_h + 1);
    int start_w = (im_w < dilation_w * (wei_w - 1) + 1)
                          ? 0
                          : (im_w - (dilation_w * (wei_w - 1) + 1)) / stride_w + 1;
    int end_w = min(col_w, im_w / stride_w + 1);

    int ch_offset = im_ch * col_w * col_h * wei_w * wei_h;
    col += ch_offset;

    float tmp = (float)0.0f;
    for (int cy = start_h; cy < end_h; cy++) {
        for (int cx = start_w; cx < end_w; cx++) {
            if ((im_h - cy * stride_h) % dilation_h == 0
                && (im_w - cx * stride_w) % dilation_w == 0) {
                int col_off_y = cy + (((im_h - cy * stride_h) / dilation_h) * wei_w * col_h);
                int col_off_x = cx + (((im_w - cx * stride_w) / dilation_w) * col_w * col_h);

                tmp += col[col_off_y * col_w + col_off_x];
            }
        }
    }
    im_off[gid] = tmp;
}
