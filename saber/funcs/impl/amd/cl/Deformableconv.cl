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
static inline float deformable_im2col_bilinear(
    const global float* bottom_data,
    const int data_width,
    const int height,
    const int width,
    float h,
    float w) {
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high;
    int w_high;

    if (h_low >= height - 1) {
        h_high = h_low = height - 1;
        h              = (float)h_low;
    } else {
        h_high = h_low + 1;
    }

    if (w_low >= width - 1) {
        w_high = w_low = width - 1;
        w              = (float)w_low;
    } else {
        w_high = w_low + 1;
    }

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;
    float v1 = bottom_data[h_low * data_width + w_low];
    float v2 = bottom_data[h_low * data_width + w_high];
    float v3 = bottom_data[h_high * data_width + w_low];
    float v4 = bottom_data[h_high * data_width + w_high];
    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

__kernel void deformable_im2col_gpu_kernel(
    const int n,
    global const float* data_im,
    global const float* data_offset,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int height_col,
    const int width_col,
    global float* data_col) {

    int global_idx  = get_global_id(0);
    const int w_col = global_idx % width_col;
    const int h_col = (global_idx / width_col) % height_col;
    const int c_im  = (global_idx / width_col) / height_col;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    // THIS IS THE TRUE CHANNEL
    const int deformable_group_index = c_im / channel_per_deformable_group;

    // input map coord(h_in, w_in)
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    // data_col (data & offset)
    global float* data_col_ptr      = data_col + (c_col * height_col + h_col) * width_col + w_col;
    const global float* data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
    const global float* data_offset_ptr =
        data_offset + deformable_group_index * 2 * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
            // offset_h and offset_w in the same channel
            const int data_offset_h_ptr =
                ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;

            const int data_offset_w_ptr =
                ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
            const float offset_h = data_offset_ptr[data_offset_h_ptr];
            const float offset_w = data_offset_ptr[data_offset_w_ptr];
            float val            = 0.f;
            const float h_im     = h_in + i * dilation_h + offset_h;
            const float w_im     = w_in + j * dilation_w + offset_w;

            if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
                const float map_h = i * dilation_h + offset_h;
                const float map_w = j * dilation_w + offset_w;
                // cur_height (from h_in to height)
                const int cur_height = height - h_in;
                const int cur_width  = width - w_in;
                val                  = deformable_im2col_bilinear(
                                           data_im_ptr, width, cur_height, cur_width, map_h, map_w);
            }

            *data_col_ptr = val;
            data_col_ptr += height_col * width_col;
        }
    }
}
__kernel void gpu_add_bias(
    global float* out_data,
    const int count,
    int in_n,
    int in_c,
    int in_h,
    int in_w,
    int in_n_stride,
    int in_c_stride,
    int in_h_stride,
    int in_w_stride,
    global const float* bias) {
    int global_idx = get_global_id(0);
    int read_w     = global_idx % in_w;
    int read_h     = (global_idx / (in_w)) % in_h;
    int read_c     = (global_idx / (in_h * in_w)) % in_c;
    int read_n     = (global_idx / (in_c * in_h * in_w)) % in_n;

    int in_idx = read_n * in_n_stride + read_c * in_c_stride + read_h * in_h_stride
                 + read_w * in_w_stride;

    float in_var     = out_data[in_idx];
    float in_bias    = bias[read_c];
    out_data[in_idx] = in_var + in_bias;
}
