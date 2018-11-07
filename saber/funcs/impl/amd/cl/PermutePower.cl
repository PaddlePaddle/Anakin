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
#define BLOCK_SIZE 256
__kernel void ker_permute_power_fwd(
        global float* out_data,
        const int num_axes,
        const int count,
        global const int* permute_order,
        global const int* new_steps,
        global const int* old_steps,
        const float scale,
        const float shift,
        const float power,
        global const float* in_data) {
    int global_idx = get_global_id(0);
    int org_idx    = global_idx;
    int in_idx     = 0;
    for (int i = 0; i < num_axes; i++) {
        int order    = permute_order[i];
        int new_step = new_steps[i];
        int old_step = old_steps[order];
        in_idx += (org_idx / new_step) * old_step;
        org_idx %= new_step;
    }
    out_data[global_idx] = pow(scale * in_data[in_idx] + shift, power);
}

__kernel void ker_permute_power_fwd_transpose(
        global float* out_data,
        const int out_h,
        const int out_w,
        const float scale,
        const float shift,
        const float power,
        global const float* in_data) {
    local float tile[3][BLOCK_SIZE];
    int global_idx = get_global_id(0);
    int local_idx  = get_local_id(0);
    if (global_idx < out_w) {
        tile[0][local_idx] = pow(in_data[global_idx * out_h + 0] * scale + shift, power);
        tile[1][local_idx] = pow(in_data[global_idx * out_h + 1] * scale + shift, power);
        tile[2][local_idx] = pow(in_data[global_idx * out_h + 2] * scale + shift, power);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_idx < out_w) {
        out_data[0 * out_w + global_idx] = tile[0][local_idx];
        out_data[1 * out_w + global_idx] = tile[1][local_idx];
        out_data[2 * out_w + global_idx] = tile[2][local_idx];
    }
}

__kernel void ker_permute_scale_fwd(
        global float* out_data,
        const int num_axes,
        const int count,
        global const int* permute_order,
        global const int* new_steps,
        global const int* old_steps,
        const float scale,
        const float shift,
        global const float* in_data) {
    int global_idx = get_global_id(0);
    int org_idx    = global_idx;
    int in_idx     = 0;
    for (int i = 0; i < num_axes; i++) {
        int order    = permute_order[i];
        int new_step = new_steps[i];
        int old_step = old_steps[order];
        in_idx += (org_idx / new_step) * old_step;
        org_idx %= new_step;
    }
    out_data[global_idx] = scale * in_data[in_idx] + shift;
}

__kernel void ker_permute_scale_fwd_transpose(
        global float* out_data,
        const int out_h,
        const int out_w,
        const float scale,
        const float shift,
        global const float* in_data) {
    local float tile[3][BLOCK_SIZE];
    int global_idx = get_global_id(0);
    int local_idx  = get_local_id(0);
    if (global_idx < out_w) {
        tile[0][local_idx] = in_data[global_idx * out_h + 0] * scale + shift;
        tile[1][local_idx] = in_data[global_idx * out_h + 1] * scale + shift;
        tile[2][local_idx] = in_data[global_idx * out_h + 2] * scale + shift;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_idx < out_w) {
        out_data[0 * out_w + global_idx] = tile[0][local_idx];
        out_data[1 * out_w + global_idx] = tile[1][local_idx];
        out_data[2 * out_w + global_idx] = tile[2][local_idx];
    }
}

__kernel void ker_nhwc_to_nchw_scale(
        global float* out_data,
        const int n,
        const int c,
        const int h,
        const int w,
        const int out_stride_n,
        const int out_stride_c,
        const int out_stride_h,
        const int out_stride_w,
        const int in_stride_n,
        const int in_stride_c,
        const int in_stride_h,
        const int in_stride_w,
        const float scale,
        const float shift,
        global const float* in_data) {
    local float tile[3][BLOCK_SIZE];
    int global_idx = get_global_id(0);
    int local_idx  = get_local_id(0);
    int w_id       = global_idx % w;
    int h_id       = (global_idx / w) % h;
    int n_id       = (global_idx / (h * w)) % n;
    int in_offset  = n_id * in_stride_n + h_id * in_stride_h + w_id * in_stride_w;
    int out_offset = n_id * out_stride_n + h_id * out_stride_h + w_id * out_stride_w;
    if (global_idx < n * h * w) {
        tile[0][local_idx] = in_data[in_offset + 0] * scale + shift;
        tile[1][local_idx] = in_data[in_offset + 1] * scale + shift;
        tile[2][local_idx] = in_data[in_offset + 2] * scale + shift;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_idx < n * h * w) {
        out_data[0 * out_stride_c + out_offset] = tile[0][local_idx];
        out_data[1 * out_stride_c + out_offset] = tile[1][local_idx];
        out_data[2 * out_stride_c + out_offset] = tile[2][local_idx];
    }
}

__kernel void ker_nhwc_to_nchw_power(
        global float* out_data,
        const int n,
        const int c,
        const int h,
        const int w,
        const int out_stride_n,
        const int out_stride_c,
        const int out_stride_h,
        const int out_stride_w,
        const int in_stride_n,
        const int in_stride_c,
        const int in_stride_h,
        const int in_stride_w,
        const float scale,
        const float shift,
        const float power,
        global const float* in_data) {
    local float tile[3][BLOCK_SIZE];
    int global_idx = get_global_id(0);
    int local_idx  = get_local_id(0);
    int w_id       = global_idx % w;
    int h_id       = (global_idx / w) % h;
    int n_id       = (global_idx / (h * w)) % n;
    int in_offset  = n_id * in_stride_n + h_id * in_stride_h + w_id * in_stride_w;
    int out_offset = n_id * out_stride_n + h_id * out_stride_h + w_id * out_stride_w;
    if (global_idx < n * h * w) {
        tile[0][local_idx] = pow(in_data[in_offset + 0] * scale + shift, power);
        tile[1][local_idx] = pow(in_data[in_offset + 1] * scale + shift, power);
        tile[2][local_idx] = pow(in_data[in_offset + 2] * scale + shift, power);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_idx < n * h * w) {
        out_data[0 * out_stride_c + out_offset] = tile[0][local_idx];
        out_data[1 * out_stride_c + out_offset] = tile[1][local_idx];
        out_data[2 * out_stride_c + out_offset] = tile[2][local_idx];
    }
}

__kernel void ker_permute_power_valid_fwd(
        global float* out_data,
        const int num_axes,
        const int count,
        global const int* permute_order,
        global const int* new_steps,
        global const int* old_steps,
        global const int* new_valid_shape,
        const float scale,
        const float shift,
        const float power,
        global const float* in_data) {

    int global_idx       = get_global_id(0);
    int in_idx           = 0;
    int out_idx          = 0;
    int new_valid_stride = 1;

    for (int i = num_axes - 1; i >= 0; --i) {
        int order    = permute_order[i];
        int new_step = new_steps[i];
        int old_step = old_steps[order];
        int id       = (global_idx / new_valid_stride) % new_valid_shape[i];
        in_idx += id * old_step;
        out_idx += id * new_step;
        new_valid_stride *= new_valid_shape[i];
    }
    out_data[out_idx] = pow(in_data[in_idx] * scale + shift, power);
}

__kernel void ker_permute_scale_valid_fwd(
        global float* out_data,
        const int num_axes,
        const int count,
        global const int* permute_order,
        global const int* new_steps,
        global const int* old_steps,
        global const int* new_valid_shape,
        const float scale,
        const float shift,
        global const float* in_data) {

    int global_idx       = get_global_id(0);
    int in_idx           = 0;
    int out_idx          = 0;
    int new_valid_stride = 1;

    for (int i = num_axes - 1; i >= 0; --i) {
        int order    = permute_order[i];
        int new_step = new_steps[i];
        int old_step = old_steps[order];
        int id       = (global_idx / new_valid_stride) % new_valid_shape[i];
        in_idx += id * old_step;
        out_idx += id * new_step;
        new_valid_stride *= new_valid_shape[i];
    }
    out_data[out_idx] = in_data[in_idx] * scale + shift;
}
