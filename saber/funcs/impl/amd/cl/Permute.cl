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
#define TRANS_BLOCK_SIZE 16
#define BLOCK_SIZE 256

__kernel void ker_permute_fwd(
        global float* out_data,
        const int num_axes,
        const int count,
        global const int* permute_order,
        global const int* new_steps,
        global const int* old_steps,
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
    out_data[global_idx] = in_data[in_idx];
}

__kernel void ker_permute_fwd_new_shape(
        global float* out_data,
        const int num_axes,
        const int count,
        global const int* permute_order,
        global const int* new_steps,
        global const int* old_steps,
        global const int* new_valid_shape,
        global const float* in_data) {
    int global_idx = get_global_id(0);

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
    out_data[out_idx] = in_data[in_idx];
}
/*in this kernel, we suppose img with format (1, h, w, c) tranform to (1, c, h, w),
and c = 3. out_h = c, out_w = h * w. each thread process one pixel*/
__kernel void ker_permute_fwd_transpose(
        global float* out_data,
        const int out_h,
        const int out_w,
        global const float* in_data) {
    int global_idx = get_global_id(0);
    int local_idx  = get_local_id(0);
    local float tile[3][BLOCK_SIZE];
    if (global_idx < out_w) {
        int offset         = global_idx * out_h;
        tile[0][local_idx] = in_data[offset];
        tile[1][local_idx] = in_data[offset + 1];
        tile[2][local_idx] = in_data[offset + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_idx < out_w) {
        out_data[0 * out_w + global_idx] = tile[0][local_idx];
        out_data[1 * out_w + global_idx] = tile[1][local_idx];
        out_data[2 * out_w + global_idx] = tile[2][local_idx];
    }
}

__kernel void ker_permute_fwd_transpose_stride(
        global float* out_data,
        const int n,
        const int c,
        const int h,
        const int w,
        global const int* out_stride,
        global const int* in_stride,
        global const float* in_data) {
    int global_idx = get_global_id(0);
    int local_idx  = get_local_id(0);
    local float tile[3][BLOCK_SIZE];
    int out_w_id   = global_idx % w;
    int out_h_id   = (global_idx / w) % h;
    int out_n_id   = global_idx / (h * w);
    int out_offset = out_n_id * out_stride[0] + out_h_id * out_stride[2] + out_w_id * out_stride[3];
    int in_offset  = out_n_id * in_stride[0] + out_h_id * in_stride[1] + out_w_id * in_stride[2];
    if (global_idx < n * h * w) {
        tile[0][local_idx] = in_data[in_offset];
        tile[1][local_idx] = in_data[in_offset + 1];
        tile[2][local_idx] = in_data[in_offset + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_idx < n * h * w) {
        out_data[out_offset + out_stride[1] * 0] = tile[0][local_idx];
        out_data[out_offset + out_stride[1] * 1] = tile[1][local_idx];
        out_data[out_offset + out_stride[1] * 2] = tile[2][local_idx];
    }
}

/*in this kernel, we suppose img with format (1, c, h, w) tranform to (1, h, w, c),
and out_h = h*w, out_w = c. each thread process one data. we use share memory*/
/* 2d part, so block is x,y*/
__kernel void ker_transpose(
        global float* out_data,
        const int out_h,
        const int out_w,
        global const float* in_data) {
    local float tile[TRANS_BLOCK_SIZE][TRANS_BLOCK_SIZE];
    int global_idx = get_global_id(0); // in index
    int global_idy = get_global_id(1); // in index
    int local_idx  = get_local_id(0);
    int local_idy  = get_local_id(1);

    if (global_idx < out_h && global_idy < out_w) {
        tile[local_idx][local_idy] = in_data[global_idx + global_idy * out_h];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_idx < out_h && global_idy < out_w) {
        out_data[global_idx * out_w + global_idy] = tile[local_idx][local_idy];
    }
}

__kernel void ker_nchw_to_nhwc(
        global float* out_data,
        const int n,
        const int c,
        const int h,
        const int w,
        global const int* out_stride,
        global const int* in_stride,
        global const float* in_data) {
    local float tile[TRANS_BLOCK_SIZE][TRANS_BLOCK_SIZE];
    int global_idx = get_global_id(0); // in index
    int global_idy = get_global_id(1); // in index
    int local_idx  = get_local_id(0);
    int local_idy  = get_local_id(1);
    int w_id       = global_idy % w;
    int h_id       = global_idy / w;
    int c_id       = global_idx % c;
    int n_id       = global_idx / c;
    int in_offset =
            n_id * in_stride[0] + c_id * in_stride[1] + h_id * in_stride[2] + w_id * in_stride[3];
    int out_offset = n_id * out_stride[0] + h_id * out_stride[1] + w_id * out_stride[2]
                     + c_id * out_stride[3];
    if (global_idx < n * c && global_idy < h * w) {
        tile[local_idx][local_idy] = in_data[in_offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_idx < n * c && global_idy < h * w) {
        out_data[out_offset] = tile[local_idx][local_idy];
    }
}
