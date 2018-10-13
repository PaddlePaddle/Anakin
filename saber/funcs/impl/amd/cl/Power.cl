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
__kernel void ker_power_fwd(
        global float* out_data,
        const int count,
        const float scale,
        const float shift,
        const float power,
        global const float* in_data) {
    int global_idx       = get_global_id(0);
    out_data[global_idx] = pow(in_data[global_idx] * scale + shift, power);
}

__kernel void ker_scale_fwd(
        global float* out_data,
        const int count,
        const float scale,
        const float shift,
        global const float* in_data) {
    int global_idx       = get_global_id(0);
    out_data[global_idx] = in_data[global_idx] * scale + shift;
}

__kernel void ker_power_stride_fwd(
        global float* out_data,
        const int count,
        const float scale,
        const float shift,
        const float power,
        global const int* out_shape,
        global const int* out_stride,
        global const int* in_stride,
        const int num_axis,
        global const float* in_data) {

    int global_idx   = get_global_id(0);
    int in_offset    = 0;
    int out_offset   = 0;
    int valid_stride = 1;
    for (int i = num_axis - 1; i >= 0; --i) {
        int id = (global_idx / valid_stride) % out_shape[i];
        in_offset += id * in_stride[i];
        out_offset += id * out_stride[i];
        valid_stride *= out_shape[i];
    }
    out_data[out_offset] = pow(in_data[in_offset] * scale + shift, power);
}

__kernel void ker_scale_stride_fwd(
        global float* out_data,
        const int count,
        const float scale,
        const float shift,
        global const int* out_shape,
        global const int* out_stride,
        global const int* in_stride,
        const int num_axis,
        global const float* in_data) {
    int global_idx   = get_global_id(0);
    int in_offset    = 0;
    int out_offset   = 0;
    int valid_stride = 1;
    for (int i = num_axis - 1; i >= 0; --i) {
        int id = (global_idx / valid_stride) % out_shape[i];
        in_offset += id * in_stride[i];
        out_offset += id * out_stride[i];
        valid_stride *= out_shape[i];
    }
    // printf("%d, %d, %d\n", tid, in_offset, out_offset);
    out_data[out_offset] = in_data[in_offset] * scale + shift;
    // printf("out_offset:%d, %f\n", out_offset, out_data[out_offset]);
}
