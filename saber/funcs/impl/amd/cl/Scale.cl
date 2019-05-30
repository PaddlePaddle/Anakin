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

#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void Scale_singleBias(
    __global float* out_data,
    __global float* in_data,
    const float scale,
    const float bias,
    const int count) {
    int tid = get_global_id(0);

    if (tid < count) {
        out_data[tid] = scale * in_data[tid] + bias;
    }
}
__kernel void Scale_singleBias_float4(
    __global float4* out_data,
    __global float4* in_data,
    const float scale,
    const float bias,
    const int count) {
    int tid = get_global_id(0);

    if (tid < count) {
        out_data[tid] = scale * in_data[tid] + bias;
    }
}

__kernel void Scale_multiBias(
    __global float* out_data,
    __global float* in_data,
    __global float* scale_data,
    __global float* bias_data,
    const int count,
    const int scale_dim,
    const int inner_dim,
    const int bias_flag) {
    int tid = get_global_id(0);

    if (tid < count) {
        int scale_id = (tid / inner_dim) % scale_dim;
        float scale  = scale_data[scale_id];
        out_data[tid] = (bias_flag == 1)
                        ? scale * in_data[tid] + bias_data[scale_id]
                          : scale * in_data[tid];
    }
}
