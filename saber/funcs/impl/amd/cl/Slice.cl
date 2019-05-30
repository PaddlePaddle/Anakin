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
__attribute__((reqd_work_group_size(512,1,1)))
__kernel void Slice_normal_512(
        const int nthreads,
        __global const float* in_data,
        const int num_slices,
        const int slice_size,
        const int in_slice_axis_size,
        const int out_slice_axis_size,
        const int offset_slice_axis,
        __global float* out_data) {
    int index = get_global_id(0);
    if (index < nthreads ) {
        const int total_slice_size = slice_size * out_slice_axis_size;
        const int slice_num        = index / total_slice_size;
        const int slice_index      = index % total_slice_size;
        const int in_index =
                slice_index + (slice_num * in_slice_axis_size + offset_slice_axis) * slice_size;
        out_data[index] = in_data[in_index];
    }
}
__attribute__((reqd_work_group_size(512,1,1)))
__kernel void Slice_normal_512_f4(
        const int nthreads,
        __global const float * in_data,
        const int num_slices,
        const int slice_size,
        const int in_slice_axis_size,
        const int out_slice_axis_size,
        const int offset_slice_axis,
        __global float * out_data) {
    int index = get_global_id(0);
    __global float4*   vsrc ;
    __global float4*   vdst ;
    if (index < nthreads ) {
        const int total_slice_size = slice_size * out_slice_axis_size;
        const int slice_num        = index / (total_slice_size >> 2);
        int slice_index            = index % (total_slice_size >> 2);
        int top_index = index << 2;
        int bottom_index = ((slice_num * in_slice_axis_size + offset_slice_axis) * slice_size);
        vsrc = (__global float4 *)(& in_data[ bottom_index + slice_index * 4]);
        vdst = (__global float4 *)(& out_data[top_index]);
        *vdst = *vsrc;
    }
}

__kernel void Slice_normal(
        const int nthreads,
        __global const float* in_data,
        const int num_slices,
        const int slice_size,
        const int in_slice_axis_size,
        const int out_slice_axis_size,
        const int offset_slice_axis,
        __global float* out_data) {
    int index = get_global_id(0);
    if (index < nthreads) {
        const int total_slice_size = slice_size * out_slice_axis_size;
        const int slice_num        = index / total_slice_size;
        const int slice_index      = index % total_slice_size;
        const int in_index =
                slice_index + (slice_num * in_slice_axis_size + offset_slice_axis) * slice_size;
        out_data[index] = in_data[in_index];
    }
}
