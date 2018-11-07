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

#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void Concat_normal(
        const int nthreads,
        __global const float* in_data,
        const int num_concats,
        const int concat_size,
        const int top_concat_axis,
        const int bottom_concat_axis,
        const int offset_concat_axis,
        __global float* out_data) {
    int index = get_global_id(0);
    if (index < nthreads) {
        const int total_concat_size = concat_size * bottom_concat_axis;
        const int concat_num        = index / total_concat_size;
        const int concat_index      = index % total_concat_size;
        const int top_index =
                concat_index + (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
        out_data[top_index] = in_data[index];
    }
}
__kernel void Concat_2d_impl(
        const int inner_size,
        const int num_concats,
        __global const float* in_data,
        const int concat_size,
        const int out_concat_axis,
        const int offset_concat_axis,
        __global float* out_data) {
    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    int wgidx = get_group_id(0);
    int wgidy = get_group_id(1);

    int wgsizex = get_local_size(0);
    int wgsizey = get_local_size(1);

    // int idx_inner = lidx + wgidx * wgsizex;
    // int idx_outer = lidy + wgidy * wgsizey;

    int idx_inner = get_global_id(0);
    int idx_outer = get_global_id(0);

    if ((idx_inner < inner_size) && (idx_outer < num_concats)) {
        int idx_input = idx_outer * inner_size + idx_inner;
        int idx_output =
                (idx_outer * out_concat_axis + offset_concat_axis) * concat_size + idx_inner;
        out_data[idx_output] = in_data[idx_input];
    }
}
