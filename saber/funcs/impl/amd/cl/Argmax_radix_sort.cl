

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

__kernel void topk_radix_sort2(
    __global float* __restrict out_data,
    int n,
    int inner_dim,
    int top_k,
    int out_max_val,
    __global const float* __restrict key_data,
    __global const int* __restrict value_data) {

    const int tid = get_global_id(0);

    if (tid < inner_dim) {
        if (out_max_val) {
            out_data[tid] = value_data[inner_dim - tid - 1];
            out_data[tid + top_k] = key_data[inner_dim - tid - 1];
        } else {
            out_data[tid] = value_data[inner_dim - tid - 1];
        }
    }
}
__kernel void topk_radix_sort(
    __global float* __restrict out_data,
    int n,
    int inner_dim,
    int top_k,
    int out_max_val,
    __global const float* __restrict key_data,
    __global const int* __restrict value_data) {

    const int tid = get_global_id(0);

    if (tid < inner_dim) {
        if (out_max_val) {
            const int offset = n * 2 * top_k;
            __global float* real_out_data = out_data + offset;
            real_out_data[tid] = value_data[inner_dim - tid - 1];
            real_out_data[tid + top_k] = key_data[inner_dim - tid - 1];
        } else {
            const int offset = n * 1 * top_k;
            __global float* real_out_data = out_data + offset;
            real_out_data[tid] = value_data[inner_dim - tid - 1];
        }

    }
}

