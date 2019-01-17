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

__kernel void
Unpool(__global float* out_data,
       __global const float* in_data,
       __global const float* in_max_index,
       const int in_n_stride,
       const int in_c_stride,
       const int out_n_stride,
       const int out_c_stride,
       const int in_n,
       const int in_c,
       const int num_threads) {
    int tid = get_global_id(0);
    if (tid < num_threads) {
        int n                        = (tid / in_n_stride) % in_n;
        int c                        = (tid / in_c_stride) % in_c;
        int out_offset               = n * out_n_stride + c * out_c_stride;
        int index                    = in_max_index[tid];
        out_data[out_offset + index] = in_data[tid];
    }
}
