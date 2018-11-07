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

__kernel void CtcAlign(
        __global float* out_data,
        __global int* out_offset,
        __global const float* in_data,
        __global const int* in_offset,
        const int seq_num,
        const int blank,
        const int merge_repeated,
        const int num_threads) {

    int tid = get_global_id(0);
    if (tid == 0) {
        int index = 0;
        for (int seq_id = 0; seq_id < seq_num; seq_id++) {
            float prev_token   = -1;
            out_offset[seq_id] = index;
            for (int i = in_offset[seq_id]; i < in_offset[seq_id + 1]; i++) {
                if (in_data[i] != blank && !(merge_repeated && in_data[i] == prev_token)) {
                    out_data[index++] = in_data[i];
                    prev_token        = in_data[i];
                }
            }
        }
        out_offset[seq_num] = index;
    }
}
