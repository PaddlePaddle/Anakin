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

__kernel void Axpy(int n,
                   int img_size,
                   __global const float* scale,
                   __global const float* x,
                   __global const float* y,
                   __global float* dst) {
    int idx      = get_global_id(0);
    int scale_id = idx / img_size;

    if (idx < n) {
        dst[idx] = scale[scale_id] * x[idx] + y[idx];
    }
}

__kernel void Axpy_float2(int n,
                          int img_size,
                          __global const float* scale,
                          __global const float2* x,
                          __global const float2* y,
                          __global       float2* dst) {
    int idx      = get_global_id(0);
    int scale_id = idx / (img_size >> 1);

    if (idx < n) {
        dst[idx] = scale[scale_id] * x[idx] + y[idx];
    }
}

__kernel void Axpy_float4(int n,
                          int img_size,
                          __global const float* scale,
                          __global const float4* x,
                          __global const float4* y,
                          __global       float4* dst) {
    int idx      = get_global_id(0);
    int scale_id = idx / (img_size >> 2);

    if (idx < n) {
        dst[idx] = scale[scale_id] * x[idx] + y[idx];
    }
}
__kernel void Axpy_float8(int n,
                          int img_size,
                          __global const float* scale,
                          __global const float8* x,
                          __global const float8* y,
                          __global       float8* dst) {
    int idx      = get_global_id(0);
    int scale_id = idx / (img_size >> 3);

    if (idx < n) {
        dst[idx] = scale[scale_id] * x[idx] + y[idx];
    }
}
