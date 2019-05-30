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
reduce_mean(int total_size, int inner_size, __global const float* src, __global float* mean) {
    // int  blockSize = 256;
    int blockSize  = get_local_size(0);
    int blockIdx_x = get_group_id(0);
    __local float sdata[256]; // CUDA_NUM_THREADS
    int tid    = get_local_id(0);
    sdata[tid] = 0.0f;

    for (int j = tid; j < inner_size; j += blockSize) {
        sdata[tid] += src[blockIdx_x * inner_size + j];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (blockSize >= 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < 32) {
        // volatile float  *vsum = sdata;
        volatile __local float* vsum = sdata;

        // fault__local float  *vsum = sdata;
        if (blockSize >= 64) {
            vsum[tid] += vsum[tid + 32];
        }

        if (blockSize >= 32) {
            vsum[tid] += vsum[tid + 16];
        }

        if (blockSize >= 16) {
            vsum[tid] += vsum[tid + 8];
        }

        if (blockSize >= 8) {
            vsum[tid] += vsum[tid + 4];
        }

        if (blockSize >= 4) {
            vsum[tid] += vsum[tid + 2];
        }

        if (blockSize >= 2) {
            vsum[tid] += vsum[tid + 1];
        }

        //! write result for this block to global mem
        if (tid == 0) {
            vsum[0]          = vsum[0] / inner_size;
            mean[blockIdx_x] = vsum[0];
            // printf("mean: %d, %.6f\n", blockIdx_x, vsum[0]);
        }
    }
}

__kernel void reduce_std(
    int total_size,
    int inner_size,
    const float eps,
    __global const float* src,
    __global const float* mean,
    __global float* std) {

    __local float sdata[256];
    int tid    = get_local_id(0);
    sdata[tid] = (float)0.0f;
    // int blockSize = 256;
    int blockSize  = get_local_size(0);
    int blockIdx_x = get_group_id(0);

    for (int j = tid; j < inner_size; j += blockSize) {
        sdata[tid] += (src[blockIdx_x * inner_size + j] - mean[blockIdx_x])
                      * (src[blockIdx_x * inner_size + j] - mean[blockIdx_x]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (blockSize >= 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < 32) {
        // volatile float *vsum = sdata;
        volatile __local float* vsum = sdata;

        if (blockSize >= 64) {
            vsum[tid] += vsum[tid + 32];
        }

        if (blockSize >= 32) {
            vsum[tid] += vsum[tid + 16];
        }

        if (blockSize >= 16) {
            vsum[tid] += vsum[tid + 8];
        }

        if (blockSize >= 8) {
            vsum[tid] += vsum[tid + 4];
        }

        if (blockSize >= 4) {
            vsum[tid] += vsum[tid + 2];
        }

        if (blockSize >= 2) {
            vsum[tid] += vsum[tid + 1];
        }

        //! write result for this block to global mem
        if (tid == 0) {
            vsum[0] = vsum[0] / inner_size;
            // printf("std pre: %d, %.6f\n", blockIdx_x, vsum[0]);
            // std[blockIdx_x] = 1.f / (sqrtf(vsum[0]) + eps);
            std[blockIdx_x] = 1.f / (sqrt(vsum[0]) + eps);
            // printf("std: %d, %.6f\n", blockIdx_x, std[blockIdx_x]);
        }
    }
}

//! normalize with scale and bias
__kernel void normalize_with_scale_bias(
    int total_size,
    int inner_size,
    __global const float* mean,
    __global const float* std,
    __global const float* scale,
    __global const float* bias,
    __global const float* src,
    __global float* dst,
    int flag_scale,
    int flag_bias) {
    int idx = get_global_id(0);

    if (idx < total_size) {
        int outer_idx = (idx / inner_size);
        int inner_idx = (idx % inner_size);

        if (flag_scale) {
            if (flag_bias) {
                dst[idx] = (src[idx] - mean[outer_idx]) * std[outer_idx] * scale[inner_idx]
                           + bias[inner_idx];
            } else {
                dst[idx] = (src[idx] - mean[outer_idx]) * std[outer_idx] * scale[inner_idx];
            }
        } else {
            if (flag_bias) {
                dst[idx] = (src[idx] - mean[outer_idx]) * std[outer_idx] + bias[inner_idx];
            } else {
                dst[idx] = (src[idx] - mean[outer_idx]) * std[outer_idx];
            }
        }
    }
}
