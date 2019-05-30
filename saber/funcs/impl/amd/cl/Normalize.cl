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
#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8

inline void AtomicAdd(volatile __global _FLOAT* source, const _FLOAT operand) {
    union {
        unsigned int intVal;
        _FLOAT floatVal;
    } newVal;
    union {
        unsigned int intVal;
        _FLOAT floatVal;
    } prevVal;

    prevVal.floatVal = *source;

    while (true) {
        newVal.floatVal = prevVal.floatVal + operand;
        newVal.intVal   = atomic_cmpxchg(
                              (volatile __global unsigned int*)source, prevVal.intVal, newVal.intVal);

        // equal to pass
        if (newVal.intVal == prevVal.intVal) {
            break;
        }

        prevVal.intVal = newVal.intVal;
    }
}

__kernel void GpuPowReverse(
    int n,
    __global const _FLOAT* __restrict src,
    __global _FLOAT* __restrict dst,
    _FLOAT alpha,
    _FLOAT eps) {
    int idx         = get_global_id(0);
    int global_size = get_global_size(0);

    for (; idx < n; idx += global_size) {
        dst[idx] = 1 / (pow(src[idx], alpha) + eps);
    }
}



__kernel void ReduceAddAtomic(
    int inner_size,
    int p,
    int inner_group_num,
    __global const _FLOAT* __restrict src,
    __global _FLOAT* __restrict dst) {

    __local _FLOAT sdata[SHARE_MEMORY_DIM];
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);
    int tid = get_local_id(0);
    sdata[tid] = 0;

    int batch_id = group_id / inner_group_num;
    int count = inner_size / inner_group_num + 1;
    int src_offset = batch_id * inner_size;
    src = src + src_offset;
    int src_idx = (group_id - batch_id * inner_group_num) * count + tid;

    for (int tid_count = tid; tid_count < count
            && src_idx < inner_size; src_idx += local_size, tid_count += local_size) {
        //! L1 norm
        if (p == 1) {
            sdata[tid] += fabs(src[src_idx]);
        } else {
            //! L2 norm
            sdata[tid] += src[src_idx] * src[src_idx];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_size >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 64) {
        if (tid < 32) {
            sdata[tid] += sdata[tid + 32];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 32) {
        if (tid < 16) {
            sdata[tid] += sdata[tid + 16];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 16) {
        if (tid < 8) {
            sdata[tid] += sdata[tid + 8];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 8) {
        if (tid < 4) {
            sdata[tid] += sdata[tid + 4];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 4) {
        if (tid < 2) {
            sdata[tid] += sdata[tid + 2];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_size >= 2) {
        if (tid < 1) {
            sdata[tid] += sdata[tid + 1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //! write result for this block to global mem
    if (tid == 0) {
        AtomicAdd(dst + batch_id, *sdata);
    }

};

__kernel void NormalizeNoAcrossSpatial(
    int size_in_channel,
    int n,
    int channels,
    __global const _FLOAT* __restrict scale,
    __global const _FLOAT* __restrict bottom_data,
    __global _FLOAT* __restrict top_data,
    _FLOAT eps,
    int p,
    int has_scale,
    int shared) {

    int index             = get_global_id(0);
    const int global_size = get_global_size(0);

    for (; index < size_in_channel * n; index += global_size) {
        float sqr_sum        = 0.f;
        int num_index        = index / size_in_channel;
        int index_in_channel = index % size_in_channel;
        int data_index       = num_index * channels * size_in_channel + index_in_channel;

        for (int i = 0; i < channels; ++i) {
            //float src_val = fabs(bottom_data[data_index + i * size_in_channel]);
            if (p == 1) {
                sqr_sum += fabs(bottom_data[data_index + i * size_in_channel]);
            } else {
                sqr_sum += bottom_data[data_index + i * size_in_channel] *
                           bottom_data[data_index + i * size_in_channel];
            }
        }

        float norm;

        if (p == 1) {
            norm = 1.f / (sqr_sum + eps);
        } else {
            norm = 1.f / (sqrt(sqr_sum) + eps);
        }


        for (int i = 0; i < channels; ++i) {
            if (has_scale) {
                if (shared) {
                    top_data[data_index + i * size_in_channel] =
                        bottom_data[data_index + i * size_in_channel] * scale[0] * norm;
                } else {
                    top_data[data_index + i * size_in_channel] =
                        bottom_data[data_index + i * size_in_channel] * scale[i] * norm;
                }
            } else {
                top_data[data_index + i * size_in_channel] =
                    bottom_data[data_index + i * size_in_channel] * norm;
            }
        }
    }
}

//! normalize with scale
__kernel void NormalizeWithScale(
    int n,
    int inner_size,
    int channel_stride,
    int channel_size,
    __global const _FLOAT* __restrict norm,
    __global const _FLOAT* __restrict scale,
    __global const _FLOAT* __restrict src,
    __global _FLOAT* __restrict dst,
    int channel_shared) {
    int idx         = get_global_id(0);
    int global_size = get_global_size(0);

    for (; idx < n; idx += global_size) {
        int outer_idx = idx / inner_size;

        if (channel_shared) {
            dst[idx] = src[idx] * norm[outer_idx] * scale[0];
        } else {
            int channel = (idx / channel_stride) % channel_size;
            dst[idx]    = src[idx] * norm[outer_idx] * scale[channel];
            // printf("channel: %d, scale: %.2f\n", channel, scale[channel]);
        }
    }
}

//! normalize without scale
__kernel void Normalize(
    int n,
    int inner_size,
    __global const _FLOAT* __restrict norm,
    __global const _FLOAT* __restrict src,
    __global _FLOAT* dst) {
    int idx         = get_global_id(0);
    int global_size = get_global_size(0);

    for (; idx < n; idx += global_size) {
        int outer_idx = idx / inner_size;
        dst[idx]      = src[idx] * norm[outer_idx];
    }
}

//! normalize with scale
__kernel void NormalizeWithScaleComputeNorm(
    int n,
    int inner_size,
    int channel_stride,
    int channel_size,
    __global const _FLOAT* __restrict norm,
    __global const _FLOAT* __restrict scale,
    __global const _FLOAT* __restrict src,
    __global _FLOAT* __restrict dst,
    int channel_shared) {
    __local _FLOAT sdata[1];
    int tid        = get_local_id(0);
    int i          = get_global_id(0);
    int in_n_index = i / inner_size;
    int idx_limit  = (in_n_index + 1) * inner_size;

    if (tid == 0) {
        sdata[0] = 1 / (sqrt(norm[in_n_index] / inner_size) + 1e-6f);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (channel_shared) {
        if (i < idx_limit) {
            dst[i] = src[i] * sdata[0] * scale[0];
        }
    } else {
        if (i < idx_limit) {
            int channel = (i / channel_stride) % channel_size;
            dst[i]      = src[i] * sdata[0] * scale[channel];
        }
    }
}

//! normalize without scale
__kernel void NormalizeComputeNorm(
    int n,
    int inner_size,
    __global const _FLOAT* __restrict norm,
    __global const _FLOAT* __restrict src,
    __global _FLOAT* __restrict dst) {
    __local _FLOAT sdata[1];
    int tid        = get_local_id(0);
    int i          = get_global_id(0);
    int in_n_index = i / inner_size;
    int idx_limit  = (in_n_index + 1) * inner_size;

    if (tid == 0) {
        sdata[0] = 1 / (sqrt(norm[in_n_index] / inner_size) + 1e-6f);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < idx_limit) {
        dst[i] = src[i] * sdata[0];
    }
}
