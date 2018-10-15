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
union AtomicFloat {
    unsigned int u32;
    float f32;
};

void cl_atomic_add_float(volatile __global float* addr, float val) {
    union AtomicFloat current, expected, next;

    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32     = current.f32 + val;

        current.u32 = atomic_cmpxchg((volatile __global unsigned int*)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

__kernel void sum_square(
        const global float* in_data,
        const int height,
        const int width,
        global float* sum,
        global float* square_sum) {
    int blockSize = get_local_size(0);
    int local_idx = get_local_id(0);
    __local float share_data[256];
    __local float share_square[256];
    int offset = get_local_id(1) * width;

    const global float* tmp_in_data = in_data + offset;
    int index                       = get_global_id(0);
    int stride                      = get_global_size(0);
    if (index < width) {
        float result        = tmp_in_data[index];
        float square_result = tmp_in_data[index] * tmp_in_data[index];
        for (int idx = index + stride; idx < width; idx += stride) {
            float data = tmp_in_data[idx];
            result += data;
            square_result += data * data;
        }
        share_data[local_idx]   = result;
        share_square[local_idx] = square_result;
    } else {
        share_data[local_idx]   = 0;
        share_square[local_idx] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (blockSize >= 256) {
        if (local_idx < 128) {
            share_data[local_idx] += share_data[local_idx + 128];
            share_square[local_idx] += share_square[local_idx + 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (blockSize >= 128) {
        if (local_idx < 64) {
            share_data[local_idx] += share_data[local_idx + 64];
            share_square[local_idx] += share_square[local_idx + 64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_idx < 32) {
        volatile local float* vsum    = share_data;
        volatile local float* vsquare = share_square;
        if (blockSize >= 64) {
            vsum[local_idx] += vsum[local_idx + 32];
            vsquare[local_idx] += vsquare[local_idx + 32];
        }
        if (blockSize >= 32) {
            vsum[local_idx] += vsum[local_idx + 16];
            vsquare[local_idx] += vsquare[local_idx + 16];
        }
        if (blockSize >= 16) {
            vsum[local_idx] += vsum[local_idx + 8];
            vsquare[local_idx] += vsquare[local_idx + 8];
        }
        if (blockSize >= 8) {
            vsum[local_idx] += vsum[local_idx + 4];
            vsquare[local_idx] += vsquare[local_idx + 4];
        }
        if (blockSize >= 4) {
            vsum[local_idx] += vsum[local_idx + 2];
            vsquare[local_idx] += vsquare[local_idx + 2];
        }
        if (blockSize >= 2) {
            vsum[local_idx] += vsum[local_idx + 1];
            vsquare[local_idx] += vsquare[local_idx + 1];
        }
    }
    if (local_idx == 0) {
        cl_atomic_add_float(&sum[get_group_id(1)], share_data[0]);
        cl_atomic_add_float(&square_sum[get_group_id(1)], share_square[0]);
    }
}

__kernel void normalize_square(
        const global float* in_data,
        const int height,
        const int width,
        const float scale,
        const global float* sum,
        const global float* square_sum,
        const float eps,
        global float* out_data) {
    int idx       = get_global_id(0);
    int idy       = get_group_id(1);
    int local_idx = get_local_id(0);
    __local float share_data[2];
    if (local_idx == 0) {
        float mean    = sum[idy] * scale;
        share_data[0] = mean;
        share_data[1] = 1.0f / (sqrt(square_sum[idy] * scale - mean * mean) + eps);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx < width && idy < height) {
        int index       = idx + idy * width;
        out_data[index] = (in_data[index] - share_data[0]) * share_data[1];
    }
}

__kernel void normalize(
        const global float* in_data,
        const int height,
        const int width,
        const float scale,
        const global float* sum,
        global float* out_data) {
    int idx       = get_global_id(0);
    int idy       = get_local_size(1);
    int local_idx = get_local_id(0);
    __local float share_data[1];
    if (local_idx == 0) {
        share_data[0] = sum[idy] * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (idx < width && idy < height) {
        int index       = idx + idy * width;
        out_data[index] = in_data[index] - share_data[0];
    }
}

__kernel void
sum(const global float* in_data, const int height, const int width, global float* sum) {
    int blockSize = get_local_size(0);
    __local float share_data[256];
    int offset = get_local_id(1) * width;

    const global float* tmp_in_data = in_data + offset;
    int index                       = get_global_id(0);
    int stride                      = get_global_size(0);
    int local_idx                   = get_local_id(0);
    if (index < width) {
        float result = tmp_in_data[index];
        for (int tid = index + stride; tid < width; tid += stride) {
            float data = tmp_in_data[tid];
            result += data;
        }
        share_data[local_idx] = result;
    } else {
        share_data[local_idx] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (blockSize >= 512) {
        if (local_idx < 256) {
            share_data[local_idx] += share_data[local_idx + 256];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (blockSize >= 256) {
        if (local_idx < 128) {
            share_data[local_idx] += share_data[local_idx + 128];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (blockSize >= 128) {
        if (local_idx < 64) {
            share_data[local_idx] += share_data[local_idx + 64];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_idx < 32) {
        if (blockSize >= 64) {
            share_data[local_idx] += share_data[local_idx + 32];
        }
        if (blockSize >= 32) {
            share_data[local_idx] += share_data[local_idx + 16];
        }
        if (blockSize >= 16) {
            share_data[local_idx] += share_data[local_idx + 8];
        }
        if (blockSize >= 8) {
            share_data[local_idx] += share_data[local_idx + 4];
        }
        if (blockSize >= 4) {
            share_data[local_idx] += share_data[local_idx + 2];
        }
        if (blockSize >= 2) {
            share_data[local_idx] += share_data[local_idx + 1];
        }
    }
    if (local_idx == 0) {
        cl_atomic_add_float(&sum[get_group_id(1)], share_data[0]);
    }
}
