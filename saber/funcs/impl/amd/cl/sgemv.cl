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

#if METHOD == 1
#define ITER ((WIDTH + 63) / 64)
#define ALIGNED_WIDTH (ITER * 64)

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b,
#if BIAS == 1
    __global const float* bias,
#endif
    __global float* c, uint WIDTH, uint OUTPUT) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float result[66];

    __global const float* pB = (__global const float*)(b + grid_x * WIDTH + lid_x); // correct
    __global float* pC;


    uint offset = (grid_x >> 2 << 6) % ALIGNED_WIDTH;

    float sum = 0.0f;

    if (grid_x < OUTPUT) {
        sum = 0.0f;

        pB = (__global const float*)(b + grid_x * WIDTH + lid_x);
        pC = (__global float*)(c + grid_x);

        for (uint i = 0; i < ITER; i++, offset = (offset + 64) % ALIGNED_WIDTH) {
            if (offset + lid_x < WIDTH) {
                result[lid_x + (lid_x >> 5)] = a[offset + lid_x];

                sum += result[lid_x + (lid_x >> 5)] * pB[offset]; // correct
            }
        }

        result[lid_x] = sum;

        if (lid_x < 32) {
            result[lid_x] += result[lid_x + 32];
        }

        if (lid_x < 16) {
            result[lid_x] += result[lid_x + 16];
        }

        if (lid_x < 8) {
            result[lid_x] += result[lid_x + 8];
        }

        if (lid_x < 4) {
            result[lid_x] += result[lid_x + 4];
        }

        if (lid_x < 2) {
            result[lid_x] += result[lid_x + 2];
        }

        if (lid_x < 1) {
            result[lid_x] += result[lid_x + 1];
        }

        if (lid_x == 0) {
#if BIAS == 1
            pC[0] = bias[grid_x] + result[0];
#else
            pC[0] = result[0];
#endif
        }
    }
}
#elif METHOD == 2
#define W_PER_CU    32
#define M       (4096 / WIDTH)
#define WORKLOAD_STRIDE (64 * W_PER_CU * M)
#define L       ((OUTPUT + WORKLOAD_STRIDE - 1) / WORKLOAD_STRIDE)
#define WORKLOAD    (W_PER_CU * L)
#define WG_WORKLOAD (WORKLOAD * M)
#define ITER        (WIDTH >> 7)

void reduce(__local float* buffer, uint tid) {
    buffer[tid + (tid >> 5)] += buffer[(tid + 64) + ((tid + 64) >> 5)];

    if (tid < 32) {
        buffer[tid] += buffer[(tid + 32) + ((tid + 32) >> 5)];
    }

    if (tid < 16) {
        buffer[tid] += buffer[tid + 16];
    }

    if (tid < 8) {
        buffer[tid] += buffer[tid + 8];
    }

    if (tid < 4) {
        buffer[tid] += buffer[tid + 4];
    }

    if (tid < 2) {
        buffer[tid] += buffer[tid + 2];
    }

    if (tid < 1) {
        buffer[tid] += buffer[tid + 1];
    }
}

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __constant float* a, __global const float* b,
#if BIAS == 1
    __global const float* bias,
#endif
    __global float* c, uint WIDTH, uint OUTPUT) {
    __local float result[2][132];

    uint lid_x  = get_local_id(0);
    uint grid_x = get_group_id(0);

    uint s_idx = (grid_x % W_PER_CU);
    uint b_idx = (s_idx + (grid_x / W_PER_CU) * WG_WORKLOAD);

    uint out_idx = b_idx;

    __global const float* pB = (__global const float*)(b + b_idx * WIDTH);

    for (uint n = 0; n < M; n++) {
        for (uint l = 0; l < L; l++) {
            uint idx = ((n & 1) == 1) ? l : (L - 1 - l);

            if (out_idx + n * WORKLOAD + idx * W_PER_CU < OUTPUT) {
                float sum[2] = {0.0f};

                for (uint i = 0; i < ITER; i++) {
                    sum[0] += a[lid_x + i * 128] * pB[lid_x + (n * WORKLOAD + idx * W_PER_CU) * WIDTH + i * 128];
                    sum[1] += a[lid_x + 64 + i * 128] *
                              pB[lid_x + 64 + (n * WORKLOAD + idx * W_PER_CU) * WIDTH + i * 128];
                }

                result[l & 1][lid_x + (lid_x >> 5)] = sum[0];
                result[l & 1][lid_x + (lid_x >> 5) + 66] = sum[1];

                reduce(result[l & 1], lid_x);

                if (lid_x == 0) {
#if BIAS == 1
                    c[out_idx + n * WORKLOAD + idx * W_PER_CU] = result[l & 1][0] +
                            bias[out_idx + n * WORKLOAD + idx * W_PER_CU];
#else
                    c[out_idx + n * WORKLOAD + idx * W_PER_CU] = result[l & 1][0];
#endif
                }
            }
        }
    }
}
#endif
