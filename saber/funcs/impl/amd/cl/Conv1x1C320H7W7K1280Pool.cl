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
#define DIR

#ifndef N
#define N   (1)
#endif
#ifndef C
#define C   (320)
#endif
#ifndef H
#define H   (7)
#endif
#ifndef W
#define W   (7)
#endif
#ifndef K
#define K   (1280)
#endif

#define PER_ITER_STRIDE (80)
#define STRIDE_WEI  C
#define STRIDE_IN   (H * W)
#define ITER    (STRIDE_WEI / PER_ITER_STRIDE)

#define LDS_WEI_STRIDE  (132)
#define LDS_WEI_COL (128)
#define LDS_WEI_COL_REAL    (80)
#define LDS_WEI_ROW (20)
#define LDS_IN_STRIDE   (66)
#define LDS_IN_COL_REAL (STRIDE_IN)
#define LDS_IN_COL  (64)
#define LDS_IN_ROW  (80)
#define LDS_WEI_READ_ITER   (3)
#define LDS_IN_READ_ITER    (5)
#define PER_WI_TILE_ROW (1)
#define PER_WI_TILE_COL (1)

#define COUNTER_STRIDE  64
#define IN_COL_WG   1
#define GROUP_ITER  (grid_x / COUNTER_STRIDE)
#define WEI_ROW_START   ((grid_x % COUNTER_STRIDE) / IN_COL_WG * LDS_WEI_ROW)
#define WEI_COL_START   (grid_x / COUNTER_STRIDE)
#define WEI_WI_COL_START    (lid_x % LDS_WEI_COL)
#define IN_COL_START    ((grid_x % IN_COL_WG) * LDS_IN_ROW)
#define IN_WI_COL_START (lid_x % COUNTER_STRIDE)
#define OUT_WI_ROW_START    (lid_x / STRIDE_IN)
#define OUT_WI_COL_START    (lid_x & 15)
#define OUT_PER_WG  (STRIDE_IN * LDS_WEI_ROW)
#define OUT_PER_WG_REAL (LDS_WEI_ROW)
#define OUT_WI_INDEX    (lid_x)
#define LDS_WEI_WI  (lid_x % LDS_WEI_COL)
#define LDS_WEI_ROW_START   ((lid_x / LDS_WEI_COL))
#define LDS_IN_ROW_START    ((lid_x >> 6) * LDS_IN_READ_ITER)
#define COMPUTE_WEI_WI_INDEX    (lid_x / STRIDE_IN)
#define COMPUTE_IN_WI_INDEX (lid_x % STRIDE_IN)
#define OFFSET  ((grid_x % COUNTER_STRIDE) * PER_ITER_STRIDE % STRIDE_WEI)

#define IN_BATCH_STRIDE     (C * H * W)
#define OUT_BATCH_STRIDE    (K)

#define ATOMIC

#ifdef DIR
void reduce(__local float* buffer, uint tid, uint start, uint upper) {
    tid += start;

    if (tid < upper) {
        if ((tid & 63) < 32) {
            buffer[tid + (tid >> 5)] += buffer[tid + 32 + ((tid + 32) >> 5)];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < upper) {
        if ((tid & 63) < 16) {
            buffer[tid + (tid >> 5)] += buffer[tid + 16 + ((tid + 16) >> 5)];
            buffer[tid + (tid >> 5)] += buffer[tid + 8 + ((tid + 8) >> 5)];
            buffer[tid + (tid >> 5)] += buffer[tid + 4 + ((tid + 4) >> 5)];
            buffer[tid + (tid >> 5)] += buffer[tid + 2 + ((tid + 2) >> 5)];
            buffer[tid + (tid >> 5)] += buffer[tid + 1 + ((tid + 1) >> 5)];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
#else
void reduce(__local float* buffer, int tid) {
    if ((tid & 15) < 8) {
        buffer[tid + (tid >> 5)] += buffer[tid + 8 + ((tid + 8) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 4 + ((tid + 4) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 2 + ((tid + 2) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 1 + ((tid + 1) >> 5)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
#if 1

    if (tid < 256 && (tid & 15) == 0) {
        buffer[tid + (tid >> 5)] += buffer[tid + 256 + ((tid + 256) >> 5)] +
                                    buffer[tid + 512 + ((tid + 512) >> 5)] + buffer[tid + 768 + ((tid + 768) >> 5)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
#endif
}
#endif

__attribute__((reqd_work_group_size(1024, 1, 1)))
__kernel void conv1x1_act_pool(
    __global const float* wei,
    __global const float* in,
#ifdef BIAS
    __constant float* bias,
#endif
    __global float* out,
    float slope) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float shared_wei[1024 * 4];
    __local float shared_in[2560 * 4];
    __local float shared_result[512 * 4];
    __local float* pShared_wei = (__local float*)shared_wei;
    __local float* pShared_in = (__local float*)shared_in;

    __global const float* pWei = (__global const float*)(wei + (WEI_ROW_START * STRIDE_WEI +
                                 WEI_WI_COL_START));//
    __global const float* pIn = (__global const float*)(in + (IN_WI_COL_START));//
    __global float* pOut = (__global float*)(out + (WEI_ROW_START + lid_x));//
#ifdef BIAS
    __constant float* pBias = (__constant float*)(bias + (WEI_ROW_START + OUT_WI_ROW_START));//
#endif


    float sum[N] = { 0.0f };

    uint offset = OFFSET;

    for (uint i = 0; i < 2; i++) {
        shared_result[lid_x + i * 1024] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef DIR
#if 0

    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % STRIDE_WEI) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * 17 + (LDS_WEI_ROW_START + i)] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
            shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START + (IN_WI_COL_START >> 5)] =
                (IN_WI_COL_START < STRIDE_IN ? pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN] :
                 0.0f); //pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
            sum += pShared_wei[j * 17 + OUT_WI_ROW_START] *
                   pShared_in[j * LDS_IN_STRIDE + IN_WI_COL_START + (IN_WI_COL_START >> 5)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

#elif 0

    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % STRIDE_WEI) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START + (WEI_WI_COL_START >> 5) + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
            shared_in[(LDS_IN_ROW_START + i) + ((LDS_IN_ROW_START + i) >> 5) + IN_WI_COL_START * LDS_WEI_STRIDE]
                = (IN_WI_COL_START < STRIDE_IN ? pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN] :
                   0.0f); //pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
            sum += pShared_wei[j + (j >> 5) + (OUT_WI_ROW_START * LDS_WEI_STRIDE)] *
                   pShared_in[j + (j >> 5) + IN_WI_COL_START * LDS_WEI_STRIDE];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

#else

    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % STRIDE_WEI) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            if (LDS_WEI_ROW_START + i * 8 < LDS_WEI_ROW) {
                shared_wei[WEI_WI_COL_START + (WEI_WI_COL_START >> 5) + (LDS_WEI_ROW_START + i * 8) * LDS_WEI_STRIDE]
                    = (LDS_WEI_WI < LDS_WEI_COL_REAL ? pWei[(LDS_WEI_ROW_START + i * 8) * STRIDE_WEI + offset] : 0.0f);
            }
        }

        for (uchar n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START + (IN_WI_COL_START >> 5)] =
                    (IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] :
                     0.0f); //pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                sum[n] += (lid_x < OUT_PER_WG ?
                           pShared_wei[j + (j >> 5) + (COMPUTE_WEI_WI_INDEX * LDS_WEI_STRIDE)] *
                           pShared_in[j * LDS_IN_STRIDE + COMPUTE_IN_WI_INDEX + (COMPUTE_IN_WI_INDEX >> 5)] : 0.0f);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#endif
#elif 1

    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % STRIDE_WEI) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START + (WEI_WI_COL_START >> 5) + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
            shared_in[(LDS_IN_ROW_START + i) + ((LDS_IN_ROW_START + i) >> 5) + IN_WI_COL_START * LDS_IN_STRIDE]
                = (IN_WI_COL_START < STRIDE_IN ? pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN] :
                   0.0f); //pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uchar j = 0; j < 32; j++) {
            for (uchar l = 0; l < 4; l++) {
                sum += pShared_wei[(lid_x >> 8 << 5) + j + (((lid_x >> 8 << 5) + j) >> 5) + ((((lid_x & 255) >> 4)) * LDS_WEI_STRIDE)] *
                       pShared_in[(lid_x >> 8 << 5) + j + (((lid_x >> 8 << 5) + j) >> 5) + (((lid_x & 15) << 2) + l) * LDS_IN_STRIDE];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

#else

    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % STRIDE_WEI) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * 17 + (LDS_WEI_ROW_START + i)] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if 0
            shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START + (IN_WI_COL_START >> 5)] =
                pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN];
#else
            shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START + (IN_WI_COL_START >> 5)] =
                (IN_WI_COL_START < STRIDE_IN ? pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN] :
                 0.0f); //pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN];
#endif
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uchar j = 0; j < 32; j++) {
            for (uchar l = 0; l < 4; l++) {
                sum += pShared_wei[((lid_x >> 8 << 5) + j) * 17 + ((lid_x & 255) >> 4)] *
                       pShared_in[((lid_x >> 8 << 5) + j) * LDS_IN_STRIDE + ((lid_x & 15) << 2) + l + ((((lid_x & 15) << 2) + l) >> 5)];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

#endif

#ifdef DIR

    for (uint n = 0; n < N; n++) {
#ifdef BIAS
        sum[n] += lid_x < OUT_PER_WG ? pBias[0] : 0.0f;
#endif
        sum[n] *= (sum[n] > 0.0f ? 1.0f : slope);
        shared_result[(COMPUTE_WEI_WI_INDEX * 64 + COMPUTE_IN_WI_INDEX) + ((COMPUTE_WEI_WI_INDEX * 64 + COMPUTE_IN_WI_INDEX) >> 5)]
            = sum[n];
        barrier(CLK_LOCAL_MEM_FENCE);

        reduce(shared_result, lid_x, 0, K);
        reduce(shared_result, lid_x, 1024, K);

        if (lid_x < OUT_PER_WG_REAL) {
            pOut[n * OUT_BATCH_STRIDE] = shared_result[(lid_x << 6) + (lid_x << 6 >> 5)] / 49.0f;
        }
    }

#else
#ifdef BIAS
    sum += (COMPUTE_WEI_WI_INDEX * 64 + COMPUTE_IN_WI_INDEX) < OUT_PER_WG ? pBias[0] : 0.0f;
#endif
    sum *= (sum > 0.0f ? 1.0f : slope);
    shared_result[lid_x + (lid_x >> 5)] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    reduce(shared_result, lid_x);

    if (lid_x < 256 && (lid_x & 15) == 0) {
        pOut[0] = shared_result[(lid_x) + (lid_x >> 5)] / 49.0f;
    }

#endif
}

