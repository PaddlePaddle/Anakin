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
#ifndef N
#define N (1)
#endif
#ifndef C
#define C (384)
#endif
#ifndef H
#define H (14)
#endif
#ifndef W
#define W (14)
#endif
#ifndef K
#define K (96)
#endif

#define GLOBAL_SPLITU (4)
#define GSU_MINUS_ONE (GLOBAL_SPLITU - 1)
#define BARRIER_MARK (0xffffffff)
#define STRIDE_WEI (C)
#define STRIDE_IN (H * W)
#define QSTRIDE (STRIDE_WEI >> 2)
#define ITER (QSTRIDE >> 4)
#define PER_ITER_STRIDE (16)
#define LDS_WEI_STRIDE (17)
#define LDS_WEI_ROW (32)
#define LDS_IN_STRIDE (17)
#define LDS_IN_ROW (16)
#define LDS_WEI_READ_ITER (2)
#define LDS_IN_READ_ITER (1)
#define PER_WI_TILE_ROW (1)
#define PER_WI_TILE_COL (2)

#define COUNTER_STRIDE 39
#define GROUP_ITER (grid_x / COUNTER_STRIDE)
#define WEI_ROW_START ((grid_x % COUNTER_STRIDE) / 13 << 5)
#define WEI_COL_START (grid_x / COUNTER_STRIDE)
#define WEI_WI_COL_START (lid_x & 15)
#define IN_COL_START ((grid_x % 13) << 4)
#define IN_WI_COL_START (lid_x & 15)
#define OUT_WI_ROW_START (lid_x >> 3)
#define OUT_WI_COL_START ((lid_x & 7) << 1)
#define LDS_WEI_ROW_START (lid_x >> 4 << 1)
#define LDS_IN_ROW_START (lid_x >> 4)
#define OFFSET ((grid_x % COUNTER_STRIDE) / 13 << 4)

#define COUNTER_INDEX (grid_x % COUNTER_STRIDE)

//#define ATOMIC
#define NOFILLED

// global: (256 * 39 * 4, 1, 1)

__attribute__((reqd_work_group_size(256, 1, 1))) __kernel void conv1x1_act(
        __global const float* wei,
        __global const float* in,
#ifdef BIAS
        __constant float* bias,
#endif
        __global float* out,
        float slope) {
    uint lid_x  = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float shared_wei[LDS_WEI_ROW * LDS_WEI_STRIDE];
    __local float shared_in[LDS_IN_ROW * LDS_IN_STRIDE];
    __local float* pShared_wei = (__local float*)shared_wei;
    __local float* pShared_in  = (__local float*)shared_in;

    __global const float* pWei =
            (__global const float*)(wei + WEI_ROW_START * STRIDE_WEI + WEI_COL_START * QSTRIDE + WEI_WI_COL_START); //
    __global const float* pIn =
            (__global const float*)(in + (IN_COL_START + IN_WI_COL_START) + WEI_COL_START * QSTRIDE * STRIDE_IN); //
    __global float* pOut =
            (__global float*)(out + (IN_COL_START + OUT_WI_COL_START) + (WEI_ROW_START + OUT_WI_ROW_START) * STRIDE_IN); //
    __global uint* pBarrier = (__global uint*)(out + W * H * K);                  //
    __global uint* pCounter = (__global uint*)(out + W * H * K + COUNTER_STRIDE); //
#ifdef BIAS
    __constant float* pBias = (__constant float*)(bias + WEI_ROW_START + OUT_WI_ROW_START); //
#endif

    if (grid_x < COUNTER_STRIDE) {
        if (lid_x == 0) {
            *(pBarrier + COUNTER_INDEX) = 0;
            *(pCounter + COUNTER_INDEX) = 0;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    float sum[PER_WI_TILE_COL] = {0.0f};

    uint offset = OFFSET;

#if 1
    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % QSTRIDE) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                    pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }
        for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if 0
            shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] = pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN];
#else
            shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN
                             ? pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN]
                             : 0.0f);
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
            for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                sum[l] += pShared_wei[j + (OUT_WI_ROW_START * LDS_WEI_STRIDE)]
                          * pShared_in[j + (OUT_WI_COL_START + l) * LDS_WEI_STRIDE];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#else
    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % QSTRIDE) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * 33 + (LDS_WEI_ROW_START + i)] =
                    pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }
        for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if 0
            shared_in[(LDS_IN_ROW_START + i) * 33 + IN_WI_COL_START] = pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN];
#else
            shared_in[(LDS_IN_ROW_START + i) * 33 + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN
                             ? pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN]
                             : 0.0f);
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
            for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                sum[l] += pShared_wei[j * 33 + OUT_WI_ROW_START]
                          * pShared_in[j * 33 + OUT_WI_COL_START + l];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif

    float previous_value;
    uint prevVal;
    uint newVal;
    if (GROUP_ITER == 0) {
        if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
#ifdef NOFILLED
#ifdef BIAS
            for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                *(pOut + i) = sum[i] + pBias[0];
            }
#else
            for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                *(pOut + i) = sum[i];
            }
#endif
#else
#ifdef BIAS
            for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                do {
                    previous_value = *(pOut + i);
                    prevVal        = as_uint(previous_value);
                    newVal         = as_uint(sum[i] + pBias[0] + previous_value);
                } while (atomic_cmpxchg((__global uint*)(pOut + i), prevVal, newVal) != prevVal);
            }
#else
            for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                do {
                    previous_value = *(pOut + i);
                    prevVal        = as_uint(previous_value);
                    newVal         = as_uint(sum[i] + previous_value);
                } while (atomic_cmpxchg((__global uint*)(pOut + i), prevVal, newVal) != prevVal);
            }
#endif
#endif
        }

#ifdef NOFILLED
#if 0
        if (lid_x == 0)
        {
            *(pCounter + COUNTER_INDEX) = 0;
        }
#endif
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (lid_x == 0) {
            *(pBarrier + COUNTER_INDEX) = BARRIER_MARK;
        }
#endif
    } else {
#ifdef NOFILLED
        if (lid_x == 0) {
            do {
                prevVal = *(pBarrier + COUNTER_INDEX);
                newVal  = BARRIER_MARK;
            } while (
                    atomic_cmpxchg((__global uint*)(pBarrier + COUNTER_INDEX), BARRIER_MARK, newVal)
                    != BARRIER_MARK);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
#endif

        if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                do {
                    previous_value = *(pOut + i);
                    prevVal        = as_uint(previous_value);
                    newVal         = as_uint(sum[i] + previous_value);
                } while (atomic_cmpxchg((__global uint*)(pOut + i), prevVal, newVal) != prevVal);
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }

        if (GROUP_ITER == GSU_MINUS_ONE) {
            mem_fence(CLK_GLOBAL_MEM_FENCE);
            if (lid_x == 0) {
                do {
                    prevVal = *(pCounter + COUNTER_INDEX);
                    newVal  = GSU_MINUS_ONE;
                } while (atomic_cmpxchg(
                                 (__global uint*)(pCounter + COUNTER_INDEX), GSU_MINUS_ONE, newVal)
                         != GSU_MINUS_ONE);
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

            if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
                for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                    pOut[i] *= (pOut[i] > 0.0f ? 1.0f : slope);
                }
            }
        }
    }
}
