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

#if GLOBAL_SPLITU == 1
#define ATOMIC  0
#else
#define ATOMIC  2
#endif

///////////////////////////////////////////
#ifndef LOCAL_X
#define LOCAL_X     (256)
#endif

#ifndef METHOD
#define METHOD  1
#endif

#ifndef BRANCH
#if N == 1
#define BRANCH  1
#elif N == 2
#define BRANCH  3
#endif
#endif

#define GSU_MINUS_ONE   (GLOBAL_SPLITU - 1)
#define BARRIER_MARK    (0xffffffff)
#define STRIDE_WEI  (C)
#if STRIDE == 2
#define OW  (W / 2)
#define OH  (H / 2)
#define STRIDE_IN_REAL  (H * W)
#else
#define OW  (W)
#define OH  (H)
#endif
#define STRIDE_IN   (OH * OW)


#define WG_PER_IN   ((STRIDE_IN + TILE_COL - 1) / TILE_COL)
#define WG_PER_WEI  ((K + TILE_ROW - 1) / TILE_ROW)

#define QSTRIDE (STRIDE_WEI / GLOBAL_SPLITU)
#define ITER    (QSTRIDE / PER_ITER_STRIDE)

#if BRANCH == 1 || BRANCH == 2
#define LDS_WEI_COL (PER_ITER_STRIDE)
#define LDS_WEI_STRIDE  (LDS_WEI_COL + (LDS_WEI_COL + 31) / 32)
#define LDS_WEI_ROW (TILE_ROW)
#define WEI_READ_LINE   LDS_WEI_COL
#elif BRANCH == 3 || BRANCH == 4
#define LDS_WEI_COL (TILE_ROW)
#define LDS_WEI_STRIDE  (LDS_WEI_COL + (LDS_WEI_COL + 31) / 32)
#define LDS_WEI_ROW (PER_ITER_STRIDE)
#define WEI_READ_LINE   LDS_WEI_ROW
#endif
#if BRANCH == 1 || BRANCH == 3
#define LDS_IN_COL  (PER_ITER_STRIDE)
#define LDS_IN_STRIDE   (LDS_IN_COL + (LDS_IN_COL + 31) / 32)
#define LDS_IN_ROW  (TILE_COL)
#define IN_READ_LINE    LDS_IN_ROW
#elif BRANCH == 2 || BRANCH == 4
#define LDS_IN_COL  (TILE_COL)
#define LDS_IN_STRIDE   (LDS_IN_COL + (LDS_IN_COL + 31) / 32)
#define LDS_IN_ROW  (PER_ITER_STRIDE)
#define IN_READ_LINE    LDS_IN_COL
#endif

#define LDS_WEI_READ_ITER   (LDS_WEI_COL * LDS_WEI_ROW / LOCAL_X)
#define LDS_IN_READ_ITER    (LDS_IN_COL * LDS_IN_ROW / LOCAL_X)
#define LDS_WEI_ROW_START   ((lid_x / WEI_READ_LINE) * LDS_WEI_READ_ITER)
#define LDS_IN_ROW_START    ((lid_x / IN_READ_LINE) * LDS_IN_READ_ITER)

#define IN_BATCH_STRIDE     (C * H * W)
#define OUT_BATCH_STRIDE    (K * OH * OW)

#define COUNTER_STRIDE  (WG_PER_IN * WG_PER_WEI)
#define GROUP_ITER  (grid_x / COUNTER_STRIDE)
#define WEI_ROW_START   ((grid_x % COUNTER_STRIDE) / WG_PER_IN * TILE_ROW)
#define WEI_COL_START   (grid_x / COUNTER_STRIDE)
#define WEI_WI_COL_START    (lid_x % PER_ITER_STRIDE)
#if STRIDE == 2
#define IN_COL_START_REAL   (((grid_x % WG_PER_IN) * TILE_COL) + (lid_x % TILE_COL))
#define IN_WI_COL_START_REAL    (IN_COL_START_REAL / OW * OW * 4 + IN_COL_START_REAL % OW * 2)
#define IN_COL_START    ((grid_x % WG_PER_IN) * TILE_COL)
#define IN_WI_COL_START (lid_x % TILE_COL)
#else
#define IN_COL_START    ((grid_x % WG_PER_IN) * TILE_COL)
#define IN_WI_COL_START (lid_x % TILE_COL)
#endif
#define OUT_WI_PER_ROW      (TILE_COL / PER_WI_TILE_COL)
#define OUT_WI_ROW_START    (lid_x / OUT_WI_PER_ROW * PER_WI_TILE_ROW)
#define OUT_WI_COL_START    ((lid_x % OUT_WI_PER_ROW) * PER_WI_TILE_COL)

#if 1
#define OFFSET  ((grid_x % COUNTER_STRIDE * STRIDE_WEI / 4096 * PER_ITER_STRIDE) % QSTRIDE)
#else
#define OFFSET  (((grid_x % COUNTER_STRIDE) / WG_PER_IN * PER_ITER_STRIDE) % QSTRIDE)
#endif

#define COUNTER_INDEX   (grid_x % COUNTER_STRIDE)

#define NOFILLED

//global: (256 * WG_PER_IN * WG_PER_WEI * GLOBAL_SPLITU, 1, 1)

#if METHOD == 1
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void conv1x1_act(
    __global const float* wei,
    __global const float* in,
#if BIAS == 1
    __constant float* bias,
#endif
    __global float* out,
    float slope) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float shared_wei[LDS_WEI_ROW * LDS_WEI_STRIDE];
    __local float shared_in[LDS_IN_ROW * LDS_IN_STRIDE];
    __local float* pShared_wei = (__local float*)shared_wei;
    __local float* pShared_in = (__local float*)shared_in;

    __global const float* pWei = (__global const float*)(wei + WEI_ROW_START * STRIDE_WEI +
                                 WEI_COL_START * QSTRIDE + WEI_WI_COL_START);//
#if STRIDE == 2
    __global const float* pIn = (__global const float*)(in + (IN_WI_COL_START_REAL) + WEI_COL_START *
                                QSTRIDE * STRIDE_IN_REAL);//
#else
    __global const float* pIn = (__global const float*)(in + (IN_COL_START + IN_WI_COL_START) +
                                WEI_COL_START * QSTRIDE * STRIDE_IN);//
#endif
    __global float* pOut = (__global float*)(out + (IN_COL_START + OUT_WI_COL_START) +
                           (WEI_ROW_START + OUT_WI_ROW_START) * STRIDE_IN);//

#if ATOMIC == 1
    __global uint* pBarrier = (__global uint*)(out + OUT_BATCH_STRIDE * N);//
    __global uint* pCounter = (__global uint*)(out + OUT_BATCH_STRIDE * N + COUNTER_STRIDE);//
#elif ATOMIC == 2
    volatile __global uint* pCounter = (volatile __global uint*)(out + OUT_BATCH_STRIDE * N);//
#endif

#if BIAS == 1
    __constant float* pBias = (__constant float*)(bias + WEI_ROW_START + OUT_WI_ROW_START);//
#endif

    float previous_value;
    uint prevVal;
    uint newVal;

#if ATOMIC == 1

    if (grid_x < COUNTER_STRIDE) {
        if (lid_x == 0) {
            *(pBarrier + COUNTER_INDEX) = 0;
            *(pCounter + COUNTER_INDEX) = 0;
        }
    }

    //barrier(CLK_GLOBAL_MEM_FENCE);
#elif ATOMIC == 2

    if (grid_x < COUNTER_STRIDE) {
        if (lid_x == 0) {
            *(pCounter + COUNTER_INDEX) = 0;
        }
    }

    //barrier(CLK_GLOBAL_MEM_FENCE);
#endif

    float sum[N][PER_WI_TILE_ROW][PER_WI_TILE_COL] = { { { 0.0f } } };

    uint offset = OFFSET;

#if BRANCH == 1

    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % QSTRIDE) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
            prefetch(pWei + (LDS_WEI_ROW_START + i) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % QSTRIDE, 32);
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % QSTRIDE) * STRIDE_IN_REAL -
                         IN_WI_COL_START_REAL + IN_COL_START_REAL / OW * OW * 4 + n * IN_BATCH_STRIDE, 64);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % QSTRIDE) * STRIDE_IN -
                         IN_WI_COL_START + n * IN_BATCH_STRIDE, 32);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] += pShared_wei[j + (OUT_WI_ROW_START + m) * LDS_WEI_STRIDE] *
                                        pShared_in[j + (OUT_WI_COL_START + l) * LDS_IN_STRIDE];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 2

    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % QSTRIDE) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] += pShared_wei[j + (OUT_WI_ROW_START + m) * LDS_WEI_STRIDE] *
                                        pShared_in[j * LDS_IN_STRIDE + (OUT_WI_COL_START + l)];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 3

    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % QSTRIDE) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * LDS_WEI_STRIDE + (LDS_WEI_ROW_START + i) + (LDS_WEI_ROW_START + i) / 32]
                = pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
            prefetch(pWei + (LDS_WEI_ROW_START + i) * STRIDE_WEI - WEI_WI_COL_START +
                     (offset + PER_ITER_STRIDE) % QSTRIDE, 32);
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % QSTRIDE) * STRIDE_IN_REAL -
                         IN_WI_COL_START_REAL + IN_COL_START_REAL / OW * OW * 4 + n * IN_BATCH_STRIDE, 64);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_IN_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
                prefetch(pIn + (LDS_IN_ROW_START + i + (offset + PER_ITER_STRIDE) % QSTRIDE) * STRIDE_IN -
                         IN_WI_COL_START + n * IN_BATCH_STRIDE, 32);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar l = 0; l < PER_WI_TILE_COL; l++)
                    for (uchar m = 0; m < PER_WI_TILE_ROW; m++) {
                        sum[n][m][l] +=
                            pShared_wei[j * LDS_WEI_STRIDE + (OUT_WI_ROW_START + m) + (OUT_WI_ROW_START + m) / 32] *
                            pShared_in[j + (OUT_WI_COL_START + l) * LDS_IN_STRIDE];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#elif BRANCH == 4

    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % QSTRIDE) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START * LDS_WEI_STRIDE + (LDS_WEI_ROW_START + i) + (LDS_WEI_ROW_START + i) / 32]
                = pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) * LDS_IN_STRIDE + IN_WI_COL_START] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] +=
                            pShared_wei[j * LDS_WEI_STRIDE + (OUT_WI_ROW_START + m) + (OUT_WI_ROW_START + m) / 32] *
                            pShared_in[j * LDS_IN_STRIDE + (OUT_WI_COL_START + l)];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#endif

#if ATOMIC == 0

    if (GROUP_ITER == 0) {
        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = sum[n][j][i] + pBias[j];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = sum[n][j][i];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }

#endif
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = sum[n][j][i] + pBias[j];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = sum[n][j][i];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }

#endif
        }
    }

#elif ATOMIC == 1

    if (GROUP_ITER == 0) {
        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
                    }

#endif
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
                    }

#endif
        }

#ifdef NOFILLED
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            *(pBarrier + COUNTER_INDEX) = BARRIER_MARK;
        }

#endif
    } else {
#ifdef NOFILLED

        if (lid_x == 0) {
            do {
                newVal = BARRIER_MARK;
            } while (atomic_cmpxchg((__global uint*)(pBarrier + COUNTER_INDEX), BARRIER_MARK,
                                    newVal) != BARRIER_MARK);
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
#endif

        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[n][j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[n][j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }

        if (GROUP_ITER == GSU_MINUS_ONE) {
            if (lid_x == 0) {
                do {
                    newVal = GSU_MINUS_ONE;
                } while (atomic_cmpxchg((__global uint*)(pCounter + COUNTER_INDEX), GSU_MINUS_ONE,
                                        newVal) != GSU_MINUS_ONE);
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

            if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
                for (uint n = 0; n < N; n++)
                    for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                        for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                            pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] *= (pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] >
                                    0.0f ? 1.0f : slope);
                        }
            } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
                for (uint n = 0; n < N; n++)
                    for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                        for (uint i = 0; i < 1; i++) {
                            pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] *= (pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] >
                                    0.0f ? 1.0f : slope);
                        }
            }
        }
    }

#elif ATOMIC == 2

    if (GROUP_ITER == 0) {
        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
                    }

#endif
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
#if BIAS == 1

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i] + pBias[j];
                    }

#else

            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[n][j][i];
                    }

#endif
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }
    } else if (GROUP_ITER == GSU_MINUS_ONE) {
#ifdef NOFILLED

        if (lid_x == 0) {
            do {
            } while (atomic_cmpxchg((volatile __global uint*)(pCounter + COUNTER_INDEX), GROUP_ITER,
                                    GROUP_ITER) < GROUP_ITER);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
#endif

        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        temp += sum[n][j][i];
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp * (temp > 0.0f ? 1.0f : slope);
                    }
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        temp += sum[n][j][i];
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp * (temp > 0.0f ? 1.0f : slope);
                    }
        }
    } else {
        if (lid_x == 0) {
            do {
            } while (atomic_cmpxchg((volatile __global uint*)(pCounter + COUNTER_INDEX), GROUP_ITER,
                                    GROUP_ITER) < 1);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

#if 1

        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[n][j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[n][j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
        }

#else

        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        temp += sum[n][j][i];
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp;
                    }
        } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
            for (uint n = 0; n < N; n++)
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                        temp += sum[n][j][i];
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = temp;
                    }
        }

#endif
        barrier(CLK_GLOBAL_MEM_FENCE);

        if (lid_x == 0) {
            atomic_inc(pCounter + COUNTER_INDEX);
        }
    }

#endif
}

#elif  METHOD == 2


__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void conv1x1_act(
    __global const float* wei,
    __global const float* in,
#if BIAS == 1
    __constant float* bias,
#endif
    __global float* out,
    float slope) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float shared_wei[LDS_WEI_ROW * LDS_WEI_STRIDE];
    __local float shared_in[LDS_IN_ROW * LDS_IN_STRIDE];
    __local float* pShared_wei = (__local float*)shared_wei;
    __local float* pShared_in = (__local float*)shared_in;

    __global const float* pWei = (__global const float*)(wei + WEI_ROW_START * STRIDE_WEI +
                                 WEI_COL_START * QSTRIDE + WEI_WI_COL_START);//
#if STRIDE == 2
    __global const float* pIn = (__global const float*)(in + (IN_WI_COL_START_REAL) + WEI_COL_START *
                                QSTRIDE * STRIDE_IN_REAL);//
#else
    __global const float* pIn = (__global const float*)(in + (IN_COL_START + IN_WI_COL_START) +
                                WEI_COL_START * QSTRIDE * STRIDE_IN);//
#endif
    __global float* pOut = (__global float*)(out + (IN_COL_START + OUT_WI_COL_START) +
                           (WEI_ROW_START + OUT_WI_ROW_START) * STRIDE_IN);//

#if ATOMIC == 1
    __global uint* pBarrier = (__global uint*)(out + OUT_BATCH_STRIDE * N);//
    __global uint* pCounter = (__global uint*)(out + OUT_BATCH_STRIDE * N + COUNTER_STRIDE);//
#endif

#if BIAS == 1
    __constant float* pBias = (__constant float*)(bias + WEI_ROW_START + OUT_WI_ROW_START);//
#endif

    float previous_value;
    uint prevVal;
    uint newVal;

#if ATOMIC == 1

    if (grid_x < COUNTER_STRIDE) {
        if (lid_x == 0) {
            *(pBarrier + COUNTER_INDEX) = 0;
            *(pCounter + COUNTER_INDEX) = 0;
        }
    }

    //barrier(CLK_GLOBAL_MEM_FENCE);
#endif

    for (uint n = 0; n < N; n++) {
        float sum[PER_WI_TILE_ROW][PER_WI_TILE_COL] = { { 0.0f } };

        uint offset = OFFSET;

        for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % QSTRIDE) {
            for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
                shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                    pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
            }

            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#if STRIDE == 2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_WEI_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL + n * IN_BATCH_STRIDE] : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_WEI_STRIDE] =
                    (IN_COL_START + IN_WI_COL_START < STRIDE_IN ?
                     pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] : 0.0f);
#endif
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[m][l] += pShared_wei[j + ((OUT_WI_ROW_START + m) * LDS_WEI_STRIDE)] *
                                     pShared_in[j + (OUT_WI_COL_START + l) * LDS_WEI_STRIDE];
                    }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

#if ATOMIC == 0
#elif ATOMIC == 1

        if (GROUP_ITER == 0) {

            if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
#if BIAS == 1

                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[j][i] + pBias[j];
                    }

#else

                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[j][i];
                    }

#endif
            } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
#if BIAS == 1

                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[j][i] + pBias[j];
                    }

#else

                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE) = sum[j][i];
                    }

#endif
            }

#ifdef NOFILLED
            barrier(CLK_GLOBAL_MEM_FENCE);

            if (lid_x == 0) {
                *(pBarrier + COUNTER_INDEX) = BARRIER_MARK;
            }

#endif
        } else {
#ifdef NOFILLED

            if (lid_x == 0) {
                do {
                    newVal = BARRIER_MARK;
                } while (atomic_cmpxchg((__global uint*)(pBarrier + COUNTER_INDEX), BARRIER_MARK,
                                        newVal) != BARRIER_MARK);
            }

            barrier(CLK_GLOBAL_MEM_FENCE);
#endif

            if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
            } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        do {
                            previous_value = *(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE);
                            prevVal = as_uint(previous_value);
                            newVal = as_uint(sum[j][i] + previous_value);
                        } while (atomic_cmpxchg((__global uint*)(pOut + j * STRIDE_IN + i + n * OUT_BATCH_STRIDE), prevVal,
                                                newVal) != prevVal);
                    }
            }

            barrier(CLK_GLOBAL_MEM_FENCE);

            if (lid_x == 0) {
                atomic_inc(pCounter + COUNTER_INDEX);
            }

            if (GROUP_ITER == GSU_MINUS_ONE) {
#ifdef NOFILLED

                if (lid_x == 0) {
                    do {
                        newVal = 0;
                    } while (atomic_cmpxchg((__global uint*)(pCounter + COUNTER_INDEX), GSU_MINUS_ONE,
                                            newVal) != GSU_MINUS_ONE);

                    //*(pInit + COUNTER_INDEX) = 0;
                }

                barrier(CLK_GLOBAL_MEM_FENCE);
#endif

                if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
                    for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                        for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                            pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] *= (pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] >
                                    0.0f ? 1.0f : slope);
                        }
                } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
                    for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                        for (uint i = 0; i < 1; i++) {
                            pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] *= (pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] >
                                    0.0f ? 1.0f : slope);
                        }
                }
            }
        }

#endif
    }
}
#endif
