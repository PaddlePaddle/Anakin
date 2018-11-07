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
#define STRIDE2

#ifndef N
#define N (1)
#endif
#ifndef C
#define C (256)
#endif
#ifndef H
#define H (56)
#endif
#ifndef W
#define W (56)
#endif
#ifndef K
#define K (512)
#endif

#define GLOBAL_SPLITU (1)
#define GSU_MINUS_ONE (GLOBAL_SPLITU - 1)
#define BARRIER_MARK (0xffffffff)
#define STRIDE_WEI (C)
#ifdef STRIDE2
#define OW (W / 2)
#define OH (H / 2)
#define STRIDE_IN_REAL (H * W)
#else
#define OW (W)
#define OH (H)
#endif
#define STRIDE_IN (OH * OW)

#define PER_ITER_STRIDE (32)
#define QSTRIDE (STRIDE_WEI / GLOBAL_SPLITU)
#define ITER (QSTRIDE / PER_ITER_STRIDE)
#define LDS_WEI_STRIDE (33)
#define LDS_WEI_ROW (64)
#define LDS_IN_STRIDE (33)
#define LDS_IN_ROW (32)
#define LDS_WEI_READ_ITER (8)
#define LDS_IN_READ_ITER (4)
#define PER_WI_TILE_ROW (4)
#define PER_WI_TILE_COL (2)

#define IN_BATCH_STRIDE (C * H * W)
#define OUT_BATCH_STRIDE (K * OH * OW)

#define WG_PER_IN 25
#define WG_PER_WEI 8
#define COUNTER_STRIDE (WG_PER_IN * WG_PER_WEI)
#define GROUP_ITER (grid_x / COUNTER_STRIDE)
#define WEI_ROW_START ((grid_x % COUNTER_STRIDE) / WG_PER_IN << 6)
#define WEI_COL_START (grid_x / COUNTER_STRIDE)
#define WEI_WI_COL_START (lid_x & 31)
#ifdef STRIDE2
#define IN_COL_START_REAL (((grid_x % WG_PER_IN) << 5) + (lid_x & 31))
#define IN_WI_COL_START_REAL (IN_COL_START_REAL / OW * OW * 4 + IN_COL_START_REAL % OW * 2)
#define IN_COL_START ((grid_x % WG_PER_IN) << 5)
#define IN_WI_COL_START (lid_x & 31)
#else
#define IN_COL_START ((grid_x % WG_PER_IN) << 5)
#define IN_WI_COL_START (lid_x & 31)
#endif
#define OUT_WI_ROW_START (lid_x >> 4 << 2)
#define OUT_WI_COL_START ((lid_x & 15) << 1)
#define LDS_WEI_ROW_START ((lid_x >> 5) * LDS_WEI_READ_ITER)
#define LDS_IN_ROW_START ((lid_x >> 5) * LDS_IN_READ_ITER)
#define OFFSET (((grid_x % COUNTER_STRIDE) / WG_PER_IN << 5) % QSTRIDE)

#define COUNTER_INDEX (grid_x % COUNTER_STRIDE)

#define NOFILLED

// global: (256 * 200 * 1, 1, 1)

#if 1
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
#ifdef STRIDE2
    __global const float* pIn =
            (__global const float*)(in + (IN_WI_COL_START_REAL) + WEI_COL_START * QSTRIDE * STRIDE_IN_REAL); //
#else
    __global const float* pIn =
            (__global const float*)(in + (IN_COL_START + IN_WI_COL_START) + WEI_COL_START * QSTRIDE * STRIDE_IN); //
#endif
    __global float* pOut =
            (__global float*)(out + (IN_COL_START + OUT_WI_COL_START) + (WEI_ROW_START + OUT_WI_ROW_START) * STRIDE_IN); //
#ifdef BIAS
    __constant float* pBias = (__constant float*)(bias + WEI_ROW_START + OUT_WI_ROW_START); //
#endif

    float previous_value;
    uint prevVal;
    uint newVal;

    float sum[N][PER_WI_TILE_ROW][PER_WI_TILE_COL] = {{{0.0f}}};

    uint offset = OFFSET;

    for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % QSTRIDE) {
        for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
            shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                    pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
        }

        for (uint n = 0; n < N; n++) {
            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#ifdef STRIDE2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_WEI_STRIDE] =
                        (IN_COL_START + IN_WI_COL_START < STRIDE_IN
                                 ? pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL
                                       + n * IN_BATCH_STRIDE]
                                 : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_WEI_STRIDE] =
                        (IN_COL_START + IN_WI_COL_START < STRIDE_IN
                                 ? pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN
                                       + n * IN_BATCH_STRIDE]
                                 : 0.0f);
#endif
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[n][m][l] += pShared_wei[j + ((OUT_WI_ROW_START + m) * LDS_WEI_STRIDE)]
                                        * pShared_in[j + (OUT_WI_COL_START + l) * LDS_WEI_STRIDE];
                    }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (GROUP_ITER == 0) {
        if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
#ifdef BIAS
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
#ifdef BIAS
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
}

#else
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
#ifdef STRIDE2
    __global const float* pIn =
            (__global const float*)(in + (IN_WI_COL_START_REAL) + WEI_COL_START * QSTRIDE * STRIDE_IN_REAL); //
#else
    __global const float* pIn =
            (__global const float*)(in + (IN_COL_START + IN_WI_COL_START) + WEI_COL_START * QSTRIDE * STRIDE_IN); //
#endif
    __global float* pOut =
            (__global float*)(out + (IN_COL_START + OUT_WI_COL_START) + (WEI_ROW_START + OUT_WI_ROW_START) * STRIDE_IN); //
#ifdef BIAS
    __constant float* pBias = (__constant float*)(bias + WEI_ROW_START + OUT_WI_ROW_START); //
#endif

    float previous_value;
    uint prevVal;
    uint newVal;

    for (uint n = 0; n < N; n++) {
        float sum[PER_WI_TILE_ROW][PER_WI_TILE_COL] = {{0.0f}};

        uint offset = OFFSET;

        for (uchar k = 0; k < ITER; k++, offset = (offset + PER_ITER_STRIDE) % QSTRIDE) {
            for (uchar i = 0; i < LDS_WEI_READ_ITER; i++) {
                shared_wei[WEI_WI_COL_START + (LDS_WEI_ROW_START + i) * LDS_WEI_STRIDE] =
                        pWei[(LDS_WEI_ROW_START + i) * STRIDE_WEI + offset];
            }

            for (uchar i = 0; i < LDS_IN_READ_ITER; i++) {
#ifdef STRIDE2
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_WEI_STRIDE] =
                        (IN_COL_START + IN_WI_COL_START < STRIDE_IN
                                 ? pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN_REAL
                                       + n * IN_BATCH_STRIDE]
                                 : 0.0f);
#else
                shared_in[(LDS_IN_ROW_START + i) + IN_WI_COL_START * LDS_WEI_STRIDE] =
                        (IN_COL_START + IN_WI_COL_START < STRIDE_IN
                                 ? pIn[(LDS_IN_ROW_START + i + offset) * STRIDE_IN
                                       + n * IN_BATCH_STRIDE]
                                 : 0.0f);
#endif
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < PER_ITER_STRIDE; j++) {
                for (uchar m = 0; m < PER_WI_TILE_ROW; m++)
                    for (uchar l = 0; l < PER_WI_TILE_COL; l++) {
                        sum[m][l] += pShared_wei[j + ((OUT_WI_ROW_START + m) * LDS_WEI_STRIDE)]
                                     * pShared_in[j + (OUT_WI_COL_START + l) * LDS_WEI_STRIDE];
                    }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (GROUP_ITER == 0) {

            if (IN_COL_START + OUT_WI_COL_START + 1 < STRIDE_IN) {
#ifdef BIAS
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = sum[j][i] + pBias[j];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }
#else
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < PER_WI_TILE_COL; i++) {
                        float temp = sum[j][i];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }
#endif
            } else if (IN_COL_START + OUT_WI_COL_START < STRIDE_IN) {
#ifdef BIAS
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = sum[j][i] + pBias[j];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }
#else
                for (uint j = 0; j < PER_WI_TILE_ROW; j++)
                    for (uint i = 0; i < 1; i++) {
                        float temp = sum[j][i];
                        temp *= (temp > 0.0f ? 1.0f : slope);
                        pOut[j * STRIDE_IN + i + n * OUT_BATCH_STRIDE] = temp;
                    }
#endif
            }
        }
    }

#endif
