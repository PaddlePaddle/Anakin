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
#define C   (1024)
#endif
#ifndef H
#define H   (7)
#endif
#ifndef W
#define W   (7)
#endif
#ifndef K
#define K   (1024)
#endif

#define STRIDE_WEI  C
#define STRIDE_IN   (49)
#define QSTRIDE (C >> 2)
#define ITER    (QSTRIDE >> 5)
#define LDS_WEI_STRIDE  (17)
#define LDS_IN_STRIDE   (132)

#define IN_BATCH_STRIDE     (C * H * W)
#define OUT_BATCH_STRIDE    (K)

#define ATOMIC

#ifdef DIR
void reduce(__local float* buffer, int tid) {
    if ((tid & 63) < 32) {
        buffer[tid + (tid >> 5)] += buffer[tid + 32 + ((tid + 32) >> 5)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((tid & 63) < 16) {
        buffer[tid + (tid >> 5)] += buffer[tid + 16 + ((tid + 16) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 8 + ((tid + 8) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 4 + ((tid + 4) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 2 + ((tid + 2) >> 5)];
        buffer[tid + (tid >> 5)] += buffer[tid + 1 + ((tid + 1) >> 5)];
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

    __global const float* pWei = (__global const float*)(wei + ((grid_x & 63) << 4) * STRIDE_WEI +
                                 (lid_x & 127));//
    __global const float* pIn = (__global const float*)(in + ((lid_x & 63)));//
    __global float* pOut = (__global float*)(out + ((grid_x & 63) << 4) + (lid_x >> 6));//
#ifdef BIAS
    __constant float* pBias = (__constant float*)(bias + ((grid_x & 63) << 4) + (lid_x >> 6));//
#endif


    float sum[N] = { 0.0f };

    uint offset = ((grid_x & 63) << 7) % STRIDE_WEI;

#ifdef DIR
#if 0

    for (uchar k = 0; k < ITER; k++, offset = (offset + 128) % STRIDE_WEI) {
        for (uchar i = 0; i < 2; i++) {
            shared_wei[(lid_x & 127) * 17 + ((lid_x >> 7 << 1) + i)] =
                pWei[((lid_x >> 7 << 1) + i) * STRIDE_WEI + offset];
        }

        for (uchar i = 0; i < 8; i++) {
            shared_in[((lid_x >> 6 << 3) + i) * 66 + (lid_x & 63) + ((lid_x & 63) >> 5)] = ((
                        lid_x & 63) < STRIDE_IN ? pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN] :
                    0.0f); //pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uchar j = 0; j < 128; j++) {
            sum += pShared_wei[j * 17 + (lid_x >> 6)] * pShared_in[j * 66 + (lid_x & 63) + ((lid_x & 63) >> 5)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

#elif 0

    for (uchar k = 0; k < ITER; k++, offset = (offset + 128) % STRIDE_WEI) {
        for (uchar i = 0; i < 2; i++) {
            shared_wei[(lid_x & 127) + ((lid_x & 127) >> 5) + ((lid_x >> 7 << 1) + i) * LDS_IN_STRIDE] =
                pWei[((lid_x >> 7 << 1) + i) * STRIDE_WEI + offset];
        }

        for (uchar i = 0; i < 8; i++) {
            shared_in[((lid_x >> 6 << 3) + i) + (((lid_x >> 6 << 3) + i) >> 5) + (lid_x & 63) * LDS_IN_STRIDE]
                = ((lid_x & 63) < STRIDE_IN ? pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN] :
                   0.0f); //pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uchar j = 0; j < 128; j++) {
            sum += pShared_wei[j + (j >> 5) + ((lid_x >> 6) * LDS_IN_STRIDE)] *
                   pShared_in[j + (j >> 5) + (lid_x & 63) * LDS_IN_STRIDE];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

#else

    for (uchar k = 0; k < ITER; k++, offset = (offset + 128) % STRIDE_WEI) {
        for (uchar i = 0; i < 2; i++) {
            shared_wei[(lid_x & 127) + ((lid_x & 127) >> 5) + ((lid_x >> 7 << 1) + i) * LDS_IN_STRIDE] =
                pWei[((lid_x >> 7 << 1) + i) * STRIDE_WEI + offset];
        }

        for (uchar n = 0; n < N; n++) {
            for (uchar i = 0; i < 8; i++) {
                shared_in[((lid_x >> 6 << 3) + i) * 66 + (lid_x & 63) + ((lid_x & 63) >> 5)] = ((
                            lid_x & 63) < STRIDE_IN ? pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN + n * IN_BATCH_STRIDE] :
                        0.0f); //pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for (uchar j = 0; j < 128; j++) {
                sum[n] += pShared_wei[j + (j >> 5) + ((lid_x >> 6) * LDS_IN_STRIDE)] *
                          pShared_in[j * 66 + (lid_x & 63) + ((lid_x & 63) >> 5)];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

#endif
#elif 1

    for (uchar k = 0; k < ITER; k++, offset = (offset + 128) % STRIDE_WEI) {
        for (uchar i = 0; i < 2; i++) {
            shared_wei[(lid_x & 127) + ((lid_x & 127) >> 5) + ((lid_x >> 7 << 1) + i) * LDS_IN_STRIDE] =
                pWei[((lid_x >> 7 << 1) + i) * STRIDE_WEI + offset];
        }

        for (uchar i = 0; i < 8; i++) {
            shared_in[((lid_x >> 6 << 3) + i) + (((lid_x >> 6 << 3) + i) >> 5) + (lid_x & 63) * LDS_IN_STRIDE]
                = ((lid_x & 63) < STRIDE_IN ? pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN] :
                   0.0f); //pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uchar j = 0; j < 32; j++) {
            for (uchar l = 0; l < 4; l++) {
                sum += pShared_wei[(lid_x >> 8 << 5) + j + (((lid_x >> 8 << 5) + j) >> 5) + ((((lid_x & 255) >> 4)) * LDS_IN_STRIDE)] *
                       pShared_in[(lid_x >> 8 << 5) + j + (((lid_x >> 8 << 5) + j) >> 5) + (((lid_x & 15) << 2) + l) * LDS_IN_STRIDE];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

#else

    for (uchar k = 0; k < ITER; k++, offset = (offset + 128) % STRIDE_WEI) {
        for (uchar i = 0; i < 2; i++) {
            shared_wei[(lid_x & 127) * 17 + ((lid_x >> 7 << 1) + i)] =
                pWei[((lid_x >> 7 << 1) + i) * STRIDE_WEI + offset];
        }

        for (uchar i = 0; i < 8; i++) {
#if 0
            shared_in[((lid_x >> 6 << 3) + i) * 66 + (lid_x & 63) + ((lid_x & 63) >> 5)] =
                pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN];
#else
            shared_in[((lid_x >> 6 << 3) + i) * 66 + (lid_x & 63) + ((lid_x & 63) >> 5)] = ((
                        lid_x & 63) < STRIDE_IN ? pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN] :
                    0.0f); //pIn[((lid_x >> 6 << 3) + i + offset) * STRIDE_IN];
#endif
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uchar j = 0; j < 32; j++) {
            for (uchar l = 0; l < 4; l++) {
                sum += pShared_wei[((lid_x >> 8 << 5) + j) * 17 + ((lid_x & 255) >> 4)] *
                       pShared_in[((lid_x >> 8 << 5) + j) * 66 + ((lid_x & 15) << 2) + l + ((((lid_x & 15) << 2) + l) >> 5)];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

#endif

#ifdef DIR

    for (uint n = 0; n < N; n++) {
#ifdef BIAS
        sum[n] += (lid_x & 63) < STRIDE_IN ? pBias[0] : 0.0f;
#endif
        sum[n] *= (sum[n] > 0.0f ? 1.0f : slope);
        shared_result[lid_x + (lid_x >> 5)] = sum[n];
        barrier(CLK_LOCAL_MEM_FENCE);

        reduce(shared_result, lid_x);

        if ((lid_x & 63) == 0) {
            pOut[n * OUT_BATCH_STRIDE] = shared_result[(lid_x) + (lid_x >> 5)] / 49.0f;
        }
    }

#else
#ifdef BIAS
    sum += (lid_x & 63) < STRIDE_IN ? pBias[0] : 0.0f;
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

