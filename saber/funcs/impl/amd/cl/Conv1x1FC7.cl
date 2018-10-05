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
#define N   1
#endif

#ifndef STRIDE
#define STRIDE (1024)
#endif


#define ITER (STRIDE >> 6)

#define OUTPUT 1000

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __constant float* a, __global const float* b,
#ifdef BIAS
    __global const float* bias,
#endif
    __global float* c,
    float slope) {
    uint lid_x = get_local_id(0);
    uint grid_x = get_group_id(0);

    __local float result[N][66];

    if (grid_x < OUTPUT) {
        __constant float* pA;
        __global const float* pB = (__global const float*)(b + grid_x * STRIDE + lid_x); // correct
        __global float* pC;

        pA = (__constant float*)(a + lid_x);
        pC = (__global float*)(c + grid_x);

        uint offset = (grid_x >> 2 << 6) % STRIDE;

        float sum[N] = { 0.0f };

        for (uint i = 0; i < ITER; i++, offset = (offset + 64) % STRIDE) {
            for (uint n = 0; n < N; n++) {
                sum[n] += pA[offset + n * STRIDE] * pB[offset]; // correct
            }
        }

        for (uint n = 0; n < N; n++) {
            result[n][lid_x] = sum[n];

            if (lid_x < 32) {
                result[n][lid_x] += result[n][lid_x + 32];
            }

            if (lid_x < 16) {
                result[n][lid_x] += result[n][lid_x + 16];
            }

            if (lid_x < 8) {
                result[n][lid_x] += result[n][lid_x + 8];
            }

            if (lid_x < 4) {
                result[n][lid_x] += result[n][lid_x + 4];
            }

            if (lid_x < 2) {
                result[n][lid_x] += result[n][lid_x + 2];
            }

            if (lid_x < 1) {
                result[n][lid_x] += result[n][lid_x + 1];
            }

            if (lid_x == 0) {
#ifdef BIAS
                pC[n * OUTPUT] = bias[grid_x] + result[n][0];
#else
                pC[n * OUTPUT] = result[n][0];
#endif
                pC[n * OUTPUT] *= (pC[n * OUTPUT] > 0 ? 1.0f : slope);
            }
        }
    }
}
