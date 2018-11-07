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

//#ifndef STRIDE
//#define STRIDE (1024)
//#endif

#define ITER (STRIDE >> 8)

#define OUTPUT 1000

__attribute__((reqd_work_group_size(256, 1, 1))) __kernel void InnerProduct(
        __constant float* a,
        __global const float* b,
#ifdef BIAS
        __constant float* bias,
#endif
        __global float* c) {
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __local float shared_b[64][66];
    __local float result[256];

    __constant float* pA = (__constant float*)a;
    __global const float* pB =
            (__global const float*)(b + ((grid_x << 4)) * STRIDE + (lid_x & 63)); // correct

    int offset = ((grid_x >> 2 << 6) + (lid_x >> 6 << 6) * ITER) % STRIDE;

    float sum = 0.0f;

    for (int i = 0; i < ITER; i++, offset = (offset + 64) % STRIDE) {
        for (int j = 0; j < 16; j++) {
            shared_b[(lid_x >> 6 << 4) + j][(lid_x & 63) + ((lid_x & 63) >> 5)] =
                    (offset + j * STRIDE + ((grid_x << 4)) * STRIDE + (lid_x & 63) < OUTPUT * STRIDE
                             ? pB[(offset + j * STRIDE)]
                             : 0.0f); // correct
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < 16; k++) {
            sum += pA[(offset + ((lid_x & 3) << 4) + k) % STRIDE]
                   * shared_b[(lid_x >> 2)]
                             [(((lid_x & 3) << 4) + k)
                              + ((((lid_x & 3) << 4) + k) >> 5)]; // correct
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    result[lid_x] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid_x < 64) {
        result[(lid_x << 2)] +=
                result[(lid_x << 2) + 1] + result[(lid_x << 2) + 2] + result[(lid_x << 2) + 3];
        barrier(CLK_LOCAL_MEM_FENCE);
        result[(lid_x << 2)] +=
                result[(lid_x << 2) + 64] + result[(lid_x << 2) + 128] + result[(lid_x << 2) + 192];

        if (lid_x < 16 && (grid_x << 4) + lid_x < OUTPUT) {
#ifdef BIAS
            c[(grid_x << 4) + lid_x] = bias[(grid_x << 4) + lid_x] + result[(lid_x << 2)];
#else
            c[(grid_x << 4) + lid_x] = result[(lid_x << 2)];
#endif
        }
    }
}
