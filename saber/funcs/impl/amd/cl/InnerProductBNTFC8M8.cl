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

#define STRIDE (4096)
#define ITER (64)

#define OUTPUT 1000

void reduce(__local float* buffer, int tid)
{
    if(tid < 32)
    {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }
    if(tid < 16)
    {
        buffer[tid << 2] += buffer[(tid << 2) + 2];
    }
    if(tid < 8)
    {
        buffer[tid << 3] += buffer[(tid << 3) + 4];
    }
}

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c)
{
    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __local float shared_a[65];
    __local float shared_b[8][65];
    __local float result[64];

    __global const float* pA = (__global const float*)(a + (grid_x >> 7 << 12)); // correct
    __global const float* pB = (__global const float*)(b);                       // correct

    int offset = (((grid_x & 127) << 6)) % STRIDE;

    float sum = 0.0f;

    for(int i = 0; i < ITER; i++, offset = (offset + 64) % STRIDE)
    {
        shared_a[lid_x] = pA[offset + lid_x];

        for(int j = 0; j < 8; j++)
        {
            shared_b[j][(lid_x)] =
                (offset + (j + ((grid_x & 127) << 3)) * STRIDE + (lid_x) < OUTPUT * STRIDE
                     ? pB[offset + (j + ((grid_x & 127) << 3)) * STRIDE + (lid_x)]
                     : 0.0f); // correct
        }

        for(int k = 0; k < 8; k++)
        {
#if 1
            sum = mad(shared_a[((lid_x & 7) << 3) + k],
                      shared_b[((lid_x) >> 3)][((lid_x & 7) << 3) + k],
                      sum); // correct
#else
            sum += shared_a[((lid_x & 7) << 3) + k] *
                   shared_b[((lid_x) >> 3)][((lid_x & 7) << 3) + k]; // correct
#endif
        }
    }

    result[lid_x] = sum;

    reduce(result, lid_x);

    if(lid_x < 8 && ((grid_x & 127) << 3) + lid_x < OUTPUT)
    {
        int out_offset = ((grid_x >> 7) * OUTPUT + ((grid_x & 127) << 3) + lid_x);
        c[out_offset]  = bias[((grid_x & 127) << 3) + lid_x] + result[(lid_x << 3)];
    }
}
