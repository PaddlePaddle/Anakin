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
#define STRIDE (25088)
#define CSTRIDE (4096)
#define ITER (392)

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
}

__attribute__((reqd_work_group_size(64, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c)
{
    __local float shared_a[2][66];
    __local float shared_b[8][66];
    __local float result[65];

    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __global const float* pA = (__global const float*)(a);
    __global const float* pB = (__global const float*)(b + (grid_x << 3) * STRIDE);

    int offset = ((grid_x << 6)) % STRIDE;
    float sum  = 0.0f;

    for(int i = 0; i < ITER; i++, offset = (offset + 64) % STRIDE)
    {
        for(int j = 0; j < 2; j++)
        {
            shared_a[j][lid_x + (lid_x >> 5)] = pA[offset + lid_x + j * STRIDE];
        }

        for(int j = 0; j < 8; j++)
        {
            shared_b[j][lid_x + (lid_x >> 5)] = pB[offset + lid_x + j * STRIDE];
        }

        for(int k = 0; k < 16; k++)
        {
            sum += shared_a[lid_x >> 5][((lid_x & 3) << 4) + k + ((((lid_x & 3) << 4) + k) >> 5)] *
                   shared_b[(lid_x & 31) >> 2]
                           [((lid_x & 3) << 4) + k + ((((lid_x & 3) << 4) + k) >> 5)];
        }
    }

    result[lid_x] = sum;
    reduce(result, lid_x);

    if(lid_x < 2)
    {
        float8 out;
        float* pOut = (float*)&out;

        for(int i = 0; i < 8; i++)
        {
            pOut[i] = result[((lid_x * 8 + i) << 2)] + bias[(grid_x << 3) + i];
        }

        __global float8* pC = (__global float8*)(c + (grid_x << 3) + lid_x * CSTRIDE);
        *pC                 = out;
    }
}
