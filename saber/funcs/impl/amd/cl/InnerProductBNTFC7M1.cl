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
#define HSTRIDE (2048)
#define ITER (32)

void reduce(__local float* buffer, int tid)
{
    if(tid < 64)
    {
        buffer[tid] += buffer[tid + 64];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid < 32)
    {
        buffer[tid << 1] += buffer[(tid << 1) + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid < 16)
    {
        buffer[tid << 2] += buffer[(tid << 2) + 2];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid < 8)
    {
        buffer[tid << 3] += buffer[(tid << 3) + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(tid < 4)
    {
        buffer[tid << 4] += buffer[(tid << 4) + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void InnerProduct(
    __global const float* a, __global const float* b, __global const float* bias, __global float* c)
{
    __local float shared_a[129];
    __local float shared_b[8][65];

    __local float result[2][129];

    int gid_x  = get_global_id(0);
    int lid_x  = get_local_id(0);
    int grid_x = get_group_id(0);

    __global const float* pA = (__global const float*)(a + (grid_x >> 9) * STRIDE);
    __global const float* pB = (__global const float*)(b + ((grid_x & 511) << 3) * STRIDE);

    int offset = ((grid_x << 6) + ((lid_x >> 6) * HSTRIDE) + (lid_x & 63)) % STRIDE;

    int temp_offset = offset;
    for(int l = 0; l < 2; l++, offset = temp_offset)
    {
        float sum = 0.0f;
        for(int i = 0; i < ITER; i++, offset = (offset + 64) % STRIDE)
        {
            shared_a[lid_x] = pA[offset];

            for(int j = l * 4; j < (l + 1) * 4; j++)
            {
                shared_b[(lid_x >> 6 << 2) + (j & 3)][(lid_x & 63)] = pB[offset + j * STRIDE];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for(int k = 0; k < 4; k++)
            {
                sum += shared_a[(lid_x >> 6 << 6) + ((lid_x & 15) << 2) + k] *
                       shared_b[(lid_x >> 6 << 2) + ((lid_x & 63) >> 4)][((lid_x & 15) << 2) + k];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        result[l][lid_x] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        reduce(result[l], lid_x);
    }

    if(lid_x < 8)
    {
        c[(grid_x << 3) + lid_x] =
            result[lid_x >> 2][(lid_x & 3) << 4] + bias[(grid_x << 3) + lid_x];
    }
}
