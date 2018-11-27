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
#define STRIDE	(4096)
#define ITER	(64)

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void InnerProduct(
	__constant float *a,
    __global const float *b, 
    __global const float *bias,
    __global float *c)
{
	__local float shared_b[16][65];

	__local float result[64];

	int gid_x = get_global_id(0);
	int lid_x = get_local_id(0);
	int grid_x = get_group_id(0);

	__constant float* pA = (__constant float*)(a + (grid_x >> 8 << 12)); //correct
	__global const float* pB = (__global const float*)(b + ((grid_x & 255) << 4) * STRIDE + lid_x);

	int offset = ((grid_x & 255) << 6) % STRIDE;

	float sum = 0.0f;

	for (int i = 0; i < ITER; i++, offset = (offset + 64) % STRIDE)
	{
		for (int j = 0; j < 16; j++)
		{
			shared_b[j][lid_x] = pB[offset + j * STRIDE];
		}
		//barrier(CLK_LOCAL_MEM_FENCE);

		for (int k = 0; k < 16; k++)
		{
			sum += pA[offset + (lid_x >> 4 << 4) + k] * shared_b[(lid_x & 15)][(lid_x >> 4 << 4) + k];
		}
		//barrier(CLK_LOCAL_MEM_FENCE);
	}

	result[lid_x] = sum;
	//barrier(CLK_LOCAL_MEM_FENCE);

	if (lid_x < 16)
	{
		result[lid_x] += result[lid_x + 16] + result[lid_x + 32] + result[lid_x + 48];
		c[(grid_x << 4) + lid_x] = result[lid_x] + bias[((grid_x & 255) << 4) + lid_x];
	}
}

