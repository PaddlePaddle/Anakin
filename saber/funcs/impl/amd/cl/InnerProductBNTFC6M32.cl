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
/////////////////////////////////////////////////////////
// FC6 batch 32 Version 3 2018.6.25

#define STRIDE	(25088)
#define CSTRIDE (4096)
#define ITER	(1568)

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void InnerProduct(
    __global const float *a, 
    __global const float *b, 
    __global const float *bias,
    __global float *c)
{
	__local float shared_a[1024];
	__local float shared_b[1024];
	__local float4* pShared_a = (__local float4*)shared_a;
	__local float4* pShared_b = (__local float4*)shared_b;

	float4 sha;
	float4 shb;

	float4 sum = 0.0f;
	float* pSum = (float*)&sum;

	int gid_x = get_global_id(0);
	int lid_x = get_local_id(0);
	int grid_x = get_group_id(0);

	__global const float* pA = (__global const float*)(a);
	__global const float* pB = (__global const float*)(b + (grid_x << 5) * STRIDE);
	__global float4* pC = (__global float4*)(c + (lid_x >> 3) * CSTRIDE + (grid_x << 5) + ((lid_x & 7) << 2));

	int offset = (grid_x << 5) % STRIDE;

	for (int i = 0; i < ITER; i++, offset = (offset + 16) % STRIDE)
	{
		for (int j = 0; j < 2; j++)
		{
			shared_a[((j << 4) + (lid_x >> 4)) + (lid_x & 15) * 32 + ((((j << 4) + (lid_x >> 4)) + (lid_x & 15) * 32) >> 5 << 2)] = pA[((j << 4) + (lid_x >> 4)) * STRIDE + (lid_x & 15) + offset];
			shared_b[((j << 4) + (lid_x >> 4)) + (lid_x & 15) * 32 + ((((j << 4) + (lid_x >> 4)) + (lid_x & 15) * 32) >> 5 << 2)] = pB[((j << 4) + (lid_x >> 4)) * STRIDE + (lid_x & 15) + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k = 0; k < 16; k++)
		{
			sha = shared_a[(lid_x >> 3) + k * 32 + (((lid_x >> 3) + k * 32) >> 5 << 2)];
			shb = pShared_b[(((lid_x & 7))) + k * 8 + (((((lid_x & 7))) + k * 8) >> 3)];
			sum += sha * shb;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	for (int i = 0; i < 4; i++)
	{
		shared_a[(lid_x >> 3) * 32 + ((lid_x & 7) << 2) + i] = pSum[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	float4 result;
	float* pResult = (float*)&result;

	for (int i = 0; i < 4; i++)
	{
		pResult[i] = shared_a[(lid_x << 2) + i] + bias[(grid_x << 5) + ((lid_x & 7) << 2) + i];
	}

	*pC = result;
}

