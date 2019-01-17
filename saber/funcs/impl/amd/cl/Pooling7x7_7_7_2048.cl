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

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#define _FLOAT float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)
#define _INT_MASK_GLOBAL uchar
#define _INT_MASK_LOCAL uchar

#define UNUSED __attribute__((__unused__))

#define MLO_POOLING_OP_AVE 0
#define MLO_POOLING_OP_MAX 1
#define MLO_POOLING_OP_STC 2

#define MLO_POOLING_GROUP_SZ2 1

#ifndef MLO_POOLING_OP_ID
#define MLO_POOLING_OP_ID 0
#endif
// max
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
#define MLO_POOLING_OP(A, B) fmax(A, B);
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
#define MLO_POOLING_OP(A, B) (A + B);
#endif

/*********************************************************************************

**********************************************************************************/

#define KERNEL_SIZE (MLO_POOLING_KERNEL_SZ0 * MLO_POOLING_KERNEL_SZ1)

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void mloPoolingG(const __global _FLOAT* bot, __global _FLOAT* top, UNUSED __global _FLOAT* mask)
{
    int id = get_global_id(0);
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
    float result = -MAX_VAL;
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
    float result = 0.0f;
#endif

    for(int i=0; i<KERNEL_SIZE; i++)
    {
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
        result = MLO_POOLING_OP(result, bot[id * KERNEL_SIZE + i]);
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        result += bot[id * KERNEL_SIZE + i];
#endif
    }

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
    result = result / KERNEL_SIZE;
#endif

    top[id] = result;
}
