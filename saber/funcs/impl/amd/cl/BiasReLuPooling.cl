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

#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8
#define _INT_MASK_GLOBAL uchar
#define _INT_MASK_LOCAL uchar

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F /* max value */
#endif

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

#define THREAD_PER_WAVE 64
#define WAVE_PER_4SIMD 40

#define MLO_BOT_DATA_SZ0 2
#define MLO_BOT_DATA_SZ1 2

//#define LOCAL_MEMORY

__attribute__((reqd_work_group_size(256, 1, 1))) __kernel void mloPooling(
    const __global _FLOAT* bot,
    __global _FLOAT* top,
#if MLO_CONV_BIAS
    __global _FLOAT* bias,
#endif
    float slope) {
    uint gid     = get_global_id(0);
    uint ob      = BATCH_NUM * MLO_POOLING_N_OUTPUTS; // output * batch_sz
    uint bot_off = 0;
    uint top_off = gid;

    _FLOAT2 bot_data[MLO_BOT_DATA_SZ1];
    _FLOAT res;

#ifdef LOCAL_MEMORY
    __local _FLOAT write_combine[256];
    __local _FLOAT4* p_write_combine = (__local _FLOAT4*)write_combine;
    __global _FLOAT4* p_top;
#endif

    uint loop_num =
        ((ob * MLO_POOLING_TOP_STRIDE * MLO_POOLING_TOP_HEIGHT
          + THREAD_PER_WAVE * CU_NUM * WAVE_PER_4SIMD - 1)
         / (THREAD_PER_WAVE * CU_NUM * WAVE_PER_4SIMD));
    uint top_loop_stride = THREAD_PER_WAVE * CU_NUM * WAVE_PER_4SIMD;

    for (int index = 0;
            index < loop_num && top_off < ob * MLO_POOLING_TOP_STRIDE * MLO_POOLING_TOP_HEIGHT;
            index++, top_off += top_loop_stride) {
        uint bot_b = (top_off / MLO_POOLING_TOP_BATCH_STRIDE);
        uint bot_c = (top_off % MLO_POOLING_TOP_BATCH_STRIDE / MLO_POOLING_TOP_CHANNEL_STRIDE);
        uint bot_y = (top_off % MLO_POOLING_TOP_CHANNEL_STRIDE / MLO_POOLING_TOP_STRIDE) << 1;
        uint bot_x = (top_off % MLO_POOLING_TOP_STRIDE) << 1;

        bot_off = bot_b * MLO_POOLING_BOT_BATCH_STRIDE + bot_c * MLO_POOLING_BOT_CHANNEL_STRIDE
                  + bot_y * MLO_POOLING_BOT_STRIDE + bot_x;
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
        res = -FLT_MAX;
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        res          = 0;
#endif

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        uint pool_size = 0;
#endif

        for (uint j = 0; j < MLO_BOT_DATA_SZ1; ++j) {
            uint bot_gbl_off       = bot_off + j * MLO_POOLING_BOT_STRIDE;
            __global _FLOAT2* read = (__global _FLOAT2*)(bot + bot_gbl_off);
            bot_data[j]            = *read;
#if MLO_CONV_BIAS
            bot_data[j] += bias[bot_c];
#endif
            bot_data[j].s0 *= (bot_data[j].s0 > 0.0f ? 1.0f : slope);
            bot_data[j].s1 *= (bot_data[j].s1 > 0.0f ? 1.0f : slope);

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
            pool_size += (uint)vis;
#endif
            res = MLO_POOLING_OP(res, bot_data[j].s0);

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
            pool_size += (uint)vis;
#endif
            res = MLO_POOLING_OP(res, bot_data[j].s1);
        }

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        res *= 1.f / (_FLOAT)pool_size;
#endif

#ifdef LOCAL_MEMORY
        write_combine[get_local_id(0)] = res;

        if (get_local_id(0) % 4 == 0) {
            p_top  = (__global _FLOAT4*)(top + top_off);
            *p_top = p_write_combine[get_local_id(0) / 4];
        }

#else
        top[top_off] = res;
#endif
    }
}
