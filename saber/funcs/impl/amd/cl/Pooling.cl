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

// example, change to #if 1 to enable
#if 0
#define MLO_POOLING_OP_ID 1
#define BATCH_NUM 1
#define CU_NUM 64
#define MLO_POOLING_N_OUTPUTS 64
#define MLO_POOLING_TOP_STRIDE 112
#define MLO_POOLING_TOP_HEIGHT 112
#define MLO_POOLING_TOP_CHANNEL_STRIDE (MLO_POOLING_TOP_STRIDE * MLO_POOLING_TOP_HEIGHT)
#define MLO_POOLING_TOP_BATCH_STRIDE (MLO_POOLING_TOP_CHANNEL_STRIDE * MLO_POOLING_N_OUTPUTS)
#define MLO_POOLING_BOT_STRIDE 224
#define MLO_POOLING_BOT_HEIGHT 224
#define MLO_POOLING_BOT_WIDTH 224
#define MLO_POOLING_BOT_CHANNEL_STRIDE (MLO_POOLING_BOT_STRIDE * MLO_POOLING_BOT_HEIGHT)
#define MLO_POOLING_BOT_BATCH_STRIDE (MLO_POOLING_BOT_CHANNEL_STRIDE * MLO_POOLING_N_OUTPUTS)
#endif

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

__attribute__((reqd_work_group_size(256, 1, 1))) __kernel void
mloPooling(const __global _FLOAT* bot, __global _FLOAT* top, UNUSED __global _FLOAT* mask)
{
    uint gid     = get_global_id(0);
    uint ob      = BATCH_NUM * MLO_POOLING_N_OUTPUTS; // output * batch_sz
    uint bot_off = 0;
    uint top_off = gid;

    _FLOAT2 bot_data[MLO_BOT_DATA_SZ1];
    _FLOAT res;

    uint loop_num = ((ob * MLO_POOLING_TOP_STRIDE * MLO_POOLING_TOP_HEIGHT +
                      THREAD_PER_WAVE * CU_NUM * WAVE_PER_4SIMD - 1) /
                     (THREAD_PER_WAVE * CU_NUM * WAVE_PER_4SIMD));
    uint top_loop_stride = THREAD_PER_WAVE * CU_NUM * WAVE_PER_4SIMD;

    for(int index = 0;
        index < loop_num && top_off < ob * MLO_POOLING_TOP_STRIDE * MLO_POOLING_TOP_HEIGHT;
        index++, top_off += top_loop_stride)
    {
        uint bot_b = (top_off / MLO_POOLING_TOP_BATCH_STRIDE);
        uint bot_c = (top_off % MLO_POOLING_TOP_BATCH_STRIDE / MLO_POOLING_TOP_CHANNEL_STRIDE);
        uint bot_y = (top_off % MLO_POOLING_TOP_CHANNEL_STRIDE / MLO_POOLING_TOP_STRIDE) * MLO_POOLING_STRIDE1;
        uint bot_x = (top_off % MLO_POOLING_TOP_STRIDE) * MLO_POOLING_STRIDE0;

        bot_off = bot_b * MLO_POOLING_BOT_BATCH_STRIDE + bot_c * MLO_POOLING_BOT_CHANNEL_STRIDE +
                  bot_y * MLO_POOLING_BOT_STRIDE + bot_x;
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_MAX
        res = -FLT_MAX;
#elif MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        res = 0;
#endif

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        uint pool_size = 0;
#endif
        // for window x, y
        for(uint j = 0; j < MLO_BOT_DATA_SZ1; ++j)
        {
            uint bot_gbl_off       = bot_off + j * MLO_POOLING_BOT_STRIDE;
            __global _FLOAT2* read = (__global _FLOAT2*)(bot + bot_gbl_off);
            bot_data[j]            = *read;

            int run_y = (int)j;
            int run_x = (int)bot_x;
            // for w_x = 0
            uint vis  = ((run_y >= 0 && run_y < MLO_POOLING_BOT_HEIGHT) &&
                        (run_x >= 0 && run_x < MLO_POOLING_BOT_WIDTH))
                           ? 1
                           : 0;
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
            bot_data[j].s0 = (vis) ? bot_data[j].s0 : 0;
            pool_size += vis;
#else
            bot_data[j].s0 = (vis) ? bot_data[j].s0 : -FLT_MAX;
#endif
            res = MLO_POOLING_OP(res, bot_data[j].s0);
            run_x++;

            // for w_x = 1
            vis = ((run_y >= 0 && run_y < MLO_POOLING_BOT_HEIGHT) &&
                   (run_x >= 0 && run_x < MLO_POOLING_BOT_WIDTH))
                      ? 1
                      : 0;
#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
            bot_data[j].s1 = (vis) ? bot_data[j].s1 : 0;
            pool_size += (uint)vis;
#else
            bot_data[j].s1 = (vis) ? bot_data[j].s1 : -FLT_MAX;
#endif
            res = MLO_POOLING_OP(res, bot_data[j].s1);
        }

#if MLO_POOLING_OP_ID == MLO_POOLING_OP_AVE
        res *= 1.f / (_FLOAT)pool_size;
#endif

        top[top_off] = res;
    }
}
