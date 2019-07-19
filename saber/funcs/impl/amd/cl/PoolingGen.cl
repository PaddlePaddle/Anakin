/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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

#define Pooling_max 1
#define Pooling_average_include_padding 2
#define Pooling_average_exclude_padding 3

#ifndef MLO_POOLING_OP_ID
#define MLO_POOLING_OP_ID Pooling_max
#endif
// max
#if MLO_POOLING_OP_ID == Pooling_max
#define MLO_POOLING_OP(A, B) fmax(A, B);
#else
#define MLO_POOLING_OP(A, B) (A + B);
#endif

/*********************************************************************************

**********************************************************************************/

#define THREAD_PER_WAVE 64
#define WAVE_PER_4SIMD 40

__attribute__((reqd_work_group_size(256, 1, 1))) __kernel void
mloPooling(const __global _FLOAT* bot,
           __global _FLOAT* top,
#if MLO_CONV_BIAS
           const __global _FLOAT* bias,
#endif
#if MLO_CONV_PRELU
           _FLOAT negSlope,
#endif
           UNUSED __global _FLOAT* mask) {
    int gid     = get_global_id(0);
    int ob      = BATCH_NUM * MLO_POOLING_N_OUTPUTS; // output * batch_sz
    int top_off = gid;
    int bot_off = 0;

    _FLOAT res;

    uint loop_num = ((ob * MLO_POOLING_TOP_STRIDE * MLO_POOLING_TOP_HEIGHT +
                      THREAD_PER_WAVE * CU_NUM * WAVE_PER_4SIMD - 1) /
                     (THREAD_PER_WAVE * CU_NUM * WAVE_PER_4SIMD));
    int top_loop_stride = THREAD_PER_WAVE * CU_NUM * WAVE_PER_4SIMD;

    for (int index = 0;
            index < loop_num && top_off < ob * MLO_POOLING_TOP_STRIDE * MLO_POOLING_TOP_HEIGHT;
            index++, top_off += top_loop_stride) {
        int bot_b = (top_off / MLO_POOLING_TOP_BATCH_STRIDE);
        int bot_c = (top_off % MLO_POOLING_TOP_BATCH_STRIDE / MLO_POOLING_TOP_CHANNEL_STRIDE);
        int bot_y = (top_off % MLO_POOLING_TOP_CHANNEL_STRIDE / MLO_POOLING_TOP_STRIDE) *
                    MLO_POOLING_STRIDE1 - MLO_POOLING_PAD1;
        int bot_x = (top_off % MLO_POOLING_TOP_STRIDE) * MLO_POOLING_STRIDE0 - MLO_POOLING_PAD0;

        bot_off = bot_b * MLO_POOLING_BOT_BATCH_STRIDE + bot_c * MLO_POOLING_BOT_CHANNEL_STRIDE +
                  bot_y * MLO_POOLING_BOT_STRIDE + bot_x;
#if MLO_POOLING_OP_ID == Pooling_max
        res = -FLT_MAX;
#else
        res = 0;
#endif

#if MLO_POOLING_OP_ID != Pooling_max
        uint pool_size = 0;
#endif

        // for window x, y
        for (int j = 0; j < MLO_POOLING_KERNEL_SZ1; ++j) {
            for (int i = 0; i < MLO_POOLING_KERNEL_SZ0; ++i) {
                int run_y = (int)bot_y + j;
                int run_x = (int)bot_x + i;
                uint vis  = ((run_y >= 0 && run_y < MLO_POOLING_BOT_HEIGHT) &&
                             (run_x >= 0 && run_x < MLO_POOLING_BOT_WIDTH))
                            ? 1 : 0;

                if (vis) {
                    uint bot_gbl_off = bot_off + j * MLO_POOLING_BOT_STRIDE + i;
                    float bot_data   = *(bot + bot_gbl_off);

#if MLO_CONV_BIAS
                    bot_data = bot_data + bias[bot_c];
#endif

#if MLO_CONV_PRELU
                    bot_data = (bot_data > 0) ? bot_data : bot_data * negSlope;
#endif

#if MLO_POOLING_OP_ID != Pooling_max
                    bot_data = (vis) ? bot_data : 0;
#else
                    bot_data = (vis) ? bot_data : -FLT_MAX;
#endif
                    res = MLO_POOLING_OP(res, bot_data);
                }

#if MLO_POOLING_OP_ID != Pooling_max
#if MLO_POOLING_OP_ID == Pooling_average_include_padding
                vis = ((run_y < (MLO_POOLING_BOT_HEIGHT + MLO_POOLING_PAD1)) &&
                       (run_x < (MLO_POOLING_BOT_WIDTH  + MLO_POOLING_PAD0)))
                      ? 1 : 0;
#endif
                pool_size += vis;
#endif
            }
        }

#if MLO_POOLING_OP_ID != Pooling_max
        res *= (1.0f / (_FLOAT)pool_size);
#endif

        top[top_off] = res;
    }
}
