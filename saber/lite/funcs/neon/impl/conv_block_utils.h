/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_LITE_FUNCS_CONV_BLOCK_UTILS_H
#define ANAKIN_SABER_LITE_FUNCS_CONV_BLOCK_UTILS_H
#include "saber/lite/core/common_lite.h"

#ifdef USE_ARM_PLACE
namespace anakin{
namespace saber{
namespace lite{

#define AKMAX(a, b) (a) > (b)? (a) : (b)

/*preprocessing weights
* input weights: [chout, chin/ group, 3, 3] --> outputs weights: [chout / n, chin/ group, 3, 3 * n]
*/
template <typename dtype>
static SaberStatus conv_trans_weights_numc(const dtype* din, dtype* dout, int chout, int chin, int n, int kernel_size) {
    if (n <= 0){
        LOGE("ch_n and hei_n are more than zero\n");
        return SaberInvalidValue;
    }
    int c_loop = chout / n;
    int chout_round = (chout + n - 1) / n;
    int win_stride = chin * kernel_size;
    int wout_stride = n * win_stride;
    int co = 0;
    for (; co < c_loop; ++co) {
        dtype* dout_c = dout + co * wout_stride;
        const dtype *din_array[n];
        din_array[0] = din + co * wout_stride;
        for (int i = 1; i < n; i++){
            din_array[i] = din_array[i - 1] + win_stride;
        }
        for (int ci = 0; ci < chin; ++ci) {
            for (int k = 0; k < kernel_size; ++k) {
                for (int i = 0; i < n; i++){
                    *(dout_c++) = * (din_array[i]++);
                }
            }
        }
    }
    // pad final chout
    if (chout_round > c_loop) {
        dtype* dout_c = dout + c_loop * wout_stride;
        const dtype *din_array[n];
        din_array[0] = din + c_loop * wout_stride;
        for (int i = 1; i < n; i++){
            din_array[i] = din_array[i - 1] + win_stride;
        }
        //deal remain
        int cremain = chout_round * n - chout;
        for (int i = 1; i <= cremain; i++){
            din_array[n - i] = din_array[0];
        }
        for (int ci = 0; ci < chin; ++ci) {
            for (int k = 0; k < kernel_size; ++k) {
                for (int i = 0; i < n; i++){
                    *(dout_c++) = * (din_array[i]++);
                }
            }
        }
    }
    return SaberSuccess;
}
/*preprocessing inputs
* input din: [1, chin, he-hs, we - ws] --> outputs dout: [n, chin, 1, we - ws]
* n = he - hs
*/
template <typename dtype>
static SaberStatus prepack_input_nxw(const dtype* din, dtype* dout, int n, int hs, int he, int ws, int we, \
    int channel, int width, int height, dtype* zero_ptr) {

    if (n <= 0){
        LOGE("hei_n is more than zero\n");
        return SaberInvalidValue;
    }
    int w0 = ws < 0? 0 : ws;
    int w1 = we > width? width : we;

    int size_w = we - ws;
    int size_wc_len = size_w * channel;
    int size_c = width * height;

    int valid_w = w1 - w0;
    size_t valid_w_byte = valid_w * sizeof(dtype);

    dtype *out_array[n];
    out_array[0] = dout;
    for (int i = 1; i < n; i++){
        out_array[i] = out_array[i - 1] + size_wc_len;
    }

    for (int c = 0; c < channel; ++c) {
        int j = 0;
        //valid height
        for (int i = hs; i < he; i++){
            //get address
            dtype *in_array;
            if (i < 0 || i >= height){
                in_array = zero_ptr;
            }else{
                in_array = din + i * width;
            }

            for (int w = ws; w < w0; ++w) {
                *(out_array[j]++) = 0.f;
            }
            memcpy(out_array[j], in_array, valid_w_byte);
            out_array[j] += valid_w;
            for (int w = w1; w < we; ++w) {
                *(out_array[j]++) = 0.f;
            }
            j++;
        }
        din += size_c;
    }
    return SaberSuccess;
}

/*wirte result in outputs
* input din: [n, c / 4, h, w * 4], output dout: [n, c, h, w]
*/
inline SaberStatus write_to_output_c4_fp32(const float* din, float* dout, int ch_n, int hei_n, int cs, int ce, \
    int hs, int he, int ws, int we, int channel, int height, int width, bool flag_relu, float* trash_ptr) {
    if (ch_n != 4 || hei_n <= 0){
        LOGE("ch_n must be equal 4 and hei_n is more than zero\n");
        return SaberInvalidValue;
    }
    int size_c_out = width * height;

    float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
    float* doutc1r0 = doutc0r0 + size_c_out;
    float* doutc2r0 = doutc1r0 + size_c_out;
    float* doutc3r0 = doutc2r0 + size_c_out;

    const float* ptr_din = din;

    int size_h = (he > height ? height : he) - hs; //size_h == hei_n

    int valid_w = we - ws;
    int cnt = valid_w / 4;

    if (we > width) {
        cnt--;
    }
    if (flag_relu){
        for (int i = 0; i < size_h; i++){
            int size_w = i * width;
            float* doutc0_ptr = doutc0r0 + size_w; //doutc0r0 + width;
            float* doutc1_ptr = doutc1r0 + size_w;
            float* doutc2_ptr = doutc2r0 + size_w;
            float* doutc3_ptr = doutc3r0 + size_w;
            if (ce > channel) {
                switch (ce - channel) {
                    case 3:
                        doutc1_ptr = trash_ptr;
                    case 2:
                        doutc2_ptr = trash_ptr;
                    case 1:
                        doutc3_ptr = trash_ptr;
                    default:
                        break;
                }
            }
            const float* din_hei_ptr = ptr_din + i * valid_w * ch_n;
            if (cnt > 0){
                int cnt_loop = cnt;
#ifdef __aarch64__
                asm volatile(
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "movi v20.4s, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "trn1   v8.4s, v0.4s, v1.4s     \n"         /* trans q0, q1*/
                    "trn2   v9.4s, v0.4s, v1.4s     \n"         /* trans q0, q1*/
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "trn1   v10.4s, v2.4s, v3.4s    \n"         /* trans q2, q3*/
                    "trn2   v11.4s, v2.4s, v3.4s    \n"         /* trans q2, q3*/
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "trn1   v16.2d, v8.2d, v10.2d   \n"         /* trans q8, q10*/
                    "trn2   v17.2d, v8.2d, v10.2d   \n"         /* trans q8, q10*/
                    "trn1   v18.2d, v9.2d, v11.2d   \n"         /* trans q9, q11*/
                    "trn2   v19.2d, v9.2d, v11.2d   \n"         /* trans q9, q11*/
                    "fmax   v16.4s, v16.4s, v20.4s  \n"         /*relu*/
                    "fmax   v17.4s, v17.4s, v20.4s  \n"         /*relu*/
                    "fmax   v18.4s, v18.4s, v20.4s  \n"         /*relu*/
                    "fmax   v19.4s, v19.4s, v20.4s  \n"         /*relu*/
                    "str    q16, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q17, [%[doutc2r0]], #16 \n"         /* store c2r0*/
                    "str    q18, [%[doutc1r0]], #16 \n"         /* store c1r0*/
                    "str    q19, [%[doutc3r0]], #16 \n"         /* store c3r0*/

                    "subs   %w[cnt], %w[cnt], #1    \n"         /* loop count -1*/
                    "bne    1b                      \n"         /* jump to main loop*/

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                [doutc3r0]"+r"(doutc3_ptr), [cnt] "+r"(cnt_loop), [ptr_din]"+r"(din_hei_ptr)
                :
                : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", \
                "v14", "v15", "v16", "v17", "v18", "v19", "v20"
                );
#else
                asm volatile(
                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"
                    "vmov.u32 q15, #0                @ dump zero\n"
                    "1:                              @ main loop\n"
                    "vtrn.32 q0, q1                  @ trans data:c00c01c20c21 \n"
                    "vtrn.32 q2, q3                  @ trans data:c02c03c22c23 \n"

                    "vmov.f32  d8, d1                  @ mov d1 to d8\n"
                    "vmov.f32  d9, d3                  @ mov d1 to d8\n"

                    "vmov.f32 d1, d4                   @ mov d1 to d8\n"
                    "vmov.f32 d3, d6                   @ mov d1 to d8\n"
                    "vmov.f32 d4, d8                   @ mov d1 to d8\n"
                    "vmov.f32 d6, d9                   @ mov d1 to d8\n"

                    "vmax.f32   q0, q0, q15        @ relu\n"
                    "vmax.f32   q1, q1, q15        @ relu\n"
                    "vmax.f32   q2, q2, q15        @ relu\n"
                    "vmax.f32   q3, q3, q15        @ relu\n"

                    "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d2-d3}, [%[doutc1r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d4-d5}, [%[doutc2r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d6-d7}, [%[doutc3r0]]!     @ store result, add pointer\n"

                    "subs   %[cnt], %[cnt], #1    @ loop count - 1\n"

                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"

                    "bne    1b                            @ jump to main loop\n"

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                [doutc3r0]"+r"(doutc3_ptr), [ptr_din]"+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
                :
                : "q0", "q1", "q2", "q3", "q4", "q15"
                );
#endif
            }
            if (we > width) {
                int offset = 16 * (valid_w / 4 - 1);
                din_hei_ptr  = ptr_din + offset;
                int i = we - 4;
                for (; i < width; ++i) {
                    *(doutc0_ptr++) = AKMAX(din_hei_ptr[0], 0.f);
                    *(doutc1_ptr++) = AKMAX(din_hei_ptr[1], 0.f);
                    *(doutc2_ptr++) = AKMAX(din_hei_ptr[2], 0.f);
                    *(doutc3_ptr++) = AKMAX(din_hei_ptr[3], 0.f);
                    din_hei_ptr += 4;
                }
            }
        }
    }else{
        for (int i = 0; i < size_h; i++){
            int size_w = i * width;
            float* doutc0_ptr = doutc0r0 + size_w; //doutc0r0 + width;
            float* doutc1_ptr = doutc1r0 + size_w;
            float* doutc2_ptr = doutc2r0 + size_w;
            float* doutc3_ptr = doutc3r0 + size_w;
            if (ce > channel) {
                switch (ce - channel) {
                    case 3:
                        doutc1_ptr = trash_ptr;
                    case 2:
                        doutc2_ptr = trash_ptr;
                    case 1:
                        doutc3_ptr = trash_ptr;
                    default:
                        break;
                }
            }
            const float* din_hei_ptr = ptr_din + i * valid_w * ch_n;
            if (cnt > 0){
                int cnt_loop = cnt;
#ifdef __aarch64__
                asm volatile(
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "movi v20.4s, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "trn1   v8.4s, v0.4s, v1.4s     \n"         /* trans q0, q1*/
                    "trn2   v9.4s, v0.4s, v1.4s     \n"         /* trans q0, q1*/
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "trn1   v10.4s, v2.4s, v3.4s    \n"         /* trans q2, q3*/
                    "trn2   v11.4s, v2.4s, v3.4s    \n"         /* trans q2, q3*/
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "trn1   v16.2d, v8.2d, v10.2d   \n"         /* trans q8, q10*/
                    "trn2   v17.2d, v8.2d, v10.2d   \n"         /* trans q8, q10*/
                    "trn1   v18.2d, v9.2d, v11.2d   \n"         /* trans q9, q11*/
                    "trn2   v19.2d, v9.2d, v11.2d   \n"         /* trans q9, q11*/

                    "str    q16, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q17, [%[doutc2r0]], #16 \n"         /* store c2r0*/
                    "str    q18, [%[doutc1r0]], #16 \n"         /* store c1r0*/
                    "str    q19, [%[doutc3r0]], #16 \n"         /* store c3r0*/

                    "subs   %w[cnt], %w[cnt], #1    \n"         /* loop count -1*/
                    "bne    1b                      \n"         /* jump to main loop*/

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                [doutc3r0]"+r"(doutc3_ptr), [cnt] "+r"(cnt_loop), [ptr_din]"+r"(din_hei_ptr)
                :
                : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", \
                "v14", "v15", "v16", "v17", "v18", "v19", "v20"
                );
#else
                asm volatile(
                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"
                    "vmov.u32 q15, #0                @ dump zero\n"
                    "1:                              @ main loop\n"
                    "vtrn.32 q0, q1                  @ trans data:c00c01c20c21 \n"
                    "vtrn.32 q2, q3                  @ trans data:c02c03c22c23 \n"

                    "vmov.f32  d8, d1                  @ mov d1 to d8\n"
                    "vmov.f32  d9, d3                  @ mov d1 to d8\n"

                    "vmov.f32 d1, d4                   @ mov d1 to d8\n"
                    "vmov.f32 d3, d6                   @ mov d1 to d8\n"
                    "vmov.f32 d4, d8                   @ mov d1 to d8\n"
                    "vmov.f32 d6, d9                   @ mov d1 to d8\n"

                    "subs   %[cnt], %[cnt], #1    @ loop count - 1\n"
                    "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d2-d3}, [%[doutc1r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d4-d5}, [%[doutc2r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d6-d7}, [%[doutc3r0]]!     @ store result, add pointer\n"

                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"

                    "bne    1b                            @ jump to main loop\n"

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                [doutc3r0]"+r"(doutc3_ptr), [ptr_din]"+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
                :
                : "q0", "q1", "q2", "q3", "q4", "q15"
                );
#endif
            }
            if (we > width) {
                int offset = 16 * (valid_w / 4 - 1);
                din_hei_ptr  = ptr_din + offset;
                int i = we - 4;
                for (; i < width; ++i) {
                    *(doutc0_ptr++) = din_hei_ptr[0];
                    *(doutc1_ptr++) = din_hei_ptr[1];
                    *(doutc2_ptr++) = din_hei_ptr[2];
                    *(doutc3_ptr++) = din_hei_ptr[3];
                    din_hei_ptr += 4;
                }
            }
        }
    }
    return SaberSuccess;
}

/*wirte result in outputs
* input din: [n, c / 8, h, w * 8], output dout: [n, c, h, w]
*/
inline SaberStatus write_to_output_c8_fp32(const float* din, float* dout, int ch_n, int hei_n, int cs, int ce, \
    int hs, int he, int ws, int we, int channel, int height, int width, bool flag_relu, float* trash_ptr) {
    if (ch_n != 8 || hei_n <= 0){
        LOGE("ch_n must be equal 8 and hei_n is more than zero\n");
        return SaberInvalidValue;
    }
    int size_c_out = width * height;

    float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
    float* doutc1r0 = doutc0r0 + size_c_out;
    float* doutc2r0 = doutc1r0 + size_c_out;
    float* doutc3r0 = doutc2r0 + size_c_out;
    float* doutc4r0 = doutc3r0 + size_c_out;
    float* doutc5r0 = doutc4r0 + size_c_out;
    float* doutc6r0 = doutc5r0 + size_c_out;
    float* doutc7r0 = doutc6r0 + size_c_out;

    const float* ptr_din = din;

    int size_h = (he > height ? height : he) - hs; //size_h == hei_n

    int valid_w = we - ws;
    int cnt = valid_w / 4;

    if (we > width) {
        cnt--;
    }
    if (flag_relu){
        for (int i = 0; i < size_h; i++){
            int size_w = i * width;
            float* doutc0_ptr = doutc0r0 + size_w; //doutc0r0 + width;
            float* doutc1_ptr = doutc1r0 + size_w;
            float* doutc2_ptr = doutc2r0 + size_w;
            float* doutc3_ptr = doutc3r0 + size_w;
            float* doutc4_ptr = doutc4r0 + size_w; //doutc0r0 + width;
            float* doutc5_ptr = doutc5r0 + size_w;
            float* doutc6_ptr = doutc6r0 + size_w;
            float* doutc7_ptr = doutc7r0 + size_w;
            if (ce > channel) {
                switch (ce - channel) {
                    case 7:
                        doutc1_ptr = trash_ptr;
                    case 6:
                        doutc2_ptr = trash_ptr;
                    case 5:
                        doutc3_ptr = trash_ptr;
                    case 4:
                        doutc4_ptr = trash_ptr;
                    case 3:
                        doutc5_ptr = trash_ptr;
                    case 2:
                        doutc6_ptr = trash_ptr;
                    case 1:
                        doutc7_ptr = trash_ptr;
                    default:
                        break;
                }
            }

            const float* din_hei_ptr = ptr_din + i * valid_w * ch_n;
            if (cnt > 0){
                int cnt_loop = cnt;
#ifdef __aarch64__
                asm volatile(
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "ldp q4, q5, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q6, q7, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "movi v20.4s, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "trn1   v8.4s, v0.4s, v2.4s     \n"         /* trans q0, q1*/
                    "trn2   v9.4s, v0.4s, v2.4s     \n"         /* trans q0, q1*/
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "trn1   v10.4s, v1.4s, v3.4s    \n"         /* trans q2, q3*/
                    "trn2   v11.4s, v1.4s, v3.4s    \n"         /* trans q2, q3*/
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */

                    "trn1   v12.4s, v4.4s, v6.4s     \n"         /* trans q0, q1*/
                    "trn2   v13.4s, v4.4s, v6.4s     \n"         /* trans q0, q1*/
                    "ldp q4, q5, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "trn1   v14.4s, v5.4s, v7.4s    \n"         /* trans q2, q3*/
                    "trn2   v15.4s, v5.4s, v7.4s    \n"         /* trans q2, q3*/
                    "ldp q6, q7, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */

                    "trn1   v16.2d, v8.2d, v12.2d   \n"         /* trans q8, q10 00 01 02 03*/
                    "trn2   v17.2d, v8.2d, v12.2d   \n"         /* trans q8, q10 20 21 22 23*/
                    "trn1   v18.2d, v9.2d, v13.2d   \n"         /* trans q9, q11 10 11 12 13*/
                    "trn2   v19.2d, v9.2d, v13.2d   \n"         /* trans q9, q11 30 31 32 33*/

                    "trn1   v8.2d, v10.2d, v14.2d   \n"         /* trans q8, q10 40 41 42 43*/
                    "trn2   v9.2d, v10.2d, v14.2d   \n"         /* trans q8, q10 60 61 62 63*/
                    "trn1   v12.2d, v11.2d, v15.2d   \n"         /* trans q9, q11 50 51 52 53*/
                    "trn2   v13.2d, v11.2d, v15.2d   \n"         /* trans q9, q11 70 71 72 73*/

                    "fmax   v16.4s, v16.4s, v20.4s  \n"         /*relu*/
                    "fmax   v17.4s, v17.4s, v20.4s  \n"         /*relu*/
                    "fmax   v18.4s, v18.4s, v20.4s  \n"         /*relu*/
                    "fmax   v19.4s, v19.4s, v20.4s  \n"         /*relu*/

                    "fmax   v8.4s, v8.4s, v20.4s  \n"         /*relu*/
                    "fmax   v9.4s, v9.4s, v20.4s  \n"         /*relu*/
                    "fmax   v12.4s, v12.4s, v20.4s  \n"         /*relu*/
                    "fmax   v13.4s, v13.4s, v20.4s  \n"         /*relu*/

                    "str    q16, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q17, [%[doutc2r0]], #16 \n"         /* store c2r0*/
                    "str    q18, [%[doutc1r0]], #16 \n"         /* store c1r0*/
                    "str    q19, [%[doutc3r0]], #16 \n"         /* store c3r0*/

                    "subs   %w[cnt], %w[cnt], #1    \n"         /* loop count -1*/
                    "str    q8, [%[doutc4r0]], #16 \n"         /* store c0r0*/
                    "str    q9, [%[doutc6r0]], #16 \n"         /* store c2r0*/
                    "str    q12, [%[doutc5r0]], #16 \n"         /* store c1r0*/
                    "str    q13, [%[doutc7r0]], #16 \n"         /* store c3r0*/

                    "bne    1b                      \n"         /* jump to main loop*/

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                  [doutc3r0]"+r"(doutc3_ptr), [doutc4r0]"+r"(doutc4_ptr), [doutc5r0]"+r"(doutc5_ptr), \
                  [doutc6r0]"+r"(doutc6_ptr), [doutc7r0]"+r"(doutc7_ptr),\
                  [cnt] "+r"(cnt_loop), [ptr_din]"+r"(din_hei_ptr)
                :
                : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", \
                "v14", "v15", "v16", "v17", "v18", "v19", "v20"
                );
#else
                asm volatile(
                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d8-d11}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d12-d15}, [%[ptr_din]]!        @load data \n"
                    "vmov.u32 q15, #0                @ dump zero\n"
                    "1:                              @ main loop\n"
                    "vtrn.32 q0, q2                  @ trans data:c00c01c20c21 c10c11c30c31\n"
                    "vtrn.32 q1, q3                  @ trans data:c40c41c60c61 c50c51c70c7n"
                    "vtrn.32 q4, q6                  @ trans data:c02c03c22c23 c12c13c32c33\n"
                    "vtrn.32 q5, q7                  @ trans data:c42c43c62c63 c52c53c72c73\n"

                    "vmov.f32  d16, d1                  @ mov d1 to d8\n"
                    "vmov.f32  d17, d3                  @ mov d1 to d8\n"
                    "vmov.f32  d18, d5                  @ mov d1 to d8\n"
                    "vmov.f32  d19, d7                  @ mov d1 to d8\n"

                    "vmov.f32 d1, d8                   @ mov d1 to d8 c00c01c02c03\n"
                    "vmov.f32 d3, d10                   @ mov d1 to d8 c40c41c42c43\n"
                    "vmov.f32 d5, d12                   @ mov d1 to d8 c10c11c12c13\n"
                    "vmov.f32 d7, d14                   @ mov d1 to d8 c50c51c52c53\n"

                    "vmov.f32 d8, d1                   @ mov d1 to d8 c20c21c22c23\n"
                    "vmov.f32 d10, d3                   @ mov d1 to d8 c60c61c62c63 n"
                    "vmov.f32 d12, d5                   @ mov d1 to d8 c10c11c12c13\n"
                    "vmov.f32 d14, d7                   @ mov d1 to d8 d70d71d72d73\n"

                    "vmax.f32   q0, q0, q15        @ relu\n"
                    "vmax.f32   q1, q1, q15        @ relu\n"
                    "vmax.f32   q2, q2, q15        @ relu\n"
                    "vmax.f32   q3, q3, q15        @ relu\n"

                    "vmax.f32   q4, q4, q15        @ relu\n"
                    "vmax.f32   q5, q5, q15        @ relu\n"
                    "vmax.f32   q6, q6, q15        @ relu\n"
                    "vmax.f32   q7, q7, q15        @ relu\n"

                    "subs   %[cnt], %[cnt], #1    @ loop count - 1\n"
                    "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d2-d3}, [%[doutc4r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d4-d5}, [%[doutc1r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d6-d7}, [%[doutc5r0]]!     @ store result, add pointer\n"

                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"

                    "vst1.32  {d8-d9}, [%[doutc2r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d10-d11}, [%[doutc6r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d12-d13}, [%[doutc1r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d14-d15}, [%[doutc7r0]]!     @ store result, add pointer\n"

                    "vld1.32 {d8-d11}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d12-d15}, [%[ptr_din]]!        @load data \n"

                    "bne    1b                            @ jump to main loop\n"

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                  [doutc3r0]"+r"(doutc3_ptr), [doutc4r0]"+r"(doutc4_ptr), [doutc5r0]"+r"(doutc5_ptr), \
                  [doutc6r0]"+r"(doutc6_ptr), [doutc7r0]"+r"(doutc7_ptr),\
                  [ptr_din]"+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
                :
                : "q0", "q1", "q2", "q3", "q4", "q15"
                );
#endif
            }
            if (we > width) {
                int offset = 32 * (valid_w / 4 - 1);
                din_hei_ptr  = ptr_din + offset;
                int i = we - 4;
                for (; i < width; ++i) {
                    *(doutc0_ptr++) = AKMAX(din_hei_ptr[0], 0.f);
                    *(doutc1_ptr++) = AKMAX(din_hei_ptr[1], 0.f);
                    *(doutc2_ptr++) = AKMAX(din_hei_ptr[2], 0.f);
                    *(doutc3_ptr++) = AKMAX(din_hei_ptr[3], 0.f);
                    *(doutc4_ptr++) = AKMAX(din_hei_ptr[4], 0.f);
                    *(doutc5_ptr++) = AKMAX(din_hei_ptr[5], 0.f);
                    *(doutc6_ptr++) = AKMAX(din_hei_ptr[6], 0.f);
                    *(doutc7_ptr++) = AKMAX(din_hei_ptr[7], 0.f);
                    din_hei_ptr += 8;
                }
            }
        }
    }else{
        for (int i = 0; i < size_h; i++){
            int size_w = i * width;
            float* doutc0_ptr = doutc0r0 + size_w; //doutc0r0 + width;
            float* doutc1_ptr = doutc1r0 + size_w;
            float* doutc2_ptr = doutc2r0 + size_w;
            float* doutc3_ptr = doutc3r0 + size_w;
            float* doutc4_ptr = doutc4r0 + size_w; //doutc0r0 + width;
            float* doutc5_ptr = doutc5r0 + size_w;
            float* doutc6_ptr = doutc6r0 + size_w;
            float* doutc7_ptr = doutc7r0 + size_w;
            if (ce > channel) {
                switch (ce - channel) {
                    case 7:
                        doutc1_ptr = trash_ptr;
                    case 6:
                        doutc2_ptr = trash_ptr;
                    case 5:
                        doutc3_ptr = trash_ptr;
                    case 4:
                        doutc4_ptr = trash_ptr;
                    case 3:
                        doutc5_ptr = trash_ptr;
                    case 2:
                        doutc6_ptr = trash_ptr;
                    case 1:
                        doutc7_ptr = trash_ptr;
                    default:
                        break;
                }
            }

            const float* din_hei_ptr = ptr_din + i * valid_w * ch_n;
            if (cnt > 0){
                int cnt_loop = cnt;
#ifdef __aarch64__
                asm volatile(
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "ldp q4, q5, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q6, q7, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "movi v20.4s, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "trn1   v8.4s, v0.4s, v2.4s     \n"         /* trans q0, q1*/
                    "trn2   v9.4s, v0.4s, v2.4s     \n"         /* trans q0, q1*/
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "trn1   v10.4s, v1.4s, v3.4s    \n"         /* trans q2, q3*/
                    "trn2   v11.4s, v1.4s, v3.4s    \n"         /* trans q2, q3*/
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */

                    "trn1   v12.4s, v4.4s, v6.4s     \n"         /* trans q0, q1*/
                    "trn2   v13.4s, v4.4s, v6.4s     \n"         /* trans q0, q1*/
                    "ldp q4, q5, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "trn1   v14.4s, v5.4s, v7.4s    \n"         /* trans q2, q3*/
                    "trn2   v15.4s, v5.4s, v7.4s    \n"         /* trans q2, q3*/
                    "ldp q6, q7, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */

                    "trn1   v16.2d, v8.2d, v12.2d   \n"         /* trans q8, q10 00 01 02 03*/
                    "trn2   v17.2d, v8.2d, v12.2d   \n"         /* trans q8, q10 20 21 22 23*/
                    "trn1   v18.2d, v9.2d, v13.2d   \n"         /* trans q9, q11 10 11 12 13*/
                    "trn2   v19.2d, v9.2d, v13.2d   \n"         /* trans q9, q11 30 31 32 33*/

                    "trn1   v8.2d, v10.2d, v14.2d   \n"         /* trans q8, q10 40 41 42 43*/
                    "trn2   v9.2d, v10.2d, v14.2d   \n"         /* trans q8, q10 60 61 62 63*/
                    "trn1   v12.2d, v11.2d, v15.2d   \n"         /* trans q9, q11 50 51 52 53*/
                    "trn2   v13.2d, v11.2d, v15.2d   \n"         /* trans q9, q11 70 71 72 73*/

                    "str    q16, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q17, [%[doutc2r0]], #16 \n"         /* store c2r0*/
                    "str    q18, [%[doutc1r0]], #16 \n"         /* store c1r0*/
                    "str    q19, [%[doutc3r0]], #16 \n"         /* store c3r0*/

                    "subs   %w[cnt], %w[cnt], #1    \n"         /* loop count -1*/
                    "str    q8, [%[doutc4r0]], #16 \n"         /* store c0r0*/
                    "str    q9, [%[doutc6r0]], #16 \n"         /* store c2r0*/
                    "str    q12, [%[doutc5r0]], #16 \n"         /* store c1r0*/
                    "str    q13, [%[doutc7r0]], #16 \n"         /* store c3r0*/

                    "bne    1b                      \n"         /* jump to main loop*/

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                  [doutc3r0]"+r"(doutc3_ptr), [doutc4r0]"+r"(doutc4_ptr), [doutc5r0]"+r"(doutc5_ptr), \
                  [doutc6r0]"+r"(doutc6_ptr), [doutc7r0]"+r"(doutc7_ptr),\
                  [cnt] "+r"(cnt_loop), [ptr_din]"+r"(din_hei_ptr)
                :
                : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", \
                "v14", "v15", "v16", "v17", "v18", "v19", "v20"
                );
#else
                asm volatile(
                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d8-d11}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d12-d15}, [%[ptr_din]]!        @load data \n"
                    "vmov.u32 q15, #0                @ dump zero\n"
                    "1:                              @ main loop\n"
                    "vtrn.32 q0, q2                  @ trans data:c00c01c20c21 c10c11c30c31\n"
                    "vtrn.32 q1, q3                  @ trans data:c40c41c60c61 c50c51c70c7n"
                    "vtrn.32 q4, q6                  @ trans data:c02c03c22c23 c12c13c32c33\n"
                    "vtrn.32 q5, q7                  @ trans data:c42c43c62c63 c52c53c72c73\n"

                    "vmov.f32  d16, d1                  @ mov d1 to d8\n"
                    "vmov.f32  d17, d3                  @ mov d1 to d8\n"
                    "vmov.f32  d18, d5                  @ mov d1 to d8\n"
                    "vmov.f32  d19, d7                  @ mov d1 to d8\n"

                    "vmov.f32 d1, d8                   @ mov d1 to d8 c00c01c02c03\n"
                    "vmov.f32 d3, d10                   @ mov d1 to d8 c40c41c42c43\n"
                    "vmov.f32 d5, d12                   @ mov d1 to d8 c10c11c12c13\n"
                    "vmov.f32 d7, d14                   @ mov d1 to d8 c50c51c52c53\n"

                    "vmov.f32 d8, d1                   @ mov d1 to d8 c20c21c22c23\n"
                    "vmov.f32 d10, d3                   @ mov d1 to d8 c60c61c62c63 n"
                    "vmov.f32 d12, d5                   @ mov d1 to d8 c10c11c12c13\n"
                    "vmov.f32 d14, d7                   @ mov d1 to d8 d70d71d72d73\n"

                    "subs   %[cnt], %[cnt], #1    @ loop count - 1\n"
                    "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d2-d3}, [%[doutc4r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d4-d5}, [%[doutc1r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d6-d7}, [%[doutc5r0]]!     @ store result, add pointer\n"

                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"

                    "vst1.32  {d8-d9}, [%[doutc2r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d10-d11}, [%[doutc6r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d12-d13}, [%[doutc1r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d14-d15}, [%[doutc7r0]]!     @ store result, add pointer\n"

                    "vld1.32 {d8-d11}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d12-d15}, [%[ptr_din]]!        @load data \n"

                    "bne    1b                            @ jump to main loop\n"

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                  [doutc3r0]"+r"(doutc3_ptr), [doutc4r0]"+r"(doutc4_ptr), [doutc5r0]"+r"(doutc5_ptr), \
                  [doutc6r0]"+r"(doutc6_ptr), [doutc7r0]"+r"(doutc7_ptr),\
                  [ptr_din]"+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
                :
                : "q0", "q1", "q2", "q3", "q4", "q15"
                );
#endif
            }
            if (we > width) {
                int offset = 32 * (valid_w / 4 - 1);
                din_hei_ptr  = ptr_din + offset;
                int i = we - 4;
                for (; i < width; ++i) {
                    *(doutc0_ptr++) = din_hei_ptr[0];
                    *(doutc1_ptr++) = din_hei_ptr[1];
                    *(doutc2_ptr++) = din_hei_ptr[2];
                    *(doutc3_ptr++) = din_hei_ptr[3];
                    *(doutc4_ptr++) = din_hei_ptr[4];
                    *(doutc5_ptr++) = din_hei_ptr[5];
                    *(doutc6_ptr++) = din_hei_ptr[6];
                    *(doutc7_ptr++) = din_hei_ptr[7];
                    din_hei_ptr += 8;
                }
            }
        }
    }
    return SaberSuccess;
}

/*wirte result in outputs
* input din: [n, c / 4, h, w * 4], output dout: [n, c, h, w]
*/
inline SaberStatus write_to_output_c4_int32(const int* din, int* dout, int ch_n, int hei_n, int cs, int ce, \
    int hs, int he, int ws, int we, int channel, int height, int width, bool flag_relu, int* trash_ptr) {
    if (ch_n != 4 || hei_n <= 0){
        LOGE("ch_n must be equal 4 and hei_n is more than zero\n");
        return SaberInvalidValue;
    }
    int size_c_out = width * height;

    int* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
    int* doutc1r0 = doutc0r0 + size_c_out;
    int* doutc2r0 = doutc1r0 + size_c_out;
    int* doutc3r0 = doutc2r0 + size_c_out;

    const int* ptr_din = din;

    int size_h = (he > height ? height : he) - hs; //size_h == hei_n

    int valid_w = we - ws;
    int cnt = valid_w / 4;

    if (we > width) {
        cnt--;
    }
    if (flag_relu){
        for (int i = 0; i < size_h; i++){
            int size_w = i * width;
            int* doutc0_ptr = doutc0r0 + size_w; //doutc0r0 + width;
            int* doutc1_ptr = doutc1r0 + size_w;
            int* doutc2_ptr = doutc2r0 + size_w;
            int* doutc3_ptr = doutc3r0 + size_w;
            if (ce > channel) {
                switch (ce - channel) {
                    case 3:
                        doutc1_ptr = trash_ptr;
                    case 2:
                        doutc2_ptr = trash_ptr;
                    case 1:
                        doutc3_ptr = trash_ptr;
                    default:
                        break;
                }
            }
            const int* din_hei_ptr = ptr_din + i * valid_w * ch_n;
            if (cnt > 0){
                int cnt_loop = cnt;
#ifdef __aarch64__
                asm volatile(
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "movi v20.4s, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "trn1   v8.4s, v0.4s, v1.4s     \n"         /* trans q0, q1*/
                    "trn2   v9.4s, v0.4s, v1.4s     \n"         /* trans q0, q1*/
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "trn1   v10.4s, v2.4s, v3.4s    \n"         /* trans q2, q3*/
                    "trn2   v11.4s, v2.4s, v3.4s    \n"         /* trans q2, q3*/
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "trn1   v16.2d, v8.2d, v10.2d   \n"         /* trans q8, q10*/
                    "trn2   v17.2d, v8.2d, v10.2d   \n"         /* trans q8, q10*/
                    "trn1   v18.2d, v9.2d, v11.2d   \n"         /* trans q9, q11*/
                    "trn2   v19.2d, v9.2d, v11.2d   \n"         /* trans q9, q11*/
                    "smax   v16.4s, v16.4s, v20.4s  \n"         /*relu*/
                    "smax   v17.4s, v17.4s, v20.4s  \n"         /*relu*/
                    "smax   v18.4s, v18.4s, v20.4s  \n"         /*relu*/
                    "smax   v19.4s, v19.4s, v20.4s  \n"         /*relu*/
                    "str    q16, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q17, [%[doutc2r0]], #16 \n"         /* store c2r0*/
                    "str    q18, [%[doutc1r0]], #16 \n"         /* store c1r0*/
                    "str    q19, [%[doutc3r0]], #16 \n"         /* store c3r0*/

                    "subs   %w[cnt], %w[cnt], #1    \n"         /* loop count -1*/
                    "bne    1b                      \n"         /* jump to main loop*/

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                [doutc3r0]"+r"(doutc3_ptr), [cnt] "+r"(cnt_loop), [ptr_din]"+r"(din_hei_ptr)
                :
                : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", \
                "v14", "v15", "v16", "v17", "v18", "v19", "v20"
                );
#else
                asm volatile(
                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"
                    "vmov.u32 q15, #0                @ dump zero\n"
                    "1:                              @ main loop\n"
                    "vtrn.32 q0, q1                  @ trans data:c00c01c20c21 \n"
                    "vtrn.32 q2, q3                  @ trans data:c02c03c22c23 \n"

                    "vmov.s32  d8, d1                  @ mov d1 to d8\n"
                    "vmov.s32  d9, d3                  @ mov d1 to d8\n"

                    "vmov.s32 d1, d4                   @ mov d1 to d8\n"
                    "vmov.s32 d3, d6                   @ mov d1 to d8\n"
                    "vmov.s32 d4, d8                   @ mov d1 to d8\n"
                    "vmov.s32 d6, d9                   @ mov d1 to d8\n"

                    "vmax.s32   q0, q0, q15        @ relu\n"
                    "vmax.s32   q1, q1, q15        @ relu\n"
                    "vmax.s32   q2, q2, q15        @ relu\n"
                    "vmax.s32   q3, q3, q15        @ relu\n"

                    "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d2-d3}, [%[doutc1r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d4-d5}, [%[doutc2r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d6-d7}, [%[doutc3r0]]!     @ store result, add pointer\n"

                    "subs   %[cnt], %[cnt], #1    @ loop count - 1\n"

                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"

                    "bne    1b                            @ jump to main loop\n"

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                [doutc3r0]"+r"(doutc3_ptr), [ptr_din]"+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
                :
                : "q0", "q1", "q2", "q3", "q4", "q15"
                );
#endif
            }
            if (we > width) {
                int offset = 16 * (valid_w / 4 - 1);
                din_hei_ptr  = ptr_din + offset;
                int i = we - 4;
                for (; i < width; ++i) {
                    *(doutc0_ptr++) = AKMAX(din_hei_ptr[0], 0.f);
                    *(doutc1_ptr++) = AKMAX(din_hei_ptr[1], 0.f);
                    *(doutc2_ptr++) = AKMAX(din_hei_ptr[2], 0.f);
                    *(doutc3_ptr++) = AKMAX(din_hei_ptr[3], 0.f);
                    din_hei_ptr += 4;
                }
            }
        }
    }else{
        for (int i = 0; i < size_h; i++){
            int size_w = i * width;
            int* doutc0_ptr = doutc0r0 + size_w; //doutc0r0 + width;
            int* doutc1_ptr = doutc1r0 + size_w;
            int* doutc2_ptr = doutc2r0 + size_w;
            int* doutc3_ptr = doutc3r0 + size_w;
            if (ce > channel) {
                switch (ce - channel) {
                    case 3:
                        doutc1_ptr = trash_ptr;
                    case 2:
                        doutc2_ptr = trash_ptr;
                    case 1:
                        doutc3_ptr = trash_ptr;
                    default:
                        break;
                }
            }
            const int* din_hei_ptr = ptr_din + i * valid_w * ch_n;
            if (cnt > 0){
                int cnt_loop = cnt;
#ifdef __aarch64__
                asm volatile(
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "movi v20.4s, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "trn1   v8.4s, v0.4s, v1.4s     \n"         /* trans q0, q1*/
                    "trn2   v9.4s, v0.4s, v1.4s     \n"         /* trans q0, q1*/
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "trn1   v10.4s, v2.4s, v3.4s    \n"         /* trans q2, q3*/
                    "trn2   v11.4s, v2.4s, v3.4s    \n"         /* trans q2, q3*/
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "trn1   v16.2d, v8.2d, v10.2d   \n"         /* trans q8, q10*/
                    "trn2   v17.2d, v8.2d, v10.2d   \n"         /* trans q8, q10*/
                    "trn1   v18.2d, v9.2d, v11.2d   \n"         /* trans q9, q11*/
                    "trn2   v19.2d, v9.2d, v11.2d   \n"         /* trans q9, q11*/

                    "str    q16, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q17, [%[doutc2r0]], #16 \n"         /* store c2r0*/
                    "str    q18, [%[doutc1r0]], #16 \n"         /* store c1r0*/
                    "str    q19, [%[doutc3r0]], #16 \n"         /* store c3r0*/

                    "subs   %w[cnt], %w[cnt], #1    \n"         /* loop count -1*/
                    "bne    1b                      \n"         /* jump to main loop*/

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                [doutc3r0]"+r"(doutc3_ptr), [cnt] "+r"(cnt_loop), [ptr_din]"+r"(din_hei_ptr)
                :
                : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", \
                "v14", "v15", "v16", "v17", "v18", "v19", "v20"
                );
#else
                asm volatile(
                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"
                    "vmov.u32 q15, #0                @ dump zero\n"
                    "1:                              @ main loop\n"
                    "vtrn.32 q0, q1                  @ trans data:c00c01c20c21 \n"
                    "vtrn.32 q2, q3                  @ trans data:c02c03c22c23 \n"

                    "vmov.s32  d8, d1                  @ mov d1 to d8\n"
                    "vmov.s32  d9, d3                  @ mov d1 to d8\n"

                    "vmov.s32 d1, d4                   @ mov d1 to d8\n"
                    "vmov.s32 d3, d6                   @ mov d1 to d8\n"
                    "vmov.s32 d4, d8                   @ mov d1 to d8\n"
                    "vmov.s32 d6, d9                   @ mov d1 to d8\n"

                    "subs   %[cnt], %[cnt], #1    @ loop count - 1\n"
                    "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d2-d3}, [%[doutc1r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d4-d5}, [%[doutc2r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d6-d7}, [%[doutc3r0]]!     @ store result, add pointer\n"

                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"

                    "bne    1b                            @ jump to main loop\n"

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                [doutc3r0]"+r"(doutc3_ptr), [ptr_din]"+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
                :
                : "q0", "q1", "q2", "q3", "q4", "q15"
                );
#endif
            }
            if (we > width) {
                int offset = 16 * (valid_w / 4 - 1);
                din_hei_ptr  = ptr_din + offset;
                int i = we - 4;
                for (; i < width; ++i) {
                    *(doutc0_ptr++) = din_hei_ptr[0];
                    *(doutc1_ptr++) = din_hei_ptr[1];
                    *(doutc2_ptr++) = din_hei_ptr[2];
                    *(doutc3_ptr++) = din_hei_ptr[3];
                    din_hei_ptr += 4;
                }
            }
        }
    }
    return SaberSuccess;
}

/*wirte result in outputs
* input din: [n, c / 8, h, w * 8], output dout: [n, c, h, w]
*/
inline SaberStatus write_to_output_c8_int32(const int* din, int* dout, int ch_n, int hei_n, int cs, int ce, \
    int hs, int he, int ws, int we, int channel, int height, int width, bool flag_relu, int* trash_ptr) {
    if (ch_n != 8 || hei_n <= 0){
        LOGE("ch_n must be equal 8 and hei_n is more than zero\n");
        return SaberInvalidValue;
    }
    int size_c_out = width * height;

    int* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
    int* doutc1r0 = doutc0r0 + size_c_out;
    int* doutc2r0 = doutc1r0 + size_c_out;
    int* doutc3r0 = doutc2r0 + size_c_out;
    int* doutc4r0 = doutc3r0 + size_c_out;
    int* doutc5r0 = doutc4r0 + size_c_out;
    int* doutc6r0 = doutc5r0 + size_c_out;
    int* doutc7r0 = doutc6r0 + size_c_out;

    const int* ptr_din = din;

    int size_h = (he > height ? height : he) - hs; //size_h == hei_n

    int valid_w = we - ws;
    int cnt = valid_w / 4;

    if (we > width) {
        cnt--;
    }
    if (flag_relu){
        for (int i = 0; i < size_h; i++){
            int size_w = i * width;
            int* doutc0_ptr = doutc0r0 + size_w; //doutc0r0 + width;
            int* doutc1_ptr = doutc1r0 + size_w;
            int* doutc2_ptr = doutc2r0 + size_w;
            int* doutc3_ptr = doutc3r0 + size_w;
            int* doutc4_ptr = doutc4r0 + size_w; //doutc0r0 + width;
            int* doutc5_ptr = doutc5r0 + size_w;
            int* doutc6_ptr = doutc6r0 + size_w;
            int* doutc7_ptr = doutc7r0 + size_w;
            if (ce > channel) {
                switch (ce - channel) {
                    case 7:
                        doutc1_ptr = trash_ptr;
                    case 6:
                        doutc2_ptr = trash_ptr;
                    case 5:
                        doutc3_ptr = trash_ptr;
                    case 4:
                        doutc4_ptr = trash_ptr;
                    case 3:
                        doutc5_ptr = trash_ptr;
                    case 2:
                        doutc6_ptr = trash_ptr;
                    case 1:
                        doutc7_ptr = trash_ptr;
                    default:
                        break;
                }
            }

            const int* din_hei_ptr = ptr_din + i * valid_w * ch_n;
            if (cnt > 0){
                int cnt_loop = cnt;
#ifdef __aarch64__
                asm volatile(
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "ldp q4, q5, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q6, q7, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "movi v20.4s, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "trn1   v8.4s, v0.4s, v2.4s     \n"         /* trans q0, q1*/
                    "trn2   v9.4s, v0.4s, v2.4s     \n"         /* trans q0, q1*/
                    "trn1   v10.4s, v1.4s, v3.4s    \n"         /* trans q2, q3*/
                    "trn2   v11.4s, v1.4s, v3.4s    \n"         /* trans q2, q3*/
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */

                    "trn1   v12.4s, v4.4s, v6.4s     \n"         /* trans q0, q1*/
                    "trn2   v13.4s, v4.4s, v6.4s     \n"         /* trans q0, q1*/
                    "trn1   v14.4s, v5.4s, v7.4s    \n"         /* trans q2, q3*/
                    "trn2   v15.4s, v5.4s, v7.4s    \n"         /* trans q2, q3*/
                    "ldp q4, q5, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q6, q7, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */

                    "trn1   v16.2d, v8.2d, v12.2d   \n"         /* trans q8, q10 00 01 02 03*/
                    "trn2   v17.2d, v8.2d, v12.2d   \n"         /* trans q8, q10 20 21 22 23*/
                    "trn1   v18.2d, v9.2d, v13.2d   \n"         /* trans q9, q11 10 11 12 13*/
                    "trn2   v19.2d, v9.2d, v13.2d   \n"         /* trans q9, q11 30 31 32 33*/

                    "trn1   v8.2d, v10.2d, v14.2d   \n"         /* trans q8, q10 40 41 42 43*/
                    "trn2   v9.2d, v10.2d, v14.2d   \n"         /* trans q8, q10 60 61 62 63*/
                    "trn1   v12.2d, v11.2d, v15.2d   \n"         /* trans q9, q11 50 51 52 53*/
                    "trn2   v13.2d, v11.2d, v15.2d   \n"         /* trans q9, q11 70 71 72 73*/

                    "smax   v16.4s, v16.4s, v20.4s  \n"         /*relu*/
                    "smax   v17.4s, v17.4s, v20.4s  \n"         /*relu*/
                    "smax   v18.4s, v18.4s, v20.4s  \n"         /*relu*/
                    "smax   v19.4s, v19.4s, v20.4s  \n"         /*relu*/

                    "smax   v8.4s, v8.4s, v20.4s  \n"         /*relu*/
                    "smax   v9.4s, v9.4s, v20.4s  \n"         /*relu*/
                    "smax   v12.4s, v12.4s, v20.4s  \n"         /*relu*/
                    "smax   v13.4s, v13.4s, v20.4s  \n"         /*relu*/

                    "str    q16, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q17, [%[doutc2r0]], #16 \n"         /* store c2r0*/
                    "str    q18, [%[doutc1r0]], #16 \n"         /* store c1r0*/
                    "str    q19, [%[doutc3r0]], #16 \n"         /* store c3r0*/

                    "subs   %w[cnt], %w[cnt], #1    \n"         /* loop count -1*/
                    "str    q8, [%[doutc4r0]], #16 \n"         /* store c0r0*/
                    "str    q9, [%[doutc6r0]], #16 \n"         /* store c2r0*/
                    "str    q12, [%[doutc5r0]], #16 \n"         /* store c1r0*/
                    "str    q13, [%[doutc7r0]], #16 \n"         /* store c3r0*/

                    "bne    1b                      \n"         /* jump to main loop*/

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                  [doutc3r0]"+r"(doutc3_ptr), [doutc4r0]"+r"(doutc4_ptr), [doutc5r0]"+r"(doutc5_ptr), \
                  [doutc6r0]"+r"(doutc6_ptr), [doutc7r0]"+r"(doutc7_ptr),\
                  [cnt] "+r"(cnt_loop), [ptr_din]"+r"(din_hei_ptr)
                :
                : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", \
                "v14", "v15", "v16", "v17", "v18", "v19", "v20"
                );
#else
                asm volatile(
                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d8-d11}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d12-d15}, [%[ptr_din]]!        @load data \n"
                    "vmov.u32 q15, #0                @ dump zero\n"
                    "1:                              @ main loop\n"
                    "vtrn.32 q0, q2                  @ trans data:c00c01c20c21 c10c11c30c31\n"
                    "vtrn.32 q1, q3                  @ trans data:c40c41c60c61 c50c51c70c7n"
                    "vtrn.32 q4, q6                  @ trans data:c02c03c22c23 c12c13c32c33\n"
                    "vtrn.32 q5, q7                  @ trans data:c42c43c62c63 c52c53c72c73\n"

                    "vmov.s32  d16, d1                  @ mov d1 to d8\n"
                    "vmov.s32  d17, d3                  @ mov d1 to d8\n"
                    "vmov.s32  d18, d5                  @ mov d1 to d8\n"
                    "vmov.s32  d19, d7                  @ mov d1 to d8\n"

                    "vmov.s32 d1, d8                   @ mov d1 to d8 c00c01c02c03\n"
                    "vmov.s32 d3, d10                   @ mov d1 to d8 c40c41c42c43\n"
                    "vmov.s32 d5, d12                   @ mov d1 to d8 c10c11c12c13\n"
                    "vmov.s32 d7, d14                   @ mov d1 to d8 c50c51c52c53\n"

                    "vmov.s32 d8, d1                   @ mov d1 to d8 c20c21c22c23\n"
                    "vmov.s32 d10, d3                   @ mov d1 to d8 c60c61c62c63 n"
                    "vmov.s32 d12, d5                   @ mov d1 to d8 c10c11c12c13\n"
                    "vmov.s32 d14, d7                   @ mov d1 to d8 d70d71d72d73\n"

                    "vmax.s32   q0, q0, q15        @ relu\n"
                    "vmax.s32   q1, q1, q15        @ relu\n"
                    "vmax.s32   q2, q2, q15        @ relu\n"
                    "vmax.s32   q3, q3, q15        @ relu\n"

                    "vmax.s32   q4, q4, q15        @ relu\n"
                    "vmax.s32   q5, q5, q15        @ relu\n"
                    "vmax.s32   q6, q6, q15        @ relu\n"
                    "vmax.s32   q7, q7, q15        @ relu\n"

                    "subs   %[cnt], %[cnt], #1    @ loop count - 1\n"
                    "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d2-d3}, [%[doutc4r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d4-d5}, [%[doutc1r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d6-d7}, [%[doutc5r0]]!     @ store result, add pointer\n"

                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"

                    "vst1.32  {d8-d9}, [%[doutc2r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d10-d11}, [%[doutc6r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d12-d13}, [%[doutc1r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d14-d15}, [%[doutc7r0]]!     @ store result, add pointer\n"

                    "vld1.32 {d8-d11}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d12-d15}, [%[ptr_din]]!        @load data \n"

                    "bne    1b                            @ jump to main loop\n"

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                  [doutc3r0]"+r"(doutc3_ptr), [doutc4r0]"+r"(doutc4_ptr), [doutc5r0]"+r"(doutc5_ptr), \
                  [doutc6r0]"+r"(doutc6_ptr), [doutc7r0]"+r"(doutc7_ptr),\
                  [ptr_din]"+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
                :
                : "q0", "q1", "q2", "q3", "q4", "q15"
                );
#endif
            }
            if (we > width) {
                int offset = 32 * (valid_w / 4 - 1);
                din_hei_ptr  = ptr_din + offset;
                int i = we - 4;
                for (; i < width; ++i) {
                    *(doutc0_ptr++) = AKMAX(din_hei_ptr[0], 0.f);
                    *(doutc1_ptr++) = AKMAX(din_hei_ptr[1], 0.f);
                    *(doutc2_ptr++) = AKMAX(din_hei_ptr[2], 0.f);
                    *(doutc3_ptr++) = AKMAX(din_hei_ptr[3], 0.f);
                    *(doutc4_ptr++) = AKMAX(din_hei_ptr[4], 0.f);
                    *(doutc5_ptr++) = AKMAX(din_hei_ptr[5], 0.f);
                    *(doutc6_ptr++) = AKMAX(din_hei_ptr[6], 0.f);
                    *(doutc7_ptr++) = AKMAX(din_hei_ptr[7], 0.f);
                    din_hei_ptr += 8;
                }
            }
        }
    }else{
        for (int i = 0; i < size_h; i++){
            int size_w = i * width;
            int* doutc0_ptr = doutc0r0 + size_w; //doutc0r0 + width;
            int* doutc1_ptr = doutc1r0 + size_w;
            int* doutc2_ptr = doutc2r0 + size_w;
            int* doutc3_ptr = doutc3r0 + size_w;
            int* doutc4_ptr = doutc4r0 + size_w; //doutc0r0 + width;
            int* doutc5_ptr = doutc5r0 + size_w;
            int* doutc6_ptr = doutc6r0 + size_w;
            int* doutc7_ptr = doutc7r0 + size_w;
            if (ce > channel) {
                switch (ce - channel) {
                    case 7:
                        doutc1_ptr = trash_ptr;
                    case 6:
                        doutc2_ptr = trash_ptr;
                    case 5:
                        doutc3_ptr = trash_ptr;
                    case 4:
                        doutc4_ptr = trash_ptr;
                    case 3:
                        doutc5_ptr = trash_ptr;
                    case 2:
                        doutc6_ptr = trash_ptr;
                    case 1:
                        doutc7_ptr = trash_ptr;
                    default:
                        break;
                }
            }
            const int* din_hei_ptr = ptr_din + i * valid_w * ch_n;
            // printf("din_hei_ptr: %x, ptr_din: %x, cnt: %d \n", din_hei_ptr, ptr_din, cnt);
            if (cnt > 0){
                int cnt_loop = cnt;
#ifdef __aarch64__
                asm volatile(
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "ldp q4, q5, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "ldp q6, q7, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                    "movi v20.4s, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "trn1   v8.4s, v0.4s, v2.4s     \n"         /* trans q0, q1*/
                    "trn2   v9.4s, v0.4s, v2.4s     \n"         /* trans q0, q1*/
                    "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "trn1   v10.4s, v1.4s, v3.4s    \n"         /* trans q2, q3*/
                    "trn2   v11.4s, v1.4s, v3.4s    \n"         /* trans q2, q3*/
                    "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */

                    "trn1   v12.4s, v4.4s, v6.4s     \n"         /* trans q0, q1*/
                    "trn2   v13.4s, v4.4s, v6.4s     \n"         /* trans q0, q1*/
                    "ldp q4, q5, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                    "trn1   v14.4s, v5.4s, v7.4s    \n"         /* trans q2, q3*/
                    "trn2   v15.4s, v5.4s, v7.4s    \n"         /* trans q2, q3*/
                    "ldp q6, q7, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */

                    "trn1   v16.2d, v8.2d, v12.2d   \n"         /* trans q8, q10 00 01 02 03*/
                    "trn2   v17.2d, v8.2d, v12.2d   \n"         /* trans q8, q10 20 21 22 23*/
                    "trn1   v18.2d, v9.2d, v13.2d   \n"         /* trans q9, q11 10 11 12 13*/
                    "trn2   v19.2d, v9.2d, v13.2d   \n"         /* trans q9, q11 30 31 32 33*/

                    "trn1   v8.2d, v10.2d, v14.2d   \n"         /* trans q8, q10 40 41 42 43*/
                    "trn2   v9.2d, v10.2d, v14.2d   \n"         /* trans q8, q10 60 61 62 63*/
                    "trn1   v12.2d, v11.2d, v15.2d   \n"         /* trans q9, q11 50 51 52 53*/
                    "trn2   v13.2d, v11.2d, v15.2d   \n"         /* trans q9, q11 70 71 72 73*/

                    "str    q16, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q17, [%[doutc2r0]], #16 \n"         /* store c2r0*/
                    "str    q18, [%[doutc1r0]], #16 \n"         /* store c1r0*/
                    "str    q19, [%[doutc3r0]], #16 \n"         /* store c3r0*/

                    "subs   %w[cnt], %w[cnt], #1    \n"         /* loop count -1*/
                    "str    q8, [%[doutc4r0]], #16 \n"         /* store c0r0*/
                    "str    q9, [%[doutc6r0]], #16 \n"         /* store c2r0*/
                    "str    q12, [%[doutc5r0]], #16 \n"         /* store c1r0*/
                    "str    q13, [%[doutc7r0]], #16 \n"         /* store c3r0*/

                    "bne    1b                      \n"         /* jump to main loop*/

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                  [doutc3r0]"+r"(doutc3_ptr), [doutc4r0]"+r"(doutc4_ptr), [doutc5r0]"+r"(doutc5_ptr), \
                  [doutc6r0]"+r"(doutc6_ptr), [doutc7r0]"+r"(doutc7_ptr),\
                  [cnt] "+r"(cnt_loop), [ptr_din]"+r"(din_hei_ptr)
                :
                : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", \
                "v14", "v15", "v16", "v17", "v18", "v19", "v20"
                );
#else
                asm volatile(
                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d8-d11}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d12-d15}, [%[ptr_din]]!        @load data \n"
                    "vmov.u32 q15, #0                @ dump zero\n"
                    "1:                              @ main loop\n"
                    "vtrn.32 q0, q2                  @ trans data:c00c01c20c21 c10c11c30c31\n"
                    "vtrn.32 q1, q3                  @ trans data:c40c41c60c61 c50c51c70c7n"
                    "vtrn.32 q4, q6                  @ trans data:c02c03c22c23 c12c13c32c33\n"
                    "vtrn.32 q5, q7                  @ trans data:c42c43c62c63 c52c53c72c73\n"

                    "vmov.s32  d16, d1                  @ mov d1 to d8\n"
                    "vmov.s32  d17, d3                  @ mov d1 to d8\n"
                    "vmov.s32  d18, d5                  @ mov d1 to d8\n"
                    "vmov.s32  d19, d7                  @ mov d1 to d8\n"

                    "vmov.s32 d1, d8                   @ mov d1 to d8 c00c01c02c03\n"
                    "vmov.s32 d3, d10                   @ mov d1 to d8 c40c41c42c43\n"
                    "vmov.s32 d5, d12                   @ mov d1 to d8 c10c11c12c13\n"
                    "vmov.s32 d7, d14                   @ mov d1 to d8 c50c51c52c53\n"

                    "vmov.s32 d8, d1                   @ mov d1 to d8 c20c21c22c23\n"
                    "vmov.s32 d10, d3                   @ mov d1 to d8 c60c61c62c63 n"
                    "vmov.s32 d12, d5                   @ mov d1 to d8 c10c11c12c13\n"
                    "vmov.s32 d14, d7                   @ mov d1 to d8 d70d71d72d73\n"

                    "subs   %[cnt], %[cnt], #1    @ loop count - 1\n"
                    "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d2-d3}, [%[doutc4r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d4-d5}, [%[doutc1r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d6-d7}, [%[doutc5r0]]!     @ store result, add pointer\n"

                    "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"

                    "vst1.32  {d8-d9}, [%[doutc2r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d10-d11}, [%[doutc6r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d12-d13}, [%[doutc1r0]]!     @ store result, add pointer\n"
                    "vst1.32  {d14-d15}, [%[doutc7r0]]!     @ store result, add pointer\n"

                    "vld1.32 {d8-d11}, [%[ptr_din]]!        @load data \n"
                    "vld1.32 {d12-d15}, [%[ptr_din]]!        @load data \n"

                    "bne    1b                            @ jump to main loop\n"

                : [doutc0r0]"+r"(doutc0_ptr), [doutc1r0]"+r"(doutc1_ptr), [doutc2r0]"+r"(doutc2_ptr), \
                  [doutc3r0]"+r"(doutc3_ptr), [doutc4r0]"+r"(doutc4_ptr), [doutc5r0]"+r"(doutc5_ptr), \
                  [doutc6r0]"+r"(doutc6_ptr), [doutc7r0]"+r"(doutc7_ptr),\
                  [ptr_din]"+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
                :
                : "q0", "q1", "q2", "q3", "q4", "q15"
                );
#endif
            }
            if (we > width) {
                int offset = 32 * (valid_w / 4 - 1);
                din_hei_ptr  = ptr_din + offset;
                int i = we - 4;
                for (; i < width; ++i) {
                    *(doutc0_ptr++) = din_hei_ptr[0];
                    *(doutc1_ptr++) = din_hei_ptr[1];
                    *(doutc2_ptr++) = din_hei_ptr[2];
                    *(doutc3_ptr++) = din_hei_ptr[3];
                    *(doutc4_ptr++) = din_hei_ptr[4];
                    *(doutc5_ptr++) = din_hei_ptr[5];
                    *(doutc6_ptr++) = din_hei_ptr[6];
                    *(doutc7_ptr++) = din_hei_ptr[7];
                    din_hei_ptr += 8;
                }
            }
        }
    }
    return SaberSuccess;
}

/*
* din [n, hei_n, ch_n, w]
* dout [n, ch_n, hei_n, w]
*/
template <typename dtype>
static SaberStatus write_to_output_numc(const dtype* din, dtype* dout, int ch_n, int hei_n, int cs, int ce, \
    int hs, int he, int ws, int we, int channel, int height, int width, bool flag_relu, dtype* trash_ptr) {
    if (ch_n <= 0 || hei_n <= 0){
        LOGE("ch_n and hei_n are more than zero\n");
        return SaberInvalidValue;
    }
    int size_c_out = width * height;

    dtype *out_array[ch_n];
    out_array[0] = dout + cs * size_c_out + hs * width + ws;

    for (int i = 1; i < ch_n; i++){
        out_array[i] = out_array[i - 1] + size_c_out;
    }

    const dtype* ptr_din = din;

    int cremain = ce - channel;
    for (int i = 1; i <= cremain; i++){
        out_array[ch_n - i] = trash_ptr;
    }

    int size_h = (he > height ? height : he) - hs; //size_h == hei_n

    int size_w = we - ws;

    int size_c_in = ch_n * size_w;

    size_t valid_w_byte = width * sizeof(dtype);

    if (flag_relu){
        for (int h = 0; h < size_h; h++){
            dtype* din_ptr = din + h * size_c_in;
            for (int i = 0; i < ch_n; i++){
                dtype* dout_ptr = out_array[i] + h * width;
                for (int k = 0; k < width; k++){
                    *(dout_ptr++) = AKMAX(din_ptr[k], 0);
                }
                din_ptr += size_w;
            }
        }
    }else{
        for (int h = 0; h < size_h; h++){
            dtype* din_ptr = din + h * size_c_in;
            for (int i = 0; i < ch_n; i++){
                dtype* dout_ptr = out_array[i] + h * width;
                memcpy(dout_ptr, din_ptr, valid_w_byte);
                din_ptr += size_w;
            }
        }
    }
    return SaberSuccess;
}
/**
* innput din: nchwc(num)
*/
inline void fill_packed_bias_nxmw_fp32(const float* bias, float* dout, int ch_n, int hei_n, int wround) {
    if (ch_n <= 0 || hei_n <= 0){
        LOGE("ch_n and hei_n are more than zero\n");
        return SaberInvalidValue;
    }
    int cnt_ch = ch_n / 4;
    int size = wround * ch_n;
    for (int h = 0; h < hei_n; h++){
        float* dout_ptr = dout + h * size;
        for (int i = 0; i < wround; i++){
            const float* bias_ptr = bias;
            int j = 0;
            for (; j < cnt_ch; j++){
                float32x4_t vb = vld1q_f32(bias_ptr);
                bias_ptr += 4;

                vst1q_f32(dout_ptr, vb);
                dout_ptr += 4;
            }
            j = j * 4;
            for (;j < ch_n; j++){
                *dout_ptr = *bias_ptr;
                dout_ptr++;
                bias_ptr++;
            }
        }
    }
}

inline void fill_packed_bias_nxmw_int8(const int* bias, int* dout, int ch_n, int hei_n, int wround) {

    if (ch_n <= 0 || hei_n <= 0){
        LOGE("ch_n and hei_n are more than zero\n");
        return SaberInvalidValue;
    }
    int cnt_ch = ch_n / 4;
    int size = wround * ch_n;
    for (int h = 0; h < hei_n; h++){
        int* dout_ptr = dout + h * size;
        for (int i = 0; i < wround; i++){
            const int* bias_ptr = bias;
            int j = 0;
            for (; j < cnt_ch; j++){
                int32x4_t vb = vld1q_s32(bias_ptr);
                bias_ptr += 4;

                vst1q_s32(dout_ptr, vb);
                dout_ptr += 4;
            }
            j = j * 4;
            for (;j < ch_n; j++){
                *dout_ptr = *bias_ptr;
                dout_ptr++;
                bias_ptr++;
            }
        }
    }
    return SaberSuccess;
}
}//namespace lite
}//namespace saber
}//namespace anakin
#endif
#endif
