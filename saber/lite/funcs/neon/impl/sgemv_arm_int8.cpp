#include "saber/lite/funcs/neon/impl/sgemv_arm_int8.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

void sgemv_int8(const bool transA, const int M, const int N, \
    const signed char* A, const signed char* x, int* y) {
    int* data_out = y;
    const signed char* data_in = x;
    const signed char* weights_ptr = A;

#ifdef __aarch64__
    int cnt_loop = N >> 4;
    int tail = N & 15;
    int out_cnt = M >> 3;

    unsigned char imask[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    uint8x8_t vmask1 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask));
    uint8x8_t vmask2 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask + 8));

    int8x8_t vzero = vdup_n_s8(0);
    // printf("tail: %d, cnt_loop: %d, out_cnt: %d \n", tail, cnt_loop, out_cnt);

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {

        int out_idx = j * 8;
        int *ptr_out = data_out + out_idx;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * out_idx);
        const signed char *ptr_w1 = ptr_w0 + N;
        const signed char *ptr_w2 = ptr_w1 + N;
        const signed char *ptr_w3 = ptr_w2 + N;
        const signed char *ptr_w4 = ptr_w3 + N;
        const signed char *ptr_w5 = ptr_w4 + N;
        const signed char *ptr_w6 = ptr_w5 + N;
        const signed char *ptr_w7 = ptr_w6 + N;

         asm volatile(
            "prfm  pldl1keep, [%[in]]   \n"
                    "prfm  pldl1keep, [%[w0]]   \n"
                    "prfm  pldl1keep, [%[w1]]   \n"
                    "prfm  pldl1keep, [%[w2]]   \n"
                    "prfm  pldl1keep, [%[w3]]   \n"
                    "prfm  pldl1keep, [%[w4]]   \n"
                    "prfm  pldl1keep, [%[w5]]   \n"
                    "prfm  pldl1keep, [%[w6]]   \n"
                    "prfm  pldl1keep, [%[w7]]   \n"
                     :
        :[in] "r"(ptr_in), [w0] "r"(ptr_w0), [w1] "r"(ptr_w1), [w2] "r"(ptr_w2), \
                [w3] "r"(ptr_w3), [w4] "r"(ptr_w4), [w5] "r"(ptr_w5), [w6] "r"(ptr_w6), \
                [w7] "r" (ptr_w7)
        :"memory"
        );
        int32x4_t sum0 = vdupq_n_s32(0);
        int32x4_t sum1 = vdupq_n_s32(0);
        int32x4_t sum2 = vdupq_n_s32(0);
        int32x4_t sum3 = vdupq_n_s32(0);
        int32x4_t sum4 = vdupq_n_s32(0);
        int32x4_t sum5 = vdupq_n_s32(0);
        int32x4_t sum6 = vdupq_n_s32(0);
        int32x4_t sum7 = vdupq_n_s32(0);

        for (int i = 0; i < cnt_loop; i++){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);
            int8x8_t w1_0 = vld1_s8(ptr_w1);
            int8x8_t w2_0 = vld1_s8(ptr_w2);
            int8x8_t w3_0 = vld1_s8(ptr_w3);
            int8x8_t w4_0 = vld1_s8(ptr_w4);
            int8x8_t w5_0 = vld1_s8(ptr_w5);
            int8x8_t w6_0 = vld1_s8(ptr_w6);
            int8x8_t w7_0 = vld1_s8(ptr_w7);

            int16x8_t sum0_16 = vmull_s8(din0, w0_0);
            int16x8_t sum1_16 = vmull_s8(din0, w1_0);
            int16x8_t sum2_16 = vmull_s8(din0, w2_0);
            int16x8_t sum3_16 = vmull_s8(din0, w3_0);
            int16x8_t sum4_16 = vmull_s8(din0, w4_0);
            int16x8_t sum5_16 = vmull_s8(din0, w5_0);
            int16x8_t sum6_16 = vmull_s8(din0, w6_0);
            int16x8_t sum7_16 = vmull_s8(din0, w7_0);

            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);
            int8x8_t w1_1 = vld1_s8(ptr_w1 + 8);
            int8x8_t w2_1 = vld1_s8(ptr_w2 + 8);
            int8x8_t w3_1 = vld1_s8(ptr_w3 + 8);
            int8x8_t w4_1 = vld1_s8(ptr_w4 + 8);
            int8x8_t w5_1 = vld1_s8(ptr_w5 + 8);
            int8x8_t w6_1 = vld1_s8(ptr_w6 + 8);
            int8x8_t w7_1 = vld1_s8(ptr_w7 + 8);

            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum1_16 = vmlal_s8(sum1_16, din1, w1_1);
            sum2_16 = vmlal_s8(sum2_16, din1, w2_1);
            sum3_16 = vmlal_s8(sum3_16, din1, w3_1);
            sum4_16 = vmlal_s8(sum4_16, din1, w4_1);
            sum5_16 = vmlal_s8(sum5_16, din1, w5_1);
            sum6_16 = vmlal_s8(sum6_16, din1, w6_1);
            sum7_16 = vmlal_s8(sum7_16, din1, w7_1);

            ptr_in += 16;
            ptr_w0 += 16;
            ptr_w1 += 16;
            ptr_w2 += 16;
            ptr_w3 += 16;
            ptr_w4 += 16;
            ptr_w5 += 16;
            ptr_w6 += 16;
            ptr_w7 += 16;

            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_low_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_low_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_low_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_low_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_low_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_low_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_low_s16(sum7_16));

            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_high_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_high_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_high_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_high_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_high_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_high_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_high_s16(sum7_16));

        }
        if (tail > 0){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);
            int8x8_t w1_0 = vld1_s8(ptr_w1);
            int8x8_t w2_0 = vld1_s8(ptr_w2);
            int8x8_t w3_0 = vld1_s8(ptr_w3);

            din0 = vbsl_s8(vmask1, din0, vzero);

            int8x8_t w4_0 = vld1_s8(ptr_w4);
            int8x8_t w5_0 = vld1_s8(ptr_w5);
            int8x8_t w6_0 = vld1_s8(ptr_w6);
            int8x8_t w7_0 = vld1_s8(ptr_w7);

            int16x8_t sum0_16 = vmull_s8(din0, w0_0);
            int16x8_t sum1_16 = vmull_s8(din0, w1_0);
            int16x8_t sum2_16 = vmull_s8(din0, w2_0);
            int16x8_t sum3_16 = vmull_s8(din0, w3_0);
            int16x8_t sum4_16 = vmull_s8(din0, w4_0);
            int16x8_t sum5_16 = vmull_s8(din0, w5_0);
            int16x8_t sum6_16 = vmull_s8(din0, w6_0);
            int16x8_t sum7_16 = vmull_s8(din0, w7_0);

            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);
            int8x8_t w1_1 = vld1_s8(ptr_w1 + 8);
            int8x8_t w2_1 = vld1_s8(ptr_w2 + 8);
            int8x8_t w3_1 = vld1_s8(ptr_w3 + 8);

            din1 = vbsl_s8(vmask2, din1, vzero);

            int8x8_t w4_1 = vld1_s8(ptr_w4 + 8);
            int8x8_t w5_1 = vld1_s8(ptr_w5 + 8);
            int8x8_t w6_1 = vld1_s8(ptr_w6 + 8);
            int8x8_t w7_1 = vld1_s8(ptr_w7 + 8);

            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum1_16 = vmlal_s8(sum1_16, din1, w1_1);
            sum2_16 = vmlal_s8(sum2_16, din1, w2_1);
            sum3_16 = vmlal_s8(sum3_16, din1, w3_1);
            sum4_16 = vmlal_s8(sum4_16, din1, w4_1);
            sum5_16 = vmlal_s8(sum5_16, din1, w5_1);
            sum6_16 = vmlal_s8(sum6_16, din1, w6_1);
            sum7_16 = vmlal_s8(sum7_16, din1, w7_1);

            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_low_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_low_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_low_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_low_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_low_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_low_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_low_s16(sum7_16));

            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_high_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_high_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_high_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_high_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_high_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_high_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_high_s16(sum7_16));
        }
        //add
        int32x4_t vout0 = vpaddq_s32(sum0, sum1);
        int32x4_t vout1 = vpaddq_s32(sum2, sum3);
        int32x4_t vout2 = vpaddq_s32(sum4, sum5);
        int32x4_t vout3 = vpaddq_s32(sum6, sum7);
        int32x4_t vdout = vpaddq_s32(vout0, vout1);
        int32x4_t vdout1 = vpaddq_s32(vout2, vout3);
        vst1q_s32(ptr_out, vdout);
        vst1q_s32(ptr_out + 4, vdout1);
    }
    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 8; j < M; ++j){
        int *ptr_out = data_out + j;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;

         asm volatile(
            "prfm  pldl1keep, [%[in]]   \n"
                    "prfm  pldl1keep, [%[w0]]   \n"
                     :
        :[in] "r"(ptr_in), [w0] "r"(ptr_w0)
        :"memory"
        );

        int32x4_t sum0 = vdupq_n_s32(0);
        for (int i = 0; i < cnt_loop; i++){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);//a = 01234567

            int16x8_t sum0_16 = vmull_s8(din0, w0_0);

            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);//a = 01234567

            ptr_in += 16;
            ptr_w0 += 16;

            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
        }
        if (tail > 0){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);//a = 01234567

            din0 = vbsl_s8(vmask1, din0, vzero);
            din1 = vbsl_s8(vmask2, din1, vzero);
            int16x8_t sum0_16 = vmull_s8(din0, w0_0);
            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
        }
        //add
        int32x2_t vout = vpadd_s32(vget_low_s32(sum0), vget_high_s32(sum0));
        int32x2_t vdout = vpadd_s32(vout, vout);
        *ptr_out = vget_lane_s32(vdout, 0);
    }
#else
    int cnt_loop = N >> 4;
    int tail = N & 15;
    int out_cnt = M >> 2;

    unsigned char imask[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    uint8x8_t vmask1 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask));
    uint8x8_t vmask2 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask + 8));

    // printf("cnt_loop: %d, tail: %d, out_cnt: %d \n", cnt_loop, tail, out_cnt);

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {
        int out_idx = j * 4;
        int *ptr_out = data_out + out_idx;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * out_idx);
        const signed char *ptr_w1 = ptr_w0 + N;
        const signed char *ptr_w2 = ptr_w1 + N;
        const signed char *ptr_w3 = ptr_w2 + N;

        int cnt = cnt_loop;
        asm volatile(
        "pld [%[in]] @ preload cache line, input\n"
                "pld [%[w0]]                    @ preload cache line, weights r0\n"
                "pld [%[w1]]                    @ preload cache line, weights r1\n"
                "pld [%[w2]]                    @ preload cache line, weights r2\n"
                "pld [%[w3]]                    @ preload cache line, weights r3\n"

                "vmov.u32 q0, #0                @ set q0 to 0\n"
                "vmov.u32 q1, #0                @ set q1 to 0\n"
                "vmov.u32 q2, #0                @ set q2 to 0\n"
                "vmov.u32 q3, #0                @ set q3 to 0\n"

                "cmp %[cnt], #1                 @ check whether has main loop\n"
                "blt  3f                        @ jump to tail\n"

                "1:                             @ main loop\n"
                "vld1.8 {d8-d9}, [%[in]]!    @ load input, q4, q5\n"
                "vld1.8 {d12-d13}, [%[w0]]!    @ load weights r0, q6\n"
                "vld1.8 {d14-d15}, [%[w1]]!    @ load weights r1, q7\n"
                "vld1.8 {d16-d17}, [%[w2]]!    @ load weights r2, q8\n"
                "vld1.8 {d18-d19}, [%[w3]]!    @ load weights r3, q9\n"

                "vmull.s8 q12, d8, d12            @ mul add\n"
                "vmull.s8 q13, d8, d14            @ mul add\n"
                "vmull.s8 q14, d8, d16            @ mul add\n"
                "vmull.s8 q15, d8, d18            @ mul add\n"

                "pld [%[in]]                    @ preload cache line, input\n"
                "pld [%[w0]]                    @ preload cache line, weights r0\n"

                "vmlal.s8 q12, d9, d13            @ mul add\n"
                "vmlal.s8 q13, d9, d15            @ mul add\n"
                "vmlal.s8 q14, d9, d17           @ mul add\n"
                "vmlal.s8 q15, d9, d19           @ mul add\n"

                "pld [%[w1]]                    @ preload cache line, weights r1\n"
                "pld [%[w2]]                    @ preload cache line, weights r2\n"
                "pld [%[w3]]                    @ preload cache line, weights r3\n"

                "vaddw.s16 q0, q0, d24                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d26                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d28                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d30                 @addw \n" // out1 += vget_low_s16(out10)

                "vaddw.s16 q0, q0, d25                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d27                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d29                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d31                 @addw \n" // out1 += vget_low_s16(out10)

                // check loop end
                "subs %[cnt], #1                @ sub loop count \n"

                "bne 1b                         @ jump to main loop\n"

                // check tails
                "3:                             @ check tail\n"
                "cmp %[tail], #1                @ check whether has mid cols\n"
                "blt  2f                        @ jump to end\n"

                // process tail
                "vld1.8 {d8-d9}, [%[in]]!    @ load input, q4, q5\n"
                // deal with right pad
                "vmov.u32 d10, #0                @ dump q8 to zero, for bit select in tail\n"
                "vld1.8 {d12-d13}, [%[w0]]!    @ load weights r0, q6\n"
                "vld1.8 {d14-d15}, [%[w1]]!    @ load weights r1, q7\n"
                "vld1.8 {d16-d17}, [%[w2]]!    @ load weights r2, q8\n"
                "vld1.8 {d18-d19}, [%[w3]]!    @ load weights r3, q9\n"

                "vbif d8, d10, %[mask1]         @ bit select, deal with right pad\n"
                "vbif d9, d10, %[mask2]         @ bit select, deal with right pad\n"
                "pld [%[in]]                    @ preload cache line, input\n"

                "vmull.s8 q12, d8, d12            @ mul add\n"
                "vmull.s8 q13, d8, d14            @ mul add\n"
                "vmull.s8 q14, d8, d16            @ mul add\n"
                "vmull.s8 q15, d8, d18            @ mul add\n"

                "pld [%[w0]]                    @ preload cache line, weights r1\n"
                "pld [%[w1]]                    @ preload cache line, weights r1\n"

                "vmlal.s8 q12, d9, d13            @ mul add\n"
                "vmlal.s8 q13, d9, d15            @ mul add\n"
                "vmlal.s8 q14, d9, d17           @ mul add\n"
                "vmlal.s8 q15, d9, d19           @ mul add\n"

                "pld [%[w2]]                    @ preload cache line, weights r2\n"
                "pld [%[w3]]                    @ preload cache line, weights r3\n"

                "vaddw.s16 q0, q0, d24                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d26                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d28                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d30                 @addw \n" // out1 += vget_low_s16(out10)

                "vaddw.s16 q0, q0, d25                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d27                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d29                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d31                 @addw \n" // out1 += vget_low_s16(out10)

                // pair add to final result
                "2:                             @ end processing\n"
                "vpadd.s32 d8, d0, d1           @ pair add, first step\n"
                "vpadd.s32 d9, d2, d3           @ pair add, first step\n"
                "vpadd.s32 d10, d4, d5          @ pair add, first step\n"
                "vpadd.s32 d11, d6, d7          @ pair add, first step\n"

                "vpadd.s32 d0, d8, d9           @ pair add, second step\n"
                "vpadd.s32 d1, d10, d11         @ pair add, second step\n"

                "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

        :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1), \
                 [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [out] "+r"(ptr_out), \
                 [cnt] "+r"(cnt)
        :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2)
        :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", \
                "q12", "q13", "q14", "q15"
        );
    }

    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 4; j < M; j++) {
        int *ptr_out = data_out + j;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;
        asm volatile(
        "pld [%[in]] @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"
                "vmov.u32 q0, #0                    @ set q0 to 0\n"
                "cmp %[cnt], #1                     @ check whether has main loop\n"
                "blt  3f                            @ jump to tail\n"

                "1:                                 @ main loop\n"
                "vld1.8 {d24-d25}, [%[in]]!    @ load input, q12\n"
                "vld1.8 {d28-d29}, [%[w0]]!    @ load weights 14\n"

                "pld [%[in]]                        @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"

                "vmull.s8 q10, d24, d28            @ mul add\n"

                "vmlal.s8 q10, d25, d29             @ mul add\n"
                "subs %[cnt] , #1                   @ sub loop count \n"
                "vaddw.s16 q0, q0, d20                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q0, q0, d21                 @addw \n" // out1 += vget_low_s16(out10)
                "bne 1b                             @ jump to main loop\n"

                // check tails
                "3:                                 @ tail\n"
                "cmp %[tail], #1                    @ check whether has mid cols\n"
                "blt  2f                            @ jump to end\n"

                // process tail
                "vld1.8 {d24-d25}, [%[in]]!        @ load input, q12,q13\n"
                // deal with right pad
                "vmov.u32 d2, #0                    @ dump q8 to zero, for bit select in tail\n"
                "vld1.8 {d28-d29}, [%[w0]]!    @ load weights 14\n"

                "vbif d24, d2, %[mask1]            @ bit select, deal with right pad\n"
                "vbif d25, d2, %[mask2]            @ bit select, deal with right pad\n"

                "vmull.s8 q10, d24, d28            @ mul add\n"
                "vmlal.s8 q10, d25, d29             @ mul add\n"

                "vaddw.s16 q0, q0, d20                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q0, q0, d21                 @addw \n" // out1 += vget_low_s16(out10)
                // pair add to final result
                "2:                                 @ end processing\n"
                "vpadd.s32 d2, d0, d1               @ pair add, first step\n"
                "vpadd.s32 d3, d2, d2               @ pair add, final step\n"
                "vst1.32 {d3[0]}, [%[out]]          @ save result\n"
        :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [out] "+r"(ptr_out), \
            [cnt] "+r"(cnt)
        :[tail] "r"(tail), [mask1] "w"(vmask1), [mask2] "w"(vmask2)
        :"q0", "q1", "q10", "q12", "q13", "q14", "q15"
        );
    }
#endif //__aarch64__
}


void sgemv_relu_int8(const bool transA, const int M, const int N, \
    const signed char* A, const signed char* x, int* y) {
    int* data_out = y;
    const signed char* data_in = x;
    const signed char* weights_ptr = A;

#ifdef __aarch64__
    int cnt_loop = N >> 4;
    int tail = N & 15;
    int out_cnt = M >> 3;

    unsigned char imask[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    uint8x8_t vmask1 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask));
    uint8x8_t vmask2 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask + 8));

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {

        int out_idx = j * 8;
        int *ptr_out = data_out + out_idx;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * out_idx);
        const signed char *ptr_w1 = ptr_w0 + N;
        const signed char *ptr_w2 = ptr_w1 + N;
        const signed char *ptr_w3 = ptr_w2 + N;
        const signed char *ptr_w4 = ptr_w3 + N;
        const signed char *ptr_w5 = ptr_w4 + N;
        const signed char *ptr_w6 = ptr_w5 + N;
        const signed char *ptr_w7 = ptr_w6 + N;

         asm volatile(
            "prfm  pldl1keep, [%[in]]   \n"
                    "prfm  pldl1keep, [%[w0]]   \n"
                    "prfm  pldl1keep, [%[w1]]   \n"
                    "prfm  pldl1keep, [%[w2]]   \n"
                    "prfm  pldl1keep, [%[w3]]   \n"
                    "prfm  pldl1keep, [%[w4]]   \n"
                    "prfm  pldl1keep, [%[w5]]   \n"
                    "prfm  pldl1keep, [%[w6]]   \n"
                    "prfm  pldl1keep, [%[w7]]   \n"
                     :
        :[in] "r"(ptr_in), [w0] "r"(ptr_w0), [w1] "r"(ptr_w1), [w2] "r"(ptr_w2), \
                [w3] "r"(ptr_w3), [w4] "r"(ptr_w4), [w5] "r"(ptr_w5), [w6] "r"(ptr_w6), \
                [w7] "r" (ptr_w7)
        :"memory"
        );
        int32x4_t sum0 = vdupq_n_s32(0);
        int32x4_t sum1 = vdupq_n_s32(0);
        int32x4_t sum2 = vdupq_n_s32(0);
        int32x4_t sum3 = vdupq_n_s32(0);
        int32x4_t sum4 = vdupq_n_s32(0);
        int32x4_t sum5 = vdupq_n_s32(0);
        int32x4_t sum6 = vdupq_n_s32(0);
        int32x4_t sum7 = vdupq_n_s32(0);

        for (int i = 0; i < cnt_loop; i++){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);
            int8x8_t w1_0 = vld1_s8(ptr_w1);
            int8x8_t w2_0 = vld1_s8(ptr_w2);
            int8x8_t w3_0 = vld1_s8(ptr_w3);
            int8x8_t w4_0 = vld1_s8(ptr_w4);
            int8x8_t w5_0 = vld1_s8(ptr_w5);
            int8x8_t w6_0 = vld1_s8(ptr_w6);
            int8x8_t w7_0 = vld1_s8(ptr_w7);

            int16x8_t sum0_16 = vmull_s8(din0, w0_0);
            int16x8_t sum1_16 = vmull_s8(din0, w1_0);
            int16x8_t sum2_16 = vmull_s8(din0, w2_0);
            int16x8_t sum3_16 = vmull_s8(din0, w3_0);
            int16x8_t sum4_16 = vmull_s8(din0, w4_0);
            int16x8_t sum5_16 = vmull_s8(din0, w5_0);
            int16x8_t sum6_16 = vmull_s8(din0, w6_0);
            int16x8_t sum7_16 = vmull_s8(din0, w7_0);

            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);
            int8x8_t w1_1 = vld1_s8(ptr_w1 + 8);
            int8x8_t w2_1 = vld1_s8(ptr_w2 + 8);
            int8x8_t w3_1 = vld1_s8(ptr_w3 + 8);
            int8x8_t w4_1 = vld1_s8(ptr_w4 + 8);
            int8x8_t w5_1 = vld1_s8(ptr_w5 + 8);
            int8x8_t w6_1 = vld1_s8(ptr_w6 + 8);
            int8x8_t w7_1 = vld1_s8(ptr_w7 + 8);

            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum1_16 = vmlal_s8(sum1_16, din1, w1_1);
            sum2_16 = vmlal_s8(sum2_16, din1, w2_1);
            sum3_16 = vmlal_s8(sum3_16, din1, w3_1);
            sum4_16 = vmlal_s8(sum4_16, din1, w4_1);
            sum5_16 = vmlal_s8(sum5_16, din1, w5_1);
            sum6_16 = vmlal_s8(sum6_16, din1, w6_1);
            sum7_16 = vmlal_s8(sum7_16, din1, w7_1);

            ptr_in += 16;
            ptr_w0 += 16;
            ptr_w1 += 16;
            ptr_w2 += 16;
            ptr_w3 += 16;
            ptr_w4 += 16;
            ptr_w5 += 16;
            ptr_w6 += 16;
            ptr_w7 += 16;

            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_low_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_low_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_low_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_low_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_low_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_low_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_low_s16(sum7_16));

            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_high_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_high_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_high_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_high_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_high_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_high_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_high_s16(sum7_16));

        }
        if (tail > 0){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);
            int8x8_t w1_0 = vld1_s8(ptr_w1);
            int8x8_t w2_0 = vld1_s8(ptr_w2);
            int8x8_t w3_0 = vld1_s8(ptr_w3);

            din0 = vbsl_s8(vmask1, din0, vzero);

            int8x8_t w4_0 = vld1_s8(ptr_w4);
            int8x8_t w5_0 = vld1_s8(ptr_w5);
            int8x8_t w6_0 = vld1_s8(ptr_w6);
            int8x8_t w7_0 = vld1_s8(ptr_w7);

            int16x8_t sum0_16 = vmull_s8(din0, w0_0);
            int16x8_t sum1_16 = vmull_s8(din0, w1_0);
            int16x8_t sum2_16 = vmull_s8(din0, w2_0);
            int16x8_t sum3_16 = vmull_s8(din0, w3_0);
            int16x8_t sum4_16 = vmull_s8(din0, w4_0);
            int16x8_t sum5_16 = vmull_s8(din0, w5_0);
            int16x8_t sum6_16 = vmull_s8(din0, w6_0);
            int16x8_t sum7_16 = vmull_s8(din0, w7_0);

            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);
            int8x8_t w1_1 = vld1_s8(ptr_w1 + 8);
            int8x8_t w2_1 = vld1_s8(ptr_w2 + 8);
            int8x8_t w3_1 = vld1_s8(ptr_w3 + 8);

            din1 = vbsl_s8(vmask2, din1, vzero);

            int8x8_t w4_1 = vld1_s8(ptr_w4 + 8);
            int8x8_t w5_1 = vld1_s8(ptr_w5 + 8);
            int8x8_t w6_1 = vld1_s8(ptr_w6 + 8);
            int8x8_t w7_1 = vld1_s8(ptr_w7 + 8);

            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum1_16 = vmlal_s8(sum1_16, din1, w1_1);
            sum2_16 = vmlal_s8(sum2_16, din1, w2_1);
            sum3_16 = vmlal_s8(sum3_16, din1, w3_1);
            sum4_16 = vmlal_s8(sum4_16, din1, w4_1);
            sum5_16 = vmlal_s8(sum5_16, din1, w5_1);
            sum6_16 = vmlal_s8(sum6_16, din1, w6_1);
            sum7_16 = vmlal_s8(sum7_16, din1, w7_1);

            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_low_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_low_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_low_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_low_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_low_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_low_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_low_s16(sum7_16));

            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_high_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_high_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_high_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_high_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_high_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_high_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_high_s16(sum7_16));
        }
        //add
        int32x4_t vout0 = vpaddq_s32(sum0, sum1);
        int32x4_t vout1 = vpaddq_s32(sum2, sum3);
        int32x4_t vout2 = vpaddq_s32(sum4, sum5);
        int32x4_t vout3 = vpaddq_s32(sum6, sum7);
        int32x4_t vdout = vpaddq_s32(vout0, vout1);
        int32x4_t vdout1 = vpaddq_s32(vout2, vout3);
        vdout = vmaxq_s32(vdout, vzero_32);
        vdout1 = vmaxq_s32(vdout1, vzero_32);
        vst1q_s32(ptr_out, vdout);
        vst1q_s32(ptr_out + 4, vdout1);
    }
    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 8; j < M; ++j){
        int *ptr_out = data_out + j;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;

         asm volatile(
            "prfm  pldl1keep, [%[in]]   \n"
                    "prfm  pldl1keep, [%[w0]]   \n"
                     :
        :[in] "r"(ptr_in), [w0] "r"(ptr_w0)
        :"memory"
        );

        int32x4_t sum0 = vdupq_n_s32(0);
        for (int i = 0; i < cnt_loop; i++){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);//a = 01234567

            int16x8_t sum0_16 = vmull_s8(din0, w0_0);

            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);//a = 01234567

            ptr_in += 16;
            ptr_w0 += 16;

            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
        }
        if (tail > 0){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);//a = 01234567

            din0 = vbsl_s8(vmask1, din0, vzero);
            din1 = vbsl_s8(vmask2, din1, vzero);
            int16x8_t sum0_16 = vmull_s8(din0, w0_0);
            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
        }
        //add
        int32x2_t vout = vpadd_s32(vget_low_s32(sum0), vget_high_s32(sum0));
        int32x2_t vdout = vpadd_s32(vout, vout);
        int tmp = vget_lane_s32(vdout, 0);
        *ptr_out = tmp > 0 ? tmp : 0;
    }
#else
    int cnt_loop = N >> 4;
    int tail = N & 15;
    int out_cnt = M >> 2;

    unsigned char imask[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    uint8x8_t vmask1 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask));
    uint8x8_t vmask2 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask + 8));

    // printf("cnt_loop: %d, tail: %d, out_cnt: %d \n", cnt_loop, tail, out_cnt);

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {
        int out_idx = j * 4;
        int *ptr_out = data_out + out_idx;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * out_idx);
        const signed char *ptr_w1 = ptr_w0 + N;
        const signed char *ptr_w2 = ptr_w1 + N;
        const signed char *ptr_w3 = ptr_w2 + N;

        int cnt = cnt_loop;
        asm volatile(
        "pld [%[in]] @ preload cache line, input\n"
                "pld [%[w0]]                    @ preload cache line, weights r0\n"
                "pld [%[w1]]                    @ preload cache line, weights r1\n"
                "pld [%[w2]]                    @ preload cache line, weights r2\n"
                "pld [%[w3]]                    @ preload cache line, weights r3\n"

                "vmov.u32 q0, #0                @ set q0 to 0\n"
                "vmov.u32 q1, #0                @ set q1 to 0\n"
                "vmov.u32 q2, #0                @ set q2 to 0\n"
                "vmov.u32 q3, #0                @ set q3 to 0\n"

                "cmp %[cnt], #1                 @ check whether has main loop\n"
                "blt  3f                        @ jump to tail\n"

                "1:                             @ main loop\n"
                "vld1.8 {d8-d9}, [%[in]]!    @ load input, q4, q5\n"
                "vld1.8 {d12-d13}, [%[w0]]!    @ load weights r0, q6\n"
                "vld1.8 {d14-d15}, [%[w1]]!    @ load weights r1, q7\n"
                "vld1.8 {d16-d17}, [%[w2]]!    @ load weights r2, q8\n"
                "vld1.8 {d18-d19}, [%[w3]]!    @ load weights r3, q9\n"

                "vmull.s8 q12, d8, d12            @ mul add\n"
                "vmull.s8 q13, d8, d14            @ mul add\n"
                "vmull.s8 q14, d8, d16            @ mul add\n"
                "vmull.s8 q15, d8, d18            @ mul add\n"

                "pld [%[in]]                    @ preload cache line, input\n"
                "pld [%[w0]]                    @ preload cache line, weights r0\n"

                "vmlal.s8 q12, d9, d13            @ mul add\n"
                "vmlal.s8 q13, d9, d15            @ mul add\n"
                "vmlal.s8 q14, d9, d17           @ mul add\n"
                "vmlal.s8 q15, d9, d19           @ mul add\n"

                "pld [%[w1]]                    @ preload cache line, weights r1\n"
                "pld [%[w2]]                    @ preload cache line, weights r2\n"
                "pld [%[w3]]                    @ preload cache line, weights r3\n"

                "vaddw.s16 q0, q0, d24                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d26                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d28                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d30                 @addw \n" // out1 += vget_low_s16(out10)

                "vaddw.s16 q0, q0, d25                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d27                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d29                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d31                 @addw \n" // out1 += vget_low_s16(out10)

                // check loop end
                "subs %[cnt], #1                @ sub loop count \n"

                "bne 1b                         @ jump to main loop\n"

                // check tails
                "3:                             @ check tail\n"
                "cmp %[tail], #1                @ check whether has mid cols\n"
                "blt  2f                        @ jump to end\n"

                // process tail
                "vld1.8 {d8-d9}, [%[in]]!    @ load input, q4, q5\n"
                // deal with right pad
                "vmov.u32 d10, #0                @ dump q8 to zero, for bit select in tail\n"
                "vld1.8 {d12-d13}, [%[w0]]!    @ load weights r0, q6\n"
                "vld1.8 {d14-d15}, [%[w1]]!    @ load weights r1, q7\n"
                "vld1.8 {d16-d17}, [%[w2]]!    @ load weights r2, q8\n"
                "vld1.8 {d18-d19}, [%[w3]]!    @ load weights r3, q9\n"

                "vbif d8, d10, %[mask1]         @ bit select, deal with right pad\n"
                "vbif d9, d10, %[mask2]         @ bit select, deal with right pad\n"
                "pld [%[in]]                    @ preload cache line, input\n"

                "vmull.s8 q12, d8, d12            @ mul add\n"
                "vmull.s8 q13, d8, d14            @ mul add\n"
                "vmull.s8 q14, d8, d16            @ mul add\n"
                "vmull.s8 q15, d8, d18            @ mul add\n"

                "pld [%[w0]]                    @ preload cache line, weights r1\n"
                "pld [%[w1]]                    @ preload cache line, weights r1\n"

                "vmlal.s8 q12, d9, d13            @ mul add\n"
                "vmlal.s8 q13, d9, d15            @ mul add\n"
                "vmlal.s8 q14, d9, d17           @ mul add\n"
                "vmlal.s8 q15, d9, d19           @ mul add\n"

                "pld [%[w2]]                    @ preload cache line, weights r2\n"
                "pld [%[w3]]                    @ preload cache line, weights r3\n"

                "vaddw.s16 q0, q0, d24                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d26                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d28                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d30                 @addw \n" // out1 += vget_low_s16(out10)

                "vaddw.s16 q0, q0, d25                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d27                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d29                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d31                 @addw \n" // out1 += vget_low_s16(out10)

                // pair add to final result
                "2:                             @ end processing\n"
                "vpadd.s32 d8, d0, d1           @ pair add, first step\n"
                "vpadd.s32 d9, d2, d3           @ pair add, first step\n"
                "vpadd.s32 d10, d4, d5          @ pair add, first step\n"
                "vpadd.s32 d11, d6, d7          @ pair add, first step\n"

                "vmov.u32 d12, #0                @ set q0 to 0\n"

                "vpadd.s32 d0, d8, d9           @ pair add, second step\n"
                "vpadd.s32 d1, d10, d11         @ pair add, second step\n"

                "vmax.s32 d0, d0, d12            @ relu \n"
                "vmax.s32 d1, d1, d12            @ relu \n"

                "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

        :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1), \
                 [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [out] "+r"(ptr_out), \
                 [cnt] "+r"(cnt)
        :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2)
        :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", \
                "q12", "q13", "q14", "q15"
        );
    }

    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 4; j < M; j++) {
        int *ptr_out = data_out + j;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;
        asm volatile(
        "pld [%[in]] @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"
                "vmov.u32 q0, #0                    @ set q0 to 0\n"
                "cmp %[cnt], #1                     @ check whether has main loop\n"
                "blt  3f                            @ jump to tail\n"

                "1:                                 @ main loop\n"
                "vld1.8 {d24-d25}, [%[in]]!    @ load input, q12\n"
                "vld1.8 {d28-d29}, [%[w0]]!    @ load weights 14\n"

                "pld [%[in]]                        @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"

                "vmull.s8 q10, d24, d28            @ mul add\n"

                "vmlal.s8 q10, d25, d29             @ mul add\n"
                "subs %[cnt] , #1                   @ sub loop count \n"
                "vaddw.s16 q0, q0, d20                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q0, q0, d21                 @addw \n" // out1 += vget_low_s16(out10)
                "bne 1b                             @ jump to main loop\n"

                // check tails
                "3:                                 @ tail\n"
                "cmp %[tail], #1                    @ check whether has mid cols\n"
                "blt  2f                            @ jump to end\n"

                // process tail
                "vld1.8 {d24-d25}, [%[in]]!        @ load input, q12,q13\n"
                // deal with right pad
                "vmov.u32 d2, #0                    @ dump q8 to zero, for bit select in tail\n"
                "vld1.8 {d28-d29}, [%[w0]]!    @ load weights 14\n"

                "vbif d24, d2, %[mask1]            @ bit select, deal with right pad\n"
                "vbif d25, d2, %[mask2]            @ bit select, deal with right pad\n"

                "vmull.s8 q10, d24, d28            @ mul add\n"
                "vmlal.s8 q10, d25, d29             @ mul add\n"

                "vaddw.s16 q0, q0, d20                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q0, q0, d21                 @addw \n" // out1 += vget_low_s16(out10)
                // pair add to final result
                "2:                                 @ end processing\n"
                "vpadd.s32 d2, d0, d1               @ pair add, first step\n"
                "vpadd.s32 d3, d2, d2               @ pair add, final step\n"
                "vmov.u32 d5, #0                    @ dump q8 to zero, for bit select in tail\n"
                "vmax.s32 d3, d3, d5            @ relu \n"
                "vst1.32 {d3[0]}, [%[out]]          @ save result\n"
        :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [out] "+r"(ptr_out), \
            [cnt] "+r"(cnt)
        :[tail] "r"(tail), [mask1] "w"(vmask1), [mask2] "w"(vmask2)
        :"q0", "q1", "q10", "q12", "q13", "q14", "q15"
        );
    }
#endif //__aarch64__
}


void sgemv_bias_int8(const bool transA, const int M, const int N, \
    const signed char* A, const signed char* x, int* y, const int* bias) {
    int* data_out = y;
    const signed char* data_in = x;
    const signed char* weights_ptr = A;

#ifdef __aarch64__
    int cnt_loop = N >> 4;
    int tail = N & 15;
    int out_cnt = M >> 3;

    unsigned char imask[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    uint8x8_t vmask1 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask));
    uint8x8_t vmask2 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask + 8));

    int8x8_t vzero = vdup_n_s8(0);

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {

        int out_idx = j * 8;
        int *ptr_out = data_out + out_idx;
        const int* bias_ptr = bias + out_idx;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * out_idx);
        const signed char *ptr_w1 = ptr_w0 + N;
        const signed char *ptr_w2 = ptr_w1 + N;
        const signed char *ptr_w3 = ptr_w2 + N;
        const signed char *ptr_w4 = ptr_w3 + N;
        const signed char *ptr_w5 = ptr_w4 + N;
        const signed char *ptr_w6 = ptr_w5 + N;
        const signed char *ptr_w7 = ptr_w6 + N;

         asm volatile(
            "prfm  pldl1keep, [%[in]]   \n"
                    "prfm  pldl1keep, [%[w0]]   \n"
                    "prfm  pldl1keep, [%[w1]]   \n"
                    "prfm  pldl1keep, [%[w2]]   \n"
                    "prfm  pldl1keep, [%[w3]]   \n"
                    "prfm  pldl1keep, [%[w4]]   \n"
                    "prfm  pldl1keep, [%[w5]]   \n"
                    "prfm  pldl1keep, [%[w6]]   \n"
                    "prfm  pldl1keep, [%[w7]]   \n"
                     :
        :[in] "r"(ptr_in), [w0] "r"(ptr_w0), [w1] "r"(ptr_w1), [w2] "r"(ptr_w2), \
                [w3] "r"(ptr_w3), [w4] "r"(ptr_w4), [w5] "r"(ptr_w5), [w6] "r"(ptr_w6), \
                [w7] "r" (ptr_w7)
        :"memory"
        );
        int32x4_t sum0 = vdupq_n_s32(0);
        int32x4_t sum1 = vdupq_n_s32(0);
        int32x4_t sum2 = vdupq_n_s32(0);
        int32x4_t sum3 = vdupq_n_s32(0);
        int32x4_t sum4 = vdupq_n_s32(0);
        int32x4_t sum5 = vdupq_n_s32(0);
        int32x4_t sum6 = vdupq_n_s32(0);
        int32x4_t sum7 = vdupq_n_s32(0);

        for (int i = 0; i < cnt_loop; i++){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);
            int8x8_t w1_0 = vld1_s8(ptr_w1);
            int8x8_t w2_0 = vld1_s8(ptr_w2);
            int8x8_t w3_0 = vld1_s8(ptr_w3);
            int8x8_t w4_0 = vld1_s8(ptr_w4);
            int8x8_t w5_0 = vld1_s8(ptr_w5);
            int8x8_t w6_0 = vld1_s8(ptr_w6);
            int8x8_t w7_0 = vld1_s8(ptr_w7);

            int16x8_t sum0_16 = vmull_s8(din0, w0_0);
            int16x8_t sum1_16 = vmull_s8(din0, w1_0);
            int16x8_t sum2_16 = vmull_s8(din0, w2_0);
            int16x8_t sum3_16 = vmull_s8(din0, w3_0);
            int16x8_t sum4_16 = vmull_s8(din0, w4_0);
            int16x8_t sum5_16 = vmull_s8(din0, w5_0);
            int16x8_t sum6_16 = vmull_s8(din0, w6_0);
            int16x8_t sum7_16 = vmull_s8(din0, w7_0);

            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);
            int8x8_t w1_1 = vld1_s8(ptr_w1 + 8);
            int8x8_t w2_1 = vld1_s8(ptr_w2 + 8);
            int8x8_t w3_1 = vld1_s8(ptr_w3 + 8);
            int8x8_t w4_1 = vld1_s8(ptr_w4 + 8);
            int8x8_t w5_1 = vld1_s8(ptr_w5 + 8);
            int8x8_t w6_1 = vld1_s8(ptr_w6 + 8);
            int8x8_t w7_1 = vld1_s8(ptr_w7 + 8);

            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum1_16 = vmlal_s8(sum1_16, din1, w1_1);
            sum2_16 = vmlal_s8(sum2_16, din1, w2_1);
            sum3_16 = vmlal_s8(sum3_16, din1, w3_1);
            sum4_16 = vmlal_s8(sum4_16, din1, w4_1);
            sum5_16 = vmlal_s8(sum5_16, din1, w5_1);
            sum6_16 = vmlal_s8(sum6_16, din1, w6_1);
            sum7_16 = vmlal_s8(sum7_16, din1, w7_1);

            ptr_in += 16;
            ptr_w0 += 16;
            ptr_w1 += 16;
            ptr_w2 += 16;
            ptr_w3 += 16;
            ptr_w4 += 16;
            ptr_w5 += 16;
            ptr_w6 += 16;
            ptr_w7 += 16;

            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_low_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_low_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_low_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_low_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_low_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_low_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_low_s16(sum7_16));

            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_high_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_high_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_high_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_high_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_high_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_high_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_high_s16(sum7_16));

        }
        if (tail > 0){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);
            int8x8_t w1_0 = vld1_s8(ptr_w1);
            int8x8_t w2_0 = vld1_s8(ptr_w2);
            int8x8_t w3_0 = vld1_s8(ptr_w3);

            din0 = vbsl_s8(vmask1, din0, vzero);

            int8x8_t w4_0 = vld1_s8(ptr_w4);
            int8x8_t w5_0 = vld1_s8(ptr_w5);
            int8x8_t w6_0 = vld1_s8(ptr_w6);
            int8x8_t w7_0 = vld1_s8(ptr_w7);

            int16x8_t sum0_16 = vmull_s8(din0, w0_0);
            int16x8_t sum1_16 = vmull_s8(din0, w1_0);
            int16x8_t sum2_16 = vmull_s8(din0, w2_0);
            int16x8_t sum3_16 = vmull_s8(din0, w3_0);
            int16x8_t sum4_16 = vmull_s8(din0, w4_0);
            int16x8_t sum5_16 = vmull_s8(din0, w5_0);
            int16x8_t sum6_16 = vmull_s8(din0, w6_0);
            int16x8_t sum7_16 = vmull_s8(din0, w7_0);

            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);
            int8x8_t w1_1 = vld1_s8(ptr_w1 + 8);
            int8x8_t w2_1 = vld1_s8(ptr_w2 + 8);
            int8x8_t w3_1 = vld1_s8(ptr_w3 + 8);

            din1 = vbsl_s8(vmask2, din1, vzero);

            int8x8_t w4_1 = vld1_s8(ptr_w4 + 8);
            int8x8_t w5_1 = vld1_s8(ptr_w5 + 8);
            int8x8_t w6_1 = vld1_s8(ptr_w6 + 8);
            int8x8_t w7_1 = vld1_s8(ptr_w7 + 8);

            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum1_16 = vmlal_s8(sum1_16, din1, w1_1);
            sum2_16 = vmlal_s8(sum2_16, din1, w2_1);
            sum3_16 = vmlal_s8(sum3_16, din1, w3_1);
            sum4_16 = vmlal_s8(sum4_16, din1, w4_1);
            sum5_16 = vmlal_s8(sum5_16, din1, w5_1);
            sum6_16 = vmlal_s8(sum6_16, din1, w6_1);
            sum7_16 = vmlal_s8(sum7_16, din1, w7_1);

            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_low_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_low_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_low_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_low_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_low_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_low_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_low_s16(sum7_16));

            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_high_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_high_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_high_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_high_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_high_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_high_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_high_s16(sum7_16));
        }
        int32x4_t bias_val0 = vld1q_s32(bias_ptr);
        int32x4_t bias_val1 = vld1q_s32(bias_ptr + 4);
        //add
        int32x4_t vout0 = vpaddq_s32(sum0, sum1);
        int32x4_t vout1 = vpaddq_s32(sum2, sum3);
        int32x4_t vout2 = vpaddq_s32(sum4, sum5);
        int32x4_t vout3 = vpaddq_s32(sum6, sum7);

        int32x4_t vdout = vpaddq_s32(vout0, vout1);
        int32x4_t vdout1 = vpaddq_s32(vout2, vout3);

        vdout = vaddq_s32(vdout, bias_val0);
        vdout1 = vaddq_s32(vdout1, bias_val1);
        vst1q_s32(ptr_out, vdout);
        vst1q_s32(ptr_out + 4, vdout1);
    }
    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 8; j < M; ++j){
        int *ptr_out = data_out + j;
        const int *bias_ptr = bias + j;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;

         asm volatile(
            "prfm  pldl1keep, [%[in]]   \n"
                    "prfm  pldl1keep, [%[w0]]   \n"
                     :
        :[in] "r"(ptr_in), [w0] "r"(ptr_w0)
        :"memory"
        );

        int32x4_t sum0 = vdupq_n_s32(0);
        for (int i = 0; i < cnt_loop; i++){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);//a = 01234567

            int16x8_t sum0_16 = vmull_s8(din0, w0_0);

            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);//a = 01234567

            ptr_in += 16;
            ptr_w0 += 16;

            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
        }
        if (tail > 0){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);//a = 01234567

            din0 = vbsl_s8(vmask1, din0, vzero);
            din1 = vbsl_s8(vmask2, din1, vzero);
            int16x8_t sum0_16 = vmull_s8(din0, w0_0);
            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
        }
        //add
        int32x2_t vout = vpadd_s32(vget_low_s32(sum0), vget_high_s32(sum0));
        int32x2_t vdout = vpadd_s32(vout, vout);
        *ptr_out = vget_lane_s32(vdout, 0) + bias_ptr[0];
    }
#else
    int cnt_loop = N >> 4;
    int tail = N & 15;
    int out_cnt = M >> 2;

    unsigned char imask[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    uint8x8_t vmask1 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask));
    uint8x8_t vmask2 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask + 8));

    // printf("cnt_loop: %d, tail: %d, out_cnt: %d \n", cnt_loop, tail, out_cnt);

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {
        int out_idx = j * 4;
        int *ptr_out = data_out + out_idx;
        const int* bias_ptr = bias + out_idx;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * out_idx);
        const signed char *ptr_w1 = ptr_w0 + N;
        const signed char *ptr_w2 = ptr_w1 + N;
        const signed char *ptr_w3 = ptr_w2 + N;

        int cnt = cnt_loop;
        asm volatile(
        "pld [%[in]] @ preload cache line, input\n"
                "pld [%[w0]]                    @ preload cache line, weights r0\n"
                "pld [%[w1]]                    @ preload cache line, weights r1\n"
                "pld [%[w2]]                    @ preload cache line, weights r2\n"
                "pld [%[w3]]                    @ preload cache line, weights r3\n"

                "vmov.u32 q0, #0                @ set q0 to 0\n"
                "vmov.u32 q1, #0                @ set q1 to 0\n"
                "vmov.u32 q2, #0                @ set q2 to 0\n"
                "vmov.u32 q3, #0                @ set q3 to 0\n"

                "cmp %[cnt], #1                 @ check whether has main loop\n"
                "blt  3f                        @ jump to tail\n"

                "1:                             @ main loop\n"
                "vld1.8 {d8-d9}, [%[in]]!    @ load input, q4, q5\n"
                "vld1.8 {d12-d13}, [%[w0]]!    @ load weights r0, q6\n"
                "vld1.8 {d14-d15}, [%[w1]]!    @ load weights r1, q7\n"
                "vld1.8 {d16-d17}, [%[w2]]!    @ load weights r2, q8\n"
                "vld1.8 {d18-d19}, [%[w3]]!    @ load weights r3, q9\n"

                "vmull.s8 q12, d8, d12            @ mul add\n"
                "vmull.s8 q13, d8, d14            @ mul add\n"
                "vmull.s8 q14, d8, d16            @ mul add\n"
                "vmull.s8 q15, d8, d18            @ mul add\n"

                "pld [%[in]]                    @ preload cache line, input\n"
                "pld [%[w0]]                    @ preload cache line, weights r0\n"

                "vmlal.s8 q12, d9, d13            @ mul add\n"
                "vmlal.s8 q13, d9, d15            @ mul add\n"
                "vmlal.s8 q14, d9, d17           @ mul add\n"
                "vmlal.s8 q15, d9, d19           @ mul add\n"

                "pld [%[w1]]                    @ preload cache line, weights r1\n"
                "pld [%[w2]]                    @ preload cache line, weights r2\n"
                "pld [%[w3]]                    @ preload cache line, weights r3\n"

                "vaddw.s16 q0, q0, d24                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d26                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d28                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d30                 @addw \n" // out1 += vget_low_s16(out10)

                "vaddw.s16 q0, q0, d25                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d27                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d29                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d31                 @addw \n" // out1 += vget_low_s16(out10)

                // check loop end
                "subs %[cnt], #1                @ sub loop count \n"

                "bne 1b                         @ jump to main loop\n"

                // check tails
                "3:                             @ check tail\n"
                "cmp %[tail], #1                @ check whether has mid cols\n"
                "blt  2f                        @ jump to end\n"

                // process tail
                "vld1.8 {d8-d9}, [%[in]]!    @ load input, q4, q5\n"
                // deal with right pad
                "vmov.u32 d10, #0                @ dump q8 to zero, for bit select in tail\n"
                "vld1.8 {d12-d13}, [%[w0]]!    @ load weights r0, q6\n"
                "vld1.8 {d14-d15}, [%[w1]]!    @ load weights r1, q7\n"
                "vld1.8 {d16-d17}, [%[w2]]!    @ load weights r2, q8\n"
                "vld1.8 {d18-d19}, [%[w3]]!    @ load weights r3, q9\n"

                "vbif d8, d10, %[mask1]         @ bit select, deal with right pad\n"
                "vbif d9, d10, %[mask2]         @ bit select, deal with right pad\n"
                "pld [%[in]]                    @ preload cache line, input\n"

                "vmull.s8 q12, d8, d12            @ mul add\n"
                "vmull.s8 q13, d8, d14            @ mul add\n"
                "vmull.s8 q14, d8, d16            @ mul add\n"
                "vmull.s8 q15, d8, d18            @ mul add\n"

                "pld [%[w0]]                    @ preload cache line, weights r1\n"
                "pld [%[w1]]                    @ preload cache line, weights r1\n"

                "vmlal.s8 q12, d9, d13            @ mul add\n"
                "vmlal.s8 q13, d9, d15            @ mul add\n"
                "vmlal.s8 q14, d9, d17           @ mul add\n"
                "vmlal.s8 q15, d9, d19           @ mul add\n"

                "pld [%[w2]]                    @ preload cache line, weights r2\n"
                "pld [%[w3]]                    @ preload cache line, weights r3\n"

                "vaddw.s16 q0, q0, d24                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d26                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d28                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d30                 @addw \n" // out1 += vget_low_s16(out10)

                "vaddw.s16 q0, q0, d25                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d27                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d29                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d31                 @addw \n" // out1 += vget_low_s16(out10)

                // pair add to final result
                "2:                             @ end processing\n"
                "vld1.32 {d12-d13}, [%[bias]]   @ load weights r0, q6,q7\n"
                "vpadd.s32 d8, d0, d1           @ pair add, first step\n"
                "vpadd.s32 d9, d2, d3           @ pair add, first step\n"
                "vpadd.s32 d10, d4, d5          @ pair add, first step\n"
                "vpadd.s32 d11, d6, d7          @ pair add, first step\n"

                "vpadd.s32 d0, d8, d9           @ pair add, second step\n"
                "vpadd.s32 d1, d10, d11         @ pair add, second step\n"

                "vadd.s32 q0, q0, q6           @ add \n"

                "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

        :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1), \
                 [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [out] "+r"(ptr_out), \
                 [cnt] "+r"(cnt)
        :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2), [bias] "r" (bias_ptr)
        :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", \
                "q12", "q13", "q14", "q15"
        );
    }

    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 4; j < M; j++) {
        int *ptr_out = data_out + j;
        const int *bias_ptr = bias + j;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;
        asm volatile(
        "pld [%[in]] @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"
                "vmov.u32 q0, #0                    @ set q0 to 0\n"
                "cmp %[cnt], #1                     @ check whether has main loop\n"
                "blt  3f                            @ jump to tail\n"

                "1:                                 @ main loop\n"
                "vld1.8 {d24-d25}, [%[in]]!    @ load input, q12\n"
                "vld1.8 {d28-d29}, [%[w0]]!    @ load weights 14\n"

                "pld [%[in]]                        @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"

                "vmull.s8 q10, d24, d28            @ mul add\n"

                "vmlal.s8 q10, d25, d29             @ mul add\n"
                "subs %[cnt] , #1                   @ sub loop count \n"
                "vaddw.s16 q0, q0, d20                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q0, q0, d21                 @addw \n" // out1 += vget_low_s16(out10)
                "bne 1b                             @ jump to main loop\n"

                // check tails
                "3:                                 @ tail\n"
                "cmp %[tail], #1                    @ check whether has mid cols\n"
                "blt  2f                            @ jump to end\n"

                // process tail
                "vld1.8 {d24-d25}, [%[in]]!        @ load input, q12,q13\n"
                // deal with right pad
                "vmov.u32 d2, #0                    @ dump q8 to zero, for bit select in tail\n"
                "vld1.8 {d28-d29}, [%[w0]]!    @ load weights 14\n"

                "vbif d24, d2, %[mask1]            @ bit select, deal with right pad\n"
                "vbif d25, d2, %[mask2]            @ bit select, deal with right pad\n"

                "vmull.s8 q10, d24, d28            @ mul add\n"
                "vmlal.s8 q10, d25, d29             @ mul add\n"

                "vaddw.s16 q0, q0, d20                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q0, q0, d21                 @addw \n" // out1 += vget_low_s16(out10)
                // pair add to final result
                "2:                                 @ end processing\n"
                "vld1.32 {d12}, [%[bias]]   @ load weights r0, q6,q7\n"
                "vpadd.s32 d2, d0, d1               @ pair add, first step\n"

                "vpadd.s32 d3, d2, d2               @ pair add, final step\n"

                "vadd.s32 d3, d3, d12           @ add \n"

                "vst1.32 {d3[0]}, [%[out]]          @ save result\n"
        :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [out] "+r"(ptr_out), \
            [cnt] "+r"(cnt)
        :[tail] "r"(tail), [mask1] "w"(vmask1), [mask2] "w"(vmask2), [bias] "r" (bias_ptr)
        :"q0", "q1", "q10", "q12", "q13", "q14", "q15"
        );
    }
#endif //__aarch64__
}


void sgemv_bias_relu_int8(const bool transA, const int M, const int N, \
    const signed char* A, const signed char* x, int* y, const int* bias) {
    int* data_out = y;
    const signed char* data_in = x;
    const signed char* weights_ptr = A;

#ifdef __aarch64__
    int cnt_loop = N >> 4;
    int tail = N & 15;
    int out_cnt = M >> 3;

    unsigned char imask[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    uint8x8_t vmask1 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask));
    uint8x8_t vmask2 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask + 8));

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {

        int out_idx = j * 8;
        int *ptr_out = data_out + out_idx;
        const int* bias_ptr = bias + out_idx;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * out_idx);
        const signed char *ptr_w1 = ptr_w0 + N;
        const signed char *ptr_w2 = ptr_w1 + N;
        const signed char *ptr_w3 = ptr_w2 + N;
        const signed char *ptr_w4 = ptr_w3 + N;
        const signed char *ptr_w5 = ptr_w4 + N;
        const signed char *ptr_w6 = ptr_w5 + N;
        const signed char *ptr_w7 = ptr_w6 + N;

         asm volatile(
            "prfm  pldl1keep, [%[in]]   \n"
                    "prfm  pldl1keep, [%[w0]]   \n"
                    "prfm  pldl1keep, [%[w1]]   \n"
                    "prfm  pldl1keep, [%[w2]]   \n"
                    "prfm  pldl1keep, [%[w3]]   \n"
                    "prfm  pldl1keep, [%[w4]]   \n"
                    "prfm  pldl1keep, [%[w5]]   \n"
                    "prfm  pldl1keep, [%[w6]]   \n"
                    "prfm  pldl1keep, [%[w7]]   \n"
                     :
        :[in] "r"(ptr_in), [w0] "r"(ptr_w0), [w1] "r"(ptr_w1), [w2] "r"(ptr_w2), \
                [w3] "r"(ptr_w3), [w4] "r"(ptr_w4), [w5] "r"(ptr_w5), [w6] "r"(ptr_w6), \
                [w7] "r" (ptr_w7)
        :"memory"
        );
        int32x4_t sum0 = vdupq_n_s32(0);
        int32x4_t sum1 = vdupq_n_s32(0);
        int32x4_t sum2 = vdupq_n_s32(0);
        int32x4_t sum3 = vdupq_n_s32(0);
        int32x4_t sum4 = vdupq_n_s32(0);
        int32x4_t sum5 = vdupq_n_s32(0);
        int32x4_t sum6 = vdupq_n_s32(0);
        int32x4_t sum7 = vdupq_n_s32(0);

        for (int i = 0; i < cnt_loop; i++){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);
            int8x8_t w1_0 = vld1_s8(ptr_w1);
            int8x8_t w2_0 = vld1_s8(ptr_w2);
            int8x8_t w3_0 = vld1_s8(ptr_w3);
            int8x8_t w4_0 = vld1_s8(ptr_w4);
            int8x8_t w5_0 = vld1_s8(ptr_w5);
            int8x8_t w6_0 = vld1_s8(ptr_w6);
            int8x8_t w7_0 = vld1_s8(ptr_w7);

            int16x8_t sum0_16 = vmull_s8(din0, w0_0);
            int16x8_t sum1_16 = vmull_s8(din0, w1_0);
            int16x8_t sum2_16 = vmull_s8(din0, w2_0);
            int16x8_t sum3_16 = vmull_s8(din0, w3_0);
            int16x8_t sum4_16 = vmull_s8(din0, w4_0);
            int16x8_t sum5_16 = vmull_s8(din0, w5_0);
            int16x8_t sum6_16 = vmull_s8(din0, w6_0);
            int16x8_t sum7_16 = vmull_s8(din0, w7_0);

            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);
            int8x8_t w1_1 = vld1_s8(ptr_w1 + 8);
            int8x8_t w2_1 = vld1_s8(ptr_w2 + 8);
            int8x8_t w3_1 = vld1_s8(ptr_w3 + 8);
            int8x8_t w4_1 = vld1_s8(ptr_w4 + 8);
            int8x8_t w5_1 = vld1_s8(ptr_w5 + 8);
            int8x8_t w6_1 = vld1_s8(ptr_w6 + 8);
            int8x8_t w7_1 = vld1_s8(ptr_w7 + 8);

            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum1_16 = vmlal_s8(sum1_16, din1, w1_1);
            sum2_16 = vmlal_s8(sum2_16, din1, w2_1);
            sum3_16 = vmlal_s8(sum3_16, din1, w3_1);
            sum4_16 = vmlal_s8(sum4_16, din1, w4_1);
            sum5_16 = vmlal_s8(sum5_16, din1, w5_1);
            sum6_16 = vmlal_s8(sum6_16, din1, w6_1);
            sum7_16 = vmlal_s8(sum7_16, din1, w7_1);

            ptr_in += 16;
            ptr_w0 += 16;
            ptr_w1 += 16;
            ptr_w2 += 16;
            ptr_w3 += 16;
            ptr_w4 += 16;
            ptr_w5 += 16;
            ptr_w6 += 16;
            ptr_w7 += 16;

            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_low_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_low_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_low_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_low_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_low_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_low_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_low_s16(sum7_16));

            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_high_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_high_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_high_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_high_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_high_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_high_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_high_s16(sum7_16));

        }
        if (tail > 0){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);
            int8x8_t w1_0 = vld1_s8(ptr_w1);
            int8x8_t w2_0 = vld1_s8(ptr_w2);
            int8x8_t w3_0 = vld1_s8(ptr_w3);

            din0 = vbsl_s8(vmask1, din0, vzero);

            int8x8_t w4_0 = vld1_s8(ptr_w4);
            int8x8_t w5_0 = vld1_s8(ptr_w5);
            int8x8_t w6_0 = vld1_s8(ptr_w6);
            int8x8_t w7_0 = vld1_s8(ptr_w7);

            int16x8_t sum0_16 = vmull_s8(din0, w0_0);
            int16x8_t sum1_16 = vmull_s8(din0, w1_0);
            int16x8_t sum2_16 = vmull_s8(din0, w2_0);
            int16x8_t sum3_16 = vmull_s8(din0, w3_0);
            int16x8_t sum4_16 = vmull_s8(din0, w4_0);
            int16x8_t sum5_16 = vmull_s8(din0, w5_0);
            int16x8_t sum6_16 = vmull_s8(din0, w6_0);
            int16x8_t sum7_16 = vmull_s8(din0, w7_0);

            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);
            int8x8_t w1_1 = vld1_s8(ptr_w1 + 8);
            int8x8_t w2_1 = vld1_s8(ptr_w2 + 8);
            int8x8_t w3_1 = vld1_s8(ptr_w3 + 8);

            din1 = vbsl_s8(vmask2, din1, vzero);

            int8x8_t w4_1 = vld1_s8(ptr_w4 + 8);
            int8x8_t w5_1 = vld1_s8(ptr_w5 + 8);
            int8x8_t w6_1 = vld1_s8(ptr_w6 + 8);
            int8x8_t w7_1 = vld1_s8(ptr_w7 + 8);

            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum1_16 = vmlal_s8(sum1_16, din1, w1_1);
            sum2_16 = vmlal_s8(sum2_16, din1, w2_1);
            sum3_16 = vmlal_s8(sum3_16, din1, w3_1);
            sum4_16 = vmlal_s8(sum4_16, din1, w4_1);
            sum5_16 = vmlal_s8(sum5_16, din1, w5_1);
            sum6_16 = vmlal_s8(sum6_16, din1, w6_1);
            sum7_16 = vmlal_s8(sum7_16, din1, w7_1);

            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_low_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_low_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_low_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_low_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_low_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_low_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_low_s16(sum7_16));

            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
            sum1 = vaddw_s16(sum1, vget_high_s16(sum1_16));
            sum2 = vaddw_s16(sum2, vget_high_s16(sum2_16));
            sum3 = vaddw_s16(sum3, vget_high_s16(sum3_16));
            sum4 = vaddw_s16(sum4, vget_high_s16(sum4_16));
            sum5 = vaddw_s16(sum5, vget_high_s16(sum5_16));
            sum6 = vaddw_s16(sum6, vget_high_s16(sum6_16));
            sum7 = vaddw_s16(sum7, vget_high_s16(sum7_16));
        }
        int32x4_t bias_val0 = vld1q_s32(bias_ptr);
        int32x4_t bias_val1 = vld1q_s32(bias_ptr + 4);
        //add
        int32x4_t vout0 = vpaddq_s32(sum0, sum1);
        int32x4_t vout1 = vpaddq_s32(sum2, sum3);
        int32x4_t vout2 = vpaddq_s32(sum4, sum5);
        int32x4_t vout3 = vpaddq_s32(sum6, sum7);

        int32x4_t vdout = vpaddq_s32(vout0, vout1);
        int32x4_t vdout1 = vpaddq_s32(vout2, vout3);

        vdout = vaddq_s32(vdout, bias_val0);
        vdout1 = vaddq_s32(vdout1, bias_val1);

        vdout = vmaxq_s32(vdout, vzero_32);
        vdout1 = vmaxq_s32(vdout1, vzero_32);

        vst1q_s32(ptr_out, vdout);
        vst1q_s32(ptr_out + 4, vdout1);
    }
    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 8; j < M; ++j){
        int *ptr_out = data_out + j;
        const int *bias_ptr = bias + j;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;

         asm volatile(
            "prfm  pldl1keep, [%[in]]   \n"
                    "prfm  pldl1keep, [%[w0]]   \n"
                     :
        :[in] "r"(ptr_in), [w0] "r"(ptr_w0)
        :"memory"
        );

        int32x4_t sum0 = vdupq_n_s32(0);
        for (int i = 0; i < cnt_loop; i++){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);//a = 01234567

            int16x8_t sum0_16 = vmull_s8(din0, w0_0);

            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);//a = 01234567

            ptr_in += 16;
            ptr_w0 += 16;

            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
        }
        if (tail > 0){
            int8x8_t din0 = vld1_s8(ptr_in);//a = 01234567
            int8x8_t din1 = vld1_s8(ptr_in + 8);//a = 01234567
            int8x8_t w0_0 = vld1_s8(ptr_w0);//a = 01234567
            int8x8_t w0_1 = vld1_s8(ptr_w0 + 8);//a = 01234567

            din0 = vbsl_s8(vmask1, din0, vzero);
            din1 = vbsl_s8(vmask2, din1, vzero);
            int16x8_t sum0_16 = vmull_s8(din0, w0_0);
            sum0_16 = vmlal_s8(sum0_16, din1, w0_1);
            sum0 = vaddw_s16(sum0, vget_low_s16(sum0_16));
            sum0 = vaddw_s16(sum0, vget_high_s16(sum0_16));
        }
        //add
        int32x2_t vout = vpadd_s32(vget_low_s32(sum0), vget_high_s32(sum0));
        int32x2_t vdout = vpadd_s32(vout, vout);
        int tmp = vget_lane_s32(vdout, 0) + bias_ptr[0];
        *ptr_out = tmp > 0 ? tmp : 0;
    }
#else
    int cnt_loop = N >> 4;
    int tail = N & 15;
    int out_cnt = M >> 2;

    unsigned char imask[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    uint8x8_t vmask1 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask));
    uint8x8_t vmask2 = vcgt_u8(vdup_n_u8(tail), vld1_u8(imask + 8));

    // printf("cnt_loop: %d, tail: %d, out_cnt: %d \n", cnt_loop, tail, out_cnt);

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {
        int out_idx = j * 4;
        int *ptr_out = data_out + out_idx;
        const int* bias_ptr = bias + out_idx;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * out_idx);
        const signed char *ptr_w1 = ptr_w0 + N;
        const signed char *ptr_w2 = ptr_w1 + N;
        const signed char *ptr_w3 = ptr_w2 + N;

        int cnt = cnt_loop;
        asm volatile(
        "pld [%[in]] @ preload cache line, input\n"
                "pld [%[w0]]                    @ preload cache line, weights r0\n"
                "pld [%[w1]]                    @ preload cache line, weights r1\n"
                "pld [%[w2]]                    @ preload cache line, weights r2\n"
                "pld [%[w3]]                    @ preload cache line, weights r3\n"

                "vmov.u32 q0, #0                @ set q0 to 0\n"
                "vmov.u32 q1, #0                @ set q1 to 0\n"
                "vmov.u32 q2, #0                @ set q2 to 0\n"
                "vmov.u32 q3, #0                @ set q3 to 0\n"

                "cmp %[cnt], #1                 @ check whether has main loop\n"
                "blt  3f                        @ jump to tail\n"

                "1:                             @ main loop\n"
                "vld1.8 {d8-d9}, [%[in]]!    @ load input, q4, q5\n"
                "vld1.8 {d12-d13}, [%[w0]]!    @ load weights r0, q6\n"
                "vld1.8 {d14-d15}, [%[w1]]!    @ load weights r1, q7\n"
                "vld1.8 {d16-d17}, [%[w2]]!    @ load weights r2, q8\n"
                "vld1.8 {d18-d19}, [%[w3]]!    @ load weights r3, q9\n"

                "vmull.s8 q12, d8, d12            @ mul add\n"
                "vmull.s8 q13, d8, d14            @ mul add\n"
                "vmull.s8 q14, d8, d16            @ mul add\n"
                "vmull.s8 q15, d8, d18            @ mul add\n"

                "pld [%[in]]                    @ preload cache line, input\n"
                "pld [%[w0]]                    @ preload cache line, weights r0\n"

                "vmlal.s8 q12, d9, d13            @ mul add\n"
                "vmlal.s8 q13, d9, d15            @ mul add\n"
                "vmlal.s8 q14, d9, d17           @ mul add\n"
                "vmlal.s8 q15, d9, d19           @ mul add\n"

                "pld [%[w1]]                    @ preload cache line, weights r1\n"
                "pld [%[w2]]                    @ preload cache line, weights r2\n"
                "pld [%[w3]]                    @ preload cache line, weights r3\n"

                "vaddw.s16 q0, q0, d24                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d26                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d28                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d30                 @addw \n" // out1 += vget_low_s16(out10)

                "vaddw.s16 q0, q0, d25                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d27                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d29                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d31                 @addw \n" // out1 += vget_low_s16(out10)

                // check loop end
                "subs %[cnt], #1                @ sub loop count \n"

                "bne 1b                         @ jump to main loop\n"

                // check tails
                "3:                             @ check tail\n"
                "cmp %[tail], #1                @ check whether has mid cols\n"
                "blt  2f                        @ jump to end\n"

                // process tail
                "vld1.8 {d8-d9}, [%[in]]!    @ load input, q4, q5\n"
                // deal with right pad
                "vmov.u32 d10, #0                @ dump q8 to zero, for bit select in tail\n"
                "vld1.8 {d12-d13}, [%[w0]]!    @ load weights r0, q6\n"
                "vld1.8 {d14-d15}, [%[w1]]!    @ load weights r1, q7\n"
                "vld1.8 {d16-d17}, [%[w2]]!    @ load weights r2, q8\n"
                "vld1.8 {d18-d19}, [%[w3]]!    @ load weights r3, q9\n"

                "vbif d8, d10, %[mask1]         @ bit select, deal with right pad\n"
                "vbif d9, d10, %[mask2]         @ bit select, deal with right pad\n"
                "pld [%[in]]                    @ preload cache line, input\n"

                "vmull.s8 q12, d8, d12            @ mul add\n"
                "vmull.s8 q13, d8, d14            @ mul add\n"
                "vmull.s8 q14, d8, d16            @ mul add\n"
                "vmull.s8 q15, d8, d18            @ mul add\n"

                "pld [%[w0]]                    @ preload cache line, weights r1\n"
                "pld [%[w1]]                    @ preload cache line, weights r1\n"

                "vmlal.s8 q12, d9, d13            @ mul add\n"
                "vmlal.s8 q13, d9, d15            @ mul add\n"
                "vmlal.s8 q14, d9, d17           @ mul add\n"
                "vmlal.s8 q15, d9, d19           @ mul add\n"

                "pld [%[w2]]                    @ preload cache line, weights r2\n"
                "pld [%[w3]]                    @ preload cache line, weights r3\n"

                "vaddw.s16 q0, q0, d24                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d26                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d28                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d30                 @addw \n" // out1 += vget_low_s16(out10)

                "vaddw.s16 q0, q0, d25                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q1, q1, d27                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q2, q2, d29                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q3, q3, d31                 @addw \n" // out1 += vget_low_s16(out10)

                // pair add to final result
                "2:                             @ end processing\n"
                "vld1.32 {d12-d13}, [%[bias]]   @ load weights r0, q6,q7\n"
                "vpadd.s32 d8, d0, d1           @ pair add, first step\n"
                "vpadd.s32 d9, d2, d3           @ pair add, first step\n"
                "vpadd.s32 d10, d4, d5          @ pair add, first step\n"
                "vpadd.s32 d11, d6, d7          @ pair add, first step\n"

                "vpadd.s32 d0, d8, d9           @ pair add, second step\n"
                "vpadd.s32 d1, d10, d11         @ pair add, second step\n"

                "vmov.u32 d12, #0                @ set q0 to 0\n"

                "vadd.s32 q0, q0, q6           @ add \n"

                "vmax.s32 d0, d0, d12            @ relu \n"
                "vmax.s32 d1, d1, d12            @ relu \n"

                "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

        :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1), \
                 [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [out] "+r"(ptr_out), \
                 [cnt] "+r"(cnt)
        :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2), [bias] "r" (bias_ptr)
        :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", \
                "q12", "q13", "q14", "q15"
        );
    }

    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 4; j < M; j++) {
        int *ptr_out = data_out + j;
        const int *bias_ptr = bias + j;
        const signed char *ptr_in = data_in;
        const signed char *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;
        asm volatile(
        "pld [%[in]] @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"
                "vmov.u32 q0, #0                    @ set q0 to 0\n"
                "cmp %[cnt], #1                     @ check whether has main loop\n"
                "blt  3f                            @ jump to tail\n"

                "1:                                 @ main loop\n"
                "vld1.8 {d24-d25}, [%[in]]!    @ load input, q12\n"
                "vld1.8 {d28-d29}, [%[w0]]!    @ load weights 14\n"

                "pld [%[in]]                        @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"

                "vmull.s8 q10, d24, d28            @ mul add\n"

                "vmlal.s8 q10, d25, d29             @ mul add\n"
                "subs %[cnt] , #1                   @ sub loop count \n"
                "vaddw.s16 q0, q0, d20                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q0, q0, d21                 @addw \n" // out1 += vget_low_s16(out10)
                "bne 1b                             @ jump to main loop\n"

                // check tails
                "3:                                 @ tail\n"
                "cmp %[tail], #1                    @ check whether has mid cols\n"
                "blt  2f                            @ jump to end\n"

                // process tail
                "vld1.8 {d24-d25}, [%[in]]!        @ load input, q12,q13\n"
                // deal with right pad
                "vmov.u32 d2, #0                    @ dump q8 to zero, for bit select in tail\n"
                "vld1.8 {d28-d29}, [%[w0]]!    @ load weights 14\n"

                "vbif d24, d2, %[mask1]            @ bit select, deal with right pad\n"
                "vbif d25, d2, %[mask2]            @ bit select, deal with right pad\n"

                "vmull.s8 q10, d24, d28            @ mul add\n"
                "vmlal.s8 q10, d25, d29             @ mul add\n"

                "vaddw.s16 q0, q0, d20                 @addw \n" // out1 += vget_low_s16(out10)
                "vaddw.s16 q0, q0, d21                 @addw \n" // out1 += vget_low_s16(out10)
                // pair add to final result
                "2:                                 @ end processing\n"
                "vld1.32 {d12}, [%[bias]]   @ load weights r0, q6,q7\n"
                "vpadd.s32 d2, d0, d1               @ pair add, first step\n"

                "vpadd.s32 d3, d2, d2               @ pair add, final step\n"
                "vmov.u32 d5, #0                    @ dump q8 to zero, for bit select in tail\n"

                "vadd.s32 d3, d3, d12           @ add \n"
                "vmax.s32 d3, d3, d5            @ relu \n"

                "vst1.32 {d3[0]}, [%[out]]          @ save result\n"
        :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [out] "+r"(ptr_out), \
            [cnt] "+r"(cnt)
        :[tail] "r"(tail), [mask1] "w"(vmask1), [mask2] "w"(vmask2), [bias] "r" (bias_ptr)
        :"q0", "q1", "q10", "q12", "q13", "q14", "q15"
        );
    }
#endif //__aarch64__
}

} //lite

} //saber

} //namespace anakin

#endif //USE_ARM_PLACE
