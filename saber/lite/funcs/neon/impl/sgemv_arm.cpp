#include "saber/lite/funcs/neon/impl/sgemv_arm.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

void sgemv(const bool transA, const int M, const int N, \
    const float* A, const float* x, float* y) {
    float* data_out = y;
    const float* data_in = x;
    const float* weights_ptr = A;

    int cnt_loop = N >> 3;
    int tail = N & 7;
    int out_cnt = M >> 2;

    unsigned int imask[8] = {7, 6, 5, 4, 3, 2, 1, 0};

    uint32x4_t vmask1 = vcgtq_u32(vld1q_u32(imask), vdupq_n_u32(tail));
    uint32x4_t vmask2 = vcgtq_u32(vld1q_u32(imask + 4), vdupq_n_u32(tail));

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {

        int out_idx = j * 4;
        float *ptr_out = data_out + out_idx;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * out_idx);
        const float *ptr_w1 = ptr_w0 + N;
        const float *ptr_w2 = ptr_w1 + N;
        const float *ptr_w3 = ptr_w2 + N;

        int cnt = cnt_loop;
        if (cnt > 0) {
            asm volatile(
            "pld [%[in]] @ preload cache line, input\n"
                    "pld [%[in]]                    @ preload cache line, input\n"
                    "pld [%[w0]]                    @ preload cache line, weights r0\n"
                    "pld [%[w1]]                    @ preload cache line, weights r1\n"
                    "pld [%[w2]]                    @ preload cache line, weights r2\n"
                    "pld [%[w3]]                    @ preload cache line, weights r3\n"

                    "vmov.u32 q0, #0                @ set q0 to 0\n"
                    "vmov.u32 q1, #0                @ set q1 to 0\n"
                    "vmov.u32 q2, #0                @ set q2 to 0\n"
                    "vmov.u32 q3, #0                @ set q3 to 0\n"

                    "pld [%[w0], #128]              @ preload cache line, weights r0\n"
                    "pld [%[w1], #128]              @ preload cache line, weights r1\n"
                    "pld [%[w2], #128]              @ preload cache line, weights r2\n"
                    "pld [%[w3], #128]              @ preload cache line, weights r3\n"

                    "1:                             @ main loop\n"
                    "vld1.32 {d8-d11}, [%[in]]!     @ load input, q4, q5\n"
                    //"pld [%[in]]                    @ preload cache line, in\n"
                    //"pld [%[in], #128]              @ preload cache line, in\n"

                    "vld1.32 {d12-d15}, [%[w0]]!    @ load weights r0, q6,q7\n"
                    "pld [%[w0]]                    @ preload cache line, weights r0\n"
                    "pld [%[w0], #128]              @ preload cache line, weights r0\n"

                    "vld1.32 {d16-d19}, [%[w1]]!    @ load weights r1, q8,q9\n"
                    "pld [%[w1]]                    @ preload cache line, weights r1\n"
                    "pld [%[w1], #128]              @ preload cache line, weights r1\n"

                    "vld1.32 {d20-d23}, [%[w2]]!    @ load weights r2, q10,q11\n"
                    "pld [%[w2]]                    @ preload cache line, weights r2\n"
                    "pld [%[w2], #128]              @ preload cache line, weights r2\n"

                    "vld1.32 {d24-d27}, [%[w3]]!    @ load weights r3, q12,q13\n"
                    "pld [%[w3]]                    @ preload cache line, weights r3\n"
                    "pld [%[w3], #128]              @ preload cache line, weights r3\n"

                    "vmla.f32 q0, q4, q6            @ mul add\n"
                    "vmla.f32 q1, q4, q8            @ mul add\n"
                    "vmla.f32 q2, q4, q10           @ mul add\n"
                    "vmla.f32 q3, q4, q12           @ mul add\n"

                    "vmla.f32 q0, q5, q7            @ mul add\n"
                    "vmla.f32 q1, q5, q9            @ mul add\n"
                    "vmla.f32 q2, q5, q11           @ mul add\n"
                    "vmla.f32 q3, q5, q13           @ mul add\n"

                    // check loop end
                    "subs %[cnt], #1                @ sub loop count \n"
                    "bne 1b                         @ jump to main loop\n"

                    // check tails
                    "cmp %[tail], #1                @ check whether has mid cols\n"
                    "blt  2f                        @ jump to end\n"

                    // process tail
                    "vld1.32 {d8-d11}, [%[in]]!     @ load input, q4, q5\n"
                    // deal with right pad
                    "vmov.u32 q6, #0                @ dump q8 to zero, for bit select in tail\n"
                    "vbif q4, q6, %q[mask1]         @ bit select, deal with right pad\n"
                    "vbif q5, q6, %q[mask2]         @ bit select, deal with right pad\n"

                    "vld1.32 {d12-d15}, [%[w0]]!    @ load weights r0, q6,q7\n"
                    "vld1.32 {d16-d19}, [%[w1]]!    @ load weights r1, q8,q9\n"
                    "vld1.32 {d20-d23}, [%[w2]]!    @ load weights r2, q10,q11\n"
                    "vld1.32 {d24-d27}, [%[w3]]!    @ load weights r3, q12, q13\n"

                    "vmla.f32 q0, q4, q6            @ mul add\n"
                    "vmla.f32 q1, q4, q8            @ mul add\n"
                    "vmla.f32 q2, q4, q10           @ mul add\n"
                    "vmla.f32 q3, q4, q12           @ mul add\n"

                    "vmla.f32 q0, q5, q7            @ mul add\n"
                    "vmla.f32 q1, q5, q9            @ mul add\n"
                    "vmla.f32 q2, q5, q11           @ mul add\n"
                    "vmla.f32 q3, q5, q13           @ mul add\n"

                    // pair add to final result
                    "2:                             @ end processing\n"
                    "vpadd.f32 d8, d0, d1           @ pair add, first step\n"
                    "vpadd.f32 d9, d2, d3           @ pair add, first step\n"
                    "vpadd.f32 d10, d4, d5          @ pair add, first step\n"
                    "vpadd.f32 d11, d6, d7          @ pair add, first step\n"

                    "vpadd.f32 d0, d8, d9           @ pair add, second step\n"
                    "vpadd.f32 d1, d10, d11         @ pair add, second step\n"

                    "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

            :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1), \
                 [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [out] "+r"(ptr_out), \
                 [cnt] "+r"(cnt)
            :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2)
            :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", \
                "q10", "q11", "q12", "q13"
            );
        }
    }

    //! deal with remains
    #pragma omp parallel for
    for (int j = out_cnt * 4; j < M; ++j) {
        float *ptr_out = data_out + j;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;
        asm volatile(
        "pld [%[in]] @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"
                "vmov.u32 q0, #0                    @ set q0 to 0\n"
                "pld [%[in], #128]                  @ preload cache line, input\n"
                "pld [%[w0], #128]                  @ preload cache line, weights r0\n"

                "1:                                 @ main loop\n"
                "vld1.32 {d24-d25}, [%[in]]!        @ load input, q12,q13\n"
                "vld1.32 {d28-d29}, [%[w0]]!        @ load weights r0, q14\n"

                "pld [%[in]]                        @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"

                "vmla.f32 q0, q12, q14              @ mul add\n"

                "vld1.32 {d26-d27}, [%[in]]!        @ load input, q12,q13\n"
                "vld1.32 {d30-d31}, [%[w0]]!        @ load weights r0, q14\n"
                "pld [%[in]]                        @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"

                "vmla.f32 q0, q13, q15              @ mul add\n"
                "subs %[cnt] , #1                   @ sub loop count \n"
                "bne 1b                             @ jump to main loop\n"

                // check tails
                "cmp %[tail], #1                    @ check whether has mid cols\n"
                "blt  2f                            @ jump to end\n"

                // process tail
                "vld1.32 {d24-d27}, [%[in]]!        @ load input, q12,q13\n"
                // deal with right pad
                "vmov.u32 q1, #0                    @ dump q8 to zero, for bit select in tail\n"
                "vbif q12, q1, %q[mask1]            @ bit select, deal with right pad\n"
                "vbif q13, q1, %q[mask2]            @ bit select, deal with right pad\n"

                "vld1.32 {d28-d29}, [%[w0]]!        @ load weights r0, q14\n"
                "vmla.f32 q0, q12, q14              @ mul add\n"
                "vld1.32 {d30-d31}, [%[w0]]!        @ load weights r0, q15\n"
                "vmla.f32 q0, q13, q15              @ mul add\n"

                // pair add to final result
                "2:                                 @ end processing\n"
                "vpadd.f32 d2, d0, d1               @ pair add, first step\n"
                "vpadd.f32 d3, d2, d2               @ pair add, final step\n"
                "vst1.32 {d3[0]}, [%[out]]          @ save result\n"
        :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [out] "+r"(ptr_out), \
            [cnt] "+r"(cnt)
        :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2)
        :"q0", "q1", "q12", "q13", "q14", "q15"
        );
    }
}


void sgemv_relu(const bool transA, const int M, const int N, \
    const float* A, const float* x, float* y) {
    float* data_out = y;
    const float* data_in = x;
    const float* weights_ptr = A;

    int cnt_loop = N >> 3;
    int tail = N & 7;
    int out_cnt = M >> 2;

    unsigned int imask[8] = {7, 6, 5, 4, 3, 2, 1, 0};

    uint32x4_t vmask1 = vcgtq_u32(vld1q_u32(imask), vdupq_n_u32(tail));
    uint32x4_t vmask2 = vcgtq_u32(vld1q_u32(imask + 4), vdupq_n_u32(tail));

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {

        int out_idx = j * 4;
        float *ptr_out = data_out + out_idx;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * out_idx);
        const float *ptr_w1 = ptr_w0 + N;
        const float *ptr_w2 = ptr_w1 + N;
        const float *ptr_w3 = ptr_w2 + N;

        int cnt = cnt_loop;
        if (cnt > 0) {
            asm volatile(
            "pld [%[in]] @ preload cache line, input\n"
                    "pld [%[in]]                    @ preload cache line, input\n"
                    "pld [%[w0]]                    @ preload cache line, weights r0\n"
                    "pld [%[w1]]                    @ preload cache line, weights r1\n"
                    "pld [%[w2]]                    @ preload cache line, weights r2\n"
                    "pld [%[w3]]                    @ preload cache line, weights r3\n"

                    "vmov.u32 q0, #0                @ set q0 to 0\n"
                    "vmov.u32 q1, #0                @ set q1 to 0\n"
                    "vmov.u32 q2, #0                @ set q2 to 0\n"
                    "vmov.u32 q3, #0                @ set q3 to 0\n"

                    "pld [%[w0], #128]              @ preload cache line, weights r0\n"
                    "pld [%[w1], #128]              @ preload cache line, weights r1\n"
                    "pld [%[w2], #128]              @ preload cache line, weights r2\n"
                    "pld [%[w3], #128]              @ preload cache line, weights r3\n"

                    "1:                             @ main loop\n"
                    "vld1.32 {d8-d11}, [%[in]]!     @ load input, q4, q5\n"
                    //"pld [%[in]]                    @ preload cache line, in\n"
                    //"pld [%[in], #128]              @ preload cache line, in\n"

                    "vld1.32 {d12-d15}, [%[w0]]!    @ load weights r0, q6,q7\n"
                    "pld [%[w0]]                    @ preload cache line, weights r0\n"
                    "pld [%[w0], #128]              @ preload cache line, weights r0\n"

                    "vld1.32 {d16-d19}, [%[w1]]!    @ load weights r1, q8,q9\n"
                    "pld [%[w1]]                    @ preload cache line, weights r1\n"
                    "pld [%[w1], #128]              @ preload cache line, weights r1\n"

                    "vld1.32 {d20-d23}, [%[w2]]!    @ load weights r2, q10,q11\n"
                    "pld [%[w2]]                    @ preload cache line, weights r2\n"
                    "pld [%[w2], #128]              @ preload cache line, weights r2\n"

                    "vld1.32 {d24-d27}, [%[w3]]!    @ load weights r3, q12,q13\n"
                    "pld [%[w3]]                    @ preload cache line, weights r3\n"
                    "pld [%[w3], #128]              @ preload cache line, weights r3\n"

                    "vmla.f32 q0, q4, q6            @ mul add\n"
                    "vmla.f32 q1, q4, q8            @ mul add\n"
                    "vmla.f32 q2, q4, q10           @ mul add\n"
                    "vmla.f32 q3, q4, q12           @ mul add\n"

                    "vmla.f32 q0, q5, q7            @ mul add\n"
                    "vmla.f32 q1, q5, q9            @ mul add\n"
                    "vmla.f32 q2, q5, q11           @ mul add\n"
                    "vmla.f32 q3, q5, q13           @ mul add\n"

                    // check loop end
                    "subs %[cnt], #1                @ sub loop count \n"
                    "bne 1b                         @ jump to main loop\n"

                    // check tails
                    "cmp %[tail], #1                @ check whether has mid cols\n"
                    "blt  2f                        @ jump to end\n"

                    // process tail
                    "vld1.32 {d8-d11}, [%[in]]!     @ load input, q4, q5\n"
                    // deal with right pad
                    "vmov.u32 q6, #0                @ dump q8 to zero, for bit select in tail\n"
                    "vbif q4, q6, %q[mask1]         @ bit select, deal with right pad\n"
                    "vbif q5, q6, %q[mask2]         @ bit select, deal with right pad\n"

                    "vld1.32 {d12-d15}, [%[w0]]!    @ load weights r0, q6,q7\n"
                    "vld1.32 {d16-d19}, [%[w1]]!    @ load weights r1, q8,q9\n"
                    "vld1.32 {d20-d23}, [%[w2]]!    @ load weights r2, q10,q11\n"
                    "vld1.32 {d24-d27}, [%[w3]]!    @ load weights r3, q12, q13\n"

                    "vmla.f32 q0, q4, q6            @ mul add\n"
                    "vmla.f32 q1, q4, q8            @ mul add\n"
                    "vmla.f32 q2, q4, q10           @ mul add\n"
                    "vmla.f32 q3, q4, q12           @ mul add\n"

                    "vmla.f32 q0, q5, q7            @ mul add\n"
                    "vmla.f32 q1, q5, q9            @ mul add\n"
                    "vmla.f32 q2, q5, q11           @ mul add\n"
                    "vmla.f32 q3, q5, q13           @ mul add\n"

                    // pair add to final result
                    "2:                             @ end processing\n"
                    "vpadd.f32 d8, d0, d1           @ pair add, first step\n"
                    "vpadd.f32 d9, d2, d3           @ pair add, first step\n"
                    "vpadd.f32 d10, d4, d5          @ pair add, first step\n"
                    "vpadd.f32 d11, d6, d7          @ pair add, first step\n"

                    "vpadd.f32 d0, d8, d9           @ pair add, second step\n"
                    "vpadd.f32 d1, d10, d11         @ pair add, second step\n"

                    "vmov.u32   q1, #0              @ set q1 to zero, for relu\n"
                    "vmax.f32   q2, q0, q1          @ relu\n"

                    "vst1.32 {d4-d5}, [%[out]]      @ save result\n"

            :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1), \
                 [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [out] "+r"(ptr_out), \
                 [cnt] "+r"(cnt)
            :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2)
            :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", \
                "q10", "q11", "q12", "q13"
            );
        }
    }

    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 4; j < M; ++j) {
        float *ptr_out = data_out + j;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;
        asm volatile(
        "pld [%[in]] @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"
                "vmov.u32 q0, #0                    @ set q0 to 0\n"
                "pld [%[in], #128]                  @ preload cache line, input\n"
                "pld [%[w0], #128]                  @ preload cache line, weights r0\n"

                "1:                                 @ main loop\n"
                "vld1.32 {d24-d25}, [%[in]]!        @ load input, q12,q13\n"
                "vld1.32 {d28-d29}, [%[w0]]!        @ load weights r0, q14\n"

                "pld [%[in]]                        @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"

                "vmla.f32 q0, q12, q14              @ mul add\n"

                "vld1.32 {d26-d27}, [%[in]]!        @ load input, q12,q13\n"
                "vld1.32 {d30-d31}, [%[w0]]!        @ load weights r0, q14\n"
                "pld [%[in]]                        @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"

                "vmla.f32 q0, q13, q15              @ mul add\n"
                "subs %[cnt] , #1                   @ sub loop count \n"
                "bne 1b                             @ jump to main loop\n"

                // check tails
                "cmp %[tail], #1                    @ check whether has mid cols\n"
                "blt  2f                            @ jump to end\n"

                // process tail
                "vld1.32 {d24-d27}, [%[in]]!        @ load input, q12,q13\n"
                // deal with right pad
                "vmov.u32 q1, #0                    @ dump q8 to zero, for bit select in tail\n"
                "vbif q12, q1, %q[mask1]            @ bit select, deal with right pad\n"
                "vbif q13, q1, %q[mask2]            @ bit select, deal with right pad\n"

                "vld1.32 {d28-d29}, [%[w0]]!        @ load weights r0, q14\n"
                "vmla.f32 q0, q12, q14              @ mul add\n"
                "vld1.32 {d30-d31}, [%[w0]]!        @ load weights r0, q15\n"
                "vmla.f32 q0, q13, q15              @ mul add\n"

                // pair add to final result
                "2:                                 @ end processing\n"
                "vpadd.f32 d2, d0, d1               @ pair add, first step\n"
                "vpadd.f32 d3, d2, d2               @ pair add, final step\n"

                "vmov.u32   d0, #0                  @ set q1 to zero, for relu\n"
                "vmax.f32   d1, d3, d0              @ relu\n"

                "vst1.32 {d1[0]}, [%[out]]          @ save result\n"

        :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [out] "+r"(ptr_out), \
            [cnt] "+r"(cnt)
        :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2)
        :"q0", "q1", "q12", "q13", "q14", "q15"
        );
    }
}

void sgemv_bias(const bool transA, const int M, const int N, \
    const float* A, const float* x, float* y, const float* bias) {
    float* data_out = y;
    const float* data_in = x;
    const float* weights_ptr = A;

    int cnt_loop = N >> 3;
    int tail = N & 7;
    int out_cnt = M >> 2;

    unsigned int imask[8] = {7, 6, 5, 4, 3, 2, 1, 0};

    uint32x4_t vmask1 = vcgtq_u32(vld1q_u32(imask), vdupq_n_u32(tail));
    uint32x4_t vmask2 = vcgtq_u32(vld1q_u32(imask + 4), vdupq_n_u32(tail));

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {

        int out_idx = j * 4;
        float *ptr_out = data_out + out_idx;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * out_idx);
        const float *ptr_w1 = ptr_w0 + N;
        const float *ptr_w2 = ptr_w1 + N;
        const float *ptr_w3 = ptr_w2 + N;

        const float* ptr_bias = bias + out_idx;

        int cnt = cnt_loop;
        if (cnt > 0) {
            asm volatile(
            "pld [%[in]] @ preload cache line, input\n"
                    "pld [%[in]]                    @ preload cache line, input\n"
                    "pld [%[w0]]                    @ preload cache line, weights r0\n"
                    "pld [%[w1]]                    @ preload cache line, weights r1\n"
                    "pld [%[w2]]                    @ preload cache line, weights r2\n"
                    "pld [%[w3]]                    @ preload cache line, weights r3\n"

                    "vmov.u32 q0, #0                @ set q0 to 0\n"
                    "vmov.u32 q1, #0                @ set q1 to 0\n"
                    "vmov.u32 q2, #0                @ set q2 to 0\n"
                    "vmov.u32 q3, #0                @ set q3 to 0\n"

                    "pld [%[w0], #128]              @ preload cache line, weights r0\n"
                    "pld [%[w1], #128]              @ preload cache line, weights r1\n"
                    "pld [%[w2], #128]              @ preload cache line, weights r2\n"
                    "pld [%[w3], #128]              @ preload cache line, weights r3\n"

                    "1:                             @ main loop\n"
                    "vld1.32 {d8-d11}, [%[in]]!     @ load input, q4, q5\n"
                    //"pld [%[in]]                    @ preload cache line, in\n"
                    //"pld [%[in], #128]              @ preload cache line, in\n"

                    "vld1.32 {d12-d15}, [%[w0]]!    @ load weights r0, q6,q7\n"
                    "pld [%[w0]]                    @ preload cache line, weights r0\n"
                    "pld [%[w0], #128]              @ preload cache line, weights r0\n"

                    "vld1.32 {d16-d19}, [%[w1]]!    @ load weights r1, q8,q9\n"
                    "pld [%[w1]]                    @ preload cache line, weights r1\n"
                    "pld [%[w1], #128]              @ preload cache line, weights r1\n"

                    "vld1.32 {d20-d23}, [%[w2]]!    @ load weights r2, q10,q11\n"
                    "pld [%[w2]]                    @ preload cache line, weights r2\n"
                    "pld [%[w2], #128]              @ preload cache line, weights r2\n"

                    "vld1.32 {d24-d27}, [%[w3]]!    @ load weights r3, q12,q13\n"
                    "pld [%[w3]]                    @ preload cache line, weights r3\n"
                    "pld [%[w3], #128]              @ preload cache line, weights r3\n"

                    "vmla.f32 q0, q4, q6            @ mul add\n"
                    "vmla.f32 q1, q4, q8            @ mul add\n"
                    "vmla.f32 q2, q4, q10           @ mul add\n"
                    "vmla.f32 q3, q4, q12           @ mul add\n"

                    "vmla.f32 q0, q5, q7            @ mul add\n"
                    "vmla.f32 q1, q5, q9            @ mul add\n"
                    "vmla.f32 q2, q5, q11           @ mul add\n"
                    "vmla.f32 q3, q5, q13           @ mul add\n"

                    // check loop end
                    "subs %[cnt], #1                @ sub loop count \n"
                    "bne 1b                         @ jump to main loop\n"

                    // check tails
                    "cmp %[tail], #1                @ check whether has mid cols\n"
                    "blt  2f                        @ jump to end\n"

                    // process tail
                    "vld1.32 {d8-d11}, [%[in]]!     @ load input, q4, q5\n"
                    // deal with right pad
                    "vmov.u32 q6, #0                @ dump q8 to zero, for bit select in tail\n"
                    "vbif q4, q6, %q[mask1]         @ bit select, deal with right pad\n"
                    "vbif q5, q6, %q[mask2]         @ bit select, deal with right pad\n"

                    "vld1.32 {d12-d15}, [%[w0]]!    @ load weights r0, q6,q7\n"
                    "vld1.32 {d16-d19}, [%[w1]]!    @ load weights r1, q8,q9\n"
                    "vld1.32 {d20-d23}, [%[w2]]!    @ load weights r2, q10,q11\n"
                    "vld1.32 {d24-d27}, [%[w3]]!    @ load weights r3, q12, q13\n"

                    "vmla.f32 q0, q4, q6            @ mul add\n"
                    "vmla.f32 q1, q4, q8            @ mul add\n"
                    "vmla.f32 q2, q4, q10           @ mul add\n"
                    "vmla.f32 q3, q4, q12           @ mul add\n"

                    "vmla.f32 q0, q5, q7            @ mul add\n"
                    "vmla.f32 q1, q5, q9            @ mul add\n"
                    "vmla.f32 q2, q5, q11           @ mul add\n"
                    "vmla.f32 q3, q5, q13           @ mul add\n"

                    // pair add to final result
                    "2:                             @ end processing\n"
                    "vld1.32 {d12-d13}, [%[bias]]   @ load weights r0, q6,q7\n"
                    "vpadd.f32 d8, d0, d1           @ pair add, first step\n"
                    "vpadd.f32 d9, d2, d3           @ pair add, first step\n"
                    "vpadd.f32 d10, d4, d5          @ pair add, first step\n"
                    "vpadd.f32 d11, d6, d7          @ pair add, first step\n"

                    "vpadd.f32 d0, d8, d9           @ pair add, second step\n"
                    "vpadd.f32 d1, d10, d11         @ pair add, second step\n"

                    "vadd.f32 q1, q0, q6                @ add bias\n"

                    "vst1.32 {d2-d3}, [%[out]]      @ save result\n"

            :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1), \
                 [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [out] "+r"(ptr_out), \
                 [cnt] "+r"(cnt)
            :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2), \
                 [bias] "r" (ptr_bias)
            :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", \
                "q10", "q11", "q12", "q13"
            );
        }
    }

    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 4; j < M; ++j) {
        float *ptr_out = data_out + j;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;
        float32x2_t vbias = vdup_n_f32(bias[j]);
        asm volatile(
        "pld [%[in]] @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"
                "vmov.u32 q0, #0                    @ set q0 to 0\n"
                "pld [%[in], #128]                  @ preload cache line, input\n"
                "pld [%[w0], #128]                  @ preload cache line, weights r0\n"

                "1:                                 @ main loop\n"
                "vld1.32 {d24-d25}, [%[in]]!        @ load input, q12,q13\n"
                "vld1.32 {d28-d29}, [%[w0]]!        @ load weights r0, q14\n"

                "pld [%[in]]                        @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"

                "vmla.f32 q0, q12, q14              @ mul add\n"

                "vld1.32 {d26-d27}, [%[in]]!        @ load input, q12,q13\n"
                "vld1.32 {d30-d31}, [%[w0]]!        @ load weights r0, q14\n"
                "pld [%[in]]                        @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"

                "vmla.f32 q0, q13, q15              @ mul add\n"
                "subs %[cnt] , #1                   @ sub loop count \n"
                "bne 1b                             @ jump to main loop\n"

                // check tails
                "cmp %[tail], #1                    @ check whether has mid cols\n"
                "blt  2f                            @ jump to end\n"

                // process tail
                "vld1.32 {d24-d27}, [%[in]]!        @ load input, q12,q13\n"
                // deal with right pad
                "vmov.u32 q1, #0                    @ dump q8 to zero, for bit select in tail\n"
                "vbif q12, q1, %q[mask1]            @ bit select, deal with right pad\n"
                "vbif q13, q1, %q[mask2]            @ bit select, deal with right pad\n"

                "vld1.32 {d28-d29}, [%[w0]]!        @ load weights r0, q14\n"
                "vmla.f32 q0, q12, q14              @ mul add\n"
                "vld1.32 {d30-d31}, [%[w0]]!        @ load weights r0, q15\n"
                "vmla.f32 q0, q13, q15              @ mul add\n"

                // pair add to final result
                "2:                                 @ end processing\n"
                "vpadd.f32 d2, d0, d1               @ pair add, first step\n"
                "vpadd.f32 d3, d2, d2               @ pair add, final step\n"

                "vadd.f32  d3, %P[bias]             @ add bias\n"

                "vst1.32 {d3[0]}, [%[out]]          @ save result\n"

        :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [out] "+r"(ptr_out), \
            [cnt] "+r"(cnt)
        :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2), \
            [bias] "w" (vbias)
        :"q0", "q1", "q12", "q13", "q14", "q15"
        );
    }
}


void sgemv_bias_relu(const bool transA, const int M, const int N, \
    const float* A, const float* x, float* y, const float* bias) {
    float* data_out = y;
    const float* data_in = x;
    const float* weights_ptr = A;

    int cnt_loop = N >> 3;
    int tail = N & 7;
    int out_cnt = M >> 2;

    unsigned int imask[8] = {7, 6, 5, 4, 3, 2, 1, 0};

    uint32x4_t vmask1 = vcgtq_u32(vld1q_u32(imask), vdupq_n_u32(tail));
    uint32x4_t vmask2 = vcgtq_u32(vld1q_u32(imask + 4), vdupq_n_u32(tail));

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {

        int out_idx = j * 4;
        float *ptr_out = data_out + out_idx;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * out_idx);
        const float *ptr_w1 = ptr_w0 + N;
        const float *ptr_w2 = ptr_w1 + N;
        const float *ptr_w3 = ptr_w2 + N;

        const float* ptr_bias = bias + out_idx;

        int cnt = cnt_loop;
        if (cnt > 0) {
            asm volatile(
            "pld [%[in]] @ preload cache line, input\n"
                    "pld [%[in]]                    @ preload cache line, input\n"
                    "pld [%[w0]]                    @ preload cache line, weights r0\n"
                    "pld [%[w1]]                    @ preload cache line, weights r1\n"
                    "pld [%[w2]]                    @ preload cache line, weights r2\n"
                    "pld [%[w3]]                    @ preload cache line, weights r3\n"

                    "vmov.u32 q0, #0                @ set q0 to 0\n"
                    "vmov.u32 q1, #0                @ set q1 to 0\n"
                    "vmov.u32 q2, #0                @ set q2 to 0\n"
                    "vmov.u32 q3, #0                @ set q3 to 0\n"

                    "pld [%[w0], #128]              @ preload cache line, weights r0\n"
                    "pld [%[w1], #128]              @ preload cache line, weights r1\n"
                    "pld [%[w2], #128]              @ preload cache line, weights r2\n"
                    "pld [%[w3], #128]              @ preload cache line, weights r3\n"

                    "1:                             @ main loop\n"
                    "vld1.32 {d8-d11}, [%[in]]!     @ load input, q4, q5\n"
                    //"pld [%[in]]                    @ preload cache line, in\n"
                    //"pld [%[in], #128]              @ preload cache line, in\n"

                    "vld1.32 {d12-d15}, [%[w0]]!    @ load weights r0, q6,q7\n"
                    "pld [%[w0]]                    @ preload cache line, weights r0\n"
                    "pld [%[w0], #128]              @ preload cache line, weights r0\n"

                    "vld1.32 {d16-d19}, [%[w1]]!    @ load weights r1, q8,q9\n"
                    "pld [%[w1]]                    @ preload cache line, weights r1\n"
                    "pld [%[w1], #128]              @ preload cache line, weights r1\n"

                    "vld1.32 {d20-d23}, [%[w2]]!    @ load weights r2, q10,q11\n"
                    "pld [%[w2]]                    @ preload cache line, weights r2\n"
                    "pld [%[w2], #128]              @ preload cache line, weights r2\n"

                    "vld1.32 {d24-d27}, [%[w3]]!    @ load weights r3, q12,q13\n"
                    "pld [%[w3]]                    @ preload cache line, weights r3\n"
                    "pld [%[w3], #128]              @ preload cache line, weights r3\n"

                    "vmla.f32 q0, q4, q6            @ mul add\n"
                    "vmla.f32 q1, q4, q8            @ mul add\n"
                    "vmla.f32 q2, q4, q10           @ mul add\n"
                    "vmla.f32 q3, q4, q12           @ mul add\n"

                    "vmla.f32 q0, q5, q7            @ mul add\n"
                    "vmla.f32 q1, q5, q9            @ mul add\n"
                    "vmla.f32 q2, q5, q11           @ mul add\n"
                    "vmla.f32 q3, q5, q13           @ mul add\n"

                    // check loop end
                    "subs %[cnt], #1                @ sub loop count \n"
                    "bne 1b                         @ jump to main loop\n"

                    // check tails
                    "cmp %[tail], #1                @ check whether has mid cols\n"
                    "blt  2f                        @ jump to end\n"

                    // process tail
                    "vld1.32 {d8-d11}, [%[in]]!     @ load input, q4, q5\n"
                    // deal with right pad
                    "vmov.u32 q6, #0                @ dump q8 to zero, for bit select in tail\n"
                    "vbif q4, q6, %q[mask1]         @ bit select, deal with right pad\n"
                    "vbif q5, q6, %q[mask2]         @ bit select, deal with right pad\n"

                    "vld1.32 {d12-d15}, [%[w0]]!    @ load weights r0, q6,q7\n"
                    "vld1.32 {d16-d19}, [%[w1]]!    @ load weights r1, q8,q9\n"
                    "vld1.32 {d20-d23}, [%[w2]]!    @ load weights r2, q10,q11\n"
                    "vld1.32 {d24-d27}, [%[w3]]!    @ load weights r3, q12, q13\n"

                    "vmla.f32 q0, q4, q6            @ mul add\n"
                    "vmla.f32 q1, q4, q8            @ mul add\n"
                    "vmla.f32 q2, q4, q10           @ mul add\n"
                    "vmla.f32 q3, q4, q12           @ mul add\n"

                    "vmla.f32 q0, q5, q7            @ mul add\n"
                    "vmla.f32 q1, q5, q9            @ mul add\n"
                    "vmla.f32 q2, q5, q11           @ mul add\n"
                    "vmla.f32 q3, q5, q13           @ mul add\n"

                    // pair add to final result
                    "2:                             @ end processing\n"
                    "vld1.32 {d12-d13}, [%[bias]]   @ load weights r0, q6,q7\n"
                    "vpadd.f32 d8, d0, d1           @ pair add, first step\n"
                    "vpadd.f32 d9, d2, d3           @ pair add, first step\n"
                    "vpadd.f32 d10, d4, d5          @ pair add, first step\n"
                    "vpadd.f32 d11, d6, d7          @ pair add, first step\n"

                    "vpadd.f32 d0, d8, d9           @ pair add, second step\n"
                    "vpadd.f32 d1, d10, d11         @ pair add, second step\n"

                    "vmov.u32 q2, #0                @ for relu\n"
                    "vadd.f32 q1, q0, q6            @ add bias\n"
                    "vmax.f32 q0, q1, q2            @ relu\n"

                    "vst1.32 {d0-d1}, [%[out]]      @ save result\n"

            :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1), \
                 [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [out] "+r"(ptr_out), \
                 [cnt] "+r"(cnt)
            :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2), \
                 [bias] "r" (ptr_bias)
            :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", \
                "q10", "q11", "q12", "q13"
            );
        }
    }

    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 4; j < M; ++j) {
        float *ptr_out = data_out + j;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;
        float32x2_t vbias = vdup_n_f32(bias[j]);
        asm volatile(
        "pld [%[in]] @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"
                "vmov.u32 q0, #0                    @ set q0 to 0\n"
                "pld [%[in], #128]                  @ preload cache line, input\n"
                "pld [%[w0], #128]                  @ preload cache line, weights r0\n"

                "1:                                 @ main loop\n"
                "vld1.32 {d24-d25}, [%[in]]!        @ load input, q12,q13\n"
                "vld1.32 {d28-d29}, [%[w0]]!        @ load weights r0, q14\n"

                "pld [%[in]]                        @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"

                "vmla.f32 q0, q12, q14              @ mul add\n"

                "vld1.32 {d26-d27}, [%[in]]!        @ load input, q12,q13\n"
                "vld1.32 {d30-d31}, [%[w0]]!        @ load weights r0, q14\n"
                "pld [%[in]]                        @ preload cache line, input\n"
                "pld [%[w0]]                        @ preload cache line, weights r0\n"

                "vmla.f32 q0, q13, q15              @ mul add\n"
                "subs %[cnt] , #1                   @ sub loop count \n"
                "bne 1b                             @ jump to main loop\n"

                // check tails
                "cmp %[tail], #1                    @ check whether has mid cols\n"
                "blt  2f                            @ jump to end\n"

                // process tail
                "vld1.32 {d24-d27}, [%[in]]!        @ load input, q12,q13\n"
                // deal with right pad
                "vmov.u32 q1, #0                    @ dump q8 to zero, for bit select in tail\n"
                "vbif q12, q1, %q[mask1]            @ bit select, deal with right pad\n"
                "vbif q13, q1, %q[mask2]            @ bit select, deal with right pad\n"

                "vld1.32 {d28-d29}, [%[w0]]!        @ load weights r0, q14\n"
                "vmla.f32 q0, q12, q14              @ mul add\n"
                "vld1.32 {d30-d31}, [%[w0]]!        @ load weights r0, q15\n"
                "vmla.f32 q0, q13, q15              @ mul add\n"

                // pair add to final result
                "2:                                 @ end processing\n"
                "vpadd.f32 d2, d0, d1               @ pair add, first step\n"
                "vpadd.f32 d3, d2, d2               @ pair add, final step\n"

                "vadd.f32  d3, %P[bias]             @ add bias\n"

                "vmov.u32  d2, #0                   @ for relu\n"
                "vmax.f32  d1, d2, d3               @ relu\n"

                "vst1.32 {d1[0]}, [%[out]]          @ save result\n"

        :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [out] "+r"(ptr_out), \
            [cnt] "+r"(cnt)
        :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2), \
            [bias] "w" (vbias)
        :"q0", "q1", "q12", "q13", "q14", "q15"
        );
    }
}

} //lite

} //saber

} //namespace anakin

#endif //USE_ARM_PLACE