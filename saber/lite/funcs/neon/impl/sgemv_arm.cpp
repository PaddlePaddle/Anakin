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

#ifdef __aarch64__
    int cnt_loop = N >> 3;
    int tail = N & 7;
    int out_cnt = M >> 3;

    unsigned int imask[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(imask), vdupq_n_u32(tail));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(imask + 4), vdupq_n_u32(tail));

    float32x4_t vzero = vdupq_n_f32(0.f);

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {

        int out_idx = j * 8;
        float *ptr_out = data_out + out_idx;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * out_idx);
        const float *ptr_w1 = ptr_w0 + N;
        const float *ptr_w2 = ptr_w1 + N;
        const float *ptr_w3 = ptr_w2 + N;
        const float *ptr_w4 = ptr_w3 + N;
        const float *ptr_w5 = ptr_w4 + N;
        const float *ptr_w6 = ptr_w5 + N;
        const float *ptr_w7 = ptr_w6 + N;

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
#if 1
        float32x4_t sum0 = vdupq_n_f32(0.f);
        float32x4_t sum1 = vdupq_n_f32(0.f);
        float32x4_t sum2 = vdupq_n_f32(0.f);
        float32x4_t sum3 = vdupq_n_f32(0.f);
        float32x4_t sum4 = vdupq_n_f32(0.f);
        float32x4_t sum5 = vdupq_n_f32(0.f);
        float32x4_t sum6 = vdupq_n_f32(0.f);
        float32x4_t sum7 = vdupq_n_f32(0.f);

        for (int i = 0; i < cnt_loop; i++){

            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);
            float32x4_t w1_0 = vld1q_f32(ptr_w1);
            float32x4_t w2_0 = vld1q_f32(ptr_w2);
            float32x4_t w3_0 = vld1q_f32(ptr_w3);
            float32x4_t w4_0 = vld1q_f32(ptr_w4);
            float32x4_t w5_0 = vld1q_f32(ptr_w5);
            float32x4_t w6_0 = vld1q_f32(ptr_w6);
            float32x4_t w7_0 = vld1q_f32(ptr_w7);

            sum0 = vmlaq_f32(sum0, din0, w0_0);
            sum1 = vmlaq_f32(sum1, din0, w1_0);
            sum2 = vmlaq_f32(sum2, din0, w2_0);
            sum3 = vmlaq_f32(sum3, din0, w3_0);
            sum4 = vmlaq_f32(sum4, din0, w4_0);
            sum5 = vmlaq_f32(sum5, din0, w5_0);
            sum6 = vmlaq_f32(sum6, din0, w6_0);
            sum7 = vmlaq_f32(sum7, din0, w7_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);
            float32x4_t w1_1 = vld1q_f32(ptr_w1 + 4);
            float32x4_t w2_1 = vld1q_f32(ptr_w2 + 4);
            float32x4_t w3_1 = vld1q_f32(ptr_w3 + 4);
            float32x4_t w4_1 = vld1q_f32(ptr_w4 + 4);
            float32x4_t w5_1 = vld1q_f32(ptr_w5 + 4);
            float32x4_t w6_1 = vld1q_f32(ptr_w6 + 4);
            float32x4_t w7_1 = vld1q_f32(ptr_w7 + 4);

            sum0 = vmlaq_f32(sum0, din1, w0_1);
            sum1 = vmlaq_f32(sum1, din1, w1_1);
            sum2 = vmlaq_f32(sum2, din1, w2_1);
            sum3 = vmlaq_f32(sum3, din1, w3_1);
            sum4 = vmlaq_f32(sum4, din1, w4_1);
            sum5 = vmlaq_f32(sum5, din1, w5_1);
            sum6 = vmlaq_f32(sum6, din1, w6_1);
            sum7 = vmlaq_f32(sum7, din1, w7_1);

            ptr_in += 8;
            ptr_w0 += 8;
            ptr_w1 += 8;
            ptr_w2 += 8;
            ptr_w3 += 8;
            ptr_w4 += 8;
            ptr_w5 += 8;
            ptr_w6 += 8;
            ptr_w7 += 8;

        }
        if (tail > 0){
            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);
            float32x4_t w1_0 = vld1q_f32(ptr_w1);
            float32x4_t w2_0 = vld1q_f32(ptr_w2);
            float32x4_t w3_0 = vld1q_f32(ptr_w3);

            din0 = vbslq_f32(vmask1, din0, vzero);

            float32x4_t w4_0 = vld1q_f32(ptr_w4);
            float32x4_t w5_0 = vld1q_f32(ptr_w5);
            float32x4_t w6_0 = vld1q_f32(ptr_w6);
            float32x4_t w7_0 = vld1q_f32(ptr_w7);

            sum0 = vmlaq_f32(sum0, din0, w0_0);
            sum1 = vmlaq_f32(sum1, din0, w1_0);
            sum2 = vmlaq_f32(sum2, din0, w2_0);
            sum3 = vmlaq_f32(sum3, din0, w3_0);
            sum4 = vmlaq_f32(sum4, din0, w4_0);
            sum5 = vmlaq_f32(sum5, din0, w5_0);
            sum6 = vmlaq_f32(sum6, din0, w6_0);
            sum7 = vmlaq_f32(sum7, din0, w7_0);

            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);
            float32x4_t w1_1 = vld1q_f32(ptr_w1 + 4);
            float32x4_t w2_1 = vld1q_f32(ptr_w2 + 4);

            din1 = vbslq_f32(vmask2, din1, vzero);

            float32x4_t w3_1 = vld1q_f32(ptr_w3 + 4);
            float32x4_t w4_1 = vld1q_f32(ptr_w4 + 4);
            float32x4_t w5_1 = vld1q_f32(ptr_w5 + 4);
            float32x4_t w6_1 = vld1q_f32(ptr_w6 + 4);
            float32x4_t w7_1 = vld1q_f32(ptr_w7 + 4);

            sum0 = vmlaq_f32(sum0, din1, w0_1);
            sum1 = vmlaq_f32(sum1, din1, w1_1);
            sum2 = vmlaq_f32(sum2, din1, w2_1);
            sum3 = vmlaq_f32(sum3, din1, w3_1);
            sum4 = vmlaq_f32(sum4, din1, w4_1);
            sum5 = vmlaq_f32(sum5, din1, w5_1);
            sum6 = vmlaq_f32(sum6, din1, w6_1);
            sum7 = vmlaq_f32(sum7, din1, w7_1);
        }
        //add
        float32x4_t vout0 = vpaddq_f32(sum0, sum1);
        float32x4_t vout1 = vpaddq_f32(sum2, sum3);
        float32x4_t vout2 = vpaddq_f32(sum4, sum5);
        float32x4_t vout3 = vpaddq_f32(sum6, sum7);
        float32x4_t vdout = vpaddq_f32(vout0, vout1);
        float32x4_t vdout1 = vpaddq_f32(vout2, vout3);
        vst1q_f32(ptr_out, vdout);
        vst1q_f32(ptr_out + 4, vdout1);
    }
    //! deal with remains
    #pragma omp parallel for
    for (int j = out_cnt * 8; j < M; ++j){
        float *ptr_out = data_out + j;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;

         asm volatile(
            "prfm  pldl1keep, [%[in]]   \n"
                    "prfm  pldl1keep, [%[w0]]   \n"
                     :
        :[in] "r"(ptr_in), [w0] "r"(ptr_w0)
        :"memory"
        );

        float32x4_t sum0 = vdupq_n_f32(0.f);
        for (int i = 0; i < cnt_loop; i++){
            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);

            sum0 = vmlaq_f32(sum0, din0, w0_0);

            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);

            sum0 = vmlaq_f32(sum0, din1, w0_1);

            ptr_in += 8;
            ptr_w0 += 8;

        }
        if (tail > 0){
            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);

            din0 = vbslq_f32(vmask1, din0, vzero);
            sum0 = vmlaq_f32(sum0, din0, w0_0);

            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);

            din1 = vbslq_f32(vmask2, din1, vzero);
            sum0 = vmlaq_f32(sum0, din1, w0_1);
        }
        //add
        float32x2_t vout = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
        float32x2_t vdout = vpadd_f32(vout, vout);
        *ptr_out = vget_lane_f32(vdout, 0);
    }
#else
        int cnt = cnt_loop;
        if (cnt > 0) {
            asm volatile(
                    "1:                             \n"
                    "LDP q30, q31, [%[in]], #32   \n" // q0=A0A1A2A3
                    "LDP q8, q9, [%[w0]], #32       \n" // q0=A0A1A2A3
                    "LDP q10, q11, [%[w1]], #32       \n" // q0=A0A1A2A3
                    "LDP q12, q13, [%[w2]], #32     \n" // q0=A0A1A2A3
                    "LDP q14, q15, [%[w3]], #32     \n" // q0=A0A1A2A

                    "prfm  pldl1keep, [%[w0]]   \n"
                    "prfm  pldl1keep, [%[w1]]   \n"
                    "prfm  pldl1keep, [%[w2]]   \n"
                    "prfm  pldl1keep, [%[w3]]   \n"


                    "fmul v0.4s, v30.4s, v8.4s       \n"
                    "fmul v1.4s, v30.4s, v10.4s       \n"
                    "fmul v2.4s, v30.4s, v12.4s       \n"
                    "fmul v3.4s, v30.4s, v14.4s       \n"

                    "LDP q16, q17, [%[w4]], #32     \n" // q0=A0A1A2A3
                    "LDP q18, q19, [%[w5]], #32     \n" // q0=A0A1A2A3
                    "LDP q20, q21, [%[w6]], #32     \n" // q0=A0A1A2A3
                    "LDP q22, q23, [%[w7]], #32     \n" // q0=A0A1A2A3

                    "prfm  pldl1keep, [%[w4]]   \n"
                    "prfm  pldl1keep, [%[w5]]   \n"
                    "prfm  pldl1keep, [%[w6]]   \n"
                    "prfm  pldl1keep, [%[w7]]   \n"

                    "fmul v4.4s, v30.4s, v16.4s       \n"
                    "fmul v5.4s, v30.4s, v18.4s       \n"
                    "fmul v6.4s, v30.4s, v20.4s       \n"
                    "fmul v7.4s, v30.4s, v22.4s       \n"


                    "prfm  pldl1keep, [%[in]]   \n"
                    "prfm   pldl1keep, [%[in], #64]   \n"

                    "fmla v0.4s, v31.4s, v9.4s       \n"
                    "fmla v1.4s, v31.4s, v11.4s       \n"
                    "fmla v2.4s, v31.4s, v13.4s       \n"
                    "fmla v3.4s, v31.4s, v15.4s       \n"

                    "fmla v4.4s, v31.4s, v17.4s       \n"
                    "fmla v5.4s, v31.4s, v19.4s       \n"
                    "fmla v6.4s, v31.4s, v21.4s       \n"
                    "fmla v7.4s, v31.4s, v23.4s       \n"


                    // check loop end
                    "subs %[cnt], %[cnt], #1        \n"
                    "bne 1b                         \n"

                    // check tails
                    "cmp %[tail], #1                \n"
                    "blt  2f                       \n"

                    // process tail
                    "LDP q30, q31, [%[in]], #32   \n" // q0=A0A1A2A3

                    "movi  v26.4s, #0x0           \n"
                    // deal with right pad
                    "bif v30.8b, v16.8b, %[mask1].8b         \n"
                    "bif v31.8b, v16.8b, %[mask2].8b         \n"

                    "LDP q8, q9, [%[w0]], #32       \n" // q0=A0A1A2A3
                    "LDP q10, q11, [%[w1]], #32       \n" // q0=A0A1A2A3
                    "LDP q12, q13, [%[w2]], #32     \n" // q0=A0A1A2A3
                    "LDP q14, q15, [%[w3]], #32     \n" // q0=A0A1A2A

                    "LDP q16, q17, [%[w4]], #32     \n" // q0=A0A1A2A3
                    "LDP q18, q19, [%[w5]], #32     \n" // q0=A0A1A2A3
                    "LDP q20, q21, [%[w6]], #32     \n" // q0=A0A1A2A3
                    "LDP q22, q23, [%[w7]], #32     \n" // q0=A0A1A2A3

                    "fmul v0.4s, v30.4s, v8.4s       \n"
                    "fmul v1.4s, v30.4s, v10.4s       \n"
                    "fmul v2.4s, v30.4s, v12.4s       \n"
                    "fmul v3.4s, v30.4s, v14.4s       \n"
                    "fmul v4.4s, v30.4s, v16.4s       \n"
                    "fmul v5.4s, v30.4s, v18.4s       \n"
                    "fmul v6.4s, v30.4s, v20.4s       \n"
                    "fmul v7.4s, v30.4s, v22.4s       \n"


                    "fmla v0.4s, v31.4s, v9.4s       \n"
                    "fmla v1.4s, v31.4s, v11.4s       \n"
                    "fmla v2.4s, v31.4s, v13.4s       \n"
                    "fmla v3.4s, v31.4s, v15.4s       \n"

                    "fmla v4.4s, v31.4s, v17.4s       \n"
                    "fmla v5.4s, v31.4s, v19.4s       \n"
                    "fmla v6.4s, v31.4s, v21.4s       \n"
                    "fmla v7.4s, v31.4s, v23.4s       \n"

                    // pair add to final result
                    "2:                             \n"
                    "addp v8.4s, v0.4s, v1.4s      \n"
                    "addp v9.4s, v2.4s, v3.4s      \n"
                    "addp v10.4s, v4.4s, v5.4s     \n"
                    "addp v11.4s, v6.4s, v7.4s     \n"

                    "addp v12.4s, v8.4s, v9.4s      \n"
                    "addp v13.4s, v10.4s, v11.4s    \n"

                    "STR   q12, [%[out]], #16       \n"
                    "STR   q13, [%[out]], #16      \n"

            :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [w1] "+r"(ptr_w1), \
                 [w2] "+r"(ptr_w2), [w3] "+r"(ptr_w3), [w4] "+r"(ptr_w4), \
                 [w5] "+r"(ptr_w5), [w6] "+r"(ptr_w6), [w7] "+r"(ptr_w7), [out] "+r"(ptr_out), \
                 [cnt] "+r"(cnt)
            :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2)
            :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", \
                "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",  \
                "v18", "v19", "v20", "v21",  "v22", "v23", "v26", "v30", "v31"
            );
        }
    }

    //! deal with remains
    #pragma omp parallel for
    for (int j = out_cnt * 8; j < M; ++j) {
        float *ptr_out = data_out + j;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;
        if (cnt > 0){
            asm volatile(
                "prfm  pldl1keep, [%[in]]   \n"
                "prfm  pldl1keep, [%[w0]]   \n"

                "movi  v0.4s, #0x0           \n"

                "1:                                 \n"
                "LDP q12, q13, [%[in]], #32       \n" // q0=A0A1A2A3
                "LDP q14, q15, [%[w0]], #32      \n" // q0=A0A1A2A3

                "prfm  pldl1keep, [%[in]]   \n"
                "prfm  pldl1keep, [%[w0]]   \n"

                "fmla v0.4s, v12.4s, v14.4s       \n"
                "fmla v0.4s, v13.4s, v15.4s       \n"

                "subs %[cnt], %[cnt], #1                  \n"
                "bne 1b                             \n"

                // check tails
                "cmp %[tail], #1                    \n"
                "blt  2f                            \n"

                // process tail
                "LDP q12, q13, [%[in]], #32       \n" // q0=A0A1A2A3
                // deal with right pad
                "movi  v1.4s, #0x0           \n"
                "bif v12.8b, v1.8b, %[mask1].8b            \n"
                "bif v13.8b, v1.8b, %[mask2].8b            \n"

                "LDP q14, q15, [%[in]], #32       \n" // q0=A0A1A2A3

                "fmla v0.4s, v12.4s, v14.4s       \n"
                "fmla v0.4s, v13.4s, v15.4s       \n"

                // pair add to final result
                "2:                                 \n"
                "addp v1.4s, v0.4s, v0.4s               \n"
                "addp v3.4s, v1.4s, v2.4s              \n"
                "str   q3, [%[out]], #4      \n"
            :[in] "+r"(ptr_in), [w0] "+r"(ptr_w0), [out] "+r"(ptr_out), \
            [cnt] "+r"(cnt)
            :[tail] "r" (tail), [mask1] "w" (vmask1), [mask2] "w" (vmask2)
            :"v0", "v1", "v12", "v13", "v14", "v15"
            );
        }
    }
#endif
#else
    int cnt_loop = N >> 3;
    int tail = N & 7;
    int out_cnt = M >> 2;

    unsigned int imask[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(imask), vdupq_n_u32(tail));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(imask + 4), vdupq_n_u32(tail));

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

                "cmp %[cnt], #1                 @ check whether has main loop\n"
                "blt  3f                        @ jump to tail\n"

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
                "3:                             @ check tail\n"
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

                "cmp %[cnt], #1                     @ check whether has main loop\n"
                "blt  3f                            @ jump to tail\n"

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
                "3:                                 @ tail\n"
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
        :[tail] "r"(tail), [mask1] "w"(vmask1), [mask2] "w"(vmask2)
        :"q0", "q1", "q12", "q13", "q14", "q15"
        );
    }
#endif //__aarch64__
}


void sgemv_relu(const bool transA, const int M, const int N, \
    const float* A, const float* x, float* y) {
    float* data_out = y;
    const float* data_in = x;
    const float* weights_ptr = A;

#ifdef __aarch64__
//todo
    int cnt_loop = N >> 3;
    int tail = N & 7;
    int out_cnt = M >> 3;

    unsigned int imask[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(imask), vdupq_n_u32(tail));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(imask + 4), vdupq_n_u32(tail));

    float32x4_t vzero = vdupq_n_f32(0.f);

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {

        int out_idx = j * 8;
        float *ptr_out = data_out + out_idx;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * out_idx);
        const float *ptr_w1 = ptr_w0 + N;
        const float *ptr_w2 = ptr_w1 + N;
        const float *ptr_w3 = ptr_w2 + N;
        const float *ptr_w4 = ptr_w3 + N;
        const float *ptr_w5 = ptr_w4 + N;
        const float *ptr_w6 = ptr_w5 + N;
        const float *ptr_w7 = ptr_w6 + N;

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

        float32x4_t sum0 = vdupq_n_f32(0.f);
        float32x4_t sum1 = vdupq_n_f32(0.f);
        float32x4_t sum2 = vdupq_n_f32(0.f);
        float32x4_t sum3 = vdupq_n_f32(0.f);
        float32x4_t sum4 = vdupq_n_f32(0.f);
        float32x4_t sum5 = vdupq_n_f32(0.f);
        float32x4_t sum6 = vdupq_n_f32(0.f);
        float32x4_t sum7 = vdupq_n_f32(0.f);

        for (int i = 0; i < cnt_loop; i++){

            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);
            float32x4_t w1_0 = vld1q_f32(ptr_w1);
            float32x4_t w2_0 = vld1q_f32(ptr_w2);
            float32x4_t w3_0 = vld1q_f32(ptr_w3);
            float32x4_t w4_0 = vld1q_f32(ptr_w4);
            float32x4_t w5_0 = vld1q_f32(ptr_w5);
            float32x4_t w6_0 = vld1q_f32(ptr_w6);
            float32x4_t w7_0 = vld1q_f32(ptr_w7);

            sum0 = vmlaq_f32(sum0, din0, w0_0);
            sum1 = vmlaq_f32(sum1, din0, w1_0);
            sum2 = vmlaq_f32(sum2, din0, w2_0);
            sum3 = vmlaq_f32(sum3, din0, w3_0);
            sum4 = vmlaq_f32(sum4, din0, w4_0);
            sum5 = vmlaq_f32(sum5, din0, w5_0);
            sum6 = vmlaq_f32(sum6, din0, w6_0);
            sum7 = vmlaq_f32(sum7, din0, w7_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);
            float32x4_t w1_1 = vld1q_f32(ptr_w1 + 4);
            float32x4_t w2_1 = vld1q_f32(ptr_w2 + 4);
            float32x4_t w3_1 = vld1q_f32(ptr_w3 + 4);
            float32x4_t w4_1 = vld1q_f32(ptr_w4 + 4);
            float32x4_t w5_1 = vld1q_f32(ptr_w5 + 4);
            float32x4_t w6_1 = vld1q_f32(ptr_w6 + 4);
            float32x4_t w7_1 = vld1q_f32(ptr_w7 + 4);

            sum0 = vmlaq_f32(sum0, din1, w0_1);
            sum1 = vmlaq_f32(sum1, din1, w1_1);
            sum2 = vmlaq_f32(sum2, din1, w2_1);
            sum3 = vmlaq_f32(sum3, din1, w3_1);
            sum4 = vmlaq_f32(sum4, din1, w4_1);
            sum5 = vmlaq_f32(sum5, din1, w5_1);
            sum6 = vmlaq_f32(sum6, din1, w6_1);
            sum7 = vmlaq_f32(sum7, din1, w7_1);

            ptr_in += 8;
            ptr_w0 += 8;
            ptr_w1 += 8;
            ptr_w2 += 8;
            ptr_w3 += 8;
            ptr_w4 += 8;
            ptr_w5 += 8;
            ptr_w6 += 8;
            ptr_w7 += 8;

        }
        if (tail >= 1){
            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);
            float32x4_t w1_0 = vld1q_f32(ptr_w1);
            float32x4_t w2_0 = vld1q_f32(ptr_w2);
            float32x4_t w3_0 = vld1q_f32(ptr_w3);

            din0 = vbslq_f32(vmask1, din0, vzero);

            float32x4_t w4_0 = vld1q_f32(ptr_w4);
            float32x4_t w5_0 = vld1q_f32(ptr_w5);
            float32x4_t w6_0 = vld1q_f32(ptr_w6);
            float32x4_t w7_0 = vld1q_f32(ptr_w7);


            sum0 = vmlaq_f32(sum0, din0, w0_0);
            sum1 = vmlaq_f32(sum1, din0, w1_0);
            sum2 = vmlaq_f32(sum2, din0, w2_0);
            sum3 = vmlaq_f32(sum3, din0, w3_0);
            sum4 = vmlaq_f32(sum4, din0, w4_0);
            sum5 = vmlaq_f32(sum5, din0, w5_0);
            sum6 = vmlaq_f32(sum6, din0, w6_0);
            sum7 = vmlaq_f32(sum7, din0, w7_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);
            float32x4_t w1_1 = vld1q_f32(ptr_w1 + 4);
            float32x4_t w2_1 = vld1q_f32(ptr_w2 + 4);

            din1 = vbslq_f32(vmask2, din1, vzero);

            float32x4_t w3_1 = vld1q_f32(ptr_w3 + 4);
            float32x4_t w4_1 = vld1q_f32(ptr_w4 + 4);
            float32x4_t w5_1 = vld1q_f32(ptr_w5 + 4);
            float32x4_t w6_1 = vld1q_f32(ptr_w6 + 4);
            float32x4_t w7_1 = vld1q_f32(ptr_w7 + 4);

            sum0 = vmlaq_f32(sum0, din1, w0_1);
            sum1 = vmlaq_f32(sum1, din1, w1_1);
            sum2 = vmlaq_f32(sum2, din1, w2_1);
            sum3 = vmlaq_f32(sum3, din1, w3_1);
            sum4 = vmlaq_f32(sum4, din1, w4_1);
            sum5 = vmlaq_f32(sum5, din1, w5_1);
            sum6 = vmlaq_f32(sum6, din1, w6_1);
            sum7 = vmlaq_f32(sum7, din1, w7_1);
        }
        //add
        float32x4_t vout0 = vpaddq_f32(sum0, sum1);
        float32x4_t vout1 = vpaddq_f32(sum2, sum3);
        float32x4_t vout2 = vpaddq_f32(sum4, sum5);
        float32x4_t vout3 = vpaddq_f32(sum6, sum7);
        float32x4_t vdout = vpaddq_f32(vout0, vout1);
        float32x4_t vdout1 = vpaddq_f32(vout2, vout3);
        vdout = vmaxq_f32(vdout, vzero);
        vdout1 = vmaxq_f32(vdout1, vzero);
        vst1q_f32(ptr_out, vdout);
        vst1q_f32(ptr_out + 4, vdout1);
    }
    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 8; j < M; ++j){
        float *ptr_out = data_out + j;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;

         asm volatile(
            "prfm  pldl1keep, [%[in]]   \n"
                    "prfm  pldl1keep, [%[w0]]   \n"
                     :
        :[in] "r"(ptr_in), [w0] "r"(ptr_w0)
        :"memory"
        );

        float32x4_t sum0 = vdupq_n_f32(0.f);
        for (int i = 0; i < cnt_loop; i++){
            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);

            sum0 = vmlaq_f32(sum0, din0, w0_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);

            sum0 = vmlaq_f32(sum0, din1, w0_1);

            ptr_in += 8;
            ptr_w0 += 8;

        }
        if (tail >= 1){
            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);

            din0 = vbslq_f32(vmask1, din0, vzero);


            sum0 = vmlaq_f32(sum0, din0, w0_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);

            din1 = vbslq_f32(vmask2, din1, vzero);

            sum0 = vmlaq_f32(sum0, din1, w0_1);
        }
        //add
        float32x2_t vout = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
        float32x2_t vdout = vpadd_f32(vout, vout);
        vdout = vmax_f32(vdout, vget_low_f32(vzero));
        *ptr_out = vget_lane_f32(vdout, 0);
    }
#else

    int cnt_loop = N >> 3;
    int tail = N & 7;
    int out_cnt = M >> 2;

    unsigned int imask[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(imask), vdupq_n_u32(tail));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(imask + 4), vdupq_n_u32(tail));
    unsigned int mask[8];

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

                "cmp %[cnt], #1                 @ check whether has main loop\n"
                "blt  3f                        @ jump to tail\n"

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
                "3:                                 @ tail\n"
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

                "cmp %[cnt], #1                     @ check whether has main loop\n"
                "blt  3f                            @ jump to tail\n"

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
                "3:                                 @ tail\n"
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
#endif //__aarch64__
}

void sgemv_bias(const bool transA, const int M, const int N, \
    const float* A, const float* x, float* y, const float* bias) {

    float* data_out = y;
    const float* data_in = x;
    const float* weights_ptr = A;

#ifdef __aarch64__
//todo
    int cnt_loop = N >> 3;
    int tail = N & 7;
    int out_cnt = M >> 3;

    unsigned int imask[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(imask), vdupq_n_u32(tail));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(imask + 4), vdupq_n_u32(tail));

    float32x4_t vzero = vdupq_n_f32(0.f);

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {

        int out_idx = j * 8;
        float *ptr_out = data_out + out_idx;
        const float *ptr_bias = bias + out_idx;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * out_idx);
        const float *ptr_w1 = ptr_w0 + N;
        const float *ptr_w2 = ptr_w1 + N;
        const float *ptr_w3 = ptr_w2 + N;
        const float *ptr_w4 = ptr_w3 + N;
        const float *ptr_w5 = ptr_w4 + N;
        const float *ptr_w6 = ptr_w5 + N;
        const float *ptr_w7 = ptr_w6 + N;

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

        float32x4_t sum0 = vdupq_n_f32(0.f);
        float32x4_t sum1 = vdupq_n_f32(0.f);
        float32x4_t sum2 = vdupq_n_f32(0.f);
        float32x4_t sum3 = vdupq_n_f32(0.f);
        float32x4_t sum4 = vdupq_n_f32(0.f);
        float32x4_t sum5 = vdupq_n_f32(0.f);
        float32x4_t sum6 = vdupq_n_f32(0.f);
        float32x4_t sum7 = vdupq_n_f32(0.f);

        for (int i = 0; i < cnt_loop; i++){

            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);
            float32x4_t w1_0 = vld1q_f32(ptr_w1);
            float32x4_t w2_0 = vld1q_f32(ptr_w2);
            float32x4_t w3_0 = vld1q_f32(ptr_w3);
            float32x4_t w4_0 = vld1q_f32(ptr_w4);
            float32x4_t w5_0 = vld1q_f32(ptr_w5);
            float32x4_t w6_0 = vld1q_f32(ptr_w6);
            float32x4_t w7_0 = vld1q_f32(ptr_w7);

            sum0 = vmlaq_f32(sum0, din0, w0_0);
            sum1 = vmlaq_f32(sum1, din0, w1_0);
            sum2 = vmlaq_f32(sum2, din0, w2_0);
            sum3 = vmlaq_f32(sum3, din0, w3_0);
            sum4 = vmlaq_f32(sum4, din0, w4_0);
            sum5 = vmlaq_f32(sum5, din0, w5_0);
            sum6 = vmlaq_f32(sum6, din0, w6_0);
            sum7 = vmlaq_f32(sum7, din0, w7_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);
            float32x4_t w1_1 = vld1q_f32(ptr_w1 + 4);
            float32x4_t w2_1 = vld1q_f32(ptr_w2 + 4);
            float32x4_t w3_1 = vld1q_f32(ptr_w3 + 4);
            float32x4_t w4_1 = vld1q_f32(ptr_w4 + 4);
            float32x4_t w5_1 = vld1q_f32(ptr_w5 + 4);
            float32x4_t w6_1 = vld1q_f32(ptr_w6 + 4);
            float32x4_t w7_1 = vld1q_f32(ptr_w7 + 4);

            sum0 = vmlaq_f32(sum0, din1, w0_1);
            sum1 = vmlaq_f32(sum1, din1, w1_1);
            sum2 = vmlaq_f32(sum2, din1, w2_1);
            sum3 = vmlaq_f32(sum3, din1, w3_1);
            sum4 = vmlaq_f32(sum4, din1, w4_1);
            sum5 = vmlaq_f32(sum5, din1, w5_1);
            sum6 = vmlaq_f32(sum6, din1, w6_1);
            sum7 = vmlaq_f32(sum7, din1, w7_1);

            ptr_in += 8;
            ptr_w0 += 8;
            ptr_w1 += 8;
            ptr_w2 += 8;
            ptr_w3 += 8;
            ptr_w4 += 8;
            ptr_w5 += 8;
            ptr_w6 += 8;
            ptr_w7 += 8;

        }
        if (tail >= 1){
            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);
            float32x4_t w1_0 = vld1q_f32(ptr_w1);
            float32x4_t w2_0 = vld1q_f32(ptr_w2);
            float32x4_t w3_0 = vld1q_f32(ptr_w3);

            din0 = vbslq_f32(vmask1, din0, vzero);

            float32x4_t w4_0 = vld1q_f32(ptr_w4);
            float32x4_t w5_0 = vld1q_f32(ptr_w5);
            float32x4_t w6_0 = vld1q_f32(ptr_w6);
            float32x4_t w7_0 = vld1q_f32(ptr_w7);


            sum0 = vmlaq_f32(sum0, din0, w0_0);
            sum1 = vmlaq_f32(sum1, din0, w1_0);
            sum2 = vmlaq_f32(sum2, din0, w2_0);
            sum3 = vmlaq_f32(sum3, din0, w3_0);
            sum4 = vmlaq_f32(sum4, din0, w4_0);
            sum5 = vmlaq_f32(sum5, din0, w5_0);
            sum6 = vmlaq_f32(sum6, din0, w6_0);
            sum7 = vmlaq_f32(sum7, din0, w7_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);
            float32x4_t w1_1 = vld1q_f32(ptr_w1 + 4);
            float32x4_t w2_1 = vld1q_f32(ptr_w2 + 4);

            din1 = vbslq_f32(vmask2, din1, vzero);

            float32x4_t w3_1 = vld1q_f32(ptr_w3 + 4);
            float32x4_t w4_1 = vld1q_f32(ptr_w4 + 4);
            float32x4_t w5_1 = vld1q_f32(ptr_w5 + 4);
            float32x4_t w6_1 = vld1q_f32(ptr_w6 + 4);
            float32x4_t w7_1 = vld1q_f32(ptr_w7 + 4);

            sum0 = vmlaq_f32(sum0, din1, w0_1);
            sum1 = vmlaq_f32(sum1, din1, w1_1);
            sum2 = vmlaq_f32(sum2, din1, w2_1);
            sum3 = vmlaq_f32(sum3, din1, w3_1);
            sum4 = vmlaq_f32(sum4, din1, w4_1);
            sum5 = vmlaq_f32(sum5, din1, w5_1);
            sum6 = vmlaq_f32(sum6, din1, w6_1);
            sum7 = vmlaq_f32(sum7, din1, w7_1);
        }

        float32x4_t vbias0 = vld1q_f32(ptr_bias);
        float32x4_t vbias1 = vld1q_f32(ptr_bias + 4);
        //add
        float32x4_t vout0 = vpaddq_f32(sum0, sum1);
        float32x4_t vout1 = vpaddq_f32(sum2, sum3);
        float32x4_t vout2 = vpaddq_f32(sum4, sum5);
        float32x4_t vout3 = vpaddq_f32(sum6, sum7);
        float32x4_t vdout = vpaddq_f32(vout0, vout1);
        float32x4_t vdout1 = vpaddq_f32(vout2, vout3);


        vdout = vaddq_f32(vdout, vbias0);
        vdout1 = vaddq_f32(vdout1, vbias1);

        vst1q_f32(ptr_out, vdout);
        vst1q_f32(ptr_out + 4, vdout1);
    }
    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 8; j < M; ++j){
        float *ptr_out = data_out + j;
        const float *ptr_bias = bias + j;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;

         asm volatile(
            "prfm  pldl1keep, [%[in]]   \n"
                    "prfm  pldl1keep, [%[w0]]   \n"
                    "prfm  pldl1keep, [%[bias]]   \n"
                     :
        :[in] "r"(ptr_in), [w0] "r"(ptr_w0), [bias] "r" (ptr_bias)
        :"memory"
        );

        float32x4_t sum0 = vdupq_n_f32(0.f);
        for (int i = 0; i < cnt_loop; i++){
            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);

            sum0 = vmlaq_f32(sum0, din0, w0_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);

            sum0 = vmlaq_f32(sum0, din1, w0_1);

            ptr_in += 8;
            ptr_w0 += 8;

        }
        if (tail >= 1){
            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);

            din0 = vbslq_f32(vmask1, din0, vzero);


            sum0 = vmlaq_f32(sum0, din0, w0_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);

            din1 = vbslq_f32(vmask2, din1, vzero);

            sum0 = vmlaq_f32(sum0, din1, w0_1);
        }

        float32x2_t vbias = vld1_f32(ptr_bias);
        //add
        float32x2_t vout = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
        float32x2_t vdout = vpadd_f32(vout, vout);
        vdout = vadd_f32(vdout, vbias);
        *ptr_out = vget_lane_f32(vdout, 0);
    }
#else

    int cnt_loop = N >> 3;
    int tail = N & 7;
    int out_cnt = M >> 2;

    unsigned int imask[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(imask), vdupq_n_u32(tail));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(imask + 4), vdupq_n_u32(tail));

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

                    "cmp %[cnt], #1                 @ check whether has main loop\n"
                    "blt  3f                        @ jump to tail\n"

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
                    "3:                                 @ tail\n"
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

                "cmp %[cnt], #1                     @ check whether has main loop\n"
                "blt  3f                            @ jump to tail\n"

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
                "3:                                 @ tail\n"
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

#endif //__aarch64__
}


void sgemv_bias_relu(const bool transA, const int M, const int N, \
    const float* A, const float* x, float* y, const float* bias) {
    float* data_out = y;
    const float* data_in = x;
    const float* weights_ptr = A;

#ifdef __aarch64__
//todo

    int cnt_loop = N >> 3;
    int tail = N & 7;
    int out_cnt = M >> 3;

    unsigned int imask[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(imask), vdupq_n_u32(tail));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(imask + 4), vdupq_n_u32(tail));

    float32x4_t vzero = vdupq_n_f32(0.f);

#pragma omp parallel for
    for (int j = 0; j < out_cnt; j++) {

        int out_idx = j * 8;
        float *ptr_out = data_out + out_idx;
        const float *ptr_bias = bias + out_idx;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * out_idx);
        const float *ptr_w1 = ptr_w0 + N;
        const float *ptr_w2 = ptr_w1 + N;
        const float *ptr_w3 = ptr_w2 + N;
        const float *ptr_w4 = ptr_w3 + N;
        const float *ptr_w5 = ptr_w4 + N;
        const float *ptr_w6 = ptr_w5 + N;
        const float *ptr_w7 = ptr_w6 + N;

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

        float32x4_t sum0 = vdupq_n_f32(0.f);
        float32x4_t sum1 = vdupq_n_f32(0.f);
        float32x4_t sum2 = vdupq_n_f32(0.f);
        float32x4_t sum3 = vdupq_n_f32(0.f);
        float32x4_t sum4 = vdupq_n_f32(0.f);
        float32x4_t sum5 = vdupq_n_f32(0.f);
        float32x4_t sum6 = vdupq_n_f32(0.f);
        float32x4_t sum7 = vdupq_n_f32(0.f);

        for (int i = 0; i < cnt_loop; i++){

            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);
            float32x4_t w1_0 = vld1q_f32(ptr_w1);
            float32x4_t w2_0 = vld1q_f32(ptr_w2);
            float32x4_t w3_0 = vld1q_f32(ptr_w3);
            float32x4_t w4_0 = vld1q_f32(ptr_w4);
            float32x4_t w5_0 = vld1q_f32(ptr_w5);
            float32x4_t w6_0 = vld1q_f32(ptr_w6);
            float32x4_t w7_0 = vld1q_f32(ptr_w7);

            sum0 = vmlaq_f32(sum0, din0, w0_0);
            sum1 = vmlaq_f32(sum1, din0, w1_0);
            sum2 = vmlaq_f32(sum2, din0, w2_0);
            sum3 = vmlaq_f32(sum3, din0, w3_0);
            sum4 = vmlaq_f32(sum4, din0, w4_0);
            sum5 = vmlaq_f32(sum5, din0, w5_0);
            sum6 = vmlaq_f32(sum6, din0, w6_0);
            sum7 = vmlaq_f32(sum7, din0, w7_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);
            float32x4_t w1_1 = vld1q_f32(ptr_w1 + 4);
            float32x4_t w2_1 = vld1q_f32(ptr_w2 + 4);
            float32x4_t w3_1 = vld1q_f32(ptr_w3 + 4);
            float32x4_t w4_1 = vld1q_f32(ptr_w4 + 4);
            float32x4_t w5_1 = vld1q_f32(ptr_w5 + 4);
            float32x4_t w6_1 = vld1q_f32(ptr_w6 + 4);
            float32x4_t w7_1 = vld1q_f32(ptr_w7 + 4);

            sum0 = vmlaq_f32(sum0, din1, w0_1);
            sum1 = vmlaq_f32(sum1, din1, w1_1);
            sum2 = vmlaq_f32(sum2, din1, w2_1);
            sum3 = vmlaq_f32(sum3, din1, w3_1);
            sum4 = vmlaq_f32(sum4, din1, w4_1);
            sum5 = vmlaq_f32(sum5, din1, w5_1);
            sum6 = vmlaq_f32(sum6, din1, w6_1);
            sum7 = vmlaq_f32(sum7, din1, w7_1);

            ptr_in += 8;
            ptr_w0 += 8;
            ptr_w1 += 8;
            ptr_w2 += 8;
            ptr_w3 += 8;
            ptr_w4 += 8;
            ptr_w5 += 8;
            ptr_w6 += 8;
            ptr_w7 += 8;

        }
        if (tail >= 1){
            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);
            float32x4_t w1_0 = vld1q_f32(ptr_w1);
            float32x4_t w2_0 = vld1q_f32(ptr_w2);
            float32x4_t w3_0 = vld1q_f32(ptr_w3);

            din0 = vbslq_f32(vmask1, din0, vzero);

            float32x4_t w4_0 = vld1q_f32(ptr_w4);
            float32x4_t w5_0 = vld1q_f32(ptr_w5);
            float32x4_t w6_0 = vld1q_f32(ptr_w6);
            float32x4_t w7_0 = vld1q_f32(ptr_w7);


            sum0 = vmlaq_f32(sum0, din0, w0_0);
            sum1 = vmlaq_f32(sum1, din0, w1_0);
            sum2 = vmlaq_f32(sum2, din0, w2_0);
            sum3 = vmlaq_f32(sum3, din0, w3_0);
            sum4 = vmlaq_f32(sum4, din0, w4_0);
            sum5 = vmlaq_f32(sum5, din0, w5_0);
            sum6 = vmlaq_f32(sum6, din0, w6_0);
            sum7 = vmlaq_f32(sum7, din0, w7_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);
            float32x4_t w1_1 = vld1q_f32(ptr_w1 + 4);
            float32x4_t w2_1 = vld1q_f32(ptr_w2 + 4);

            din1 = vbslq_f32(vmask2, din1, vzero);

            float32x4_t w3_1 = vld1q_f32(ptr_w3 + 4);
            float32x4_t w4_1 = vld1q_f32(ptr_w4 + 4);
            float32x4_t w5_1 = vld1q_f32(ptr_w5 + 4);
            float32x4_t w6_1 = vld1q_f32(ptr_w6 + 4);
            float32x4_t w7_1 = vld1q_f32(ptr_w7 + 4);

            sum0 = vmlaq_f32(sum0, din1, w0_1);
            sum1 = vmlaq_f32(sum1, din1, w1_1);
            sum2 = vmlaq_f32(sum2, din1, w2_1);
            sum3 = vmlaq_f32(sum3, din1, w3_1);
            sum4 = vmlaq_f32(sum4, din1, w4_1);
            sum5 = vmlaq_f32(sum5, din1, w5_1);
            sum6 = vmlaq_f32(sum6, din1, w6_1);
            sum7 = vmlaq_f32(sum7, din1, w7_1);
        }

        float32x4_t vbias0 = vld1q_f32(ptr_bias);
        float32x4_t vbias1 = vld1q_f32(ptr_bias + 4);
        //add
        float32x4_t vout0 = vpaddq_f32(sum0, sum1);
        float32x4_t vout1 = vpaddq_f32(sum2, sum3);
        float32x4_t vout2 = vpaddq_f32(sum4, sum5);
        float32x4_t vout3 = vpaddq_f32(sum6, sum7);
        float32x4_t vdout = vpaddq_f32(vout0, vout1);
        float32x4_t vdout1 = vpaddq_f32(vout2, vout3);


        vdout = vaddq_f32(vdout, vbias0);
        vdout1 = vaddq_f32(vdout1, vbias1);

        vdout = vmaxq_f32(vdout, vzero);
        vdout1 = vmaxq_f32(vdout1, vzero);

        vst1q_f32(ptr_out, vdout);
        vst1q_f32(ptr_out + 4, vdout1);
    }
    //! deal with remains
#pragma omp parallel for
    for (int j = out_cnt * 8; j < M; ++j){
        float *ptr_out = data_out + j;
        const float *ptr_bias = bias + j;
        const float *ptr_in = data_in;
        const float *ptr_w0 = weights_ptr + (N * j);
        int cnt = cnt_loop;

         asm volatile(
            "prfm  pldl1keep, [%[in]]   \n"
                    "prfm  pldl1keep, [%[w0]]   \n"
                    "prfm  pldl1keep, [%[bias]]   \n"
                     :
        :[in] "r"(ptr_in), [w0] "r"(ptr_w0), [bias] "r" (ptr_bias)
        :"memory"
        );

        float32x4_t sum0 = vdupq_n_f32(0.f);
        for (int i = 0; i < cnt_loop; i++){
            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);

            sum0 = vmlaq_f32(sum0, din0, w0_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);

            sum0 = vmlaq_f32(sum0, din1, w0_1);

            ptr_in += 8;
            ptr_w0 += 8;

        }
        if (tail >= 1){
            float32x4_t din0 = vld1q_f32(ptr_in);
            float32x4_t w0_0 = vld1q_f32(ptr_w0);

            din0 = vbslq_f32(vmask1, din0, vzero);


            sum0 = vmlaq_f32(sum0, din0, w0_0);


            float32x4_t din1 = vld1q_f32(ptr_in + 4);
            float32x4_t w0_1 = vld1q_f32(ptr_w0 + 4);

            din1 = vbslq_f32(vmask2, din1, vzero);

            sum0 = vmlaq_f32(sum0, din1, w0_1);
        }

        float32x2_t vbias = vld1_f32(ptr_bias);
        //add
        float32x2_t vout = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
        float32x2_t vdout = vpadd_f32(vout, vout);
        vdout = vadd_f32(vdout, vbias);
        vdout = vmax_f32(vdout, vget_low_f32(vzero));
        *ptr_out = vget_lane_f32(vdout, 0);
    }
#else

    int cnt_loop = N >> 3;
    int tail = N & 7;
    int out_cnt = M >> 2;

    unsigned int imask[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    uint32x4_t vmask1 = vcltq_u32(vld1q_u32(imask), vdupq_n_u32(tail));
    uint32x4_t vmask2 = vcltq_u32(vld1q_u32(imask + 4), vdupq_n_u32(tail));

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

                    "cmp %[cnt], #1                 @ check whether has main loop\n"
                    "blt  3f                        @ jump to tail\n"

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
                    "3:                             @ tail\n"
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

                "cmp %[cnt], #1                     @ check whether has main loop\n"
                "blt  3f                            @ jump to tail\n"

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
                "3:                                 @ tail\n"
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
#endif //__aarch64__
}

} //lite

} //saber

} //namespace anakin

#endif //USE_ARM_PLACE
