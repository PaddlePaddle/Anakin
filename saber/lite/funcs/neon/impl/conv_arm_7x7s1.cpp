#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

void conv7x7_mid_top2(const float* din, float* dout, int w_out, int width, int height, const float* weight_ch_in){
    float32x4_t vweights10 = vld1q_f32(weight_ch_in + 7);
    float32x4_t vweights11 = vld1q_f32(weight_ch_in + 11);
    vweights11 = vsetq_lane_f32(0.f, vweights11, 3);

    float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
    float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
    vweights21 = vsetq_lane_f32(0.f, vweights21, 3);

    float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
    float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
    vweights31 = vsetq_lane_f32(0.f, vweights31, 3);

    float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
    float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
    vweights41 = vsetq_lane_f32(0.f, vweights41, 3);

    float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
    float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
    vweights51 = vsetq_lane_f32(0.f, vweights51, 3);

    float32x4_t vweights60 = vld1q_f32(weight_ch_in + 42);
    float32x4_t vweights61 = vld1q_f32(weight_ch_in + 46);
    vweights61 = vsetq_lane_f32(0.f, vweights61, 3);
    for (int i = 0; i < height; i++){
        const float* inptr_ch0 = din + i * width;
        const float* inptr_ch1 = inptr_ch0 + width;
        const float* inptr_ch2 = inptr_ch1 + width;
        const float* inptr_ch3 = inptr_ch2 + width;
        const float* inptr_ch4 = inptr_ch3 + width;
        const float* inptr_ch5 = inptr_ch4 + width;
        const float* wei_ptr = weight_ch_in + 7;//1
        float* outptr = dout + 3;
        int cnt = (width - 6) / 4;
        int remain = (width - 6) - cnt * 4;
        //printf("din: %x, inptr_ch0: %x, inptr_ch1: %x, weight_ch_in: %x \n", din, inptr_ch0, inptr_ch1, weight_ch_in);
#ifdef __arrch64__
        for(; cnt > 0; cnt --){
            //0
            float32x4_t vdin00 = vld1q_f32(inptr_ch0);
            float32x4_t vdin01 = vld1q_f32(inptr_ch0 + 4);
            float32x4_t vdin02 = vld1q_f32(inptr_ch0 + 8);
            float32x4_t vdin01234 = vextq_f32(vdin00, vdin01, 1);
            float32x4_t vdin02345 = vextq_f32(vdin00, vdin01, 2);
            float32x4_t vdin03456 = vextq_f32(vdin00, vdin01, 3);
            //printf("vdin00: %.2f, vdin01234: %.2f\n", vgetq_lane_f32(vdin00, 0), vgetq_lane_f32(vdin01234, 0));

            float32x4_t vsum0 = vmulq_f32(vdin00, vweights10);
            float32x4_t vsum1 = vmulq_f32(vdin01234, vweights10);
            float32x4_t vsum2 = vmulq_f32(vdin02345, vweights10);
            float32x4_t vsum3 = vmulq_f32(vdin03456, vweights10);

            float32x4_t vdin05678 = vextq_f32(vdin01, vdin02, 1);
            float32x4_t vdin06789 = vextq_f32(vdin01, vdin02, 2);
            float32x4_t vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights11);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights11);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights11);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights11);

            //1
            vdin00 = vld1q_f32(inptr_ch1);
            vdin01 = vld1q_f32(inptr_ch1 + 4);
            vdin02 = vld1q_f32(inptr_ch1 + 8);
            vdin01234 = vextq_f32(vdin00, vdin01, 1);
            vdin02345 = vextq_f32(vdin00, vdin01, 2);
            vdin03456 = vextq_f32(vdin00, vdin01, 3);

            vsum0 = vmlaq_f32(vsum0, vdin00, vweights20);
            vsum1 = vmlaq_f32(vsum1, vdin01234, vweights20);
            vsum2 = vmlaq_f32(vsum2, vdin02345, vweights20);
            vsum3 = vmlaq_f32(vsum3, vdin03456, vweights20);

            vdin05678 = vextq_f32(vdin01, vdin02, 1);
            vdin06789 = vextq_f32(vdin01, vdin02, 2);
            vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights21);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights21);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights21);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights21);

            //2
            vdin00 = vld1q_f32(inptr_ch2);
            vdin01 = vld1q_f32(inptr_ch2 + 4);
            vdin02 = vld1q_f32(inptr_ch2 + 8);
            vdin01234 = vextq_f32(vdin00, vdin01, 1);
            vdin02345 = vextq_f32(vdin00, vdin01, 2);
            vdin03456 = vextq_f32(vdin00, vdin01, 3);

            vsum0 = vmlaq_f32(vsum0, vdin00, vweights30);
            vsum1 = vmlaq_f32(vsum1, vdin01234, vweights30);
            vsum2 = vmlaq_f32(vsum2, vdin02345, vweights30);
            vsum3 = vmlaq_f32(vsum3, vdin03456, vweights30);

            vdin05678 = vextq_f32(vdin01, vdin02, 1);
            vdin06789 = vextq_f32(vdin01, vdin02, 2);
            vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights31);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights31);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights31);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights31);

            //3
            vdin00 = vld1q_f32(inptr_ch3);
            vdin01 = vld1q_f32(inptr_ch3 + 4);
            vdin02 = vld1q_f32(inptr_ch3 + 8);
            vdin01234 = vextq_f32(vdin00, vdin01, 1);
            vdin02345 = vextq_f32(vdin00, vdin01, 2);
            vdin03456 = vextq_f32(vdin00, vdin01, 3);

            vsum0 = vmlaq_f32(vsum0, vdin00, vweights40);
            vsum1 = vmlaq_f32(vsum1, vdin01234, vweights40);
            vsum2 = vmlaq_f32(vsum2, vdin02345, vweights40);
            vsum3 = vmlaq_f32(vsum3, vdin03456, vweights40);

            vdin05678 = vextq_f32(vdin01, vdin02, 1);
            vdin06789 = vextq_f32(vdin01, vdin02, 2);
            vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights41);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights41);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights41);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights41);

            //4
            vdin00 = vld1q_f32(inptr_ch4);
            vdin01 = vld1q_f32(inptr_ch4 + 4);
            vdin02 = vld1q_f32(inptr_ch4 + 8);
            vdin01234 = vextq_f32(vdin00, vdin01, 1);
            vdin02345 = vextq_f32(vdin00, vdin01, 2);
            vdin03456 = vextq_f32(vdin00, vdin01, 3);

            vsum0 = vmlaq_f32(vsum0, vdin00, vweights50);
            vsum1 = vmlaq_f32(vsum1, vdin01234, vweights50);
            vsum2 = vmlaq_f32(vsum2, vdin02345, vweights50);
            vsum3 = vmlaq_f32(vsum3, vdin03456, vweights50);

            vdin05678 = vextq_f32(vdin01, vdin02, 1);
            vdin06789 = vextq_f32(vdin01, vdin02, 2);
            vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights51);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights51);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights51);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights51);

            //5
            vdin00 = vld1q_f32(inptr_ch5);
            vdin01 = vld1q_f32(inptr_ch5 + 4);
            vdin02 = vld1q_f32(inptr_ch5 + 8);
            vdin01234 = vextq_f32(vdin00, vdin01, 1);
            vdin02345 = vextq_f32(vdin00, vdin01, 2);
            vdin03456 = vextq_f32(vdin00, vdin01, 3);

            vsum0 = vmlaq_f32(vsum0, vdin00, vweights60);
            vsum1 = vmlaq_f32(vsum1, vdin01234, vweights60);
            vsum2 = vmlaq_f32(vsum2, vdin02345, vweights60);
            vsum3 = vmlaq_f32(vsum3, vdin03456, vweights60);

            vdin05678 = vextq_f32(vdin01, vdin02, 1);
            vdin06789 = vextq_f32(vdin01, vdin02, 2);
            vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights61);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights61);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights61);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights61);

            float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum0), vget_high_f32(vsum0));
            vtotal = vpadd_f32(vtotal, vtotal);
            float32x2_t vtotal1 = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));
            vtotal1 = vpadd_f32(vtotal1, vtotal1);
            float32x2_t vtotal2 = vpadd_f32(vget_low_f32(vsum2), vget_high_f32(vsum2));
            vtotal2 = vpadd_f32(vtotal2, vtotal2);
            float32x2_t vtotal3 = vpadd_f32(vget_low_f32(vsum3), vget_high_f32(vsum3));
            vtotal3 = vpadd_f32(vtotal3, vtotal3);

            float32x4_t vsum = vdupq_n_f32(vget_lane_f32(vtotal, 0));
            vsum = vsetq_lane_f32(vget_lane_f32(vtotal1, 0), vsum, 1);
            vsum = vsetq_lane_f32(vget_lane_f32(vtotal2, 0), vsum, 2);
            vsum = vsetq_lane_f32(vget_lane_f32(vtotal3, 0), vsum, 3);
            float32x4_t vdataout = vld1q_f32(outptr);
            vdataout = vaddq_f32(vdataout, vsum);
            vst1q_f32(outptr, vdataout);
            outptr += 4;
            inptr_ch0 += 4;
            inptr_ch1 += 4;
            inptr_ch2 += 4;
            inptr_ch3 += 4;
            inptr_ch4 += 4;
            inptr_ch5 += 4;
            inptr_ch6 += 4;
        }
#else
        asm volatile(
            "cmp      %[cnt], #0                                    @ cnt > 0 \n"
            "ble       exit_top2                                    @ exit \n"
            "loop_top2:                                             @ loop \n"
            //0.0
            "vld1.f32 {d0-d1}, [%[inptr_ch0]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch0]]!                      @ load data \n"
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch0]]!                    @ load data \n"
            "vld1.f32 {d14-d15}, [%[outptr]]                        @ load out \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmul.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmul.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmul.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //0.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch0], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch1]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch1]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"
            
            //1.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch1]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //1.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch1], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                 @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch2]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch2]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //2.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch2]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //2.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch2], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch3]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch3]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //3.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch3]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //3.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch3], #32                              @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch4]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch4]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //4.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch4]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //4.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch4], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                 @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch5]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch5]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //5.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch5]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //5.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch5], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"
            "sub      %[wei_ptr], #172                              @ sub wei_ptr - 4 \n"
            "vadd.f32 q5, q10, q7                                  @ add \n"
            "vadd.f32 q6, q9, q8                                  @ add \n"
            "subs     %[cnt], #1                                   @ subs \n"
            "vadd.f32 q4, q5, q6                                  @ add \n"
            "vst1.f32 {d8-d9}, [%[outptr]]!                       @ stroe \n"
            "bne loop_top2                                        @ loop \n"
            "exit_top2:                                           @ exit \n"
            :[cnt] "+r" (cnt), [wei_ptr] "+r" (wei_ptr), [outptr] "+r" (outptr), \
            [inptr_ch0] "+r" (inptr_ch0), [inptr_ch1] "+r" (inptr_ch1), \
            [inptr_ch2] "+r" (inptr_ch2), [inptr_ch3] "+r" (inptr_ch3), \
            [inptr_ch4] "+r" (inptr_ch4), [inptr_ch5] "+r" (inptr_ch5)
            :
            :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"
        );
      // printf("inptr_ch0: %x, inptr_ch1: %x, weight_ch_in: %x, cnt: %d \n", inptr_ch0, inptr_ch1, wei_ptr, cnt);
#endif
        for (; remain > 0; remain--){
            float32x4_t vdin00 = vld1q_f32(inptr_ch0);
            float32x4_t vdin10 = vld1q_f32(inptr_ch1);
            float32x4_t vdin20 = vld1q_f32(inptr_ch2);
            float32x4_t vdin30 = vld1q_f32(inptr_ch3);
            float32x4_t vdin40 = vld1q_f32(inptr_ch4);
            float32x4_t vdin50 = vld1q_f32(inptr_ch5);
            float32x4_t vsum = vmulq_f32(vdin00, vweights10);
            vsum = vmlaq_f32(vsum, vdin10, vweights20);
            vsum = vmlaq_f32(vsum, vdin20, vweights30);
            vsum = vmlaq_f32(vsum, vdin30, vweights40);
            vsum = vmlaq_f32(vsum, vdin40, vweights50);
            vsum = vmlaq_f32(vsum, vdin50, vweights60);
            float32x4_t vdin01 = vld1q_f32(inptr_ch0 + 4);
            float32x4_t vdin11 = vld1q_f32(inptr_ch1 + 4);
            float32x4_t vdin21 = vld1q_f32(inptr_ch2 + 4);
            float32x4_t vdin31 = vld1q_f32(inptr_ch3 + 4);
            float32x4_t vdin41 = vld1q_f32(inptr_ch4 + 4);
            float32x4_t vdin51 = vld1q_f32(inptr_ch5 + 4);
            vsum = vmlaq_f32(vsum, vdin01, vweights11);
            vsum = vmlaq_f32(vsum, vdin11, vweights21);
            vsum = vmlaq_f32(vsum, vdin21, vweights31);
            vsum = vmlaq_f32(vsum, vdin31, vweights41);
            vsum = vmlaq_f32(vsum, vdin41, vweights51);
            vsum = vmlaq_f32(vsum, vdin51, vweights61);
            float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
            vtotal = vpadd_f32(vtotal, vtotal);
            float32x2_t vdataout = vld1_f32(outptr);
            vtotal = vset_lane_f32(0.f, vtotal, 1);
            vdataout = vadd_f32(vdataout, vtotal);
            vst1_f32(outptr, vdataout);
            outptr++;
            inptr_ch0++;
            inptr_ch1++;
            inptr_ch2++;
            inptr_ch3++;
            inptr_ch4++;
            inptr_ch5++;
        }
    }
}

void conv7x7_mid_top1(const float* din, float* dout, int w_out, int width, int height, const float* weight_ch_in){
    
    const float* inptr_ch0 = din;
    const float* inptr_ch1 = inptr_ch0 + width;
    const float* inptr_ch2 = inptr_ch1 + width;
    const float* inptr_ch3 = inptr_ch2 + width;
    const float* inptr_ch4 = inptr_ch3 + width;
    const float* wei_ptr = weight_ch_in + 14;//2
    float* outptr = dout + 3;
    int cnt = (width - 6) / 4;
    int remain = (width - 6) - cnt * 4;
    asm volatile(
            "cmp      %[cnt], #0                                    @ cnt > 0 \n"
            "ble       exit_top1                                    @ exit \n"
            "loop_top1:                                             @ loop \n"
            //0.0
            "vld1.f32 {d0-d1}, [%[inptr_ch0]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch0]]!                      @ load data \n"
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch0]]!                    @ load data \n"
            "vld1.f32 {d14-d15}, [%[outptr]]                        @ load out \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmul.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmul.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmul.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //0.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch0], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch1]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch1]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"
            
            //1.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch1]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //1.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch1], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                 @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch2]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch2]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //2.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch2]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //2.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch2], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch3]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch3]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //3.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch3]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //3.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch3], #32                              @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch4]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch4]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //4.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch4]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //4.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch4], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"
            "sub      %[wei_ptr], #144                              @ sub wei_ptr - 4 \n"
            "vadd.f32 q5, q10, q7                                  @ add \n"
            "vadd.f32 q6, q9, q8                                  @ add \n"
            "subs     %[cnt], #1                                   @ subs \n"
            "vadd.f32 q4, q5, q6                                  @ add \n"
            "vst1.f32 {d8-d9}, [%[outptr]]!                       @ stroe \n"
            "bne loop_top1                                        @ loop \n"
            "exit_top1:                                           @ exit \n"
            :[cnt] "+r" (cnt), [wei_ptr] "+r" (wei_ptr), [outptr] "+r" (outptr), \
            [inptr_ch0] "+r" (inptr_ch0), [inptr_ch1] "+r" (inptr_ch1), \
            [inptr_ch2] "+r" (inptr_ch2), [inptr_ch3] "+r" (inptr_ch3), \
            [inptr_ch4] "+r" (inptr_ch4)
            :
            :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"
    );

    float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
    float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
    vweights21 = vsetq_lane_f32(0.f, vweights21, 3);

    float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
    float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
    vweights31 = vsetq_lane_f32(0.f, vweights31, 3);

    float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
    float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
    vweights41 = vsetq_lane_f32(0.f, vweights41, 3);

    float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
    float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
    vweights51 = vsetq_lane_f32(0.f, vweights51, 3);

    float32x4_t vweights60 = vld1q_f32(weight_ch_in + 42);
    float32x4_t vweights61 = vld1q_f32(weight_ch_in + 46);
    vweights61 = vsetq_lane_f32(0.f, vweights61, 3);
    for (; remain > 0; remain--){
        float32x4_t vdin00 = vld1q_f32(inptr_ch0);
        float32x4_t vdin10 = vld1q_f32(inptr_ch1);
        float32x4_t vdin20 = vld1q_f32(inptr_ch2);
        float32x4_t vdin30 = vld1q_f32(inptr_ch3);
        float32x4_t vdin40 = vld1q_f32(inptr_ch4);
        float32x4_t vsum = vmulq_f32(vdin00, vweights20);
        vsum = vmlaq_f32(vsum, vdin10, vweights30);
        vsum = vmlaq_f32(vsum, vdin20, vweights40);
        vsum = vmlaq_f32(vsum, vdin30, vweights50);
        vsum = vmlaq_f32(vsum, vdin40, vweights60);
        float32x4_t vdin01 = vld1q_f32(inptr_ch0 + 4);
        float32x4_t vdin11 = vld1q_f32(inptr_ch1 + 4);
        float32x4_t vdin21 = vld1q_f32(inptr_ch2 + 4);
        float32x4_t vdin31 = vld1q_f32(inptr_ch3 + 4);
        float32x4_t vdin41 = vld1q_f32(inptr_ch4 + 4);
        vsum = vmlaq_f32(vsum, vdin01, vweights21);
        vsum = vmlaq_f32(vsum, vdin11, vweights31);
        vsum = vmlaq_f32(vsum, vdin21, vweights41);
        vsum = vmlaq_f32(vsum, vdin31, vweights51);
        vsum = vmlaq_f32(vsum, vdin41, vweights61);
        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
        vtotal = vpadd_f32(vtotal, vtotal);
        float32x2_t vdataout = vld1_f32(outptr);
        vtotal = vset_lane_f32(0.f, vtotal, 1);
        vdataout = vadd_f32(vdataout, vtotal);
        vst1_f32(outptr, vdataout);
        outptr++;
        inptr_ch0++;
        inptr_ch1++;
        inptr_ch2++;
        inptr_ch3++;
        inptr_ch4++;
    }
}

void conv7x7_mid_top0(const float* din, float* dout, int w_out, int width, int height, const float* weight_ch_in){
    
    const float* inptr_ch0 = din;
    const float* inptr_ch1 = inptr_ch0 + width;
    const float* inptr_ch2 = inptr_ch1 + width;
    const float* inptr_ch3 = inptr_ch2 + width;
    const float* wei_ptr = weight_ch_in + 21;//3
    float* outptr = dout + 3;
    int cnt = (width - 6) / 4;
    int remain = (width - 6) - cnt * 4;
    asm volatile(
            "cmp      %[cnt], #0                                    @ cnt > 0 \n"
            "ble       exit_top0                                    @ exit \n"
            "loop_top0:                                             @ loop \n"
            //0.0
            "vld1.f32 {d0-d1}, [%[inptr_ch0]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch0]]!                      @ load data \n"
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch0]]!                    @ load data \n"
            "vld1.f32 {d14-d15}, [%[outptr]]                        @ load out \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmul.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmul.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmul.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //0.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch0], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch1]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch1]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"
            
            //1.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch1]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //1.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch1], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                 @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch2]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch2]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //2.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch2]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //2.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch2], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch3]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch3]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //3.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch3]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //3.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch3], #32                              @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"
            "sub      %[wei_ptr], #116                              @ sub wei_ptr - 4 \n"
            "vadd.f32 q5, q10, q7                                  @ add \n"
            "vadd.f32 q6, q9, q8                                  @ add \n"
            "subs     %[cnt], #1                                   @ subs \n"
            "vadd.f32 q4, q5, q6                                  @ add \n"
            "vst1.f32 {d8-d9}, [%[outptr]]!                       @ stroe \n"
            "bne loop_top0                                        @ loop \n"
            "exit_top0:                                           @ exit \n"
            :[cnt] "+r" (cnt), [wei_ptr] "+r" (wei_ptr), [outptr] "+r" (outptr), \
            [inptr_ch0] "+r" (inptr_ch0), [inptr_ch1] "+r" (inptr_ch1), \
            [inptr_ch2] "+r" (inptr_ch2), [inptr_ch3] "+r" (inptr_ch3)
            :
            :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"
    );

    
    float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
    float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
    vweights31 = vsetq_lane_f32(0.f, vweights31, 3);

    float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
    float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
    vweights41 = vsetq_lane_f32(0.f, vweights41, 3);

    float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
    float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
    vweights51 = vsetq_lane_f32(0.f, vweights51, 3);

    float32x4_t vweights60 = vld1q_f32(weight_ch_in + 42);
    float32x4_t vweights61 = vld1q_f32(weight_ch_in + 46);
    vweights61 = vsetq_lane_f32(0.f, vweights61, 3);
    for (; remain > 0; remain--){
        float32x4_t vdin00 = vld1q_f32(inptr_ch0);
        float32x4_t vdin10 = vld1q_f32(inptr_ch1);
        float32x4_t vdin20 = vld1q_f32(inptr_ch2);
        float32x4_t vdin30 = vld1q_f32(inptr_ch3);
        float32x4_t vsum = vmulq_f32(vdin00, vweights30);
        vsum = vmlaq_f32(vsum, vdin10, vweights40);
        vsum = vmlaq_f32(vsum, vdin20, vweights50);
        vsum = vmlaq_f32(vsum, vdin30, vweights60);
        float32x4_t vdin01 = vld1q_f32(inptr_ch0 + 4);
        float32x4_t vdin11 = vld1q_f32(inptr_ch1 + 4);
        float32x4_t vdin21 = vld1q_f32(inptr_ch2 + 4);
        float32x4_t vdin31 = vld1q_f32(inptr_ch3 + 4);
        vsum = vmlaq_f32(vsum, vdin01, vweights31);
        vsum = vmlaq_f32(vsum, vdin11, vweights41);
        vsum = vmlaq_f32(vsum, vdin21, vweights51);
        vsum = vmlaq_f32(vsum, vdin31, vweights61);
        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
        vtotal = vpadd_f32(vtotal, vtotal);
        float32x2_t vdataout = vld1_f32(outptr);
        vtotal = vset_lane_f32(0.f, vtotal, 1);
        vdataout = vadd_f32(vdataout, vtotal);
        vst1_f32(outptr, vdataout);
        outptr++;
        inptr_ch0++;
        inptr_ch1++;
        inptr_ch2++;
        inptr_ch3++;
    }
}


void conv7x7_mid_mid(const float* din, float* dout, int w_out, int width, int height, const float* weight_ch_in){
    float32x4_t vweights00 = vld1q_f32(weight_ch_in);
    float32x4_t vweights01 = vld1q_f32(weight_ch_in + 4);
    vweights01 = vsetq_lane_f32(0.f, vweights01, 3);

    float32x4_t vweights10 = vld1q_f32(weight_ch_in + 7);
    float32x4_t vweights11 = vld1q_f32(weight_ch_in + 11);
    vweights11 = vsetq_lane_f32(0.f, vweights11, 3);

    float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
    float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
    vweights21 = vsetq_lane_f32(0.f, vweights21, 3);

    float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
    float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
    vweights31 = vsetq_lane_f32(0.f, vweights31, 3);

    float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
    float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
    vweights41 = vsetq_lane_f32(0.f, vweights41, 3);

    float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
    float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
    vweights51 = vsetq_lane_f32(0.f, vweights51, 3);

    float32x4_t vweights60 = vld1q_f32(weight_ch_in + 42);
    float32x4_t vweights61 = vld1q_f32(weight_ch_in + 46);
    vweights61 = vsetq_lane_f32(0.f, vweights61, 3);
    height -= 3;
    for (int i = 0; i < height; i++){
        const float* inptr_ch0 = din + i * width;
        const float* inptr_ch1 = inptr_ch0 + width;
        const float* inptr_ch2 = inptr_ch1 + width;
        const float* inptr_ch3 = inptr_ch2 + width;
        const float* inptr_ch4 = inptr_ch3 + width;
        const float* inptr_ch5 = inptr_ch4 + width;
        const float* inptr_ch6 = inptr_ch5 + width;
        const float* wei_ptr = weight_ch_in;
        float* outptr = dout + i * w_out + 3;
        int cnt = (width - 6) / 4;
        int remain = (width - 6) - cnt * 4;
        //printf("din: %x, inptr_ch0: %x, inptr_ch1: %x, weight_ch_in: %x \n", din, inptr_ch0, inptr_ch1, weight_ch_in);
#ifdef __arrch64__
        for(; cnt > 0; cnt --){
            //0
            float32x4_t vdin00 = vld1q_f32(inptr_ch0);
            float32x4_t vdin01 = vld1q_f32(inptr_ch0 + 4);
            float32x4_t vdin02 = vld1q_f32(inptr_ch0 + 8);
            float32x4_t vdin01234 = vextq_f32(vdin00, vdin01, 1);
            float32x4_t vdin02345 = vextq_f32(vdin00, vdin01, 2);
            float32x4_t vdin03456 = vextq_f32(vdin00, vdin01, 3);
            //printf("vdin00: %.2f, vdin01234: %.2f\n", vgetq_lane_f32(vdin00, 0), vgetq_lane_f32(vdin01234, 0));

            float32x4_t vsum0 = vmulq_f32(vdin00, vweights00);
            float32x4_t vsum1 = vmulq_f32(vdin01234, vweights00);
            float32x4_t vsum2 = vmulq_f32(vdin02345, vweights00);
            float32x4_t vsum3 = vmulq_f32(vdin03456, vweights00);

            float32x4_t vdin05678 = vextq_f32(vdin01, vdin02, 1);
            float32x4_t vdin06789 = vextq_f32(vdin01, vdin02, 2);
            float32x4_t vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights01);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights01);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights01);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights01);

            //1
            vdin00 = vld1q_f32(inptr_ch1);
            vdin01 = vld1q_f32(inptr_ch1 + 4);
            vdin02 = vld1q_f32(inptr_ch1 + 8);
            vdin01234 = vextq_f32(vdin00, vdin01, 1);
            vdin02345 = vextq_f32(vdin00, vdin01, 2);
            vdin03456 = vextq_f32(vdin00, vdin01, 3);

            vsum0 = vmlaq_f32(vsum0, vdin00, vweights10);
            vsum1 = vmlaq_f32(vsum1, vdin01234, vweights10);
            vsum2 = vmlaq_f32(vsum2, vdin02345, vweights10);
            vsum3 = vmlaq_f32(vsum3, vdin03456, vweights10);

            vdin05678 = vextq_f32(vdin01, vdin02, 1);
            vdin06789 = vextq_f32(vdin01, vdin02, 2);
            vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights11);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights11);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights11);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights11);

            //2
            vdin00 = vld1q_f32(inptr_ch2);
            vdin01 = vld1q_f32(inptr_ch2 + 4);
            vdin02 = vld1q_f32(inptr_ch2 + 8);
            vdin01234 = vextq_f32(vdin00, vdin01, 1);
            vdin02345 = vextq_f32(vdin00, vdin01, 2);
            vdin03456 = vextq_f32(vdin00, vdin01, 3);

            vsum0 = vmlaq_f32(vsum0, vdin00, vweights20);
            vsum1 = vmlaq_f32(vsum1, vdin01234, vweights20);
            vsum2 = vmlaq_f32(vsum2, vdin02345, vweights20);
            vsum3 = vmlaq_f32(vsum3, vdin03456, vweights20);

            vdin05678 = vextq_f32(vdin01, vdin02, 1);
            vdin06789 = vextq_f32(vdin01, vdin02, 2);
            vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights21);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights21);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights21);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights21);

            //3
            vdin00 = vld1q_f32(inptr_ch3);
            vdin01 = vld1q_f32(inptr_ch3 + 4);
            vdin02 = vld1q_f32(inptr_ch3 + 8);
            vdin01234 = vextq_f32(vdin00, vdin01, 1);
            vdin02345 = vextq_f32(vdin00, vdin01, 2);
            vdin03456 = vextq_f32(vdin00, vdin01, 3);

            vsum0 = vmlaq_f32(vsum0, vdin00, vweights30);
            vsum1 = vmlaq_f32(vsum1, vdin01234, vweights30);
            vsum2 = vmlaq_f32(vsum2, vdin02345, vweights30);
            vsum3 = vmlaq_f32(vsum3, vdin03456, vweights30);

            vdin05678 = vextq_f32(vdin01, vdin02, 1);
            vdin06789 = vextq_f32(vdin01, vdin02, 2);
            vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights31);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights31);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights31);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights31);

            //4
            vdin00 = vld1q_f32(inptr_ch4);
            vdin01 = vld1q_f32(inptr_ch4 + 4);
            vdin02 = vld1q_f32(inptr_ch4 + 8);
            vdin01234 = vextq_f32(vdin00, vdin01, 1);
            vdin02345 = vextq_f32(vdin00, vdin01, 2);
            vdin03456 = vextq_f32(vdin00, vdin01, 3);

            vsum0 = vmlaq_f32(vsum0, vdin00, vweights40);
            vsum1 = vmlaq_f32(vsum1, vdin01234, vweights40);
            vsum2 = vmlaq_f32(vsum2, vdin02345, vweights40);
            vsum3 = vmlaq_f32(vsum3, vdin03456, vweights40);

            vdin05678 = vextq_f32(vdin01, vdin02, 1);
            vdin06789 = vextq_f32(vdin01, vdin02, 2);
            vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights41);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights41);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights41);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights41);

            //5
            vdin00 = vld1q_f32(inptr_ch5);
            vdin01 = vld1q_f32(inptr_ch5 + 4);
            vdin02 = vld1q_f32(inptr_ch5 + 8);
            vdin01234 = vextq_f32(vdin00, vdin01, 1);
            vdin02345 = vextq_f32(vdin00, vdin01, 2);
            vdin03456 = vextq_f32(vdin00, vdin01, 3);

            vsum0 = vmlaq_f32(vsum0, vdin00, vweights50);
            vsum1 = vmlaq_f32(vsum1, vdin01234, vweights50);
            vsum2 = vmlaq_f32(vsum2, vdin02345, vweights50);
            vsum3 = vmlaq_f32(vsum3, vdin03456, vweights50);

            vdin05678 = vextq_f32(vdin01, vdin02, 1);
            vdin06789 = vextq_f32(vdin01, vdin02, 2);
            vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights51);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights51);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights51);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights51);

            //6
            vdin00 = vld1q_f32(inptr_ch6);
            vdin01 = vld1q_f32(inptr_ch6 + 4);
            vdin02 = vld1q_f32(inptr_ch6 + 8);
            vdin01234 = vextq_f32(vdin00, vdin01, 1);
            vdin02345 = vextq_f32(vdin00, vdin01, 2);
            vdin03456 = vextq_f32(vdin00, vdin01, 3);

            vsum0 = vmlaq_f32(vsum0, vdin00, vweights60);
            vsum1 = vmlaq_f32(vsum1, vdin01234, vweights60);
            vsum2 = vmlaq_f32(vsum2, vdin02345, vweights60);
            vsum3 = vmlaq_f32(vsum3, vdin03456, vweights60);

            vdin05678 = vextq_f32(vdin01, vdin02, 1);
            vdin06789 = vextq_f32(vdin01, vdin02, 2);
            vdin078910 = vextq_f32(vdin01, vdin02,3);

            vsum0 = vmlaq_f32(vsum0, vdin01, vweights61);
            vsum1 = vmlaq_f32(vsum1, vdin05678, vweights61);
            vsum2 = vmlaq_f32(vsum2, vdin06789, vweights61);
            vsum3 = vmlaq_f32(vsum3, vdin078910, vweights61);

            float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum0), vget_high_f32(vsum0));
            vtotal = vpadd_f32(vtotal, vtotal);
            float32x2_t vtotal1 = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));
            vtotal1 = vpadd_f32(vtotal1, vtotal1);
            float32x2_t vtotal2 = vpadd_f32(vget_low_f32(vsum2), vget_high_f32(vsum2));
            vtotal2 = vpadd_f32(vtotal2, vtotal2);
            float32x2_t vtotal3 = vpadd_f32(vget_low_f32(vsum3), vget_high_f32(vsum3));
            vtotal3 = vpadd_f32(vtotal3, vtotal3);

            float32x4_t vsum = vdupq_n_f32(vget_lane_f32(vtotal, 0));
            vsum = vsetq_lane_f32(vget_lane_f32(vtotal1, 0), vsum, 1);
            vsum = vsetq_lane_f32(vget_lane_f32(vtotal2, 0), vsum, 2);
            vsum = vsetq_lane_f32(vget_lane_f32(vtotal3, 0), vsum, 3);
            float32x4_t vdataout = vld1q_f32(outptr);

           // printf("vdin05678: %.2f, vdin00: %.2f \n", vgetq_lane_f32(vdin05678, 0), vgetq_lane_f32(vdin00, 0));
            //printf("outptr: %x, vdataout: %.2f \n", outptr, vgetq_lane_f32(vdataout, 0));
            vdataout = vaddq_f32(vdataout, vsum);
            //printf("outptr: %x, vdataout: %.2f \n", outptr, vgetq_lane_f32(vdataout, 0));
            vst1q_f32(outptr, vdataout);
            outptr += 4;
            inptr_ch0 += 4;
            inptr_ch1 += 4;
            inptr_ch2 += 4;
            inptr_ch3 += 4;
            inptr_ch4 += 4;
            inptr_ch5 += 4;
            inptr_ch6 += 4;
        }
#else
        asm volatile(
            "cmp      %[cnt], #0                                    @ cnt > 0 \n"
            "ble       exit1                                        @ exit \n"
            "loop1:                                                 @ loop \n"
            //0.0
            "vld1.f32 {d0-d1}, [%[inptr_ch0]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch0]]!                      @ load data \n"
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch0]]!                    @ load data \n"
            "vld1.f32 {d14-d15}, [%[outptr]]                        @ load out \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmul.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmul.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmul.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //0.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch0], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch1]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch1]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"
            
            //1.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch1]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //1.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch1], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                 @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch2]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch2]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //2.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch2]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //2.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch2], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch3]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch3]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //3.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch3]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //3.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch3], #32                              @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch4]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch4]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //4.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch4]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //4.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch4], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                 @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch5]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch5]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

             //5.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch5]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //5.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch5], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "sub      %[wei_ptr], #4                                 @ sub wei_ptr - 4 \n"
            "vld1.f32 {d0-d1}, [%[inptr_ch6]]!                      @ load data \n"
            "vld1.f32 {d2-d3}, [%[inptr_ch6]]!                      @ load data \n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"

            //6.0
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                      @ load weights00 \n"
            "vld1.f32 {d12-d13}, [%[inptr_ch6]]!                    @ load data \n"
            "vext.f32 q2, q0, q1, #1                                @ extq 1234 \n"
            "vext.f32 q3, q0, q1, #2                                @ extq 2345 \n"
            "vext.f32 q4, q0, q1, #3                                @ extq 3456 \n"
            "vmla.f32 q7, q0, d10[0]                                @ mla vdin0 * wei00[0]\n"
            "vmla.f32 q8, q2, d10[1]                                @ mla 1234 * wei00[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 2345 * wei00[2]\n"
            "vmla.f32 q10, q4, d11[1]                               @ mla 3456 * wei00[3]\n"
            //6.1
            "vld1.f32 {d10-d11}, [%[wei_ptr]]!                       @ load weights01 \n"
            "vext.f32 q2, q1, q6, #1                               @ extq 5678 \n"
            "vext.f32 q3, q1, q6, #2                                @ extq 6789 \n"
            "sub      %[inptr_ch6], #32                             @ sub inptr_ch0 - 32 \n"
            "vmla.f32 q7, q1, d10[0]                                @ mla 4567 * wei01[0]\n"
            "vmla.f32 q8, q2, d10[1]                               @ mla 5678 * wei01[1]\n"
            "vmla.f32 q9, q3, d11[0]                                @ mla 6789 * wei01[2]\n"
            "sub      %[wei_ptr], #200                             @ sub wei_ptr - 4 \n"
            "vadd.f32 q5, q10, q7                                  @ add \n"
            "vadd.f32 q6, q9, q8                                  @ add \n"
            "subs     %[cnt], #1                                   @ subs \n"
            "vadd.f32 q4, q5, q6                                  @ add \n"
            "vst1.f32 {d8-d9}, [%[outptr]]!                       @ stroe \n"
            "bne loop1                                            @ loop \n"
            "exit1:                                               @ exit \n"
            :[cnt] "+r" (cnt), [wei_ptr] "+r" (wei_ptr), [outptr] "+r" (outptr), \
            [inptr_ch0] "+r" (inptr_ch0), [inptr_ch1] "+r" (inptr_ch1), \
            [inptr_ch2] "+r" (inptr_ch2), [inptr_ch3] "+r" (inptr_ch3), \
            [inptr_ch4] "+r" (inptr_ch4), [inptr_ch5] "+r" (inptr_ch5), \
            [inptr_ch6] "+r" (inptr_ch6)
            :
            :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"
        );
      // printf("inptr_ch0: %x, inptr_ch1: %x, weight_ch_in: %x, cnt: %d \n", inptr_ch0, inptr_ch1, wei_ptr, cnt);
#endif
        for (; remain > 0; remain--){
            float32x4_t vdin00 = vld1q_f32(inptr_ch0);
            float32x4_t vdin10 = vld1q_f32(inptr_ch1);
            float32x4_t vdin20 = vld1q_f32(inptr_ch2);
            float32x4_t vdin30 = vld1q_f32(inptr_ch3);
            float32x4_t vdin40 = vld1q_f32(inptr_ch4);
            float32x4_t vdin50 = vld1q_f32(inptr_ch5);
            float32x4_t vdin60 = vld1q_f32(inptr_ch6);
            float32x4_t vsum = vmulq_f32(vdin00, vweights00);
            vsum = vmlaq_f32(vsum, vdin10, vweights10);
            vsum = vmlaq_f32(vsum, vdin20, vweights20);
            vsum = vmlaq_f32(vsum, vdin30, vweights30);
            vsum = vmlaq_f32(vsum, vdin40, vweights40);
            vsum = vmlaq_f32(vsum, vdin50, vweights50);
            vsum = vmlaq_f32(vsum, vdin60, vweights60);
            float32x4_t vdin01 = vld1q_f32(inptr_ch0 + 4);
            float32x4_t vdin11 = vld1q_f32(inptr_ch1 + 4);
            float32x4_t vdin21 = vld1q_f32(inptr_ch2 + 4);
            float32x4_t vdin31 = vld1q_f32(inptr_ch3 + 4);
            float32x4_t vdin41 = vld1q_f32(inptr_ch4 + 4);
            float32x4_t vdin51 = vld1q_f32(inptr_ch5 + 4);
            float32x4_t vdin61 = vld1q_f32(inptr_ch6 + 4);
            vsum = vmlaq_f32(vsum, vdin01, vweights01);
            vsum = vmlaq_f32(vsum, vdin11, vweights11);
            vsum = vmlaq_f32(vsum, vdin21, vweights21);
            vsum = vmlaq_f32(vsum, vdin31, vweights31);
            vsum = vmlaq_f32(vsum, vdin41, vweights41);
            vsum = vmlaq_f32(vsum, vdin51, vweights51);
            vsum = vmlaq_f32(vsum, vdin61, vweights61);
            float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
            vtotal = vpadd_f32(vtotal, vtotal);
            float32x2_t vdataout = vld1_f32(outptr);
            vtotal = vset_lane_f32(0.f, vtotal, 1);
            vdataout = vadd_f32(vdataout, vtotal);
            vst1_f32(outptr, vdataout);
            outptr++;
            inptr_ch0++;
            inptr_ch1++;
            inptr_ch2++;
            inptr_ch3++;
            inptr_ch4++;
            inptr_ch5++;
            inptr_ch6++;
        }
    }
}

void conv_7x7s1_direct(const float* din, float* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const float* weights, const float* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Sgemm& gemmer, void* work_space) {

    int w_in = win;
    int h_in = hin;
    int ch_in = chin;

    int w_out = wout;
    int h_out = hout;
    int ch_out = chout;

    const int size_kernel = kernel_h * kernel_w;

    const int ch_out_g = ch_out / group;
    const int ch_in_g = ch_in / group;
    const int size_in_channel = w_in * h_in;
    const int size_in_batch = size_in_channel * ch_in;
    const int size_out_channel = w_out * h_out;
    const int size_out_batch = size_out_channel * ch_out;

    //printf("extend kernel size: %d, %d\n", kernel_ext_w, kernel_ext_h);
    const float *data_in = din;
    float *outptr = dout;

    int kernel_w_even = kernel_w >> 1;
    int kernel_h_even = kernel_h >> 1;

    for (int b = 0; b < num; ++b) {
        float *outptr_batch = outptr + b * size_out_batch;
        const float* data_in_batch = data_in + b * size_in_batch;
//#pragma omp parallel for collapse(2)
        for (int g = 0; g < group; ++g) {
#pragma omp parallel for
            for (int c = 0; c < ch_out_g; ++c) {
                const float *inptr_group = data_in_batch + g * ch_in_g * size_in_channel;
                float *outptr_ch = outptr_batch + (g * ch_out_g + c) * size_out_channel;
                const float *weight_ch = weights + (g * ch_out_g + c) * ch_in_g * size_kernel;

                float bias_value = flag_bias? bias[g * ch_out_g + c] : 0.f;
                fill_bias(outptr_ch, &bias_value, 1, w_out * h_out);
                for(int cin = 0; cin < ch_in_g; cin++){
                    const float *inptr_ch = inptr_group + cin * size_in_channel;
                    const float *weight_ch_in = weight_ch + cin * size_kernel;
                    float32x4_t vzero = vdupq_n_f32(0.f);
                    const float *inptr_ch0 = inptr_ch;
                    const float *inptr_ch1 = inptr_ch0 + w_in;
                    const float *inptr_ch2 = inptr_ch1 + w_in;
                    const float *inptr_ch3 = inptr_ch2 + w_in;
                    const float *inptr_ch4 = inptr_ch3 + w_in;
                    const float *inptr_ch5 = inptr_ch4 + w_in;
                    const float *inptr_ch6 = inptr_ch5 + w_in;

                    //mid
                    float* outptr_imd = outptr_ch ;
                    const float* inptr_imd = inptr_ch0;
                    const float* wei_ptr = weight_ch_in;
                    conv7x7_mid_top0(inptr_imd, outptr_imd, w_out, w_in, 1, wei_ptr);

                    outptr_imd = outptr_ch + 1 * w_out;
                    inptr_imd = inptr_ch0;
                    wei_ptr = weight_ch_in;
                    conv7x7_mid_top1(inptr_imd, outptr_imd, w_out, w_in, 1, wei_ptr);

                    outptr_imd = outptr_ch + 2 * w_out;
                    inptr_imd = inptr_ch0;
                    wei_ptr = weight_ch_in;
                    conv7x7_mid_top2(inptr_imd, outptr_imd, w_out, w_in, 1, wei_ptr);

                    outptr_imd = outptr_ch + 3 * w_out;
                    inptr_imd = inptr_ch0;
                    wei_ptr = weight_ch_in;
                    conv7x7_mid_mid(inptr_imd, outptr_imd, w_out, w_in, h_out - kernel_h_even, wei_ptr);

                    outptr_imd = outptr_ch + (h_out - 3) * w_out;
                    inptr_imd = inptr_ch + (h_out - kernel_h + 1) * w_in;//1
                    wei_ptr = weight_ch_in - 7;
                    conv7x7_mid_top2(inptr_imd, outptr_imd, w_out, w_in, 1, wei_ptr);

                    outptr_imd = outptr_ch + (h_out - 2) * w_out;
                    inptr_imd = inptr_ch + (h_out - kernel_h + 2) * w_in;//1
                    wei_ptr = weight_ch_in - 14;
                    conv7x7_mid_top1(inptr_imd, outptr_imd, w_out, w_in, 1, wei_ptr);

                    outptr_imd = outptr_ch + (h_out - 1) * w_out;
                    inptr_imd = inptr_ch + (h_out - kernel_h + 3) * w_in;//1
                    wei_ptr = weight_ch_in - 21;
                    conv7x7_mid_top0(inptr_imd, outptr_imd, w_out, w_in, 1, wei_ptr);

                    int h = 0;
                    //border
                    for(; h < h_out; h++){ // 0-3
                        float *outptr_ch_wh = outptr_ch + h * w_out;
                        if (h > kernel_h_even && h < h_out - kernel_h_even){
                            inptr_ch0 = inptr_ch + (h - kernel_h_even) * w_in;
                            inptr_ch1 = inptr_ch0 + w_in;
                            inptr_ch2 = inptr_ch1 + w_in;
                            inptr_ch3 = inptr_ch2 + w_in;
                            inptr_ch4 = inptr_ch3 + w_in;
                            inptr_ch5 = inptr_ch4 + w_in;
                            inptr_ch6 = inptr_ch5 + w_in;
                        }else{
                            if(h <= kernel_h_even)
                                inptr_ch0 = inptr_ch;
                            else
                                inptr_ch0 = inptr_ch + (h_out - kernel_h) * w_in;
                            inptr_ch1 = inptr_ch0 + w_in;
                            inptr_ch2 = inptr_ch1 + w_in;
                            inptr_ch3 = inptr_ch2 + w_in;
                            inptr_ch4 = inptr_ch3 + w_in;
                            inptr_ch5 = inptr_ch4 + w_in;
                            inptr_ch6 = inptr_ch5 + w_in;
                        }
                        if (h < h_out - kernel_h_even){
                            int w = 0;
                            for(w = 0; w < 3; w++){
                                if (h == 0){//0
                                    //load weights 7x7
                                    float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                                    float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                                    vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                                    float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
                                    float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
                                    vweights41 = vsetq_lane_f32(0.f, vweights41, 3);
                                    float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
                                    float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
                                    vweights51 = vsetq_lane_f32(0.f, vweights51, 3);
                                    float32x4_t vweights60 = vld1q_f32(weight_ch_in + 42);
                                    float32x4_t vweights61 = vld1q_f32(weight_ch_in + 46);
                                    vweights61 = vsetq_lane_f32(0.f, vweights61, 3);
                                    if (w == 0){//3456
                                        float32x4_t vw0_3456 = vextq_f32(vweights30, vweights31, 3);
                                        float32x4_t vw1_3456 = vextq_f32(vweights40, vweights41, 3);
                                        float32x4_t vw2_3456 = vextq_f32(vweights50, vweights51, 3);
                                        float32x4_t vw3_3456 = vextq_f32(vweights60, vweights61, 3);
                                        float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                        float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                        float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                        float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                        float32x4_t vsum = vmulq_f32(vdatain00, vw0_3456);
                                        vsum = vmlaq_f32(vsum, vdatain10, vw1_3456);
                                        vsum = vmlaq_f32(vsum, vdatain20, vw2_3456);
                                        vsum = vmlaq_f32(vsum, vdatain30, vw3_3456);
                                        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                        float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                        vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                        vtotal = vset_lane_f32(0.f, vtotal, 1);
                                        vdataout = vadd_f32(vtotal, vdataout);
                                        vst1_f32(outptr_ch_wh, vdataout);
                                        if(flag_relu)
                                            outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                        outptr_ch_wh++;
                                        continue;
                                    } else if (w == 1){//23456
                                        float32x4_t vw0_2345 = vextq_f32(vweights30, vweights31, 2);
                                        float32x4_t vw1_2345 = vextq_f32(vweights40, vweights41, 2);
                                        float32x4_t vw2_2345 = vextq_f32(vweights50, vweights51, 2);
                                        float32x4_t vw3_2345 = vextq_f32(vweights60, vweights61, 2);
                                        float32x4_t vw0_6700 = vextq_f32(vweights31, vzero, 2);
                                        float32x4_t vw1_6700 = vextq_f32(vweights41, vzero, 2);
                                        float32x4_t vw2_6700 = vextq_f32(vweights51, vzero, 2);
                                        float32x4_t vw3_6700 = vextq_f32(vweights61, vzero, 2);
                                        float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                        float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                        float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                        float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                        float32x4_t vsum = vmulq_f32(vdatain00, vw0_2345);
                                        vsum = vmlaq_f32(vsum, vdatain10, vw1_2345);
                                        vsum = vmlaq_f32(vsum, vdatain20, vw2_2345);
                                        vsum = vmlaq_f32(vsum, vdatain30, vw3_2345);
                                        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                        vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                        float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                        vsum1 = vextq_f32(vsum1, vzero, 3);
                                        float32x4_t vdatain01 = vld1q_f32(inptr_ch0 + 4);
                                        float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                        float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                        float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                        vsum1 = vmlaq_f32(vsum1, vdatain01, vw0_6700);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain11, vw1_6700);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain21, vw2_6700);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain31, vw3_6700);//6
                                        vtotal = vget_low_f32(vsum1);
                                        float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                       // vtotal = vset_lane_f32(0.f, vtotal, 1);
                                        vdataout = vadd_f32(vtotal, vdataout);
                                        vst1_f32(outptr_ch_wh, vdataout);
                                        if(flag_relu)
                                            outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                        outptr_ch_wh++;
                                        continue;
                                    } else if (w == 2){//123456
                                        float32x4_t vw0_1234 = vextq_f32(vweights30, vweights31, 1);
                                        float32x4_t vw1_1234 = vextq_f32(vweights40, vweights41, 1);
                                        float32x4_t vw2_1234 = vextq_f32(vweights50, vweights51, 1);
                                        float32x4_t vw3_1234 = vextq_f32(vweights60, vweights61, 1);
                                        float32x4_t vw0_5670 = vextq_f32(vweights31, vzero, 1);
                                        float32x4_t vw1_5670 = vextq_f32(vweights41, vzero, 1);
                                        float32x4_t vw2_5670 = vextq_f32(vweights51, vzero, 1);
                                        float32x4_t vw3_5670 = vextq_f32(vweights61, vzero, 1);
                                        float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                        float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                        float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                        float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                        float32x4_t vsum = vmulq_f32(vdatain00, vw0_1234);
                                        vsum = vmlaq_f32(vsum, vdatain10, vw1_1234);
                                        vsum = vmlaq_f32(vsum, vdatain20, vw2_1234);
                                        vsum = vmlaq_f32(vsum, vdatain30, vw3_1234);
                                        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                        vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                        float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                        vsum1 = vextq_f32(vsum1, vzero, 3);
                                        float32x4_t vdatain01 = vld1q_f32(inptr_ch0 + 4);
                                        float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                        float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                        float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                        vsum1 = vmlaq_f32(vsum1, vdatain01, vw0_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain11, vw1_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain21, vw2_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain31, vw3_5670);//6
                                        vtotal = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));//x, 0
                                        float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                        vdataout = vadd_f32(vtotal, vdataout);
                                        vst1_f32(outptr_ch_wh, vdataout);
                                        if(flag_relu)
                                            outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                        outptr_ch_wh++;
                                        continue;
                                    } 
                                } else if (h == 1){//1
                                    //load weights 7x7
                                    float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
                                    float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
                                    vweights21 = vsetq_lane_f32(0.f, vweights21, 3);
                                    float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                                    float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                                    vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                                    float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
                                    float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
                                    vweights41 = vsetq_lane_f32(0.f, vweights41, 3);
                                    float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
                                    float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
                                    vweights51 = vsetq_lane_f32(0.f, vweights51, 3);
                                    float32x4_t vweights60 = vld1q_f32(weight_ch_in + 42);
                                    float32x4_t vweights61 = vld1q_f32(weight_ch_in + 46);
                                    vweights61 = vsetq_lane_f32(0.f, vweights61, 3);
                                    if (w == 0){//3456
                                        float32x4_t vw0_3456 = vextq_f32(vweights20, vweights21, 3);
                                        float32x4_t vw1_3456 = vextq_f32(vweights30, vweights31, 3);
                                        float32x4_t vw2_3456 = vextq_f32(vweights40, vweights41, 3);
                                        float32x4_t vw3_3456 = vextq_f32(vweights50, vweights51, 3);
                                        float32x4_t vw4_3456 = vextq_f32(vweights60, vweights61, 3);
                                        float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                        float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                        float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                        float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                        float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                        float32x4_t vsum = vmulq_f32(vdatain00, vw0_3456);
                                        vsum = vmlaq_f32(vsum, vdatain10, vw1_3456);
                                        vsum = vmlaq_f32(vsum, vdatain20, vw2_3456);
                                        vsum = vmlaq_f32(vsum, vdatain30, vw3_3456);
                                        vsum = vmlaq_f32(vsum, vdatain40, vw4_3456);
                                        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                        vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                        vtotal = vset_lane_f32(0.f, vtotal, 1);
                                        float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                        vdataout = vadd_f32(vtotal, vdataout);
                                        vst1_f32(outptr_ch_wh, vdataout);
                                        if(flag_relu)
                                            outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                        outptr_ch_wh++;
                                        continue;
                                    }else if (w == 1){//23456
                                        float32x4_t vw0_2345 = vextq_f32(vweights20, vweights21, 2);
                                        float32x4_t vw1_2345 = vextq_f32(vweights30, vweights31, 2);
                                        float32x4_t vw2_2345 = vextq_f32(vweights40, vweights41, 2);
                                        float32x4_t vw3_2345 = vextq_f32(vweights50, vweights51, 2);
                                        float32x4_t vw4_2345 = vextq_f32(vweights60, vweights61, 2);
                                        float32x4_t vw0_6700 = vextq_f32(vweights21, vzero, 2);
                                        float32x4_t vw1_6700 = vextq_f32(vweights31, vzero, 2);
                                        float32x4_t vw2_6700 = vextq_f32(vweights41, vzero, 2);
                                        float32x4_t vw3_6700 = vextq_f32(vweights51, vzero, 2);
                                        float32x4_t vw4_6700 = vextq_f32(vweights61, vzero, 2);
                                        float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                        float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                        float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                        float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                        float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                        float32x4_t vsum = vmulq_f32(vdatain00, vw0_2345);
                                        vsum = vmlaq_f32(vsum, vdatain10, vw1_2345);
                                        vsum = vmlaq_f32(vsum, vdatain20, vw2_2345);
                                        vsum = vmlaq_f32(vsum, vdatain30, vw3_2345);
                                        vsum = vmlaq_f32(vsum, vdatain40, vw4_2345);
                                        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                        vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                        float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                        vsum1 = vextq_f32(vsum1, vzero, 3);
                                        float32x4_t vdatain01 = vld1q_f32(inptr_ch0 + 4);
                                        float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                        float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                        float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                        float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                        vsum1 = vmlaq_f32(vsum1, vdatain01, vw0_6700);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain11, vw1_6700);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain21, vw2_6700);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain31, vw3_6700);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain41, vw4_6700);//6
                                        vtotal = vget_low_f32(vsum1);
                                        float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                        vdataout = vadd_f32(vtotal, vdataout);
                                        vst1_f32(outptr_ch_wh, vdataout);
                                        if(flag_relu)
                                            outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                        outptr_ch_wh++;
                                        continue;
                                    }else if (w == 2){//123456
                                        float32x4_t vw0_1234 = vextq_f32(vweights20, vweights21, 1);
                                        float32x4_t vw1_1234 = vextq_f32(vweights30, vweights31, 1);
                                        float32x4_t vw2_1234 = vextq_f32(vweights40, vweights41, 1);
                                        float32x4_t vw3_1234 = vextq_f32(vweights50, vweights51, 1);
                                        float32x4_t vw4_1234 = vextq_f32(vweights60, vweights61, 1);

                                        float32x4_t vw0_5670 = vextq_f32(vweights21, vzero, 1);
                                        float32x4_t vw1_5670 = vextq_f32(vweights31, vzero, 1);
                                        float32x4_t vw2_5670 = vextq_f32(vweights41, vzero, 1);
                                        float32x4_t vw3_5670 = vextq_f32(vweights51, vzero, 1);
                                        float32x4_t vw4_5670 = vextq_f32(vweights61, vzero, 1);

                                        float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                        float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                        float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                        float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                        float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                        float32x4_t vsum = vmulq_f32(vdatain00, vw0_1234);
                                        vsum = vmlaq_f32(vsum, vdatain10, vw1_1234);
                                        vsum = vmlaq_f32(vsum, vdatain20, vw2_1234);
                                        vsum = vmlaq_f32(vsum, vdatain30, vw3_1234);
                                        vsum = vmlaq_f32(vsum, vdatain40, vw4_1234);
                                        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                        vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                        float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                        vsum1 = vextq_f32(vsum1, vzero, 3);

                                        float32x4_t vdatain01 = vld1q_f32(inptr_ch0 + 4);
                                        float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                        float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                        float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                        float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                        vsum1 = vmlaq_f32(vsum1, vdatain01, vw0_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain11, vw1_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain21, vw2_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain31, vw3_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain41, vw4_5670);//6
                                        vtotal = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));//x, 0
                                        float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                        vdataout = vadd_f32(vtotal, vdataout);
                                        vst1_f32(outptr_ch_wh, vdataout);
                                        if(flag_relu)
                                            outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                        outptr_ch_wh++;
                                        continue;
                                    }
                                } else if (h == 2){//2 
                                    //load weights 7x7
                                    float32x4_t vweights10 = vld1q_f32(weight_ch_in + 7);
                                    float32x4_t vweights11 = vld1q_f32(weight_ch_in + 11);
                                    vweights11 = vsetq_lane_f32(0.f, vweights11, 3);
                                    float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
                                    float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
                                    vweights21 = vsetq_lane_f32(0.f, vweights21, 3);
                                    float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                                    float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                                    vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                                    float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
                                    float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
                                    vweights41 = vsetq_lane_f32(0.f, vweights41, 3);
                                    float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
                                    float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
                                    vweights51 = vsetq_lane_f32(0.f, vweights51, 3);
                                    float32x4_t vweights60 = vld1q_f32(weight_ch_in + 42);
                                    float32x4_t vweights61 = vld1q_f32(weight_ch_in + 46);
                                    vweights61 = vsetq_lane_f32(0.f, vweights61, 3);
                                    
                                    if (w == 0){//3456
                                        float32x4_t vw0_3456 = vextq_f32(vweights10, vweights11, 3);
                                        float32x4_t vw1_3456 = vextq_f32(vweights20, vweights21, 3);
                                        float32x4_t vw2_3456 = vextq_f32(vweights30, vweights31, 3);
                                        float32x4_t vw3_3456 = vextq_f32(vweights40, vweights41, 3);
                                        float32x4_t vw4_3456 = vextq_f32(vweights50, vweights51, 3);
                                        float32x4_t vw5_3456 = vextq_f32(vweights60, vweights61, 3);
                                        float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                        float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                        float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                        float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                        float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                        float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                        float32x4_t vsum = vmulq_f32(vdatain00, vw0_3456);
                                        vsum = vmlaq_f32(vsum, vdatain10, vw1_3456);
                                        vsum = vmlaq_f32(vsum, vdatain20, vw2_3456);
                                        vsum = vmlaq_f32(vsum, vdatain30, vw3_3456);
                                        vsum = vmlaq_f32(vsum, vdatain40, vw4_3456);
                                        vsum = vmlaq_f32(vsum, vdatain50, vw5_3456);
                                        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                        vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                        vtotal = vset_lane_f32(0.f, vtotal, 1);
                                        float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                        vdataout = vadd_f32(vtotal, vdataout);
                                        vst1_f32(outptr_ch_wh, vdataout);
                                        if(flag_relu)
                                            outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                        outptr_ch_wh++;
                                        continue;
                                    }else if (w == 1){//23456
                                        float32x4_t vw0_2345 = vextq_f32(vweights10, vweights11, 2);
                                        float32x4_t vw1_2345 = vextq_f32(vweights20, vweights21, 2);
                                        float32x4_t vw2_2345 = vextq_f32(vweights30, vweights31, 2);
                                        float32x4_t vw3_2345 = vextq_f32(vweights40, vweights41, 2);
                                        float32x4_t vw4_2345 = vextq_f32(vweights50, vweights51, 2);
                                        float32x4_t vw5_2345 = vextq_f32(vweights60, vweights61, 2);
                                        float32x4_t vw0_6700 = vextq_f32(vweights11, vzero, 2);
                                        float32x4_t vw1_6700 = vextq_f32(vweights21, vzero, 2);
                                        float32x4_t vw2_6700 = vextq_f32(vweights31, vzero, 2);
                                        float32x4_t vw3_6700 = vextq_f32(vweights41, vzero, 2);
                                        float32x4_t vw4_6700 = vextq_f32(vweights51, vzero, 2);
                                        float32x4_t vw5_6700 = vextq_f32(vweights61, vzero, 2);

                                        float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                        float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                        float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                        float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                        float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                        float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                        float32x4_t vsum = vmulq_f32(vdatain00, vw0_2345);
                                        vsum = vmlaq_f32(vsum, vdatain10, vw1_2345);
                                        vsum = vmlaq_f32(vsum, vdatain20, vw2_2345);
                                        vsum = vmlaq_f32(vsum, vdatain30, vw3_2345);
                                        vsum = vmlaq_f32(vsum, vdatain40, vw4_2345);
                                        vsum = vmlaq_f32(vsum, vdatain50, vw5_2345);
                                        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                        vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                        float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                        vsum1 = vextq_f32(vsum1, vzero, 3);
                                        
                                        float32x4_t vdatain01 = vld1q_f32(inptr_ch0 + 4);
                                        float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                        float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                        float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                        float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                        float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);

                                        vsum1 = vmlaq_f32(vsum1, vdatain01, vw0_6700);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain11, vw1_6700);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain21, vw2_6700);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain31, vw3_6700);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain41, vw4_6700);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain51, vw5_6700);//6
                                        vtotal = vget_low_f32(vsum1);
                                        float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                        vdataout = vadd_f32(vtotal, vdataout);
                                        vst1_f32(outptr_ch_wh, vdataout);
                                        if(flag_relu)
                                            outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                        outptr_ch_wh++;
                                        continue;
                                    }else if (w == 2){//123456
                                       
                                        float32x4_t vw0_1234 = vextq_f32(vweights10, vweights11, 1);
                                        float32x4_t vw1_1234 = vextq_f32(vweights20, vweights21, 1);
                                        float32x4_t vw2_1234 = vextq_f32(vweights30, vweights31, 1);
                                        float32x4_t vw3_1234 = vextq_f32(vweights40, vweights41, 1);
                                        float32x4_t vw4_1234 = vextq_f32(vweights50, vweights51, 1);
                                        float32x4_t vw5_1234 = vextq_f32(vweights60, vweights61, 1);
                                        float32x4_t vw0_5670 = vextq_f32(vweights11, vzero, 1);
                                        float32x4_t vw1_5670 = vextq_f32(vweights21, vzero, 1);
                                        float32x4_t vw2_5670 = vextq_f32(vweights31, vzero, 1);
                                        float32x4_t vw3_5670 = vextq_f32(vweights41, vzero, 1);
                                        float32x4_t vw4_5670 = vextq_f32(vweights51, vzero, 1);
                                        float32x4_t vw5_5670 = vextq_f32(vweights61, vzero, 1);
                                        float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                        float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                        float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                        float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                        float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                        float32x4_t vdatain50 = vld1q_f32(inptr_ch5);

                                        float32x4_t vsum = vmulq_f32(vdatain00, vw0_1234);
                                        vsum = vmlaq_f32(vsum, vdatain10, vw1_1234);
                                        vsum = vmlaq_f32(vsum, vdatain20, vw2_1234);
                                        vsum = vmlaq_f32(vsum, vdatain30, vw3_1234);
                                        vsum = vmlaq_f32(vsum, vdatain40, vw4_1234);
                                        vsum = vmlaq_f32(vsum, vdatain50, vw5_1234);
                                        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                        vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                        float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                        vsum1 = vextq_f32(vsum1, vzero, 3);
                                        float32x4_t vdatain01 = vld1q_f32(inptr_ch0 + 4);
                                        float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                        float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                        float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                        float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                        float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);

                                        vsum1 = vmlaq_f32(vsum1, vdatain01, vw0_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain11, vw1_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain21, vw2_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain31, vw3_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain41, vw4_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain51, vw5_5670);//6
                                        vtotal = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));//x, 0
                                        float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                        vdataout = vadd_f32(vtotal, vdataout);
                                        vst1_f32(outptr_ch_wh, vdataout);
                                        if(flag_relu)
                                            outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                        outptr_ch_wh++;
                                        continue;
                                    }
                                }else{//mid
                                    //load weights 7x7
                                    float32x4_t vweights00 = vld1q_f32(weight_ch_in);
                                    float32x4_t vweights01 = vld1q_f32(weight_ch_in + 4);
                                    vweights01 = vsetq_lane_f32(0.f, vweights01, 3);
                                    float32x4_t vweights10 = vld1q_f32(weight_ch_in + 7);
                                    float32x4_t vweights11 = vld1q_f32(weight_ch_in + 11);
                                    vweights11 = vsetq_lane_f32(0.f, vweights11, 3);
                                    float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
                                    float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
                                    vweights21 = vsetq_lane_f32(0.f, vweights21, 3);
                                    float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                                    float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                                    vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                                    float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
                                    float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
                                    vweights41 = vsetq_lane_f32(0.f, vweights41, 3);
                                    float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
                                    float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
                                    vweights51 = vsetq_lane_f32(0.f, vweights51, 3);
                                    float32x4_t vweights60 = vld1q_f32(weight_ch_in + 42);
                                    float32x4_t vweights61 = vld1q_f32(weight_ch_in + 46);
                                    vweights61 = vsetq_lane_f32(0.f, vweights61, 3);
                                   if (w == 0){//3456
                                        float32x4_t vw0_3456 = vextq_f32(vweights00, vweights01, 3);
                                        float32x4_t vw1_3456 = vextq_f32(vweights10, vweights11, 3);
                                        float32x4_t vw2_3456 = vextq_f32(vweights20, vweights21, 3);
                                        float32x4_t vw3_3456 = vextq_f32(vweights30, vweights31, 3);
                                        float32x4_t vw4_3456 = vextq_f32(vweights40, vweights41, 3);
                                        float32x4_t vw5_3456 = vextq_f32(vweights50, vweights51, 3);
                                        float32x4_t vw6_3456 = vextq_f32(vweights60, vweights61, 3);
                                        float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                        float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                        float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                        float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                        float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                        float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                        float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                                        float32x4_t vsum = vmulq_f32(vdatain00, vw0_3456);
                                        vsum = vmlaq_f32(vsum, vdatain10, vw1_3456);
                                        vsum = vmlaq_f32(vsum, vdatain20, vw2_3456);
                                        vsum = vmlaq_f32(vsum, vdatain30, vw3_3456);
                                        vsum = vmlaq_f32(vsum, vdatain40, vw4_3456);
                                        vsum = vmlaq_f32(vsum, vdatain50, vw5_3456);
                                        vsum = vmlaq_f32(vsum, vdatain60, vw6_3456);
                                        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                        vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                        vtotal = vset_lane_f32(0.f, vtotal, 1);
                                        float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                        vdataout = vadd_f32(vtotal, vdataout);
                                        vst1_f32(outptr_ch_wh, vdataout);
                                        if(flag_relu)
                                            outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                        outptr_ch_wh++;
                                        continue;
                                    }else if (w == 1){//23456
                                        float32x4_t vw0_2345 = vextq_f32(vweights00, vweights01, 2);
                                        float32x4_t vw1_2345 = vextq_f32(vweights10, vweights11, 2);
                                        float32x4_t vw2_2345 = vextq_f32(vweights20, vweights21, 2);
                                        float32x4_t vw3_2345 = vextq_f32(vweights30, vweights31, 2);
                                        float32x4_t vw4_2345 = vextq_f32(vweights40, vweights41, 2);
                                        float32x4_t vw5_2345 = vextq_f32(vweights50, vweights51, 2);
                                        float32x4_t vw6_2345 = vextq_f32(vweights60, vweights61, 2);

                                        float32x4_t vw0_5670 = vextq_f32(vweights01, vzero, 2);
                                        float32x4_t vw1_5670 = vextq_f32(vweights11, vzero, 2);
                                        float32x4_t vw2_5670 = vextq_f32(vweights21, vzero, 2);
                                        float32x4_t vw3_5670 = vextq_f32(vweights31, vzero, 2);
                                        float32x4_t vw4_5670 = vextq_f32(vweights41, vzero, 2);
                                        float32x4_t vw5_5670 = vextq_f32(vweights51, vzero, 2);
                                        float32x4_t vw6_5670 = vextq_f32(vweights61, vzero, 2);

                                        float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                                        float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                        float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                        float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                        float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                        float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                        float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                        float32x4_t vsum = vmulq_f32(vdatain00, vw0_2345);
                                        vsum = vmlaq_f32(vsum, vdatain10, vw1_2345);
                                        vsum = vmlaq_f32(vsum, vdatain20, vw2_2345);
                                        vsum = vmlaq_f32(vsum, vdatain30, vw3_2345);
                                        vsum = vmlaq_f32(vsum, vdatain40, vw4_2345);
                                        vsum = vmlaq_f32(vsum, vdatain50, vw5_2345);
                                        vsum = vmlaq_f32(vsum, vdatain60, vw6_2345);
                                        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                        vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                        float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                        vsum1 = vextq_f32(vsum1, vzero, 3);

                                        float32x4_t vdatain01 = vld1q_f32(inptr_ch0 + 4);
                                        float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                        float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                        float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                        float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                        float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);
                                        float32x4_t vdatain61 = vld1q_f32(inptr_ch6 + 4);
                                        vsum1 = vmlaq_f32(vsum1, vdatain01, vw0_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain11, vw1_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain21, vw2_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain31, vw3_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain41, vw4_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain51, vw5_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain61, vw6_5670);//6
                                        vtotal = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));//x, 0
                                        float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                        vdataout = vadd_f32(vtotal, vdataout);
                                        vst1_f32(outptr_ch_wh, vdataout);
                                        if(flag_relu)
                                            outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                        outptr_ch_wh++;
                                        continue;
                                    }else if (w == 2){//123456
                                        float32x4_t vw0_5670 = vextq_f32(vweights01, vzero, 1);
                                        float32x4_t vw1_5670 = vextq_f32(vweights11, vzero, 1);
                                        float32x4_t vw2_5670 = vextq_f32(vweights21, vzero, 1);
                                        float32x4_t vw3_5670 = vextq_f32(vweights31, vzero, 1);
                                        float32x4_t vw4_5670 = vextq_f32(vweights41, vzero, 1);
                                        float32x4_t vw5_5670 = vextq_f32(vweights51, vzero, 1);
                                        float32x4_t vw6_5670 = vextq_f32(vweights61, vzero, 1); 
                                        
                                        float32x4_t vw0_1234 = vextq_f32(vweights00, vweights01, 1);
                                        float32x4_t vw1_1234 = vextq_f32(vweights10, vweights11, 1);
                                        float32x4_t vw2_1234 = vextq_f32(vweights20, vweights21, 1);
                                        float32x4_t vw3_1234 = vextq_f32(vweights30, vweights31, 1);
                                        float32x4_t vw4_1234 = vextq_f32(vweights40, vweights41, 1);
                                        float32x4_t vw5_1234 = vextq_f32(vweights50, vweights51, 1);
                                        float32x4_t vw6_1234 = vextq_f32(vweights60, vweights61, 1);
                                       // printf("vw0_1234_0: %.2f, vw0_1234_1: %.2f, vw0_1234_2: %.2f, vw0_1234_3: %.2f\n", \ 
                                        //    vgetq_lane_f32(vw0_1234,0), vgetq_lane_f32(vw0_1234,1), vgetq_lane_f32(vw0_1234,2), vgetq_lane_f32(vw0_1234,3));

                                        float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                        float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                        float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                        float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                        float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                        float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                        float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                                        float32x4_t vsum = vmulq_f32(vdatain00, vw0_1234);
                                        vsum = vmlaq_f32(vsum, vdatain10, vw1_1234);
                                        vsum = vmlaq_f32(vsum, vdatain20, vw2_1234);
                                        vsum = vmlaq_f32(vsum, vdatain30, vw3_1234);
                                        vsum = vmlaq_f32(vsum, vdatain40, vw4_1234);
                                        vsum = vmlaq_f32(vsum, vdatain50, vw5_1234);
                                        vsum = vmlaq_f32(vsum, vdatain60, vw6_1234);
                                        float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                        vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                        float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                        vsum1 = vextq_f32(vsum1, vzero, 3);
                                       
                                        
                                        float32x4_t vdatain01 = vld1q_f32(inptr_ch0 + 4);
                                        float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                        float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                        float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                        float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                        float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);
                                        float32x4_t vdatain61 = vld1q_f32(inptr_ch6 + 4);
                                       // printf("vw0_5670_0: %.2f, vw0_5670_1: %.2f, vw0_5670_2: %.2f, vw0_5670_3: %.2f\n", \ 
                                         //   vgetq_lane_f32(vw0_5670,0), vgetq_lane_f32(vw0_5670,1), vgetq_lane_f32(vw0_5670,2), vgetq_lane_f32(vw0_5670,3));
                                        
                                        vsum1 = vmlaq_f32(vsum1, vdatain01, vw0_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain11, vw1_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain21, vw2_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain31, vw3_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain41, vw4_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain51, vw5_5670);//6
                                        vsum1 = vmlaq_f32(vsum1, vdatain61, vw6_5670);//6
                                        vtotal = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));//x, 0
                                        //printf("vtotal_0: %.2f, vatotal_1: %.2f, vdataout_0: %.2f, vdataout_1: %.2f\n", \ 
                                        //    vget_lane_f32(vtotal,0), vget_lane_f32(vtotal,1), vget_lane_f32(vdataout,0), vget_lane_f32(vdataout,1));
                                        float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                        vdataout = vadd_f32(vdataout, vtotal);
                                       // printf("vtotal_0: %.2f, vatotal_1: %.2f, vdataout_0: %.2f, vdataout_1: %.2f\n", \ 
                                         //   vget_lane_f32(vtotal,0), vget_lane_f32(vtotal,1), vget_lane_f32(vdataout,0), vget_lane_f32(vdataout,1));
                                        
                                        vst1_f32(outptr_ch_wh, vdataout);
                                        if(flag_relu)
                                            outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;

                                        outptr_ch_wh++;
                                        continue;
                                    }
                                } 
                            }
                            //right
                            inptr_ch0 += w_in - 7;
                            inptr_ch1 += w_in - 7;
                            inptr_ch2 += w_in - 7;
                            inptr_ch3 += w_in - 7;
                            inptr_ch4 += w_in - 7;
                            inptr_ch5 += w_in - 7;
                            inptr_ch6 += w_in - 7;
                            outptr_ch_wh +=w_out - 6;
                           
                            if (h == 0){
                                
                                float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                float32x4_t vdatain01 = vld1q_f32(inptr_ch0 + 4);
                                float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                //3 123456-012345
                                float32x4_t vdin0_1234 = vextq_f32(vdatain00, vdatain01, 1);
                                float32x4_t vdin1_1234 = vextq_f32(vdatain10, vdatain11, 1);
                                float32x4_t vdin2_1234 = vextq_f32(vdatain20, vdatain21, 1);
                                float32x4_t vdin3_1234 = vextq_f32(vdatain30, vdatain31, 1);
                                float32x4_t vdin0_5670 = vextq_f32(vdatain01, vzero, 1);
                                float32x4_t vdin1_5670 = vextq_f32(vdatain11, vzero, 1);
                                float32x4_t vdin2_5670 = vextq_f32(vdatain21, vzero, 1);
                                float32x4_t vdin3_5670 = vextq_f32(vdatain31, vzero, 1);

                                float32x4_t vdin0_2345 = vextq_f32(vdatain00, vdatain01, 2);
                                float32x4_t vdin1_2345 = vextq_f32(vdatain10, vdatain11, 2);
                                float32x4_t vdin2_2345 = vextq_f32(vdatain20, vdatain21, 2);
                                float32x4_t vdin3_2345 = vextq_f32(vdatain30, vdatain31, 2);
                                float32x4_t vdin0_6700 = vextq_f32(vdatain01, vzero, 2);
                                float32x4_t vdin1_6700 = vextq_f32(vdatain11, vzero, 2);
                                float32x4_t vdin2_6700 = vextq_f32(vdatain21, vzero, 2);
                                float32x4_t vdin3_6700 = vextq_f32(vdatain31, vzero, 2);

                                float32x4_t vdin0_3456 = vextq_f32(vdatain00, vdatain01, 3);
                                float32x4_t vdin1_3456 = vextq_f32(vdatain10, vdatain11, 3);
                                float32x4_t vdin2_3456 = vextq_f32(vdatain20, vdatain21, 3);
                                float32x4_t vdin3_3456 = vextq_f32(vdatain30, vdatain31, 3);

                                float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                                float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
                                float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
                                float32x4_t vweights60 = vld1q_f32(weight_ch_in + 42);
                                float32x4_t vsum = vmulq_f32(vdin0_1234, vweights30);
                                vsum = vmlaq_f32(vsum, vdin1_1234, vweights40);
                                vsum = vmlaq_f32(vsum, vdin2_1234, vweights50);
                                vsum = vmlaq_f32(vsum, vdin3_1234, vweights60);
                                
                                //out
                                float32x2_t vdataout = vld1_f32(outptr_ch_wh);


                                float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                                vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                                float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
                                vweights41 = vsetq_lane_f32(0.f, vweights41, 3);
                                float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
                                vweights51 = vsetq_lane_f32(0.f, vweights51, 3);
                                float32x4_t vweights61 = vld1q_f32(weight_ch_in + 46);
                                vweights61 = vsetq_lane_f32(0.f, vweights61, 3);
                                float32x4_t vw0_4500 = vsetq_lane_f32(0.f, vweights31, 2);
                                float32x4_t vw1_4500 = vsetq_lane_f32(0.f, vweights41, 2);
                                float32x4_t vw2_4500 = vsetq_lane_f32(0.f, vweights51, 2);
                                float32x4_t vw3_4500 = vsetq_lane_f32(0.f, vweights61, 2);
                                vsum = vmlaq_f32(vsum, vdin0_5670, vw0_4500);
                                vsum = vmlaq_f32(vsum, vdin1_5670, vw1_4500);
                                vsum = vmlaq_f32(vsum, vdin2_5670, vw2_4500);
                                vsum = vmlaq_f32(vsum, vdin3_5670, vw3_4500);
                                float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                vtotal = vpadd_f32(vtotal, vtotal);
                                vtotal = vset_lane_f32(0.f, vtotal, 1);

                                vdataout = vadd_f32(vdataout, vtotal);
                                vst1_f32(outptr_ch_wh, vdataout);
                                if(flag_relu)
                                    outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                outptr_ch_wh++;

                                //2 23456-01234
                                
                                vsum = vmulq_f32(vdin0_2345, vweights30);
                                vsum = vmlaq_f32(vsum, vdin1_2345, vweights40);
                                vsum = vmlaq_f32(vsum, vdin2_2345, vweights50);
                                vsum = vmlaq_f32(vsum, vdin3_2345, vweights60);
                                vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                vtotal = vpadd_f32(vtotal, vtotal);
                                
                                //out
                                vdataout = vld1_f32(outptr_ch_wh);
                                float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal,0));
                                vsum1 = vextq_f32(vsum1, vzero, 3);
                                
                                vsum1 = vmlaq_f32(vsum1, vdin0_6700, vweights31);
                                vsum1 = vmlaq_f32(vsum1, vdin1_6700, vweights41);
                                vsum1 = vmlaq_f32(vsum1, vdin2_6700, vweights51);
                                vsum1 = vmlaq_f32(vsum1, vdin3_6700, vweights61);
                                
                                vtotal = vget_low_f32(vsum1);
                                vtotal = vset_lane_f32(0.f, vtotal, 1);
                                vdataout = vadd_f32(vdataout, vtotal);
                                vst1_f32(outptr_ch_wh, vdataout);
                                if(flag_relu)
                                    outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                outptr_ch_wh++;

                                //1 3456 - 0123
                                vsum = vmulq_f32(vdin0_3456, vweights30);
                                vsum = vmlaq_f32(vsum, vdin1_3456, vweights40);
                                vsum = vmlaq_f32(vsum, vdin2_3456, vweights50);
                                vsum = vmlaq_f32(vsum, vdin3_3456, vweights60);
                                vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                vtotal = vpadd_f32(vtotal, vtotal);
                                vtotal = vset_lane_f32(0.f, vtotal, 1);
                                vdataout = vld1_f32(outptr_ch_wh);
                                vdataout = vadd_f32(vdataout, vtotal);
                                vst1_f32(outptr_ch_wh, vdataout);
                                if(flag_relu)
                                    outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                outptr_ch_wh++; 
                            } else if (h == 1){
                                //3
                                float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                float32x4_t vdatain01 = vld1q_f32(inptr_ch0 + 4);
                                float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                
                                float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
                                float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                                float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
                                float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
                                float32x4_t vweights60 = vld1q_f32(weight_ch_in + 42);
                                //1 123456-012345
                                float32x4_t vdin0_1234 = vextq_f32(vdatain00, vdatain01, 1);
                                float32x4_t vdin1_1234 = vextq_f32(vdatain10, vdatain11, 1);
                                float32x4_t vdin2_1234 = vextq_f32(vdatain20, vdatain21, 1);
                                float32x4_t vdin3_1234 = vextq_f32(vdatain30, vdatain31, 1);
                                float32x4_t vdin4_1234 = vextq_f32(vdatain40, vdatain41, 1);
                                float32x4_t vdin0_5670 = vextq_f32(vdatain01, vzero, 1);
                                float32x4_t vdin1_5670 = vextq_f32(vdatain11, vzero, 1);
                                float32x4_t vdin2_5670 = vextq_f32(vdatain21, vzero, 1);
                                float32x4_t vdin3_5670 = vextq_f32(vdatain31, vzero, 1);
                                float32x4_t vdin4_5670 = vextq_f32(vdatain41, vzero, 1);

                                float32x4_t vdin0_2345 = vextq_f32(vdatain00, vdatain01, 2);
                                float32x4_t vdin1_2345 = vextq_f32(vdatain10, vdatain11, 2);
                                float32x4_t vdin2_2345 = vextq_f32(vdatain20, vdatain21, 2);
                                float32x4_t vdin3_2345 = vextq_f32(vdatain30, vdatain31, 2);
                                float32x4_t vdin4_2345 = vextq_f32(vdatain40, vdatain41, 2);
                                float32x4_t vdin0_6700 = vextq_f32(vdatain01, vzero, 2);
                                float32x4_t vdin1_6700 = vextq_f32(vdatain11, vzero, 2);
                                float32x4_t vdin2_6700 = vextq_f32(vdatain21, vzero, 2);
                                float32x4_t vdin3_6700 = vextq_f32(vdatain31, vzero, 2);
                                float32x4_t vdin4_6700 = vextq_f32(vdatain41, vzero, 2);

                                float32x4_t vdin0_3456 = vextq_f32(vdatain00, vdatain01, 3);
                                float32x4_t vdin1_3456 = vextq_f32(vdatain10, vdatain11, 3);
                                float32x4_t vdin2_3456 = vextq_f32(vdatain20, vdatain21, 3);
                                float32x4_t vdin3_3456 = vextq_f32(vdatain30, vdatain31, 3);
                                float32x4_t vdin4_3456 = vextq_f32(vdatain40, vdatain41, 3);

                                float32x4_t vsum = vmulq_f32(vdin0_1234, vweights20);
                                vsum = vmlaq_f32(vsum, vdin1_1234, vweights30);
                                vsum = vmlaq_f32(vsum, vdin2_1234, vweights40);
                                vsum = vmlaq_f32(vsum, vdin3_1234, vweights50);
                                vsum = vmlaq_f32(vsum, vdin4_1234, vweights60);
                                
                                //out
                                float32x2_t vdataout = vld1_f32(outptr_ch_wh);

                                float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
                                vweights21 = vsetq_lane_f32(0.f, vweights21, 3);
                                float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                                vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                                float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
                                vweights41 = vsetq_lane_f32(0.f, vweights41, 3);
                                float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
                                vweights51 = vsetq_lane_f32(0.f, vweights51, 3);
                                float32x4_t vweights61 = vld1q_f32(weight_ch_in + 46);
                                vweights61 = vsetq_lane_f32(0.f, vweights61, 3);

                                float32x4_t vw0_4500 = vsetq_lane_f32(0.f, vweights21, 2);
                                float32x4_t vw1_4500 = vsetq_lane_f32(0.f, vweights31, 2);
                                float32x4_t vw2_4500 = vsetq_lane_f32(0.f, vweights41, 2);
                                float32x4_t vw3_4500 = vsetq_lane_f32(0.f, vweights51, 2);
                                float32x4_t vw4_4500 = vsetq_lane_f32(0.f, vweights61, 2);
                                
                                vsum = vmlaq_f32(vsum, vdin0_5670, vw0_4500);
                                vsum = vmlaq_f32(vsum, vdin1_5670, vw1_4500);
                                vsum = vmlaq_f32(vsum, vdin2_5670, vw2_4500);
                                vsum = vmlaq_f32(vsum, vdin3_5670, vw3_4500);
                                vsum = vmlaq_f32(vsum, vdin4_5670, vw4_4500);
                                float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                vtotal = vpadd_f32(vtotal, vtotal);
                                vtotal = vset_lane_f32(0.f, vtotal, 1);
                                vdataout = vadd_f32(vdataout, vtotal);
                                vst1_f32(outptr_ch_wh, vdataout);
                                if(flag_relu)
                                    outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                outptr_ch_wh++;

                                //2 23456-01234
                                
                                vsum = vmulq_f32(vdin0_2345, vweights20);
                                vsum = vmlaq_f32(vsum, vdin1_2345, vweights30);
                                vsum = vmlaq_f32(vsum, vdin2_2345, vweights40);
                                vsum = vmlaq_f32(vsum, vdin3_2345, vweights50);
                                vsum = vmlaq_f32(vsum, vdin4_2345, vweights60);
                                vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                vtotal = vpadd_f32(vtotal, vtotal);
                                
                                //out
                                vdataout = vld1_f32(outptr_ch_wh);
                                float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal,0));
                                vsum1 = vextq_f32(vsum1, vzero, 3);

                                
                                vsum1 = vmlaq_f32(vsum1, vdin0_6700, vweights21);
                                vsum1 = vmlaq_f32(vsum1, vdin1_6700, vweights31);
                                vsum1 = vmlaq_f32(vsum1, vdin2_6700, vweights41);
                                vsum1 = vmlaq_f32(vsum1, vdin3_6700, vweights51);
                                vsum1 = vmlaq_f32(vsum1, vdin4_6700, vweights61);
                                
                                vtotal = vget_low_f32(vsum1);
                                vtotal = vset_lane_f32(0.f, vtotal, 1);
                                vdataout = vadd_f32(vdataout, vtotal);
                                vst1_f32(outptr_ch_wh, vdataout);
                                if(flag_relu)
                                    outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                outptr_ch_wh++;
                                
                                //1 3456 - 0123
                                
                                vsum = vmulq_f32(vdin0_3456, vweights20);
                                vsum = vmlaq_f32(vsum, vdin1_3456, vweights30);
                                vsum = vmlaq_f32(vsum, vdin2_3456, vweights40);
                                vsum = vmlaq_f32(vsum, vdin3_3456, vweights50);
                                vsum = vmlaq_f32(vsum, vdin4_3456, vweights60);
                                vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                vtotal = vpadd_f32(vtotal, vtotal);
                                vtotal = vset_lane_f32(0.f, vtotal, 1);
                                vdataout = vld1_f32(outptr_ch_wh);
                                vdataout = vadd_f32(vdataout, vtotal);
                                vst1_f32(outptr_ch_wh, vdataout);
                                if(flag_relu)
                                    outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                outptr_ch_wh++; 
                            }else if (h == 2){
                                
                                float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                float32x4_t vdatain01 = vld1q_f32(inptr_ch0 + 4);
                                float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);
                                
                                //1 123456-012345
                                float32x4_t vdin0_1234 = vextq_f32(vdatain00, vdatain01, 1);
                                float32x4_t vdin1_1234 = vextq_f32(vdatain10, vdatain11, 1);
                                float32x4_t vdin2_1234 = vextq_f32(vdatain20, vdatain21, 1);
                                float32x4_t vdin3_1234 = vextq_f32(vdatain30, vdatain31, 1);
                                float32x4_t vdin4_1234 = vextq_f32(vdatain40, vdatain41, 1);
                                float32x4_t vdin5_1234 = vextq_f32(vdatain50, vdatain51, 1);
                                float32x4_t vdin0_5670 = vextq_f32(vdatain01, vzero, 1);
                                float32x4_t vdin1_5670 = vextq_f32(vdatain11, vzero, 1);
                                float32x4_t vdin2_5670 = vextq_f32(vdatain21, vzero, 1);
                                float32x4_t vdin3_5670 = vextq_f32(vdatain31, vzero, 1);
                                float32x4_t vdin4_5670 = vextq_f32(vdatain41, vzero, 1);
                                float32x4_t vdin5_5670 = vextq_f32(vdatain51, vzero, 1);

                                float32x4_t vdin0_2345 = vextq_f32(vdatain00, vdatain01, 2);
                                float32x4_t vdin1_2345 = vextq_f32(vdatain10, vdatain11, 2);
                                float32x4_t vdin2_2345 = vextq_f32(vdatain20, vdatain21, 2);
                                float32x4_t vdin3_2345 = vextq_f32(vdatain30, vdatain31, 2);
                                float32x4_t vdin4_2345 = vextq_f32(vdatain40, vdatain41, 2);
                                float32x4_t vdin5_2345 = vextq_f32(vdatain50, vdatain51, 2);
                                float32x4_t vdin0_6700 = vextq_f32(vdatain01, vzero, 2);
                                float32x4_t vdin1_6700 = vextq_f32(vdatain11, vzero, 2);
                                float32x4_t vdin2_6700 = vextq_f32(vdatain21, vzero, 2);
                                float32x4_t vdin3_6700 = vextq_f32(vdatain31, vzero, 2);
                                float32x4_t vdin4_6700 = vextq_f32(vdatain41, vzero, 2);
                                float32x4_t vdin5_6700 = vextq_f32(vdatain51, vzero, 2);
                                float32x4_t vdin0_3456 = vextq_f32(vdatain00, vdatain01, 3);
                                float32x4_t vdin1_3456 = vextq_f32(vdatain10, vdatain11, 3);
                                float32x4_t vdin2_3456 = vextq_f32(vdatain20, vdatain21, 3);
                                float32x4_t vdin3_3456 = vextq_f32(vdatain30, vdatain31, 3);
                                float32x4_t vdin4_3456 = vextq_f32(vdatain40, vdatain41, 3);
                                float32x4_t vdin5_3456 = vextq_f32(vdatain50, vdatain51, 3);

                                float32x4_t vweights10 = vld1q_f32(weight_ch_in + 7);
                                float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
                                float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                                float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
                                float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
                                float32x4_t vweights60 = vld1q_f32(weight_ch_in + 42);

                                float32x4_t vsum = vmulq_f32(vdin0_1234, vweights10);
                                vsum = vmlaq_f32(vsum, vdin1_1234, vweights20);
                                vsum = vmlaq_f32(vsum, vdin2_1234, vweights30);
                                vsum = vmlaq_f32(vsum, vdin3_1234, vweights40);
                                vsum = vmlaq_f32(vsum, vdin4_1234, vweights50);
                                vsum = vmlaq_f32(vsum, vdin5_1234, vweights60);
                                
                                //out
                                float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                
                                float32x4_t vweights11 = vld1q_f32(weight_ch_in + 11);
                                vweights11 = vsetq_lane_f32(0.f, vweights11, 3);
                                float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
                                vweights21 = vsetq_lane_f32(0.f, vweights21, 3);
                                float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                                vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                                float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
                                vweights41 = vsetq_lane_f32(0.f, vweights41, 3);
                                float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
                                vweights51 = vsetq_lane_f32(0.f, vweights51, 3);
                                float32x4_t vweights61 = vld1q_f32(weight_ch_in + 46);
                                vweights61 = vsetq_lane_f32(0.f, vweights61, 3);

                                float32x4_t vw0_4500 = vsetq_lane_f32(0.f, vweights11, 2);
                                float32x4_t vw1_4500 = vsetq_lane_f32(0.f, vweights21, 2);
                                float32x4_t vw2_4500 = vsetq_lane_f32(0.f, vweights31, 2);
                                float32x4_t vw3_4500 = vsetq_lane_f32(0.f, vweights41, 2);
                                float32x4_t vw4_4500 = vsetq_lane_f32(0.f, vweights51, 2);
                                float32x4_t vw5_4500 = vsetq_lane_f32(0.f, vweights61, 2);
                                
                                vsum = vmlaq_f32(vsum, vdin0_5670, vw0_4500);
                                vsum = vmlaq_f32(vsum, vdin1_5670, vw1_4500);
                                vsum = vmlaq_f32(vsum, vdin2_5670, vw2_4500);
                                vsum = vmlaq_f32(vsum, vdin3_5670, vw3_4500);
                                vsum = vmlaq_f32(vsum, vdin4_5670, vw4_4500);
                                vsum = vmlaq_f32(vsum, vdin5_5670, vw5_4500);
                                float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                vtotal = vpadd_f32(vtotal, vtotal);
                                vtotal = vset_lane_f32(0.f, vtotal, 1);
                                vdataout = vadd_f32(vdataout, vtotal);
                                vst1_f32(outptr_ch_wh, vdataout);
                                if(flag_relu)
                                    outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                outptr_ch_wh++;

                                //2 23456-01234
                                
                                vsum = vmulq_f32(vdin0_2345, vweights10);
                                vsum = vmlaq_f32(vsum, vdin1_2345, vweights20);
                                vsum = vmlaq_f32(vsum, vdin2_2345, vweights30);
                                vsum = vmlaq_f32(vsum, vdin3_2345, vweights40);
                                vsum = vmlaq_f32(vsum, vdin4_2345, vweights50);
                                vsum = vmlaq_f32(vsum, vdin5_2345, vweights60);
                                vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                vtotal = vpadd_f32(vtotal, vtotal);
                                
                                //out
                                vdataout = vld1_f32(outptr_ch_wh);
                                float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal,0));
                                vsum1 = vextq_f32(vsum1, vzero, 3);

                                
                                vsum1 = vmlaq_f32(vsum1, vdin0_6700, vweights11);
                                vsum1 = vmlaq_f32(vsum1, vdin1_6700, vweights21);
                                vsum1 = vmlaq_f32(vsum1, vdin2_6700, vweights31);
                                vsum1 = vmlaq_f32(vsum1, vdin3_6700, vweights41);
                                vsum1 = vmlaq_f32(vsum1, vdin4_6700, vweights51);
                                vsum1 = vmlaq_f32(vsum1, vdin5_6700, vweights61);
                                
                                vtotal = vget_low_f32(vsum1);
                                vtotal = vset_lane_f32(0.f, vtotal, 1);
                                vdataout = vadd_f32(vdataout, vtotal);
                                vst1_f32(outptr_ch_wh, vdataout);
                                if(flag_relu)
                                    outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                outptr_ch_wh++;
                                
                                //1 3456 - 0123
                                
                                vsum = vmulq_f32(vdin0_3456, vweights10);
                                vsum = vmlaq_f32(vsum, vdin1_3456, vweights20);
                                vsum = vmlaq_f32(vsum, vdin2_3456, vweights30);
                                vsum = vmlaq_f32(vsum, vdin3_3456, vweights40);
                                vsum = vmlaq_f32(vsum, vdin4_3456, vweights50);
                                vsum = vmlaq_f32(vsum, vdin5_3456, vweights60);
                                vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                vtotal = vpadd_f32(vtotal, vtotal);
                                vtotal = vset_lane_f32(0.f, vtotal, 1);
                                vdataout = vld1_f32(outptr_ch_wh);
                                vdataout = vadd_f32(vdataout, vtotal);
                                vst1_f32(outptr_ch_wh, vdataout);
                                if(flag_relu)
                                    outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                outptr_ch_wh++; 
                            } else {//mid
                                
                                float32x4_t vdatain00 = vld1q_f32(inptr_ch0);
                                float32x4_t vdatain01 = vld1q_f32(inptr_ch0 + 4);
                                float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);
                                float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                                float32x4_t vdatain61 = vld1q_f32(inptr_ch6 + 4);
                                //1 123456-012345
                                float32x4_t vdin0_1234 = vextq_f32(vdatain00, vdatain01, 1);
                                float32x4_t vdin1_1234 = vextq_f32(vdatain10, vdatain11, 1);
                                float32x4_t vdin2_1234 = vextq_f32(vdatain20, vdatain21, 1);
                                float32x4_t vdin3_1234 = vextq_f32(vdatain30, vdatain31, 1);
                                float32x4_t vdin4_1234 = vextq_f32(vdatain40, vdatain41, 1);
                                float32x4_t vdin5_1234 = vextq_f32(vdatain50, vdatain51, 1);
                                float32x4_t vdin6_1234 = vextq_f32(vdatain60, vdatain61, 1);
                                float32x4_t vweights00 = vld1q_f32(weight_ch_in);
                                float32x4_t vweights10 = vld1q_f32(weight_ch_in + 7);
                                float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
                                float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                                float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
                                float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
                                float32x4_t vweights60 = vld1q_f32(weight_ch_in + 42);

                                float32x4_t vsum = vmulq_f32(vdin0_1234, vweights00);
                                vsum = vmlaq_f32(vsum, vdin1_1234, vweights10);
                                vsum = vmlaq_f32(vsum, vdin2_1234, vweights20);
                                vsum = vmlaq_f32(vsum, vdin3_1234, vweights30);
                                vsum = vmlaq_f32(vsum, vdin4_1234, vweights40);
                                vsum = vmlaq_f32(vsum, vdin5_1234, vweights50);
                                vsum = vmlaq_f32(vsum, vdin6_1234, vweights60);
                                
                                //out
                                float32x2_t vdataout = vld1_f32(outptr_ch_wh);

                                float32x4_t vweights01 = vld1q_f32(weight_ch_in + 4);
                                vweights01 = vsetq_lane_f32(0.f, vweights01, 3);
                                float32x4_t vweights11 = vld1q_f32(weight_ch_in + 11);
                                vweights11 = vsetq_lane_f32(0.f, vweights11, 3);
                                float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
                                vweights21 = vsetq_lane_f32(0.f, vweights21, 3);
                                float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                                vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                                float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
                                vweights41 = vsetq_lane_f32(0.f, vweights41, 3);
                                float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
                                vweights51 = vsetq_lane_f32(0.f, vweights51, 3);
                                float32x4_t vweights61 = vld1q_f32(weight_ch_in + 46);
                                vweights61 = vsetq_lane_f32(0.f, vweights61, 3);

                                float32x4_t vw0_4500 = vsetq_lane_f32(0.f, vweights01, 2);
                                float32x4_t vw1_4500 = vsetq_lane_f32(0.f, vweights11, 2);
                                float32x4_t vw2_4500 = vsetq_lane_f32(0.f, vweights21, 2);
                                float32x4_t vw3_4500 = vsetq_lane_f32(0.f, vweights31, 2);
                                float32x4_t vw4_4500 = vsetq_lane_f32(0.f, vweights41, 2);
                                float32x4_t vw5_4500 = vsetq_lane_f32(0.f, vweights51, 2);
                                float32x4_t vw6_4500 = vsetq_lane_f32(0.f, vweights61, 2);
                                float32x4_t vdin0_5670 = vextq_f32(vdatain01, vzero, 1);
                                float32x4_t vdin1_5670 = vextq_f32(vdatain11, vzero, 1);
                                float32x4_t vdin2_5670 = vextq_f32(vdatain21, vzero, 1);
                                float32x4_t vdin3_5670 = vextq_f32(vdatain31, vzero, 1);
                                float32x4_t vdin4_5670 = vextq_f32(vdatain41, vzero, 1);
                                float32x4_t vdin5_5670 = vextq_f32(vdatain51, vzero, 1);
                                float32x4_t vdin6_5670 = vextq_f32(vdatain61, vzero, 1);
                                vsum = vmlaq_f32(vsum, vdin0_5670, vw0_4500);
                                vsum = vmlaq_f32(vsum, vdin1_5670, vw1_4500);
                                vsum = vmlaq_f32(vsum, vdin2_5670, vw2_4500);
                                vsum = vmlaq_f32(vsum, vdin3_5670, vw3_4500);
                                vsum = vmlaq_f32(vsum, vdin4_5670, vw4_4500);
                                vsum = vmlaq_f32(vsum, vdin5_5670, vw5_4500);
                                vsum = vmlaq_f32(vsum, vdin6_5670, vw6_4500);
                                float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                vtotal = vpadd_f32(vtotal, vtotal);
                                vtotal = vset_lane_f32(0.f, vtotal, 1);
                                vdataout = vadd_f32(vdataout, vtotal);
                                vst1_f32(outptr_ch_wh, vdataout);
                                if(flag_relu)
                                    outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                outptr_ch_wh++;

                                //2 23456-01234
                                float32x4_t vdin0_2345 = vextq_f32(vdatain00, vdatain01, 2);
                                float32x4_t vdin1_2345 = vextq_f32(vdatain10, vdatain11, 2);
                                float32x4_t vdin2_2345 = vextq_f32(vdatain20, vdatain21, 2);
                                float32x4_t vdin3_2345 = vextq_f32(vdatain30, vdatain31, 2);
                                float32x4_t vdin4_2345 = vextq_f32(vdatain40, vdatain41, 2);
                                float32x4_t vdin5_2345 = vextq_f32(vdatain50, vdatain51, 2);
                                float32x4_t vdin6_2345 = vextq_f32(vdatain60, vdatain61, 2);
                                vsum = vmulq_f32(vdin0_2345, vweights00);
                                vsum = vmlaq_f32(vsum, vdin1_2345, vweights10);
                                vsum = vmlaq_f32(vsum, vdin2_2345, vweights20);
                                vsum = vmlaq_f32(vsum, vdin3_2345, vweights30);
                                vsum = vmlaq_f32(vsum, vdin4_2345, vweights40);
                                vsum = vmlaq_f32(vsum, vdin5_2345, vweights50);
                                vsum = vmlaq_f32(vsum, vdin6_2345, vweights60);
                                vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                vtotal = vpadd_f32(vtotal, vtotal);
                                
                                //out
                                vdataout = vld1_f32(outptr_ch_wh);
                                float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal,0));
                                vsum1 = vextq_f32(vsum1, vzero, 3);

                                float32x4_t vdin0_6700 = vextq_f32(vdatain01, vzero, 2);
                                float32x4_t vdin1_6700 = vextq_f32(vdatain11, vzero, 2);
                                float32x4_t vdin2_6700 = vextq_f32(vdatain21, vzero, 2);
                                float32x4_t vdin3_6700 = vextq_f32(vdatain31, vzero, 2);
                                float32x4_t vdin4_6700 = vextq_f32(vdatain41, vzero, 2);
                                float32x4_t vdin5_6700 = vextq_f32(vdatain51, vzero, 2);
                                float32x4_t vdin6_6700 = vextq_f32(vdatain61, vzero, 2);
                                vsum1 = vmlaq_f32(vsum1, vdin0_6700, vweights01);
                                vsum1 = vmlaq_f32(vsum1, vdin1_6700, vweights11);
                                vsum1 = vmlaq_f32(vsum1, vdin2_6700, vweights21);
                                vsum1 = vmlaq_f32(vsum1, vdin3_6700, vweights31);
                                vsum1 = vmlaq_f32(vsum1, vdin4_6700, vweights41);
                                vsum1 = vmlaq_f32(vsum1, vdin5_6700, vweights51);
                                vsum1 = vmlaq_f32(vsum1, vdin6_6700, vweights61);
                                
                                vtotal = vget_low_f32(vsum1);
                                vtotal = vset_lane_f32(0.f, vtotal, 1);
                                vdataout = vadd_f32(vdataout, vtotal);
                                vst1_f32(outptr_ch_wh, vdataout);
                                if(flag_relu)
                                    outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                outptr_ch_wh++;
                                
                                //1 3456 - 0123
                                float32x4_t vdin0_3456 = vextq_f32(vdatain00, vdatain01, 3);
                                float32x4_t vdin1_3456 = vextq_f32(vdatain10, vdatain11, 3);
                                float32x4_t vdin2_3456 = vextq_f32(vdatain20, vdatain21, 3);
                                float32x4_t vdin3_3456 = vextq_f32(vdatain30, vdatain31, 3);
                                float32x4_t vdin4_3456 = vextq_f32(vdatain40, vdatain41, 3);
                                float32x4_t vdin5_3456 = vextq_f32(vdatain50, vdatain51, 3);
                                float32x4_t vdin6_3456 = vextq_f32(vdatain60, vdatain61, 3);
                                vsum = vmulq_f32(vdin0_3456, vweights00);
                                vsum = vmlaq_f32(vsum, vdin1_3456, vweights10);
                                vsum = vmlaq_f32(vsum, vdin2_3456, vweights20);
                                vsum = vmlaq_f32(vsum, vdin3_3456, vweights30);
                                vsum = vmlaq_f32(vsum, vdin4_3456, vweights40);
                                vsum = vmlaq_f32(vsum, vdin5_3456, vweights50);
                                vsum = vmlaq_f32(vsum, vdin6_3456, vweights60);
                                vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                vtotal = vpadd_f32(vtotal, vtotal);
                                vtotal = vset_lane_f32(0.f, vtotal, 1);
                                vdataout = vld1_f32(outptr_ch_wh);
                                vdataout = vadd_f32(vdataout, vtotal);
                                vst1_f32(outptr_ch_wh, vdataout);
                                if(flag_relu)
                                    outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                outptr_ch_wh++; 
                            }
                        } else if (h == h_out - kernel_h_even){//3 123456
                            int w = 0;
                            for (w = 0; w < 3; w++){
                                float32x4_t vweights00 = vld1q_f32(weight_ch_in);
                                float32x4_t vweights01 = vld1q_f32(weight_ch_in + 4);
                                vweights01 = vsetq_lane_f32(0.f, vweights01, 3);
                                float32x4_t vweights10 = vld1q_f32(weight_ch_in + 7);
                                float32x4_t vweights11 = vld1q_f32(weight_ch_in + 11);
                                vweights11 = vsetq_lane_f32(0.f, vweights11, 3);
                                float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
                                float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
                                vweights21 = vsetq_lane_f32(0.f, vweights21, 3);
                                float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                                float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                                vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                                float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
                                float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
                                vweights41 = vsetq_lane_f32(0.f, vweights41, 3);
                                float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
                                float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
                                vweights51 = vsetq_lane_f32(0.f, vweights51, 3);
                                 
                                if (w == 0){//3456
                                    float32x4_t vw0_3456 = vextq_f32(vweights00, vweights01, 3);
                                    float32x4_t vw1_3456 = vextq_f32(vweights10, vweights11, 3);
                                    float32x4_t vw2_3456 = vextq_f32(vweights20, vweights21, 3);
                                    float32x4_t vw3_3456 = vextq_f32(vweights30, vweights31, 3);
                                    float32x4_t vw4_3456 = vextq_f32(vweights40, vweights41, 3);
                                    float32x4_t vw5_3456 = vextq_f32(vweights50, vweights51, 3);
                                    //load data
                                    float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                    float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                    float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                    float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                    float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                    float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                                    
                                    float32x4_t vsum = vmulq_f32(vdatain10, vw0_3456);
                                    vsum = vmlaq_f32(vsum, vdatain20, vw1_3456);
                                    vsum = vmlaq_f32(vsum, vdatain30, vw2_3456);
                                    vsum = vmlaq_f32(vsum, vdatain40, vw3_3456);
                                    vsum = vmlaq_f32(vsum, vdatain50, vw4_3456);
                                    vsum = vmlaq_f32(vsum, vdatain60, vw5_3456);
                                    float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                    vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                    vtotal = vset_lane_f32(0.f, vtotal, 1);
                                    float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                    vdataout = vadd_f32(vtotal, vdataout);
                                    vst1_f32(outptr_ch_wh, vdataout);
                                    if(flag_relu)
                                        outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                    outptr_ch_wh++;
                                    continue;
                                }else if (w == 1){//23456
                                    float32x4_t vw0_2345 = vextq_f32(vweights00, vweights01, 2);
                                    float32x4_t vw1_2345 = vextq_f32(vweights10, vweights11, 2);
                                    float32x4_t vw2_2345 = vextq_f32(vweights20, vweights21, 2);
                                    float32x4_t vw3_2345 = vextq_f32(vweights30, vweights31, 2);
                                    float32x4_t vw4_2345 = vextq_f32(vweights40, vweights41, 2);
                                    float32x4_t vw5_2345 = vextq_f32(vweights50, vweights51, 2);
                                    //load data
                                    float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                    float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                    float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                    float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                    float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                    float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                                    float32x4_t vsum = vmulq_f32(vdatain10, vw0_2345);
                                    vsum = vmlaq_f32(vsum, vdatain20, vw1_2345);
                                    vsum = vmlaq_f32(vsum, vdatain30, vw2_2345);
                                    vsum = vmlaq_f32(vsum, vdatain40, vw3_2345);
                                    vsum = vmlaq_f32(vsum, vdatain50, vw4_2345);
                                    vsum = vmlaq_f32(vsum, vdatain60, vw5_2345);
                                    float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                    vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                    float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                    vsum1 = vextq_f32(vsum1, vzero, 3);
                                    float32x4_t vw0_5670 = vextq_f32(vweights01, vzero, 2);
                                    float32x4_t vw1_5670 = vextq_f32(vweights11, vzero, 2);
                                    float32x4_t vw2_5670 = vextq_f32(vweights21, vzero, 2);
                                    float32x4_t vw3_5670 = vextq_f32(vweights31, vzero, 2);
                                    float32x4_t vw4_5670 = vextq_f32(vweights41, vzero, 2);
                                    float32x4_t vw5_5670 = vextq_f32(vweights51, vzero, 2);

                                    float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                    float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                    float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                    float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                    float32x4_t vdatain61 = vld1q_f32(inptr_ch6 + 4);
                                    float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);

                                    vsum1 = vmlaq_f32(vsum1, vdatain11, vw0_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain21, vw1_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain31, vw2_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain41, vw3_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain51, vw4_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain61, vw5_5670);//6
                                    vtotal = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));//x, 0
                                    float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                    vdataout = vadd_f32(vtotal, vdataout);
                                    vst1_f32(outptr_ch_wh, vdataout);
                                    if(flag_relu)
                                       outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                    outptr_ch_wh++;
                                    continue;
                                }else if (w == 2){//123456
                                    float32x4_t vw0_1234 = vextq_f32(vweights00, vweights01, 1);
                                    float32x4_t vw1_1234 = vextq_f32(vweights10, vweights11, 1);
                                    float32x4_t vw2_1234 = vextq_f32(vweights20, vweights21, 1);
                                    float32x4_t vw3_1234 = vextq_f32(vweights30, vweights31, 1);
                                    float32x4_t vw4_1234 = vextq_f32(vweights40, vweights41, 1);
                                    float32x4_t vw5_1234 = vextq_f32(vweights50, vweights51, 1);
                                    //load data
                                    float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                                    float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                    float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                    float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                    float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                    float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                                    float32x4_t vsum = vmulq_f32(vdatain10, vw0_1234);
                                    vsum = vmlaq_f32(vsum, vdatain20, vw1_1234);
                                    vsum = vmlaq_f32(vsum, vdatain30, vw2_1234);
                                    vsum = vmlaq_f32(vsum, vdatain40, vw3_1234);
                                    vsum = vmlaq_f32(vsum, vdatain50, vw4_1234);
                                    vsum = vmlaq_f32(vsum, vdatain60, vw5_1234);
                                    float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                    vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                    float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                    vsum1 = vextq_f32(vsum1, vzero, 3);
                                    float32x4_t vw0_5670 = vextq_f32(vweights01, vzero, 1);
                                    float32x4_t vw1_5670 = vextq_f32(vweights11, vzero, 1);
                                    float32x4_t vw2_5670 = vextq_f32(vweights21, vzero, 1);
                                    float32x4_t vw3_5670 = vextq_f32(vweights31, vzero, 1);
                                    float32x4_t vw4_5670 = vextq_f32(vweights41, vzero, 1);
                                    float32x4_t vw5_5670 = vextq_f32(vweights51, vzero, 1);
                                    float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                                    float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                    float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                    float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                    float32x4_t vdatain61 = vld1q_f32(inptr_ch6 + 4);
                                    float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);

                                    vsum1 = vmlaq_f32(vsum1, vdatain11, vw0_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain21, vw1_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain31, vw2_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain41, vw3_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain51, vw4_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain61, vw5_5670);//6
                                    vtotal = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));//x, 0
                                    float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                    vdataout = vadd_f32(vdataout, vtotal);
                                    vst1_f32(outptr_ch_wh, vdataout);
                                    if(flag_relu)
                                        outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                    outptr_ch_wh++;
                                    continue;
                                }
                            } 
                            //right
                           //right
                            inptr_ch1 += w_in - 7;
                            inptr_ch2 += w_in - 7;
                            inptr_ch3 += w_in - 7;
                            inptr_ch4 += w_in - 7;
                            inptr_ch5 += w_in - 7;
                            inptr_ch6 += w_in - 7;
                            outptr_ch_wh +=w_out - 6;
                           // printf("inptr_ch1: %x, inptr_ch2: %x, inptr_ch3: %x, inptr_ch4: %x, inptr_ch5: %x, inptr_ch6: %x \n", \
                                     inptr_ch1, inptr_ch2, inptr_ch3, inptr_ch4, inptr_ch5, inptr_ch6);
                            float32x4_t vdatain10 = vld1q_f32(inptr_ch1);
                            float32x4_t vdatain11 = vld1q_f32(inptr_ch1 + 4);
                            float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                            float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                            float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                            float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                            float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                            float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                            float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                            float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);
                            float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                            float32x4_t vdatain61 = vld1q_f32(inptr_ch6 + 4);

                            //1 123456-012345
                            float32x4_t vdin0_1234 = vextq_f32(vdatain10, vdatain11, 1);
                            float32x4_t vdin1_1234 = vextq_f32(vdatain20, vdatain21, 1);
                            float32x4_t vdin2_1234 = vextq_f32(vdatain30, vdatain31, 1);
                            float32x4_t vdin3_1234 = vextq_f32(vdatain40, vdatain41, 1);
                            float32x4_t vdin4_1234 = vextq_f32(vdatain50, vdatain51, 1);
                            float32x4_t vdin5_1234 = vextq_f32(vdatain60, vdatain61, 1);
                            float32x4_t vweights00 = vld1q_f32(weight_ch_in);
                            float32x4_t vweights10 = vld1q_f32(weight_ch_in + 7);
                            float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
                            float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                            float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
                            float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);

                            float32x4_t vsum = vmulq_f32(vdin0_1234, vweights00);
                            vsum = vmlaq_f32(vsum, vdin1_1234, vweights10);
                            vsum = vmlaq_f32(vsum, vdin2_1234, vweights20);
                            vsum = vmlaq_f32(vsum, vdin3_1234, vweights30);
                            vsum = vmlaq_f32(vsum, vdin4_1234, vweights40);
                            vsum = vmlaq_f32(vsum, vdin5_1234, vweights50);
                                
                            //out
                            float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                            
                            float32x4_t vweights01 = vld1q_f32(weight_ch_in + 4);
                            vweights01 = vsetq_lane_f32(0.f, vweights01, 3);
                            float32x4_t vweights11 = vld1q_f32(weight_ch_in + 11);
                            vweights11 = vsetq_lane_f32(0.f, vweights11, 3);
                            float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
                            vweights21 = vsetq_lane_f32(0.f, vweights21, 3);
                            float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                            vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                            float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
                            vweights41 = vsetq_lane_f32(0.f, vweights41, 3);
                            float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
                            vweights51 = vsetq_lane_f32(0.f, vweights51, 3);
                            float32x4_t vw0_4500 = vsetq_lane_f32(0.f, vweights01, 2);
                            float32x4_t vw1_4500 = vsetq_lane_f32(0.f, vweights11, 2);
                            float32x4_t vw2_4500 = vsetq_lane_f32(0.f, vweights21, 2);
                            float32x4_t vw3_4500 = vsetq_lane_f32(0.f, vweights31, 2);
                            float32x4_t vw4_4500 = vsetq_lane_f32(0.f, vweights41, 2);
                            float32x4_t vw5_4500 = vsetq_lane_f32(0.f, vweights51, 2);
                            float32x4_t vdin0_5670 = vextq_f32(vdatain11, vzero, 1);
                            float32x4_t vdin1_5670 = vextq_f32(vdatain21, vzero, 1);
                            float32x4_t vdin2_5670 = vextq_f32(vdatain31, vzero, 1);
                            float32x4_t vdin3_5670 = vextq_f32(vdatain41, vzero, 1);
                            float32x4_t vdin4_5670 = vextq_f32(vdatain51, vzero, 1);
                            float32x4_t vdin5_5670 = vextq_f32(vdatain61, vzero, 1);
                            vsum = vmlaq_f32(vsum, vdin0_5670, vw0_4500);
                            vsum = vmlaq_f32(vsum, vdin1_5670, vw1_4500);
                            vsum = vmlaq_f32(vsum, vdin2_5670, vw2_4500);
                            vsum = vmlaq_f32(vsum, vdin3_5670, vw3_4500);
                            vsum = vmlaq_f32(vsum, vdin4_5670, vw4_4500);
                            vsum = vmlaq_f32(vsum, vdin5_5670, vw5_4500);
                            float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                            vtotal = vpadd_f32(vtotal, vtotal);
                            vtotal = vset_lane_f32(0.f, vtotal, 1);
                            vdataout = vadd_f32(vdataout, vtotal);
                            vst1_f32(outptr_ch_wh, vdataout);
                            if(flag_relu)
                                outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                            outptr_ch_wh++;

                            //2 23456-01234;
                            float32x4_t vdin0_2345 = vextq_f32(vdatain10, vdatain11, 2);
                            float32x4_t vdin1_2345 = vextq_f32(vdatain20, vdatain21, 2);
                            float32x4_t vdin2_2345 = vextq_f32(vdatain30, vdatain31, 2);
                            float32x4_t vdin3_2345 = vextq_f32(vdatain40, vdatain41, 2);
                            float32x4_t vdin4_2345 = vextq_f32(vdatain50, vdatain51, 2);
                            float32x4_t vdin5_2345 = vextq_f32(vdatain60, vdatain61, 2);
                            vsum = vmulq_f32(vdin0_2345, vweights00);
                            vsum = vmlaq_f32(vsum, vdin1_2345, vweights10);
                            vsum = vmlaq_f32(vsum, vdin2_2345, vweights20);
                            vsum = vmlaq_f32(vsum, vdin3_2345, vweights30);
                            vsum = vmlaq_f32(vsum, vdin4_2345, vweights40);
                            vsum = vmlaq_f32(vsum, vdin5_2345, vweights50);
                            vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                            vtotal = vpadd_f32(vtotal, vtotal);
                                
                            //out
                            vdataout = vld1_f32(outptr_ch_wh);
                            float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal,0));
                            vsum1 = vextq_f32(vsum1, vzero, 3);

                            float32x4_t vdin0_6700 = vextq_f32(vdatain11, vzero, 2);
                            float32x4_t vdin1_6700 = vextq_f32(vdatain21, vzero, 2);
                            float32x4_t vdin2_6700 = vextq_f32(vdatain31, vzero, 2);
                            float32x4_t vdin3_6700 = vextq_f32(vdatain41, vzero, 2);
                            float32x4_t vdin4_6700 = vextq_f32(vdatain51, vzero, 2);
                            float32x4_t vdin5_6700 = vextq_f32(vdatain61, vzero, 2);
                            vsum1 = vmlaq_f32(vsum1, vdin0_6700, vweights01);
                            vsum1 = vmlaq_f32(vsum1, vdin1_6700, vweights11);
                            vsum1 = vmlaq_f32(vsum1, vdin2_6700, vweights21);
                            vsum1 = vmlaq_f32(vsum1, vdin3_6700, vweights31);
                            vsum1 = vmlaq_f32(vsum1, vdin4_6700, vweights41);
                            vsum1 = vmlaq_f32(vsum1, vdin5_6700, vweights51);
                                
                            vtotal = vget_low_f32(vsum1);
                            vtotal = vset_lane_f32(0.f, vtotal, 1);
                            vdataout = vadd_f32(vdataout, vtotal);
                            vst1_f32(outptr_ch_wh, vdataout);
                            if(flag_relu)
                                outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                            outptr_ch_wh++;
                                
                            //1 3456 - 0123
                            float32x4_t vdin0_3456 = vextq_f32(vdatain10, vdatain11, 3);
                            float32x4_t vdin1_3456 = vextq_f32(vdatain20, vdatain21, 3);
                            float32x4_t vdin2_3456 = vextq_f32(vdatain30, vdatain31, 3);
                            float32x4_t vdin3_3456 = vextq_f32(vdatain40, vdatain41, 3);
                            float32x4_t vdin4_3456 = vextq_f32(vdatain50, vdatain51, 3);
                            float32x4_t vdin5_3456 = vextq_f32(vdatain60, vdatain61, 3);
                            vsum = vmulq_f32(vdin0_3456, vweights00);
                            vsum = vmlaq_f32(vsum, vdin1_3456, vweights10);
                            vsum = vmlaq_f32(vsum, vdin2_3456, vweights20);
                            vsum = vmlaq_f32(vsum, vdin3_3456, vweights30);
                            vsum = vmlaq_f32(vsum, vdin4_3456, vweights40);
                            vsum = vmlaq_f32(vsum, vdin5_3456, vweights50);
                            vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                            vtotal = vpadd_f32(vtotal, vtotal);
                            vtotal = vset_lane_f32(0.f, vtotal, 1);
                            vdataout = vld1_f32(outptr_ch_wh);
                            vdataout = vadd_f32(vdataout, vtotal);
                            vst1_f32(outptr_ch_wh, vdataout);
                            if(flag_relu)
                                outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                            outptr_ch_wh++; 
                                    
                        }else if (h == h_out - kernel_h_even + 1){//1
                            int w = 0;
                            for (w = 0; w < w_out - kernel_w_even; w++){
                                float32x4_t vweights00 = vld1q_f32(weight_ch_in);
                                float32x4_t vweights01 = vld1q_f32(weight_ch_in + 4);
                                vweights01 = vsetq_lane_f32(0.f, vweights01, 3);
                                float32x4_t vweights10 = vld1q_f32(weight_ch_in + 7);
                                float32x4_t vweights11 = vld1q_f32(weight_ch_in + 11);
                                vweights11 = vsetq_lane_f32(0.f, vweights11, 3);
                                float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
                                float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
                                vweights21 = vsetq_lane_f32(0.f, vweights21, 3);
                                float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                                float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                                vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                                float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);
                                float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
                                vweights41 = vsetq_lane_f32(0.f, vweights41, 3);
                                float32x4_t vweights50 = vld1q_f32(weight_ch_in + 35);
                                float32x4_t vweights51 = vld1q_f32(weight_ch_in + 39);
                                vweights51 = vsetq_lane_f32(0.f, vweights51, 3);
                                
                                if (w == 0){//3456
                                    float32x4_t vw0_3456 = vextq_f32(vweights00, vweights01, 3);
                                    float32x4_t vw1_3456 = vextq_f32(vweights10, vweights11, 3);
                                    float32x4_t vw2_3456 = vextq_f32(vweights20, vweights21, 3);
                                    float32x4_t vw3_3456 = vextq_f32(vweights30, vweights31, 3);
                                    float32x4_t vw4_3456 = vextq_f32(vweights40, vweights41, 3);
                                    float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                    float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                    float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                    float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                    float32x4_t vdatain60 = vld1q_f32(inptr_ch6);

                                    float32x4_t vsum = vmulq_f32(vdatain20, vw0_3456);
                                    vsum = vmlaq_f32(vsum, vdatain30, vw1_3456);
                                    vsum = vmlaq_f32(vsum, vdatain40, vw2_3456);
                                    vsum = vmlaq_f32(vsum, vdatain50, vw3_3456);
                                    vsum = vmlaq_f32(vsum, vdatain60, vw4_3456);
                                    float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                    vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                    vtotal = vset_lane_f32(0.f, vtotal, 1);
                                    float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                    vdataout = vadd_f32(vtotal, vdataout);
                                    vst1_f32(outptr_ch_wh, vdataout);
                                    if(flag_relu)
                                       outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                    outptr_ch_wh++;
                                    continue;
                                }else if (w == 1){//23456
                                    float32x4_t vw0_2345 = vextq_f32(vweights00, vweights01, 2);
                                    float32x4_t vw1_2345 = vextq_f32(vweights10, vweights11, 2);
                                    float32x4_t vw2_2345 = vextq_f32(vweights20, vweights21, 2);
                                    float32x4_t vw3_2345 = vextq_f32(vweights30, vweights31, 2);
                                    float32x4_t vw4_2345 = vextq_f32(vweights40, vweights41, 2);
                                    float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                    float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                    float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                    float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                    float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                                    float32x4_t vsum = vmulq_f32(vdatain20, vw0_2345);
                                    vsum = vmlaq_f32(vsum, vdatain30, vw1_2345);
                                    vsum = vmlaq_f32(vsum, vdatain40, vw2_2345);
                                    vsum = vmlaq_f32(vsum, vdatain50, vw3_2345);
                                    vsum = vmlaq_f32(vsum, vdatain60, vw4_2345);
                                    float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                    vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                    float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                    vsum1 = vextq_f32(vsum1, vzero, 3);

                                    float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                    float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                    float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                    float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);
                                    float32x4_t vdatain61 = vld1q_f32(inptr_ch6 + 4);
                                    float32x4_t vw0_5670 = vextq_f32(vweights01, vzero, 2);
                                    float32x4_t vw1_5670 = vextq_f32(vweights11, vzero, 2);
                                    float32x4_t vw2_5670 = vextq_f32(vweights21, vzero, 2);
                                    float32x4_t vw3_5670 = vextq_f32(vweights31, vzero, 2);
                                    float32x4_t vw4_5670 = vextq_f32(vweights41, vzero, 2);
                                    vsum1 = vmlaq_f32(vsum1, vdatain21, vw0_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain31, vw1_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain41, vw2_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain51, vw3_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain61, vw4_5670);//6
                                    vtotal = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));//x, 0

                                    float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                    vdataout = vadd_f32(vtotal, vdataout);
                                    vtotal = vget_low_f32(vsum1);
                                    vtotal = vset_lane_f32(0.f, vtotal, 1);
                                    vst1_f32(outptr_ch_wh, vdataout);
                                    if(flag_relu)
                                       outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                    outptr_ch_wh++;
                                    continue;
                                }else if (w == 2){//123456
                                    float32x4_t vw0_1234 = vextq_f32(vweights00, vweights01, 1);
                                    float32x4_t vw1_1234 = vextq_f32(vweights10, vweights11, 1);
                                    float32x4_t vw2_1234 = vextq_f32(vweights20, vweights21, 1);
                                    float32x4_t vw3_1234 = vextq_f32(vweights30, vweights31, 1);
                                    float32x4_t vw4_1234 = vextq_f32(vweights40, vweights41, 1);
                                    float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                                    float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                    float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                    float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                    float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                                    float32x4_t vsum = vmulq_f32(vdatain20, vw0_1234);
                                    vsum = vmlaq_f32(vsum, vdatain30, vw1_1234);
                                    vsum = vmlaq_f32(vsum, vdatain40, vw2_1234);
                                    vsum = vmlaq_f32(vsum, vdatain50, vw3_1234);
                                    vsum = vmlaq_f32(vsum, vdatain60, vw4_1234);
                                    float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                    vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                    float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                    vsum1 = vextq_f32(vsum1, vzero, 3);

                                    float32x4_t vw0_5670 = vextq_f32(vweights01, vzero, 1);
                                    float32x4_t vw1_5670 = vextq_f32(vweights11, vzero, 1);
                                    float32x4_t vw2_5670 = vextq_f32(vweights21, vzero, 1);
                                    float32x4_t vw3_5670 = vextq_f32(vweights31, vzero, 1);
                                    float32x4_t vw4_5670 = vextq_f32(vweights41, vzero, 1);
                                    float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                                    float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                    float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                    float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);
                                    float32x4_t vdatain61 = vld1q_f32(inptr_ch6 + 4);

                                    vsum1 = vmlaq_f32(vsum1, vdatain21, vw0_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain31, vw1_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain41, vw2_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain51, vw3_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain61, vw4_5670);//6
                                    vtotal = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));//x, 0
                                    float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                    vdataout = vadd_f32(vdataout, vtotal);
                                    vst1_f32(outptr_ch_wh, vdataout);
                                    if(flag_relu)
                                       outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                    outptr_ch_wh++;
                                    continue;
                                }
                            } 
                            //right
                            inptr_ch2 += w_in - 7;
                            inptr_ch3 += w_in - 7;
                            inptr_ch4 += w_in - 7;
                            inptr_ch5 += w_in - 7;
                            inptr_ch6 += w_in - 7;
                            outptr_ch_wh +=w_out - 6;
                            float32x4_t vdatain20 = vld1q_f32(inptr_ch2);
                            float32x4_t vdatain21 = vld1q_f32(inptr_ch2 + 4);
                            float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                            float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                            float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                            float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                            float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                            float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);
                            float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                            float32x4_t vdatain61 = vld1q_f32(inptr_ch6 + 4);

                            //1 123456-012345
                            float32x4_t vdin0_1234 = vextq_f32(vdatain20, vdatain21, 1);
                            float32x4_t vdin1_1234 = vextq_f32(vdatain30, vdatain31, 1);
                            float32x4_t vdin2_1234 = vextq_f32(vdatain40, vdatain41, 1);
                            float32x4_t vdin3_1234 = vextq_f32(vdatain50, vdatain51, 1);
                            float32x4_t vdin4_1234 = vextq_f32(vdatain60, vdatain61, 1);
                            float32x4_t vweights00 = vld1q_f32(weight_ch_in );
                            float32x4_t vweights10 = vld1q_f32(weight_ch_in + 7);
                            float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
                            float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                            float32x4_t vweights40 = vld1q_f32(weight_ch_in + 28);

                            float32x4_t vsum = vmulq_f32(vdin0_1234, vweights00);
                            vsum = vmlaq_f32(vsum, vdin1_1234, vweights10);
                            vsum = vmlaq_f32(vsum, vdin2_1234, vweights20);
                            vsum = vmlaq_f32(vsum, vdin3_1234, vweights30);
                            vsum = vmlaq_f32(vsum, vdin4_1234, vweights40);
                                
                            //out
                            float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                            float32x4_t vweights01 = vld1q_f32(weight_ch_in + 4);
                            vweights01 = vsetq_lane_f32(0.f, vweights01, 3);
                            float32x4_t vweights11 = vld1q_f32(weight_ch_in + 11);
                            vweights11 = vsetq_lane_f32(0.f, vweights11, 3);
                            float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
                            vweights21 = vsetq_lane_f32(0.f, vweights21, 3);
                            float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                            vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                            float32x4_t vweights41 = vld1q_f32(weight_ch_in + 32);
                            vweights41 = vsetq_lane_f32(0.f, vweights41, 3);

                            float32x4_t vw0_4500 = vsetq_lane_f32(0.f, vweights01, 2);
                            float32x4_t vw1_4500 = vsetq_lane_f32(0.f, vweights11, 2);
                            float32x4_t vw2_4500 = vsetq_lane_f32(0.f, vweights21, 2);
                            float32x4_t vw3_4500 = vsetq_lane_f32(0.f, vweights31, 2);
                            float32x4_t vw4_4500 = vsetq_lane_f32(0.f, vweights41, 2);
                            float32x4_t vdin0_5670 = vextq_f32(vdatain21, vzero, 1);
                            float32x4_t vdin1_5670 = vextq_f32(vdatain31, vzero, 1);
                            float32x4_t vdin2_5670 = vextq_f32(vdatain41, vzero, 1);
                            float32x4_t vdin3_5670 = vextq_f32(vdatain51, vzero, 1);
                            float32x4_t vdin4_5670 = vextq_f32(vdatain61, vzero, 1);
                            vsum = vmlaq_f32(vsum, vdin0_5670, vw0_4500);
                            vsum = vmlaq_f32(vsum, vdin1_5670, vw1_4500);
                            vsum = vmlaq_f32(vsum, vdin2_5670, vw2_4500);
                            vsum = vmlaq_f32(vsum, vdin3_5670, vw3_4500);
                            vsum = vmlaq_f32(vsum, vdin4_5670, vw4_4500);
                            float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                            vtotal = vpadd_f32(vtotal, vtotal);
                            vtotal = vset_lane_f32(0.f, vtotal, 1);
                            vdataout = vadd_f32(vdataout, vtotal);
                            vst1_f32(outptr_ch_wh, vdataout);
                            if(flag_relu)
                               outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                            outptr_ch_wh++;

                            //2 23456-01234;
                            float32x4_t vdin0_2345 = vextq_f32(vdatain20, vdatain21, 2);
                            float32x4_t vdin1_2345 = vextq_f32(vdatain30, vdatain31, 2);
                            float32x4_t vdin2_2345 = vextq_f32(vdatain40, vdatain41, 2);
                            float32x4_t vdin3_2345 = vextq_f32(vdatain50, vdatain51, 2);
                            float32x4_t vdin4_2345 = vextq_f32(vdatain60, vdatain61, 2);
                            vsum = vmulq_f32(vdin0_2345, vweights00);
                            vsum = vmlaq_f32(vsum, vdin1_2345, vweights10);
                            vsum = vmlaq_f32(vsum, vdin2_2345, vweights20);
                            vsum = vmlaq_f32(vsum, vdin3_2345, vweights30);
                            vsum = vmlaq_f32(vsum, vdin4_2345, vweights40);
                            vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                            vtotal = vpadd_f32(vtotal, vtotal);
                                
                            //out
                            vdataout = vld1_f32(outptr_ch_wh);
                            float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal,0));
                            vsum1 = vextq_f32(vsum1, vzero, 3);

                            float32x4_t vdin0_6700 = vextq_f32(vdatain21, vzero, 2);
                            float32x4_t vdin1_6700 = vextq_f32(vdatain31, vzero, 2);
                            float32x4_t vdin2_6700 = vextq_f32(vdatain41, vzero, 2);
                            float32x4_t vdin3_6700 = vextq_f32(vdatain51, vzero, 2);
                            float32x4_t vdin4_6700 = vextq_f32(vdatain61, vzero, 2);
                            vsum1 = vmlaq_f32(vsum1, vdin0_6700, vweights01);
                            vsum1 = vmlaq_f32(vsum1, vdin1_6700, vweights11);
                            vsum1 = vmlaq_f32(vsum1, vdin2_6700, vweights21);
                            vsum1 = vmlaq_f32(vsum1, vdin3_6700, vweights31);
                            vsum1 = vmlaq_f32(vsum1, vdin4_6700, vweights41);
                                
                            vtotal = vget_low_f32(vsum1);
                            vtotal = vset_lane_f32(0.f, vtotal, 1);
                            vdataout = vadd_f32(vdataout, vtotal);
                            vst1_f32(outptr_ch_wh, vdataout);
                            if(flag_relu)
                                outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                            outptr_ch_wh++;
                                
                            //1 3456 - 0123
                            vdataout = vld1_f32(outptr_ch_wh);
                            float32x4_t vdin0_3456 = vextq_f32(vdatain20, vdatain21, 3);
                            float32x4_t vdin1_3456 = vextq_f32(vdatain30, vdatain31, 3);
                            float32x4_t vdin2_3456 = vextq_f32(vdatain40, vdatain41, 3);
                            float32x4_t vdin3_3456 = vextq_f32(vdatain50, vdatain51, 3);
                            float32x4_t vdin4_3456 = vextq_f32(vdatain60, vdatain61, 3);
                            vsum = vmulq_f32(vdin0_3456, vweights00);
                            vsum = vmlaq_f32(vsum, vdin1_3456, vweights10);
                            vsum = vmlaq_f32(vsum, vdin2_3456, vweights20);
                            vsum = vmlaq_f32(vsum, vdin3_3456, vweights30);
                            vsum = vmlaq_f32(vsum, vdin4_3456, vweights40);
                            vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                            vtotal = vpadd_f32(vtotal, vtotal);
                            vtotal = vset_lane_f32(0.f, vtotal, 1);
                            vdataout = vadd_f32(vdataout, vtotal);
                            vst1_f32(outptr_ch_wh, vdataout);
                            if(flag_relu)
                               outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                            outptr_ch_wh++; 
                        }else if (h == h_out - kernel_h_even + 2){//2
                            for (int w = 0; w < w_out - kernel_w_even; w++){
                                float32x4_t vweights00 = vld1q_f32(weight_ch_in);
                                float32x4_t vweights01 = vld1q_f32(weight_ch_in + 4);
                                vweights01 = vsetq_lane_f32(0.f, vweights01, 3);
                                float32x4_t vweights10 = vld1q_f32(weight_ch_in + 7);
                                float32x4_t vweights11 = vld1q_f32(weight_ch_in + 11);
                                vweights11 = vsetq_lane_f32(0.f, vweights11, 3);
                                float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
                                float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
                                vweights21 = vsetq_lane_f32(0.f, vweights21, 3);
                                float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);
                                float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                                vweights31 = vsetq_lane_f32(0.f, vweights31, 3);
                                if (w == 0){//3456
                                    float32x4_t vw0_3456 = vextq_f32(vweights00, vweights01, 3);
                                    float32x4_t vw1_3456 = vextq_f32(vweights10, vweights11, 3);
                                    float32x4_t vw2_3456 = vextq_f32(vweights20, vweights21, 3);
                                    float32x4_t vw3_3456 = vextq_f32(vweights30, vweights31, 3);
                                    float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                    float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                    float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                    float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                                    

                                    float32x4_t vsum = vmulq_f32(vdatain30, vw0_3456);
                                    vsum = vmlaq_f32(vsum, vdatain40, vw1_3456);
                                    vsum = vmlaq_f32(vsum, vdatain50, vw2_3456);
                                    vsum = vmlaq_f32(vsum, vdatain60, vw3_3456);
                                    float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                    vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                    vtotal = vset_lane_f32(0.f, vtotal, 1);
                                    float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                    vdataout = vadd_f32(vtotal, vdataout);
                                    vst1_f32(outptr_ch_wh, vdataout);
                                    if(flag_relu)
                                       outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                    outptr_ch_wh++;
                                    continue;
                                }else if (w == 1){//23456
                                    float32x4_t vw0_2345 = vextq_f32(vweights00, vweights01, 2);
                                    float32x4_t vw1_2345 = vextq_f32(vweights10, vweights11, 2);
                                    float32x4_t vw2_2345 = vextq_f32(vweights20, vweights21, 2);
                                    float32x4_t vw3_2345 = vextq_f32(vweights30, vweights31, 2);
                                    float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                    float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                    float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                    float32x4_t vdatain60 = vld1q_f32(inptr_ch6);

                                    float32x4_t vsum = vmulq_f32(vdatain30, vw0_2345);
                                    vsum = vmlaq_f32(vsum, vdatain40, vw1_2345);
                                    vsum = vmlaq_f32(vsum, vdatain50, vw2_2345);
                                    vsum = vmlaq_f32(vsum, vdatain60, vw3_2345);
                                    float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                    vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                    float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                    vsum1 = vextq_f32(vsum1, vzero, 3);
                                    
                                    float32x4_t vw0_5670 = vextq_f32(vweights01, vzero, 2);
                                    float32x4_t vw1_5670 = vextq_f32(vweights11, vzero, 2);
                                    float32x4_t vw2_5670 = vextq_f32(vweights21, vzero, 2);
                                    float32x4_t vw3_5670 = vextq_f32(vweights31, vzero, 2);
                                    float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                    float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                    float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);
                                    float32x4_t vdatain61 = vld1q_f32(inptr_ch6 + 4);
                                    vsum1 = vmlaq_f32(vsum1, vdatain31, vw0_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain41, vw1_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain51, vw2_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain61, vw3_5670);//6
                                    vtotal = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));//x, 0
                                    float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                    vdataout = vadd_f32(vtotal, vdataout);
                                    vst1_f32(outptr_ch_wh, vdataout);
                                    if(flag_relu)
                                       outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                    outptr_ch_wh++;
                                    continue;
                                }else if (w == 2){//123456
                                    float32x4_t vw0_1234 = vextq_f32(vweights00, vweights01, 1);
                                    float32x4_t vw1_1234 = vextq_f32(vweights10, vweights11, 1);
                                    float32x4_t vw2_1234 = vextq_f32(vweights20, vweights21, 1);
                                    float32x4_t vw3_1234 = vextq_f32(vweights30, vweights31, 1);
                                    float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                                    float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                                    float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                                    float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                                    float32x4_t vsum = vmulq_f32(vdatain30, vw0_1234);
                                    vsum = vmlaq_f32(vsum, vdatain40, vw1_1234);
                                    vsum = vmlaq_f32(vsum, vdatain50, vw2_1234);
                                    vsum = vmlaq_f32(vsum, vdatain60, vw3_1234);
                                    float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                                    vtotal = vpadd_f32(vtotal, vtotal);//0+1+2+3
                                    float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal, 0));
                                    vsum1 = vextq_f32(vsum1, vzero, 3);

                                    float32x4_t vw0_5670 = vextq_f32(vweights01, vzero, 1);
                                    float32x4_t vw1_5670 = vextq_f32(vweights11, vzero, 1);
                                    float32x4_t vw2_5670 = vextq_f32(vweights21, vzero, 1);
                                    float32x4_t vw3_5670 = vextq_f32(vweights31, vzero, 1);
                                    float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                                    float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                                    float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);
                                    float32x4_t vdatain61 = vld1q_f32(inptr_ch6 + 4);
                                    vsum1 = vmlaq_f32(vsum1, vdatain31, vw0_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain41, vw1_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain51, vw2_5670);//6
                                    vsum1 = vmlaq_f32(vsum1, vdatain61, vw3_5670);//6
                                    vtotal = vpadd_f32(vget_low_f32(vsum1), vget_high_f32(vsum1));//x, 0
                                    float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                                    vdataout = vadd_f32(vdataout, vtotal);
                                    vst1_f32(outptr_ch_wh, vdataout);
                                    if(flag_relu)
                                       outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                                    outptr_ch_wh++;
                                    continue;
                                }
                            } 
                            //right
                            inptr_ch3 += w_in - 7;
                            inptr_ch4 += w_in - 7;
                            inptr_ch5 += w_in - 7;
                            inptr_ch6 += w_in - 7;
                            outptr_ch_wh +=w_out - 6;
                            float32x4_t vdatain30 = vld1q_f32(inptr_ch3);
                            float32x4_t vdatain31 = vld1q_f32(inptr_ch3 + 4);
                            float32x4_t vdatain40 = vld1q_f32(inptr_ch4);
                            float32x4_t vdatain41 = vld1q_f32(inptr_ch4 + 4);
                            float32x4_t vdatain50 = vld1q_f32(inptr_ch5);
                            float32x4_t vdatain51 = vld1q_f32(inptr_ch5 + 4);
                            float32x4_t vdatain60 = vld1q_f32(inptr_ch6);
                            float32x4_t vdatain61 = vld1q_f32(inptr_ch6 + 4);
                            float32x4_t vweights00 = vld1q_f32(weight_ch_in);
                            float32x4_t vweights10 = vld1q_f32(weight_ch_in + 7);
                            float32x4_t vweights20 = vld1q_f32(weight_ch_in + 14);
                            float32x4_t vweights30 = vld1q_f32(weight_ch_in + 21);

                            //1 123456-012345
                            float32x4_t vdin0_1234 = vextq_f32(vdatain30, vdatain31, 1);
                            float32x4_t vdin1_1234 = vextq_f32(vdatain40, vdatain41, 1);
                            float32x4_t vdin2_1234 = vextq_f32(vdatain50, vdatain51, 1);
                            float32x4_t vdin3_1234 = vextq_f32(vdatain60, vdatain61, 1);

                            float32x4_t vsum = vmulq_f32(vdin0_1234, vweights00);
                            vsum = vmlaq_f32(vsum, vdin1_1234, vweights10);
                            vsum = vmlaq_f32(vsum, vdin2_1234, vweights20);
                            vsum = vmlaq_f32(vsum, vdin3_1234, vweights30);
                                
                            //out
                            float32x2_t vdataout = vld1_f32(outptr_ch_wh);
                            
                            float32x4_t vweights01 = vld1q_f32(weight_ch_in + 4);
                            vweights01 = vsetq_lane_f32(0.f, vweights01, 3);
                            float32x4_t vweights11 = vld1q_f32(weight_ch_in + 11);
                            vweights11 = vsetq_lane_f32(0.f, vweights11, 3);
                            float32x4_t vweights21 = vld1q_f32(weight_ch_in + 18);
                            vweights21 = vsetq_lane_f32(0.f, vweights21, 3);
                            float32x4_t vweights31 = vld1q_f32(weight_ch_in + 25);
                            vweights31 = vsetq_lane_f32(0.f, vweights31, 3);

                            float32x4_t vw0_4500 = vsetq_lane_f32(0.f, vweights01, 2);
                            float32x4_t vw1_4500 = vsetq_lane_f32(0.f, vweights11, 2);
                            float32x4_t vw2_4500 = vsetq_lane_f32(0.f, vweights21, 2);
                            float32x4_t vw3_4500 = vsetq_lane_f32(0.f, vweights31, 2);
                            float32x4_t vdin0_5670 = vextq_f32(vdatain31, vzero, 1);
                            float32x4_t vdin1_5670 = vextq_f32(vdatain41, vzero, 1);
                            float32x4_t vdin2_5670 = vextq_f32(vdatain51, vzero, 1);
                            float32x4_t vdin3_5670 = vextq_f32(vdatain61, vzero, 1);
                            vsum = vmlaq_f32(vsum, vdin0_5670, vw0_4500);
                            vsum = vmlaq_f32(vsum, vdin1_5670, vw1_4500);
                            vsum = vmlaq_f32(vsum, vdin2_5670, vw2_4500);
                            vsum = vmlaq_f32(vsum, vdin3_5670, vw3_4500);
                            float32x2_t vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                            vtotal = vpadd_f32(vtotal, vtotal);
                            vtotal = vset_lane_f32(0.f, vtotal, 1);
                            vdataout = vadd_f32(vdataout, vtotal);
                            vst1_f32(outptr_ch_wh, vdataout);
                            if(flag_relu)
                                outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                            outptr_ch_wh++;

                            //2 23456-01234;
                            float32x4_t vdin0_2345 = vextq_f32(vdatain30, vdatain31, 2);
                            float32x4_t vdin1_2345 = vextq_f32(vdatain40, vdatain41, 2);
                            float32x4_t vdin2_2345 = vextq_f32(vdatain50, vdatain51, 2);
                            float32x4_t vdin3_2345 = vextq_f32(vdatain60, vdatain61, 2);
                            vsum = vmulq_f32(vdin0_2345, vweights00);
                            vsum = vmlaq_f32(vsum, vdin1_2345, vweights10);
                            vsum = vmlaq_f32(vsum, vdin2_2345, vweights20);
                            vsum = vmlaq_f32(vsum, vdin3_2345, vweights30);
                            vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                            vtotal = vpadd_f32(vtotal, vtotal);
                                
                            //out
                            vdataout = vld1_f32(outptr_ch_wh);
                            float32x4_t vsum1 = vdupq_n_f32(vget_lane_f32(vtotal,0));
                            vsum1 = vextq_f32(vsum1, vzero, 3);

                            float32x4_t vdin0_6700 = vextq_f32(vdatain31, vzero, 2);
                            float32x4_t vdin1_6700 = vextq_f32(vdatain41, vzero, 2);
                            float32x4_t vdin2_6700 = vextq_f32(vdatain51, vzero, 2);
                            float32x4_t vdin3_6700 = vextq_f32(vdatain61, vzero, 2);
                            vsum1 = vmlaq_f32(vsum1, vdin0_6700, vweights01);
                            vsum1 = vmlaq_f32(vsum1, vdin1_6700, vweights11);
                            vsum1 = vmlaq_f32(vsum1, vdin2_6700, vweights21);
                            vsum1 = vmlaq_f32(vsum1, vdin3_6700, vweights31);
                                
                            vtotal = vget_low_f32(vsum1);
                            vtotal = vset_lane_f32(0.f, vtotal, 1);
                            vdataout = vadd_f32(vdataout, vtotal);
                            vst1_f32(outptr_ch_wh, vdataout);
                            if(flag_relu)
                                outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                            outptr_ch_wh++;
                                
                            //1 3456 - 0123
                            float32x4_t vdin0_3456 = vextq_f32(vdatain30, vdatain31, 3);
                            float32x4_t vdin1_3456 = vextq_f32(vdatain40, vdatain41, 3);
                            float32x4_t vdin2_3456 = vextq_f32(vdatain50, vdatain51, 3);
                            float32x4_t vdin3_3456 = vextq_f32(vdatain60, vdatain61, 3);
                            vsum = vmulq_f32(vdin0_3456, vweights00);
                            vsum = vmlaq_f32(vsum, vdin1_3456, vweights10);
                            vsum = vmlaq_f32(vsum, vdin2_3456, vweights20);
                            vsum = vmlaq_f32(vsum, vdin3_3456, vweights30);
                            vtotal = vpadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
                            vtotal = vpadd_f32(vtotal, vtotal);
                            vtotal = vset_lane_f32(0.f, vtotal, 1);
                            vdataout = vld1_f32(outptr_ch_wh);
                            vdataout = vadd_f32(vdataout, vtotal);
                            vst1_f32(outptr_ch_wh, vdataout);
                            if(flag_relu)
                                outptr_ch_wh[0] = outptr_ch_wh[0] > 0 ? outptr_ch_wh[0]: 0.f;
                            outptr_ch_wh++; 
                        }
                    }
                    
                }
                
            }
        }
    }
}

} //lite

}//saber

}//anakin
#endif //USE_ARM_PLACE