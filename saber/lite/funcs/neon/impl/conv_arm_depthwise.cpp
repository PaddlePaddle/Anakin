#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

void conv_depthwise_3x3s1p1_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out);

void conv_depthwise_3x3s2p1_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out);

void conv_depthwise_3x3s1p1_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out);

void conv_depthwise_3x3s2p1_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out);


void conv_depthwise_3x3(const float* din, float* dout, \
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

    //const float *din = tensor_in.data();
    //float *dout = tensor_out.mutable_data();

    //! only support stride = 1 or 2
    //CHECK_EQ(stride_h, stride_w) << "stride w and h must = 1 or 2";

    if (stride_h == 1) {
        if (flag_relu) {
            conv_depthwise_3x3s1p1_bias_relu(dout, din, weights, bias, flag_bias, \
            num, ch_in, h_in, w_in, h_out, w_out);
        } else {
            conv_depthwise_3x3s1p1_bias(dout, din, weights, bias, flag_bias, \
            num, ch_in, h_in, w_in, h_out, w_out);
        }
    } else { //! stride = 2
        if (flag_relu) {
            conv_depthwise_3x3s2p1_bias_relu(dout, din, weights, bias, flag_bias, \
                num, ch_in, h_in, w_in, h_out, w_out);
        } else {
            conv_depthwise_3x3s2p1_bias(dout, din, weights, bias, flag_bias, \
                num, ch_in, h_in, w_in, h_out, w_out);
        }
    }
}

/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias
 */
void conv_depthwise_3x3s1p1_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    const float zero[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    //! for 4x6 convolution window
    const int right_pad_idx[4] = {3, 2, 1, 0};

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    int tile_w = (w_in + 3) >> 2;
    int tile_h = (h_in + 1) >> 1;
    int w_in_twice = w_in << 1;
    int cnt_col = tile_w - 2;

    int size_pad_right = 1 + (tile_w << 2) - w_in;
    int size_pad_bottom = 1 + (tile_h << 1) - h_in;

    uint32x4_t vmask_rp = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(size_pad_right));
    int right_pad_sub = (size_pad_right - 1) * sizeof(float);

    //printf("size_pad_right: %d, right_pad_sub: %d, cnt_col: %d\n", size_pad_right, right_pad_sub, cnt_col);
    unsigned int tmp1[4];
    vst1q_u32(tmp1, vmask_rp);
    //printf("mask_rp: %d, %d, %d, %d\n", tmp1[0], tmp1[1], tmp1[2], tmp1[3]);

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int i = 0; i < ch_in; ++i) {
            const float *din_channel = din_batch + i * size_in_channel;

            const float *weight_ptr = weights + i * 9;
            float32x4_t wr0 = vld1q_f32(weight_ptr);
            float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
            float32x4_t wr2 = vld1q_f32(weight_ptr + 6);
            float32x4_t wbias;
            if (flag_bias) {
                wbias  = vdupq_n_f32(bias[i]);
            } else {
                wbias = vdupq_n_f32(0.f);
            }

            float *dout_channel = dout_batch + i * size_out_channel;

            const float *dr0 = din_channel;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;
            const float *dr3 = dr2 + w_in;

            const float *din0_ptr = dr0;
            const float *din1_ptr = dr1;
            const float *din2_ptr = dr2;
            const float *din3_ptr = dr3;

            float *doutr0 = dout_channel;
            float *doutr1 = dout_channel + w_out;

            float* ptr_zero = const_cast<float*>(zero);

            //! deal with top pad
            int h = 0;
            //! process
#ifdef __aarch64__
            // todo
#else
            int cnt = cnt_col;
            asm volatile(
            //! process left pad
            "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vmul.f32 q5, q10, %e[wr0][1]           @ mul weight 00, outr1\n"
                    "vmul.f32 q4, q10, %e[wr1][1]           @ mul weight 10, outr0\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vmla.f32 q5, q12, %e[wr1][1]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][1]           @ mul weight 20, outr0\n"

                    "pld [%[din2_ptr], #192]                @ preload data\n"
                    "vld1.32  {d28-d30}, [%[din2_ptr]]!     @ load din r3\n"
                    "vmla.f32 q5, q14, %e[wr2][1]           @ mul weight 20, outr1\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 21, outr0\n"

                    "vext.32  q6, q14, q15, #1              @ shift left r3\n"
                    "vmla.f32 q5, q6,  %f[wr2][0]           @ mul weight 21, outr1\n"

                    "vmov.u32 d31, #0 @ zero\n"

                    "vext.32  d12, d31, d20, #1             @ shift right r1\n"
                    "vext.32  d13, d20, d21, #1             @ shift right r1\n"
                    "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"

                    "vext.32  d12, d31, d24, #1             @ shift right r0\n"
                    "vext.32  d13, d24, d25, #1             @ shift right r0\n"
                    "vmla.f32 q5, q6,  %e[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][0]           @ mul weight 22, outr0\n"

                    "vext.32  d12, d31, d28, #1             @ shift right r0\n"
                    "vext.32  d13, d28, d29, #1             @ shift right r0\n"
                    "vmla.f32 q5, q6,  %e[wr2][0]           @ mul weight 22, outr1\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "sub %[din0_ptr], #12                   @ 1pad + 2 float data overlap\n"
                    "sub %[din1_ptr], #12                   @ 1pad + 2 float data overlap\n"
                    "sub %[din2_ptr], #12                   @ 1pad + 2 float data overlap\n"

                    //! process mid cols
                    "cmp %[cnt], #1                             @ check whether has mid cols\n"
                    "blt  start_top_right                   @ jump to main loop start point\n"
                    "start_top_mid:                         @ main loop start point\n"
                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmul.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "pld [%[din2_ptr], #192]                @ preload data\n"
                    "vld1.32  {d28-d30}, [%[din2_ptr]]!     @ load din r3\n"
                    "vmla.f32 q5, q14, %e[wr2][0]           @ mul weight 20, outr1\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q6, q14, q15, #1              @ shift left r3\n"
                    "vmla.f32 q5, q6,  %e[wr2][1]           @ mul weight 21, outr1\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "vext.32  q6, q14, q15, #2              @ shift right r3\n"
                    "vmla.f32 q5, q6,  %f[wr2][0]           @ mul weight 22, outr1\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din2_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "subs %[cnt], #1                         @ loop count minus 1\n"
                    "bne    start_top_mid                   @ jump to main loop start point\n"

                    //! process right pad
                    "start_top_right:                       @ right pad entry\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vbif d21, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmul.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vbif d25, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "pld [%[din2_ptr], #192]                @ preload data\n"
                    "vld1.32  {d28-d30}, [%[din2_ptr]]!     @ load din r3\n"
                    "vbif d29, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vmla.f32 q5, q14, %e[wr2][0]           @ mul weight 20, outr1\n"

                    "vbif  d22, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vbif  d26, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vbif  d30, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q14, q15, #1              @ shift left r3\n"
                    "vmla.f32 q5, q6,  %e[wr2][1]           @ mul weight 21, outr1\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "pld [%[dout_ptr1], #128]               @ preload data\n"

                    "vext.32  q6, q14, q15, #2              @ shift right r3\n"
                    "vmla.f32 q5, q6,  %f[wr2][0]           @ mul weight 22, outr1\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"

                    "vmvn.32  d24, d31                      @ \n"
                    "vmvn.32  d25, d31                      @ \n"
                    "vext.32  q13, q12, %q[mask], #3        @ shift mask right 1\n"
                    "vbif q8, q10, q13                      @ bit select\n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"
                    "sub %[dout_ptr2], %[dout_ptr2], %[pad_right] @ sub \n"

            :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [pad_right] "+r" (right_pad_sub), \
                            [cnt] "+r"(cnt)
            :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                        [bias] "w"(wbias), [mask] "w" (vmask_rp)
            :"q4", "q5", "q6", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
            );

#endif //__aarch64__

            //! after process, increase pointer
            doutr0 += w_out;
            doutr1 = doutr0 + w_out;
            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            dr3 = dr2 + w_in;
            //! end of process top row

            //! process mid row
            for (h = tile_h - 2; h > 0; h--) {

                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                din3_ptr = dr3;

#ifdef __aarch64__
                // todo
#else
                cnt = cnt_col;
                asm volatile(
                //! process left pad
                "pld [%[din0_ptr], #192]               @ preload data\n"
                        "pld [%[din1_ptr], #192]               @ preload data\n"
                        "pld [%[din2_ptr], #192]               @ preload data\n"
                        "pld [%[din3_ptr], #192]               @ preload data\n"
                        "vld1.32  {d16-d18}, [%[din0_ptr]]!    @ load din r0\n"
                        "vmul.f32 q4, q8, %e[wr0][1]   @mul weight 00, outr0\n"

                        "vld1.32  {d20-d22}, [%[din1_ptr]]!    @ load din r1\n"
                        "vmul.f32 q5, q10, %e[wr0][1]  @mul weight 00, outr1\n"
                        "vmla.f32 q4, q10, %e[wr1][1]  @mul weight 10, outr0\n"

                        "vld1.32  {d24-d26}, [%[din2_ptr]]!    @ load din r2\n"
                        "vmla.f32 q5, q12, %e[wr1][1]  @mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][1]  @mul weight 20, outr0\n"


                        "vld1.32  {d28-d30}, [%[din3_ptr]]!    @ load din r3\n"
                        "vmla.f32 q5, q14, %e[wr2][1]  @mul weight 20, outr1\n"

                        "vext.32  q6, q8, q9, #1     @ shift left r0\n"
                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 01, outr0\n"

                        "vext.32  q6, q10, q11, #1   @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]  @mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]  @mul weight 11, outr0\n"

                        "pld [%[din0_ptr], #192]               @ preload data\n"

                        "vext.32  q6, q12, q13, #1  @ shift left r2\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 21, outr0\n"

                        "vext.32  q6, q14, q15, #1  @ shift left r3\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 21, outr1\n"

                        "vmov.u32 d31, #0 @ zero\n"
                        "vext.32  d12, d31, d16, #1     @ shift right r0\n"
                        "vext.32  d13, d16, d17, #1     @ shift right r0\n"
                        "vmla.f32 q4, q6, %e[wr0][0]  @mul weight 02, outr0\n"

                        "pld [%[din1_ptr], #192]               @ preload data\n"

                        "vext.32  d12, d31, d20, #1     @ shift right r1\n"
                        "vext.32  d13, d20, d21, #1     @ shift right r1\n"
                        "vmla.f32 q5, q6, %e[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][0]  @mul weight 12, outr0\n"

                        "pld [%[din2_ptr], #192]               @ preload data\n"

                        "vext.32  d12, d31, d24, #1     @ shift right r0\n"
                        "vext.32  d13, d24, d25, #1     @ shift right r0\n"
                        "vmla.f32 q5, q6,  %e[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][0] @mul weight 22, outr0\n"

                        "pld [%[din3_ptr], #192]               @ preload data\n"

                        "vext.32  d12, d31, d28, #1     @ shift right r0\n"
                        "vext.32  d13, d28, d29, #1     @ shift right r0\n"
                        "vmla.f32 q5, q6,  %e[wr2][0] @mul weight 22, outr1\n"

                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!  @ store result, add pointer\n"

                        "sub %[din0_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "sub %[din2_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "sub %[din3_ptr], #12 @ 1pad + 2 float data overlap\n"

                        //! process mid cols
                        "cmp %[cnt], #1                             @ check whether has mid cols\n"
                        "blt  start_mid_right                   @ jump to main loop start point\n"
                        "start_mid_mid:                 @ main loop start point\n"
                        "vld1.32  {d16-d18}, [%[din0_ptr]]!    @ load din r0\n"
                        "vmul.f32 q4, q8, %e[wr0][0]    @ mul weight 00, outr0\n"

                        "vld1.32  {d20-d22}, [%[din1_ptr]]!    @ load din r1\n"
                        "vmul.f32 q5, q10, %e[wr0][0]   @ mul weight 00, outr1\n"
                        "vmla.f32 q4, q10, %e[wr1][0]   @ mul weight 10, outr0\n"

                        "vld1.32  {d24-d26}, [%[din2_ptr]]!    @ load din r2\n"
                        "vmla.f32 q5, q12, %e[wr1][0]   @ mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][0]   @ mul weight 20, outr0\n"

                        "pld [%[din0_ptr], #192]               @ preload data\n"

                        "vld1.32  {d28-d30}, [%[din3_ptr]]!    @ load din r3\n"
                        "vmla.f32 q5, q14, %e[wr2][0]   @ mul weight 20, outr1\n"

                        "vext.32  q6, q8, q9, #1        @ shift left r0\n"
                        "vmla.f32 q4, q6, %e[wr0][1]    @ mul weight 01, outr0\n"

                        "vext.32  q6, q10, q11, #1      @ shift left r1\n"
                        "vmla.f32 q5, q6, %e[wr0][1]    @ mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][1]    @ mul weight 11, outr0\n"

                        "pld [%[din1_ptr], #192]               @ preload data\n"

                        "vext.32  q6, q12, q13, #1      @ shift left r2\n"
                        "vmla.f32 q5, q6,  %e[wr1][1]   @ mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][1]   @ mul weight 21, outr0\n"

                        "pld [%[din2_ptr], #192]               @ preload data\n"

                        "vext.32  q6, q14, q15, #1      @ shift left r3\n"
                        "vmla.f32 q5, q6,  %e[wr2][1]   @ mul weight 21, outr1\n"

                        "vext.32  q6, q8, q9, #2        @ shift left r0\n"
                        "vmla.f32 q4, q6, %f[wr0][0]    @ mul weight 02, outr0\n"

                        "vext.32  q6, q10, q11, #2      @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]    @ mul weight 02, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]    @ mul weight 12, outr0\n"

                        "pld [%[din3_ptr], #192]               @ preload data\n"

                        "vext.32  q6, q12, q13, #2      @ shift left r2\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 22, outr0\n"

                        "vext.32  q6, q14, q15, #2  @ shift right r3\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 22, outr1\n"

                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!  @ store result, add pointer\n"

                        "sub %[din0_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din2_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din3_ptr], #8 @ 2 float data overlap with previous data\n"
                        "subs %[cnt], #1 @ loop count minus 1\n"
                        "bne    start_mid_mid @ jump to main loop start point\n"

                        //! process right pad
                        "start_mid_right:                      @ right pad entry\n"
                        "vmov.u32  d31, #0                     @ zero buf\n"
                        "vld1.32  {d16-d18}, [%[din0_ptr]]!    @ load din r0\n"
                        "vbif d17, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vmul.f32 q4, q8, %e[wr0][0]           @ mul weight 00, outr0\n"

                        "vld1.32  {d20-d22}, [%[din1_ptr]]!    @ load din r1\n"
                        "vbif d21, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vmul.f32 q5, q10, %e[wr0][0]          @ mul weight 00, outr1\n"
                        "vmla.f32 q4, q10, %e[wr1][0]          @ mul weight 10, outr0\n"

                        "vld1.32  {d24-d26}, [%[din2_ptr]]!    @ load din r2\n"
                        "vbif d25, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q12, %e[wr1][0]          @ mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][0]          @ mul weight 20, outr0\n"

                        "vld1.32  {d28-d30}, [%[din3_ptr]]!    @ load din r3\n"
                        "vbif d29, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q14, %e[wr2][0]          @ mul weight 20, outr1\n"

                        "vbif  d18, d31, %f[mask]              @ bit select, deal with right pad\n"
                        "vext.32  q6, q8, q9, #1               @ shift left r0\n"
                        "vmla.f32 q4, q6, %e[wr0][1]           @ mul weight 01, outr0\n"

                        "vbif  d22, d31, %f[mask]              @ bit select, deal with right pad\n"
                        "vext.32  q6, q10, q11, #1             @ shift left r1\n"
                        "vmla.f32 q5, q6, %e[wr0][1]           @ mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][1]           @ mul weight 11, outr0\n"

                        "vbif  d26, d31, %f[mask]               @ bit select, deal with right pad\n"
                        "vext.32  q6, q12, q13, #1  @ shift left r2\n"
                        "vmla.f32 q5, q6,  %e[wr1][1] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][1] @mul weight 21, outr0\n"

                        "vbif  d30, d31, %f[mask] @ bit select, deal with right pad\n"
                        "vext.32  q6, q14, q15, #1  @ shift left r3\n"
                        "vmla.f32 q5, q6,  %e[wr2][1] @mul weight 21, outr1\n"

                        "vext.32  q6, q8, q9, #2     @ shift left r0\n"
                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 02, outr0\n"

                        "vext.32  q6, q10, q11, #2   @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]  @mul weight 12, outr0\n"

                        "vext.32  q6, q12, q13, #2  @ shift left r2\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 22, outr0\n"

                        "pld [%[dout_ptr1], #128]         @ preload data\n"

                        "vext.32  q6, q14, q15, #2  @ shift right r3\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 22, outr1\n"

                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "vld1.32  {d20-d21}, [%[dout_ptr1]]    @ load dout r0\n"

                        "vmvn.32  d22, d31 @ \n"
                        "vmvn.32  d23, d31 @ \n"
                        "vext.32  q12, q11, %q[mask], #3                @ shift mask right 1\n"
                        "vbif q8, q10, q12                              @ bit select\n"

                        "vst1.32  {d16-d17}, [%[dout_ptr1]]!            @ store result, add pointer\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!            @ store result, add pointer\n"

                        "sub %[dout_ptr1], %[dout_ptr1], %[pad_right]   @ sub \n"
                        "sub %[dout_ptr2], %[dout_ptr2], %[pad_right]   @ sub \n"

                :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                            [pad_right] "+r" (right_pad_sub), [cnt] "+r"(cnt)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                        [bias] "w"(wbias), [mask] "w" (vmask_rp)
                :"q4", "q5", "q6", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
                );
#endif //__aarch64__

                doutr0 += w_out;
                doutr1 = doutr0 + w_out;
                dr0 = dr2;
                dr1 = dr3;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
            } //! end of processing mid rows

            //! deal with bottom pad
            din0_ptr = dr0;
            din1_ptr = dr1;
            if (size_pad_bottom == 2){
                din2_ptr = ptr_zero;
            } else {
                din2_ptr = dr2;
            }
#ifdef __aarch64__
            // todo
#else
            cnt = cnt_col;
            asm volatile(
            // process left pad
            "pld [%[din0_ptr], #192]                        @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "pld [%[din2_ptr], #192]                @ preload data\n"

                    "vld1.32  {d16-d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vmul.f32 q4, q8, %e[wr0][1]            @ mul weight 00, outr0\n"

                    "vld1.32  {d20-d22}, [%[din1_ptr]]!     @ load din r1\n"
                    "vmul.f32 q5, q10, %e[wr0][1]           @ mul weight 00, outr1\n"
                    "vmla.f32 q4, q10, %e[wr1][1]           @ mul weight 10, outr0\n"

                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"
                    "vmla.f32 q5, q12, %e[wr1][1]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][1]           @ mul weight 20, outr0\n"

                    "vext.32  q6, q8, q9, #1                @ shift left r0\n"
                    "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 01, outr0\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 21, outr0\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"

                    "vmov.u32 d31, #0 @ zero\n"
                    "vext.32  d12, d31, d16, #1             @ shift right r0\n"
                    "vext.32  d13, d16, d17, #1             @ shift right r0\n"
                    "vmla.f32 q4, q6, %e[wr0][0]            @ mul weight 02, outr0\n"

                    "vext.32  d12, d31, d20, #1             @ shift right r1\n"
                    "vext.32  d13, d20, d21, #1             @ shift right r1\n"
                    "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"

                    "pld [%[din2_ptr], #192]                @ preload data\n"

                    "vext.32  d12, d31, d24, #1             @ shift right r2\n"
                    "vext.32  d13, d24, d25, #1             @ shift right r2\n"
                    "vmla.f32 q5, q6,  %e[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][0]           @ mul weight 22, outr0\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                    "sub %[din0_ptr], #12                   @ 1pad + 2 data overlap\n"
                    "sub %[din1_ptr], #12                   @ 1pad + 2 data overlap\n"
                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "beq    bot_mid_head                    @ jump to next block\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "add %[din2_ptr], #12                   @ 1pad + 2 data overlap\n"

                    // process mid cols
                    "bot_mid_head:                          @ header of bottom process\n"
                    "cmp %[cnt], #1                         @ check whether has mid cols\n"
                    "blt  start_bot_right                   @ jump to main loop start point\n"
                    "start_bot_mid:                         @ main loop start point\n"
                    "vld1.32  {d16-d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vmul.f32 q4, q8, %e[wr0][0]            @ mul weight 00, outr0\n"

                    "vld1.32  {d20-d22}, [%[din1_ptr]]!     @ load din r1\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmla.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "vext.32  q6, q8, q9, #1                @ shift left r0\n"
                    "vmla.f32 q4, q6, %e[wr0][1]            @ mul weight 01, outr0\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"

                    "vext.32  q6, q8, q9, #2                @ shift left r0\n"
                    "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 02, outr0\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "pld [%[din2_ptr], #192]                @ preload data\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                    "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"

                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "beq    end_bot_mid                     @ jump to check point\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "add %[din2_ptr], #16                   @ point to 4 data ahead\n"

                    "end_bot_mid:                           @ check point\n"
                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    start_bot_mid                   @ jump to main loop start point\n"

                    // process right pad
                    "start_bot_right:                       @ right pad process\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                    "vld1.32  {d16-d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vbif d17, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vmul.f32 q4, q8, %e[wr0][0]            @ mul weight 00, outr0\n"

                    "vld1.32  {d20-d22}, [%[din1_ptr]]!     @ load din r1\n"
                    "vbif d21, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmla.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"
                    "vbif d25, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "vbif  d18, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q8, q9, #1                @ shift left r0\n"
                    "vmla.f32 q4, q6, %e[wr0][1]            @ mul weight 01, outr0\n"

                    "vbif  d22, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vbif  d26, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q6, q8, q9, #2                @ shift left r0\n"
                    "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 02, outr0\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "pld [%[dout_ptr1], #128]               @ preload data\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "pld [%[dout_ptr2], #128]               @ preload data\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"


                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                    "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"

                    "vmvn.32  d24, d31                      @ \n"
                    "vmvn.32  d25, d31                      @ \n"
                    "vext.32  q13, q12, %q[mask], #3        @ shift mask right 1\n"
                    "vbif q8, q10, q13                      @ bit select\n"
                    "vbif q9, q11, q13                      @ bit select\n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"

                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "beq    end                             @ jump to end point\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "end:                                   @ end\n"

            :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [pad_right] "+r"(right_pad_sub), \
                            [bot_pad] "+r"(size_pad_bottom), [cnt] "+r"(cnt)
            :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                        [bias] "w"(wbias), [mask] "w"(vmask_rp)
            //, [test] "r"(data_test_ptr)
            :"q4", "q5", "q6", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
            );
#endif //__aarch64__
            //! end of processing bottom pad
        } // end of processing channels
    } // end of processing batchs
}

/**
 * \brief depthwise convolution kernel 3x3, stride 2
 */
void conv_depthwise_3x3s2p1_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s2 depthwise convolution, pad 1 is done implicit
    int right_pad_idx[4] = {0, 0, 0, 0};
    int right_w_idx[4] = {2, 1, 2, 1};
    int size_pad_right = w_out * 2 - w_in;
    int size_pad_bottom = h_out * 2 - h_in;
    int size_right_remain = (((w_out + 1) >> 1) << 1) - w_out;
    int cnt_col = ((w_out + 1) >> 1) - 2;
    if (size_right_remain == 0 || size_pad_right == 0) {
        right_pad_idx[0] = 1;
    }
    if (size_right_remain == 0) {
        right_pad_idx[1] = 1;
        if (size_pad_right == 0) {
            right_pad_idx[2] = 1;
        }
    }
    uint32x4_t mask_rp = vcgtq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(0));
    uint32x4_t mask_w = vcgtq_s32(vld1q_s32(right_w_idx), vdupq_n_s32(size_right_remain));

    size_right_remain *= sizeof(float);

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int i = 0; i < ch_in; ++i) {
            const float* din_channel = din_batch + i * size_in_channel;
            float* dout_channel = dout_batch + i * size_out_channel;

            const float *weight_ptr = weights + i * 9;
            float32x4_t wr0 = vld1q_f32(weight_ptr);
            float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
            float32x4_t wr2 = vld1q_f32(weight_ptr + 6);
            wr0 = vsetq_lane_f32(0.f, wr0, 3);
            wr1 = vsetq_lane_f32(0.f, wr1, 3);
            wr2 = vsetq_lane_f32(0.f, wr2, 3);
            float32x4_t wbias;
            if (flag_bias) {
                wbias  = vdupq_n_f32(bias[i]);
            } else {
                wbias = vdupq_n_f32(0.f);
            }

            const float *dr0 = din_channel;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;

            const float *din0_ptr = dr0;
            const float *din1_ptr = dr1;
            const float *din2_ptr = dr2;

            float *doutr0 = dout_channel;

            //! top pad
#ifdef __aarch64__
            // todo
#else
            int cnt = cnt_col;
            asm volatile(
            // process left pad
            "pld [%[din0_ptr], #128]                @ preload data\n"
                    "pld [%[din1_ptr], #128]                @ preload data\n"
                    "vmov.u32 q11, #0                       @ for left pad\n"
                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1\n"
                    "vext.32 q7, q11, q10, #3               @ shift right 1 data\n"
                    "vmul.f32 q8, q7, %q[wr1]               @ mul weight 1, out0\n"

                    "vext.32  q7, q10, q11, #1              @ shift left r1\n"
                    "vmul.f32 q9, q7, %q[wr1]               @ mul weight 1, out1\n"

                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r2\n"
                    "vext.32 q7, q11, q12, #3               @ shift right 1 data\n"
                    "vmla.f32 q8, q7, %q[wr2]               @ mul weight 2, out0\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"

                    "vext.32  q7, q12, q11, #1              @ shift left r2\n"
                    "vmla.f32 q9, q7,  %q[wr2]              @ mul weight 2, out1\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"

                    "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                    "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                    "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                    "vadd.f32  d17, d16, %e[bias]           @ add bias \n"

                    "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                    "sub %[din0_ptr], #4                    @ 1pad + 2 float data overlap\n"
                    "sub %[din1_ptr], #4                    @ 1pad + 2 float data overlap\n"

                    // process mid cols
                    "cmp %[cnt], #1                             @ check whether has mid loop\n"
                    "blt  s2_top_right                      @ jump to rightpad\n"
                    "s2_top_mid:                            @ main loop start point\n"

                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vmul.f32 q8, q10, %q[wr1]              @ mul weight 1, out0\n"

                    "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                    "vmul.f32 q9, q7, %q[wr1]               @ mul weight 1, outr1\n"

                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vmla.f32 q8, q12, %q[wr2]              @ mul weight 2, outr0\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"

                    "vext.32  q7, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q9, q7, %q[wr2]               @ mul weight 2, outr1\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"

                    "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                    "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                    "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                    "vadd.f32  d17, d16, %e[bias]           @ add bias \n"

                    "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                    "sub %[din0_ptr], #8                    @ 1 float data overlap and 1 redundant\n"
                    "sub %[din1_ptr], #8                    @ 1 float data overlap and 1 redundant\n"

                    "subs %[cnt], #1                         @ loop count minus 1\n"
                    "bne    s2_top_mid                      @ jump to main loop start point\n"

                    //! process right pad
                    "s2_top_right:                          @ right pad entry\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"

                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vbif d21, d31, %e[mask_din]            @ bit select, deal with right pad\n"
                    "vmul.f32 q8, q10, %q[wr1]              @ mul weight 1, out0\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"

                    "vbif d22, d31, %f[mask_din]            @ bit select, deal with right pad\n"
                    "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                    "vmul.f32 q9, q7, %q[wr1]               @ mul weight 1, out1\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "pld [%[dout_ptr1], #64]                @ preload data\n"

                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vbif d25, d31, %e[mask_din]            @ bit select, deal with right pad\n"
                    "vmla.f32 q8, q12, %q[wr2]              @ mul weight 2, outr0\n"

                    "vbif d26, d31, %f[mask_din]            @ bit select, deal with right pad\n"
                    "vext.32  q7, q12, q13, #2              @ shift left r1\n"
                    "vmla.f32 q9, q7, %q[wr2]               @ mul weight 2, outr1\n"

                    "vld1.32  {d20}, [%[dout_ptr1]]         @ load dout\n"
                    "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                    "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                    "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                    "vadd.f32  d17, d16, %e[bias]           @ add bias \n"
                    //"vst1.32  {d17},   [%[tmp_ptr]]! \n"
                    "vbif d17, d20, %e[mask_w]              @ bit select\n"
                    //"vst1.32  {d17},   [%[tmp_ptr]] \n"

                    "vst1.32  {d17}, [%[dout_ptr1]]!        @ store result, add pointer\n"
                    "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"

            :[dout_ptr1] "+r"(doutr0), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr), [cnt] "+r"(cnt)
            :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask_din] "w" (mask_rp), \
                    [mask_w] "w" (mask_w), \
                    [pad_right] "r" (size_right_remain)
            :"q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );

#endif //__aarch64__

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;

            //! process mid rows
            for (int j = h_out - size_pad_bottom - 1; j > 0; j--) {
#ifdef __aarch64__
                // todo
#else

                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;

                cnt = cnt_col;
                asm volatile(
                // process left pad
                "pld [%[din0_ptr], #128]                @ preload data\n"
                        "pld [%[din1_ptr], #128]                @ preload data\n"
                        "pld [%[din2_ptr], #128]                @ preload data\n"

                        "vmov.u32 q11, #0                       @ for left pad\n"
                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0\n"
                        "vext.32 q7, q11, q10, #3               @ shift right 1 data\n"
                        "vmul.f32 q8, q7, %q[wr0]               @ mul weight 00, outr0\n"

                        "vext.32  q7, q10, q11, #1              @ shift left r0\n"
                        "vmul.f32 q9, q7, %q[wr0]               @ mul weight 00, outr1\n"

                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1\n"
                        "vext.32 q7, q11, q12, #3               @ shift right 1 data\n"
                        "vmla.f32 q8, q7, %q[wr1]               @ mul weight 10, outr0\n"

                        "pld [%[din0_ptr], #128]                @ preload data\n"

                        "vext.32  q7, q12, q11, #1              @ shift left r1\n"
                        "vmla.f32 q9, q7,  %q[wr1]              @ mul weight 10, outr1\n"

                        "pld [%[din1_ptr], #128]                @ preload data\n"

                        "vld1.32  {d28-d29}, [%[din2_ptr]]!     @ load din r2\n"
                        "vext.32 q7, q11, q14, #3               @ shift right 1 data\n"
                        "vmla.f32 q8, q7, %q[wr2]               @ mul weight 20, outr0\n"

                        "pld [%[din2_ptr], #128]                @ preload data\n"

                        "vext.32  q7, q14, q11, #1              @ shift left r2\n"
                        "vmla.f32 q9, q7,  %q[wr2]              @ mul weight 20, outr1\n"

                        "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"

                        "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                        "sub %[din0_ptr], #4                    @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #4                    @ 1pad + 2 float data overlap\n"
                        "sub %[din2_ptr], #4                    @ 1pad + 2 float data overlap\n"

                        // process mid cols
                        "cmp %[cnt], #1                         @ check whether has mid loop\n"
                        "blt  s2_mid_right                      @ jump to rightpad\n"
                        "s2_mid_mid:                            @ main loop start point\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r0\n"
                        "vmul.f32 q8, q10, %q[wr0]              @ mul weight 00, outr0\n"
                        "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                        "vmul.f32 q9, q7, %q[wr0]               @ mul weight 00, outr1\n"

                        "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r1\n"
                        "vmla.f32 q8, q12, %q[wr1]              @ mul weight 10, outr0\n"
                        "vext.32  q7, q12, q13, #2              @ shift left r2\n"
                        "vmla.f32 q9, q7, %q[wr1]               @ mul weight 10, outr1\n"

                        "pld [%[din0_ptr], #128]                @ preload data\n"

                        "vld1.32  {d28-d30}, [%[din2_ptr]]!    @ load din r2\n"
                        "vmla.f32 q8, q14, %q[wr2]  @mul weight 10, outr0\n"

                        "pld [%[din1_ptr], #128]                @ preload data\n"

                        "vext.32  q7, q14, q15, #2      @ shift left r2\n"
                        "vmla.f32 q9, q7, %q[wr2]  @mul weight 10, outr1\n"

                        "pld [%[din2_ptr], #128]                @ preload data\n"

                        "vpadd.f32 d22, d16, d17  @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19 @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23 @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"

                        "vst1.32  {d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                        "sub %[din0_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din2_ptr], #8 @ 2 float data overlap with previous data\n"

                        "subs %[cnt], #1 @ loop count minus 1\n"
                        "bne    s2_mid_mid @ jump to main loop start point\n"

                        // process right pad
                        "s2_mid_right:  @ right pad entry\n"
                        "vmov.u32  d31, #0 @ zero buf\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!    @ load din r1\n"
                        "vbif d21, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vmul.f32 q8, q10, %q[wr0]  @mul weight 00, outr0\n"
                        "vbif d22, d31, %f[mask_din] @ bit select, deal with right pad\n"
                        "vext.32  q7, q10, q11, #2      @ shift left r0\n"
                        "vmul.f32 q9, q7, %q[wr0]  @mul weight 00, outr1\n"

                        "vld1.32  {d24-d26}, [%[din1_ptr]]!    @ load din r1\n"
                        "vbif d25, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vmla.f32 q8, q12, %q[wr1]  @mul weight 10, outr0\n"
                        "vbif d26, d31, %f[mask_din] @ bit select, deal with right pad\n"
                        "vext.32  q7, q12, q13, #2      @ shift left r1\n"
                        "vmla.f32 q9, q7, %q[wr1]  @mul weight 10, outr1\n"

                        "pld [%[dout_ptr1], #64]         @ preload ouput data\n"

                        "vld1.32  {d28-d30}, [%[din2_ptr]]!    @ load din r1\n"
                        "vbif d29, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vmla.f32 q8, q14, %q[wr2]  @mul weight 20, outr0\n"
                        "vbif d30, d31, %f[mask_din] @ bit select, deal with right pad\n"
                        "vext.32  q7, q14, q15, #2      @ shift left r2\n"
                        "vmla.f32 q9, q7, %q[wr2]  @mul weight 20, outr1\n"

                        "vld1.32  {d20}, [%[dout_ptr1]]    @ load dout\n"

                        "vpadd.f32 d22, d16, d17  @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19 @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23 @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"
                        "vbif d17, d20, %e[mask_w] @ bit select\n"

                        "vst1.32  {d17}, [%[dout_ptr1]]!  @ store result, add pointer\n"

                        "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"

                :[dout_ptr1] "+r"(doutr0), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr), [din2_ptr] "+r"(din2_ptr), \
                    [cnt] "+r"(cnt)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask_din] "w" (mask_rp), \
                    [mask_w] "w" (mask_w), \
                    [pad_right] "r" (size_right_remain)
                :"q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );

#endif //__aarch64__

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
            } // end of process mid rows

            // process bottom pad if needed
            if (size_pad_bottom) {
                din0_ptr = dr0;
                din1_ptr = dr1;
#ifdef __aarch64__
                // todo
#else
                cnt = cnt_col;
                asm volatile(
                // process left pad
                "pld [%[din0_ptr], #128]               @ preload data\n"
                        "pld [%[din1_ptr], #128]               @ preload data\n"

                        "vmov.u32 q11, #0 @ for left pad\n"
                        "vld1.32  {d20-d21}, [%[din0_ptr]]!    @ load din r0\n"
                        "vext.32 q7, q11, q10, #3 @ shift right 1 data\n"
                        "vmul.f32 q8, q7, %q[wr0]  @mul weight 00, outr0\n"
                        "vext.32  q7, q10, q11, #1   @ shift left r0\n"
                        "vmul.f32 q9, q7, %q[wr0]    @mul weight 00, outr1\n"

                        "vld1.32  {d24-d25}, [%[din1_ptr]]!    @ load din r1\n"
                        "vext.32 q7, q11, q12, #3 @ shift right 1 data\n"
                        "vmla.f32 q8, q7, %q[wr1]   @mul weight 10, outr0\n"
                        "vext.32  q7, q12, q11, #1   @ shift left r1\n"
                        "vmla.f32 q9, q7,  %q[wr1]   @mul weight 10, outr1\n"

                        "vpadd.f32 d22, d16, d17  @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19  @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23  @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"

                        "vst1.32  {d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                        "sub %[din0_ptr], #4 @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #4 @ 1pad + 2 float data overlap\n"

                        // process mid cols
                        "cmp %[cnt], #1                             @ check whether has mid loop\n"
                        "blt  s2_bot_right                      @ jump to rightpad\n"
                        "s2_bot_mid:                            @ main loop start point\n"
                        "pld [%[din0_ptr], #192]               @ preload data\n"
                        "pld [%[din1_ptr], #192]               @ preload data\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!    @ load din r0\n"
                        "vmul.f32 q8, q10, %q[wr0]      @mul weight 00, outr0\n"
                        "vext.32  q7, q10, q11, #2      @ shift left r1\n"
                        "vmul.f32 q9, q7, %q[wr0]  @mul weight 00, outr1\n"

                        "vld1.32  {d24-d26}, [%[din1_ptr]]!    @ load din r1\n"
                        "vmla.f32 q8, q12, %q[wr1]  @mul weight 10, outr0\n"
                        "vext.32  q7, q12, q13, #2      @ shift left r2\n"
                        "vmla.f32 q9, q7, %q[wr1]  @mul weight 10, outr1\n"

                        "vpadd.f32 d22, d16, d17  @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19 @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23 @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"

                        "vst1.32  {d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                        "sub %[din0_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8 @ 2 float data overlap with previous data\n"

                        "subs %[cnt], #1 @ loop count minus 1\n"
                        "bne    s2_bot_mid @ jump to main loop start point\n"

                        // process right pad
                        "s2_bot_right:    @ right pad entry\n"
                        "vmov.u32  d31, #0 @ zero buf\n"
                        "pld [%[din0_ptr], #192]               @ preload data\n"
                        "pld [%[din1_ptr], #192]               @ preload data\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!    @ load din r1\n"
                        "vbif d21, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vmul.f32 q8, q10, %q[wr0]  @mul weight 00, outr0\n"
                        "vbif d22, d31, %f[mask_din] @ bit select, deal with right pad\n"
                        "vext.32  q7, q10, q11, #2      @ shift left r0\n"
                        "vmul.f32 q9, q7, %q[wr0]  @mul weight 00, outr1\n"

                        "vld1.32  {d24-d26}, [%[din1_ptr]]!    @ load din r1\n"
                        "vbif d25, d31, %e[mask_din] @ bit select, deal with right pad\n"

                        "pld [%[dout_ptr1], #64]         @ preload data\n"

                        "vmla.f32 q8, q12, %q[wr1]  @mul weight 10, outr0\n"
                        "vbif d26, d31, %f[mask_din] @ bit select, deal with right pad\n"
                        "vext.32  q7, q12, q13, #2      @ shift left r1\n"
                        "vmla.f32 q9, q7, %q[wr1]  @mul weight 10, outr1\n"

                        "vld1.32  {d20}, [%[dout_ptr1]]    @ load dout\n"

                        "vpadd.f32 d22, d16, d17  @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19 @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23 @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"
                        "vbif d17, d20, %e[mask_w] @ bit select\n"

                        "vst1.32  {d17}, [%[dout_ptr1]]!  @ store result, add pointer\n"

                        "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"

                :[dout_ptr1] "+r"(doutr0), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr), [cnt] "+r"(cnt)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask_din] "w" (mask_rp), \
                    [mask_w] "w" (mask_w), \
                    [pad_right] "r" (size_right_remain)
                :"q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );

#endif //__aarch64__
            } // end of process bottom pad

        }
    }
}

void conv_depthwise_3x3s1p1_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    const float zero[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    //! for 4x6 convolution window
    const int right_pad_idx[4] = {3, 2, 1, 0};

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    int tile_w = (w_in + 3) >> 2;
    int tile_h = (h_in + 1) >> 1;
    int w_in_twice = w_in << 1;
    int cnt_col = tile_w - 2;

    int size_pad_right = 1 + (tile_w << 2) - w_in;
    int size_pad_bottom = 1 + (tile_h << 1) - h_in;

    uint32x4_t vmask_rp = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(size_pad_right));
    int right_pad_sub = (size_pad_right - 1) * sizeof(float);

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int i = 0; i < ch_in; ++i) {
            const float *din_channel = din_batch + i * size_in_channel;

            const float *weight_ptr = weights + i * 9;
            float32x4_t wr0 = vld1q_f32(weight_ptr);
            float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
            float32x4_t wr2 = vld1q_f32(weight_ptr + 6);
            float32x4_t wbias;
            if (flag_bias) {
                wbias  = vdupq_n_f32(bias[i]);
            } else {
                wbias = vdupq_n_f32(0.f);
            }

            float *dout_channel = dout_batch + i * size_out_channel;

            const float *dr0 = din_channel;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;
            const float *dr3 = dr2 + w_in;

            const float *din0_ptr = dr0;
            const float *din1_ptr = dr1;
            const float *din2_ptr = dr2;
            const float *din3_ptr = dr3;

            float *doutr0 = dout_channel;
            float *doutr1 = dout_channel + w_out;

            float* ptr_zero = const_cast<float*>(zero);

            int h = 0;
            //! deal with top pad
            //! process
#ifdef __aarch64__
            // todo
#else
            int cnt = cnt_col;
            asm volatile(
            //! process left pad
            "pld [%[din0_ptr], #192]                @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "pld [%[din2_ptr], #192]                @ preload data\n"

                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vmul.f32 q5, q10, %e[wr0][1]           @ mul weight 00, outr1\n"
                    "vmul.f32 q4, q10, %e[wr1][1]           @ mul weight 10, outr0\n"


                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vmla.f32 q5, q12, %e[wr1][1]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][1]           @ mul weight 20, outr0\n"


                    "vld1.32  {d28-d30}, [%[din2_ptr]]!     @ load din r3\n"
                    "vmla.f32 q5, q14, %e[wr2][1]           @ mul weight 20, outr1\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 21, outr0\n"

                    "vext.32  q6, q14, q15, #1              @ shift left r3\n"
                    "vmla.f32 q5, q6,  %f[wr2][0]           @ mul weight 21, outr1\n"

                    "vmov.u32 d31, #0 @ zero\n"

                    "vext.32  d12, d31, d20, #1             @ shift right r1\n"
                    "vext.32  d13, d20, d21, #1             @ shift right r1\n"
                    "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"

                    "vext.32  d12, d31, d24, #1             @ shift right r0\n"
                    "vext.32  d13, d24, d25, #1             @ shift right r0\n"
                    "vmla.f32 q5, q6,  %e[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][0]           @ mul weight 22, outr0\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"

                    "vext.32  d12, d31, d28, #1             @ shift right r0\n"
                    "vext.32  d13, d28, d29, #1             @ shift right r0\n"
                    "vmla.f32 q5, q6,  %e[wr2][0]           @ mul weight 22, outr1\n"

                    "pld [%[din2_ptr], #192]                @ preload data\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                    "vmax.f32 q8, q8, q4                    @ relu\n"
                    "vmax.f32 q9, q9, q4                    @ relu\n"
                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "sub %[din0_ptr], #12                   @ 1pad + 2 float data overlap\n"
                    "sub %[din1_ptr], #12                   @ 1pad + 2 float data overlap\n"
                    "sub %[din2_ptr], #12                   @ 1pad + 2 float data overlap\n"

                    //! process mid cols
                    "cmp %[cnt], #1                         @ check whether has mid cols\n"
                    "blt  start_top_right_relu              @ jump to right pad\n"
                    "start_top_mid_relu:                    @ main loop start point\n"
                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmul.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "vld1.32  {d28-d30}, [%[din2_ptr]]!     @ load din r3\n"
                    "vmla.f32 q5, q14, %e[wr2][0]           @ mul weight 20, outr1\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q6, q14, q15, #1              @ shift left r3\n"
                    "vmla.f32 q5, q6,  %e[wr2][1]           @ mul weight 21, outr1\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"

                    "vext.32  q6, q14, q15, #2              @ shift right r3\n"
                    "vmla.f32 q5, q6,  %f[wr2][0]           @ mul weight 22, outr1\n"

                    "pld [%[din2_ptr], #192]                @ preload data\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                    "vmax.f32 q8, q8, q4                    @ relu\n"
                    "vmax.f32 q9, q9, q4                    @ relu\n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din2_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    start_top_mid_relu              @ jump to main loop start point\n"

                    //! process right pad
                    "start_top_right_relu:                  @ right pad entry\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vbif d21, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmul.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vbif d25, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "vld1.32  {d28-d30}, [%[din2_ptr]]!     @ load din r3\n"
                    "vbif d29, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vmla.f32 q5, q14, %e[wr2][0]           @ mul weight 20, outr1\n"

                    "vbif  d22, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vbif  d26, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vbif  d30, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q14, q15, #1              @ shift left r3\n"
                    "vmla.f32 q5, q6,  %e[wr2][1]           @ mul weight 21, outr1\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "vext.32  q6, q14, q15, #2              @ shift right r3\n"
                    "vmla.f32 q5, q6,  %f[wr2][0]           @ mul weight 22, outr1\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                    "vmax.f32 q8, q8, q4                    @ relu\n"
                    "vmax.f32 q9, q9, q4                    @ relu\n"

                    "pld [%[dout_ptr1], #128]               @ preload data\n"
                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"

                    "vmvn.32  d24, d31                      @ \n"
                    "vmvn.32  d25, d31                      @ \n"
                    "vext.32  q13, q12, %q[mask], #3        @ shift mask right 1\n"
                    "vbif q8, q10, q13                      @ bit select\n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"
                    "sub %[dout_ptr2], %[dout_ptr2], %[pad_right] @ sub \n"

            :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                 [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                 [din2_ptr] "+r"(din2_ptr), [pad_right] "+r" (right_pad_sub), \
                 [cnt] "+r"(cnt)
            :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                [bias] "w"(wbias), [mask] "w" (vmask_rp)
            :"q4", "q5", "q6", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
            );

#endif //__aarch64__

            //! after process, increase pointer
            doutr0 += w_out;
            doutr1 = doutr0 + w_out;
            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            dr3 = dr2 + w_in;
            //! end of process top row

            //! process mid row
            for (h = tile_h - 2; h > 0; h--) {

                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                din3_ptr = dr3;

#ifdef __aarch64__
                // todo
#else
                cnt = cnt_col;
                asm volatile(
                //! process left pad
                "pld [%[din0_ptr], #192]               @ preload data\n"
                        "pld [%[din1_ptr], #192]               @ preload data\n"
                        "pld [%[din2_ptr], #192]               @ preload data\n"
                        "pld [%[din3_ptr], #192]               @ preload data\n"

                        "vld1.32  {d16-d18}, [%[din0_ptr]]!    @ load din r0\n"
                        "vmul.f32 q4, q8, %e[wr0][1]   @mul weight 00, outr0\n"

                        "vld1.32  {d20-d22}, [%[din1_ptr]]!    @ load din r1\n"
                        "vmul.f32 q5, q10, %e[wr0][1]  @mul weight 00, outr1\n"
                        "vmla.f32 q4, q10, %e[wr1][1]  @mul weight 10, outr0\n"

                        "vld1.32  {d24-d26}, [%[din2_ptr]]!    @ load din r2\n"
                        "vmla.f32 q5, q12, %e[wr1][1]  @mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][1]  @mul weight 20, outr0\n"

                        "vld1.32  {d28-d30}, [%[din3_ptr]]!    @ load din r3\n"
                        "vmla.f32 q5, q14, %e[wr2][1]  @mul weight 20, outr1\n"

                        "vext.32  q6, q8, q9, #1     @ shift left r0\n"
                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 01, outr0\n"

                        "vext.32  q6, q10, q11, #1   @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]  @mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]  @mul weight 11, outr0\n"

                        "vext.32  q6, q12, q13, #1  @ shift left r2\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 21, outr0\n"

                        "vext.32  q6, q14, q15, #1  @ shift left r3\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 21, outr1\n"

                        "vmov.u32 d31, #0 @ zero\n"
                        "vext.32  d12, d31, d16, #1     @ shift right r0\n"
                        "vext.32  d13, d16, d17, #1     @ shift right r0\n"
                        "vmla.f32 q4, q6, %e[wr0][0]  @mul weight 02, outr0\n"

                        "pld [%[din0_ptr], #192]               @ preload data\n"

                        "vext.32  d12, d31, d20, #1     @ shift right r1\n"
                        "vext.32  d13, d20, d21, #1     @ shift right r1\n"
                        "vmla.f32 q5, q6, %e[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][0]  @mul weight 12, outr0\n"

                        "pld [%[din1_ptr], #192]               @ preload data\n"

                        "vext.32  d12, d31, d24, #1     @ shift right r0\n"
                        "vext.32  d13, d24, d25, #1     @ shift right r0\n"
                        "vmla.f32 q5, q6,  %e[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][0] @mul weight 22, outr0\n"

                        "pld [%[din2_ptr], #192]               @ preload data\n"

                        "vext.32  d12, d31, d28, #1     @ shift right r0\n"
                        "vext.32  d13, d28, d29, #1     @ shift right r0\n"
                        "vmla.f32 q5, q6,  %e[wr2][0] @mul weight 22, outr1\n"

                        "pld [%[din3_ptr], #192]               @ preload data\n"

                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                        "vmax.f32 q8, q8, q4                    @ relu\n"
                        "vmax.f32 q9, q9, q4                    @ relu\n"

                        "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "vst1.32  {d18-d19},   [%[dout_ptr2]]!  @ store result, add pointer\n"

                        "sub %[din0_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "sub %[din2_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "sub %[din3_ptr], #12 @ 1pad + 2 float data overlap\n"

                        //! process mid cols
                        "cmp %[cnt], #1                             @ check whether has mid cols\n"
                        "blt  start_mid_right_relu              @ @ jump to right pad\n"
                        "start_mid_mid_relu:                 @ main loop start point\n"
                        "vld1.32  {d16-d18}, [%[din0_ptr]]!    @ load din r0\n"
                        "vmul.f32 q4, q8, %e[wr0][0]    @ mul weight 00, outr0\n"

                        "vld1.32  {d20-d22}, [%[din1_ptr]]!    @ load din r1\n"
                        "vmul.f32 q5, q10, %e[wr0][0]   @ mul weight 00, outr1\n"
                        "vmla.f32 q4, q10, %e[wr1][0]   @ mul weight 10, outr0\n"

                        "vld1.32  {d24-d26}, [%[din2_ptr]]!    @ load din r2\n"
                        "vmla.f32 q5, q12, %e[wr1][0]   @ mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][0]   @ mul weight 20, outr0\n"

                        "vld1.32  {d28-d30}, [%[din3_ptr]]!    @ load din r3\n"
                        "vmla.f32 q5, q14, %e[wr2][0]   @ mul weight 20, outr1\n"

                        "vext.32  q6, q8, q9, #1        @ shift left r0\n"
                        "vmla.f32 q4, q6, %e[wr0][1]    @ mul weight 01, outr0\n"

                        "vext.32  q6, q10, q11, #1      @ shift left r1\n"
                        "vmla.f32 q5, q6, %e[wr0][1]    @ mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][1]    @ mul weight 11, outr0\n"

                        "vext.32  q6, q12, q13, #1      @ shift left r2\n"
                        "vmla.f32 q5, q6,  %e[wr1][1]   @ mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][1]   @ mul weight 21, outr0\n"

                        "vext.32  q6, q14, q15, #1      @ shift left r3\n"
                        "vmla.f32 q5, q6,  %e[wr2][1]   @ mul weight 21, outr1\n"

                        "vext.32  q6, q8, q9, #2        @ shift left r0\n"
                        "vmla.f32 q4, q6, %f[wr0][0]    @ mul weight 02, outr0\n"

                        "pld [%[din0_ptr], #192]               @ preload data\n"

                        "vext.32  q6, q10, q11, #2      @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]    @ mul weight 02, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]    @ mul weight 12, outr0\n"

                        "pld [%[din1_ptr], #192]               @ preload data\n"

                        "vext.32  q6, q12, q13, #2      @ shift left r2\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 22, outr0\n"

                        "pld [%[din2_ptr], #192]               @ preload data\n"

                        "vext.32  q6, q14, q15, #2  @ shift right r3\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 22, outr1\n"

                        "pld [%[din3_ptr], #192]               @ preload data\n"

                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                        "vmax.f32 q8, q8, q4                    @ relu\n"
                        "vmax.f32 q9, q9, q4                    @ relu\n"

                        "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "vst1.32  {d18-d19},   [%[dout_ptr2]]!  @ store result, add pointer\n"

                        "sub %[din0_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din2_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din3_ptr], #8 @ 2 float data overlap with previous data\n"
                        "subs %[cnt], #1 @ loop count minus 1\n"
                        "bne    start_mid_mid_relu @ jump to main loop start point\n"

                        //! process right pad
                        "start_mid_right_relu:                 @ right pad entry\n"
                        "vmov.u32  d31, #0                     @ zero buf\n"
                        "vld1.32  {d16-d18}, [%[din0_ptr]]!    @ load din r0\n"
                        "vbif d17, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vmul.f32 q4, q8, %e[wr0][0]           @ mul weight 00, outr0\n"

                        "vld1.32  {d20-d22}, [%[din1_ptr]]!    @ load din r1\n"
                        "vbif d21, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vmul.f32 q5, q10, %e[wr0][0]          @ mul weight 00, outr1\n"
                        "vmla.f32 q4, q10, %e[wr1][0]          @ mul weight 10, outr0\n"

                        "vld1.32  {d24-d26}, [%[din2_ptr]]!    @ load din r2\n"
                        "vbif d25, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q12, %e[wr1][0]          @ mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][0]          @ mul weight 20, outr0\n"

                        "vld1.32  {d28-d30}, [%[din3_ptr]]!    @ load din r3\n"
                        "vbif d29, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q14, %e[wr2][0]          @ mul weight 20, outr1\n"

                        "vbif  d18, d31, %f[mask]              @ bit select, deal with right pad\n"
                        "vext.32  q6, q8, q9, #1               @ shift left r0\n"
                        "vmla.f32 q4, q6, %e[wr0][1]           @ mul weight 01, outr0\n"

                        "vbif  d22, d31, %f[mask]              @ bit select, deal with right pad\n"
                        "vext.32  q6, q10, q11, #1             @ shift left r1\n"
                        "vmla.f32 q5, q6, %e[wr0][1]           @ mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][1]           @ mul weight 11, outr0\n"

                        "vbif  d26, d31, %f[mask] @ bit select, deal with right pad\n"
                        "vext.32  q6, q12, q13, #1  @ shift left r2\n"
                        "vmla.f32 q5, q6,  %e[wr1][1] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][1] @mul weight 21, outr0\n"

                        "vbif  d30, d31, %f[mask] @ bit select, deal with right pad\n"
                        "vext.32  q6, q14, q15, #1  @ shift left r3\n"
                        "vmla.f32 q5, q6,  %e[wr2][1] @mul weight 21, outr1\n"

                        "vext.32  q6, q8, q9, #2     @ shift left r0\n"
                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 02, outr0\n"

                        "vext.32  q6, q10, q11, #2   @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]  @mul weight 12, outr0\n"

                        "vext.32  q6, q12, q13, #2  @ shift left r2\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 22, outr0\n"

                        "vext.32  q6, q14, q15, #2  @ shift right r3\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 22, outr1\n"

                        "pld [%[dout_ptr1], #128]         @ preload data\n"

                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                        "vmax.f32 q8, q8, q4                    @ relu\n"
                        "vmax.f32 q9, q9, q4                    @ relu\n"

                        "vld1.32  {d20-d21}, [%[dout_ptr1]]    @ load dout r0\n"

                        "vmvn.32  d22, d31 @ \n"
                        "vmvn.32  d23, d31 @ \n"
                        "vext.32  q12, q11, %q[mask], #3 @ shift mask right 1\n"
                        "vbif q8, q10, q12 @ bit select\n"

                        "vst1.32  {d16-d17}, [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!  @ store result, add pointer\n"

                        "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"
                        "sub %[dout_ptr2], %[dout_ptr2], %[pad_right] @ sub \n"

                :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                    [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                    [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                    [pad_right] "+r" (right_pad_sub), [cnt] "+r"(cnt)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask] "w" (vmask_rp)
                :"q4", "q5", "q6", "q8", "q9", \
                    "q10", "q11", "q12", "q13", "q14", "q15"
                );
#endif //__aarch64__

                doutr0 += w_out;
                doutr1 = doutr0 + w_out;
                dr0 = dr2;
                dr1 = dr3;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
            } //! end of processing mid rows

            //! deal with bottom pad
            din0_ptr = dr0;
            din1_ptr = dr1;
            if (size_pad_bottom == 2){
                din2_ptr = ptr_zero;
            } else {
                din2_ptr = dr2;
            }
            //! process
#ifdef __aarch64__
            // todo
#else
            cnt = cnt_col;
            asm volatile(
            //! process left pad
            "pld [%[din0_ptr], #192]                        @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "pld [%[din2_ptr], #192]                @ preload data\n"

                    "vld1.32  {d16-d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vmul.f32 q4, q8, %e[wr0][1]            @ mul weight 00, outr0\n"


                    "vld1.32  {d20-d22}, [%[din1_ptr]]!     @ load din r1\n"
                    "vmul.f32 q5, q10, %e[wr0][1]           @ mul weight 00, outr1\n"
                    "vmla.f32 q4, q10, %e[wr1][1]           @ mul weight 10, outr0\n"

                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"
                    "vmla.f32 q5, q12, %e[wr1][1]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][1]           @ mul weight 20, outr0\n"

                    "vext.32  q6, q8, q9, #1                @ shift left r0\n"
                    "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 01, outr0\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 21, outr0\n"

                    "vmov.u32 d31, #0 @ zero\n"
                    "vext.32  d12, d31, d16, #1             @ shift right r0\n"
                    "vext.32  d13, d16, d17, #1             @ shift right r0\n"
                    "vmla.f32 q4, q6, %e[wr0][0]            @ mul weight 02, outr0\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"

                    "vext.32  d12, d31, d20, #1             @ shift right r1\n"
                    "vext.32  d13, d20, d21, #1             @ shift right r1\n"
                    "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"

                    "vext.32  d12, d31, d24, #1             @ shift right r2\n"
                    "vext.32  d13, d24, d25, #1             @ shift right r2\n"
                    "vmla.f32 q5, q6,  %e[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][0]           @ mul weight 22, outr0\n"

                    "pld [%[din2_ptr], #192]                @ preload data\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                    "vmax.f32 q8, q8, q4                    @ relu\n"
                    "vmax.f32 q9, q9, q4                    @ relu\n"

                    "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                    "sub %[din0_ptr], #12                   @ 1pad + 2 data overlap\n"
                    "sub %[din1_ptr], #12                   @ 1pad + 2 data overlap\n"
                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "beq    bot_mid_head_relu               @ jump to next block\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "add %[din2_ptr], #12                   @ 1pad + 2 data overlap\n"

                    // process mid cols
                    "bot_mid_head_relu:                     @ header of bottom process\n"
                    "cmp %[cnt], #1                         @ check whether has mid cols\n"
                    "blt  start_bot_right_relu              @ jump to right pad\n"
                    "start_bot_mid_relu:                    @ main loop start point\n"
                    "vld1.32  {d16-d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vmul.f32 q4, q8, %e[wr0][0]            @ mul weight 00, outr0\n"

                    "vld1.32  {d20-d22}, [%[din1_ptr]]!     @ load din r1\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmla.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "vext.32  q6, q8, q9, #1                @ shift left r0\n"
                    "vmla.f32 q4, q6, %e[wr0][1]            @ mul weight 01, outr0\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q6, q8, q9, #2                @ shift left r0\n"
                    "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 02, outr0\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "pld [%[din2_ptr], #192]                @ preload data\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                    "vmax.f32 q8, q8, q4                    @ relu\n"
                    "vmax.f32 q9, q9, q4                    @ relu\n"

                    "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                    "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"

                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "beq    end_bot_mid_relu                @ jump to check point\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "add %[din2_ptr], #16                   @ point to 4 data ahead\n"

                    "end_bot_mid_relu:                      @ check point\n"
                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    start_bot_mid_relu              @ jump to main loop start point\n"

                    // process right pad
                    "start_bot_right_relu:                  @ right pad process\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                    "vld1.32  {d16-d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vbif d17, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vmul.f32 q4, q8, %e[wr0][0]            @ mul weight 00, outr0\n"

                    "vld1.32  {d20-d22}, [%[din1_ptr]]!     @ load din r1\n"
                    "vbif d21, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmla.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"
                    "vbif d25, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "vbif  d18, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q8, q9, #1                @ shift left r0\n"
                    "vmla.f32 q4, q6, %e[wr0][1]            @ mul weight 01, outr0\n"

                    "vbif  d22, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vbif  d26, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q6, q8, q9, #2                @ shift left r0\n"
                    "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 02, outr0\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "pld [%[dout_ptr1], #128]               @ preload data\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "pld [%[dout_ptr2], #128]               @ preload data\n"

                    "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                    "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                    "vmax.f32 q8, q8, q4                    @ relu\n"
                    "vmax.f32 q9, q9, q4                    @ relu\n"


                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                    "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"

                    "vmvn.32  d24, d31                      @ \n"
                    "vmvn.32  d25, d31                      @ \n"
                    "vext.32  q13, q12, %q[mask], #3        @ shift mask right 1\n"
                    "vbif q8, q10, q13                      @ bit select\n"
                    "vbif q9, q11, q13                      @ bit select\n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"

                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "beq    end_relu                        @ jump to end point\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "end_relu:  @ end\n"

            :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                [din2_ptr] "+r"(din2_ptr), [pad_right] "+r"(right_pad_sub), \
                [bot_pad] "+r"(size_pad_bottom), [cnt] "+r"(cnt)
            :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                [bias] "w"(wbias), [mask] "w"(vmask_rp)
            :"q4", "q5", "q6", "q8", "q9", \
                "q10", "q11", "q12", "q13", "q14", "q15"
            );
#endif //__aarch64__
            //! end of processing bottom pad
        } // end of processing channels
    } // end of processing batchs
}

/**
 * \brief depthwise convolution kernel 3x3, stride 2, with reulu
 */
void conv_depthwise_3x3s2p1_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s2 depthwise convolution, pad 1 is done implicit
    int right_pad_idx[4] = {0, 0, 0, 0};
    int right_w_idx[4] = {2, 1, 2, 1};
    int size_pad_right = w_out * 2 - w_in;
    int size_pad_bottom = h_out * 2 - h_in;
    int size_right_remain = (((w_out + 1) >> 1) << 1) - w_out;
    int cnt_col = ((w_out + 1) >> 1) - 2;

    const float zero[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    if (size_right_remain == 0 || size_pad_right == 0) {
        right_pad_idx[0] = 1;
    }
    if (size_right_remain == 0) {
        right_pad_idx[1] = 1;
        if (size_pad_right == 0) {
            right_pad_idx[2] = 1;
        }
    }
    uint32x4_t mask_rp = vcgtq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(0));
    uint32x4_t mask_w = vcgtq_s32(vld1q_s32(right_w_idx), vdupq_n_s32(size_right_remain));

    size_right_remain *= sizeof(float);

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int i = 0; i < ch_in; ++i) {
            const float* din_channel = din_batch + i * size_in_channel;
            float* dout_channel = dout_batch + i * size_out_channel;

            const float *weight_ptr = weights + i * 9;
            float32x4_t wr0 = vld1q_f32(weight_ptr);
            float32x4_t wr1 = vld1q_f32(weight_ptr + 3);
            float32x4_t wr2 = vld1q_f32(weight_ptr + 6);
            wr0 = vsetq_lane_f32(0.f, wr0, 3);
            wr1 = vsetq_lane_f32(0.f, wr1, 3);
            wr2 = vsetq_lane_f32(0.f, wr2, 3);
            float32x4_t wbias;
            if (flag_bias) {
                wbias  = vdupq_n_f32(bias[i]);
            } else {
                wbias = vdupq_n_f32(0.f);
            }

            const float *dr0 = din_channel;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;

            const float *din0_ptr = dr0;
            const float *din1_ptr = dr1;
            const float *din2_ptr = dr2;

            float *doutr0 = dout_channel;

            float32x4_t vzero = vdupq_n_f32(0.f);

            //! top pad

#ifdef __aarch64__
            // todo
#else
            int cnt = cnt_col;
            asm volatile(
            // process left pad
            "pld [%[din0_ptr], #128]                @ preload data\n"
                    "pld [%[din1_ptr], #128]                @ preload data\n"
                    "vmov.u32 q11, #0                       @ for left pad\n"
                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1\n"
                    "vext.32 q7, q11, q10, #3               @ shift right 1 data\n"
                    "vmul.f32 q8, q7, %q[wr1]               @ mul weight 1, out0\n"

                    "vext.32  q7, q10, q11, #1              @ shift left r1\n"
                    "vmul.f32 q9, q7, %q[wr1]               @ mul weight 1, out1\n"

                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r2\n"
                    "vext.32 q7, q11, q12, #3               @ shift right 1 data\n"
                    "vmla.f32 q8, q7, %q[wr2]               @ mul weight 2, out0\n"

                    "vext.32  q7, q12, q11, #1              @ shift left r2\n"
                    "vmla.f32 q9, q7,  %q[wr2]              @ mul weight 2, out1\n"

                    "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                    "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                    "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                    "vadd.f32  d17, d16, %e[bias]           @ add bias \n"
                    "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                    "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                    "sub %[din0_ptr], #4                    @ 1pad + 2 float data overlap\n"
                    "sub %[din1_ptr], #4                    @ 1pad + 2 float data overlap\n"

                    // process mid cols
                    "cmp %[cnt], #1                         @ check whether has mid loop\n"
                    "blt  s2_top_right_relu                 @ jump to rightpad\n"
                    "s2_top_mid_relu:                       @ main loop start point\n"
                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vmul.f32 q8, q10, %q[wr1]              @ mul weight 1, out0\n"

                    "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                    "vmul.f32 q9, q7, %q[wr1]               @ mul weight 1, outr1\n"

                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vmla.f32 q8, q12, %q[wr2]              @ mul weight 2, outr0\n"

                    "vext.32  q7, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q9, q7, %q[wr2]               @ mul weight 2, outr1\n"

                    "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                    "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                    "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                    "vadd.f32  d17, d16, %e[bias]           @ add bias \n"
                    "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                    "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                    "sub %[din0_ptr], #8                    @ 1 float data overlap and 1 redundant\n"
                    "sub %[din1_ptr], #8                    @ 1 float data overlap and 1 redundant\n"

                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    s2_top_mid_relu                 @ jump to main loop start point\n"

                    //! process right pad
                    "s2_top_right_relu:                     @ right pad entry\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"

                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vbif d21, d31, %e[mask_din]            @ bit select, deal with right pad\n"
                    "vmul.f32 q8, q10, %q[wr1]              @ mul weight 1, out0\n"

                    "vbif d22, d31, %f[mask_din]            @ bit select, deal with right pad\n"
                    "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                    "vmul.f32 q9, q7, %q[wr1]               @ mul weight 1, out1\n"

                    "pld [%[dout_ptr1], #64]                @ preload data\n"
                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vbif d25, d31, %e[mask_din]            @ bit select, deal with right pad\n"
                    "vmla.f32 q8, q12, %q[wr2]              @ mul weight 2, outr0\n"

                    "vbif d26, d31, %f[mask_din]            @ bit select, deal with right pad\n"
                    "vext.32  q7, q12, q13, #2              @ shift left r1\n"
                    "vmla.f32 q9, q7, %q[wr2]               @ mul weight 2, outr1\n"

                    "vld1.32  {d20}, [%[dout_ptr1]]         @ load dout\n"
                    "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                    "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                    "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                    "vadd.f32  d17, d16, %e[bias]           @ add bias \n"
                    "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                    "vbif d17, d20, %e[mask_w]              @ bit select\n"

                    "vst1.32  {d17}, [%[dout_ptr1]]!        @ store result, add pointer\n"
                    "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"

            :[dout_ptr1] "+r"(doutr0), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr), [cnt] "+r"(cnt)
            :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask_din] "w" (mask_rp), \
                    [mask_w] "w" (mask_w), [vzero] "w" (vzero), \
                    [pad_right] "r" (size_right_remain)
            :"q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );

#endif //__aarch64__

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            //! end of top pad

            //! process mid rows
            for (int j = h_out - size_pad_bottom - 1; j > 0; j--) {
#ifdef __aarch64__
                // todo
#else
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;

                cnt = cnt_col;
                asm volatile(
                // process left pad
                "pld [%[din0_ptr], #128]                @ preload data\n"
                        "pld [%[din1_ptr], #128]                @ preload data\n"
                        "pld [%[din2_ptr], #128]                @ preload data\n"

                        "vmov.u32 q11, #0                       @ for left pad\n"
                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0\n"
                        "vext.32 q7, q11, q10, #3               @ shift right 1 data\n"
                        "vmul.f32 q8, q7, %q[wr0]               @ mul weight 00, outr0\n"

                        "vext.32  q7, q10, q11, #1              @ shift left r0\n"
                        "vmul.f32 q9, q7, %q[wr0]               @ mul weight 00, outr1\n"

                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1\n"
                        "vext.32 q7, q11, q12, #3               @ shift right 1 data\n"
                        "vmla.f32 q8, q7, %q[wr1]               @ mul weight 10, outr0\n"

                        "vext.32  q7, q12, q11, #1              @ shift left r1\n"
                        "vmla.f32 q9, q7,  %q[wr1]              @ mul weight 10, outr1\n"

                        "vld1.32  {d28-d29}, [%[din2_ptr]]!     @ load din r2\n"
                        "vext.32 q7, q11, q14, #3               @ shift right 1 data\n"
                        "vmla.f32 q8, q7, %q[wr2]               @ mul weight 20, outr0\n"

                        "vext.32  q7, q14, q11, #1              @ shift left r2\n"
                        "vmla.f32 q9, q7,  %q[wr2]              @ mul weight 20, outr1\n"

                        "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"
                        "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                        "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                        "sub %[din0_ptr], #4                    @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #4                    @ 1pad + 2 float data overlap\n"
                        "sub %[din2_ptr], #4                    @ 1pad + 2 float data overlap\n"

                        // process mid cols
                        "cmp %[cnt], #1                         @ check whether has mid loop\n"
                        "blt  s2_mid_right_relu                 @ jump to rightpad\n"
                        "s2_mid_mid_relu:                       @ main loop start point\n"
                        "pld [%[din0_ptr], #192]                @ preload data\n"
                        "pld [%[din1_ptr], #192]                @ preload data\n"
                        "pld [%[din2_ptr], #192]                @ preload data\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r0\n"
                        "vmul.f32 q8, q10, %q[wr0]              @ mul weight 00, outr0\n"
                        "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                        "vmul.f32 q9, q7, %q[wr0]               @ mul weight 00, outr1\n"

                        "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r1\n"
                        "vmla.f32 q8, q12, %q[wr1]              @ mul weight 10, outr0\n"
                        "vext.32  q7, q12, q13, #2              @ shift left r2\n"
                        "vmla.f32 q9, q7, %q[wr1]               @ mul weight 10, outr1\n"

                        "vld1.32  {d28-d30}, [%[din2_ptr]]!    @ load din r2\n"
                        "vmla.f32 q8, q14, %q[wr2]  @mul weight 10, outr0\n"
                        "vext.32  q7, q14, q15, #2      @ shift left r2\n"
                        "vmla.f32 q9, q7, %q[wr2]  @mul weight 10, outr1\n"

                        "vpadd.f32 d22, d16, d17  @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19 @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23 @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"
                        "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                        "vst1.32  {d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                        "sub %[din0_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din2_ptr], #8 @ 2 float data overlap with previous data\n"

                        "subs %[cnt], #1 @ loop count minus 1\n"
                        "bne    s2_mid_mid_relu @ jump to main loop start point\n"

                        // process right pad
                        "s2_mid_right_relu:  @ right pad entry\n"
                        "vmov.u32  d31, #0 @ zero buf\n"
                        "pld [%[din0_ptr], #192]               @ preload data\n"
                        "pld [%[din1_ptr], #192]               @ preload data\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!    @ load din r1\n"
                        "vbif d21, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vmul.f32 q8, q10, %q[wr0]  @mul weight 00, outr0\n"
                        "vbif d22, d31, %f[mask_din] @ bit select, deal with right pad\n"
                        "vext.32  q7, q10, q11, #2      @ shift left r0\n"
                        "vmul.f32 q9, q7, %q[wr0]  @mul weight 00, outr1\n"

                        "vld1.32  {d24-d26}, [%[din1_ptr]]!    @ load din r1\n"
                        "vbif d25, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vmla.f32 q8, q12, %q[wr1]  @mul weight 10, outr0\n"
                        "vbif d26, d31, %f[mask_din] @ bit select, deal with right pad\n"
                        "vext.32  q7, q12, q13, #2      @ shift left r1\n"
                        "vmla.f32 q9, q7, %q[wr1]  @mul weight 10, outr1\n"

                        "pld [%[dout_ptr1], #64]         @ preload ouput data\n"

                        "vld1.32  {d28-d30}, [%[din2_ptr]]!    @ load din r1\n"
                        "vbif d29, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vmla.f32 q8, q14, %q[wr2]  @mul weight 20, outr0\n"
                        "vbif d30, d31, %f[mask_din] @ bit select, deal with right pad\n"
                        "vext.32  q7, q14, q15, #2      @ shift left r2\n"
                        "vmla.f32 q9, q7, %q[wr2]  @mul weight 20, outr1\n"

                        "vld1.32  {d20}, [%[dout_ptr1]]    @ load dout\n"

                        "vpadd.f32 d22, d16, d17  @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19 @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23 @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"
                        "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                        "vbif d17, d20, %e[mask_w] @ bit select\n"

                        "vst1.32  {d17}, [%[dout_ptr1]]!  @ store result, add pointer\n"

                        "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"

                :[dout_ptr1] "+r"(doutr0), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr), [din2_ptr] "+r"(din2_ptr), \
                    [cnt] "+r"(cnt)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias),[mask_din] "w" (mask_rp), \
                    [mask_w] "w" (mask_w), [vzero] "w" (vzero), \
                    [pad_right] "r" (size_right_remain)
                :"q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );

#endif //__aarch64__

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
            } // end of process mid rows

            //! process bottom pad if needed
            if (size_pad_bottom) {
                din0_ptr = dr0;
                din1_ptr = dr1;
#ifdef __aarch64__
                // todo
#else
                cnt = cnt_col;
                asm volatile(
                // process left pad
                "pld [%[din0_ptr], #128]               @ preload data\n"
                        "pld [%[din1_ptr], #128]                @ preload data\n"

                        "vmov.u32 q11, #0 @ for left pad\n"
                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1\n"

                        "vext.32 q7, q11, q10, #3               @ shift right 1 data\n"
                        "vmul.f32 q8, q7, %q[wr0]               @ mul weight 00, outr0\n"
                        "vext.32  q7, q10, q11, #1              @ shift left r0\n"
                        "vmul.f32 q9, q7, %q[wr0]               @ mul weight 00, outr1\n"

                        "vext.32 q7, q11, q12, #3               @ shift right 1 data\n"
                        "vmla.f32 q8, q7, %q[wr1]               @ mul weight 10, outr0\n"
                        "vext.32  q7, q12, q11, #1              @ shift left r1\n"
                        "vmla.f32 q9, q7,  %q[wr1]              @ mul weight 10, outr1\n"

                        "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]           @ add bias \n"
                        "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                        "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                        "sub %[din0_ptr], #4                    @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #4                    @ 1pad + 2 float data overlap\n"

                        // process mid cols
                        "cmp %[cnt], #1                         @ check whether has mid loop\n"
                        "blt  s2_bot_right_relu                 @ jump to rightpad\n"
                        "s2_bot_mid_relu:                       @ main loop start point\n"
                        "pld [%[din0_ptr], #192]                @ preload data\n"
                        "pld [%[din1_ptr], #192]                @ preload data\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r0\n"
                        "vmul.f32 q8, q10, %q[wr0]              @ mul weight 00, outr0\n"
                        "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                        "vmul.f32 q9, q7, %q[wr0]               @ mul weight 00, outr1\n"

                        "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r1\n"
                        "vmla.f32 q8, q12, %q[wr1]              @ mul weight 10, outr0\n"
                        "vext.32  q7, q12, q13, #2              @ shift left r2\n"
                        "vmla.f32 q9, q7, %q[wr1]               @ mul weight 10, outr1\n"

                        "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]           @ add bias \n"
                        "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                        "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                        "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"

                        "subs %[cnt], #1                        @ loop count minus 1\n"
                        "bne    s2_bot_mid_relu                 @ jump to main loop start point\n"

                        // process right pad
                        "s2_bot_right_relu:                     @ right pad entry\n"
                        "vmov.u32  d31, #0                      @ zero buf\n"
                        "pld [%[din0_ptr], #192]                @ preload data\n"
                        "pld [%[din1_ptr], #192]                @ preload data\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                        "vbif d21, d31, %e[mask_din]            @ bit select, deal with right pad\n"
                        "vmul.f32 q8, q10, %q[wr0]              @ mul weight 00, outr0\n"
                        "vbif d22, d31, %f[mask_din]            @ bit select, deal with right pad\n"
                        "vext.32  q7, q10, q11, #2              @ shift left r0\n"
                        "vmul.f32 q9, q7, %q[wr0]               @ mul weight 00, outr1\n"

                        "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r1\n"
                        "vbif d25, d31, %e[mask_din]            @ bit select, deal with right pad\n"

                        "pld [%[dout_ptr1], #64]                @ preload data\n"

                        "vmla.f32 q8, q12, %q[wr1]              @ mul weight 10, outr0\n"
                        "vbif d26, d31, %f[mask_din]            @ bit select, deal with right pad\n"
                        "vext.32  q7, q12, q13, #2              @ shift left r1\n"
                        "vmla.f32 q9, q7, %q[wr1]               @ mul weight 10, outr1\n"

                        "vld1.32  {d20}, [%[dout_ptr1]]         @ load dout\n"

                        "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]           @ add bias \n"
                        "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                        "vbif d17, d20, %e[mask_w]              @ bit select\n"

                        "vst1.32  {d17}, [%[dout_ptr1]]!        @ store result, add pointer\n"

                        //"sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"

                :[dout_ptr1] "+r"(doutr0), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr), [cnt] "+r"(cnt)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask_din] "w" (mask_rp), \
                    [mask_w] "w" (mask_w), [vzero] "w" (vzero), \
                    [pad_right] "r" (size_right_remain)
                :"q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
#endif //__aarch64__
            } //! end of process bottom pad
        }
    }
}

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE
