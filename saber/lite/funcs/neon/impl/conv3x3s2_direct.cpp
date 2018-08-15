#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"
#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/arm_utils.h"

namespace anakin{

namespace saber{

namespace lite{
#ifdef __aarch64__
void conv_3x3s2_direct(const float* din, float* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const float* weights, const float* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Sgemm& gemmer, void* work_space) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    //! for 4x6 convolution window
    const unsigned int right_pad_idx[8] = {5, 4, 3, 2, 1, 0, 0, 0};

    int w_in = win;
    int h_in = hin;
    int ch_in = chin;

    int w_out = wout;
    int h_out = hout;
    int ch_out = chout;

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = ch_in * 9;

    int w_cnt = (wout + 3) >> 2;
    int h_cnt = (hout + 1) >> 1;
    int cnt_col = w_cnt - 2; //sub left and right pad

//    unsigned int size_pad_right = (unsigned int)(1 + (w_cnt << 2) - w_in);
//    int size_pad_bottom = 1 + (h_cnt << 1) - h_in;

    int cremain = ch_out & 1;

//    uint32x4_t vmask_rp1 = vcgeq_u32(vld1q_u32(right_pad_idx), vdupq_n_u32(size_pad_right));
//    uint32x4_t vmask_rp2 = vcgeq_u32(vld1q_u32(right_pad_idx + 4), vdupq_n_u32(size_pad_right));
//    uint32x4_t vmask_result = vcgtq_u32(vld1q_u32(right_pad_idx), vdupq_n_u32(size_pad_right));

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_out - 1; c += 2) {

            float* dout_c0 = dout_batch + c * size_out_channel;
            float* dout_c1 = dout_c0 + size_out_channel;

            // fixme, fill bias
//            if (flag_bias) {
//                fill_bias(dout_c0, &bias[c], 1, size_out_channel);
//                fill_bias(dout_c1, &bias[c + 1], 1, size_out_channel);
//            } else {
//                fill_bias(dout_c0, zero, 1, size_out_channel);
//                fill_bias(dout_c1, zero, 1, size_out_channel);
//            }

            const float* wc0 = weights + c * w_stride;
            const float* wc1 = wc0 + w_stride;


            for (int i = 0; i < ch_in; ++i) {

                int relu = 0;
                if ((i == ch_in - 1) && flag_relu) {
                    relu = 1;
                }

                const float *din_channel = din_batch + i * size_in_channel;

                const float* wcin0 = wc0 + i * 9;
                const float* wcin1 = wc1 + i * 9;

                float32x4_t wr00 = vld1q_f32(wcin0);        //q0
                float32x4_t wr01 = vld1q_f32(wcin0 + 3);    //q1
                float32x4_t wr02 = vld1q_f32(wcin0 + 6);    //q2

                float32x4_t wr10 = vld1q_f32(wcin1);        //q3
                float32x4_t wr11 = vld1q_f32(wcin1 + 3);    //q4
                float32x4_t wr12 = vld1q_f32(wcin1 + 6);    //q5

                //! left for 4 out channel, q6, q7, q8, q9, q10, q11

                //! 2 output channel each has 2 output rows
                float *doutc0r0 = dout_c0;
                float *doutc0r1 = doutc0r0 + w_out;

                float *doutc1r0 = dout_c1;
                float *doutc1r1 = doutc1r0 + w_out;

                //! load 5 rows from input
                const float *dr0 = din_channel;
                const float *dr1 = dr0 + w_in;
                const float *dr2 = dr1 + w_in;
                const float *dr3 = dr2 + w_in;
                const float *dr4 = dr3 + w_in;

                const float *din0_ptr = dr0;
                const float *din1_ptr = dr1;
                const float *din2_ptr = dr2;
                const float *din3_ptr = dr3;
                const float *din4_ptr = dr4;

                //prefetch input
                prefetch(din0_ptr);
                prefetch(din1_ptr);
                prefetch(din2_ptr);
                prefetch(din3_ptr);
                prefetch(din4_ptr);

                float* ptr_zero = const_cast<float*>(zero);
                float32x4_t vzero = vdupq_n_f32(0.f);

                //! deal with top pad
                int h = 0;
                {
                    //! process
                    if (0) {
                        // process left
                        // 4 rows
                        float32x4_t vq12_oc0r0 = vld1q_f32(doutc0r0);
                        float32x4_t vq13_oc0r1 = vld1q_f32(doutc0r1);
                        float32x4_t vq14_oc1r0 = vld1q_f32(doutc1r0);
                        float32x4_t vq15_oc1r1 = vld1q_f32(doutc1r1);
                        //! q16, q17, q18, q19 left for 4 out channels

                        float32x4x2_t vq20q21_r0 = vld2q_f32(din0_ptr);
                        float32x4x2_t vq22q23_r1 = vld2q_f32(din1_ptr);
                        float32x4x2_t vq24q25_r2 = vld2q_f32(din2_ptr);
                        float32x4x2_t vq26q27_r3 = vld2q_f32(din3_ptr);
                        //! q28, q29, left for loading 5th row

                        //! mul weights[1]
                        //input r2, get output r1
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq24q25_r2.val[0], wr01, 1);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq24q25_r2.val[0], wr11, 1);
                        //input r0, get output r0
                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq20q21_r0.val[0], wr01, 1);
                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq20q21_r0.val[0], wr11, 1);

                        //input r1, get output r1,r0
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq22q23_r1.val[0], wr00, 1);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq22q23_r1.val[0], wr10, 1);
                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq22q23_r1.val[0], wr02, 1);
                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq22q23_r1.val[0], wr12, 1);

                        //input r3, get output r1
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq26q27_r3.val[0], wr02, 1);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq26q27_r3.val[0], wr12, 1);

                        //! do something here, pld
                        //! mul weights[2]
                        //input r2, get output r1
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq24q25_r2.val[1], wr01, 2);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq24q25_r2.val[1], wr11, 2);
                        //input r0, get output r0
                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq20q21_r0.val[1], wr01, 2);
                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq20q21_r0.val[1], wr11, 2);

                        //input r1, get output r1,r0
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq22q23_r1.val[1], wr00, 2);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq22q23_r1.val[1], wr10, 2);
                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq22q23_r1.val[1], wr02, 2);
                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq22q23_r1.val[1], wr12, 2);

                        //input r3, get output r1
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq26q27_r3.val[1], wr02, 2);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq26q27_r3.val[1], wr12, 2);

                        //! do something here
                        //! mul weights[0]
                        // r0, r1, r2 shift right
                        float32x4_t vtmp1 = vextq_f32(vzero, vq20q21_r0.val[1], 3);
                        float32x4_t vtmp2 = vextq_f32(vzero, vq22q23_r1.val[1], 3);
                        float32x4_t vtmp3 = vextq_f32(vzero, vq24q25_r2.val[1], 3);
                        float32x4_t vtmp4 = vextq_f32(vzero, vq26q27_r3.val[1], 3);

                        //input r2, get output r1
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vtmp3, wr01, 0);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vtmp3, wr11, 0);
                        //input r0, get output r0
                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vtmp1, wr01, 0);
                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vtmp1, wr11, 0);

                        //input r1, get output r1,r0
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vtmp2, wr00, 0);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vtmp2, wr10, 0);
                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vtmp2, wr02, 0);
                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vtmp2, wr12, 0);

                        //input r3, get output r1
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vtmp4, wr02, 0);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vtmp4, wr12, 0);

                        din0_ptr += 7;
                        //prefetch(din0_ptr);
                        din1_ptr += 7;
                        //prefetch(din1_ptr);
                        din2_ptr += 7;
                        //prefetch(din2_ptr);
                        din3_ptr += 7;
                        //prefetch(din3_ptr);

//                        if (relu) {
//                            vq12_oc0r0 = vmaxq_f32(vq12_oc0r0, vzero);
//                            vq14_oc1r0 = vmaxq_f32(vq14_oc1r0, vzero);
//                            vq13_oc0r1 = vmaxq_f32(vq13_oc0r1, vzero);
//                            vq15_oc1r1 = vmaxq_f32(vq15_oc1r1, vzero);
//                        }

                        vst1q_f32(doutc0r0, vq12_oc0r0);
                        vst1q_f32(doutc0r1, vq13_oc0r1);
                        vst1q_f32(doutc1r0, vq14_oc1r0);
                        vst1q_f32(doutc1r1, vq15_oc1r1);

                        doutc0r0 += 4;
                        doutc0r1 += 4;
                        doutc1r0 += 4;
                        doutc1r1 += 4;

                        //process mid
                        //! load input data, first iter
                        vq20q21_r0 = vld2q_f32(din0_ptr);
                        vq22q23_r1 = vld2q_f32(din1_ptr);
                        vq24q25_r2 = vld2q_f32(din2_ptr);
                        vq26q27_r3 = vld2q_f32(din3_ptr);
                        //! q28, q29, left for loading 5th row
                        for (int j = 0; j < cnt_col; ++j) {

                            // 4 rows
                            //! load output data
                            vq12_oc0r0 = vld1q_f32(doutc0r0);
                            vq13_oc0r1 = vld1q_f32(doutc0r1);
                            vq14_oc1r0 = vld1q_f32(doutc1r0);
                            vq15_oc1r1 = vld1q_f32(doutc1r1);
                            //! q16, q17, q18, q19 left for 4 out channels

                            //! mul weights[1]
                            //input r2, get output r1
                            vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq24q25_r2.val[0], wr01, 0);
                            vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq24q25_r2.val[0], wr11, 0);
                            //input r0, get output r0
                            vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq20q21_r0.val[0], wr01, 0);
                            vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq20q21_r0.val[0], wr11, 0);

                            //input r1, get output r1,r0
                            vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq22q23_r1.val[0], wr00, 0);
                            vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq22q23_r1.val[0], wr10, 0);
                            vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq22q23_r1.val[0], wr02, 0);
                            vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq22q23_r1.val[0], wr12, 0);

                            //input r3, get output r1
                            vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq26q27_r3.val[0], wr02, 0);
                            vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq26q27_r3.val[0], wr12, 0);

                            //! do something here, pld
                            //! mul weights[2]
                            //input r2, get output r1
                            vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq24q25_r2.val[1], wr01, 1);
                            vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq24q25_r2.val[1], wr11, 1);
                            //input r0, get output r0
                            vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq20q21_r0.val[1], wr01, 1);
                            vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq20q21_r0.val[1], wr11, 1);

                            //input r1, get output r1,r0
                            vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq22q23_r1.val[1], wr00, 1);
                            vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq22q23_r1.val[1], wr10, 1);
                            vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq22q23_r1.val[1], wr02, 1);
                            vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq22q23_r1.val[1], wr12, 1);

                            //input r3, get output r1
                            vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq26q27_r3.val[1], wr02, 1);
                            vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq26q27_r3.val[1], wr12, 1);

                            //! copy data
                            vtmp1 = vaddq_f32(vzero, vq20q21_r0.val[0]);
                            vtmp2 = vaddq_f32(vzero, vq22q23_r1.val[0]);
                            vtmp3 = vaddq_f32(vzero, vq24q25_r2.val[0]);
                            vtmp4 = vaddq_f32(vzero, vq26q27_r3.val[0]);

                            din0_ptr += 8;
                            //prefetch(din0_ptr);
                            din1_ptr += 8;
                            //prefetch(din1_ptr);
                            din2_ptr += 8;
                            //prefetch(din2_ptr);
                            din3_ptr += 8;
                            //prefetch(din3_ptr);

                            //! load input data, net iter
                            vq20q21_r0 = vld2q_f32(din0_ptr);
                            vq22q23_r1 = vld2q_f32(din1_ptr);
                            vq24q25_r2 = vld2q_f32(din2_ptr);
                            vq26q27_r3 = vld2q_f32(din3_ptr);
                            //! q28, q29, left for loading 5th row

                            //! do something here
                            //! mul weights[0]
                            // r0, r1, r2 shift right
                            vtmp1 = vextq_f32(vtmp1, vq20q21_r0.val[0], 1);
                            vtmp2 = vextq_f32(vtmp2, vq22q23_r1.val[0], 1);
                            vtmp3 = vextq_f32(vtmp3, vq24q25_r2.val[0], 1);
                            vtmp4 = vextq_f32(vtmp4, vq26q27_r3.val[0], 1);

                            //input r2, get output r1
                            vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vtmp3, wr01, 2);
                            vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vtmp3, wr11, 2);
                            //input r0, get output r0
                            vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vtmp1, wr01, 2);
                            vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vtmp1, wr11, 2);

                            //input r1, get output r1,r0
                            vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vtmp2, wr00, 2);
                            vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vtmp2, wr10, 2);
                            vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vtmp2, wr02, 2);
                            vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vtmp2, wr12, 2);

                            //input r3, get output r1
                            vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vtmp4, wr02, 2);
                            vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vtmp4, wr12, 2);

//                            if (relu) {
//                                vq12_oc0r0 = vmaxq_f32(vq12_oc0r0, vzero);
//                                vq14_oc1r0 = vmaxq_f32(vq14_oc1r0, vzero);
//                                vq13_oc0r1 = vmaxq_f32(vq13_oc0r1, vzero);
//                                vq15_oc1r1 = vmaxq_f32(vq15_oc1r1, vzero);
//                            }

                            vst1q_f32(doutc0r0, vq12_oc0r0);
                            vst1q_f32(doutc0r1, vq13_oc0r1);
                            vst1q_f32(doutc1r0, vq14_oc1r0);
                            vst1q_f32(doutc1r1, vq15_oc1r1);

                            doutc0r0 += 4;
                            doutc0r1 += 4;
                            doutc1r0 += 4;
                            doutc1r1 += 4;
                        }

#if 0
                        //process right
                        // 3 rows
                        vq10_oc0r0 = vld1q_f32(doutc0r0);
                        vq11_oc0r1 = vld1q_f32(doutc0r1);
                        vq12_oc1r0 = vld1q_f32(doutc1r0);
                        vq13_oc1r1 = vld1q_f32(doutc1r1);

                        vq0_r00 = vld1q_f32(din0_ptr);
                        vq1_r01 = vld1q_f32(din0_ptr + 4);
                        vq2_r10 = vld1q_f32(din1_ptr);
                        vq3_r11 = vld1q_f32(din1_ptr + 4);
                        vq4_r20 = vld1q_f32(din2_ptr);
                        vq5_r21 = vld1q_f32(din2_ptr + 4);

                        // bit select, right pad zero
                        vq0_r00 = vbslq_f32(vmask_rp1, vq0_r00, vzero);
                        vq1_r01 = vbslq_f32(vmask_rp2, vq1_r01, vzero);
                        vq2_r10 = vbslq_f32(vmask_rp1, vq2_r10, vzero);
                        vq3_r11 = vbslq_f32(vmask_rp2, vq3_r11, vzero);
                        vq4_r20 = vbslq_f32(vmask_rp1, vq4_r20, vzero);
                        vq5_r21 = vbslq_f32(vmask_rp2, vq5_r21, vzero);

                        //input r0
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq0_r00, wr01, 0);
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vq0_r00, wr00, 0);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq0_r00, wr11, 0);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vq0_r00, wr10, 0);

                        //input r1
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq2_r10, wr02, 0);
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vq2_r10, wr01, 0);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq2_r10, wr12, 0);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vq2_r10, wr11, 0);

                        //input r2
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vq4_r20, wr02, 0);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vq4_r20, wr12, 0);

                        // r0, r1, r2 shift left
                        vtmp1 = vextq_f32(vq0_r00, vq1_r01, 1);
                        vtmp2 = vextq_f32(vq2_r10, vq3_r11, 1);
                        vtmp3 = vextq_f32(vq4_r20, vq5_r21, 1);

                        // input r0
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp1, wr01, 1);
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp1, wr00, 1);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp1, wr11, 1);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp1, wr10, 1);

                        //input r1
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp2, wr02, 1);
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp2, wr01, 1);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp2, wr12, 1);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp2, wr11, 1);

                        //input r2
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp3, wr02, 1);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp3, wr12, 1);

                        // r0, r1, r2 shift right
                        vtmp1 = vextq_f32(vq0_r00, vq1_r01, 2);
                        vtmp2 = vextq_f32(vq2_r10, vq3_r11, 2);
                        vtmp3 = vextq_f32(vq4_r20, vq5_r21, 2);

                        // input r0
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp1, wr01, 2);
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp1, wr00, 2);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp1, wr11, 2);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp1, wr10, 2);

                        //input r1
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp2, wr02, 2);
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp2, wr01, 2);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp2, wr12, 2);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp2, wr11, 2);

                        vq0_r00 = vld1q_f32(doutc0r0);
                        vq1_r01 = vld1q_f32(doutc1r0);

                        //input r2
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp3, wr02, 2);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp3, wr12, 2);

                        vq4_r20 = vld1q_f32(doutc0r1);
                        vq5_r21 = vld1q_f32(doutc1r1);

                        if (relu) {
                            vq10_oc0r0 = vmaxq_f32(vq10_oc0r0, vzero);
                            vq11_oc0r1 = vmaxq_f32(vq11_oc0r1, vzero);
                            vq12_oc1r0 = vmaxq_f32(vq12_oc1r0, vzero);
                            vq13_oc1r1 = vmaxq_f32(vq13_oc1r1, vzero);
                            vq14_oc2r0 = vmaxq_f32(vq14_oc2r0, vzero);
                            vq15_oc2r1 = vmaxq_f32(vq15_oc2r1, vzero);
                            vq16_oc3r0 = vmaxq_f32(vq16_oc3r0, vzero);
                            vq17_oc3r1 = vmaxq_f32(vq17_oc3r1, vzero);
                        }


                        vq10_oc0r0 = vbslq_f32(vmask_result, vq10_oc0r0, vq0_r00);
                        vq12_oc1r0 = vbslq_f32(vmask_result, vq12_oc1r0, vq1_r01);
                        vq14_oc2r0 = vbslq_f32(vmask_result, vq14_oc2r0, vq2_r10);
                        vq16_oc3r0 = vbslq_f32(vmask_result, vq16_oc3r0, vq3_r11);

                        vq11_oc0r1 = vbslq_f32(vmask_result, vq11_oc0r1, vq4_r20);
                        vq13_oc1r1 = vbslq_f32(vmask_result, vq13_oc1r1, vq5_r21);
                        vq15_oc2r1 = vbslq_f32(vmask_result, vq15_oc2r1, vtmp1);
                        vq17_oc3r1 = vbslq_f32(vmask_result, vq17_oc3r1, vtmp2);

                        vst1q_f32(doutc0r0, vq10_oc0r0);
                        vst1q_f32(doutc0r1, vq11_oc0r1);
                        vst1q_f32(doutc1r0, vq12_oc1r0);
                        vst1q_f32(doutc1r1, vq13_oc1r1);
                        vst1q_f32(doutc2r0, vq14_oc2r0);
                        vst1q_f32(doutc2r1, vq15_oc2r1);
                        vst1q_f32(doutc3r0, vq16_oc3r0);
                        vst1q_f32(doutc3r1, vq17_oc3r1);
#endif
                    }
                    //! after process, increase address
                    dr0 = dr3;
                    dr1 = dr4;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                    dr4 = dr3 + w_in;
                } //! end of process top row

                //! process mid row
                for (h = 1; h < h_cnt - 1; h++) {

                    doutc0r0 = dout_c0 + 2 * h * w_out;
                    doutc0r1 = doutc0r0 + w_out;
                    doutc1r0 = dout_c1 + 2 * h * w_out;
                    doutc1r1 = doutc1r0 + w_out;

                    din0_ptr = dr0;
                    din1_ptr = dr1;
                    din2_ptr = dr2;
                    din3_ptr = dr3;
                    din4_ptr = dr4;

                    // process left
                    // 4 rows
                    float32x4_t vq12_oc0r0 = vld1q_f32(doutc0r0);
                    float32x4_t vq13_oc0r1 = vld1q_f32(doutc0r1);
                    float32x4_t vq14_oc1r0 = vld1q_f32(doutc1r0);
                    float32x4_t vq15_oc1r1 = vld1q_f32(doutc1r1);
                    //! q16, q17, q18, q19 left for 4 out channels

                    float32x4x2_t vq20q21_r0 = vld2q_f32(din0_ptr);
                    float32x4x2_t vq22q23_r1 = vld2q_f32(din1_ptr);
                    float32x4x2_t vq24q25_r2 = vld2q_f32(din2_ptr);
                    float32x4x2_t vq26q27_r3 = vld2q_f32(din3_ptr);
                    float32x4x2_t vq28q29_r4 = vld2q_f32(din4_ptr);

//                    //! mul weights[1]
//                    //input r0, get output r0
//                    vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq20q21_r0.val[0], wr00, 1);
//                    vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq20q21_r0.val[0], wr10, 1);
//                    //input r3, get output r1
//                    vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq26q27_r3.val[0], wr01, 1);
//                    vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq26q27_r3.val[0], wr11, 1);
//
//                    //input r1, get output r0
//                    vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq22q23_r1.val[0], wr01, 1);
//                    vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq22q23_r1.val[0], wr11, 1);
//                    //input r4, get output r1
//                    vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq28q29_r4.val[0], wr02, 1);
//                    vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq28q29_r4.val[0], wr12, 1);
//
//                    //input r2, get output r1,r0
//                    vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq24q25_r2.val[0], wr02, 1);
//                    vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq24q25_r2.val[0], wr12, 1);
//                    vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq24q25_r2.val[0], wr00, 1);
//                    vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq24q25_r2.val[0], wr10, 1);
//
//                    //! do something here, pld
//                    //! mul weights[2]
//                    //input r0, get output r0
//                    vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq20q21_r0.val[1], wr00, 2);
//                    vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq20q21_r0.val[1], wr10, 2);
//                    //input r3, get output r1
//                    vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq26q27_r3.val[1], wr01, 2);
//                    vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq26q27_r3.val[1], wr11, 2);
//
//                    //input r1, get output r0
//                    vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq22q23_r1.val[1], wr01, 2);
//                    vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq22q23_r1.val[1], wr11, 2);
//                    //input r4, get output r1
//                    vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq28q29_r4.val[1], wr02, 2);
//                    vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq28q29_r4.val[1], wr12, 2);
//
//                    //input r2, get output r1,r0
//                    vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq24q25_r2.val[1], wr02, 2);
//                    vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq24q25_r2.val[1], wr12, 2);
//                    vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq24q25_r2.val[1], wr00, 2);
//                    vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq24q25_r2.val[1], wr10, 2);

                    //! do something here
                    //! mul weights[0]
                    // r0, r1, r2, r3, r4 shift right
                    float32x4_t vtmp0 = vextq_f32(vzero, vq20q21_r0.val[1], 3);
                    float32x4_t vtmp1 = vextq_f32(vzero, vq22q23_r1.val[1], 3);
                    float32x4_t vtmp2 = vextq_f32(vzero, vq24q25_r2.val[1], 3);
                    float32x4_t vtmp3 = vextq_f32(vzero, vq26q27_r3.val[1], 3);
                    float32x4_t vtmp4 = vextq_f32(vzero, vq28q29_r4.val[1], 3);
//
//                    //input r0, get output r0
//                    vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vtmp0, wr00, 0);
//                    vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vtmp0, wr10, 0);
//                    //input r3, get output r1
//                    vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vtmp3, wr01, 0);
//                    vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vtmp3, wr11, 0);
//
//                    //input r1, get output r0
//                    vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vtmp1, wr01, 0);
//                    vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vtmp1, wr11, 0);
//                    //input r4, get output r1
//                    vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vtmp4, wr02, 0);
//                    vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vtmp4, wr12, 0);
//
//                    //input r2, get output r1,r0
//                    vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vtmp2, wr02, 0);
//                    vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vtmp2, wr12, 0);
//                    vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vtmp2, wr00, 0);
//                    vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vtmp2, wr10, 0);

                    din0_ptr += 7;
                    //prefetch(din0_ptr);
                    din1_ptr += 7;
                    //prefetch(din1_ptr);
                    din2_ptr += 7;
                    //prefetch(din2_ptr);
                    din3_ptr += 7;
                    //prefetch(din3_ptr);
                    din4_ptr += 7;
                    //prefetch(din4_ptr);

//                    if (relu) {
//                        vq12_oc0r0 = vmaxq_f32(vq12_oc0r0, vzero);
//                        vq14_oc1r0 = vmaxq_f32(vq14_oc1r0, vzero);
//                        vq13_oc0r1 = vmaxq_f32(vq13_oc0r1, vzero);
//                        vq15_oc1r1 = vmaxq_f32(vq15_oc1r1, vzero);
//                    }

                    vst1q_f32(doutc0r0, vq12_oc0r0);
                    vst1q_f32(doutc0r1, vq13_oc0r1);
                    vst1q_f32(doutc1r0, vq14_oc1r0);
                    vst1q_f32(doutc1r1, vq15_oc1r1);

                    doutc0r0 += 4;
                    doutc0r1 += 4;
                    doutc1r0 += 4;
                    doutc1r1 += 4;

                    //process mid
                    //! load input data, first iter
                    vq20q21_r0 = vld2q_f32(din0_ptr);
                    vq22q23_r1 = vld2q_f32(din1_ptr);
                    vq24q25_r2 = vld2q_f32(din2_ptr);
                    vq26q27_r3 = vld2q_f32(din3_ptr);
                    vq28q29_r4 = vld2q_f32(din4_ptr);
                    //! q28, q29, left for loading 5th row
                    for (int j = 0; j < cnt_col; ++j) {
                        // 4 rows
                        //! load output data
                        vq12_oc0r0 = vld1q_f32(doutc0r0);
                        vq13_oc0r1 = vld1q_f32(doutc0r1);
                        vq14_oc1r0 = vld1q_f32(doutc1r0);
                        vq15_oc1r1 = vld1q_f32(doutc1r1);
                        //! q16, q17, q18, q19 left for 4 out channels

                        //! mul weights[0]
                        //input r0, get output r0
                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq20q21_r0.val[0], wr00, 0);
                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq20q21_r0.val[0], wr10, 0);
                        //input r3, get output r1
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq26q27_r3.val[0], wr01, 0);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq26q27_r3.val[0], wr11, 0);

//                        //input r1, get output r0
//                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq22q23_r1.val[0], wr01, 0);
//                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq22q23_r1.val[0], wr11, 0);
//                        //input r4, get output r1
//                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq28q29_r4.val[0], wr02, 0);
//                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq28q29_r4.val[0], wr12, 0);
//
//                        //input r2, get output r1,r0
//                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq24q25_r2.val[0], wr02, 0);
//                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq24q25_r2.val[0], wr12, 0);
//                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq24q25_r2.val[0], wr00, 0);
//                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq24q25_r2.val[0], wr10, 0);
//
//                        //! do something here, pld
//                        //! mul weights[1]
//                        //input r0, get output r0
//                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq20q21_r0.val[1], wr00, 1);
//                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq20q21_r0.val[1], wr10, 1);
//                        //input r3, get output r1
//                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq26q27_r3.val[1], wr01, 1);
//                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq26q27_r3.val[1], wr11, 1);
//
//                        //input r1, get output r0
//                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq22q23_r1.val[1], wr01, 1);
//                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq22q23_r1.val[1], wr11, 1);
//                        //input r4, get output r1
//                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq28q29_r4.val[1], wr02, 1);
//                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq28q29_r4.val[1], wr12, 1);
//
//                        //input r2, get output r1,r0
//                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vq24q25_r2.val[1], wr02, 1);
//                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vq24q25_r2.val[1], wr12, 1);
//                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vq24q25_r2.val[1], wr00, 1);
//                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vq24q25_r2.val[1], wr10, 1);
//
//                        //! copy data
//                        vtmp0 = vaddq_f32(vzero, vq20q21_r0.val[0]);
//                        vtmp1 = vaddq_f32(vzero, vq22q23_r1.val[0]);
//                        vtmp2 = vaddq_f32(vzero, vq24q25_r2.val[0]);
//                        vtmp3 = vaddq_f32(vzero, vq26q27_r3.val[0]);
//                        vtmp4 = vaddq_f32(vzero, vq28q29_r4.val[0]);
//
                        din0_ptr += 8;
                        //prefetch(din0_ptr);
                        din1_ptr += 8;
                        //prefetch(din1_ptr);
                        din2_ptr += 8;
                        //prefetch(din2_ptr);
                        din3_ptr += 8;
                        //prefetch(din3_ptr);
                        din4_ptr += 8;
                        //prefetch(din4_ptr);

                        //! load input data, net iter
                        vq20q21_r0 = vld2q_f32(din0_ptr);
                        vq22q23_r1 = vld2q_f32(din1_ptr);
                        vq24q25_r2 = vld2q_f32(din2_ptr);
                        vq26q27_r3 = vld2q_f32(din3_ptr);
                        vq28q29_r4 = vld2q_f32(din4_ptr);

                        //! do something here
                        //! mul weights[2]
                        // r0, r1, r2, r3, r4 shift left
                        vtmp0 = vextq_f32(vtmp0, vq20q21_r0.val[0], 1);
                        vtmp1 = vextq_f32(vtmp1, vq22q23_r1.val[0], 1);
                        vtmp2 = vextq_f32(vtmp2, vq24q25_r2.val[0], 1);
                        vtmp3 = vextq_f32(vtmp3, vq26q27_r3.val[0], 1);
                        vtmp4 = vextq_f32(vtmp4, vq28q29_r4.val[0], 1);

                        //! mul weights[2]
                        //input r0, get output r0
                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vtmp0, wr00, 2);
                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vtmp0, wr10, 2);
                        //input r3, get output r1
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vtmp3, wr01, 2);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vtmp3, wr11, 2);

                        //input r1, get output r0
                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vtmp1, wr01, 2);
                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vtmp1, wr11, 2);
                        //input r4, get output r1
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vtmp4, wr02, 2);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vtmp4, wr12, 2);

                        //input r2, get output r1,r0
                        vq12_oc0r0 = vmlaq_laneq_f32(vq12_oc0r0, vtmp2, wr02, 2);
                        vq14_oc1r0 = vmlaq_laneq_f32(vq14_oc1r0, vtmp2, wr12, 2);
                        vq13_oc0r1 = vmlaq_laneq_f32(vq13_oc0r1, vtmp2, wr00, 2);
                        vq15_oc1r1 = vmlaq_laneq_f32(vq15_oc1r1, vtmp2, wr10, 2);

//                        if (relu) {
//                            vq12_oc0r0 = vmaxq_f32(vq12_oc0r0, vzero);
//                            vq14_oc1r0 = vmaxq_f32(vq14_oc1r0, vzero);
//                            vq13_oc0r1 = vmaxq_f32(vq13_oc0r1, vzero);
//                            vq15_oc1r1 = vmaxq_f32(vq15_oc1r1, vzero);
//                        }

                        vst1q_f32(doutc0r0, vq12_oc0r0);
                        vst1q_f32(doutc0r1, vq13_oc0r1);
                        vst1q_f32(doutc1r0, vq14_oc1r0);
                        vst1q_f32(doutc1r1, vq15_oc1r1);

                        doutc0r0 += 4;
                        doutc0r1 += 4;
                        doutc1r0 += 4;
                        doutc1r1 += 4;
                    }
#if 0
                    //process right
                    vq10_oc0r0 = vld1q_f32(doutc0r0);
                    vq11_oc0r1 = vld1q_f32(doutc0r1);
                    vq12_oc1r0 = vld1q_f32(doutc1r0);
                    vq13_oc1r1 = vld1q_f32(doutc1r1);

                    vq14_oc2r0 = vld1q_f32(doutc2r0);
                    vq15_oc2r1 = vld1q_f32(doutc2r1);
                    vq16_oc3r0 = vld1q_f32(doutc3r0);
                    vq17_oc3r1 = vld1q_f32(doutc3r1);

                    vq0_r00 = vld1q_f32(din0_ptr);
                    vq1_r01 = vld1q_f32(din0_ptr + 4);
                    vq2_r10 = vld1q_f32(din1_ptr);
                    vq3_r11 = vld1q_f32(din1_ptr + 4);
                    vq4_r20 = vld1q_f32(din2_ptr);
                    vq5_r21 = vld1q_f32(din2_ptr + 4);
                    vq6_r30 = vld1q_f32(din3_ptr);
                    vq7_r31 = vld1q_f32(din3_ptr + 4);

                    // bit select, right pad zero
                    vq0_r00 = vbslq_f32(vmask_rp1, vq0_r00, vzero);
                    vq1_r01 = vbslq_f32(vmask_rp2, vq1_r01, vzero);
                    vq2_r10 = vbslq_f32(vmask_rp1, vq2_r10, vzero);
                    vq3_r11 = vbslq_f32(vmask_rp2, vq3_r11, vzero);
                    vq4_r20 = vbslq_f32(vmask_rp1, vq4_r20, vzero);
                    vq5_r21 = vbslq_f32(vmask_rp2, vq5_r21, vzero);
                    vq6_r30 = vbslq_f32(vmask_rp1, vq6_r30, vzero);
                    vq7_r31 = vbslq_f32(vmask_rp2, vq7_r31, vzero);

                    //input r0
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq0_r00, wr00, 0);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq0_r00, wr10, 0);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vq0_r00, wr20, 0);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vq0_r00, wr30, 0);

                    //input r1
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq2_r10, wr01, 0);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vq2_r10, wr00, 0);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq2_r10, wr11, 0);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vq2_r10, wr10, 0);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vq2_r10, wr21, 0);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vq2_r10, wr20, 0);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vq2_r10, wr31, 0);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vq2_r10, wr30, 0);

                    //input r2
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq4_r20, wr02, 0);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vq4_r20, wr01, 0);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq4_r20, wr12, 0);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vq4_r20, wr11, 0);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vq4_r20, wr22, 0);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vq4_r20, wr21, 0);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vq4_r20, wr32, 0);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vq4_r20, wr31, 0);

                    //input r3
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vq6_r30, wr02, 0);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vq6_r30, wr12, 0);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vq6_r30, wr22, 0);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vq6_r30, wr32, 0);

                    // r0, r1, r2 shift left
                    vtmp1 = vextq_f32(vq0_r00, vq1_r01, 1);
                    vtmp2 = vextq_f32(vq2_r10, vq3_r11, 1);
                    vtmp3 = vextq_f32(vq4_r20, vq5_r21, 1);
                    vtmp4 = vextq_f32(vq6_r30, vq7_r31, 1);

                    // input r0
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp1, wr00, 1);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp1, wr10, 1);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp1, wr20, 1);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp1, wr30, 1);

                    // input r1
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp2, wr01, 1);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp2, wr00, 1);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp2, wr11, 1);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp2, wr10, 1);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp2, wr21, 1);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp2, wr20, 1);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp2, wr31, 1);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp2, wr30, 1);

                    //input r2
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp3, wr02, 1);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp3, wr01, 1);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp3, wr12, 1);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp3, wr11, 1);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp3, wr22, 1);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp3, wr21, 1);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp3, wr32, 1);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp3, wr31, 1);

                    //input r3
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp4, wr02, 1);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp4, wr12, 1);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp4, wr22, 1);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp4, wr32, 1);

                    // r0, r1, r2 shift right
                    vtmp1 = vextq_f32(vq0_r00, vq1_r01, 2);
                    vtmp2 = vextq_f32(vq2_r10, vq3_r11, 2);
                    vtmp3 = vextq_f32(vq4_r20, vq5_r21, 2);
                    vtmp4 = vextq_f32(vq6_r30, vq7_r31, 2);

                    // input r0
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp1, wr00, 2);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp1, wr10, 2);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp1, wr20, 2);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp1, wr30, 2);

                    // input r1
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp2, wr01, 2);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp2, wr00, 2);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp2, wr11, 2);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp2, wr10, 2);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp2, wr21, 2);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp2, wr20, 2);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp2, wr31, 2);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp2, wr30, 2);

                    //input r2
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp3, wr02, 2);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp3, wr01, 2);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp3, wr12, 2);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp3, wr11, 2);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp3, wr22, 2);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp3, wr21, 2);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp3, wr32, 2);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp3, wr31, 2);

                    // load output
                    vq0_r00 = vld1q_f32(doutc0r0);
                    vq1_r01 = vld1q_f32(doutc0r1);
                    vq2_r10 = vld1q_f32(doutc1r0);
                    vq3_r11 = vld1q_f32(doutc1r1);
                    vq4_r20 = vld1q_f32(doutc2r0);
                    vq5_r21 = vld1q_f32(doutc2r1);
                    vq6_r30 = vld1q_f32(doutc3r0);
                    vq7_r31 = vld1q_f32(doutc3r1);

                    //input r3
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp4, wr02, 2);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp4, wr12, 2);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp4, wr22, 2);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp4, wr32, 2);

                    din2_ptr += 4;
                    prefetch(din2_ptr);

                    if (relu) {
                        vq10_oc0r0 = vmaxq_f32(vq10_oc0r0, vzero);
                        vq11_oc0r1 = vmaxq_f32(vq11_oc0r1, vzero);
                        vq12_oc1r0 = vmaxq_f32(vq12_oc1r0, vzero);
                        vq13_oc1r1 = vmaxq_f32(vq13_oc1r1, vzero);
                        vq14_oc2r0 = vmaxq_f32(vq14_oc2r0, vzero);
                        vq15_oc2r1 = vmaxq_f32(vq15_oc2r1, vzero);
                        vq16_oc3r0 = vmaxq_f32(vq16_oc3r0, vzero);
                        vq17_oc3r1 = vmaxq_f32(vq17_oc3r1, vzero);
                    }

                    // bit select to get right result
                    vq10_oc0r0 = vbslq_f32(vmask_result, vq10_oc0r0, vq0_r00);
                    vq11_oc0r1 = vbslq_f32(vmask_result, vq11_oc0r1, vq1_r01);

                    vq12_oc1r0 = vbslq_f32(vmask_result, vq12_oc1r0, vq2_r10);
                    vq13_oc1r1 = vbslq_f32(vmask_result, vq13_oc1r1, vq3_r11);

                    vq14_oc2r0 = vbslq_f32(vmask_result, vq14_oc2r0, vq4_r20);
                    vq15_oc2r1 = vbslq_f32(vmask_result, vq15_oc2r1, vq5_r21);

                    vq16_oc3r0 = vbslq_f32(vmask_result, vq16_oc3r0, vq6_r30);
                    vq17_oc3r1 = vbslq_f32(vmask_result, vq17_oc3r1, vq7_r31);

                    vst1q_f32(doutc0r0, vq10_oc0r0);
                    vst1q_f32(doutc0r1, vq11_oc0r1);
                    vst1q_f32(doutc1r0, vq12_oc1r0);
                    vst1q_f32(doutc1r1, vq13_oc1r1);
                    vst1q_f32(doutc2r0, vq14_oc2r0);
                    vst1q_f32(doutc2r1, vq15_oc2r1);
                    vst1q_f32(doutc3r0, vq16_oc3r0);
                    vst1q_f32(doutc3r1, vq17_oc3r1);

                    //! after process, increase address
                    dr0 = dr2;
                    dr1 = dr3;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
#endif
                } //! end of processing mid rows
#if 0
                //! deal with bottom pad
                if (1) {

                    din0_ptr = dr0;
                    din1_ptr = dr1;
                    if (size_pad_bottom == 1) {
                        din2_ptr = dr2;
                    } else {
                        din2_ptr = ptr_zero;
                    }

                    doutc0r0 = dout_c0 + 2 * h * w_out;
                    doutc0r1 = doutc0r0 + w_out;
                    doutc1r0 = dout_c1 + 2 * h * w_out;
                    doutc1r1 = doutc1r0 + w_out;
                    doutc2r0 = dout_c2 + 2 * h * w_out;
                    doutc2r1 = doutc2r0 + w_out;
                    doutc3r0 = dout_c3 + 2 * h * w_out;
                    doutc3r1 = doutc3r0 + w_out;

                    //left pad
                    float32x4_t vq10_oc0r0 = vld1q_f32(doutc0r0);
                    float32x4_t vq11_oc0r1 = vld1q_f32(doutc0r1);
                    float32x4_t vq12_oc1r0 = vld1q_f32(doutc1r0);
                    float32x4_t vq13_oc1r1 = vld1q_f32(doutc1r1);

                    float32x4_t vq14_oc2r0 = vld1q_f32(doutc2r0);
                    float32x4_t vq15_oc2r1 = vld1q_f32(doutc2r1);
                    float32x4_t vq16_oc3r0 = vld1q_f32(doutc3r0);
                    float32x4_t vq17_oc3r1 = vld1q_f32(doutc3r1);

                    float32x4_t vq0_r00 = vld1q_f32(din0_ptr);
                    float32x4_t vq1_r01 = vld1q_f32(din0_ptr + 4);
                    float32x4_t vq2_r10 = vld1q_f32(din1_ptr);
                    float32x4_t vq3_r11 = vld1q_f32(din1_ptr + 4);
                    float32x4_t vq4_r20 = vld1q_f32(din2_ptr);
                    float32x4_t vq5_r21 = vld1q_f32(din2_ptr + 4);

                    //input r0
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq0_r00, wr00, 1);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq0_r00, wr10, 1);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vq0_r00, wr20, 1);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vq0_r00, wr30, 1);

                    din0_ptr += 3;
                    prefetch(din0_ptr);

                    //input r1
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq2_r10, wr01, 1);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vq2_r10, wr00, 1);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq2_r10, wr11, 1);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vq2_r10, wr10, 1);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vq2_r10, wr21, 1);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vq2_r10, wr20, 1);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vq2_r10, wr31, 1);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vq2_r10, wr30, 1);

                    //input r2
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq4_r20, wr02, 1);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vq4_r20, wr01, 1);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq4_r20, wr12, 1);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vq4_r20, wr11, 1);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vq4_r20, wr22, 1);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vq4_r20, wr21, 1);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vq4_r20, wr32, 1);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vq4_r20, wr31, 1);

                    // r0, r1, r2 shift left
                    float32x4_t vtmp1 = vextq_f32(vq0_r00, vq1_r01, 1);
                    float32x4_t vtmp2 = vextq_f32(vq2_r10, vq3_r11, 1);
                    float32x4_t vtmp3 = vextq_f32(vq4_r20, vq5_r21, 1);

                    din1_ptr += 3;
                    prefetch(din1_ptr);

                    // input r0
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp1, wr00, 2);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp1, wr10, 2);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp1, wr20, 2);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp1, wr30, 2);

                    // input r1
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp2, wr01, 2);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp2, wr00, 2);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp2, wr11, 2);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp2, wr10, 2);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp2, wr21, 2);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp2, wr20, 2);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp2, wr31, 2);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp2, wr30, 2);

                    //input r2
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp3, wr02, 2);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp3, wr01, 2);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp3, wr12, 2);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp3, wr11, 2);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp3, wr22, 2);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp3, wr21, 2);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp3, wr32, 2);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp3, wr31, 2);

                    // r0, r1, r2 shift right
                    vtmp1 = vextq_f32(vzero, vq0_r00, 3);
                    vtmp2 = vextq_f32(vzero, vq2_r10, 3);
                    vtmp3 = vextq_f32(vzero, vq4_r20, 3);

                    if (size_pad_bottom == 1) {
                        din2_ptr += 3;
                        prefetch(din2_ptr);
                    }

                    // input r0
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp1, wr00, 0);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp1, wr10, 0);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp1, wr20, 0);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp1, wr30, 0);

                    // input r1
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp2, wr01, 0);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp2, wr00, 0);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp2, wr11, 0);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp2, wr10, 0);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp2, wr21, 0);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp2, wr20, 0);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp2, wr31, 0);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp2, wr30, 0);

                    //input r2
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp3, wr02, 0);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp3, wr01, 0);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp3, wr12, 0);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp3, wr11, 0);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp3, wr22, 0);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp3, wr21, 0);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp3, wr32, 0);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp3, wr31, 0);

                    if (relu) {
                        vq10_oc0r0 = vmaxq_f32(vq10_oc0r0, vzero);
                        vq11_oc0r1 = vmaxq_f32(vq11_oc0r1, vzero);
                        vq12_oc1r0 = vmaxq_f32(vq12_oc1r0, vzero);
                        vq13_oc1r1 = vmaxq_f32(vq13_oc1r1, vzero);
                        vq14_oc2r0 = vmaxq_f32(vq14_oc2r0, vzero);
                        vq15_oc2r1 = vmaxq_f32(vq15_oc2r1, vzero);
                        vq16_oc3r0 = vmaxq_f32(vq16_oc3r0, vzero);
                        vq17_oc3r1 = vmaxq_f32(vq17_oc3r1, vzero);
                    }

                    vst1q_f32(doutc0r0, vq10_oc0r0);
                    vst1q_f32(doutc0r1, vq11_oc0r1);
                    vst1q_f32(doutc1r0, vq12_oc1r0);
                    vst1q_f32(doutc1r1, vq13_oc1r1);
                    vst1q_f32(doutc2r0, vq14_oc2r0);
                    vst1q_f32(doutc2r1, vq15_oc2r1);
                    vst1q_f32(doutc3r0, vq16_oc3r0);
                    vst1q_f32(doutc3r1, vq17_oc3r1);

                    doutc0r0 += 4;
                    doutc0r1 += 4;
                    doutc1r0 += 4;
                    doutc1r1 += 4;
                    doutc2r0 += 4;
                    doutc2r1 += 4;
                    doutc3r0 += 4;
                    doutc3r1 += 4;

                    //process mid
                    for (int j = 0; j < cnt_col; ++j) {

                        // 4 rows
                        vq10_oc0r0 = vld1q_f32(doutc0r0);
                        vq11_oc0r1 = vld1q_f32(doutc0r1);
                        vq12_oc1r0 = vld1q_f32(doutc1r0);
                        vq13_oc1r1 = vld1q_f32(doutc1r1);

                        vq14_oc2r0 = vld1q_f32(doutc2r0);
                        vq15_oc2r1 = vld1q_f32(doutc2r1);
                        vq16_oc3r0 = vld1q_f32(doutc3r0);
                        vq17_oc3r1 = vld1q_f32(doutc3r1);

                        vq0_r00 = vld1q_f32(din0_ptr);
                        vq1_r01 = vld1q_f32(din0_ptr + 4);
                        vq2_r10 = vld1q_f32(din1_ptr);
                        vq3_r11 = vld1q_f32(din1_ptr + 4);
                        vq4_r20 = vld1q_f32(din2_ptr);
                        vq5_r21 = vld1q_f32(din2_ptr + 4);

                        //input r0
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq0_r00, wr00, 0);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq0_r00, wr10, 0);
                        vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vq0_r00, wr20, 0);
                        vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vq0_r00, wr30, 0);

                        din0_ptr += 4;
                        prefetch(din0_ptr);

                        //input r1
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq2_r10, wr01, 0);
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vq2_r10, wr00, 0);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq2_r10, wr11, 0);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vq2_r10, wr10, 0);
                        vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vq2_r10, wr21, 0);
                        vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vq2_r10, wr20, 0);
                        vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vq2_r10, wr31, 0);
                        vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vq2_r10, wr30, 0);

                        //input r2
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq4_r20, wr02, 0);
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vq4_r20, wr01, 0);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq4_r20, wr12, 0);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vq4_r20, wr11, 0);
                        vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vq4_r20, wr22, 0);
                        vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vq4_r20, wr21, 0);
                        vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vq4_r20, wr32, 0);
                        vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vq4_r20, wr31, 0);

                        // r0, r1, r2 shift left
                        vtmp1 = vextq_f32(vq0_r00, vq1_r01, 1);
                        vtmp2 = vextq_f32(vq2_r10, vq3_r11, 1);
                        vtmp3 = vextq_f32(vq4_r20, vq5_r21, 1);

                        din1_ptr += 4;
                        prefetch(din1_ptr);

                        // input r0
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp1, wr00, 1);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp1, wr10, 1);
                        vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp1, wr20, 1);
                        vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp1, wr30, 1);

                        // input r1
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp2, wr01, 1);
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp2, wr00, 1);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp2, wr11, 1);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp2, wr10, 1);
                        vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp2, wr21, 1);
                        vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp2, wr20, 1);
                        vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp2, wr31, 1);
                        vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp2, wr30, 1);

                        //input r2
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp3, wr02, 1);
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp3, wr01, 1);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp3, wr12, 1);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp3, wr11, 1);
                        vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp3, wr22, 1);
                        vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp3, wr21, 1);
                        vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp3, wr32, 1);
                        vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp3, wr31, 1);

                        // r0, r1, r2 shift right
                        vtmp1 = vextq_f32(vq0_r00, vq1_r01, 2);
                        vtmp2 = vextq_f32(vq2_r10, vq3_r11, 2);
                        vtmp3 = vextq_f32(vq4_r20, vq5_r21, 2);

                        if (size_pad_bottom == 1) {
                            din2_ptr += 4;
                            prefetch(din2_ptr);
                        }

                        // input r0
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp1, wr00, 2);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp1, wr10, 2);
                        vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp1, wr20, 2);
                        vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp1, wr30, 2);

                        // input r1
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp2, wr01, 2);
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp2, wr00, 2);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp2, wr11, 2);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp2, wr10, 2);
                        vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp2, wr21, 2);
                        vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp2, wr20, 2);
                        vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp2, wr31, 2);
                        vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp2, wr30, 2);

                        //input r2
                        vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp3, wr02, 2);
                        vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp3, wr01, 2);
                        vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp3, wr12, 2);
                        vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp3, wr11, 2);
                        vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp3, wr22, 2);
                        vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp3, wr21, 2);
                        vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp3, wr32, 2);
                        vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp3, wr31, 2);

                        if (relu) {
                            vq10_oc0r0 = vmaxq_f32(vq10_oc0r0, vzero);
                            vq11_oc0r1 = vmaxq_f32(vq11_oc0r1, vzero);
                            vq12_oc1r0 = vmaxq_f32(vq12_oc1r0, vzero);
                            vq13_oc1r1 = vmaxq_f32(vq13_oc1r1, vzero);
                            vq14_oc2r0 = vmaxq_f32(vq14_oc2r0, vzero);
                            vq15_oc2r1 = vmaxq_f32(vq15_oc2r1, vzero);
                            vq16_oc3r0 = vmaxq_f32(vq16_oc3r0, vzero);
                            vq17_oc3r1 = vmaxq_f32(vq17_oc3r1, vzero);
                        }

                        vst1q_f32(doutc0r0, vq10_oc0r0);
                        vst1q_f32(doutc0r1, vq11_oc0r1);
                        vst1q_f32(doutc1r0, vq12_oc1r0);
                        vst1q_f32(doutc1r1, vq13_oc1r1);
                        vst1q_f32(doutc2r0, vq14_oc2r0);
                        vst1q_f32(doutc2r1, vq15_oc2r1);
                        vst1q_f32(doutc3r0, vq16_oc3r0);
                        vst1q_f32(doutc3r1, vq17_oc3r1);

                        doutc0r0 += 4;
                        doutc0r1 += 4;
                        doutc1r0 += 4;
                        doutc1r1 += 4;
                        doutc2r0 += 4;
                        doutc2r1 += 4;
                        doutc3r0 += 4;
                        doutc3r1 += 4;
                    }

                    //process right

                    vq10_oc0r0 = vld1q_f32(doutc0r0);
                    vq11_oc0r1 = vld1q_f32(doutc0r1);
                    vq12_oc1r0 = vld1q_f32(doutc1r0);
                    vq13_oc1r1 = vld1q_f32(doutc1r1);

                    vq14_oc2r0 = vld1q_f32(doutc2r0);
                    vq15_oc2r1 = vld1q_f32(doutc2r1);
                    vq16_oc3r0 = vld1q_f32(doutc3r0);
                    vq17_oc3r1 = vld1q_f32(doutc3r1);

                    vq0_r00 = vld1q_f32(din0_ptr);
                    vq1_r01 = vld1q_f32(din0_ptr + 4);
                    vq2_r10 = vld1q_f32(din1_ptr);
                    vq3_r11 = vld1q_f32(din1_ptr + 4);
                    vq4_r20 = vld1q_f32(din2_ptr);
                    vq5_r21 = vld1q_f32(din2_ptr + 4);

                    // bit select, right pad zero
                    vq0_r00 = vbslq_f32(vmask_rp1, vq0_r00, vzero);
                    vq1_r01 = vbslq_f32(vmask_rp2, vq1_r01, vzero);
                    vq2_r10 = vbslq_f32(vmask_rp1, vq2_r10, vzero);
                    vq3_r11 = vbslq_f32(vmask_rp2, vq3_r11, vzero);
                    vq4_r20 = vbslq_f32(vmask_rp1, vq4_r20, vzero);
                    vq5_r21 = vbslq_f32(vmask_rp2, vq5_r21, vzero);

                    //input r0
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq0_r00, wr00, 0);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq0_r00, wr10, 0);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vq0_r00, wr20, 0);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vq0_r00, wr30, 0);

                    //input r1
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq2_r10, wr01, 0);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vq2_r10, wr00, 0);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq2_r10, wr11, 0);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vq2_r10, wr10, 0);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vq2_r10, wr21, 0);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vq2_r10, wr20, 0);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vq2_r10, wr31, 0);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vq2_r10, wr30, 0);

                    //input r2
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vq4_r20, wr02, 0);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vq4_r20, wr01, 0);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vq4_r20, wr12, 0);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vq4_r20, wr11, 0);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vq4_r20, wr22, 0);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vq4_r20, wr21, 0);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vq4_r20, wr32, 0);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vq4_r20, wr31, 0);

                    // r0, r1, r2 shift left
                    vtmp1 = vextq_f32(vq0_r00, vq1_r01, 1);
                    vtmp2 = vextq_f32(vq2_r10, vq3_r11, 1);
                    vtmp3 = vextq_f32(vq4_r20, vq5_r21, 1);

                    // input r0
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp1, wr00, 1);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp1, wr10, 1);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp1, wr20, 1);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp1, wr30, 1);

                    // input r1
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp2, wr01, 1);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp2, wr00, 1);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp2, wr11, 1);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp2, wr10, 1);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp2, wr21, 1);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp2, wr20, 1);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp2, wr31, 1);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp2, wr30, 1);

                    //input r2
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp3, wr02, 1);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp3, wr01, 1);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp3, wr12, 1);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp3, wr11, 1);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp3, wr22, 1);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp3, wr21, 1);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp3, wr32, 1);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp3, wr31, 1);

                    // r0, r1, r2 shift right
                    vtmp1 = vextq_f32(vq0_r00, vq1_r01, 2);
                    vtmp2 = vextq_f32(vq2_r10, vq3_r11, 2);
                    vtmp3 = vextq_f32(vq4_r20, vq5_r21, 2);

                    // input r0
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp1, wr00, 2);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp1, wr10, 2);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp1, wr20, 2);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp1, wr30, 2);

                    // input r1
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp2, wr01, 2);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp2, wr00, 2);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp2, wr11, 2);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp2, wr10, 2);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp2, wr21, 2);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp2, wr20, 2);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp2, wr31, 2);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp2, wr30, 2);

                    //input r2
                    vq10_oc0r0 = vmlaq_laneq_f32(vq10_oc0r0, vtmp3, wr02, 2);
                    vq11_oc0r1 = vmlaq_laneq_f32(vq11_oc0r1, vtmp3, wr01, 2);
                    vq12_oc1r0 = vmlaq_laneq_f32(vq12_oc1r0, vtmp3, wr12, 2);
                    vq13_oc1r1 = vmlaq_laneq_f32(vq13_oc1r1, vtmp3, wr11, 2);
                    vq14_oc2r0 = vmlaq_laneq_f32(vq14_oc2r0, vtmp3, wr22, 2);
                    vq15_oc2r1 = vmlaq_laneq_f32(vq15_oc2r1, vtmp3, wr21, 2);
                    vq16_oc3r0 = vmlaq_laneq_f32(vq16_oc3r0, vtmp3, wr32, 2);
                    vq17_oc3r1 = vmlaq_laneq_f32(vq17_oc3r1, vtmp3, wr31, 2);

                    if (relu) {
                        vq10_oc0r0 = vmaxq_f32(vq10_oc0r0, vzero);
                        vq11_oc0r1 = vmaxq_f32(vq11_oc0r1, vzero);
                        vq12_oc1r0 = vmaxq_f32(vq12_oc1r0, vzero);
                        vq13_oc1r1 = vmaxq_f32(vq13_oc1r1, vzero);
                        vq14_oc2r0 = vmaxq_f32(vq14_oc2r0, vzero);
                        vq15_oc2r1 = vmaxq_f32(vq15_oc2r1, vzero);
                        vq16_oc3r0 = vmaxq_f32(vq16_oc3r0, vzero);
                        vq17_oc3r1 = vmaxq_f32(vq17_oc3r1, vzero);
                    }

                    // load output
                    vq0_r00 = vld1q_f32(doutc0r0);
                    vq1_r01 = vld1q_f32(doutc0r1);

                    vq2_r10 = vld1q_f32(doutc1r0);
                    vq3_r11 = vld1q_f32(doutc1r1);

                    vq4_r20 = vld1q_f32(doutc2r0);
                    vq5_r21 = vld1q_f32(doutc2r1);

                    vtmp1 = vld1q_f32(doutc3r0);
                    vtmp2 = vld1q_f32(doutc3r1);

                    // bit select to get right result
                    vq10_oc0r0 = vbslq_f32(vmask_result, vq10_oc0r0, vq0_r00);
                    vq11_oc0r1 = vbslq_f32(vmask_result, vq11_oc0r1, vq1_r01);

                    vq12_oc1r0 = vbslq_f32(vmask_result, vq12_oc1r0, vq2_r10);
                    vq13_oc1r1 = vbslq_f32(vmask_result, vq13_oc1r1, vq3_r11);

                    vq14_oc2r0 = vbslq_f32(vmask_result, vq14_oc2r0, vq4_r20);
                    vq15_oc2r1 = vbslq_f32(vmask_result, vq15_oc2r1, vq5_r21);

                    vq16_oc3r0 = vbslq_f32(vmask_result, vq16_oc3r0, vtmp1);
                    vq17_oc3r1 = vbslq_f32(vmask_result, vq17_oc3r1, vtmp2);

                    vst1q_f32(doutc0r0, vq10_oc0r0);
                    vst1q_f32(doutc0r1, vq11_oc0r1);
                    vst1q_f32(doutc1r0, vq12_oc1r0);
                    vst1q_f32(doutc1r1, vq13_oc1r1);
                    vst1q_f32(doutc2r0, vq14_oc2r0);
                    vst1q_f32(doutc2r1, vq15_oc2r1);
                    vst1q_f32(doutc3r0, vq16_oc3r0);
                    vst1q_f32(doutc3r1, vq17_oc3r1);

                } // end of processing bottom pad
#endif
            } // end of processing channels
        } //end of processing output channel
        if (cremain > 0) {
            for (int c = 0; c < cremain; ++c) {

                int cidx = ch_out - cremain + c;
                float* dout_c = dout_batch + cidx * size_out_channel;

                if (flag_bias) {
                    fill_bias(dout_c, &bias[cidx], 1, size_out_channel);
                } else {
                    fill_bias(dout_c, zero, 1, size_out_channel);
                }

                const float* wc0 = weights + cidx * w_stride;

                for (int i = 0; i < ch_in; ++i) {

                    bool relu = (i == ch_in - 1) && flag_relu;

                    const float* din_channel = din_batch + i * size_in_channel;
                    for (int h = 0; h < h_out; ++h) {

                        int hstart = h - pad_h;
                        int hend = hstart + 3;
                        hstart = std::max(hstart, 0);
                        hend = std::min(hend, h_in);

                        int khstart = hend < kernel_h? kernel_h - hend : 0;

                        float* dout_row = dout_c + h * w_out;

                        for (int w = 0; w < w_out; ++w) {
                            int wstart = w - pad_w;
                            int wend = wstart + 3;
                            wstart = std::max(wstart, 0);
                            wend = std::min(wend, w_in);
                            int kwstart = wend < kernel_w? kernel_w - wend : 0;

                            for (int kh = hstart; kh < hend; ++kh) {
                                for (int kw = wstart; kw < wend; ++kw) {
                                    dout_row[w] += din_channel[kh * w_in + kw] * \
                                        wc0[(khstart + kh - hstart) * 3 + kwstart + kw - wstart];
                                }
                            }
                            if (relu) {
                                dout_row[w] = dout_row[w] > 0.f? dout_row[w] : 0.f;
                            }
                        }
                    }
                    wc0 += 9;
                }
            }
        } // end of remain out channel

    } // end of processing batchs
}

#else

void conv_3x3s2_direct(const float* din, float* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const float* weights, const float* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Sgemm& gemmer, void* work_space) {
    //! 3x3s2 convolution, implemented by direct algorithm
    //! pad is done implicit

    //! each core / loop gets 3 input rows in 1 input channel and produces 1 row in 2 output channels
    // q0 = w00, q1 = w01, q2 = w02
    // q3 = w10, q4 = w11, q5 = w12
    // q6 = r00/r10/r20, q7 = r01/r11/r21
    // q8 = r30/r40, q9 = r31,r41
    // q10 = outc0r0, q11 = outc0r1
    // q12 = outc1r0, q13 = outc1r1

    const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    //! for 4x6 convolution window
    const unsigned int right_pad_idx[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    const unsigned int right_save_idx[4] = {1, 2, 3, 4};

    unsigned int right_pad_save_mask[12];
    //! flags[0] stands for "do_right_pad"
    //! flags[1] stands for "relu"
    int flags[2];

    int w_in = win;
    int h_in = hin;
    int ch_in = chin;

    int w_out = wout;
    int h_out = hout;
    int ch_out = chout;

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = ch_in * 9;

    int w_loop = (w_out + 3) >> 2;
    int cnt_col = w_loop - 2;
    int cnt_row = (hout + 1) / 2;

    int cremain = ch_out - ((ch_out >> 1) << 1);
    int bt_remain = hout - (cnt_row - 1) * 2;
    int pad_bot_size = ((hin + 1) / 2) * 2 - hin;//could be 0 or 1

    int do_right_pad = 1;
    unsigned int size_pad_right = w_loop * 4/*neon simd length*/ * 2 /*stride = 2*/ - w_in;//could be 0~7
    unsigned int right_pad_save = 4 -(w_loop * 4 - w_out);
    int right_pad_sub = (w_loop * 4 - w_out) * sizeof(float);
    if (size_pad_right == 0 && right_pad_save == 4) {
        cnt_col = w_loop - 1;
        do_right_pad = 0;
    } else {
        // right pad params
        uint32x4x2_t vrpidx = vld2q_u32(right_pad_idx);
        uint32x4_t vmask_rp1 = vcgeq_u32(vrpidx.val[0], vdupq_n_u32(size_pad_right));
        uint32x4_t vmask_rp2 = vcgeq_u32(vrpidx.val[1], vdupq_n_u32(size_pad_right));
        vst1q_u32(right_pad_save_mask, vmask_rp1);
        vst1q_u32(right_pad_save_mask + 4, vmask_rp2);

        uint32x4_t vrsidx = vld1q_u32(right_save_idx);
        uint32x4_t vmask_save = vcleq_u32(vrsidx, vdupq_n_u32(right_pad_save));
        vst1q_u32(right_pad_save_mask + 8, vmask_save);
    }

//    for (int j = 0; j < 12; ++j) {
//        printf("%d\n", right_pad_save_mask[j]);
//    }

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_out - 1; c += 2) {

            float* dout_c0 = dout_batch + c * size_out_channel;
            float* dout_c1 = dout_c0 + size_out_channel;

            if (flag_bias) {
                fill_bias(dout_c0, &bias[c], 1, size_out_channel);
                fill_bias(dout_c1, &bias[c + 1], 1, size_out_channel);
            } else {
                fill_bias(dout_c0, zero, 1, size_out_channel);
                fill_bias(dout_c1, zero, 1, size_out_channel);
            }

            const float* wc0 = weights + c * w_stride;
            const float* wc1 = wc0 + w_stride;

            for (int i = 0; i < ch_in; ++i) {

                int relu = 0;
                if ((i == ch_in - 1) && flag_relu) {
                    relu = 1;
                }

                const float *din_channel = din_batch + i * size_in_channel;

                const float* wcin0 = wc0 + i * 9;
                const float* wcin1 = wc1 + i * 9;
                float32x4_t wr00 = vld1q_f32(wcin0); //q0
                float32x4_t wr01 = vld1q_f32(wcin0 + 3); //q1
                float32x4_t wr02 = vld1q_f32(wcin0 + 6); //q2

                float32x4_t wr10 = vld1q_f32(wcin1); //q3
                float32x4_t wr11 = vld1q_f32(wcin1 + 3);//q4
                float32x4_t wr12 = vld1q_f32(wcin1 + 6);//q5

                float *doutc0r0 = dout_c0;
                float *doutc0r1 = dout_c0 + wout;
                float *doutc1r0 = dout_c1;
                float *doutc1r1 = dout_c1 + wout;

                const float *dr0 = din_channel;
                const float *dr1 = dr0 + w_in;
                const float *dr2 = dr1 + w_in;
                const float *dr3 = dr2 + w_in;
                const float *dr4 = dr3 + w_in;

                const float *din0_ptr = dr0;
                const float *din1_ptr = dr1;
                const float *din2_ptr = dr2;
                const float *din3_ptr = dr3;
                const float *din4_ptr = dr4;

                float* ptr_zero = const_cast<float*>(zero);
                float32x4_t vzero = vdupq_n_f32(0.f);

                //! deal with top pad
                int h = 0;
                {
                    //! process
                    if (1) {
                        int cnt = cnt_col;
                        asm volatile(
                        //! process left pad
                        "vmov.u32 q15, #0                       @ dump zero\n"
                                "pld [%[doutc0r0]]                      @ preload data\n" //outc0r0
                                "pld [%[doutc0r1]]                      @ preload data\n"//outc0r1
                                "pld [%[doutc1r0]]                      @ preload data\n"//outc1r0
                                "pld [%[doutc1r1]]                      @ preload data\n"//outc1r1
                                "pld [%[din0_ptr]]                      @ preload data\n"//inr0
                                "pld [%[din1_ptr]]                      @ preload data\n"//inr1
                                "pld [%[din2_ptr]]                      @ preload data\n"//inr2
                                "pld [%[din3_ptr]]                      @ preload data\n"//inr3

                                //! row0/2
                                "vld2.32  {d12-d15}, [%[din0_ptr]]!     @ load input r00, r01\n" //interleave load, q6->rx0, q7->rx1
                                "vld2.32  {d16-d19}, [%[din2_ptr]]!     @ load input r20, r21\n" //interleave load, q8->rx2, q9->rx2

                                "vld1.32  {d20-d21}, [%[doutc0r0]]      @ load dout0r0\n" //q10->outc0r0
                                "vld1.32  {d22-d23}, [%[doutc0r1]]      @ load dout0r1\n" //q11->outc0r1
                                "vld1.32  {d24-d25}, [%[doutc1r0]]      @ load dout1r0\n" //q12->outc1r0
                                "vld1.32  {d26-d27}, [%[doutc1r1]]      @ load dout1r1\n" //q13->outc1r1

                                "vmla.f32 q10, q6,   %e[wr01][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr11][1]        @ mul weight1, 01, out1r0\n"
                                "vmla.f32 q11, q8,   %e[wr01][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q8,   %e[wr11][1]        @ mul weight1, 01, out1r1\n"

                                "vmla.f32 q10, q7,   %f[wr01][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q7,   %f[wr11][0]        @ mul weight1, 02, out1r0\n"
                                "vmla.f32 q11, q9,   %f[wr01][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q9,   %f[wr11][0]        @ mul weight1, 02, out1r1\n"

                                // shift right 1
                                "vext.32  q14, q15, q7,  #3             @ shift right r0\n"
                                // load row1
                                "vld2.32  {d12-d15}, [%[din1_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q10, q14,  %e[wr01][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q14,  %e[wr11][0]        @ mul weight1, 00, out1r0\n"

                                // shift right 1
                                "vext.32  q14, q15, q9,  #3             @ shift right r2\n"
                                // load row3
                                "vld2.32  {d16-d19}, [%[din3_ptr]]!     @ load input r30, r31\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q11, q14,  %e[wr01][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q14,  %e[wr11][0]        @ mul weight1, 00, out1r1\n"

                                "sub %[din0_ptr], #4                    @ r0 address -4, overlay 1 float\n"
                                "sub %[din2_ptr], #4                    @ r2 address -4, overlay 1 float\n"

                                //! row1/3
                                "vmla.f32 q10, q6,   %e[wr02][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr12][1]        @ mul weight1, 01, out0r0\n"
                                "vmla.f32 q11, q6,   %e[wr00][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q6,   %e[wr10][1]        @ mul weight1, 01, out1r1\n"

                                "vmla.f32 q10, q7,   %f[wr02][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q7,   %f[wr12][0]        @ mul weight1, 02, out0r0\n"
                                "vmla.f32 q11, q7,   %f[wr00][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q7,   %f[wr10][0]        @ mul weight1, 02, out1r1\n"

                                // shift right 1
                                "vext.32  q6, q15, q7,  #3              @ shift right r1\n"
                                "vmla.f32 q10, q6,   %e[wr02][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr12][0]        @ mul weight1, 00, out0r0\n"
                                "vmla.f32 q11, q6,   %e[wr00][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q6,   %e[wr10][0]        @ mul weight1, 00, out1r1\n"

                                "pld [%[din0_ptr]]                      @ preload data\n"//inr0
                                "pld [%[din1_ptr]]                      @ preload data\n"//inr1

                                "vmla.f32 q11, q8,   %e[wr02][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q8,   %e[wr12][1]        @ mul weight1, 01, out1r1\n"

                                "pld [%[din2_ptr]]                      @ preload data\n"//inr2
                                "pld [%[din3_ptr]]                      @ preload data\n"//inr3

                                "sub %[din1_ptr], #4                    @ r1 address -4, overlay 1 float\n"

                                "vmla.f32 q11, q9,   %f[wr02][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q9,   %f[wr12][0]        @ mul weight1, 02, out1r1\n"

                                "sub %[din3_ptr], #4                    @ r3 address -4, overlay 1 float\n"

                                // shift right 1
                                "vext.32  q8, q15, q9,  #3              @ shift right r3\n"
                                "vmla.f32 q11, q8,   %e[wr02][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q8,   %e[wr12][0]        @ mul weight1, 00, out1r1\n"

                                "cmp %[relu], #1                        @ check whether has relu\n"
                                "blt    1f                              @ jump to top left store without relu\n"
                                "vmax.f32   q10,  q10, q15              @ relu\n"
                                "vmax.f32   q11,  q11, q15              @ relu\n"
                                "vmax.f32   q12,  q12, q15              @ relu\n"
                                "vmax.f32   q13,  q13, q15              @ relu\n"

                                "1:                                     @ store top left result\n"
                                // stroe tl result
                                "vst1.32  {d20-d21}, [%[doutc0r0]]!     @ store result, add pointer\n"
                                "vst1.32  {d22-d23}, [%[doutc0r1]]!     @ store result, add pointer\n"

                                "vst1.32  {d24-d25}, [%[doutc1r0]]!     @ store result, add pointer\n"
                                "vst1.32  {d26-d27}, [%[doutc1r1]]!     @ store result, add pointer\n"

                                "pld [%[doutc0r0]]                      @ preload data\n"
                                "pld [%[doutc0r1]]                      @ preload data\n"
                                "pld [%[doutc1r0]]                      @ preload data\n"
                                "pld [%[doutc1r1]]                      @ preload data\n"

                                //! process mid cols
                                "cmp %[cnt], #1                         @ check whether has mid cols\n"
                                "blt  2f                                @ jump to main loop start point\n"
                                "start_top_mid:                         @ main loop in top row\n"
                                //! row0/2
                                "vld2.32  {d12-d15}, [%[din0_ptr]]!     @ load input r00, r01\n" //interleave load, q6->rx0, q7->rx1
                                "vld2.32  {d16-d19}, [%[din2_ptr]]!     @ load input r20, r21\n" //interleave load, q8->rx2, q9->rx2

                                "vld1.32  {d20-d21}, [%[doutc0r0]]      @ load dout0r0\n" //q10->outc0r0
                                "vld1.32  {d22-d23}, [%[doutc0r1]]      @ load dout0r1\n" //q11->outc0r1
                                "vld1.32  {d24-d25}, [%[doutc1r0]]      @ load dout1r0\n" //q12->outc1r0
                                "vld1.32  {d26-d27}, [%[doutc1r1]]      @ load dout1r1\n" //q13->outc1r1

                                "vmla.f32 q10, q7,   %e[wr01][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q7,   %e[wr11][1]        @ mul weight1, 01, out1r0\n"
                                "vmla.f32 q11, q9,   %e[wr01][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q9,   %e[wr11][1]        @ mul weight1, 01, out1r1\n"

                                "vld1.32  {d14},    [%[din0_ptr]]       @ load the 8th element, r00\n"
                                "vld1.32  {d18},    [%[din2_ptr]]       @ load the 8th element, r20\n"

                                "vmla.f32 q10, q6,   %e[wr01][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr11][0]        @ mul weight1, 00, out1r0\n"
                                "vmla.f32 q11, q8,   %e[wr01][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q8,   %e[wr11][0]        @ mul weight1, 00, out1r1\n"

                                // shift left 1
                                "vext.32  q14, q6,  q7,  #1             @ shift left r0\n"
                                // load row1
                                "vld2.32  {d12-d15}, [%[din1_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q10, q14,  %f[wr01][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q14,  %f[wr11][0]        @ mul weight1, 02, out1r0\n"

                                // shift right 1
                                "vext.32  q14, q8,  q9,  #1             @ shift right r2\n"
                                // load row3
                                "vld2.32  {d16-d19}, [%[din3_ptr]]!     @ load input r30, r31\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q11, q14,  %f[wr01][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q14,  %f[wr11][0]        @ mul weight1, 02, out1r1\n"

                                //! row1
                                "vmla.f32 q10, q7,   %e[wr02][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q7,   %e[wr12][1]        @ mul weight1, 01, out0r0\n"
                                "vmla.f32 q11, q7,   %e[wr00][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q7,   %e[wr10][1]        @ mul weight1, 01, out1r1\n"

                                "vld1.32  {d14},    [%[din1_ptr]]       @ load the 8th element, r10\n"

                                "vmla.f32 q10, q6,   %e[wr02][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr12][0]        @ mul weight1, 00, out0r0\n"
                                "vmla.f32 q11, q6,   %e[wr00][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q6,   %e[wr10][0]        @ mul weight1, 00, out1r1\n"

                                // shift left 1
                                "vext.32  q14, q6, q7,  #1              @ shift left r1\n"
                                "vmla.f32 q10, q14,  %f[wr02][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q14,  %f[wr12][0]        @ mul weight1, 02, out0r0\n"
                                "vmla.f32 q11, q14,  %f[wr00][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q14,  %f[wr10][0]        @ mul weight1, 02, out1r1\n"

                                "pld [%[din0_ptr]]                      @ preload data\n"//inr0
                                "pld [%[din1_ptr]]                      @ preload data\n"//inr1

                                //! row3
                                "vmla.f32 q11, q9,   %e[wr02][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q9,   %e[wr12][1]        @ mul weight1, 01, out1r1\n"

                                "vld1.32  {d18},    [%[din3_ptr]]       @ load the 8th element, r30\n"

                                "vmla.f32 q11, q8,   %e[wr02][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q8,   %e[wr12][0]        @ mul weight1, 00, out1r1\n"

                                "pld [%[din2_ptr]]                      @ preload data\n"//inr2
                                "pld [%[din3_ptr]]                      @ preload data\n"//inr3

                                // shift left 1
                                "vext.32  q14, q8, q9,  #1              @ shift left r3\n"
                                "vmla.f32 q11, q14,  %f[wr02][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q14,  %f[wr12][0]        @ mul weight1, 02, out1r1\n"

                                "cmp %[relu], #1                        @ check whether has relu\n"
                                "blt    3f                              @ jump to top left store without relu\n"
                                "vmax.f32   q10,  q10, q15              @ relu\n"
                                "vmax.f32   q11,  q11, q15              @ relu\n"
                                "vmax.f32   q12,  q12, q15              @ relu\n"
                                "vmax.f32   q13,  q13, q15              @ relu\n"

                                "3:                                     @ store top mid result\n"
                                // store tm result
                                "vst1.32  {d20-d21}, [%[doutc0r0]]!     @ store result, add pointer\n"
                                "vst1.32  {d22-d23}, [%[doutc0r1]]!     @ store result, add pointer\n"

                                "vst1.32  {d24-d25}, [%[doutc1r0]]!     @ store result, add pointer\n"
                                "vst1.32  {d26-d27}, [%[doutc1r1]]!     @ store result, add pointer\n"

                                "subs %[cnt], #1                        @ loop count minus 1\n"
                                "bne    start_top_mid                   @ jump to main loop start point\n"

                                //! process right pad
                                "2:                                     @ right pad entry\n"
                                // check do_right_pad, if 0, jump to end
                                "cmp %[do_right_pad], #1                @ check whether has mid cols\n"
                                "blt  5f                                @ jump to main loop start point\n"

                                // load pad mask
                                "vld1.32  {d16-d19}, [%[right_pad_save_mask]]! @ load pad index\n" //q8, q9 //load 8
                                // load row0
                                "vld2.32  {d12-d15}, [%[din0_ptr]]!     @ load input r00, r01\n" //interleave load, q6->rx0, q7->rx1

                                // load output
                                "vld1.32  {d20-d21}, [%[doutc0r0]]      @ load dout0r0\n" //q10->outc0r0
                                "vld1.32  {d22-d23}, [%[doutc0r1]]      @ load dout0r1\n" //q11->outc0r1
                                "vld1.32  {d24-d25}, [%[doutc1r0]]      @ load dout1r0\n" //q12->outc1r0
                                "vld1.32  {d26-d27}, [%[doutc1r1]]      @ load dout1r1\n" //q13->outc1r1

                                // row0,  deal with right pad
                                "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                                "vbif q7, q15, q9                       @ bit select, deal with right pad\n"

                                "vmla.f32 q10, q7,   %e[wr01][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q7,   %e[wr11][1]        @ mul weight1, 01, out1r0\n"

                                // shift left 1
                                "vext.32  q14, q6,  q15, #1             @ shift left r0\n"

                                "vmla.f32 q10, q6,   %e[wr01][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr11][0]        @ mul weight1, 00, out1r0\n"

                                // load row1
                                "vld2.32  {d12-d15}, [%[din1_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q10, q14,  %f[wr01][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q14,  %f[wr11][0]        @ mul weight1, 02, out1r0\n"

                                // row1, deal with right pad
                                "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                                "vbif q7, q15, q9                       @ bit select, deal with right pad\n"
                                //! row1
                                "vmla.f32 q10, q7,   %e[wr02][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q7,   %e[wr12][1]        @ mul weight1, 01, out0r0\n"
                                "vmla.f32 q11, q7,   %e[wr00][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q7,   %e[wr10][1]        @ mul weight1, 01, out1r1\n"

                                // shift left 1
                                "vext.32  q14, q6, q15, #1              @ shift left r1\n"

                                "vmla.f32 q10, q6,   %e[wr02][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr12][0]        @ mul weight1, 00, out0r0\n"
                                "vmla.f32 q11, q6,   %e[wr00][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q6,   %e[wr10][0]        @ mul weight1, 00, out1r1\n"

                                // load row2
                                "vld2.32  {d12-d15}, [%[din2_ptr]]!     @ load input r20, r21\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q10, q14,  %f[wr02][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q14,  %f[wr12][0]        @ mul weight1, 02, out0r0\n"
                                "vmla.f32 q11, q14,  %f[wr00][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q14,  %f[wr10][0]        @ mul weight1, 02, out1r1\n"

                                // row2, deal with right pad
                                "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                                "vbif q7, q15, q9                       @ bit select, deal with right pad\n"

                                //! row2
                                "vmla.f32 q11, q7,   %e[wr01][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q7,   %e[wr11][1]        @ mul weight1, 01, out1r1\n"

                                // shift left 1
                                "vext.32  q14, q6,  q15, #1             @ shift left r2\n"

                                "vmla.f32 q11, q6,   %e[wr01][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q6,   %e[wr11][0]        @ mul weight1, 00, out1r1\n"

                                // load row3
                                "vld2.32  {d12-d15}, [%[din3_ptr]]!     @ load input r20, r21\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q11, q14,  %f[wr01][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q14,  %f[wr11][0]        @ mul weight1, 02, out1r1\n"

                                // row3, deal with right pad
                                "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                                "vbif q7, q15, q9                       @ bit select, deal with right pad\n"

                                //! row3
                                "vmla.f32 q11, q7,   %e[wr02][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q7,   %e[wr12][1]        @ mul weight1, 01, out1r1\n"

                                // shift left 1
                                "vext.32  q14, q6,  q15, #1             @ shift left r2\n"

                                "vmla.f32 q11, q6,   %e[wr02][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q6,   %e[wr12][0]        @ mul weight1, 00, out1r1\n"

                                "vld1.32  {d12-d13}, [%[doutc0r0]]      @ load dout0r0\n" //q6->outc0r0
                                "vld1.32  {d14-d15}, [%[doutc0r1]]      @ load dout0r1\n" //q7->outc0r1
                                "vld1.32  {d16-d17}, [%[doutc1r0]]      @ load dout1r0\n" //q8->outc1r0
                                "vld1.32  {d18-d19}, [%[doutc1r1]]      @ load dout1r1\n" //q9->outc1r1

                                "vmla.f32 q11, q14,  %f[wr02][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q14,  %f[wr12][0]        @ mul weight1, 02, out1r1\n"

                                "cmp %[relu], #1                        @ check whether has relu\n"
                                "blt    4f                              @ jump to top left store without relu\n"
                                "vmax.f32   q10,  q10, q15              @ relu\n"
                                "vmax.f32   q11,  q11, q15              @ relu\n"
                                "vmax.f32   q12,  q12, q15              @ relu\n"
                                "vmax.f32   q13,  q13, q15              @ relu\n"

                                "4:                                     @ store top mid result\n"
                                // store tr result
                                "vld1.32  {d28-d29}, [%[right_pad_save_mask]]  @ load save mask\n" //q14->save mask

                                "vbif q10, q6, q14                      @ bit select\n"
                                "vbif q11, q7, q14                      @ bit select\n"
                                "vbif q12, q8, q14                      @ bit select\n"
                                "vbif q13, q9, q14                      @ bit select\n"

                                "vst1.32  {d20-d21}, [%[doutc0r0]]!     @ store result, add pointer\n"
                                "vst1.32  {d22-d23}, [%[doutc0r1]]!     @ store result, add pointer\n"
                                "vst1.32  {d24-d25}, [%[doutc1r0]]!     @ store result, add pointer\n"
                                "vst1.32  {d26-d27}, [%[doutc1r1]]!     @ store result, add pointer\n"

                                "sub %[doutc1r0], %[doutc1r0], %[right_pad_sub] @ sub \n"
                                "sub %[doutc1r1], %[doutc1r1], %[right_pad_sub] @ sub \n"
                                "sub %[doutc0r0], %[doutc0r0], %[right_pad_sub] @ sub \n"
                                "sub %[doutc0r1], %[doutc0r1], %[right_pad_sub] @ sub \n"

                                "5:                                     @ top row ending\n"

                        :[doutc0r0] "+r"(doutc0r0), [doutc0r1] "+r"(doutc0r1), \
                            [doutc1r0] "+r" (doutc1r0), [doutc1r1] "+r" (doutc1r1),\
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), [cnt] "+r"(cnt)
                        :[wr00] "w"(wr00), [wr01] "w"(wr01), [wr02] "w"(wr02), \
                            [wr10] "w"(wr10), [wr11] "w"(wr11), [wr12] "w"(wr12), \
                            [right_pad_save_mask] "r" (right_pad_save_mask), \
                            [right_pad_sub] "r" (right_pad_sub), \
                            [do_right_pad] "r" (do_right_pad), [relu] "r" (relu)
                        :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                        );
                    }
                    //! after process, increase pointer
                    doutc0r0 += w_out;
                    doutc0r1 = doutc0r0 + w_out;
                    doutc1r0 += w_out;
                    doutc1r1 = doutc1r0 + w_out;

                    dr0 = dr3;
                    dr1 = dr4;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                    dr4 = dr3 + w_in;

                } //! end of process top row

                //! process mid row
                int row_loop_end = cnt_row - 1;
                if (bt_remain == 2 && pad_bot_size == 0) {
                    row_loop_end = cnt_row;
                }
                for (h = 1; h < row_loop_end; h++) {
                    din0_ptr = dr0;
                    din1_ptr = dr1;
                    din2_ptr = dr2;
                    din3_ptr = dr3;
                    din4_ptr = dr4;
                    {
                        int cnt = cnt_col;
                        asm volatile(
                        //! process left pad
                        "vmov.u32 q15, #0                       @ dump zero\n"
                                "pld [%[din0_ptr]]                      @ preload data\n"//inr0
                                "pld [%[din1_ptr]]                      @ preload data\n"//inr1
                                "pld [%[din2_ptr]]                      @ preload data\n"//inr2
                                "pld [%[din3_ptr]]                      @ preload data\n"//inr3
                                "pld [%[din4_ptr]]                      @ preload data\n"//inr4
                                "pld [%[doutc0r0]]                      @ preload data\n" //outc0r0
                                "pld [%[doutc0r1]]                      @ preload data\n"//outc0r1
                                "pld [%[doutc1r0]]                      @ preload data\n"//outc1r0
                                "pld [%[doutc1r1]]                      @ preload data\n"//outc1r1

                                //! row0/3
                                "vld2.32  {d12-d15}, [%[din0_ptr]]!     @ load input r00, r01\n" //interleave load, q6->rx0, q7->rx1
                                "vld1.32  {d20-d21}, [%[doutc0r0]]      @ load dout0r0\n" //q10->outc0r0
                                "vld1.32  {d24-d25}, [%[doutc1r0]]      @ load dout1r0\n" //q12->outc1r0

                                "vld2.32  {d16-d19}, [%[din3_ptr]]!     @ load input r20, r21\n" //interleave load, q8->rx2, q9->rx2
                                "vld1.32  {d22-d23}, [%[doutc0r1]]      @ load dout0r1\n" //q11->outc0r1
                                "vld1.32  {d26-d27}, [%[doutc1r1]]      @ load dout1r1\n" //q13->outc1r1

                                "vmla.f32 q10, q6,   %e[wr00][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr10][1]        @ mul weight1, 01, out1r0\n"
                                "vmla.f32 q11, q8,   %e[wr01][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q8,   %e[wr11][1]        @ mul weight1, 01, out1r1\n"

                                "vmla.f32 q10, q7,   %f[wr00][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q7,   %f[wr10][0]        @ mul weight1, 02, out1r0\n"
                                "vmla.f32 q11, q9,   %f[wr01][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q9,   %f[wr11][0]        @ mul weight1, 02, out1r1\n"

                                // shift right 1, r0
                                "vext.32  q14, q15, q7,  #3             @ shift right r0\n"
                                // load row1
                                "vld2.32  {d12-d15}, [%[din1_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1
                                "sub %[din0_ptr], #4                    @ r0 address -4, overlay 1 float\n"

                                "vmla.f32 q10, q14,  %e[wr00][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q14,  %e[wr10][0]        @ mul weight1, 00, out1r0\n"

                                // shift right 1, r3
                                "vext.32  q14, q15, q9,  #3             @ shift right r2\n"
                                // load row4
                                "vld2.32  {d16-d19}, [%[din4_ptr]]!     @ load input r30, r31\n" //interleave load, q6->rx0, q7->rx1
                                "sub %[din3_ptr], #4                    @ r2 address -4, overlay 1 float\n"

                                "vmla.f32 q11, q14,  %e[wr01][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q14,  %e[wr11][0]        @ mul weight1, 00, out1r1\n"

                                //! row1/4
                                "vmla.f32 q10, q6,   %e[wr01][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr11][1]        @ mul weight1, 01, out1r0\n"
                                "vmla.f32 q11, q8,   %e[wr02][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q8,   %e[wr12][1]        @ mul weight1, 01, out1r1\n"

                                "vmla.f32 q10, q7,   %f[wr01][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q7,   %f[wr11][0]        @ mul weight1, 02, out1r0\n"
                                "vmla.f32 q11, q9,   %f[wr02][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q9,   %f[wr12][0]        @ mul weight1, 02, out1r1\n"

                                // shift right 1, r1
                                "vext.32  q14, q15, q7,  #3             @ shift right r1\n"
                                // load row2
                                "vld2.32  {d12-d15}, [%[din2_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1
                                "sub %[din1_ptr], #4                    @ r0 address -4, overlay 1 float\n"

                                "vmla.f32 q10, q14,  %e[wr01][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q14,  %e[wr11][0]        @ mul weight1, 00, out1r0\n"

                                // shift right 1, r4
                                "vext.32  q14, q15, q9,  #3             @ shift right r4\n"
                                "sub %[din4_ptr], #4                    @ r2 address -4, overlay 1 float\n"

                                "vmla.f32 q11, q14,  %e[wr02][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q14,  %e[wr12][0]        @ mul weight1, 00, out1r1\n"

                                //! row2
                                "vmla.f32 q10, q6,   %e[wr02][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr12][1]        @ mul weight1, 01, out1r0\n"
                                "vmla.f32 q11, q6,   %e[wr00][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q6,   %e[wr10][1]        @ mul weight1, 01, out1r1\n"

                                "sub %[din2_ptr], #4                    @ r1 address -4, overlay 1 float\n"

                                "vmla.f32 q10, q7,   %f[wr02][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q7,   %f[wr12][0]        @ mul weight1, 02, out1r0\n"
                                "vmla.f32 q11, q7,   %f[wr00][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q7,   %f[wr10][0]        @ mul weight1, 02, out1r1\n"

                                // shift right 1, r2
                                "vext.32  q14, q15, q7,  #3             @ shift right r2\n"
                                "vmla.f32 q10, q14,  %e[wr02][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q14,  %e[wr12][0]        @ mul weight1, 00, out1r0\n"
                                "vmla.f32 q11, q14,  %e[wr00][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q14,  %e[wr10][0]        @ mul weight1, 00, out1r1\n"


                                "pld [%[din0_ptr]]                      @ preload data\n"//inr0
                                "pld [%[din1_ptr]]                      @ preload data\n"//inr1
                                "pld [%[din2_ptr]]                      @ preload data\n"//inr2
                                "pld [%[din3_ptr]]                      @ preload data\n"//inr3
                                "pld [%[din4_ptr]]                      @ preload data\n"//inr3

                                "cmp %[relu], #1                        @ check whether has relu\n"
                                "blt    1f                              @ jump to top left store without relu\n"
                                "vmax.f32   q10,  q10, q15              @ relu\n"
                                "vmax.f32   q11,  q11, q15              @ relu\n"
                                "vmax.f32   q12,  q12, q15              @ relu\n"
                                "vmax.f32   q13,  q13, q15              @ relu\n"

                                "1:                                     @ store top left result\n"
                                // stroe tl result
                                "vst1.32  {d20-d21}, [%[doutc0r0]]!     @ store result, add pointer\n"
                                "vst1.32  {d22-d23}, [%[doutc0r1]]!     @ store result, add pointer\n"

                                "vst1.32  {d24-d25}, [%[doutc1r0]]!     @ store result, add pointer\n"
                                "vst1.32  {d26-d27}, [%[doutc1r1]]!     @ store result, add pointer\n"

                                "pld [%[doutc0r0]]                      @ preload data\n"
                                "pld [%[doutc0r1]]                      @ preload data\n"
                                "pld [%[doutc1r0]]                      @ preload data\n"
                                "pld [%[doutc1r1]]                      @ preload data\n"

                                //! process mid cols
                                "cmp %[cnt], #1                         @ check whether has mid cols\n"
                                "blt  2f                                @ jump to main loop start point\n"
                                "start_mid_mid:                         @ main loop in mid rows\n"

                                //! row0/3
                                "vld2.32  {d12-d15}, [%[din0_ptr]]!     @ load input r00, r01\n" //interleave load, q6->rx0, q7->rx1
                                "vld1.32  {d20-d21}, [%[doutc0r0]]      @ load dout0r0\n" //q10->outc0r0
                                "vld1.32  {d24-d25}, [%[doutc1r0]]      @ load dout1r0\n" //q12->outc1r0

                                "vld2.32  {d16-d19}, [%[din3_ptr]]!     @ load input r20, r21\n" //interleave load, q8->rx2, q9->rx2
                                "vld1.32  {d22-d23}, [%[doutc0r1]]      @ load dout0r1\n" //q11->outc0r1
                                "vld1.32  {d26-d27}, [%[doutc1r1]]      @ load dout1r1\n" //q13->outc1r1

                                "vmla.f32 q10, q7,   %e[wr00][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q7,   %e[wr10][1]        @ mul weight1, 01, out1r0\n"
                                "vmla.f32 q11, q9,   %e[wr01][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q9,   %e[wr11][1]        @ mul weight1, 01, out1r1\n"

                                "vld1.32  {d14},    [%[din0_ptr]]       @ load the 8th element, r00\n"
                                "vld1.32  {d18},    [%[din3_ptr]]       @ load the 8th element, r30\n"

                                "vmla.f32 q10, q6,   %e[wr00][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr10][0]        @ mul weight1, 00, out1r0\n"
                                "vmla.f32 q11, q8,   %e[wr01][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q8,   %e[wr11][0]        @ mul weight1, 00, out1r1\n"

                                // shift left 1, r0
                                "vext.32  q14, q6,  q7,  #1             @ shift left r0\n"
                                // load row1
                                "vld2.32  {d12-d15}, [%[din1_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q10, q14,  %f[wr00][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q14,  %f[wr10][0]        @ mul weight1, 02, out1r0\n"

                                // shift left 1, r3
                                "vext.32  q14, q8,  q9,  #1             @ shift left r3\n"
                                // load row4
                                "vld2.32  {d16-d19}, [%[din4_ptr]]!     @ load input r30, r31\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q11, q14,  %f[wr01][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q14,  %f[wr11][0]        @ mul weight1, 02, out1r1\n"

                                "pld [%[din0_ptr]]                      @ preload data\n"//inr0
                                "pld [%[din3_ptr]]                      @ preload data\n"//inr3

                                //! row1/4
                                "vmla.f32 q10, q7,   %e[wr01][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q7,   %e[wr11][1]        @ mul weight1, 01, out1r0\n"
                                "vmla.f32 q11, q9,   %e[wr02][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q9,   %e[wr12][1]        @ mul weight1, 01, out1r1\n"

                                "vld1.32  {d14},    [%[din1_ptr]]       @ load the 8th element, r10\n"
                                "vld1.32  {d18},    [%[din4_ptr]]       @ load the 8th element, r40\n"

                                "vmla.f32 q10, q6,   %e[wr01][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr11][0]        @ mul weight1, 00, out1r0\n"
                                "vmla.f32 q11, q8,   %e[wr02][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q8,   %e[wr12][0]        @ mul weight1, 00, out1r1\n"

                                // shift left 1, r1
                                "vext.32  q14, q6,  q7,  #1             @ shift left r0\n"
                                // load row2
                                "vld2.32  {d12-d15}, [%[din2_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q10, q14,  %f[wr01][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q14,  %f[wr11][0]        @ mul weight1, 02, out1r0\n"

                                // shift left 1, r4
                                "vext.32  q14, q8,  q9,  #1             @ shift left r3\n"
                                "vmla.f32 q11, q14,  %f[wr02][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q14,  %f[wr12][0]        @ mul weight1, 02, out1r1\n"

                                "pld [%[din1_ptr]]                      @ preload data\n"//inr0
                                "pld [%[din4_ptr]]                      @ preload data\n"//inr3

                                //! row2
                                "vmla.f32 q10, q7,   %e[wr02][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q7,   %e[wr12][1]        @ mul weight1, 01, out1r0\n"
                                "vmla.f32 q11, q7,   %e[wr00][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q7,   %e[wr10][1]        @ mul weight1, 01, out1r1\n"

                                "vld1.32  {d14},    [%[din2_ptr]]       @ load the 8th element, r00\n"

                                "vmla.f32 q10, q6,   %e[wr02][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr12][0]        @ mul weight1, 00, out1r0\n"
                                "vmla.f32 q11, q6,   %e[wr00][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q6,   %e[wr10][0]        @ mul weight1, 00, out1r1\n"

                                // shift left 1, r12
                                "vext.32  q14, q6,  q7,  #1             @ shift left r2\n"
                                "vmla.f32 q10, q14,  %f[wr02][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q14,  %f[wr12][0]        @ mul weight1, 02, out1r0\n"
                                "vmla.f32 q11, q14,  %f[wr00][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q14,  %f[wr10][0]        @ mul weight1, 02, out1r1\n"

                                "pld [%[din2_ptr]]                      @ preload data\n"//inr2

                                "cmp %[relu], #1                        @ check whether has relu\n"
                                "blt    3f                              @ jump to top left store without relu\n"
                                "vmax.f32   q10,  q10, q15              @ relu\n"
                                "vmax.f32   q11,  q11, q15              @ relu\n"
                                "vmax.f32   q12,  q12, q15              @ relu\n"
                                "vmax.f32   q13,  q13, q15              @ relu\n"

                                "3:                                     @ store top mid result\n"
                                // store tm result
                                "vst1.32  {d20-d21}, [%[doutc0r0]]!     @ store result, add pointer\n"
                                "vst1.32  {d22-d23}, [%[doutc0r1]]!     @ store result, add pointer\n"

                                "vst1.32  {d24-d25}, [%[doutc1r0]]!     @ store result, add pointer\n"
                                "vst1.32  {d26-d27}, [%[doutc1r1]]!     @ store result, add pointer\n"

                                "subs %[cnt], #1                        @ loop count minus 1\n"
                                "bne    start_mid_mid                   @ jump to main loop start point\n"

                                //! process right pad
                                "2:                                     @ right pad entry\n"
                                // check do_right_pad, if 0, jump to end
                                "cmp %[do_right_pad], #1                @ check whether has mid cols\n"
                                "blt  5f                                @ jump to main loop start point\n"

                                // load pad mask
                                "vld1.32  {d16-d19}, [%[right_pad_save_mask]]! @ load pad index\n" //q8, q9 //load 8 uint32
                                // load row0
                                "vld2.32  {d12-d15}, [%[din0_ptr]]!     @ load input r00, r01\n" //interleave load, q6->rx0, q7->rx1

                                // load output
                                "vld1.32  {d20-d21}, [%[doutc0r0]]      @ load dout0r0\n" //q10->outc0r0
                                "vld1.32  {d22-d23}, [%[doutc0r1]]      @ load dout0r1\n" //q11->outc0r1
                                "vld1.32  {d24-d25}, [%[doutc1r0]]      @ load dout1r0\n" //q12->outc1r0
                                "vld1.32  {d26-d27}, [%[doutc1r1]]      @ load dout1r1\n" //q13->outc1r1

                                // row0,  deal with right pad
                                "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                                "vbif q7, q15, q9                       @ bit select, deal with right pad\n"
                                // shift left 1
                                "vext.32  q14, q6,  q15, #1             @ shift left r0\n"

                                "vmla.f32 q10, q7,   %e[wr00][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q7,   %e[wr10][1]        @ mul weight1, 01, out1r0\n"

                                "vmla.f32 q10, q6,   %e[wr00][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr10][0]        @ mul weight1, 00, out1r0\n"

                                // load row1
                                "vld2.32  {d12-d15}, [%[din1_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q10, q14,  %f[wr00][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q14,  %f[wr10][0]        @ mul weight1, 02, out1r0\n"

                                // row1,  deal with right pad
                                "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                                "vbif q7, q15, q9                       @ bit select, deal with right pad\n"
                                // shift left 1, r1
                                "vext.32  q14, q6,  q15, #1             @ shift left r0\n"

                                "vmla.f32 q10, q7,   %e[wr01][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q7,   %e[wr11][1]        @ mul weight1, 01, out1r0\n"

                                "vmla.f32 q10, q6,   %e[wr01][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr11][0]        @ mul weight1, 00, out1r0\n"

                                // load row2
                                "vld2.32  {d12-d15}, [%[din2_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q10, q14,  %f[wr01][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q14,  %f[wr11][0]        @ mul weight1, 02, out1r0\n"

                                // row2, deal with right pad
                                "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                                "vbif q7, q15, q9                       @ bit select, deal with right pad\n"
                                // shift left 1
                                "vext.32  q14, q6, q15, #1              @ shift left r1\n"
                                //! row2
                                "vmla.f32 q10, q7,   %e[wr02][1]        @ mul weight0, 01, out0r0\n"
                                "vmla.f32 q12, q7,   %e[wr12][1]        @ mul weight1, 01, out0r0\n"
                                "vmla.f32 q11, q7,   %e[wr00][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q7,   %e[wr10][1]        @ mul weight1, 01, out1r1\n"

                                "vmla.f32 q10, q6,   %e[wr02][0]        @ mul weight0, 00, out0r0\n"
                                "vmla.f32 q12, q6,   %e[wr12][0]        @ mul weight1, 00, out0r0\n"
                                "vmla.f32 q11, q6,   %e[wr00][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q6,   %e[wr10][0]        @ mul weight1, 00, out1r1\n"

                                // load row3
                                "vld2.32  {d12-d15}, [%[din3_ptr]]!     @ load input r20, r21\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q10, q14,  %f[wr02][0]        @ mul weight0, 02, out0r0\n"
                                "vmla.f32 q12, q14,  %f[wr12][0]        @ mul weight1, 02, out0r0\n"
                                "vmla.f32 q11, q14,  %f[wr00][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q14,  %f[wr10][0]        @ mul weight1, 02, out1r1\n"

                                // row3, deal with right pad
                                "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                                "vbif q7, q15, q9                       @ bit select, deal with right pad\n"
                                // shift left 1
                                "vext.32  q14, q6,  q15, #1             @ shift left r2\n"

                                //! row3
                                "vmla.f32 q11, q7,   %e[wr01][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q7,   %e[wr11][1]        @ mul weight1, 01, out1r1\n"

                                "vmla.f32 q11, q6,   %e[wr01][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q6,   %e[wr11][0]        @ mul weight1, 00, out1r1\n"

                                // load row4
                                "vld2.32  {d12-d15}, [%[din4_ptr]]!     @ load input r40, r41\n" //interleave load, q6->rx0, q7->rx1

                                "vmla.f32 q11, q14,  %f[wr01][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q14,  %f[wr11][0]        @ mul weight1, 02, out1r1\n"

                                // row4, deal with right pad
                                "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                                "vbif q7, q15, q9                       @ bit select, deal with right pad\n"
                                // shift left 1, r4
                                "vext.32  q14, q6,  q15, #1             @ shift left r2\n"

                                //! row4
                                "vmla.f32 q11, q7,   %e[wr02][1]        @ mul weight0, 01, out0r1\n"
                                "vmla.f32 q13, q7,   %e[wr12][1]        @ mul weight1, 01, out1r1\n"

                                "vmla.f32 q11, q6,   %e[wr02][0]        @ mul weight0, 00, out0r1\n"
                                "vmla.f32 q13, q6,   %e[wr12][0]        @ mul weight1, 00, out1r1\n"

                                "vld1.32  {d12-d13}, [%[doutc0r0]]      @ load dout0r0\n" //q6->outc0r0
                                "vld1.32  {d14-d15}, [%[doutc0r1]]      @ load dout0r1\n" //q7->outc0r1
                                "vld1.32  {d16-d17}, [%[doutc1r0]]      @ load dout1r0\n" //q8->outc1r0
                                "vld1.32  {d18-d19}, [%[doutc1r1]]      @ load dout1r1\n" //q9->outc1r1

                                "vmla.f32 q11, q14,  %f[wr02][0]        @ mul weight0, 02, out0r1\n"
                                "vmla.f32 q13, q14,  %f[wr12][0]        @ mul weight1, 02, out1r1\n"

                                "cmp %[relu], #1                        @ check whether has relu\n"
                                "blt    4f                              @ jump to top left store without relu\n"
                                "vmax.f32   q10,  q10, q15              @ relu\n"
                                "vmax.f32   q11,  q11, q15              @ relu\n"
                                "vmax.f32   q12,  q12, q15              @ relu\n"
                                "vmax.f32   q13,  q13, q15              @ relu\n"

                                "4:                                     @ store top mid result\n"
                                // store tr result
                                "vld1.32  {d28-d29}, [%[right_pad_save_mask]]  @ load save mask\n" //q14->save mask

                                "vbif q10, q6, q14                      @ bit select\n"
                                "vbif q11, q7, q14                      @ bit select\n"
                                "vbif q12, q8, q14                      @ bit select\n"
                                "vbif q13, q9, q14                      @ bit select\n"

                                "vst1.32  {d20-d21}, [%[doutc0r0]]!     @ store result, add pointer\n"
                                "vst1.32  {d22-d23}, [%[doutc0r1]]!     @ store result, add pointer\n"
                                "vst1.32  {d24-d25}, [%[doutc1r0]]!     @ store result, add pointer\n"
                                "vst1.32  {d26-d27}, [%[doutc1r1]]!     @ store result, add pointer\n"

                                "sub %[doutc1r0], %[doutc1r0], %[right_pad_sub] @ sub \n"
                                "sub %[doutc1r1], %[doutc1r1], %[right_pad_sub] @ sub \n"
                                "sub %[doutc0r0], %[doutc0r0], %[right_pad_sub] @ sub \n"
                                "sub %[doutc0r1], %[doutc0r1], %[right_pad_sub] @ sub \n"

                                "5:                                     @ mid rows ending\n"

                        :[doutc0r0] "+r"(doutc0r0), [doutc0r1] "+r"(doutc0r1), \
                            [doutc1r0] "+r" (doutc1r0), [doutc1r1] "+r" (doutc1r1),\
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                            [din4_ptr] "+r"(din4_ptr), [cnt] "+r"(cnt)
                        :[wr00] "w"(wr00), [wr01] "w"(wr01), [wr02] "w"(wr02), \
                            [wr10] "w"(wr10), [wr11] "w"(wr11), [wr12] "w"(wr12), \
                            [right_pad_sub] "r" (right_pad_sub), \
                            [right_pad_save_mask] "r" (right_pad_save_mask), \
                            [do_right_pad] "r" (do_right_pad), [relu] "r" (relu)
                        :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                        );
                    }
                    doutc0r0 += w_out;
                    doutc0r1 = doutc0r0 + w_out;
                    doutc1r0 += w_out;
                    doutc1r1 = doutc1r0 + w_out;

                    dr0 = dr4;
                    dr1 = dr0 + win;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                    dr4 = dr3 + w_in;
                } //! end of processing mid rows


                //! deal with bottom pad
                if (bt_remain == 2 && pad_bot_size > 0) {
                    din0_ptr = dr0;
                    din1_ptr = dr1;
                    din2_ptr = dr2;
                    din3_ptr = dr3;
                    int cnt = cnt_col;
                    asm volatile(
                    //! process left pad
                    "vmov.u32 q15, #0                       @ dump zero\n"
                            "pld [%[din0_ptr]]                      @ preload data\n"//inr0
                            "pld [%[din1_ptr]]                      @ preload data\n"//inr1
                            "pld [%[din2_ptr]]                      @ preload data\n"//inr2
                            "pld [%[din3_ptr]]                      @ preload data\n"//inr3
                            "pld [%[doutc0r0]]                      @ preload data\n" //outc0r0
                            "pld [%[doutc0r1]]                      @ preload data\n"//outc0r1
                            "pld [%[doutc1r0]]                      @ preload data\n"//outc1r0
                            "pld [%[doutc1r1]]                      @ preload data\n"//outc1r1

                            //! row0/3
                            "vld2.32  {d12-d15}, [%[din0_ptr]]!     @ load input r00, r01\n" //interleave load, q6->rx0, q7->rx1
                            "vld1.32  {d20-d21}, [%[doutc0r0]]      @ load dout0r0\n" //q10->outc0r0
                            "vld1.32  {d24-d25}, [%[doutc1r0]]      @ load dout1r0\n" //q12->outc1r0

                            "vld2.32  {d16-d19}, [%[din3_ptr]]!     @ load input r20, r21\n" //interleave load, q8->rx2, q9->rx2
                            "vld1.32  {d22-d23}, [%[doutc0r1]]      @ load dout0r1\n" //q11->outc0r1
                            "vld1.32  {d26-d27}, [%[doutc1r1]]      @ load dout1r1\n" //q13->outc1r1

                            "vmla.f32 q10, q6,   %e[wr00][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr10][1]        @ mul weight1, 01, out1r0\n"
                            "vmla.f32 q11, q8,   %e[wr01][1]        @ mul weight0, 01, out0r1\n"
                            "vmla.f32 q13, q8,   %e[wr11][1]        @ mul weight1, 01, out1r1\n"

                            "vmla.f32 q10, q7,   %f[wr00][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q7,   %f[wr10][0]        @ mul weight1, 02, out1r0\n"
                            "vmla.f32 q11, q9,   %f[wr01][0]        @ mul weight0, 02, out0r1\n"
                            "vmla.f32 q13, q9,   %f[wr11][0]        @ mul weight1, 02, out1r1\n"

                            // shift right 1, r0
                            "vext.32  q14, q15, q7,  #3             @ shift right r0\n"
                            // load row1
                            "vld2.32  {d12-d15}, [%[din1_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1
                            "sub %[din0_ptr], #4                    @ r0 address -4, overlay 1 float\n"

                            "vmla.f32 q10, q14,  %e[wr00][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q14,  %e[wr10][0]        @ mul weight1, 00, out1r0\n"

                            // shift right 1, r3
                            "vext.32  q14, q15, q9,  #3             @ shift right r2\n"
                            "sub %[din3_ptr], #4                    @ r2 address -4, overlay 1 float\n"

                            "vmla.f32 q11, q14,  %e[wr01][0]        @ mul weight0, 00, out0r1\n"
                            "vmla.f32 q13, q14,  %e[wr11][0]        @ mul weight1, 00, out1r1\n"

                            //! row1
                            "vmla.f32 q10, q6,   %e[wr01][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr11][1]        @ mul weight1, 01, out1r0\n"

                            "vmla.f32 q10, q7,   %f[wr01][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q7,   %f[wr11][0]        @ mul weight1, 02, out1r0\n"

                            // shift right 1, r1
                            "vext.32  q14, q15, q7,  #3             @ shift right r1\n"
                            // load row2
                            "vld2.32  {d12-d15}, [%[din2_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1
                            "sub %[din1_ptr], #4                    @ r0 address -4, overlay 1 float\n"

                            "vmla.f32 q10, q14,  %e[wr01][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q14,  %e[wr11][0]        @ mul weight1, 00, out1r0\n"

                            //! row2
                            "vmla.f32 q10, q6,   %e[wr02][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr12][1]        @ mul weight1, 01, out1r0\n"
                            "vmla.f32 q11, q6,   %e[wr00][1]        @ mul weight0, 01, out0r1\n"
                            "vmla.f32 q13, q6,   %e[wr10][1]        @ mul weight1, 01, out1r1\n"

                            "sub %[din2_ptr], #4                    @ r1 address -4, overlay 1 float\n"

                            "vmla.f32 q10, q7,   %f[wr02][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q7,   %f[wr12][0]        @ mul weight1, 02, out1r0\n"
                            "vmla.f32 q11, q7,   %f[wr00][0]        @ mul weight0, 02, out0r1\n"
                            "vmla.f32 q13, q7,   %f[wr10][0]        @ mul weight1, 02, out1r1\n"

                            // shift right 1, r2
                            "vext.32  q14, q15, q7,  #3             @ shift right r2\n"
                            "vmla.f32 q10, q14,  %e[wr02][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q14,  %e[wr12][0]        @ mul weight1, 00, out1r0\n"
                            "vmla.f32 q11, q14,  %e[wr00][0]        @ mul weight0, 00, out0r1\n"
                            "vmla.f32 q13, q14,  %e[wr10][0]        @ mul weight1, 00, out1r1\n"


                            "pld [%[din0_ptr]]                      @ preload data\n"//inr0
                            "pld [%[din1_ptr]]                      @ preload data\n"//inr1
                            "pld [%[din2_ptr]]                      @ preload data\n"//inr2
                            "pld [%[din3_ptr]]                      @ preload data\n"//inr3

                            "cmp %[relu], #1                        @ check whether has relu\n"
                            "blt    1f                              @ jump to top left store without relu\n"
                            "vmax.f32   q10,  q10, q15              @ relu\n"
                            "vmax.f32   q11,  q11, q15              @ relu\n"
                            "vmax.f32   q12,  q12, q15              @ relu\n"
                            "vmax.f32   q13,  q13, q15              @ relu\n"

                            "1:                                     @ store top left result\n"
                            // stroe tl result
                            "vst1.32  {d20-d21}, [%[doutc0r0]]!     @ store result, add pointer\n"
                            "vst1.32  {d22-d23}, [%[doutc0r1]]!     @ store result, add pointer\n"
                            "vst1.32  {d24-d25}, [%[doutc1r0]]!     @ store result, add pointer\n"
                            "vst1.32  {d26-d27}, [%[doutc1r1]]!     @ store result, add pointer\n"

                            "pld [%[doutc0r0]]                      @ preload data\n"
                            "pld [%[doutc0r1]]                      @ preload data\n"
                            "pld [%[doutc1r0]]                      @ preload data\n"
                            "pld [%[doutc1r1]]                      @ preload data\n"

                            //! process mid cols
                            "cmp %[cnt], #1                         @ check whether has mid cols\n"
                            "blt  2f                                @ jump to main loop start point\n"
                            "start_bot_mid:                         @ main loop in mid rows\n"

                            //! row0/3
                            "vld2.32  {d12-d15}, [%[din0_ptr]]!     @ load input r00, r01\n" //interleave load, q6->rx0, q7->rx1
                            "vld1.32  {d20-d21}, [%[doutc0r0]]      @ load dout0r0\n" //q10->outc0r0
                            "vld1.32  {d24-d25}, [%[doutc1r0]]      @ load dout1r0\n" //q12->outc1r0

                            "vld2.32  {d16-d19}, [%[din3_ptr]]!     @ load input r20, r21\n" //interleave load, q8->rx2, q9->rx2
                            "vld1.32  {d22-d23}, [%[doutc0r1]]      @ load dout0r1\n" //q11->outc0r1
                            "vld1.32  {d26-d27}, [%[doutc1r1]]      @ load dout1r1\n" //q13->outc1r1

                            "vmla.f32 q10, q7,   %e[wr00][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q7,   %e[wr10][1]        @ mul weight1, 01, out1r0\n"
                            "vmla.f32 q11, q9,   %e[wr01][1]        @ mul weight0, 01, out0r1\n"
                            "vmla.f32 q13, q9,   %e[wr11][1]        @ mul weight1, 01, out1r1\n"

                            "vld1.32  {d14},    [%[din0_ptr]]       @ load the 8th element, r00\n"
                            "vld1.32  {d18},    [%[din3_ptr]]       @ load the 8th element, r30\n"

                            "vmla.f32 q10, q6,   %e[wr00][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr10][0]        @ mul weight1, 00, out1r0\n"
                            "vmla.f32 q11, q8,   %e[wr01][0]        @ mul weight0, 00, out0r1\n"
                            "vmla.f32 q13, q8,   %e[wr11][0]        @ mul weight1, 00, out1r1\n"

                            // shift left 1, r0
                            "vext.32  q14, q6,  q7,  #1             @ shift left r0\n"
                            // load row1
                            "vld2.32  {d12-d15}, [%[din1_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                            "vmla.f32 q10, q14,  %f[wr00][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q14,  %f[wr10][0]        @ mul weight1, 02, out1r0\n"

                            // shift left 1, r3
                            "vext.32  q14, q8,  q9,  #1             @ shift left r3\n"
                            "vmla.f32 q11, q14,  %f[wr01][0]        @ mul weight0, 02, out0r1\n"
                            "vmla.f32 q13, q14,  %f[wr11][0]        @ mul weight1, 02, out1r1\n"

                            "pld [%[din0_ptr]]                      @ preload data\n"//inr0
                            "pld [%[din3_ptr]]                      @ preload data\n"//inr3

                            //! row1
                            "vmla.f32 q10, q7,   %e[wr01][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q7,   %e[wr11][1]        @ mul weight1, 01, out1r0\n"

                            "vld1.32  {d14},    [%[din1_ptr]]       @ load the 8th element, r10\n"

                            "vmla.f32 q10, q6,   %e[wr01][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr11][0]        @ mul weight1, 00, out1r0\n"

                            // shift left 1, r1
                            "vext.32  q14, q6,  q7,  #1             @ shift left r0\n"
                            // load row2
                            "vld2.32  {d12-d15}, [%[din2_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                            "vmla.f32 q10, q14,  %f[wr01][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q14,  %f[wr11][0]        @ mul weight1, 02, out1r0\n"

                            "pld [%[din1_ptr]]                      @ preload data\n"//inr1

                            //! row2
                            "vmla.f32 q10, q7,   %e[wr02][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q7,   %e[wr12][1]        @ mul weight1, 01, out1r0\n"
                            "vmla.f32 q11, q7,   %e[wr00][1]        @ mul weight0, 01, out0r1\n"
                            "vmla.f32 q13, q7,   %e[wr10][1]        @ mul weight1, 01, out1r1\n"

                            "vld1.32  {d14},    [%[din2_ptr]]       @ load the 8th element, r00\n"

                            "vmla.f32 q10, q6,   %e[wr02][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr12][0]        @ mul weight1, 00, out1r0\n"
                            "vmla.f32 q11, q6,   %e[wr00][0]        @ mul weight0, 00, out0r1\n"
                            "vmla.f32 q13, q6,   %e[wr10][0]        @ mul weight1, 00, out1r1\n"

                            // shift left 1, r12
                            "vext.32  q14, q6,  q7,  #1             @ shift left r2\n"
                            "vmla.f32 q10, q14,  %f[wr02][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q14,  %f[wr12][0]        @ mul weight1, 02, out1r0\n"
                            "vmla.f32 q11, q14,  %f[wr00][0]        @ mul weight0, 02, out0r1\n"
                            "vmla.f32 q13, q14,  %f[wr10][0]        @ mul weight1, 02, out1r1\n"

                            "pld [%[din2_ptr]]                      @ preload data\n"//inr2

                            "cmp %[relu], #1                        @ check whether has relu\n"
                            "blt    3f                              @ jump to top left store without relu\n"
                            "vmax.f32   q10,  q10, q15              @ relu\n"
                            "vmax.f32   q11,  q11, q15              @ relu\n"
                            "vmax.f32   q12,  q12, q15              @ relu\n"
                            "vmax.f32   q13,  q13, q15              @ relu\n"

                            "3:                                     @ store top mid result\n"
                            // store tm result
                            "vst1.32  {d20-d21}, [%[doutc0r0]]!     @ store result, add pointer\n"
                            "vst1.32  {d22-d23}, [%[doutc0r1]]!     @ store result, add pointer\n"

                            "vst1.32  {d24-d25}, [%[doutc1r0]]!     @ store result, add pointer\n"
                            "vst1.32  {d26-d27}, [%[doutc1r1]]!     @ store result, add pointer\n"

                            "subs %[cnt], #1                        @ loop count minus 1\n"
                            "bne    start_bot_mid                   @ jump to main loop start point\n"

                            //! process right pad
                            "2:                                     @ right pad entry\n"
                            // check do_right_pad, if 0, jump to end
                            "cmp %[do_right_pad], #1                @ check whether has mid cols\n"
                            "blt  5f                                @ jump to main loop start point\n"

                            // load pad mask
                            "vld1.32  {d16-d19}, [%[right_pad_save_mask]] @ load pad index\n" //q8, q9 load 8 uint32
                            // load row0
                            "vld2.32  {d12-d15}, [%[din0_ptr]]!     @ load input r00, r01\n" //interleave load, q6->rx0, q7->rx1

                            // load output
                            "vld1.32  {d20-d21}, [%[doutc0r0]]      @ load dout0r0\n" //q10->outc0r0
                            "vld1.32  {d22-d23}, [%[doutc0r1]]      @ load dout0r1\n" //q11->outc0r1
                            "vld1.32  {d24-d25}, [%[doutc1r0]]      @ load dout1r0\n" //q12->outc1r0
                            "vld1.32  {d26-d27}, [%[doutc1r1]]      @ load dout1r1\n" //q13->outc1r1

                            // row0,  deal with right pad
                            "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                            "vbif q7, q15, q9                       @ bit select, deal with right pad\n"
                            // shift left 1
                            "vext.32  q14, q6,  q15, #1             @ shift left r0\n"

                            "vmla.f32 q10, q7,   %e[wr00][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q7,   %e[wr10][1]        @ mul weight1, 01, out1r0\n"

                            "vmla.f32 q10, q6,   %e[wr00][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr10][0]        @ mul weight1, 00, out1r0\n"

                            // load row1
                            "vld2.32  {d12-d15}, [%[din1_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                            "vmla.f32 q10, q14,  %f[wr00][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q14,  %f[wr10][0]        @ mul weight1, 02, out1r0\n"

                            // row1,  deal with right pad
                            "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                            "vbif q7, q15, q9                       @ bit select, deal with right pad\n"
                            // shift left 1, r1
                            "vext.32  q14, q6,  q15, #1             @ shift left r0\n"

                            "vmla.f32 q10, q7,   %e[wr01][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q7,   %e[wr11][1]        @ mul weight1, 01, out1r0\n"

                            "vmla.f32 q10, q6,   %e[wr01][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr11][0]        @ mul weight1, 00, out1r0\n"

                            // load row2
                            "vld2.32  {d12-d15}, [%[din2_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                            "vmla.f32 q10, q14,  %f[wr01][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q14,  %f[wr11][0]        @ mul weight1, 02, out1r0\n"

                            // row2, deal with right pad
                            "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                            "vbif q7, q15, q9                       @ bit select, deal with right pad\n"
                            // shift left 1
                            "vext.32  q14, q6, q15, #1              @ shift left r1\n"
                            //! row2
                            "vmla.f32 q10, q7,   %e[wr02][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q7,   %e[wr12][1]        @ mul weight1, 01, out0r0\n"
                            "vmla.f32 q11, q7,   %e[wr00][1]        @ mul weight0, 01, out0r1\n"
                            "vmla.f32 q13, q7,   %e[wr10][1]        @ mul weight1, 01, out1r1\n"

                            "vmla.f32 q10, q6,   %e[wr02][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr12][0]        @ mul weight1, 00, out0r0\n"
                            "vmla.f32 q11, q6,   %e[wr00][0]        @ mul weight0, 00, out0r1\n"
                            "vmla.f32 q13, q6,   %e[wr10][0]        @ mul weight1, 00, out1r1\n"

                            // load row3
                            "vld2.32  {d12-d15}, [%[din3_ptr]]!     @ load input r20, r21\n" //interleave load, q6->rx0, q7->rx1

                            "vmla.f32 q10, q14,  %f[wr02][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q14,  %f[wr12][0]        @ mul weight1, 02, out0r0\n"
                            "vmla.f32 q11, q14,  %f[wr00][0]        @ mul weight0, 02, out0r1\n"
                            "vmla.f32 q13, q14,  %f[wr10][0]        @ mul weight1, 02, out1r1\n"

                            // row3, deal with right pad
                            "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                            "vbif q7, q15, q9                       @ bit select, deal with right pad\n"

                            //! row3
                            "vmla.f32 q11, q7,   %e[wr01][1]        @ mul weight0, 01, out0r1\n"
                            "vmla.f32 q13, q7,   %e[wr11][1]        @ mul weight1, 01, out1r1\n"

                            // shift left 1
                            "vext.32  q14, q6,  q15, #1             @ shift left r2\n"

                            "vmla.f32 q11, q6,   %e[wr01][0]        @ mul weight0, 00, out0r1\n"
                            "vmla.f32 q13, q6,   %e[wr11][0]        @ mul weight1, 00, out1r1\n"

                            "vld1.32  {d12-d13}, [%[doutc0r0]]      @ load dout0r0\n" //q6->outc0r0
                            "vld1.32  {d14-d15}, [%[doutc0r1]]      @ load dout0r1\n" //q7->outc0r1
                            "vld1.32  {d16-d17}, [%[doutc1r0]]      @ load dout1r0\n" //q8->outc1r0
                            "vld1.32  {d18-d19}, [%[doutc1r1]]      @ load dout1r1\n" //q9->outc1r1

                            "vmla.f32 q11, q14,  %f[wr01][0]        @ mul weight0, 02, out0r1\n"
                            "vmla.f32 q13, q14,  %f[wr11][0]        @ mul weight1, 02, out1r1\n"

                            "cmp %[relu], #1                        @ check whether has relu\n"
                            "blt    4f                              @ jump to top left store without relu\n"
                            "vmax.f32   q10,  q10, q15              @ relu\n"
                            "vmax.f32   q11,  q11, q15              @ relu\n"
                            "vmax.f32   q12,  q12, q15              @ relu\n"
                            "vmax.f32   q13,  q13, q15              @ relu\n"

                            "4:                                     @ store top mid result\n"
                            // store tr result
                            "vld1.32  {d28-d29}, [%[right_pad_save_mask]]  @ load save mask\n" //q14->save mask

                            "vbif q10, q6, q14                      @ bit select\n"
                            "vbif q11, q7, q14                      @ bit select\n"
                            "vbif q12, q8, q14                      @ bit select\n"
                            "vbif q13, q9, q14                      @ bit select\n"

                            "vst1.32  {d20-d21}, [%[doutc0r0]]!     @ store result, add pointer\n"
                            "vst1.32  {d22-d23}, [%[doutc0r1]]!     @ store result, add pointer\n"
                            "vst1.32  {d24-d25}, [%[doutc1r0]]!     @ store result, add pointer\n"
                            "vst1.32  {d26-d27}, [%[doutc1r1]]!     @ store result, add pointer\n"

                            "sub %[doutc1r0], %[doutc1r0], %[right_pad_sub] @ sub \n"
                            "sub %[doutc1r1], %[doutc1r1], %[right_pad_sub] @ sub \n"
                            "sub %[doutc0r0], %[doutc0r0], %[right_pad_sub] @ sub \n"
                            "sub %[doutc0r1], %[doutc0r1], %[right_pad_sub] @ sub \n"
                            "5:                                     @ bot row ending\n"

                    :[doutc0r0] "+r"(doutc0r0), [doutc0r1] "+r"(doutc0r1), \
                                [doutc1r0] "+r" (doutc1r0), [doutc1r1] "+r" (doutc1r1),\
                                [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                                [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                                [cnt] "+r"(cnt)
                    :[wr00] "w"(wr00), [wr01] "w"(wr01), [wr02] "w"(wr02), \
                                [wr10] "w"(wr10), [wr11] "w"(wr11), [wr12] "w"(wr12), \
                                [relu] "r" (relu), [do_right_pad] "r" (do_right_pad), \
                                [right_pad_sub] "r" (right_pad_sub), \
                                [right_pad_save_mask] "r" (right_pad_save_mask)

                    :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                } else if (bt_remain == 1){
                    din0_ptr = dr0;
                    din1_ptr = dr1;
                    if (pad_bot_size > 0) {
                        din2_ptr = zero;
                    } else {
                        din2_ptr = dr2;
                    }
                    int cnt = cnt_col;
                    asm volatile(
                    //! process left pad
                    "vmov.u32 q15, #0                       @ dump zero\n"
                            "pld [%[din0_ptr]]                      @ preload data\n"//inr0
                            "pld [%[din1_ptr]]                      @ preload data\n"//inr1
                            "pld [%[din2_ptr]]                      @ preload data\n"//inr2

                            "pld [%[doutc0r0]]                      @ preload data\n" //outc0r0
                            "pld [%[doutc1r0]]                      @ preload data\n"//outc1r0

                            //! row0
                            "vld2.32  {d12-d15}, [%[din0_ptr]]!     @ load input r00, r01\n" //interleave load, q6->rx0, q7->rx1
                            "vld1.32  {d20-d21}, [%[doutc0r0]]      @ load dout0r0\n" //q10->outc0r0
                            "vld1.32  {d24-d25}, [%[doutc1r0]]      @ load dout1r0\n" //q12->outc1r0

                            "vmla.f32 q10, q6,   %e[wr00][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr10][1]        @ mul weight1, 01, out1r0\n"

                            "vmla.f32 q10, q7,   %f[wr00][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q7,   %f[wr10][0]        @ mul weight1, 02, out1r0\n"

                            // shift right 1, r0
                            "vext.32  q14, q15, q7,  #3             @ shift right r0\n"
                            // load row1
                            "vld2.32  {d12-d15}, [%[din1_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1
                            "sub %[din0_ptr], #4                    @ r0 address -4, overlay 1 float\n"

                            "vmla.f32 q10, q14,  %e[wr00][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q14,  %e[wr10][0]        @ mul weight1, 00, out1r0\n"

                            //! row1
                            "vmla.f32 q10, q6,   %e[wr01][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr11][1]        @ mul weight1, 01, out1r0\n"

                            "vmla.f32 q10, q7,   %f[wr01][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q7,   %f[wr11][0]        @ mul weight1, 02, out1r0\n"

                            // shift right 1, r1
                            "vext.32  q14, q15, q7,  #3             @ shift right r1\n"
                            // load row2
                            "vld2.32  {d12-d15}, [%[din2_ptr]]      @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1
                            "sub %[din1_ptr], #4                    @ r0 address -4, overlay 1 float\n"

                            "vmla.f32 q10, q14,  %e[wr01][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q14,  %e[wr11][0]        @ mul weight1, 00, out1r0\n"

                            //! row2
                            "vmla.f32 q10, q6,   %e[wr02][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr12][1]        @ mul weight1, 01, out1r0\n"

                            // check bot pad
                            "cmp %[bot_pad], #1                     @ check whether has relu\n"
                            "bge    11f                             @ jump to top left store without relu\n"
                            "add %[din2_ptr], #28                   @ r1 address -4, overlay 1 float\n"

                            "11:                                    @ check point\n"
                            // shift right 1, r2
                            "vext.32  q14, q15, q7,  #3             @ shift right r2\n"

                            "vmla.f32 q10, q7,   %f[wr02][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q7,   %f[wr12][0]        @ mul weight1, 02, out1r0\n"

                            "vmla.f32 q10, q14,  %e[wr02][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q14,  %e[wr12][0]        @ mul weight1, 00, out1r0\n"

                            "pld [%[din0_ptr]]                      @ preload data\n"//inr0
                            "pld [%[din1_ptr]]                      @ preload data\n"//inr1
                            "pld [%[din2_ptr]]                      @ preload data\n"//inr2

                            "cmp %[relu], #1                        @ check whether has relu\n"
                            "blt    1f                              @ jump to top left store without relu\n"
                            "vmax.f32   q10,  q10, q15              @ relu\n"
                            "vmax.f32   q12,  q12, q15              @ relu\n"

                            "1:                                     @ store top left result\n"
                            // stroe tl result
                            "vst1.32  {d20-d21}, [%[doutc0r0]]!     @ store result, add pointer\n"
                            "vst1.32  {d24-d25}, [%[doutc1r0]]!     @ store result, add pointer\n"

                            "pld [%[doutc0r0]]                      @ preload data\n"
                            "pld [%[doutc1r0]]                      @ preload data\n"

                            //! process mid cols
                            "cmp %[cnt], #1                         @ check whether has mid cols\n"
                            "blt  2f                                @ jump to main loop start point\n"
                            "start_bot1_mid:                         @ main loop in mid rows\n"

                            //! row0
                            "vld2.32  {d12-d15}, [%[din0_ptr]]!     @ load input r00, r01\n" //interleave load, q6->rx0, q7->rx1
                            "vld1.32  {d20-d21}, [%[doutc0r0]]      @ load dout0r0\n" //q10->outc0r0
                            "vld1.32  {d24-d25}, [%[doutc1r0]]      @ load dout1r0\n" //q12->outc1r0

                            "vmla.f32 q10, q7,   %e[wr00][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q7,   %e[wr10][1]        @ mul weight1, 01, out1r0\n"

                            "vld1.32  {d14},    [%[din0_ptr]]       @ load the 8th element, r00\n"

                            "vmla.f32 q10, q6,   %e[wr00][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr10][0]        @ mul weight1, 00, out1r0\n"

                            // shift left 1, r0
                            "vext.32  q14, q6,  q7,  #1             @ shift left r0\n"
                            // load row1
                            "vld2.32  {d12-d15}, [%[din1_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1
                            "pld [%[din0_ptr]]                      @ preload data\n"//inr0

                            "vmla.f32 q10, q14,  %f[wr00][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q14,  %f[wr10][0]        @ mul weight1, 02, out1r0\n"

                            //! row1
                            "vmla.f32 q10, q7,   %e[wr01][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q7,   %e[wr11][1]        @ mul weight1, 01, out1r0\n"

                            "vld1.32  {d14},    [%[din1_ptr]]       @ load the 8th element, r10\n"

                            "vmla.f32 q10, q6,   %e[wr01][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr11][0]        @ mul weight1, 00, out1r0\n"

                            // shift left 1, r1
                            "vext.32  q14, q6,  q7,  #1             @ shift left r0\n"
                            // load row2
                            "vld2.32  {d12-d15}, [%[din2_ptr]]      @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1
                            "pld [%[din1_ptr]]                      @ preload data\n"//inr1

                            "vmla.f32 q10, q14,  %f[wr01][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q14,  %f[wr11][0]        @ mul weight1, 02, out1r0\n"

                            // check bot pad
                            "cmp %[bot_pad], #1                     @ check whether has relu\n"
                            "bge    12f                             @ jump to top left store without relu\n"
                            "add %[din2_ptr], #32                   @ r1 address -4, overlay 1 float\n"

                            "12:                                    @ check point\n"
                            //! row2
                            "vmla.f32 q10, q7,   %e[wr02][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q7,   %e[wr12][1]        @ mul weight1, 01, out1r0\n"

                            "vld1.32  {d14},    [%[din2_ptr]]       @ load the 8th element, r00\n"

                            "vmla.f32 q10, q6,   %e[wr02][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr12][0]        @ mul weight1, 00, out1r0\n"

                            // shift left 1, r12
                            "vext.32  q14, q6,  q7,  #1             @ shift left r2\n"
                            "pld [%[din2_ptr]]                      @ preload data\n"//inr2

                            "vmla.f32 q10, q14,  %f[wr02][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q14,  %f[wr12][0]        @ mul weight1, 02, out1r0\n"

                            "cmp %[relu], #1                        @ check whether has relu\n"
                            "blt    3f                              @ jump to top left store without relu\n"
                            "vmax.f32   q10,  q10, q15              @ relu\n"
                            "vmax.f32   q12,  q12, q15              @ relu\n"

                            "3:                                     @ store top mid result\n"
                            // store tm result
                            "vst1.32  {d20-d21}, [%[doutc0r0]]!     @ store result, add pointer\n"
                            "vst1.32  {d24-d25}, [%[doutc1r0]]!     @ store result, add pointer\n"

                            "subs %[cnt], #1                        @ loop count minus 1\n"
                            "bne    start_bot1_mid                   @ jump to main loop start point\n"

                            //! process right pad
                            "2:                                     @ right pad entry\n"
                            // check do_right_pad, if 0, jump to end
                            "cmp %[do_right_pad], #1                @ check whether has mid cols\n"
                            "blt  5f                                @ jump to main loop start point\n"

                            // load pad mask
                            "vld1.32  {d16-d19}, [%[right_pad_save_mask]]! @ load pad index\n" //q8, q9 load 8 uint32
                            // load row0
                            "vld2.32  {d12-d15}, [%[din0_ptr]]!     @ load input r00, r01\n" //interleave load, q6->rx0, q7->rx1

                            // load output
                            "vld1.32  {d20-d21}, [%[doutc0r0]]      @ load dout0r0\n" //q10->outc0r0
                            "vld1.32  {d24-d25}, [%[doutc1r0]]      @ load dout1r0\n" //q12->outc1r0

                            // row0,  deal with right pad
                            "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                            "vbif q7, q15, q9                       @ bit select, deal with right pad\n"
                            // shift left 1
                            "vext.32  q14, q6,  q15, #1             @ shift left r0\n"

                            "vmla.f32 q10, q7,   %e[wr00][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q7,   %e[wr10][1]        @ mul weight1, 01, out1r0\n"

                            "vmla.f32 q10, q6,   %e[wr00][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr10][0]        @ mul weight1, 00, out1r0\n"

                            // load row1
                            "vld2.32  {d12-d15}, [%[din1_ptr]]!     @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                            "vmla.f32 q10, q14,  %f[wr00][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q14,  %f[wr10][0]        @ mul weight1, 02, out1r0\n"

                            // row1,  deal with right pad
                            "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                            "vbif q7, q15, q9                       @ bit select, deal with right pad\n"
                            // shift left 1, r1
                            "vext.32  q14, q6,  q15, #1             @ shift left r0\n"

                            "vmla.f32 q10, q7,   %e[wr01][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q7,   %e[wr11][1]        @ mul weight1, 01, out1r0\n"

                            "vmla.f32 q10, q6,   %e[wr01][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr11][0]        @ mul weight1, 00, out1r0\n"

                            // load row2
                            "vld2.32  {d12-d15}, [%[din2_ptr]]      @ load input r10, r11\n" //interleave load, q6->rx0, q7->rx1

                            "vmla.f32 q10, q14,  %f[wr01][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q14,  %f[wr11][0]        @ mul weight1, 02, out1r0\n"

                            // row2, deal with right pad
                            "vbif q6, q15, q8                       @ bit select, deal with right pad\n"
                            "vbif q7, q15, q9                       @ bit select, deal with right pad\n"
                            // shift left 1
                            "vext.32  q14, q6, q15, #1              @ shift left r1\n"
                            //! row2
                            "vmla.f32 q10, q7,   %e[wr02][1]        @ mul weight0, 01, out0r0\n"
                            "vmla.f32 q12, q7,   %e[wr12][1]        @ mul weight1, 01, out0r0\n"

                            "vld1.32  {d14-d15}, [%[doutc0r0]]      @ load dout0r0\n" //q7->outc0r0

                            "vmla.f32 q10, q6,   %e[wr02][0]        @ mul weight0, 00, out0r0\n"
                            "vmla.f32 q12, q6,   %e[wr12][0]        @ mul weight1, 00, out0r0\n"

                            "vld1.32  {d16-d17}, [%[doutc1r0]]      @ load dout1r0\n" //q8->outc1r0

                            "vmla.f32 q10, q14,  %f[wr02][0]        @ mul weight0, 02, out0r0\n"
                            "vmla.f32 q12, q14,  %f[wr12][0]        @ mul weight1, 02, out0r0\n"


                            "cmp %[relu], #1                        @ check whether has relu\n"
                            "blt    4f                              @ jump to top left store without relu\n"
                            "vmax.f32   q10,  q10, q15              @ relu\n"
                            "vmax.f32   q12,  q12, q15              @ relu\n"

                            "4:                                     @ store top mid result\n"
                            // store tr result
                            "vld1.32  {d28-d29}, [%[right_pad_save_mask]]  @ load save mask\n" //q14->save mask

                            "vbif q10, q7, q14                      @ bit select\n"
                            "vbif q12, q8, q14                      @ bit select\n"

                            "vst1.32  {d20-d21}, [%[doutc0r0]]      @ store result, add pointer\n"
                            "vst1.32  {d24-d25}, [%[doutc1r0]]      @ store result, add pointer\n"
                            "5:                                     @ bot row ending\n"

                    :[doutc0r0] "+r"(doutc0r0),[doutc1r0] "+r" (doutc1r0), \
                                [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                                [din2_ptr] "+r"(din2_ptr), [cnt] "+r"(cnt)
                    :[wr00] "w"(wr00), [wr01] "w"(wr01), [wr02] "w"(wr02), \
                                [wr10] "w"(wr10), [wr11] "w"(wr11), [wr12] "w"(wr12), \
                                [relu] "r" (relu), [do_right_pad] "r" (do_right_pad), \
                                [right_pad_sub] "r" (right_pad_sub), [bot_pad] "r" (pad_bot_size), \
                                [right_pad_save_mask] "r" (right_pad_save_mask)
                    :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }
                // end of processing bottom pad
            } // end of processing channels
        } //end of processing output channel
        if (cremain > 0) {
            for (int c = 0; c < cremain; ++c) {

                int cidx = ch_out - cremain + c;
                float* dout_c = dout_batch + cidx * size_out_channel;

                if (flag_bias) {
                    fill_bias(dout_c, &bias[cidx], 1, size_out_channel);
                } else {
                    fill_bias(dout_c, zero, 1, size_out_channel);
                }

                const float* wc0 = weights + cidx * w_stride;

                for (int i = 0; i < ch_in; ++i) {

                    bool relu = (i == ch_in - 1) && flag_relu;

                    const float* din_channel = din_batch + i * size_in_channel;
                    for (int h = 0; h < h_out; ++h) {

                        int hstart = h - pad_h;
                        int hend = hstart + 3;
                        hstart = std::max(hstart, 0);
                        hend = std::min(hend, h_in);

                        int khstart = hend < kernel_h? kernel_h - hend : 0;

                        float* dout_row = dout_c + h * w_out;

                        for (int w = 0; w < w_out; ++w) {
                            int wstart = w - pad_w;
                            int wend = wstart + 3;
                            wstart = std::max(wstart, 0);
                            wend = std::min(wend, w_in);
                            int kwstart = wend < kernel_w? kernel_w - wend : 0;

                            for (int kh = hstart; kh < hend; ++kh) {
                                for (int kw = wstart; kw < wend; ++kw) {
                                    dout_row[w] += din_channel[kh * w_in + kw] * \
                                        wc0[(khstart + kh - hstart) * 3 + kwstart + kw - wstart];
                                }
                            }
                            if (relu) {
                                dout_row[w] = dout_row[w] > 0.f? dout_row[w] : 0.f;
                            }
                        }
                    }
                    wc0 += 9;
                }
            }
        } // end of remain out channel

    } // end of processing batchs
}

#endif //__aarch64__
} //namespace lite

} //namespace saber

} //namespace anakin

#endif
