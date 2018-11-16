#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

void conv_1x5s1_direct(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias_ptr, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, \
                          Context* ctx, void* work_space, const void* idx_ptr) {
    //! 1x5s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    float* zero_ptr = static_cast<float*>(ctx->get_work_space());
    memset(zero_ptr, 0, win * sizeof(float));
    float* write_ptr = zero_ptr + win;

    int size_in_sp = win * hin;
    int size_in_batch = size_in_sp * chin;
    int size_out_sp = wout * hout;
    int size_out_batch = size_out_sp * chout;

    int size_weight_sp = 5 * chin;

    // cnt - left pad - right pad
    int cnt_col = (wout + 3) / 4 - 2;

    int wremain = 8 - ((cnt_col + 2) * 4 + kernel_w - 1 - win - pad_w);
    int remain = 4 - ((cnt_col + 2) * 4 - wout);
    if (remain == 0) {
        remain = 4;
    }
    const float* bias = static_cast<const float*> (bias_ptr);

    //! layout NHWC
    for (int n = 0; n < num; ++n) {
        const float* din_batch = static_cast<const float*>(din) + n * size_in_batch;
        float* dout_batch = static_cast<float*>(dout) + n * size_out_batch;
#pragma omp parallel for
        for (int k = 0; k < chout - 1; k += 2) {

            float* dout_c0 = dout_batch + k * size_out_sp;
            float* dout_c1 = dout_c0 + size_out_sp;
            const float* w_ptr_o0 = static_cast<const float*>(weights) + k * 5 * chin;
            const float* w_ptr_o1 = w_ptr_o0 + 5 * chin;

            if (flag_bias) {
                fill_bias(dout_c0, &bias[k], 2, size_out_sp);
            } else {
                fill_bias(dout_c0, zero_ptr, 2, size_out_sp);
            }

            for (int c = 0; c < chin; ++c) {
                const float* din_c = din_batch + c * size_in_sp;
                const float* w_ptr_i0 = w_ptr_o0 + 5 * c;
                const float* w_ptr_i1 = w_ptr_o1 + 5 * c;
                //! pad zero to top and bottom row
                // q0 = w0, w1, w2, w3; q1 = w4
                float w00 = w_ptr_i0[0];
                float w01 = w_ptr_i0[1];
                float w02 = w_ptr_i0[2];
                float w03 = w_ptr_i0[3];
                float w04 = w_ptr_i0[4];

                float w10 = w_ptr_i1[0];
                float w11 = w_ptr_i1[1];
                float w12 = w_ptr_i1[2];
                float w13 = w_ptr_i1[3];
                float w14 = w_ptr_i1[4];

                int relu = 0;
                if ((c == chin - 1) && flag_relu) {
                    relu = 1;
                }

                int h = pad_h;

                const float* din_r0 = nullptr;
                const float* din_r1 = nullptr;
                const float* din_r2 = nullptr;
                const float* din_r3 = nullptr;

                float* outc0r0 = nullptr;
                float* outc0r1 = nullptr;
                float* outc0r2 = nullptr;
                float* outc0r3 = nullptr;

                float* outc1r0 = nullptr;
                float* outc1r1 = nullptr;
                float* outc1r2 = nullptr;
                float* outc1r3 = nullptr;

                float32x4_t vr10;
                float32x4_t vr20;
                float32x4_t vr11;
                float32x4_t vr21;

                float32x4_t vor00;
                float32x4_t vor01;
                float32x4_t vor02;
                float32x4_t vor03;
                float32x4_t vor10;
                float32x4_t vor11;
                float32x4_t vor12;
                float32x4_t vor13;

                const float* r0 = din_c + (h - pad_h) * win;
                const float* r1 = r0 + win;
                const float* r2 = r1 + win;
                const float* r3 = r2 + win;
                for (; h < hout - pad_h; h += 4) {
                    din_r0 = r0;
                    din_r1 = r1;
                    din_r2 = r2;
                    din_r3 = r3;

                    outc0r0 = dout_c0 + h * wout;
                    outc0r1 = outc0r0 + wout;
                    outc0r2 = outc0r1 + wout;
                    outc0r3 = outc0r2 + wout;

                    outc1r0 = dout_c1 + h * wout;
                    outc1r1 = outc1r0 + wout;
                    outc1r2 = outc1r1 + wout;
                    outc1r3 = outc1r2 + wout;

                    //! process bottom pad
                    if (h + 3 > hin) {
                        switch (h + 3 - hin) {
                            case 3:
                                din_r1 = zero_ptr;
                                outc0r1 = write_ptr;
                                outc1r1 = write_ptr;
                            case 2:
                                din_r2 = zero_ptr;
                                outc0r2 = write_ptr;
                                outc1r2 = write_ptr;
                            case 1:
                                din_r3 = zero_ptr;
                                outc0r3 = write_ptr;
                                outc1r3 = write_ptr;
                            default:
                                break;
                        }
                    }

                    //! process left pad
                    float32x4_t or00 = vld1q_f32(outc0r0);
                    float32x4_t or01 = vld1q_f32(outc0r1);
                    float32x4_t or02 = vld1q_f32(outc0r2);
                    float32x4_t or03 = vld1q_f32(outc0r3);

                    float32x4_t or10 = vld1q_f32(outc1r0);
                    float32x4_t or11 = vld1q_f32(outc1r1);
                    float32x4_t or12 = vld1q_f32(outc1r2);
                    float32x4_t or13 = vld1q_f32(outc1r3);

                    float32x4_t r00 = vld1q_f32(din_r0);
                    float32x4_t r10 = vld1q_f32(din_r1);
                    float32x4_t r01 = vld1q_f32(din_r0 + 4);
                    float32x4_t r11 = vld1q_f32(din_r1 + 4);

                    float32x4_t rext01 = vdupq_n_f32(0.f);
                    float32x4_t rext11;
                    float32x4_t rext00 = vextq_f32(rext01, r00, 3);
                    float32x4_t rext10 = vextq_f32(rext01, r10, 3);

                    or00 = vmlaq_n_f32(or00, r00, w01);
                    or01 = vmlaq_n_f32(or01, r10, w01);
                    or10 = vmlaq_n_f32(or10, r00, w11);
                    or11 = vmlaq_n_f32(or11, r10, w11);

                    rext01 = vextq_f32(r00, r01, 1);
                    rext11 = vextq_f32(r10, r11, 1);

                    or00 = vmlaq_n_f32(or00, rext00, w00);
                    or01 = vmlaq_n_f32(or01, rext10, w00);
                    or10 = vmlaq_n_f32(or10, rext00, w10);
                    or11 = vmlaq_n_f32(or11, rext10, w10);

                    rext00 = vextq_f32(r00, r01, 2);
                    rext10 = vextq_f32(r10, r11, 2);

                    or00 = vmlaq_n_f32(or00, rext01, w02);
                    or01 = vmlaq_n_f32(or01, rext11, w02);
                    or10 = vmlaq_n_f32(or10, rext01, w12);
                    or11 = vmlaq_n_f32(or11, rext11, w12);

                    rext01 = vextq_f32(r00, r01, 3);
                    rext11 = vextq_f32(r10, r11, 3);

                    or00 = vmlaq_n_f32(or00, rext00, w03);
                    r00 = vld1q_f32(din_r2);
                    or01 = vmlaq_n_f32(or01, rext10, w03);
                    r01 = vld1q_f32(din_r2 + 4);
                    or10 = vmlaq_n_f32(or10, rext00, w13);
                    r10 = vld1q_f32(din_r3);
                    or11 = vmlaq_n_f32(or11, rext10, w13);
                    r11 = vld1q_f32(din_r3 + 4);

                    or00 = vmlaq_n_f32(or00, rext01, w04);
                    or01 = vmlaq_n_f32(or01, rext11, w04);
                    or10 = vmlaq_n_f32(or10, rext01, w14);
                    or11 = vmlaq_n_f32(or11, rext11, w14);

                    rext01 = vdupq_n_f32(0.f);
                    rext00 = vextq_f32(rext01, r00, 3);
                    rext10 = vextq_f32(rext01, r10, 3);

                    or02 = vmlaq_n_f32(or02, r00, w01);
                    or03 = vmlaq_n_f32(or03, r10, w01);
                    or12 = vmlaq_n_f32(or12, r00, w11);
                    or13 = vmlaq_n_f32(or13, r10, w11);

                    rext01 = vextq_f32(r00, r01, 1);
                    rext11 = vextq_f32(r10, r11, 1);

                    or02 = vmlaq_n_f32(or02, rext00, w00);
                    or03 = vmlaq_n_f32(or03, rext10, w00);
                    or12 = vmlaq_n_f32(or12, rext00, w10);
                    or13 = vmlaq_n_f32(or13, rext10, w10);

                    rext00 = vextq_f32(r00, r01, 2);
                    rext10 = vextq_f32(r10, r11, 2);

                    or02 = vmlaq_n_f32(or02, rext01, w02);
                    or03 = vmlaq_n_f32(or03, rext11, w02);
                    or12 = vmlaq_n_f32(or12, rext01, w12);
                    or13 = vmlaq_n_f32(or13, rext11, w12);

                    rext01 = vextq_f32(r00, r01, 3);
                    rext11 = vextq_f32(r10, r11, 3);

                    or02 = vmlaq_n_f32(or02, rext00, w03);
                    or03 = vmlaq_n_f32(or03, rext10, w03);
                    or12 = vmlaq_n_f32(or12, rext00, w13);
                    or13 = vmlaq_n_f32(or13, rext10, w13);

                    din_r0 += 3;
                    din_r1 += 3;

                    or02 = vmlaq_n_f32(or02, rext01, w04);
                    din_r2 += 3;
                    or03 = vmlaq_n_f32(or03, rext11, w04);
                    din_r3 += 3;
                    or12 = vmlaq_n_f32(or12, rext01, w14);
                    r00 = vld1q_f32(din_r0);
                    or13 = vmlaq_n_f32(or13, rext11, w14);
                    r01 = vld1q_f32(din_r0 + 4);
                    r10 = vld1q_f32(din_r1);
                    r11 = vld1q_f32(din_r1 + 4);

                    if (relu) {
                        rext01 = vdupq_n_f32(0.f);
                        or00 = vmaxq_f32(or00, rext01);
                        or01 = vmaxq_f32(or01, rext01);
                        or02 = vmaxq_f32(or02, rext01);
                        or03 = vmaxq_f32(or03, rext01);
                        vst1q_f32(outc0r0, or00);
                        outc0r0 += 4;
                        or10 = vmaxq_f32(or10, rext01);
                        vst1q_f32(outc0r1, or01);
                        outc0r1 += 4;
                        or11 = vmaxq_f32(or11, rext01);
                        vst1q_f32(outc0r2, or02);
                        outc0r2 += 4;
                        or12 = vmaxq_f32(or12, rext01);
                        vst1q_f32(outc0r3, or03);
                        outc0r3 += 4;
                        or13 = vmaxq_f32(or13, rext01);
                        vst1q_f32(outc1r0, or10);
                        outc1r0 += 4;
                        vst1q_f32(outc1r1, or11);
                        outc1r1 += 4;
                        vst1q_f32(outc1r2, or12);
                        outc1r2 += 4;
                        vst1q_f32(outc1r3, or13);
                        outc1r3 += 4;
                    } else {
                        vst1q_f32(outc0r0, or00);
                        outc0r0 += 4;
                        vst1q_f32(outc0r1, or01);
                        outc0r1 += 4;
                        vst1q_f32(outc0r2, or02);
                        outc0r2 += 4;
                        vst1q_f32(outc0r3, or03);
                        outc0r3 += 4;
                        vst1q_f32(outc1r0, or10);
                        outc1r0 += 4;
                        vst1q_f32(outc1r1, or11);
                        outc1r1 += 4;
                        vst1q_f32(outc1r2, or12);
                        outc1r2 += 4;
                        vst1q_f32(outc1r3, or13);
                        outc1r3 += 4;
                    }

                    for (int w = 0; w < cnt_col; ++w) {

                        or00 = vld1q_f32(outc0r0);
                        or01 = vld1q_f32(outc0r1);
                        or10 = vld1q_f32(outc1r0);
                        or11 = vld1q_f32(outc1r1);

                        rext00 = vextq_f32(r00, r01, 1);
                        rext10 = vextq_f32(r10, r11, 1);

                        or00 = vmlaq_n_f32(or00, r00, w00);
                        or02 = vld1q_f32(outc0r2);
                        or01 = vmlaq_n_f32(or01, r10, w00);
                        or03 = vld1q_f32(outc0r3);
                        or10 = vmlaq_n_f32(or10, r00, w10);
                        or12 = vld1q_f32(outc1r2);
                        or11 = vmlaq_n_f32(or11, r10, w10);
                        or13 = vld1q_f32(outc1r3);

                        or00 = vmlaq_n_f32(or00, r01, w04);
                        or01 = vmlaq_n_f32(or01, r11, w04);
                        or10 = vmlaq_n_f32(or10, r01, w14);
                        or11 = vmlaq_n_f32(or11, r11, w14);

                        rext01 = vextq_f32(r00, r01, 2);
                        rext11 = vextq_f32(r10, r11, 2);

                        or00 = vmlaq_n_f32(or00, rext00, w01);
                        or01 = vmlaq_n_f32(or01, rext10, w01);
                        or10 = vmlaq_n_f32(or10, rext00, w11);
                        or11 = vmlaq_n_f32(or11, rext10, w11);

                        rext00 = vextq_f32(r00, r01, 3);
                        rext10 = vextq_f32(r10, r11, 3);

                        or00 = vmlaq_n_f32(or00, rext01, w02);
                        r00 = vld1q_f32(din_r2);
                        or01 = vmlaq_n_f32(or01, rext11, w02);
                        r01 = vld1q_f32(din_r2 + 4);
                        or10 = vmlaq_n_f32(or10, rext01, w12);
                        r10 = vld1q_f32(din_r3);
                        or11 = vmlaq_n_f32(or11, rext11, w12);
                        r11 = vld1q_f32(din_r3 + 4);

                        or00 = vmlaq_n_f32(or00, rext00, w03);
                        or01 = vmlaq_n_f32(or01, rext10, w03);
                        or10 = vmlaq_n_f32(or10, rext00, w13);
                        or11 = vmlaq_n_f32(or11, rext10, w13);

                        rext00 = vextq_f32(r00, r01, 1);
                        rext10 = vextq_f32(r10, r11, 1);

                        or02 = vmlaq_n_f32(or02, r00, w00);
                        or03 = vmlaq_n_f32(or03, r10, w00);
                        or12 = vmlaq_n_f32(or12, r00, w10);
                        or13 = vmlaq_n_f32(or13, r10, w10);

                        din_r0 += 4;
                        or02 = vmlaq_n_f32(or02, r01, w04);
                        din_r1 += 4;
                        or03 = vmlaq_n_f32(or03, r11, w04);
                        din_r2 += 4;
                        or12 = vmlaq_n_f32(or12, r01, w14);
                        din_r3 += 4;
                        or13 = vmlaq_n_f32(or13, r11, w14);

                        rext01 = vextq_f32(r00, r01, 2);
                        rext11 = vextq_f32(r10, r11, 2);

                        or02 = vmlaq_n_f32(or02, rext00, w01);
                        or03 = vmlaq_n_f32(or03, rext10, w01);
                        or12 = vmlaq_n_f32(or12, rext00, w11);
                        or13 = vmlaq_n_f32(or13, rext10, w11);

                        rext00 = vextq_f32(r00, r01, 3);
                        rext10 = vextq_f32(r10, r11, 3);

                        or02 = vmlaq_n_f32(or02, rext01, w02);
                        or03 = vmlaq_n_f32(or03, rext11, w02);
                        or12 = vmlaq_n_f32(or12, rext01, w12);
                        or13 = vmlaq_n_f32(or13, rext11, w12);

                        or02 = vmlaq_n_f32(or02, rext00, w03);
                        r00 = vld1q_f32(din_r0);
                        or03 = vmlaq_n_f32(or03, rext10, w03);
                        r01 = vld1q_f32(din_r0 + 4);
                        or12 = vmlaq_n_f32(or12, rext00, w13);
                        r10 = vld1q_f32(din_r1);
                        or13 = vmlaq_n_f32(or13, rext10, w13);
                        r11 = vld1q_f32(din_r1 + 4);

                        if (relu) {
                            rext01 = vdupq_n_f32(0.f);
                            or00 = vmaxq_f32(or00, rext01);
                            or01 = vmaxq_f32(or01, rext01);
                            or02 = vmaxq_f32(or02, rext01);
                            or03 = vmaxq_f32(or03, rext01);
                            vst1q_f32(outc0r0, or00);
                            outc0r0 += 4;
                            or10 = vmaxq_f32(or10, rext01);
                            vst1q_f32(outc0r1, or01);
                            outc0r1 += 4;
                            or11 = vmaxq_f32(or11, rext01);
                            vst1q_f32(outc0r2, or02);
                            outc0r2 += 4;
                            or12 = vmaxq_f32(or12, rext01);
                            vst1q_f32(outc0r3, or03);
                            outc0r3 += 4;
                            or13 = vmaxq_f32(or13, rext01);
                            vst1q_f32(outc1r0, or10);
                            outc1r0 += 4;
                            vst1q_f32(outc1r1, or11);
                            outc1r1 += 4;
                            vst1q_f32(outc1r2, or12);
                            outc1r2 += 4;
                            vst1q_f32(outc1r3, or13);
                            outc1r3 += 4;
                        } else {
                            vst1q_f32(outc0r0, or00);
                            outc0r0 += 4;
                            vst1q_f32(outc0r1, or01);
                            outc0r1 += 4;
                            vst1q_f32(outc0r2, or02);
                            outc0r2 += 4;
                            vst1q_f32(outc0r3, or03);
                            outc0r3 += 4;
                            vst1q_f32(outc1r0, or10);
                            outc1r0 += 4;
                            vst1q_f32(outc1r1, or11);
                            outc1r1 += 4;
                            vst1q_f32(outc1r2, or12);
                            outc1r2 += 4;
                            vst1q_f32(outc1r3, or13);
                            outc1r3 += 4;
                        }

                    }
                    //! process right pad
                    float tmp0[8];
                    float tmp1[8];
                    float tmp2[8];
                    float tmp3[8];
                    int j = 0;
                    for (; j < wremain; ++j) {
                        tmp0[j] = *(din_r0++);
                        tmp1[j] = *(din_r1++);
                        tmp2[j] = *(din_r2++);
                        tmp3[j] = *(din_r3++);
                    }
                    for (; j < 8; ++j) {
                        tmp0[j] = 0.f;
                        tmp1[j] = 0.f;
                        tmp2[j] = 0.f;
                        tmp3[j] = 0.f;
                    }
                    r00 = vld1q_f32(tmp0);
                    r01 = vld1q_f32(tmp0 + 4);
                    r10 = vld1q_f32(tmp1);
                    r11 = vld1q_f32(tmp1 + 4);

                    or00 = vld1q_f32(outc0r0);
                    or01 = vld1q_f32(outc0r1);
                    or10 = vld1q_f32(outc1r0);
                    or11 = vld1q_f32(outc1r1);

                    rext00 = vextq_f32(r00, r01, 1);
                    rext10 = vextq_f32(r10, r11, 1);

                    or00 = vmlaq_n_f32(or00, r00, w00);
                    or02 = vld1q_f32(outc0r2);
                    or01 = vmlaq_n_f32(or01, r10, w00);
                    or03 = vld1q_f32(outc0r3);
                    or10 = vmlaq_n_f32(or10, r00, w10);
                    or12 = vld1q_f32(outc1r2);
                    or11 = vmlaq_n_f32(or11, r10, w10);
                    or13 = vld1q_f32(outc1r3);

                    or00 = vmlaq_n_f32(or00, r01, w04);
                    or01 = vmlaq_n_f32(or01, r11, w04);
                    or10 = vmlaq_n_f32(or10, r01, w14);
                    or11 = vmlaq_n_f32(or11, r11, w14);

                    rext01 = vextq_f32(r00, r01, 2);
                    rext11 = vextq_f32(r10, r11, 2);

                    or00 = vmlaq_n_f32(or00, rext00, w01);
                    or01 = vmlaq_n_f32(or01, rext10, w01);
                    or10 = vmlaq_n_f32(or10, rext00, w11);
                    or11 = vmlaq_n_f32(or11, rext10, w11);

                    rext00 = vextq_f32(r00, r01, 3);
                    rext10 = vextq_f32(r10, r11, 3);

                    or00 = vmlaq_n_f32(or00, rext01, w02);
                    r00 = vld1q_f32(tmp2);
                    or01 = vmlaq_n_f32(or01, rext11, w02);
                    r01 = vld1q_f32(tmp2 + 4);
                    or10 = vmlaq_n_f32(or10, rext01, w12);
                    r10 = vld1q_f32(tmp3);
                    or11 = vmlaq_n_f32(or11, rext11, w12);
                    r11 = vld1q_f32(tmp3 + 4);

                    or00 = vmlaq_n_f32(or00, rext00, w03);
                    or01 = vmlaq_n_f32(or01, rext10, w03);
                    or10 = vmlaq_n_f32(or10, rext00, w13);
                    or11 = vmlaq_n_f32(or11, rext10, w13);

                    rext00 = vextq_f32(r00, r01, 1);
                    rext10 = vextq_f32(r10, r11, 1);

                    or02 = vmlaq_n_f32(or02, r00, w00);
                    or03 = vmlaq_n_f32(or03, r10, w00);
                    or12 = vmlaq_n_f32(or12, r00, w10);
                    or13 = vmlaq_n_f32(or13, r10, w10);

                    or02 = vmlaq_n_f32(or02, r01, w04);
                    or03 = vmlaq_n_f32(or03, r11, w04);
                    or12 = vmlaq_n_f32(or12, r01, w14);
                    or13 = vmlaq_n_f32(or13, r11, w14);

                    rext01 = vextq_f32(r00, r01, 2);
                    rext11 = vextq_f32(r10, r11, 2);

                    or02 = vmlaq_n_f32(or02, rext00, w01);
                    or03 = vmlaq_n_f32(or03, rext10, w01);
                    or12 = vmlaq_n_f32(or12, rext00, w11);
                    or13 = vmlaq_n_f32(or13, rext10, w11);

                    rext00 = vextq_f32(r00, r01, 3);
                    rext10 = vextq_f32(r10, r11, 3);

                    or02 = vmlaq_n_f32(or02, rext01, w02);
                    or03 = vmlaq_n_f32(or03, rext11, w02);
                    or12 = vmlaq_n_f32(or12, rext01, w12);
                    or13 = vmlaq_n_f32(or13, rext11, w12);

                    or02 = vmlaq_n_f32(or02, rext00, w03);
                    or03 = vmlaq_n_f32(or03, rext10, w03);
                    or12 = vmlaq_n_f32(or12, rext00, w13);
                    or13 = vmlaq_n_f32(or13, rext10, w13);

                    if (relu) {
                        rext01 = vdupq_n_f32(0.f);
                        or00 = vmaxq_f32(or00, rext01);
                        or01 = vmaxq_f32(or01, rext01);
                        or02 = vmaxq_f32(or02, rext01);
                        or03 = vmaxq_f32(or03, rext01);
                        vst1q_f32(tmp0, or00);
                        or10 = vmaxq_f32(or10, rext01);
                        vst1q_f32(tmp0 + 4, or01);
                        or11 = vmaxq_f32(or11, rext01);
                        vst1q_f32(tmp1, or02);
                        or12 = vmaxq_f32(or12, rext01);
                        vst1q_f32(tmp1 + 4, or03);
                        or13 = vmaxq_f32(or13, rext01);
                        vst1q_f32(tmp2, or10);
                        vst1q_f32(tmp2 + 4, or11);
                        vst1q_f32(tmp3, or12);
                        vst1q_f32(tmp3 + 4, or13);
                    } else {
                        vst1q_f32(tmp0, or00);
                        vst1q_f32(tmp0 + 4, or01);
                        vst1q_f32(tmp1, or02);
                        vst1q_f32(tmp1 + 4, or03);
                        vst1q_f32(tmp2, or10);
                        vst1q_f32(tmp2 + 4, or11);
                        vst1q_f32(tmp3, or12);
                        vst1q_f32(tmp3 + 4, or13);
                    }

                    for (int i = 0; i < remain; ++i) {
                        *(outc0r0++) = tmp0[i];
                        *(outc0r1++) = tmp0[i + 4];
                        *(outc0r2++) = tmp1[i];
                        *(outc0r3++) = tmp1[i + 4];
                        *(outc1r0++) = tmp2[i];
                        *(outc1r1++) = tmp2[i + 4];
                        *(outc1r2++) = tmp3[i];
                        *(outc1r3++) = tmp3[i + 4];
                    }

                    //! end right pad

                    r0 = r3 + win;
                    r1 = r0 + win;
                    r2 = r1 + win;
                    r3 = r2 + win;
                }
            }
        }
    }

}

} //namespace lite

} //namespace saber

} //namespace anakin

#endif
