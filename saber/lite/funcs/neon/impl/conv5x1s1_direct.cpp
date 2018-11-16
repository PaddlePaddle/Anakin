#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"
#ifdef USE_ARM_PLACE
#include "saber/lite/funcs/arm_utils.h"
namespace anakin{

namespace saber{

namespace lite{

void conv_5x1s1_direct(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias_ptr, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, \
                          int dila_w, int dila_h, int pad_w, int pad_h, \
                          bool flag_bias, bool flag_relu, \
                          Context* ctx, void* work_space, const void* idx_ptr) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    float* zero_ptr = static_cast<float*>(ctx->get_work_space());
    float* write_ptr = static_cast<float*>(ctx->get_work_space()) + win;
    memset(zero_ptr, 0, sizeof(float) * win);

    int size_in_sp = win * hin;
    int size_in_batch = size_in_sp * chin;
    int size_out_sp = wout * hout;
    int size_out_batch = size_out_sp * chout;

    int size_weight_sp = 5 * chin;

    int c_remain = chout & 1;

    const float* bias = static_cast<const float*> (bias_ptr);

    //! layout NHWC
    for (int n = 0; n < num; ++n) {
        const float* din_batch = static_cast<const float*>(din) + n * size_in_batch;
        float* dout_batch = static_cast<float*>(dout) + n * size_out_batch;
#pragma omp parallel for
        for (int k = 0; k < chout - 1; k += 2) {
            float* dout_c0 = dout_batch + k * size_out_sp;
            float* dout_c1 = dout_c0 + size_out_sp;
            const float* w_ptr_o0 = static_cast<const float*>(weights) + k * size_weight_sp;
            const float* w_ptr_o1 = w_ptr_o0 + size_weight_sp;

            if (flag_bias) {
                fill_bias(dout_c0, &bias[k], 2, size_out_sp);
            } else {
                fill_bias(dout_c0, zero_ptr, 2, size_out_sp);
            }

            float tmp00[4];
            float tmp01[4];
            float tmp02[4];
            float tmp03[4];
            float tmp10[4];
            float tmp11[4];
            float tmp12[4];
            float tmp13[4];

            for (int c = 0; c < chin; ++c) {
                const float* din_c = din_batch + c * size_in_sp;
                const float* w_ptr_i0 = w_ptr_o0 + 5 * c;
                const float* w_ptr_i1 = w_ptr_o1 + 5 * c;

                int relu = 0;
                if ((c == chin - 1) && flag_relu) {
                    relu = 1;
                }

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

                const float* din_r0 = nullptr;
                const float* din_r1 = nullptr;
                const float* din_r2 = nullptr;
                const float* din_r3 = nullptr;
                const float* din_r4 = nullptr;
                const float* din_r5 = nullptr;
                const float* din_r6 = nullptr;
                const float* din_r7 = nullptr;

                float* outc0r0 = nullptr;
                float* outc0r1 = nullptr;
                float* outc0r2 = nullptr;
                float* outc0r3 = nullptr;

                float* outc1r0 = nullptr;
                float* outc1r1 = nullptr;
                float* outc1r2 = nullptr;
                float* outc1r3 = nullptr;

                float32x4_t vzero = vdupq_n_f32(0.f);

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

                //! process pad which do nothing
                int h = (pad_h - 4) > 0? pad_h - 4 : 0;

                const float* r0 = din_c - (pad_h - h) * win;
                const float* r1 = r0 + win;
                const float* r2 = r1 + win;
                const float* r3 = r2 + win;
                const float* r4 = r3 + win;
                const float* r5 = r4 + win;
                const float* r6 = r5 + win;
                const float* r7 = r6 + win;

                for (; h < hout; h += 4) {

                    din_r0 = r0;
                    din_r1 = r1;
                    din_r2 = r2;
                    din_r3 = r3;
                    din_r4 = r4;
                    din_r5 = r5;
                    din_r6 = r6;
                    din_r7 = r7;

                    outc0r0 = dout_c0 + h * wout + pad_w;
                    outc0r1 = outc0r0 + wout;
                    outc0r2 = outc0r1 + wout;
                    outc0r3 = outc0r2 + wout;

                    outc1r0 = dout_c1 + h * wout + pad_w;
                    outc1r1 = outc1r0 + wout;
                    outc1r2 = outc1r1 + wout;
                    outc1r3 = outc1r2 + wout;

                    //! process top pad
                    if (h - pad_h < 0) {
                        switch (pad_h - h) {
                            case 4:
                                din_r3 = zero_ptr;
                            case 3:
                                din_r2 = zero_ptr;
                            case 2:
                                din_r1 = zero_ptr;
                            case 1:
                                din_r0 = zero_ptr;
                            default:
                                break;
                        }
                    }

                    //! process bottom pad
                    if (h + 8 - pad_h > hin) {
                        switch (h + 8 - pad_h - hin) {
                            case 7:
                                din_r1 = zero_ptr;
                            case 6:
                                din_r2 = zero_ptr;
                            case 5:
                                din_r3 = zero_ptr;
                            case 4:
                                din_r4 = zero_ptr;
                            case 3:
                                din_r5 = zero_ptr;
                            case 2:
                                din_r6 = zero_ptr;
                            case 1:
                                din_r7 = zero_ptr;
                            default:
                                break;
                        }
                    }

                    //! process bottom remain
                    if (h + 4 > hout) {
                        switch (h + 4 - hout) {
                            case 3:
                                outc0r1 = write_ptr;
                                outc1r1 = write_ptr;
                            case 2:
                                outc0r2 = write_ptr;
                                outc1r2 = write_ptr;
                            case 1:
                                outc0r3 = write_ptr;
                                outc1r3 = write_ptr;
                            default:
                                break;
                        }
                    }

                    if (relu) {
                        for (int i = 0; i < pad_w; ++i) {
                            outc0r0[i - pad_w] = fmaxf(0.f, outc0r0[i - pad_w]);
                            outc0r1[i - pad_w] = fmaxf(0.f, outc0r1[i - pad_w]);
                            outc0r2[i - pad_w] = fmaxf(0.f, outc0r2[i - pad_w]);
                            outc0r3[i - pad_w] = fmaxf(0.f, outc0r3[i - pad_w]);
                            outc1r0[i - pad_w] = fmaxf(0.f, outc1r0[i - pad_w]);
                            outc1r1[i - pad_w] = fmaxf(0.f, outc1r1[i - pad_w]);
                            outc1r2[i - pad_w] = fmaxf(0.f, outc1r2[i - pad_w]);
                            outc1r3[i - pad_w] = fmaxf(0.f, outc1r3[i - pad_w]);

                            outc0r0[i + win] = fmaxf(0.f, outc0r0[i + win]);
                            outc0r1[i + win] = fmaxf(0.f, outc0r1[i + win]);
                            outc0r2[i + win] = fmaxf(0.f, outc0r2[i + win]);
                            outc0r3[i + win] = fmaxf(0.f, outc0r3[i + win]);
                            outc1r0[i + win] = fmaxf(0.f, outc1r0[i + win]);
                            outc1r1[i + win] = fmaxf(0.f, outc1r1[i + win]);
                            outc1r2[i + win] = fmaxf(0.f, outc1r2[i + win]);
                            outc1r3[i + win] = fmaxf(0.f, outc1r3[i + win]);
                        }
                    }

                    vr10 = vld1q_f32(din_r3);
                    vor00 = vld1q_f32(outc0r0);
                    vor01 = vld1q_f32(outc0r1);

                    vr20 = vld1q_f32(din_r4);
                    vor10 = vld1q_f32(outc1r0);
                    vor11 = vld1q_f32(outc1r1);

                    for (int w = 0; w < win; w += 4) {
                        //! 1st
                        vor02 = vld1q_f32(outc0r2);
                        vor00 = vmlaq_n_f32(vor00, vr10, w03);
                        vor12 = vld1q_f32(outc1r2);
                        vor10 = vmlaq_n_f32(vor10, vr10, w13);
                        vor03 = vld1q_f32(outc0r3);
                        vor01 = vmlaq_n_f32(vor01, vr10, w02);
                        vor13 = vld1q_f32(outc1r3);
                        vor11 = vmlaq_n_f32(vor11, vr10, w12);
                        vr11 = vld1q_f32(din_r2);
                        vor02 = vmlaq_n_f32(vor02, vr10, w01);
                        vor12 = vmlaq_n_f32(vor12, vr10, w11);
                        vor03 = vmlaq_n_f32(vor03, vr10, w00);
                        vor13 = vmlaq_n_f32(vor13, vr10, w10);

                        din_r3 += 4;
                        vr21 = vld1q_f32(din_r5);
                        vor00 = vmlaq_n_f32(vor00, vr20, w04);
                        vor10 = vmlaq_n_f32(vor10, vr20, w14);
                        vor01 = vmlaq_n_f32(vor01, vr20, w03);
                        vor11 = vmlaq_n_f32(vor11, vr20, w13);
                        vor02 = vmlaq_n_f32(vor02, vr20, w02);
                        vor12 = vmlaq_n_f32(vor12, vr20, w12);
                        vor03 = vmlaq_n_f32(vor03, vr20, w01);
                        vor13 = vmlaq_n_f32(vor13, vr20, w11);

                        //! 2nd
                        din_r4 += 4;
                        vr10 = vld1q_f32(din_r1);
                        vor00 = vmlaq_n_f32(vor00, vr11, w02);
                        vor10 = vmlaq_n_f32(vor10, vr11, w12);
                        vor01 = vmlaq_n_f32(vor01, vr11, w01);
                        vor11 = vmlaq_n_f32(vor11, vr11, w11);
                        vor02 = vmlaq_n_f32(vor02, vr11, w00);
                        vor12 = vmlaq_n_f32(vor12, vr11, w10);

                        din_r2 += 4;
                        vr20 = vld1q_f32(din_r6);
                        vor01 = vmlaq_n_f32(vor01, vr21, w04);
                        vor11 = vmlaq_n_f32(vor11, vr21, w14);
                        vor02 = vmlaq_n_f32(vor02, vr21, w03);
                        vor12 = vmlaq_n_f32(vor12, vr21, w13);
                        vor03 = vmlaq_n_f32(vor03, vr21, w02);
                        vor13 = vmlaq_n_f32(vor13, vr21, w12);

                        //! 3rd
                        din_r5 += 4;
                        vr11 = vld1q_f32(din_r0);
                        vor00 = vmlaq_n_f32(vor00, vr10, w01);
                        vor10 = vmlaq_n_f32(vor10, vr10, w11);
                        vor01 = vmlaq_n_f32(vor01, vr10, w00);
                        vor11 = vmlaq_n_f32(vor11, vr10, w10);

                        din_r1 += 4;
                        vr21 = vld1q_f32(din_r7);
                        vor02 = vmlaq_n_f32(vor02, vr20, w04);
                        vor12 = vmlaq_n_f32(vor12, vr20, w14);
                        vor03 = vmlaq_n_f32(vor03, vr20, w03);
                        vor13 = vmlaq_n_f32(vor13, vr20, w13);

                        //! 4th
                        din_r6 += 4;
                        vr10 = vld1q_f32(din_r3);
                        vor00 = vmlaq_n_f32(vor00, vr11, w00);
                        vor10 = vmlaq_n_f32(vor10, vr11, w10);

                        din_r0 += 4;
                        vr20 = vld1q_f32(din_r4);
                        vor03 = vmlaq_n_f32(vor03, vr21, w04);
                        vor13 = vmlaq_n_f32(vor13, vr21, w14);

                        din_r7 += 4;
                        if (relu) {
                            vor00 = vmaxq_f32(vor00, vzero);
                            vor01 = vmaxq_f32(vor01, vzero);
                            vor02 = vmaxq_f32(vor02, vzero);
                            vor03 = vmaxq_f32(vor03, vzero);
                            vor10 = vmaxq_f32(vor10, vzero);
                            vor11 = vmaxq_f32(vor11, vzero);
                            vor12 = vmaxq_f32(vor12, vzero);
                            vor13 = vmaxq_f32(vor13, vzero);
                        }
                        if (w + 4 > win) {
                            vst1q_f32(tmp00, vor00);
                            vst1q_f32(tmp01, vor01);
                            vst1q_f32(tmp02, vor02);
                            vst1q_f32(tmp03, vor03);
                            vst1q_f32(tmp10, vor10);
                            vst1q_f32(tmp11, vor11);
                            vst1q_f32(tmp12, vor12);
                            vst1q_f32(tmp13, vor13);
                            for (int i = w; i < win; ++i) {
                                *outc0r0++ = tmp00[i - w];
                                *outc0r1++ = tmp01[i - w];
                                *outc0r2++ = tmp02[i - w];
                                *outc0r3++ = tmp03[i - w];
                                *outc1r0++ = tmp10[i - w];
                                *outc1r1++ = tmp11[i - w];
                                *outc1r2++ = tmp12[i - w];
                                *outc1r3++ = tmp13[i - w];
                            }
                        } else {
                            vst1q_f32(outc0r0, vor00);
                            vst1q_f32(outc1r0, vor10);
                            vst1q_f32(outc0r1, vor01);
                            vst1q_f32(outc1r1, vor11);
                            outc0r0 += 4;
                            vor00 = vld1q_f32(outc0r0);
                            vst1q_f32(outc0r2, vor02);
                            outc1r0 += 4;
                            vor10 = vld1q_f32(outc1r0);
                            vst1q_f32(outc0r3, vor03);
                            outc0r1 += 4;
                            vor01 = vld1q_f32(outc0r1);
                            vst1q_f32(outc1r2, vor12);
                            outc1r1 += 4;
                            vor11 = vld1q_f32(outc1r1);
                            vst1q_f32(outc1r3, vor13);
                        }
                        outc0r2 += 4;
                        outc0r3 += 4;
                        outc1r2 += 4;
                        outc1r3 += 4;
                    }

                    r0 = r4;
                    r1 = r5;
                    r2 = r6;
                    r3 = r7;
                    r4 = r3 + win;
                    r5 = r4 + win;
                    r6 = r5 + win;
                    r7 = r6 + win;
                }
            }
        }
        if (c_remain > 0) {
            float* dout_c0 = dout_batch + (chout - 1) * size_out_sp;
            const float* w_ptr_o0 = static_cast<const float*>(weights) + (chout - 1) * size_weight_sp;

            if (flag_bias) {
                fill_bias(dout_c0, &bias[chout - 1], 1, size_out_sp);
            } else {
                fill_bias(dout_c0, zero_ptr, 1, size_out_sp);
            }

            float tmp00[4];
            float tmp01[4];
            float tmp02[4];
            float tmp03[4];

            for (int c = 0; c < chin; ++c) {
                const float* din_c = din_batch + c * size_in_sp;
                const float* w_ptr_i0 = w_ptr_o0 + 5 * c;

                int relu = 0;
                if ((c == chin - 1) && flag_relu) {
                    relu = 1;
                }

                //! pad zero to top and bottom row
                // q0 = w0, w1, w2, w3; q1 = w4
                float w00 = w_ptr_i0[0];
                float w01 = w_ptr_i0[1];
                float w02 = w_ptr_i0[2];
                float w03 = w_ptr_i0[3];
                float w04 = w_ptr_i0[4];

                const float* din_r0 = nullptr;
                const float* din_r1 = nullptr;
                const float* din_r2 = nullptr;
                const float* din_r3 = nullptr;
                const float* din_r4 = nullptr;
                const float* din_r5 = nullptr;
                const float* din_r6 = nullptr;
                const float* din_r7 = nullptr;

                float* outc0r0 = nullptr;
                float* outc0r1 = nullptr;
                float* outc0r2 = nullptr;
                float* outc0r3 = nullptr;

                float32x4_t vzero = vdupq_n_f32(0.f);

                float32x4_t vr10;
                float32x4_t vr20;
                float32x4_t vr11;
                float32x4_t vr21;

                float32x4_t vor00;
                float32x4_t vor01;
                float32x4_t vor02;
                float32x4_t vor03;

                //! process pad which do nothing
                int h = (pad_h - 4) > 0? pad_h - 4 : 0;

                const float* r0 = din_c - (pad_h - h) * win;
                const float* r1 = r0 + win;
                const float* r2 = r1 + win;
                const float* r3 = r2 + win;
                const float* r4 = r3 + win;
                const float* r5 = r4 + win;
                const float* r6 = r5 + win;
                const float* r7 = r6 + win;

                for (; h < hout; h += 4) {

                    din_r0 = r0;
                    din_r1 = r1;
                    din_r2 = r2;
                    din_r3 = r3;
                    din_r4 = r4;
                    din_r5 = r5;
                    din_r6 = r6;
                    din_r7 = r7;

                    outc0r0 = dout_c0 + h * wout + pad_w;
                    outc0r1 = outc0r0 + wout;
                    outc0r2 = outc0r1 + wout;
                    outc0r3 = outc0r2 + wout;

                    //! process top pad
                    if (h - pad_h < 0) {
                        switch (pad_h - h) {
                            case 4:
                                din_r3 = zero_ptr;
                            case 3:
                                din_r2 = zero_ptr;
                            case 2:
                                din_r1 = zero_ptr;
                            case 1:
                                din_r0 = zero_ptr;
                            default:
                                break;
                        }
                    }

                    //! process bottom pad
                    if (h + 8 - pad_h > hin) {
                        switch (h + 8 - pad_h - hin) {
                            case 7:
                                din_r1 = zero_ptr;
                            case 6:
                                din_r2 = zero_ptr;
                            case 5:
                                din_r3 = zero_ptr;
                            case 4:
                                din_r4 = zero_ptr;
                            case 3:
                                din_r5 = zero_ptr;
                            case 2:
                                din_r6 = zero_ptr;
                            case 1:
                                din_r7 = zero_ptr;
                            default:
                                break;
                        }
                    }

                    //! process bottom remain
                    if (h + 4 > hin) {
                        switch (h + 4 - hin) {
                            case 3:
                                outc0r1 = write_ptr;
                            case 2:
                                outc0r2 = write_ptr;
                            case 1:
                                outc0r3 = write_ptr;
                            default:
                                break;
                        }
                    }

                    if (relu) {
                        for (int i = 0; i < pad_w; ++i) {
                            outc0r0[i - pad_w] = fmaxf(0.f, outc0r0[i - pad_w]);
                            outc0r1[i - pad_w] = fmaxf(0.f, outc0r1[i - pad_w]);
                            outc0r2[i - pad_w] = fmaxf(0.f, outc0r2[i - pad_w]);
                            outc0r3[i - pad_w] = fmaxf(0.f, outc0r3[i - pad_w]);

                            outc0r0[i + win] = fmaxf(0.f, outc0r0[i + win]);
                            outc0r1[i + win] = fmaxf(0.f, outc0r1[i + win]);
                            outc0r2[i + win] = fmaxf(0.f, outc0r2[i + win]);
                            outc0r3[i + win] = fmaxf(0.f, outc0r3[i + win]);
                        }
                    }

                    vr10 = vld1q_f32(din_r3);
                    vor00 = vld1q_f32(outc0r0);
                    vor01 = vld1q_f32(outc0r1);
                    vr20 = vld1q_f32(din_r4);

                    for (int w = 0; w < win; w += 4) {
                        //! 1st
                        vor02 = vld1q_f32(outc0r2);
                        vor00 = vmlaq_n_f32(vor00, vr10, w03);

                        vor03 = vld1q_f32(outc0r3);
                        vor01 = vmlaq_n_f32(vor01, vr10, w02);

                        vr11 = vld1q_f32(din_r2);
                        vor02 = vmlaq_n_f32(vor02, vr10, w01);
                        vor03 = vmlaq_n_f32(vor03, vr10, w00);

                        din_r3 += 4;
                        vr21 = vld1q_f32(din_r5);
                        vor00 = vmlaq_n_f32(vor00, vr20, w04);
                        vor01 = vmlaq_n_f32(vor01, vr20, w03);
                        vor02 = vmlaq_n_f32(vor02, vr20, w02);
                        vor03 = vmlaq_n_f32(vor03, vr20, w01);

                        //! 2nd
                        din_r4 += 4;
                        vr10 = vld1q_f32(din_r1);
                        vor00 = vmlaq_n_f32(vor00, vr11, w02);
                        vor01 = vmlaq_n_f32(vor01, vr11, w01);
                        vor02 = vmlaq_n_f32(vor02, vr11, w00);

                        din_r2 += 4;
                        vr20 = vld1q_f32(din_r6);
                        vor01 = vmlaq_n_f32(vor01, vr21, w04);
                        vor02 = vmlaq_n_f32(vor02, vr21, w03);
                        vor03 = vmlaq_n_f32(vor03, vr21, w02);

                        //! 3rd
                        din_r5 += 4;
                        vr11 = vld1q_f32(din_r0);
                        vor00 = vmlaq_n_f32(vor00, vr10, w01);
                        vor01 = vmlaq_n_f32(vor01, vr10, w00);

                        din_r1 += 4;
                        vr21 = vld1q_f32(din_r7);
                        vor02 = vmlaq_n_f32(vor02, vr20, w04);
                        vor03 = vmlaq_n_f32(vor03, vr20, w03);

                        //! 4th
                        din_r6 += 4;
                        vr10 = vld1q_f32(din_r3);
                        vor00 = vmlaq_n_f32(vor00, vr11, w00);

                        din_r0 += 4;
                        vr20 = vld1q_f32(din_r4);
                        vor03 = vmlaq_n_f32(vor03, vr21, w04);

                        din_r7 += 4;
                        if (relu) {
                            vor00 = vmaxq_f32(vor00, vzero);
                            vor01 = vmaxq_f32(vor01, vzero);
                            vor02 = vmaxq_f32(vor02, vzero);
                            vor03 = vmaxq_f32(vor03, vzero);
                        }
                        if (w + 4 > win) {
                            vst1q_f32(tmp00, vor00);
                            vst1q_f32(tmp01, vor01);
                            vst1q_f32(tmp02, vor02);
                            vst1q_f32(tmp03, vor03);
                            for (int i = w; i < win; ++i) {
                                *outc0r0++ = tmp00[i - w];
                                *outc0r1++ = tmp01[i - w];
                                *outc0r2++ = tmp02[i - w];
                                *outc0r3++ = tmp03[i - w];
                            }
                        } else {
                            vst1q_f32(outc0r0, vor00);
                            vst1q_f32(outc0r1, vor01);
                            outc0r0 += 4;
                            vor00 = vld1q_f32(outc0r0);
                            vst1q_f32(outc0r2, vor02);
                            vst1q_f32(outc0r3, vor03);
                            outc0r1 += 4;
                            vor01 = vld1q_f32(outc0r1);
                        }
                        outc0r2 += 4;
                        outc0r3 += 4;
                    }

                    r0 = r4;
                    r1 = r5;
                    r2 = r6;
                    r3 = r7;
                    r4 = r3 + win;
                    r5 = r4 + win;
                    r6 = r5 + win;
                    r7 = r6 + win;
                }
            }
        }
    }

}

} //namespace lite

} //namespace saber

} //namespace anakin

#endif
