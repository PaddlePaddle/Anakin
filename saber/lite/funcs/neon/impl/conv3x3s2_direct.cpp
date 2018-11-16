#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"
#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/arm_utils.h"

namespace anakin{

namespace saber{

namespace lite{

#define AKMAX(a, b) (a) > (b)? (a) : (b)

#ifdef __aarch64__

void conv3x3s2_trans_weights4c(void* dout, const void* din, int chout, int chin) {
    int c_loop = chout / 4;
    int chout_round = (chout + 3) / 4;
    int win_stride = chin * 9;
    int wout_stride = 4 * win_stride;
    int co = 0;
    for (; co < c_loop; ++co) {
        float* dout_c = static_cast<float*>(dout) + co * wout_stride;
        const float* din_c0 = static_cast<const float*>(din) + co * 4 * win_stride;
        const float* din_c1 = din_c0 + win_stride;
        const float* din_c2 = din_c1 + win_stride;
        const float* din_c3 = din_c2 + win_stride;
        for (int ci = 0; ci < chin; ++ci) {
            for (int k = 0; k < 9; ++k) {
                *(dout_c++) = *(din_c0++);
                *(dout_c++) = *(din_c1++);
                *(dout_c++) = *(din_c2++);
                *(dout_c++) = *(din_c3++);
            }
        }
    }
    // pad final chout
    if (chout_round > c_loop) {
        float* dout_c = static_cast<float*>(dout) + c_loop * wout_stride;
        const float* din_c0 = static_cast<const float*>(din) + co * 4 * win_stride;
        const float* din_c1 = din_c0 + win_stride;
        const float* din_c2 = din_c1 + win_stride;
        const float* din_c3 = din_c2 + win_stride;
        switch (chout_round * 4 - chout) {
            case 3:
                din_c1 = din_c0;
            case 2:
                din_c2 = din_c0;
            case 1:
                din_c3 = din_c0;
            default:
                break;
        }
        for (int ci = 0; ci < chin; ++ci) {
            for (int k = 0; k < 9; ++k) {
                *(dout_c++) = *(din_c0++);
                *(dout_c++) = *(din_c1++);
                *(dout_c++) = *(din_c2++);
                *(dout_c++) = *(din_c3++);
            }
        }
    }
}

void prepack_input5xw(const float* din, float* dout, int cs, int ce, int hs, int he, int ws, int we, \
    int channel, int width, int height, float* zero_ptr) {

    int w0 = ws < 0? 0 : ws;
    int w1 = we > width? width : we;

    int size_w = we - ws;
    int size_wc_len = size_w * channel;
    int size_c = width * height;

    int valid_w = w1 - w0;
    size_t valid_w_byte = valid_w * sizeof(float);

    float* r0 = dout;
    float* r1 = r0 + size_wc_len;
    float* r2 = r1 + size_wc_len;
    float* r3 = r2 + size_wc_len;
    float* r4 = r3 + size_wc_len;

    for (int c = 0; c < channel; ++c) {
        const float* inr0 = din + hs * width;
        const float* inr1 = inr0 + width;
        const float* inr2 = inr1 + width;
        const float* inr3 = inr2 + width;
        const float* inr4 = inr3 + width;
        if (hs < -4) {
            inr0 = zero_ptr;
            inr1 = zero_ptr;
            inr2 = zero_ptr;
            inr3 = zero_ptr;
            inr4 = zero_ptr;
        }
        if (hs < 0) {
            switch (hs) {
                case -4:
                    inr3 = zero_ptr;
                case -3:
                    inr2 = zero_ptr;
                case -2:
                    inr1 = zero_ptr;
                case -1:
                    inr0 = zero_ptr;
                default:
                    break;
            }
        }
        if (he >= height + 5) {
            inr0 = zero_ptr;
            inr1 = zero_ptr;
            inr2 = zero_ptr;
            inr3 = zero_ptr;
            inr4 = zero_ptr;
        }
        if (he > height) {
            switch (he - height) {
                case 4:
                    inr1 = zero_ptr;
                case 3:
                    inr2 = zero_ptr;
                case 2:
                    inr3 = zero_ptr;
                case 1:
                    inr4 = zero_ptr;
                default:
                    break;
            }
        }

        for (int w = ws; w < w0; ++w) {
            *(r0++) = 0.f;
            *(r1++) = 0.f;
            *(r2++) = 0.f;
            *(r3++) = 0.f;
            *(r4++) = 0.f;
        }
        memcpy(r0, inr0, valid_w_byte);
        memcpy(r1, inr1, valid_w_byte);
        memcpy(r2, inr2, valid_w_byte);
        memcpy(r3, inr3, valid_w_byte);
        memcpy(r4, inr4, valid_w_byte);
        r0 += valid_w;
        r1 += valid_w;
        r2 += valid_w;
        r3 += valid_w;
        r4 += valid_w;

        for (int w = w1; w < we; ++w) {
            *(r0++) = 0.f;
            *(r1++) = 0.f;
            *(r2++) = 0.f;
            *(r3++) = 0.f;
            *(r4++) = 0.f;
        }
        din += size_c;
    }

}

void write_to_output4xw(const float* din, float* dout, int cs, int ce, int hs, int he, int ws, int we, \
    int channel, int height, int width, bool flag_relu, float* trash_ptr) {

    int size_c_out = width * height;

    float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
    float* doutc0r1 = doutc0r0 + width;
    float* doutc1r0 = doutc0r0 + size_c_out;
    float* doutc1r1 = doutc1r0 + width;
    float* doutc2r0 = doutc1r0 + size_c_out;
    float* doutc2r1 = doutc2r0 + width;
    float* doutc3r0 = doutc2r0 + size_c_out;
    float* doutc3r1 = doutc3r0 + width;

    const float* ptr_din = din;

    if (ce > channel) {
        switch (ce - channel) {
            case 3:
                doutc1r0 = trash_ptr;
                doutc1r1 = trash_ptr;
            case 2:
                doutc2r0 = trash_ptr;
                doutc2r1 = trash_ptr;
            case 1:
                doutc3r0 = trash_ptr;
                doutc3r1 = trash_ptr;
            default:
                break;
        }
    }
    if (he > height) {
        doutc0r1 = trash_ptr;
        doutc1r1 = trash_ptr;
        doutc2r1 = trash_ptr;
        doutc3r1 = trash_ptr;
    }

    int cnt = (we - ws) / 4;

    if (we > width) {
        cnt--;
    }
    if (flag_relu) {
        if (cnt > 0) {
            asm volatile(
                "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                "ldp q4, q5, [%[ptr_din]], #32  \n"         /* load r10, r11 to q4, q5 */
                "ldp q6, q7, [%[ptr_din]], #32  \n"         /* load r12, r13 to q6, q7 */
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

                "trn1   v12.4s, v4.4s, v5.4s    \n"         /* trans q4, q5*/
                "trn2   v13.4s, v4.4s, v5.4s    \n"         /* trans q4, q5*/
                "ldp q4, q5, [%[ptr_din]], #32  \n"         /* load r10, r11 to q4, q5 */
                "trn1   v14.4s, v6.4s, v7.4s    \n"         /* trans q6, q7*/
                "trn2   v15.4s, v6.4s, v7.4s    \n"         /* trans q6, q7*/
                "ldp q6, q7, [%[ptr_din]], #32  \n"         /* load r12, r13 to q6, q7 */
                "trn1   v16.2d, v12.2d, v14.2d  \n"         /* trans q12, q14*/
                "trn2   v17.2d, v12.2d, v14.2d  \n"         /* trans q12, q14*/
                "trn1   v18.2d, v13.2d, v15.2d  \n"         /* trans q13, q15*/
                "trn2   v19.2d, v13.2d, v15.2d  \n"         /* trans q13, q15*/
                "fmax   v16.4s, v16.4s, v20.4s  \n"         /*relu*/
                "fmax   v17.4s, v17.4s, v20.4s  \n"         /*relu*/
                "fmax   v18.4s, v18.4s, v20.4s  \n"         /*relu*/
                "fmax   v19.4s, v19.4s, v20.4s  \n"         /*relu*/
                "str    q16, [%[doutc0r1]], #16 \n"         /* store c0r1*/
                "str    q17, [%[doutc2r1]], #16 \n"         /* store c2r1*/
                "str    q18, [%[doutc1r1]], #16 \n"         /* store c1r1*/
                "str    q19, [%[doutc3r1]], #16 \n"         /* store c3r1*/
                "bne    1b                      \n"         /* jump to main loop*/

            : [doutc0r0]"+r"(doutc0r0), [doutc0r1]"+r"(doutc0r1), [doutc1r0]"+r"(doutc1r0), \
            [doutc1r1]"+r"(doutc1r1), [doutc2r0]"+r"(doutc2r0), [doutc2r1]"+r"(doutc2r1), \
            [doutc3r0]"+r"(doutc3r0), [doutc3r1]"+r"(doutc3r1), [cnt] "+r"(cnt), [ptr_din]"+r"(ptr_din)
            :
            : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", \
            "v14", "v15", "v16", "v17", "v18", "v19", "v20"
            );
        }
        if (we > width) {
            int offset = 32 * ((we - ws) / 4 - 1);
            ptr_din  = din + offset;
            int i = we - 4;
            for (; i < width; ++i) {
                *(doutc0r0++) = AKMAX(ptr_din[0], 0.f);
                *(doutc1r0++) = AKMAX(ptr_din[1], 0.f);
                *(doutc2r0++) = AKMAX(ptr_din[2], 0.f);
                *(doutc3r0++) = AKMAX(ptr_din[3], 0.f);

                *(doutc0r1++) = AKMAX(ptr_din[16], 0.f);
                *(doutc1r1++) = AKMAX(ptr_din[17], 0.f);
                *(doutc2r1++) = AKMAX(ptr_din[18], 0.f);
                *(doutc3r1++) = AKMAX(ptr_din[19], 0.f);
                ptr_din += 4;
            }
        }
    } else {
        if (cnt > 0) {
            asm volatile(
                "ldp q0, q1, [%[ptr_din]], #32  \n"         /* load r00, r01 to q0, q1 */
                "ldp q2, q3, [%[ptr_din]], #32  \n"         /* load r02, r03 to q2, q3 */
                "ldp q4, q5, [%[ptr_din]], #32  \n"         /* load r10, r11 to q4, q5 */
                "ldp q6, q7, [%[ptr_din]], #32  \n"         /* load r12, r13 to q6, q7 */
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

                "trn1   v12.4s, v4.4s, v5.4s    \n"         /* trans q4, q5*/
                "trn2   v13.4s, v4.4s, v5.4s    \n"         /* trans q4, q5*/
                "ldp q4, q5, [%[ptr_din]], #32  \n"         /* load r10, r11 to q4, q5 */
                "trn1   v14.4s, v6.4s, v7.4s    \n"         /* trans q6, q7*/
                "trn2   v15.4s, v6.4s, v7.4s    \n"         /* trans q6, q7*/
                "ldp q6, q7, [%[ptr_din]], #32  \n"         /* load r12, r13 to q6, q7 */
                "trn1   v16.2d, v12.2d, v14.2d  \n"         /* trans q12, q14*/
                "trn2   v17.2d, v12.2d, v14.2d  \n"         /* trans q12, q14*/
                "trn1   v18.2d, v13.2d, v15.2d  \n"         /* trans q13, q15*/
                "trn2   v19.2d, v13.2d, v15.2d  \n"         /* trans q13, q15*/
                "str    q16, [%[doutc0r1]], #16 \n"         /* store c0r1*/
                "str    q17, [%[doutc2r1]], #16 \n"         /* store c2r1*/
                "str    q18, [%[doutc1r1]], #16 \n"         /* store c1r1*/
                "str    q19, [%[doutc3r1]], #16 \n"         /* store c3r1*/
                "bne    1b                      \n"         /* jump to main loop*/

            : [doutc0r0]"+r"(doutc0r0), [doutc0r1]"+r"(doutc0r1), [doutc1r0]"+r"(doutc1r0), \
            [doutc1r1]"+r"(doutc1r1), [doutc2r0]"+r"(doutc2r0), [doutc2r1]"+r"(doutc2r1), \
            [doutc3r0]"+r"(doutc3r0), [doutc3r1]"+r"(doutc3r1), [cnt] "+r"(cnt), [ptr_din]"+r"(ptr_din)
            :
            : "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", \
            "v14", "v15", "v16", "v17", "v18", "v19"
            );
        }
        if (we > width) {
            int offset = 32 * ((we - ws) / 4 - 1);
            ptr_din  = din + offset;
            int i = we - 4;
            for (; i < width; ++i) {
                *(doutc0r0++) = *(ptr_din++);
                *(doutc1r0++) = *(ptr_din++);
                *(doutc2r0++) = *(ptr_din++);
                *(doutc3r0++) = *(ptr_din++);

                *(doutc0r1++) = *(ptr_din + 12);
                *(doutc1r1++) = *(ptr_din + 13);
                *(doutc2r1++) = *(ptr_din + 14);
                *(doutc3r1++) = *(ptr_din + 15);
            }
        }
    }

}

void fill_packed_bias4x2w(float* dout, const float* bias, int wround) {
    float32x4_t vb = vld1q_f32(bias);
    int cnt = wround / 4;
    asm volatile(
    "1:                                         \n"     /* main loop*/
            "and v0.16b, %[vb].16b, %[vb].16b   \n"     /*fill bias*/
            "and v1.16b, %[vb].16b, %[vb].16b   \n"     /*fill bias*/
            "and v2.16b, %[vb].16b, %[vb].16b   \n"     /*fill bias*/
            "and v3.16b, %[vb].16b, %[vb].16b   \n"     /*fill bias*/
            "stp q0, q1, [%[dout]], #32         \n"     /*write back*/
            "and v4.16b, %[vb].16b, %[vb].16b   \n"     /*fill bias*/
            "stp q2, q3, [%[dout]], #32         \n"     /*write back*/
            "and v5.16b, %[vb].16b, %[vb].16b   \n"     /*fill bias*/
            "and v6.16b, %[vb].16b, %[vb].16b   \n"     /*fill bias*/
            "and v7.16b, %[vb].16b, %[vb].16b   \n"     /*fill bias*/
            "stp q4, q5, [%[dout]], #32         \n"     /*write back*/
            "stp q6, q7, [%[dout]], #32         \n"     /*write back*/
            "subs   %w[cnt], %w[cnt], #1        \n"     /*loop count -1*/
            "bne    1b                          \n"     /* jump to main loop*/

    : [dout] "+r"(dout), [cnt] "+r" (cnt)
    : [vb] "w"(vb)
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
    );
}

void conv_3x3s2_direct(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, \
                          Context* ctx, void* work_space, const void* idx_ptr) {
    //! 3x3s2 convolution, implemented by direct algorithm

    //! prepack input to tmp buffer
    //! write output to tmp buffer

    const int hin_r_block = 5;
    const int hout_c_block = 4;
    const int hout_r_block = 2;

    int wout_round = ((wout + 3) / 4) * 4;
    int win_round = wout_round * stride_w + 1;

    int threads = ctx->get_threads();

    float* tmp_work_space = static_cast<float*>(ctx->get_work_space());
    float* ptr_zero = tmp_work_space;
    memset(ptr_zero, 0, sizeof(float) * win_round);
    float* ptr_write = ptr_zero + win_round;

    int in_len = win_round * chin;
    int pre_in_size = hin_r_block * in_len;
    int pre_out_size = hout_c_block * hout_r_block * wout_round;

    float* pre_din = ptr_write + wout_round;

    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    int w_stride = chin * 9;

    int ws = -pad_w;
    int we = ws + win_round;
    int w_loop = wout_round / 4;

    for (int n = 0; n < num; ++n) {
        const float *din_batch = static_cast<const float*>(din) + n * chin * size_in_channel;
        float *dout_batch = static_cast<float*>(dout) + n * chout * size_out_channel;
        for (int h = 0; h < hout; h += 2) {
            int hs = h * 2 - pad_h;
            int he = hs + 5;
            prepack_input5xw(din_batch, pre_din, 0, chin, hs, he, ws, we, chin, win, hin, ptr_zero);

#pragma omp parallel for num_threads(threads)
            for (int c = 0; c < chout; c += 4) {

#ifdef USE_OPENMP
                float* pre_out = pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
                float* pre_out = pre_din + pre_in_size;
#endif
                const float* inr0 = pre_din;
                const float* inr1 = inr0 + in_len;
                const float* inr2 = inr1 + in_len;
                const float* inr3 = inr2 + in_len;
                const float* inr4 = inr3 + in_len;

                const float* wc0 = static_cast<const float*>(weights) + c * w_stride;

                const float* bias_ptr = ptr_zero;
                if (flag_bias) {
                    bias_ptr = static_cast<const float*>(bias) + c;
                }
                fill_packed_bias4x2w(pre_out, bias_ptr, wout_round);

                for (int i = 0; i < chin; ++i) {

                    float32x4_t v0 = vld1q_f32(wc0); //w0, v23
                    float32x4_t v1 = vld1q_f32(wc0 + 4);//w1, v24
                    float32x4_t v2 = vld1q_f32(wc0 + 8);//w2, v25

                    float32x4_t v3 = vld1q_f32(wc0 + 12);//w3, v26
                    float32x4_t v4 = vld1q_f32(wc0 + 16);//w4, v27
                    float32x4_t v5 = vld1q_f32(wc0 + 20);//w5, v28

                    float32x4_t v6 = vld1q_f32(wc0 + 24);//w6, v29
                    float32x4_t v7 = vld1q_f32(wc0 + 28);//w7, v30
                    float32x4_t v8 = vld1q_f32(wc0 + 32);//w8, v31

                    const float* r0 = inr0;
                    const float* r1 = inr1;
                    const float* r2 = inr2;
                    const float* r3 = inr3;
                    const float* r4 = inr4;

                    float* ptr_out = pre_out;

                    int cnt = w_loop;
                    asm volatile(
                    "ldp	q15, q16, [%[ptr_out]]              \n"         /* load outr00, outr01*/
                            "ldp	q17, q18, [%[ptr_out], #32] \n"         /* load outr02, outr03*/
                            "ldp	q19, q20, [%[ptr_out], #64] \n"         /* load outr10, outr11*/
                            "ldp	q21, q22, [%[ptr_out], #96] \n"         /* load outr12, outr13*/
                            "ldp    q0, q1,   [%[r0]], #32      \n"         /* load input r0*/
                            "ldp    q2, q3,   [%[r2]], #32      \n"         /* load input r1*/
                            "2:                                 \n"         /* main loop*/
                            /*  r0, r2, mul w00 */
                            "fmla 	v15.4s ,  %[v0].4s,  v0.s[0]\n"         /* outr00 = v0 * r0[0]*/
                            "fmla 	v16.4s ,  %[v0].4s,  v0.s[2]\n"         /* outr01 = v0 * r0[2]*/
                            "fmla 	v17.4s ,  %[v0].4s,  v1.s[0]\n"         /* outr02 = v0 * r0[4]*/
                            "fmla 	v18.4s ,  %[v0].4s,  v1.s[2]\n"         /* outr03 = v0 * r0[6]*/
                            "fmla 	v19.4s ,  %[v0].4s,  v2.s[0]\n"         /* outr10 = v0 * r2[0]*/
                            "fmla 	v20.4s ,  %[v0].4s,  v2.s[2]\n"         /* outr11 = v0 * r2[2]*/
                            "fmla 	v21.4s ,  %[v0].4s,  v3.s[0]\n"         /* outr12 = v0 * r2[4]*/
                            "fmla 	v22.4s ,  %[v0].4s,  v3.s[2]\n"         /* outr13 = v0 * r2[6]*/

                            "ldr    d6,      [%[r0]]            \n"         /* load r0, 9th data,v6.s[0]*/

                            /*  r0, r2, mul w01 */
                            "fmla 	v15.4s ,  %[v1].4s,  v0.s[1]\n"         /* outr00 = v1 * r0[1]*/
                            "fmla 	v16.4s ,  %[v1].4s,  v0.s[3]\n"         /* outr01 = v1 * r0[3]*/
                            "fmla 	v17.4s ,  %[v1].4s,  v1.s[1]\n"         /* outr02 = v1 * r0[5]*/
                            "fmla 	v18.4s ,  %[v1].4s,  v1.s[3]\n"         /* outr03 = v1 * r0[7]*/
                            "fmla 	v19.4s ,  %[v1].4s,  v2.s[1]\n"         /* outr10 = v1 * r2[1]*/
                            "fmla 	v20.4s ,  %[v1].4s,  v2.s[3]\n"         /* outr11 = v1 * r2[3]*/
                            "fmla 	v21.4s ,  %[v1].4s,  v3.s[1]\n"         /* outr12 = v1 * r2[5]*/
                            "fmla 	v22.4s ,  %[v1].4s,  v3.s[3]\n"         /* outr13 = v1 * r2[7]*/

                            "ldr    d7,      [%[r2]]            \n"         /* load r2, 9th data,v7.s[0]*/

                            /*  r0, r2, mul w02 */
                            "fmla 	v15.4s ,  %[v2].4s,  v0.s[2]\n"         /* outr00 = v2 * r0[2]*/
                            "fmla 	v16.4s ,  %[v2].4s,  v1.s[0]\n"         /* outr01 = v2 * r0[4]*/
                            "fmla 	v17.4s ,  %[v2].4s,  v1.s[2]\n"         /* outr02 = v2 * r0[6]*/
                            "fmla 	v18.4s ,  %[v2].4s,  v6.s[0]\n"         /* outr03 = v2 * r0[8]*/
                            "fmla 	v19.4s ,  %[v2].4s,  v2.s[2]\n"         /* outr10 = v2 * r2[2]*/
                            "fmla 	v20.4s ,  %[v2].4s,  v3.s[0]\n"         /* outr11 = v2 * r2[4]*/
                            "fmla 	v21.4s ,  %[v2].4s,  v3.s[2]\n"         /* outr12 = v2 * r2[6]*/
                            "fmla 	v22.4s ,  %[v2].4s,  v7.s[0]\n"         /* outr13 = v2 * r2[8]*/

                            /* r2, mul w08 */
                            "fmla 	v15.4s ,  %[v8].4s,  v2.s[2]\n"         /* outr00 = v8 * r2[2]*/
                            "fmla 	v16.4s ,  %[v8].4s,  v3.s[0]\n"         /* outr01 = v8 * r2[4]*/
                            "fmla 	v17.4s ,  %[v8].4s,  v3.s[2]\n"         /* outr02 = v8 * r2[6]*/
                            "fmla 	v18.4s ,  %[v8].4s,  v7.s[0]\n"         /* outr03 = v8 * r2[8]*/

                            "ldp    q4, q5,   [%[r1]], #32\n"               /* load input r1*/
                            /* r2, mul w07 */
                            "fmla 	v15.4s ,  %[v6].4s,  v2.s[0]\n"         /* outr00 = v6 * r2[0]*/
                            "fmla 	v16.4s ,  %[v6].4s,  v2.s[2]\n"         /* outr01 = v6 * r2[2]*/
                            "fmla 	v17.4s ,  %[v6].4s,  v3.s[0]\n"         /* outr02 = v6 * r2[4]*/
                            "fmla 	v18.4s ,  %[v6].4s,  v3.s[2]\n"         /* outr03 = v6 * r2[6]*/

                            "ldp    q6, q7,   [%[r3]], #32\n"               /* load input r3*/
                            /* r2, mul w06 */
                            "fmla 	v15.4s ,  %[v7].4s,  v2.s[1]\n"         /* outr00 = v7 * r2[1]*/
                            "fmla 	v16.4s ,  %[v7].4s,  v2.s[3]\n"         /* outr01 = v7 * r2[3]*/
                            "fmla 	v17.4s ,  %[v7].4s,  v3.s[1]\n"         /* outr02 = v7 * r2[5]*/
                            "fmla 	v18.4s ,  %[v7].4s,  v3.s[3]\n"         /* outr03 = v7 * r2[7]*/

                            "ldr    d0,       [%[r1]]           \n"         /* load r1, 9th data,v0.s[0]*/
                            "ldr    d1,       [%[r3]]           \n"         /* load r3, 9th data,v1.s[0]*/

                            /*  r1, r3, mul w05 */
                            "fmla 	v15.4s ,  %[v5].4s,  v4.s[2]\n"         /* outr00 = v5 * r1[2]*/
                            "fmla 	v16.4s ,  %[v5].4s,  v5.s[0]\n"         /* outr01 = v5 * r1[4]*/
                            "fmla 	v17.4s ,  %[v5].4s,  v5.s[2]\n"         /* outr02 = v5 * r1[6]*/
                            "fmla 	v18.4s ,  %[v5].4s,  v0.s[0]\n"         /* outr03 = v5 * r1[8]*/
                            "fmla 	v19.4s ,  %[v5].4s,  v6.s[2]\n"         /* outr10 = v5 * r3[2]*/
                            "fmla 	v20.4s ,  %[v5].4s,  v7.s[0]\n"         /* outr11 = v5 * r3[4]*/
                            "fmla 	v21.4s ,  %[v5].4s,  v7.s[2]\n"         /* outr12 = v5 * r3[6]*/
                            "fmla 	v22.4s ,  %[v5].4s,  v1.s[0]\n"         /* outr13 = v5 * r3[8]*/

                            "ldp    q2, q3,   [%[r4]], #32      \n"         /* load input r4*/
                            /*  r1, r3, mul w03 */
                            "fmla 	v15.4s ,  %[v3].4s,  v4.s[0]\n"         /* outr00 = v3 * r1[0]*/
                            "fmla 	v16.4s ,  %[v3].4s,  v4.s[2]\n"         /* outr01 = v3 * r1[2]*/
                            "fmla 	v17.4s ,  %[v3].4s,  v5.s[0]\n"         /* outr02 = v3 * r1[4]*/
                            "fmla 	v18.4s ,  %[v3].4s,  v5.s[2]\n"         /* outr03 = v3 * r1[6]*/
                            "fmla 	v19.4s ,  %[v3].4s,  v6.s[0]\n"         /* outr10 = v3 * r3[0]*/
                            "fmla 	v20.4s ,  %[v3].4s,  v6.s[2]\n"         /* outr11 = v3 * r3[2]*/
                            "fmla 	v21.4s ,  %[v3].4s,  v7.s[0]\n"         /* outr12 = v3 * r3[4]*/
                            "fmla 	v22.4s ,  %[v3].4s,  v7.s[2]\n"         /* outr13 = v3 * r3[6]*/

                            "subs   %w[cnt], %w[cnt], #1        \n"         /* loop count -1*/

                            "ldr    d0,      [%[r4]]            \n"         /* load r4, 9th data,v0.s[0]*/
                            /*  r1, r3, mul w04 */
                            "fmla 	v15.4s ,  %[v4].4s,  v4.s[1]\n"         /* outr00 = v4 * r1[1]*/
                            "fmla 	v16.4s ,  %[v4].4s,  v4.s[3]\n"         /* outr01 = v4 * r1[3]*/
                            "fmla 	v17.4s ,  %[v4].4s,  v5.s[1]\n"         /* outr02 = v4 * r1[5]*/
                            "fmla 	v18.4s ,  %[v4].4s,  v5.s[3]\n"         /* outr03 = v4 * r1[7]*/
                            "fmla 	v19.4s ,  %[v4].4s,  v6.s[1]\n"         /* outr10 = v4 * r3[1]*/
                            "fmla 	v20.4s ,  %[v4].4s,  v6.s[3]\n"         /* outr11 = v4 * r3[3]*/
                            "fmla 	v21.4s ,  %[v4].4s,  v7.s[1]\n"         /* outr12 = v4 * r3[5]*/
                            "fmla 	v22.4s ,  %[v4].4s,  v7.s[3]\n"         /* outr13 = v4 * r3[7]*/

                            "stp    q15, q16, [%[ptr_out]], #32 \n"         /* save to output*/

                            /* r4, mul w08 */
                            "fmla 	v21.4s ,  %[v8].4s,  v3.s[2]\n"         /* outr02 = v8 * r2[6]*/
                            "fmla 	v22.4s ,  %[v8].4s,  v0.s[0]\n"         /* outr03 = v8 * r2[8]*/

                            "ldp    q0, q1,   [%[r0]], #32      \n"         /* load input r0*/

                            "fmla 	v19.4s ,  %[v8].4s,  v2.s[2]\n"         /* outr00 = v8 * r2[2]*/
                            "fmla 	v20.4s ,  %[v8].4s,  v3.s[0]\n"         /* outr01 = v8 * r2[4]*/

                            "ldp    q15, q16, [%[ptr_out], #96] \n"         /* load next output to q15,q16*/
                            /* r4, mul w07 */
                            "fmla 	v19.4s ,  %[v6].4s,  v2.s[0]\n"         /* outr00 = v6 * r2[0]*/
                            "fmla 	v20.4s ,  %[v6].4s,  v2.s[2]\n"         /* outr01 = v6 * r2[2]*/

                            "stp    q17, q18, [%[ptr_out]], #32 \n"         /* save to output*/

                            "fmla 	v21.4s ,  %[v6].4s,  v3.s[0]\n"         /* outr02 = v6 * r2[4]*/
                            "fmla 	v22.4s ,  %[v6].4s,  v3.s[2]\n"         /* outr03 = v6 * r2[6]*/

                            /* r4, mul w06 */
                            "fmla 	v19.4s ,  %[v7].4s,  v2.s[1]\n"         /* outr00 = v7 * r2[1]*/
                            "fmla 	v20.4s ,  %[v7].4s,  v2.s[3]\n"         /* outr01 = v7 * r2[3]*/

                            "ldp    q17, q18, [%[ptr_out], #96] \n"         /* load next output to q17,q18*/

                            "fmla 	v21.4s ,  %[v7].4s,  v3.s[1]\n"         /* outr02 = v7 * r2[5]*/
                            "fmla 	v22.4s ,  %[v7].4s,  v3.s[3]\n"         /* outr03 = v7 * r2[7]*/

                            "ldp    q2, q3,   [%[r2]], #32      \n"         /* load input r2*/

                            "stp    q19, q20, [%[ptr_out]], #32 \n"         /* save to output*/
                            "stp    q21, q22, [%[ptr_out]], #32 \n"         /* save to output*/

                            "ldp	q19, q20, [%[ptr_out], #64] \n"         /* load outr10, outr11*/
                            "ldp	q21, q22, [%[ptr_out], #96] \n"         /* load outr12, outr13*/

                            "bne    2b                          \n"         /* jump to main loop*/

                    : [cnt]"+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2), \
                        [r3] "+r"(r3), [r4] "+r"(r4), [ptr_out] "+r"(ptr_out)
                    : [v0]"w"(v0), [v1]"w"(v1), [v2]"w"(v2), [v3]"w"(v3),\
                        [v4]"w"(v4),[v5]"w"(v5), [v6]"w"(v6), [v7]"w"(v7),\
                        [v8]"w"(v8)
                    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v15", "v16", \
                        "v17", "v18", "v19", "v20", "v21", "v22"
                    );

                    wc0 += 9 * hout_c_block;
                    inr0 += win_round;
                    inr1 += win_round;
                    inr2 += win_round;
                    inr3 += win_round;
                    inr4 += win_round;
                }
                write_to_output4xw(pre_out, dout_batch, c, c + 4, h, h + 2, 0, wout_round, chout, hout, wout, flag_relu, ptr_write);
            }
        }
    }
}

#else //__aarch64__

void conv_3x3s2_direct(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias_ptr, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, \
                          Context* ctx, void* work_space, const void* idx_ptr) {
    //! 3x3s2 convolution, implemented by direct algorithm
    //! pad is done implicit

    //! each core / loop gets 3 input rows in 1 input channel and produces 1 row in 2 output channels
    // q0 = w00, q1 = w01, q2 = w02
    // q3 = w10, q4 = w11, q5 = w12
    // q6 = r00/r10/r20, q7 = r01/r11/r21
    // q8 = r30/r40, q9 = r31,r41
    // q10 = outc0r0, q11 = outc0r1
    // q12 = outc1r0, q13 = outc1r1

    int threads = ctx->get_threads();

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

    int cremain = ch_out & 1;
    int bt_remain = hout - (cnt_row - 1) * 2;
    int pad_bot_size = ((hin + 1) / 2) * 2 - hin;//could be 0 or 1

    int do_right_pad = 1;
    unsigned int size_pad_right = w_loop * 4/*neon simd length*/ * 2 /*stride = 2*/ - w_in;//could be 0~7
    unsigned int right_pad_save = 4 - (w_loop * 4 - w_out);

    const float* bias = static_cast<const float*> (bias_ptr);

    //int right_pad_sub = (w_loop * 4 - w_out) * sizeof(float);
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

    for (int n = 0; n < num; ++n) {
        const float *din_batch = static_cast<const float*>(din) + n * ch_in * size_in_channel;
        float *dout_batch = static_cast<float*>(dout) + n * chout * size_out_channel;
#pragma omp parallel for num_threads(threads)
        for (int c = 0; c < ch_out - 1; c += 2) {

            float *dout_c0 = dout_batch + c * size_out_channel;
            float *dout_c1 = dout_c0 + size_out_channel;

            if (flag_bias) {
                fill_bias(dout_c0, bias + c, 1, size_out_channel);
                fill_bias(dout_c1, bias + c + 1, 1, size_out_channel);
            } else {
                fill_bias(dout_c0, zero, 1, size_out_channel);
                fill_bias(dout_c1, zero, 1, size_out_channel);
            }

            const float *wc0 = static_cast<const float*>(weights) + c * w_stride;
            const float *wc1 = wc0 + w_stride;

            for (int i = 0; i < ch_in; ++i) {

                int relu = 0;
                if ((i == ch_in - 1) && flag_relu) {
                    relu = 1;
                }

                const float *din_channel = din_batch + i * size_in_channel;

                const float *wcin0 = wc0 + i * 9;
                const float *wcin1 = wc1 + i * 9;
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

                float *ptr_zero = const_cast<float *>(zero);
                float32x4_t vzero = vdupq_n_f32(0.f);

                //! deal with top pad
                int h = 0;
                {
                    //! process
                    if (1) {
                        int cnt = cnt_col;
                        unsigned int* ptr_right_mask = right_pad_save_mask;
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
                                "5:                                     @ top row ending\n"
                        :[doutc0r0] "+r"(doutc0r0), [doutc0r1] "+r"(doutc0r1), \
                            [doutc1r0] "+r"(doutc1r0), [doutc1r1] "+r"(doutc1r1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), [cnt] "+r"(cnt)
                        :[wr00] "w"(wr00), [wr01] "w"(wr01), [wr02] "w"(wr02), \
                            [wr10] "w"(wr10), [wr11] "w"(wr11), [wr12] "w"(wr12), \
                            [right_pad_save_mask] "r"(ptr_right_mask), \
                            [do_right_pad] "r"(do_right_pad), [relu] "r"(relu)
                        :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                        );
                    }
                    //! after process, increase pointer

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
                    unsigned int *right_pad_ptr = right_pad_save_mask;

                    doutc0r0 = dout_c0 + 2 * h * w_out;
                    doutc0r1 = doutc0r0 + w_out;
                    doutc1r0 = dout_c1 + 2 * h * w_out;
                    doutc1r1 = doutc1r0 + w_out;

                    int cnt = cnt_col;
                    unsigned int* ptr_right_mask = right_pad_save_mask;
                    asm volatile(
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

                            //! process left pad
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

                            "5:                                     @ mid rows ending\n"
                    :[doutc0r0] "+r"(doutc0r0), [doutc0r1] "+r"(doutc0r1), \
                            [doutc1r0] "+r"(doutc1r0), [doutc1r1] "+r"(doutc1r1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                            [din4_ptr] "+r"(din4_ptr), [cnt] "+r"(cnt)
                    :[wr00] "w"(wr00), [wr01] "w"(wr01), [wr02] "w"(wr02), \
                            [wr10] "w"(wr10), [wr11] "w"(wr11), [wr12] "w"(wr12), \
                            [right_pad_save_mask] "r"(ptr_right_mask), \
                            [do_right_pad] "r"(do_right_pad), [relu] "r"(relu)
                    :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

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

                    doutc0r0 = dout_c0 + 2 * h * w_out;
                    doutc0r1 = doutc0r0 + w_out;
                    doutc1r0 = dout_c1 + 2 * h * w_out;
                    doutc1r1 = doutc1r0 + w_out;

                    int cnt = cnt_col;
                    unsigned int* ptr_right_mask = right_pad_save_mask;
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

                            "5:                                     @ bot row ending\n"
                    :[doutc0r0] "+r"(doutc0r0), [doutc0r1] "+r"(doutc0r1), \
                                [doutc1r0] "+r"(doutc1r0), [doutc1r1] "+r"(doutc1r1), \
                                [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                                [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                                [cnt] "+r"(cnt)
                    :[wr00] "w"(wr00), [wr01] "w"(wr01), [wr02] "w"(wr02), \
                                [wr10] "w"(wr10), [wr11] "w"(wr11), [wr12] "w"(wr12), \
                                [relu] "r"(relu), [do_right_pad] "r"(do_right_pad), \
                                [right_pad_save_mask] "r"(ptr_right_mask)

                    :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                } else if (bt_remain == 1) {
                    din0_ptr = dr0;
                    din1_ptr = dr1;
                    if (pad_bot_size > 0) {
                        din2_ptr = zero;
                    } else {
                        din2_ptr = dr2;
                    }

                    doutc0r0 = dout_c0 + 2 * h * w_out;
                    doutc1r0 = dout_c1 + 2 * h * w_out;
                    unsigned int* ptr_right_mask = right_pad_save_mask;
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

                    :[doutc0r0] "+r"(doutc0r0),[doutc1r0] "+r"(doutc1r0), \
                                [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                                [din2_ptr] "+r"(din2_ptr), [cnt] "+r"(cnt)
                    :[wr00] "w"(wr00), [wr01] "w"(wr01), [wr02] "w"(wr02), \
                                [wr10] "w"(wr10), [wr11] "w"(wr11), [wr12] "w"(wr12), \
                                [relu] "r"(relu), [do_right_pad] "r"(do_right_pad), \
                                [bot_pad] "r"(pad_bot_size), \
                                [right_pad_save_mask] "r"(ptr_right_mask)
                    :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }
                // end of processing bottom pad
            } // end of processing channels
        } //end of processing output channel
        if (cremain > 0) {
            float *dout_c = dout_batch + (chout - 1) * size_out_channel;

            if (flag_bias) {
                fill_bias(dout_c, &bias[chout - 1], 1, size_out_channel);
            } else {
                fill_bias(dout_c, zero, 1, size_out_channel);
            }

            const float *wc0 = static_cast<const float*>(weights) + (chout - 1) * w_stride;

            for (int i = 0; i < ch_in; ++i) {

                bool relu = (i == ch_in - 1) && flag_relu;

                const float *din_channel = din_batch + i * size_in_channel;

                const float *wcin0 = wc0 + i * 9;
                float w00 = wcin0[0];
                float w01 = wcin0[1];
                float w02 = wcin0[2];
                float w10 = wcin0[3];
                float w11 = wcin0[4];
                float w12 = wcin0[5];
                float w20 = wcin0[6];
                float w21 = wcin0[7];
                float w22 = wcin0[8];

                float *doutc0r0 = dout_c;
                float *doutc0r1 = dout_c + wout;

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

                float *ptr_zero = const_cast<float *>(zero);

                //float32x4_t vzero = vdupq_n_f32(0.f);
                //! deal with top pad
                float32x4x2_t vr0 = vld2q_f32(din0_ptr); //q0, q1
                float32x4x2_t vr1 = vld2q_f32(din1_ptr); //q2, q3
                float32x4x2_t vr2 = vld2q_f32(din2_ptr); //q4, q5
                float32x4x2_t vr3 = vld2q_f32(din3_ptr); //q6, q7
                float32x4x2_t vr4;  // q8, q9

                float32x4_t vor0;
		        float32x4_t vor1; //q10, q11
                int h = 0;
                //! process top
                if (1) {
                    //! process left
                    vor0 = vld1q_f32(doutc0r0);
                    float32x4_t vtmp1 = vdupq_n_f32(0.f);// = vld1q_f32(din0_ptr + 8);
                    vor1 = vld1q_f32(doutc0r1);
                    float32x4_t vtmp2;// = vld1q_f32(din0_ptr + 8);

                    vor0 = vmlaq_n_f32(vor0, vr0.val[0], w11);
                    vr1 = vld2q_f32(din1_ptr);
                    vor1 = vmlaq_n_f32(vor1, vr3.val[0], w21);
                    vr2 = vld2q_f32(din2_ptr);

                    float32x4_t vtmpr1 = vextq_f32(vtmp1, vr0.val[1], 3);
                    float32x4_t vtmpr2 = vextq_f32(vtmp1, vr3.val[1], 3);

                    vor0 = vmlaq_n_f32(vor0, vr0.val[1], w12);
                    vor1 = vmlaq_n_f32(vor1, vr3.val[1], w22);

                    din0_ptr += 7;
                    vr0 = vld2q_f32(din0_ptr);

                    vor0 = vmlaq_n_f32(vor0, vtmpr1, w10);
                    vor1 = vmlaq_n_f32(vor1, vtmpr2, w20);

                    din3_ptr += 7;
                    vr3 = vld2q_f32(din3_ptr);

                    vor0 = vmlaq_n_f32(vor0, vr1.val[0], w21);
                    vor1 = vmlaq_n_f32(vor1, vr1.val[0], w01);

                    vor0 = vmlaq_n_f32(vor0, vr1.val[1], w22);
                    vor1 = vmlaq_n_f32(vor1, vr1.val[1], w02);

                    vtmpr1 = vextq_f32(vtmp1, vr1.val[1], 3);
                    din1_ptr += 7;
                    vr1 = vld2q_f32(din1_ptr);

                    vor0 = vmlaq_n_f32(vor0, vtmpr1, w20);
                    vor1 = vmlaq_n_f32(vor1, vtmpr1, w00);

                    vor1 = vmlaq_n_f32(vor1, vr2.val[0], w11);

                    vtmpr2 = vextq_f32(vtmp1, vr2.val[1], 3);

                    vor1 = vmlaq_n_f32(vor1, vr2.val[1], w12);

                    din2_ptr += 7;
                    vr2 = vld2q_f32(din2_ptr);

                    vor1 = vmlaq_n_f32(vor1, vtmpr2, w10);

                    if (relu) {
                        vor0 = vmaxq_f32(vor0, vtmp1);
                        vor1 = vmaxq_f32(vor1, vtmp1);
                    }
                    vst1q_f32(doutc0r0, vor0);
                    vst1q_f32(doutc0r1, vor1);

                    doutc0r0 += 4;
                    doutc0r1 += 4;

                    //! process mid
                    vor0 = vld1q_f32(doutc0r0);
                    vtmp1 = vld1q_f32(din0_ptr + 8);
                    vor1 = vld1q_f32(doutc0r1);
                    vtmp2 = vld1q_f32(din3_ptr + 8);
                    for (int w = 0; w < cnt_col; ++w) {

                        vor0 = vmlaq_n_f32(vor0, vr0.val[0], w10);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[0], w20);
                        vtmpr1 = vextq_f32(vr0.val[0], vtmp1, 1);
                        vtmpr2 = vextq_f32(vr3.val[0], vtmp2, 1);

                        vtmp1 = vld1q_f32(din1_ptr + 8);
                        vor0 = vmlaq_n_f32(vor0, vr0.val[1], w11);
                        vtmp2 = vld1q_f32(din2_ptr + 8);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[1], w21);

                        din0_ptr += 8;
                        vr0 = vld2q_f32(din0_ptr);

                        vor0 = vmlaq_n_f32(vor0, vtmpr1, w12);
                        vor1 = vmlaq_n_f32(vor1, vtmpr2, w22);

                        din3_ptr += 8;
                        vr3 = vld2q_f32(din3_ptr);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[0], w20);
                        vor1 = vmlaq_n_f32(vor1, vr1.val[0], w00);

                        vtmpr1 = vextq_f32(vr1.val[0], vtmp1, 1);
                        vtmpr2 = vextq_f32(vr1.val[0], vtmp1, 1);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[1], w21);
                        vor1 = vmlaq_n_f32(vor1, vr1.val[1], w01);

                        din1_ptr += 8;
                        vr1 = vld2q_f32(din1_ptr);

                        vor0 = vmlaq_n_f32(vor0, vtmpr1, w22);
                        vor1 = vmlaq_n_f32(vor1, vtmpr2, w02);

                        vtmp1 = vld1q_f32(din0_ptr + 8);

                        vor1 = vmlaq_n_f32(vor1, vr2.val[0], w10);

                        vor1 = vmlaq_n_f32(vor1, vr2.val[1], w11);

                        vtmpr1 = vextq_f32(vr2.val[0], vtmp2, 1);
                        din2_ptr += 8;
                        vr2 = vld2q_f32(din2_ptr);

                        vor1 = vmlaq_n_f32(vor1, vtmpr1, w12);

                        vtmp2 = vld1q_f32(din3_ptr + 8);

                        if (relu) {
                            vtmpr1 = vdupq_n_f32(0.f);
                            vor0 = vmaxq_f32(vor0, vtmpr1);
                            vor1 = vmaxq_f32(vor1, vtmpr1);
                        }
                        vst1q_f32(doutc0r0, vor0);
                        vst1q_f32(doutc0r1, vor1);

                        doutc0r0 += 4;
                        doutc0r1 += 4;
                        vor0 = vld1q_f32(doutc0r0);
                        vor1 = vld1q_f32(doutc0r1);
                    }

                    //! process right
                    if (do_right_pad) {
                        // load pad mask
                        uint32x4_t vmask1= vld1q_u32(right_pad_save_mask);
                        uint32x4_t vmask2= vld1q_u32(right_pad_save_mask + 4);
                        vtmpr1 = vdupq_n_f32(0.f);
                        vr0.val[0] = vbslq_f32(vmask1, vr0.val[0], vtmpr1);
                        vr0.val[1] = vbslq_f32(vmask2, vr0.val[1], vtmpr1);
                        vr3.val[0] = vbslq_f32(vmask1, vr3.val[0], vtmpr1);
                        vr3.val[1] = vbslq_f32(vmask2, vr3.val[1], vtmpr1);

                        vor0 = vmlaq_n_f32(vor0, vr0.val[1], w11);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[1], w21);

                        vr0.val[1] = vextq_f32(vr0.val[0], vtmpr1, 1);
                        vr3.val[1] = vextq_f32(vr3.val[0], vtmpr1, 1);

                        vor0 = vmlaq_n_f32(vor0, vr0.val[0], w10);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[0], w20);

                        vor0 = vmlaq_n_f32(vor0, vr0.val[1], w12);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[1], w22);

                        vr1.val[0] = vbslq_f32(vmask1, vr1.val[0], vtmpr1);
                        vr1.val[1] = vbslq_f32(vmask2, vr1.val[1], vtmpr1);
                        vr2.val[0] = vbslq_f32(vmask1, vr2.val[0], vtmpr1);
                        vr2.val[1] = vbslq_f32(vmask2, vr2.val[1], vtmpr1);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[1], w21);
                        vor1 = vmlaq_n_f32(vor1, vr1.val[1], w01);

                        vr1.val[1] = vextq_f32(vr1.val[0], vtmpr1, 1);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[0], w20);
                        vor1 = vmlaq_n_f32(vor1, vr1.val[0], w00);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[1], w22);
                        vor1 = vmlaq_n_f32(vor1, vr1.val[1], w02);

                        vmask1 = vld1q_u32(right_pad_save_mask + 8);

                        vor1 = vmlaq_n_f32(vor1, vr2.val[1], w11);

                        vr2.val[1] = vextq_f32(vr2.val[0], vtmpr1, 1);

                        vor1 = vmlaq_n_f32(vor1, vr2.val[0], w10);

                        vr0.val[0] = vld1q_f32(doutc0r0);
                        vr0.val[1] = vld1q_f32(doutc0r1);

                        vor1 = vmlaq_n_f32(vor1, vr2.val[1], w12);

                        if (relu) {
                            vor0 = vmaxq_f32(vor0, vtmpr1);
                            vor1 = vmaxq_f32(vor1, vtmpr1);
                        }
                        vor0 = vbslq_f32(vmask1, vor0, vr0.val[0]);
                        vor1 = vbslq_f32(vmask1, vor1, vr0.val[1]);

                        vst1q_f32(doutc0r0, vor0);
                        vst1q_f32(doutc0r1, vor1);
                    }

                    dr0 = dr3;
                    dr1 = dr4;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                    dr4 = dr3 + w_in;
                }
                //! process mid rows
                int row_loop_end = cnt_row - 1;
                if (bt_remain == 2 && pad_bot_size == 0) {
                    row_loop_end = cnt_row;
                }
                for (h = 1; h < row_loop_end; h++) {

                    doutc0r0 = dout_c + h * 2 * wout;
                    doutc0r1 = doutc0r0 + wout;

                    din0_ptr = dr0;
                    din1_ptr = dr1;
                    din2_ptr = dr2;
                    din3_ptr = dr3;
                    din4_ptr = dr4;
                    //! process left
                    vr0 = vld2q_f32(din0_ptr);
                    vor0 = vld1q_f32(doutc0r0);
                    float32x4_t vtmp1 = vdupq_n_f32(0.f);// = vld1q_f32(din0_ptr + 8);
                    vr3 = vld2q_f32(din3_ptr);
                    vor1 = vld1q_f32(doutc0r1);
                    float32x4_t vtmp2;// = vld1q_f32(din0_ptr + 8);

                    vor0 = vmlaq_n_f32(vor0, vr0.val[0], w01);
                    vr1 = vld2q_f32(din1_ptr);
                    vor1 = vmlaq_n_f32(vor1, vr3.val[0], w11);
                    vr2 = vld2q_f32(din2_ptr);

                    float32x4_t vtmpr1 = vextq_f32(vtmp1, vr0.val[1], 3);
                    float32x4_t vtmpr2 = vextq_f32(vtmp1, vr3.val[1], 3);

                    vor0 = vmlaq_n_f32(vor0, vr0.val[1], w02);
                    vr4 = vld2q_f32(din4_ptr);
                    vor1 = vmlaq_n_f32(vor1, vr3.val[1], w12);

                    din0_ptr += 7;
                    vr0 = vld2q_f32(din0_ptr);

                    vor0 = vmlaq_n_f32(vor0, vtmpr1, w00);
                    vor1 = vmlaq_n_f32(vor1, vtmpr2, w10);

                    din3_ptr += 7;
                    vr3 = vld2q_f32(din3_ptr);

                    vor0 = vmlaq_n_f32(vor0, vr1.val[0], w11);
                    vor1 = vmlaq_n_f32(vor1, vr4.val[0], w21);

                    vtmpr1 = vextq_f32(vtmp1, vr1.val[1], 3);
                    vtmpr2 = vextq_f32(vtmp1, vr4.val[1], 3);

                    vor0 = vmlaq_n_f32(vor0, vr1.val[1], w12);
                    vor1 = vmlaq_n_f32(vor1, vr4.val[1], w22);

                    din1_ptr += 7;
                    vr1 = vld2q_f32(din1_ptr);

                    vor0 = vmlaq_n_f32(vor0, vtmpr1, w10);
                    vor1 = vmlaq_n_f32(vor1, vtmpr2, w20);

                    din4_ptr += 7;
                    vr4 = vld2q_f32(din4_ptr);

                    vor0 = vmlaq_n_f32(vor0, vr2.val[0], w21);
                    vor1 = vmlaq_n_f32(vor1, vr2.val[0], w01);

                    vor0 = vmlaq_n_f32(vor0, vr2.val[1], w22);
                    vor1 = vmlaq_n_f32(vor1, vr2.val[1], w02);

                    vtmpr1 = vextq_f32(vtmp1, vr2.val[1], 3);
                    din2_ptr += 7;
                    vr2 = vld2q_f32(din2_ptr);

                    vor0 = vmlaq_n_f32(vor0, vtmpr1, w20);
                    vor1 = vmlaq_n_f32(vor1, vtmpr1, w00);

                    if (relu) {
                        vor0 = vmaxq_f32(vor0, vtmp1);
                        vor1 = vmaxq_f32(vor1, vtmp1);
                    }
                    vst1q_f32(doutc0r0, vor0);
                    vst1q_f32(doutc0r1, vor1);

                    doutc0r0 += 4;
                    doutc0r1 += 4;

                    //! process mid
                    vor0 = vld1q_f32(doutc0r0);
                    vtmp1 = vld1q_f32(din0_ptr + 8);
                    vor1 = vld1q_f32(doutc0r1);
                    vtmp2 = vld1q_f32(din3_ptr + 8);
                    for (int w = 0; w < cnt_col; ++w) {

                        vor0 = vmlaq_n_f32(vor0, vr0.val[0], w00);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[0], w10);
                        vtmpr1 = vextq_f32(vr0.val[0], vtmp1, 1);
                        vtmpr2 = vextq_f32(vr3.val[0], vtmp2, 1);

                        vtmp1 = vld1q_f32(din1_ptr + 8);
                        vor0 = vmlaq_n_f32(vor0, vr0.val[1], w01);
                        vtmp2 = vld1q_f32(din4_ptr + 8);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[1], w11);

                        din0_ptr += 8;
                        vr0 = vld2q_f32(din0_ptr);

                        vor0 = vmlaq_n_f32(vor0, vtmpr1, w02);
                        vor1 = vmlaq_n_f32(vor1, vtmpr2, w12);

                        din3_ptr += 8;
                        vr3 = vld2q_f32(din3_ptr);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[0], w10);
                        vor1 = vmlaq_n_f32(vor1, vr4.val[0], w20);

                        vtmpr1 = vextq_f32(vr1.val[0], vtmp1, 1);
                        vtmpr2 = vextq_f32(vr4.val[0], vtmp2, 1);

                        vtmp1 = vld1q_f32(din1_ptr + 8);
                        vor0 = vmlaq_n_f32(vor0, vr1.val[1], w11);
                        vtmp2 = vld1q_f32(din4_ptr + 8);
                        vor1 = vmlaq_n_f32(vor1, vr4.val[1], w21);

                        din1_ptr += 8;
                        vr1 = vld2q_f32(din1_ptr);

                        vor0 = vmlaq_n_f32(vor0, vtmpr1, w12);
                        vor1 = vmlaq_n_f32(vor1, vtmpr2, w22);

                        din4_ptr += 8;
                        vr4 = vld2q_f32(din4_ptr);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[0], w20);
                        vtmp1 = vld1q_f32(din2_ptr + 8);
                        vor1 = vmlaq_n_f32(vor1, vr2.val[0], w00);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[1], w21);
                        vor1 = vmlaq_n_f32(vor1, vr2.val[1], w01);

                        vtmpr1 = vextq_f32(vr2.val[0], vtmp1, 1);
                        din2_ptr += 8;
                        vr2 = vld2q_f32(din2_ptr);

                        vor0 = vmlaq_n_f32(vor0, vtmpr1, w22);
                        vor1 = vmlaq_n_f32(vor1, vtmpr1, w02);

                        vtmp1 = vld1q_f32(din0_ptr + 8);
                        vtmp2 = vld1q_f32(din3_ptr + 8);

                        if (relu) {
                            vtmpr1 = vdupq_n_f32(0.f);
                            vor0 = vmaxq_f32(vor0, vtmpr1);
                            vor1 = vmaxq_f32(vor1, vtmpr1);
                        }
                        vst1q_f32(doutc0r0, vor0);
                        vst1q_f32(doutc0r1, vor1);

                        doutc0r0 += 4;
                        doutc0r1 += 4;
                        vor0 = vld1q_f32(doutc0r0);
                        vor1 = vld1q_f32(doutc0r1);
                    }

                    //! process right
                    if (do_right_pad) {
                        // load pad mask
                        uint32x4_t vmask1 = vld1q_u32(right_pad_save_mask);
                        uint32x4_t vmask2 = vld1q_u32(right_pad_save_mask + 4);
                        vtmpr1 = vdupq_n_f32(0.f);
                        vr0.val[0] = vbslq_f32(vmask1, vr0.val[0], vtmpr1);
                        vr0.val[1] = vbslq_f32(vmask2, vr0.val[1], vtmpr1);
                        vr3.val[0] = vbslq_f32(vmask1, vr3.val[0], vtmpr1);
                        vr3.val[1] = vbslq_f32(vmask2, vr3.val[1], vtmpr1);

                        vor0 = vmlaq_n_f32(vor0, vr0.val[1], w01);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[1], w11);

                        vr0.val[1] = vextq_f32(vr0.val[0], vtmpr1, 1);
                        vr3.val[1] = vextq_f32(vr3.val[0], vtmpr1, 1);

                        vor0 = vmlaq_n_f32(vor0, vr0.val[0], w00);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[0], w10);

                        vor0 = vmlaq_n_f32(vor0, vr0.val[1], w02);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[1], w12);

                        vr1.val[0] = vbslq_f32(vmask1, vr1.val[0], vtmpr1);
                        vr1.val[1] = vbslq_f32(vmask2, vr1.val[1], vtmpr1);
                        vr4.val[0] = vbslq_f32(vmask1, vr4.val[0], vtmpr1);
                        vr4.val[1] = vbslq_f32(vmask2, vr4.val[1], vtmpr1);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[1], w11);
                        vor1 = vmlaq_n_f32(vor1, vr4.val[1], w21);

                        vr1.val[1] = vextq_f32(vr1.val[0], vtmpr1, 1);
                        vr4.val[1] = vextq_f32(vr4.val[0], vtmpr1, 1);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[0], w10);
                        vor1 = vmlaq_n_f32(vor1, vr4.val[0], w20);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[1], w12);
                        vor1 = vmlaq_n_f32(vor1, vr4.val[1], w22);

                        vr2.val[0] = vbslq_f32(vmask1, vr2.val[0], vtmpr1);
                        vr2.val[1] = vbslq_f32(vmask2, vr2.val[1], vtmpr1);

                        vmask1 = vld1q_u32(right_pad_save_mask + 8);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[1], w21);
                        vor1 = vmlaq_n_f32(vor1, vr2.val[1], w01);

                        vr2.val[1] = vextq_f32(vr2.val[0], vtmpr1, 1);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[0], w20);
                        vor1 = vmlaq_n_f32(vor1, vr2.val[0], w00);

                        vr0.val[0] = vld1q_f32(doutc0r0);
                        vr0.val[1] = vld1q_f32(doutc0r1);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[1], w22);
                        vor1 = vmlaq_n_f32(vor1, vr2.val[1], w02);

                        if (relu) {
                            vor0 = vmaxq_f32(vor0, vtmpr1);
                            vor1 = vmaxq_f32(vor1, vtmpr1);
                        }

                        vor0 = vbslq_f32(vmask1, vor0, vr0.val[0]);
                        vor1 = vbslq_f32(vmask1, vor1, vr0.val[1]);

                        vst1q_f32(doutc0r0, vor0);
                        vst1q_f32(doutc0r1, vor1);
                    }
                    dr0 = dr4;
                    dr1 = dr0 + w_in;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                    dr4 = dr3 + w_in;
                }
                //! process bottom
                if (bt_remain == 2 && pad_bot_size > 0) {

                    doutc0r0 = dout_c + h * 2 * wout;
                    doutc0r1 = doutc0r0 + wout;

                    din0_ptr = dr0;
                    din1_ptr = dr1;
                    din2_ptr = dr2;
                    din3_ptr = dr3;
                    //! process left
                    vr0 = vld2q_f32(din0_ptr);
                    vor0 = vld1q_f32(doutc0r0);
                    float32x4_t vtmp1 = vdupq_n_f32(0.f);// = vld1q_f32(din0_ptr + 8);
                    vr3 = vld2q_f32(din3_ptr);
                    vor1 = vld1q_f32(doutc0r1);
                    float32x4_t vtmp2;// = vld1q_f32(din0_ptr + 8);

                    vor0 = vmlaq_n_f32(vor0, vr0.val[0], w01);
                    vr1 = vld2q_f32(din1_ptr);
                    vor1 = vmlaq_n_f32(vor1, vr3.val[0], w11);
                    vr2 = vld2q_f32(din2_ptr);

                    float32x4_t vtmpr1 = vextq_f32(vtmp1, vr0.val[1], 3);
                    float32x4_t vtmpr2 = vextq_f32(vtmp1, vr3.val[1], 3);

                    vor0 = vmlaq_n_f32(vor0, vr0.val[1], w02);
                    vor1 = vmlaq_n_f32(vor1, vr3.val[1], w12);

                    din0_ptr += 7;
                    vr0 = vld2q_f32(din0_ptr);

                    vor0 = vmlaq_n_f32(vor0, vtmpr1, w00);
                    vor1 = vmlaq_n_f32(vor1, vtmpr2, w10);

                    din3_ptr += 7;
                    vr3 = vld2q_f32(din3_ptr);

                    vor0 = vmlaq_n_f32(vor0, vr1.val[0], w11);

                    vtmpr1 = vextq_f32(vtmp1, vr1.val[1], 3);

                    vor0 = vmlaq_n_f32(vor0, vr1.val[1], w12);

                    din1_ptr += 7;
                    vr1 = vld2q_f32(din1_ptr);

                    vor0 = vmlaq_n_f32(vor0, vtmpr1, w10);

                    vor0 = vmlaq_n_f32(vor0, vr2.val[0], w21);
                    vor1 = vmlaq_n_f32(vor1, vr2.val[0], w01);

                    vor0 = vmlaq_n_f32(vor0, vr2.val[1], w22);
                    vor1 = vmlaq_n_f32(vor1, vr2.val[1], w02);

                    vtmpr1 = vextq_f32(vtmp1, vr2.val[1], 3);
                    din2_ptr += 7;
                    vr2 = vld2q_f32(din2_ptr);

                    vor0 = vmlaq_n_f32(vor0, vtmpr1, w20);
                    vor1 = vmlaq_n_f32(vor1, vtmpr1, w00);

                    if (relu) {
                        vor0 = vmaxq_f32(vor0, vtmp1);
                        vor1 = vmaxq_f32(vor1, vtmp1);
                    }
                    vst1q_f32(doutc0r0, vor0);
                    vst1q_f32(doutc0r1, vor1);

                    doutc0r0 += 4;
                    doutc0r1 += 4;

                    //! process mid
                    vor0 = vld1q_f32(doutc0r0);
                    vtmp1 = vld1q_f32(din0_ptr + 8);
                    vor1 = vld1q_f32(doutc0r1);
                    vtmp2 = vld1q_f32(din3_ptr + 8);
                    for (int w = 0; w < cnt_col; ++w) {

                        vor0 = vmlaq_n_f32(vor0, vr0.val[0], w00);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[0], w10);
                        vtmpr1 = vextq_f32(vr0.val[0], vtmp1, 1);
                        vtmpr2 = vextq_f32(vr3.val[0], vtmp2, 1);

                        vtmp1 = vld1q_f32(din1_ptr + 8);
                        vor0 = vmlaq_n_f32(vor0, vr0.val[1], w01);
                        vtmp2 = vld1q_f32(din2_ptr + 8);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[1], w11);

                        din0_ptr += 8;
                        vr0 = vld2q_f32(din0_ptr);

                        vor0 = vmlaq_n_f32(vor0, vtmpr1, w02);
                        vor1 = vmlaq_n_f32(vor1, vtmpr2, w12);

                        din3_ptr += 8;
                        vr3 = vld2q_f32(din3_ptr);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[0], w10);

                        vtmpr1 = vextq_f32(vr1.val[0], vtmp1, 1);

                        vtmp1 = vld1q_f32(din1_ptr + 8);
                        vor0 = vmlaq_n_f32(vor0, vr1.val[1], w11);

                        din1_ptr += 8;
                        vr1 = vld2q_f32(din1_ptr);

                        vor0 = vmlaq_n_f32(vor0, vtmpr1, w12);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[0], w20);
                        vtmp1 = vld1q_f32(din2_ptr + 8);
                        vor1 = vmlaq_n_f32(vor1, vr2.val[0], w00);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[1], w21);
                        vor1 = vmlaq_n_f32(vor1, vr2.val[1], w01);

                        vtmpr1 = vextq_f32(vr2.val[0], vtmp1, 1);
                        din2_ptr += 8;
                        vr2 = vld2q_f32(din2_ptr);

                        vor0 = vmlaq_n_f32(vor0, vtmpr1, w22);
                        vor1 = vmlaq_n_f32(vor1, vtmpr1, w02);

                        vtmp1 = vld1q_f32(din0_ptr + 8);
                        vtmp2 = vld1q_f32(din3_ptr + 8);

                        if (relu) {
                            vtmpr1 = vdupq_n_f32(0.f);
                            vor0 = vmaxq_f32(vor0, vtmpr1);
                            vor1 = vmaxq_f32(vor1, vtmpr1);
                        }
                        vst1q_f32(doutc0r0, vor0);
                        vst1q_f32(doutc0r1, vor1);

                        doutc0r0 += 4;
                        doutc0r1 += 4;
                        vor0 = vld1q_f32(doutc0r0);
                        vor1 = vld1q_f32(doutc0r1);
                    }

                    //! process right
                    if (do_right_pad) {
                        // load pad mask
                        uint32x4_t vmask1 = vld1q_u32(right_pad_save_mask);
                        uint32x4_t vmask2 = vld1q_u32(right_pad_save_mask + 4);
                        vtmpr1 = vdupq_n_f32(0.f);
                        vr0.val[0] = vbslq_f32(vmask1, vr0.val[0], vtmpr1);
                        vr0.val[1] = vbslq_f32(vmask2, vr0.val[1], vtmpr1);
                        vr3.val[0] = vbslq_f32(vmask1, vr3.val[0], vtmpr1);
                        vr3.val[1] = vbslq_f32(vmask2, vr3.val[1], vtmpr1);

                        vor0 = vmlaq_n_f32(vor0, vr0.val[1], w01);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[1], w11);

                        vr0.val[1] = vextq_f32(vr0.val[0], vtmpr1, 1);
                        vr3.val[1] = vextq_f32(vr3.val[0], vtmpr1, 1);

                        vor0 = vmlaq_n_f32(vor0, vr0.val[0], w00);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[0], w10);

                        vor0 = vmlaq_n_f32(vor0, vr0.val[1], w02);
                        vor1 = vmlaq_n_f32(vor1, vr3.val[1], w12);

                        vr1.val[0] = vbslq_f32(vmask1, vr1.val[0], vtmpr1);
                        vr1.val[1] = vbslq_f32(vmask2, vr1.val[1], vtmpr1);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[1], w11);

                        vr1.val[1] = vextq_f32(vr1.val[0], vtmpr1, 1);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[0], w10);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[1], w12);

                        vr2.val[0] = vbslq_f32(vmask1, vr2.val[0], vtmpr1);
                        vr2.val[1] = vbslq_f32(vmask2, vr2.val[1], vtmpr1);

                        vmask1 = vld1q_u32(right_pad_save_mask + 8);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[1], w21);
                        vor1 = vmlaq_n_f32(vor1, vr2.val[1], w01);

                        vr2.val[1] = vextq_f32(vr2.val[0], vtmpr1, 1);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[0], w20);
                        vor1 = vmlaq_n_f32(vor1, vr2.val[0], w00);

                        vr0.val[0] = vld1q_f32(doutc0r0);
                        vr0.val[1] = vld1q_f32(doutc0r1);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[1], w22);
                        vor1 = vmlaq_n_f32(vor1, vr2.val[1], w02);

                        if (relu) {
                            vor0 = vmaxq_f32(vor0, vtmpr1);
                            vor1 = vmaxq_f32(vor1, vtmpr1);
                        }

                        vor0 = vbslq_f32(vmask1, vor0, vr0.val[0]);
                        vor1 = vbslq_f32(vmask1, vor1, vr0.val[1]);

                        vst1q_f32(doutc0r0, vor0);
                        vst1q_f32(doutc0r1, vor1);
                    }

                } else if (bt_remain == 1) {
                    din0_ptr = dr0;
                    din1_ptr = dr1;
                    if (pad_bot_size > 0) {
                        din2_ptr = zero;
                    } else {
                        din2_ptr = dr2;
                    }

                    doutc0r0 = dout_c + 2 * h * w_out;

                    //! process left
                    vr0 = vld2q_f32(din0_ptr);
                    vor0 = vld1q_f32(doutc0r0);
                    vr1 = vld2q_f32(din1_ptr);
                    vr2 = vld2q_f32(din2_ptr);
                    float32x4_t vtmp1 = vdupq_n_f32(0.f);// = vld1q_f32(din0_ptr + 8);
                    float32x4_t vtmp2;
                    float32x4_t vtmp3;

                    vor0 = vmlaq_n_f32(vor0, vr0.val[0], w01);
                    float32x4_t vtmpr1 = vextq_f32(vtmp1, vr0.val[1], 3);
                    vor0 = vmlaq_n_f32(vor0, vr0.val[1], w02);
                    din0_ptr += 7;
                    vr0 = vld2q_f32(din0_ptr);
                    vor0 = vmlaq_n_f32(vor0, vtmpr1, w00);
                    vor0 = vmlaq_n_f32(vor0, vr1.val[0], w11);
                    vtmpr1 = vextq_f32(vtmp1, vr1.val[1], 3);
                    vor0 = vmlaq_n_f32(vor0, vr1.val[1], w12);
                    din1_ptr += 7;
                    vr1 = vld2q_f32(din1_ptr);
                    vor0 = vmlaq_n_f32(vor0, vtmpr1, w10);

                    vor0 = vmlaq_n_f32(vor0, vr2.val[0], w21);
                    vor0 = vmlaq_n_f32(vor0, vr2.val[1], w22);

                    vtmpr1 = vextq_f32(vtmp1, vr2.val[1], 3);
                    if (pad_bot_size == 0) {
                        din2_ptr += 7;
                        vr2 = vld2q_f32(din2_ptr);
                    }

                    vor0 = vmlaq_n_f32(vor0, vtmpr1, w20);

                    if (relu) {
                        vor0 = vmaxq_f32(vor0, vtmp1);
                    }
                    vst1q_f32(doutc0r0, vor0);

                    doutc0r0 += 4;

                    //! process mid
                    vor0 = vld1q_f32(doutc0r0);
                    vtmp1 = vld1q_f32(din0_ptr + 8);
                    vtmp2 = vld1q_f32(din1_ptr + 8);
                    vtmp3 = vld1q_f32(din2_ptr + 8);

                    for (int w = 0; w < cnt_col; ++w) {

                        vor0 = vmlaq_n_f32(vor0, vr0.val[0], w00);
                        vtmpr1 = vextq_f32(vr0.val[0], vtmp1, 1);
                        vor0 = vmlaq_n_f32(vor0, vr0.val[1], w01);
                        din0_ptr += 8;
                        vr0 = vld2q_f32(din0_ptr);
                        vor0 = vmlaq_n_f32(vor0, vtmpr1, w02);

                        vtmp1 = vld1q_f32(din0_ptr + 8);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[0], w10);
                        vtmpr1 = vextq_f32(vr1.val[0], vtmp2, 1);
                        vor0 = vmlaq_n_f32(vor0, vr1.val[1], w11);
                        din1_ptr += 8;
                        vr1 = vld2q_f32(din1_ptr);
                        vor0 = vmlaq_n_f32(vor0, vtmpr1, w12);

                        vtmp2 = vld1q_f32(din1_ptr + 8);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[0], w20);
                        vtmpr1 = vextq_f32(vr2.val[0], vtmp3, 1);
                        vor0 = vmlaq_n_f32(vor0, vr2.val[1], w21);
                        if (pad_bot_size == 0) {
                            din2_ptr += 8;
                            vr2 = vld2q_f32(din2_ptr);
                            vtmp3 = vld1q_f32(din2_ptr + 8);
                        }
                        vor0 = vmlaq_n_f32(vor0, vtmpr1, w22);

                        if (relu) {
                            vtmpr1 = vdupq_n_f32(0.f);
                            vor0 = vmaxq_f32(vor0, vtmpr1);
                        }
                        vst1q_f32(doutc0r0, vor0);

                        doutc0r0 += 4;
                        vor0 = vld1q_f32(doutc0r0);
                    }

                    //! process right
                    if (do_right_pad) {
                        // load pad mask
                        uint32x4_t vmask1 = vld1q_u32(right_pad_save_mask);
                        uint32x4_t vmask2 = vld1q_u32(right_pad_save_mask + 4);
                        vtmpr1 = vdupq_n_f32(0.f);
                        vr0.val[0] = vbslq_f32(vmask1, vr0.val[0], vtmpr1);
                        vr0.val[1] = vbslq_f32(vmask2, vr0.val[1], vtmpr1);

                        vor0 = vmlaq_n_f32(vor0, vr0.val[1], w01);

                        vr0.val[1] = vextq_f32(vr0.val[0], vtmpr1, 1);

                        vor0 = vmlaq_n_f32(vor0, vr0.val[0], w00);

                        vor0 = vmlaq_n_f32(vor0, vr0.val[1], w02);

                        vr1.val[0] = vbslq_f32(vmask1, vr1.val[0], vtmpr1);
                        vr1.val[1] = vbslq_f32(vmask2, vr1.val[1], vtmpr1);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[1], w11);

                        vr1.val[1] = vextq_f32(vr1.val[0], vtmpr1, 1);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[0], w10);

                        vor0 = vmlaq_n_f32(vor0, vr1.val[1], w12);

                        vr2.val[0] = vbslq_f32(vmask1, vr2.val[0], vtmpr1);
                        vr2.val[1] = vbslq_f32(vmask2, vr2.val[1], vtmpr1);

                        vmask1 = vld1q_u32(right_pad_save_mask + 8);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[1], w21);

                        vr2.val[1] = vextq_f32(vr2.val[0], vtmpr1, 1);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[0], w20);

                        vr0.val[0] = vld1q_f32(doutc0r0);

                        vor0 = vmlaq_n_f32(vor0, vr2.val[1], w22);

                        if (relu) {
                            vor0 = vmaxq_f32(vor0, vtmpr1);
                        }

                        vor0 = vbslq_f32(vmask1, vor0, vr0.val[0]);
                        vst1q_f32(doutc0r0, vor0);
                    }

                }
            } // end of remain out channel

        } // end of processing batchs
    }
}
#endif //__aarch64__
} //namespace lite

} //namespace saber

} //namespace anakin

#endif
