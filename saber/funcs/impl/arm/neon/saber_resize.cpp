#include "saber/funcs/impl/arm/saber_resize.h"

namespace anakin{

namespace saber{

// This implementation is based on https:https://github.com/Tencent/ncnn

// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.


// BSD License 3-clause

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the <organization> nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

void resize_bilinear(const float* src, int w_in, int h_in, float* dst, int w_out, \
                           int h_out, float scale_x, float scale_y, ResizeType resize_type){

    int* buf = new int[w_out + h_out + w_out * 2 + h_out * 2];

    int* xofs = buf;
    int* yofs = buf + w_out;

    float* alpha = (float*)(buf + w_out + h_out);
    float* beta = (float*)(buf + w_out + h_out + w_out * 2);

    float fx = 0.0f;
    float fy = 0.0f;
    int sx = 0;
    int sy = 0;
    if (resize_type == BILINEAR_ALIGN){
        scale_x = (float)(w_in - 1) / (w_out - 1);
        scale_y = (float)(h_in - 1) / (h_out - 1);
        //calculate x axis coordinate
        for (int dx = 0; dx < w_out; dx++){
            fx = dx * scale_x;
            sx = int(fx);
            fx -= sx;

            xofs[dx] = sx;

            alpha[dx * 2] = 1.f - fx;
            alpha[dx * 2 + 1] = fx;
        }
        //calculate y axis coordinate
        for (int dy = 0; dy < h_out; dy++){
            fy = dy * scale_y;
            sy = int(fy);
            fy -= sy;

            yofs[dy] = sy;

            beta[dy * 2] = 1.f - fy;
            beta[dy * 2 + 1] = fy;
        }
    } else if (resize_type == BILINEAR_NO_ALIGN){
        scale_x = (float)w_in / w_out;
        scale_y = (float)h_in / h_out;
        //calculate x axis coordinate
        for (int dx = 0; dx < w_out; dx++){
            fx = scale_x * (dx + 0.5f) - 0.5f;
            fx = fx < 0 ? 0.f : fx;
            sx = int(fx);
            fx -= sx;

            xofs[dx] = sx;

            alpha[dx * 2] = 1.f - fx;
            alpha[dx * 2 + 1] = fx;
        }
        //calculate y axis coordinate
        for (int dy = 0; dy < h_out; dy++){
            fy = scale_y * (dy + 0.5f) - 0.5f;
            fy = fy < 0 ? 0.f : fy;
            sy = int(fy);
            fy -= sy;

            yofs[dy] = sy;

            beta[dy * 2] = 1.f - fy;
            beta[dy * 2 + 1] = fy;
        }
    }
    float* rowsbuf0 = new float[w_out];
    float* rowsbuf1 = new float[w_out];
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    //output w , h boundary
    int w_bound = w_out;
    int h_bound = h_out;
    if (resize_type == BILINEAR_ALIGN){
        w_bound = ceil((w_in - 1) / scale_x);
        h_bound = ceil((h_in - 1) / scale_y);
    } else if (resize_type == BILINEAR_NO_ALIGN){
        w_bound = ceil((w_in - 0.5f) / scale_x - 0.5f);
        h_bound = ceil((h_in - 0.5f) / scale_y - 0.5f);
    }
    //h_bound loop
    for (int dy = 0; dy < h_bound; dy++){
        int sy = yofs[dy];

        const float* s0 = src + sy * w_in;
        const float* s1 = src + (sy + 1) * w_in;

        const float* alphap = alpha;
        float* rows0p = rows0;
        float* rows1p = rows1;

        int dx = 0;
        //w_bound loop
        for (; dx+1 < w_bound; dx += 2){
            int sx = xofs[dx];
            int sxn = xofs[dx+1];
            const float* s0p = s0 + sx;
            const float* s1p = s1 + sx;
            const float* s0np = s0 + sxn;
            const float* s1np = s1 + sxn;

            float32x4_t _a = vld1q_f32(alphap);
            float32x2_t _s0 = vld1_f32(s0p);
            float32x2_t _s1 = vld1_f32(s1p);
            float32x2_t _s0n = vld1_f32(s0np);
            float32x2_t _s1n = vld1_f32(s1np);

            float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
            float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
            float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
            float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

            float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
            vst1_f32(rows0p + dx, _rows0);
            float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
            vst1_f32(rows1p + dx, _rows1);

            alphap += 4;

        }
        //w_bound remain loop
        for (; dx < w_bound; dx++){
            int sx = xofs[dx];
            const float* s0p = s0 + sx;
            const float* s1p = s1 + sx;

            float a0 = alphap[0];
            float a1 = alphap[1];
            rows0p[dx] = s0p[0]*a0 + s0p[1]*a1;
            rows1p[dx] = s1p[0]*a0 + s1p[1]*a1;

            alphap += 2;
        }

        const float buffer1[2] = {*(src + sy * w_in + w_in - 1), *(src + sy * w_in + w_in - 1)};
        const float buffer2[2] = {*(src + (sy+1) * w_in + w_in - 1), *(src + (sy+1) * w_in + w_in - 1)};
        //w_bound - w_out loop
        for (; dx+1 < w_out; dx += 2){

            const float* s0p = buffer1;
            const float* s1p = buffer2;
            const float* s0np = buffer1;
            const float* s1np = buffer2;

            float32x4_t _a = vld1q_f32(alphap);
            float32x2_t _s0 = vld1_f32(s0p);
            float32x2_t _s1 = vld1_f32(s1p);
            float32x2_t _s0n = vld1_f32(s0np);
            float32x2_t _s1n = vld1_f32(s1np);

            float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
            float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
            float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
            float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

            float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
            vst1_f32(rows0p + dx, _rows0);
            float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
            vst1_f32(rows1p + dx, _rows1);

            alphap += 4;
        }
        //w_bound - w_out remain loop
        for (; dx < w_out; dx++){
            const float* s0p = buffer1;
            const float* s1p = buffer2;

            float a0 = alphap[0];
            float a1 = alphap[1];
            rows0p[dx] = s0p[0]*a0 + s0p[1]*a1;
            rows1p[dx] = s1p[0]*a0 + s1p[1]*a1;

            alphap += 2;
        }

        float b0 = beta[0];
        float b1 = beta[1];

        float* dp = dst + dy * w_out;

        int nn = w_out >> 3;
        int remain = w_out - (nn << 3);

#ifdef __aarch64__
        float32x4_t _b0 = vdupq_n_f32(b0);
        float32x4_t _b1 = vdupq_n_f32(b1);
        //calculate and store results
        for (; nn>0; nn--){
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _d = vmulq_f32(_rows0, _b0);
            float32x4_t _rows1 = vld1q_f32(rows1p);
            _d = vmlaq_f32(_d, _rows1, _b1);

            float32x4_t _rows0n = vld1q_f32(rows0p+4);
            float32x4_t _rows1n = vld1q_f32(rows1p+4);

            float32x4_t _dn = vmulq_f32(_rows0n, _b0);
            vst1q_f32(dp, _d);
            _dn = vmlaq_f32(_dn, _rows1n, _b1);
            vst1q_f32(dp+4, _dn);

            dp += 8;
            rows0p += 8;
            rows1p += 8;
        }

#else
        if (nn > 0){
            asm volatile(
                "vdup.32 q0, %[b0]                   @dup b0 to q1\n"
                "vdup.32 q1, %[b1]                   @dup b1 to q0\n"
                "1:                                                      \n"
                "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
                "vld1.32 {d6-d7}, [%[rows1p]]!       @loads rows0p to q3\n"
                "vmul.f32 q2, q2, q0                 @mul\n"
                "vmla.f32 q2, q3, q1                 @mul add\n"
                "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
                "pld [%[rows0p]]                     @preload rows0p\n"

                "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
                "vld1.32 {d6-d7}, [%[rows1p]]!       @load rows1p to q3\n"
                "vmul.f32 q2, q2, q0                 @mul\n"
                "vmla.f32 q2, q3, q1                 @mul add\n"
                "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
                "pld [%[rows1p]]                     @preload rows1p\n"
                "subs %[loopc], #1                   @loop count minus #1\n"
                "bne 1b                              @jump to 1\n"
                :[rows0p]"+r"(rows0p), [rows1p]"+r"(rows1p), [out]"+r"(dp), [loopc]"+r"(nn)
                :[b0]"r"(b0), [b1]"r"(b1)
                :"cc", "memory", "q0", "q1", "q2", "q3"
            );
        }
#endif
        //calculate and store remain resluts
        for (; remain; --remain){
            *dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }
        beta += 2;
    }

//h_bound - h_out loop
for (int dy = h_bound; dy < h_out; dy++){
        int sy = h_in - 1;
        const float* s0 = src + sy * w_in;
        const float* s1 = s0;
        const float* alphap = alpha;
        float* rows0p = rows0;
        float* rows1p = rows1;

        int dx = 0;
        //w_bound loop
        for (; dx+1 < w_bound; dx += 2){
            int sx = xofs[dx];
            int sxn = xofs[dx+1];
            const float* s0p = s0 + sx;
            const float* s1p = s1 + sx;
            const float* s0np = s0 + sxn;
            const float* s1np = s1 + sxn;

            float32x4_t _a = vld1q_f32(alphap);
            float32x2_t _s0 = vld1_f32(s0p);
            float32x2_t _s1 = vld1_f32(s1p);
            float32x2_t _s0n = vld1_f32(s0np);
            float32x2_t _s1n = vld1_f32(s1np);

            float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
            float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
            float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
            float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

            float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
            vst1_f32(rows0p + dx, _rows0);
            float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
            vst1_f32(rows1p + dx, _rows1);

            alphap += 4;

        }
        //w_bound remain loop
        for (; dx < w_bound; dx++){
            int sx = xofs[dx];
            const float* s0p = s0 + sx;
            float a0 = alphap[0];
            float a1 = alphap[1];
            rows0p[dx] = s0p[0]*a0 + s0p[1]*a1;
            rows1p[dx] = rows0p[dx];

            alphap += 2;
        }

        const float buffer1[2] = {*(src + sy * w_in + w_in - 1), *(src + sy * w_in + w_in - 1)};
        //w_bound - w_out loop
        for (; dx+1 < w_out; dx += 2){

            const float* s0p = buffer1;
            const float* s1p = buffer1;
            const float* s0np = buffer1;
            const float* s1np = buffer1;

            float32x4_t _a = vld1q_f32(alphap);
            float32x2_t _s0 = vld1_f32(s0p);
            float32x2_t _s1 = vld1_f32(s1p);
            float32x2_t _s0n = vld1_f32(s0np);
            float32x2_t _s1n = vld1_f32(s1np);

            float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
            float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
            float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
            float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

            float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
            vst1_f32(rows0p + dx, _rows0);
            float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
            vst1_f32(rows1p + dx, _rows1);

            alphap += 4;
        }
        //w_bound - wout remain loop
        for (; dx < w_out; dx++){
            const float* s0p = buffer1;
            float a0 = alphap[0];
            float a1 = alphap[1];
            rows0p[dx] = s0p[0]*a0 + s0p[1]*a1;
            rows1p[dx] = rows0p[dx];
            alphap += 2;
        }

        float b0 = beta[0];
        float b1 = beta[1];

        float* dp = dst + dy * w_out;

        int nn = w_out >> 3;
        int remain = w_out - (nn << 3);

#ifdef __aarch64__
        float32x4_t _b0 = vdupq_n_f32(b0);
        float32x4_t _b1 = vdupq_n_f32(b1);
        //calculate and store results
        for (; nn>0; nn--){
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _d = vmulq_f32(_rows0, _b0);
            float32x4_t _rows1 = vld1q_f32(rows1p);
            _d = vmlaq_f32(_d, _rows1, _b1);

            float32x4_t _rows0n = vld1q_f32(rows0p + 4);
            float32x4_t _rows1n = vld1q_f32(rows1p + 4);

            float32x4_t _dn = vmulq_f32(_rows0n, _b0);
            vst1q_f32(dp, _d);
            _dn = vmlaq_f32(_dn, _rows1n, _b1);
            vst1q_f32(dp+4, _dn);

            dp += 8;
            rows0p += 8;
            rows1p += 8;
        }

#else
        if (nn > 0){
            asm volatile(
                "vdup.32 q0, %[b0]                   @dup b0 to q1\n"
                "vdup.32 q1, %[b1]                   @dup b1 to q0\n"
                "1:                                                      \n"
                "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
                "vld1.32 {d6-d7}, [%[rows1p]]!       @loads rows0p to q3\n"
                "vmul.f32 q2, q2, q0                 @mul\n"
                "vmla.f32 q2, q3, q1                 @mul add\n"
                "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
                "pld [%[rows0p]]                     @preload rows0p\n"

                "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
                "vld1.32 {d6-d7}, [%[rows1p]]!       @load rows1p to q3\n"
                "vmul.f32 q2, q2, q0                 @mul\n"
                "vmla.f32 q2, q3, q1                 @mul add\n"
                "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
                "pld [%[rows1p]]                     @preload rows1p\n"
                "subs %[loopc], #1                   @loop count minus #1\n"
                "bne 1b                              @jump to 1\n"
                :[rows0p]"+r"(rows0p), [rows1p]"+r"(rows1p), [out]"+r"(dp), [loopc]"+r"(nn)
                :[b0]"r"(b0), [b1]"r"(b1)
                :"cc", "memory", "q0", "q1", "q2", "q3"
            );
        }
#endif
        //calculate and store remain results
        for (; remain; --remain){
            *dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }
    delete[] buf;
    delete[] rowsbuf0;
    delete[] rowsbuf1;
}

void resize_bilinear_custom(const float* src, int w_in, int h_in, float* dst, int w_out, \
                           int h_out, float scale_x, float scale_y, ResizeType resize_type){

    int* buf = new int[w_out + h_out + w_out * 2 + h_out * 2];

    int* xofs = buf;
    int* yofs = buf + w_out;

    float* alpha = (float*)(buf + w_out + h_out);
    float* beta = (float*)(buf + w_out + h_out + w_out * 2);

    float fx = 0.0f;
    float fy = 0.0f;
    int sx = 0;
    int sy = 0;
    //calculate x axis coordinate
    for (int dx = 0; dx < w_out; dx++){
        fx = dx * scale_x;
        sx = int(fx);
        fx -= sx;

        xofs[dx] = sx;

        alpha[dx * 2] = 1.f - fx;
        alpha[dx * 2 + 1] = fx;
    }
    //calculate y axis coordinate
    for (int dy = 0; dy < h_out; dy++){
        fy = dy * scale_y;
        sy = int(fy);
        fy -= sy;

        yofs[dy] = sy;

        beta[dy * 2] = 1.f - fy;
        beta[dy * 2 + 1] = fy;
    }

    float* rowsbuf0 = new float[w_out];
    float* rowsbuf1 = new float[w_out];
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    //output w , h boundary
    int w_bound = ceil((w_in - 1) / scale_x);
    int h_bound = ceil((h_in - 1) / scale_y);
    //h_bound loop
    for (int dy = 0; dy < h_bound; dy++){
        int sy = yofs[dy];

        const float* s0 = src + sy * w_in;
        const float* s1 = src + (sy+1) * w_in;

        const float* alphap = alpha;
        float* rows0p = rows0;
        float* rows1p = rows1;

        int dx = 0;
        //w_bound loop
        for (; dx+1 < w_bound; dx += 2){
            int sx = xofs[dx];
            int sxn = xofs[dx+1];
            const float* s0p = s0 + sx;
            const float* s1p = s1 + sx;
            const float* s0np = s0 + sxn;
            const float* s1np = s1 + sxn;

            float32x4_t _a = vld1q_f32(alphap);
            float32x2_t _s0 = vld1_f32(s0p);
            float32x2_t _s1 = vld1_f32(s1p);
            float32x2_t _s0n = vld1_f32(s0np);
            float32x2_t _s1n = vld1_f32(s1np);

            float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
            float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
            float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
            float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

            float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
            vst1_f32(rows0p + dx, _rows0);
            float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
            vst1_f32(rows1p + dx, _rows1);

            alphap += 4;

        }
        //w_bound remain loop
        for (; dx < w_bound; dx++){
            int sx = xofs[dx];
            const float* s0p = s0 + sx;
            const float* s1p = s1 + sx;

            float a0 = alphap[0];
            float a1 = alphap[1];
            rows0p[dx] = s0p[0]*a0 + s0p[1]*a1;
            rows1p[dx] = s1p[0]*a0 + s1p[1]*a1;

            alphap += 2;
        }

        const float buffer1[2] = {*(src + sy * w_in + w_in - 1), 0.0f};
        const float buffer2[2] = {*(src + (sy+1) * w_in + w_in - 1), 0.0f};
        //w_bound - w_out loop
        for (; dx+1 < w_out; dx += 2){

            const float* s0p = buffer1;
            const float* s1p = buffer2;
            const float* s0np = buffer1;
            const float* s1np = buffer2;

            float32x4_t _a = vld1q_f32(alphap);
            float32x2_t _s0 = vld1_f32(s0p);
            float32x2_t _s1 = vld1_f32(s1p);
            float32x2_t _s0n = vld1_f32(s0np);
            float32x2_t _s1n = vld1_f32(s1np);

            float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
            float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
            float32x4_t _s1s1n = vcombine_f32(_s1, _s1n);
            float32x4_t _ms1 = vmulq_f32(_s1s1n, _a);

            float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
            vst1_f32(rows0p + dx, _rows0);
            float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));
            vst1_f32(rows1p + dx, _rows1);

            alphap += 4;
        }
        //w_bound - w_out remain loop
        for (; dx < w_out; dx++){
            const float* s0p = buffer1;
            const float* s1p = buffer2;

            float a0 = alphap[0];
            float a1 = alphap[1];
            rows0p[dx] = s0p[0]*a0 + s0p[1]*a1;
            rows1p[dx] = s1p[0]*a0 + s1p[1]*a1;

            alphap += 2;
        }

        float b0 = beta[0];
        float b1 = beta[1];

        float* dp = dst + dy * w_out;

        int nn = w_out >> 3;
        int remain = w_out - (nn << 3);

#ifdef __aarch64__
        float32x4_t _b0 = vdupq_n_f32(b0);
        float32x4_t _b1 = vdupq_n_f32(b1);
        //calculate and store results
        for (; nn>0; nn--){
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _d = vmulq_f32(_rows0, _b0);
            float32x4_t _rows1 = vld1q_f32(rows1p);
            _d = vmlaq_f32(_d, _rows1, _b1);

            float32x4_t _rows0n = vld1q_f32(rows0p + 4);
            float32x4_t _rows1n = vld1q_f32(rows1p + 4);

            float32x4_t _dn = vmulq_f32(_rows0n, _b0);
            vst1q_f32(dp, _d);
            _dn = vmlaq_f32(_dn, _rows1n, _b1);
            vst1q_f32(dp+4, _dn);

            dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#else
        if (nn > 0){
            asm volatile(
                "vdup.32 q0, %[b0]                   @dup b0 to q1\n"
                "vdup.32 q1, %[b1]                   @dup b1 to q0\n"
                "1:                                                      \n"
                "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
                "vld1.32 {d6-d7}, [%[rows1p]]!       @loads rows0p to q3\n"
                "vmul.f32 q2, q2, q0                 @mul\n"
                "vmla.f32 q2, q3, q1                 @mul add\n"
                "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
                "pld [%[rows0p]]                     @preload rows0p\n"

                "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
                "vld1.32 {d6-d7}, [%[rows1p]]!       @load rows1p to q3\n"
                "vmul.f32 q2, q2, q0                 @mul\n"
                "vmla.f32 q2, q3, q1                 @mul add\n"
                "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
                "pld [%[rows1p]]                     @preload rows1p\n"
                "subs %[loopc], #1                   @loop count minus #1\n"
                "bne 1b                              @jump to 1\n"
                :[rows0p]"+r"(rows0p), [rows1p]"+r"(rows1p), [out]"+r"(dp), [loopc]"+r"(nn)
                :[b0]"r"(b0), [b1]"r"(b1)
                :"cc", "memory", "q0", "q1", "q2", "q3"
            );
        }
#endif
        //calculate and store remain resluts
        for (; remain; --remain){
            *dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }
        beta += 2;
    }

//h_bound - h_out loop
for (int dy = h_bound; dy < h_out; dy++){
        int sy = h_in - 1;
        const float* s0 = src + sy * w_in;
        const float* alphap = alpha;
        float* rows0p = rows0;
        float* rows1p = rows1;

        int dx = 0;
        //w_bound loop
        for (; dx+1 < w_bound; dx += 2){
            int sx = xofs[dx];
            int sxn = xofs[dx+1];
            const float* s0p = s0 + sx;
            const float* s0np = s0 + sxn;

            float32x4_t _a = vld1q_f32(alphap);
            float32x2_t _s0 = vld1_f32(s0p);
            float32x2_t _s0n = vld1_f32(s0np);
            float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);

            float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
            float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
            float32x2_t _rows1 = vdup_n_f32(0.0f);
            vst1_f32(rows0p + dx, _rows0);
            vst1_f32(rows1p + dx, _rows1);

            alphap += 4;

        }
        //w_bound remain loop
        for (; dx < w_bound; dx++){
            int sx = xofs[dx];
            const float* s0p = s0 + sx;
            float a0 = alphap[0];
            float a1 = alphap[1];
            rows0p[dx] = s0p[0]*a0 + s0p[1]*a1;
            rows1p[dx] = 0.0f;

            alphap += 2;
        }

        const float buffer1[2] = {*(src + sy * w_in + w_in - 1), 0.0f};
        //w_bound - w_out loop
        for (; dx+1 < w_out; dx += 2){

            const float* s0p = buffer1;
            const float* s0np = buffer1;

            float32x4_t _a = vld1q_f32(alphap);
            float32x2_t _s0 = vld1_f32(s0p);
            float32x2_t _s0n = vld1_f32(s0np);

            float32x4_t _s0s0n = vcombine_f32(_s0, _s0n);
            float32x4_t _ms0 = vmulq_f32(_s0s0n, _a);
            float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
            float32x2_t _rows1 = vdup_n_f32(0.0f);

            vst1_f32(rows0p + dx, _rows0);
            vst1_f32(rows1p + dx, _rows1);

            alphap += 4;
        }
        //w_bound - wout remain loop
        for (; dx < w_out; dx++){
            const float* s0p = buffer1;
            float a0 = alphap[0];
            float a1 = alphap[1];
            rows0p[dx] = s0p[0]*a0 + s0p[1]*a1;
            rows1p[dx] = 0.0f;
            alphap += 2;
        }

        float b0 = beta[0];
        float b1 = beta[1];

        float* dp = dst + dy * w_out;

        int nn = w_out >> 3;
        int remain = w_out - (nn << 3);

#ifdef __aarch64__
        float32x4_t _b0 = vdupq_n_f32(b0);
        float32x4_t _b1 = vdupq_n_f32(b1);
        //calculate and store results
        for (; nn>0; nn--){
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _d = vmulq_f32(_rows0, _b0);
            float32x4_t _rows1 = vld1q_f32(rows1p);
            _d = vmlaq_f32(_d, _rows1, _b1);

            float32x4_t _rows0n = vld1q_f32(rows0p + 4);
            float32x4_t _rows1n = vld1q_f32(rows1p + 4);

            float32x4_t _dn = vmulq_f32(_rows0n, _b0);
            vst1q_f32(dp, _d);
            _dn = vmlaq_f32(_dn, _rows1n, _b1);
            vst1q_f32(dp+4, _dn);

            dp += 8;
            rows0p += 8;
            rows1p += 8;
        }

#else
        if (nn > 0){
            asm volatile(
                "vdup.32 q0, %[b0]                   @dup b0 to q1\n"
                "vdup.32 q1, %[b1]                   @dup b1 to q0\n"
                "1:                                                      \n"
                "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
                "vld1.32 {d6-d7}, [%[rows1p]]!       @loads rows0p to q3\n"
                "vmul.f32 q2, q2, q0                 @mul\n"
                "vmla.f32 q2, q3, q1                 @mul add\n"
                "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
                "pld [%[rows0p]]                     @preload rows0p\n"

                "vld1.32 {d4-d5}, [%[rows0p]]!       @loads rows0p to q2\n"
                "vld1.32 {d6-d7}, [%[rows1p]]!       @load rows1p to q3\n"
                "vmul.f32 q2, q2, q0                 @mul\n"
                "vmla.f32 q2, q3, q1                 @mul add\n"
                "vst1.32 {d4-d5}, [%[out]]!          @store out to q2 \n"
                "pld [%[rows1p]]                     @preload rows1p\n"
                "subs %[loopc], #1                   @loop count minus #1\n"
                "bne 1b                              @jump to 1\n"
                :[rows0p]"+r"(rows0p), [rows1p]"+r"(rows1p), [out]"+r"(dp), [loopc]"+r"(nn)
                :[b0]"r"(b0), [b1]"r"(b1)
                :"cc", "memory", "q0", "q1", "q2", "q3"
            );
        }
#endif
        //calculate and store remain results
        for (; remain; --remain){
            *dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }
    delete[] buf;
    delete[] rowsbuf0;
    delete[] rowsbuf1;
}
void resize_nearest_kernel(const float* src, int w_in, int h_in, float* dst, int w_out, \
                           int h_out, float scale_x, float scale_y, ResizeType resize_type){
    float scale_w_new = (float)(w_in - 1) / (w_out - 1);
    float scale_h_new = (float)(h_in - 1) / (h_out - 1);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < h_out; ++h) {
        for (int w = 0; w < w_out; ++w) {

            int near_x = static_cast<int>(scale_w_new * w + 0.5);
            int near_y = static_cast<int>(scale_h_new * h + 0.5);
            near_x = near_x < 0 ? 0 : near_x;
            near_y = near_y < 0 ? 0 : near_y;
            dst[h * w_out + w] = src[near_y * w_in + near_x];

            // for (int n = 0; n < n_in; ++n) {
            //     for (int c = 0; c < c_in; ++c) {
            //         int src_index = n * src_stride_batch + c * src_stride_channel;
            //         int dst_index = n * dst_stride_batch + c * dst_stride_channel + h * dst_stride_h + w * dst_stride_w;
            //         dst[dst_index] = src[src_index + near_y * src_stride_h + near_x * src_stride_w];
            //     }
            // }
        }
    }
}

template <>
SaberStatus SaberResize<ARM, AK_FLOAT>::create(const std::vector<Tensor<ARM> *>& inputs,
                            std::vector<Tensor<ARM> *>& outputs,
                            ResizeParam<ARM>& param, Context<ARM> &ctx) {
    if (param.out_width != -1 && param.out_height != -1){
        _width_scale = (float)param.out_width / inputs[0]->width();
        _height_scale = (float)param.out_height / inputs[0]->height();
    } else {
        _width_scale = param.width_scale;
        _height_scale = param.height_scale;
    }
    if (inputs.size() > 1){
        int* out_size_data = static_cast<int*>(inputs[1]->data());
        int h_out = out_size_data[0];
        int w_out = out_size_data[1];
        int num_cout = outputs[0]->num();
        int c_cout = outputs[0]->channel();
        outputs[0]->reshape(Shape({num_cout, c_cout, h_out, w_out}));
    }
    _resize_type = param.resize_type;
    switch (_resize_type){
        case BILINEAR_ALIGN:
            _impl = resize_bilinear;
            break;
        case BILINEAR_NO_ALIGN:
            _impl = resize_bilinear;
            break;
        case RESIZE_CUSTOM:
            _impl = resize_bilinear_custom;
            break;
        case NEAREST_ALIGN:
            _impl = resize_nearest_kernel;
            break;
        default:
            LOG(ERROR) << "unimply resize type: " << _resize_type;
    }
    return SaberSuccess;
}

template <>
SaberStatus SaberResize<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        ResizeParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    float* dout = static_cast<float*>(outputs[0]->mutable_data());
    const float* din = static_cast<const float*>(inputs[0]->data());

    int out_num = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int count = out_num * out_c;
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    int spatial_in = in_h * in_w;
    int spatial_out = out_h * out_w;

#pragma omp parallel for
    for (int i = 0; i < count; ++i){
        _impl(din + i * spatial_in, in_w, in_h, dout + i * spatial_out, out_w, out_h, \
            1.f / _width_scale, 1.f / _height_scale, _resize_type);
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Resize : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("Resize", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberResize, ResizeParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberResize, ResizeParam, ARM, AK_INT8);

} //namespace anakin

} //namespace anakin
