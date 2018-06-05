#include "saber/lite/funcs/neon/impl/pooling_arm_impl.h"
#include <limits>
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

void pooling_basic(Tensor<float>& tensor_out, Tensor<float>& tensor_in, \
    PoolingType type, bool global, int kernel_w, int kernel_h, \
    int stride_w, int stride_h, int pad_w, int pad_h) {
    //no need to pad input tensor, border is zero pad inside this function
    int w_in = tensor_in.width();
    int h_in = tensor_in.height();
    int ch_in = tensor_in.channel();
    int num = tensor_in.num();

    int size_channel_in = w_in * h_in;

    int w_out = tensor_out.width();
    int h_out = tensor_out.height();
    int ch_out = tensor_out.channel();

    int size_channel_out = w_out * h_out;

    float* data_out = tensor_out.mutable_data();
    const float* data_in = tensor_in.data();

    if (global) {
        switch (type) {
            case Pooling_max:
                for (int n = 0; n < num; ++n) {
                    float* data_out_batch = data_out + n * ch_out * size_channel_out;
                    const float* data_in_batch = data_in + n * ch_in * size_channel_in;
#pragma omp parallel for
                    for (int c = 0; c < ch_out; ++c) {
                        const float* data_in_channel = data_in_batch + c * size_channel_in;//in address
                        data_out_batch[c] = data_in_channel[0];
                        for (int i = 0; i < size_channel_in; ++i) {
                            data_out_batch[c] = data_out_batch[c] > data_in_channel[i] ? \
                            data_out_batch[c] : data_in_channel[i];
                        }
                    }
                }
                break;

            case Pooling_average_include_padding:

            case Pooling_average_exclude_padding:
                for (int n = 0; n < num; ++n) {
                    float* data_out_batch = data_out + n * ch_out * size_channel_out;
                    const float* data_in_batch = data_in + n * ch_in * size_channel_in;
#pragma omp parallel for
                    for (int c = 0; c < ch_out; ++c) {
                        const float* data_in_channel = data_in_batch + c * size_channel_in;//in address
                        float sum = 0.f;
                        for (int i = 0; i < size_channel_in; ++i) {
                            sum += data_in_channel[i];
                        }
                        data_out_batch[c] = sum / size_channel_in;
                    }
                }
                break;
            default:
                LOG(INFO) << "not support";
        }
        return;
    }

    switch (type) {
        case Pooling_max:
            for (int n = 0; n < num; ++n) {
                float* data_out_channel = data_out + n * ch_out * size_channel_out;
                const float* data_in_batch = data_in + n * ch_in * size_channel_in;
#pragma omp parallel for
                for (int q = 0; q < ch_out; q++) {

                    float* data_out_row = data_out_channel + q * size_channel_out;
                    const float* data_in_channel = data_in_batch + q * size_channel_in;

                    for (int i = 0; i < h_out; i++) {
                        for (int j = 0; j < w_out; j++) {
                            int hstart = i * stride_h - pad_h;
                            int wstart = j * stride_w - pad_w;
                            int hend = std::min(hstart + kernel_h, h_in + pad_h);
                            int wend = std::min(wstart + kernel_w, w_in + pad_w);
                            hstart = std::max(hstart, 0);
                            wstart = std::max(wstart, 0);
                            hend = std::min(hend, h_in);
                            wend = std::min(wend, w_in);

                            data_out_row[j] = data_in_channel[hstart * w_in + wstart];
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    data_out_row[j] = data_out_row[j] > \
                                        data_in_channel[h * w_in + w] ? \
                                        data_out_row[j] : data_in_channel[h * w_in + w];
                                }
                            }
                        }
                        data_out_row += w_out;
                    }
                }
            }
            break;

        case Pooling_average_include_padding:
            for (int n = 0; n < num; ++n) {
                int pool_size = kernel_w * kernel_h;//(hend - hstart) * (wend - wstart);//problem
                float* data_out_channel = data_out + n * ch_out * size_channel_out;
                const float* data_in_batch = data_in + n * ch_in * size_channel_in;
#pragma omp parallel for
                for (int q = 0; q < ch_out; q++) {

                    float* data_out_row = data_out_channel + q * size_channel_out;
                    const float* data_in_channel = data_in_batch + q * size_channel_in;
                    for (int i = 0; i < h_out; i++) {
                        for (int j = 0; j < w_out; j++) {
                            int hstart = i * stride_h - pad_h;
                            int wstart = j * stride_w - pad_w;
                            int hend = std::min(hstart + kernel_h, h_in + pad_h);
                            int wend = std::min(wstart + kernel_w, w_in + pad_w);
                            hstart = std::max(hstart, 0);
                            wstart = std::max(wstart, 0);
                            hend = std::min(hend, h_in);
                            wend = std::min(wend, w_in);

                            data_out_row[j] = data_in_channel[hstart * w_in + wstart];
                            float sum = 0.f;
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    sum += data_in_channel[h * w_in + w];
                                }
                            }
                            data_out_row[j] = sum / pool_size;
                        }
                        data_out_row += w_out;
                    }
                }
            }
            break;
        case Pooling_average_exclude_padding:
            for (int n = 0; n < num; ++n) {
                float* data_out_channel = data_out + n * ch_out * size_channel_out;
                const float* data_in_batch = data_in + n * ch_in * size_channel_in;
#pragma omp parallel for
                for (int q = 0; q < ch_out; q++) {

                    float* data_out_row = data_out_channel + q * size_channel_out;
                    const float* data_in_channel = data_in_batch + q * size_channel_in;
                    for (int i = 0; i < h_out; i++) {
                        for (int j = 0; j < w_out; j++) {
                            int hstart = i * stride_h - pad_h;
                            int wstart = j * stride_w - pad_w;
                            int hend = std::min(hstart + kernel_h, h_in + pad_h);
                            int wend = std::min(wstart + kernel_w, w_in + pad_w);
                            hstart = std::max(hstart, 0);
                            wstart = std::max(wstart, 0);
                            hend = std::min(hend, h_in);
                            wend = std::min(wend, w_in);

                            data_out_row[j] = data_in_channel[hstart * w_in + wstart];
                            float sum = 0.f;
                            for (int h = hstart; h < hend; ++h) {
                                for (int w = wstart; w < wend; ++w) {
                                    sum += data_in_channel[h * w_in + w];
                                }
                            }
                            int pool_size = (hend - hstart) * (wend - wstart);
                            data_out_row[j] = sum / pool_size;
                        }
                        data_out_row += w_out;
                    }
                }
            }
            break;
        default:
            LOG(FATAL) << "not support";
    }
}

void pooling_global(Tensor<float>& tensor_out, Tensor<float>& tensor_in, \
    PoolingType type, bool global, int kernel_w, int kernel_h, \
    int stride_w, int stride_h, int pad_w, int pad_h) {

    int w_in = tensor_in.width();
    int h_in = tensor_in.height();
    int ch_in = tensor_in.channel();
    int num = tensor_in.num();

    int size_channel_in = w_in * h_in;

    int ch_out = tensor_out.channel();

    float* data_out = tensor_out.mutable_data();
    const float* data_in = tensor_in.data();
    
    int cnt = size_channel_in / 8;

    for (int n = 0; n < num; ++n) {
        float* data_out_batch = data_out + n * ch_out;
        const float* data_in_batch = data_in + n * ch_in * size_channel_in;
        if (type == Pooling_max) {
#pragma omp parallel for
            for (int c = 0; c < ch_out; ++c) {
                const float* data_in_channel = data_in_batch + c * size_channel_in;
                int i = 0;
                float32x4_t vmax = vdupq_n_f32(std::numeric_limits<float>::min());
#ifdef __aarch64__
                for(; i < cnt; i++) {
                    float32x4_t vdin1 = vld1q_f32(data_in_channel);
                    vmax = vmaxq_f32(vdin1, vmax);
                    float32x4_t vdin2 = vld1q_f32(data_in_channel + 4);
                    vmax = vmaxq_f32(vmax, vdin2);
                    data_in_channel += 8;
                }
#else
                int num = cnt;
                if (num > 0) {
                    asm volatile(
                    "max_loop:                                        @main loop\n"
                    "vld1.f32   {d0-d1}, [%[data_in_channel]]!        @load q1, data_in_channel\n"
                    "vmax.f32   %q[vmax], %q[vmax], q0                @max vmax, vmax, data_in_channel\n"
                    "vld1.f32   {d2-d3}, [%[data_in_channel]]!        @ load 2nd 4 data"
                    "vmax.f32   %q[vmax], %q[vmax], q1                @ compare 2nd 4 datas\n"
                    "subs       %[num], #1                            @subs num, 1\n"
                    "bne        max_loop                              @bne num\n"
                    :[data_in_channel] "+r" (data_in_channel), [num] "+r" (num), [vmax] "+w" (vmax)
                    :"r" (data_in_channel), "r" (num)
                    : "q0"
                    );
                }
#endif //__aarch64__
                float32x2_t vmax_tmp = vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
                float tmp1 = vget_lane_f32(vmax_tmp, 0);
                float tmp2 = vget_lane_f32(vmax_tmp, 1);
                float max_tmp = tmp1 > tmp2? tmp1 : tmp2;
                for (i = cnt * 8; i < size_channel_in; ++i) {
                    /* code */
                    max_tmp = max_tmp > data_in_channel[0] ? max_tmp : data_in_channel[0];
                    data_in_channel++;
                }
                data_out_batch[c] = max_tmp;
            }
        }
        else {
#pragma omp parallel for
            for(int c = 0;c < ch_out; c++){
                const float* data_in_channel = data_in_batch + c * size_channel_in;//in address
                int i = 0;
                float32x4_t vsum = vdupq_n_f32(0.0f);
#ifdef  __aarch64__
                for(; i < cnt; i++){//
                    vsum = vaddq_f32(vld1q_f32(data_in_channel),vsum);
                    data_in_channel += 4;
                }
#else
                int num = cnt;
                if (num > 0) {
                    asm volatile(
                    "add_loop:                                        @main loop\n"
                    "vld1.f32   {d0-d1}, [%[data_in_channel]]!        @load q1, data_in_channel\n"
                    "vadd.f32   %q[vsum], %q[vsum], q0                @add vmax, vmax, data_in_channel\n"
                    "subs        %[num], #1                           @subs num, 1\n"
                    "bne        add_loop                              @bne num\n"
                    :[data_in_channel] "+r" (data_in_channel), [num] "+r" (num), [vsum] "+w" (vsum)
                    :"r" (data_in_channel), "r" (num), "w" (vsum)
                    : "q0"
                    );
                }
#endif //__aarch64__
                float32x2_t vsum_tmp = vadd_f32(vget_low_f32(vsum),vget_high_f32(vsum));
                float sum = vget_lane_f32(vsum_tmp,0) + vget_lane_f32(vsum_tmp,1);
                for(i = cnt * 4;i < size_channel_in; i++) {
                    sum += data_in_channel[0];
                    data_in_channel++;
                }
                data_out_batch[c] = sum / size_channel_in;
            }
        }
    }
}

void pooling2x2s2_max(Tensor<float>& tensor_out, Tensor<float>& tensor_in, \
    PoolingType type, bool global, int kernel_w, int kernel_h, \
    int stride_w, int stride_h, int pad_w, int pad_h) {

    int w_in = tensor_in.width();
    int h_in = tensor_in.height();
    int ch_in = tensor_in.channel();
    int num = tensor_in.num();

    int w_out = tensor_out.width();
    int h_out = tensor_out.height();
    int ch_out = tensor_out.channel();

    int size_channel_out = w_out * h_out;
    int size_channel_in = w_in * h_in;

    float* data_out = tensor_out.mutable_data();
    const float* data_in = tensor_in.data();

    int right_pad = w_out * 2 - w_in;
    int bottom_pad = h_out * 2 - h_in;

    uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    int col_cnt = (w_out - right_pad) / 4;
    int right_remian = w_out - col_cnt * 4;

    int row_cnt = (h_out - bottom_pad) / 2;
    int bottom_remian = h_out - row_cnt * 2;

    float32x4_t vzero = vdupq_n_f32(0.f);

    for (int n = 0; n < num; ++n) {
        float* data_out_batch = data_out + n * ch_out * size_channel_out;
        const float* data_in_batch = data_in + n * ch_in * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < ch_out; c++) {
            const float* data_in_channel = data_in_batch + c * size_channel_in;
            float* data_out_channel = data_out_batch + c * size_channel_out;
            int h = 0;
            for (; h < h_out - 1; h += 2) {
                const float* r0 = data_in_channel + 2 * h * w_in;
                const float* r1 = r0 + w_in;
                const float* r2 = r1 + w_in;
                const float* r3 = r2 + w_in;
                float* out_r0 = data_out_channel + h * w_out;
                float* out_r1 = out_r0 + w_out;
                for (int w = 0; w < col_cnt; ++w) {
#ifdef __aarch64__
                    //todo
#else
                    asm volatile (
                    "vld1.32 {d0-d3}, [%[ptr0]]!      @ load r0, 8 elements, q0=r00,r01,r02,r03;q1=r04,r05,r06,r07\n"
                    "vld1.32 {d4-d7}, [%[ptr1]]!      @ load r1, 8 elements, q2=r10,r11,r12,r13;q1=r14,r15,r16,r17\n"
                    "vld1.32 {d16-d19}, [%[ptr2]]!    @ load r2, 8 elements, q8=r20,r21,r22,r23;q9=r24,r25,r26,r27\n"
                    "vld1.32 {d20-d23}, [%[ptr3]]!    @ load r3, 8 elements, q10=r30,r31,r32,r33;q11=r34,r35,r36,r37\n"

                    "vmax.f32   q0, q0, q2            @ get max of r0\n"
                    "vmax.f32   q1, q1, q3            @ get max of r0\n"
                    "vpmax.f32   d0, d0, d1           @ get max in q0, q1\n"
                    "vpmax.f32   d1, d2, d3           @ get max in q0, q1\n"
                    "vst1.32    {d0-d1},[%[outptr0]]! @ write q0, 1st row\n"

                    "vmax.f32   q8, q8, q10           @ get result of r2, r3\n"
                    "vmax.f32   q9, q9, q11           @ get result of r2, r3\n"
                    "vpmax.f32   d16, d16, d17        @ get max in q8, q9\n"
                    "vpmax.f32   d17, d18, d19        @ get max in q8, q9\n"
                    "vst1.32  {d16-d17},[%[outptr1]]! @ write q4, 2nd row\n"
                    : [ptr0] "+r" (r0), [ptr1] "+r" (r1), [ptr2] "+r" (r2), \
                        [ptr3] "+r" (r3), [outptr0] "+r" (out_r0), [outptr1] "+r" (out_r1)
                    :
                    : "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
                    );
#endif //__aarch64__

                }
            }
        }
    }
}

void pooling2x2s2_ave(Tensor<float>& tensor_out, Tensor<float>& tensor_in, \
    PoolingType type, bool global, int kernel_w, int kernel_h, \
    int stride_w, int stride_h, int pad_w, int pad_h) {

    int w_in = tensor_in.width();
    int h_in = tensor_in.height();
    int ch_in = tensor_in.channel();
    int num = tensor_in.num();

    int w_out = tensor_out.width();
    int h_out = tensor_out.height();
    int ch_out = tensor_out.channel();

    int size_channel_out = w_out * h_out;
    int size_channel_in = w_in * h_in;
    float* data_out = tensor_out.mutable_data();
    const float* data_in = tensor_in.data();

    int w_even = (w_in >> 1) << 1;
    //int w_remains = w_in - w_even; // should be 0 or 1
    int h_even = (h_in >> 1) << 1;
    //int h_remains = h_in - h_even; // should be 0 or 1
    int w_unroll_size = (w_even >> 3) << 3;
    //int w_unroll_remian = w_even - w_unroll_size;
    int w_in_2 = w_in << 1;
    float32x4_t vcoef = vdupq_n_f32(0.25f); //divided by 4

    for (int n = 0; n < num; ++n) {

        float* data_out_batch = data_out + n * ch_out * size_channel_out;
        const float* data_in_batch = data_in + n * ch_in * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < ch_out; c++) {
            float* data_out_channel = data_out_batch + c * size_channel_out;
            const float* data_in_channel = data_in_batch + c * size_channel_in;
            const float* r0 = data_in_channel;
            const float* r1 = r0 + w_in;
            int h = 0;
            for (; h < h_even; h += 2) {
                int w = 0;
            #ifdef __aarch64__
                for (; w < w_unroll_size; w += 8) {
                    prefetch_2x(r0);
                    prefetch_2x(r1);
                    float32x4_t dr00 = vld1q_f32(&r0[w]);
                    float32x4_t dr01 = vld1q_f32(&r0[w + 4]);
                    float32x4_t dr10 = vld1q_f32(&r1[w]);
                    float32x4_t dr11 = vld1q_f32(&r1[w + 4]);
                    float32x4_t dsum1 = vaddq_f32(dr00, dr10);
                    float32x4_t dsum2 = vaddq_f32(dr01, dr11);
                #ifdef __aarch64__
                    float32x4_t dsum = vpaddq_f32(dsum1, dsum2);
                #else
                    float32x2_t dsuml = vpadd_f32(vget_low_f32(dsum1), vget_high_f32(dsum1));
                    float32x2_t dsumh = vpadd_f32(vget_low_f32(dsum2), vget_high_f32(dsum2));
                    float32x4_t dsum = vcombine_f32(dsuml, dsumh);
                #endif
                    float32x4_t res = vmulq_f32(dsum, vcoef);
                    vst1q_f32(&data_out_channel[w >> 1], res);

                }
            #else
                w = w_unroll_size;
                int num = w_unroll_size >> 3;
                float* dr0 = (float *)r0;
                float* dr1 = (float *)r1;
                float* dr_out = data_out_channel;
                //printf("c: %d, num: %d, dr0: %x, dr1: %x, dr_out: %x\n",c,num,dr0,dr1,dr_out);
                if (num > 0){
                    asm volatile(
                    "s2_ave_loop:                                     @main loop\n"  
                    "vld1.f32   {d0-d3}, [%[dr0]]!                    @load q0, dr0\n"
                    "vld1.f32   {d4-d7}, [%[dr1]]!                    @load q1, dr1\n"
                    "vadd.f32   q0, q0, q2                            @add q0, q0, q2\n"
                    "vadd.f32   q1, q1, q3                            @add q1, q1, q2\n"
                    "vpadd.f32  d4, d0, d1                            @add d4, d0, d1\n"
                    "vpadd.f32  d5, d2, d3                            @add d5, d2, d3\n"
                    "vmul.f32   q2, q2, %q[vcoef]                      @mul q2, q2, vcoef\n"
                    "vst1.f32   {d4-d5}, [%[dr_out]]!                 @vst1 q2, dr_out\n"
                    "subs       %[num], #1                            @subs num, 1\n"
                    "bne        s2_ave_loop                           @bne num\n"
                    :[dr0] "+r" (dr0), [dr1] "+r" (dr1), [dr_out] "+r" (dr_out), [vcoef] "+w" (vcoef), [num] "+r" (num)
                    :"r" (dr0), "r" (dr1), "r" (dr_out), "r" (num), "w" (vcoef)
                    :"q0", "q1", "q2", "q3"
                    );
                }
            #endif //__aarch64__
                for (; w < w_even; w += 2) {
                    data_out_channel[w >> 1] = (r0[w] + r0[w + 1] + r1[w] + r1[w + 1]) / 4.f;
                }
                for (; w < w_in; ++w) { // run 0 or 1 time
                    data_out_channel[w >> 1] = (r0[w] + r1[w]) / 4.f;
                }
                r0 += w_in_2;// << 1;
                r1 += w_in_2;// << 1;
                data_out_channel += w_out;
            }
            // process remain row (odd, last row)
            for (; h < h_in; h++) { //run 0 or 1 time
                int w = 0;
            #ifdef __aarch64__
                for (; w < w_unroll_size; w += 8) {
                    prefetch_2x(r0);
                    float32x4_t dr00 = vld1q_f32(&r0[w]);
                    float32x4_t dr01 = vld1q_f32(&r0[w + 4]);
                #ifdef __aarch64__
                    float32x4_t dsum = vpaddq_f32(dr00, dr01);
                #else
                    float32x2_t dsuml = vpadd_f32(vget_low_f32(dr00), vget_high_f32(dr00));
                    float32x2_t dsumh = vpadd_f32(vget_low_f32(dr01), vget_high_f32(dr01));
                    float32x4_t dsum = vcombine_f32(dsuml, dsumh);
                #endif
                    float32x4_t res = vmulq_f32(dsum, vcoef);
                    vst1q_f32(&data_out_channel[w >> 1], res);

                }
            #else
                w = w_unroll_size;
                int num = w_unroll_size >> 3;
                float* dr0 = (float *)r0;
                float* dr_out = data_out_channel;
                //printf("c: %d, num: %d, dr0: %x, dr1: %x, dr_out: %x\n",c,num,dr0,dr1,dr_out);
                if (num > 0){
                    asm volatile(
                    "s2_ave_loop1:                                    @main loop\n"  
                    "vld1.f32   {d0-d3}, [%[dr0]]!                    @load q0, dr0\n"
                    "vpadd.f32  d4, d0, d1                            @add d4, d0, d1\n"
                    "vpadd.f32  d5, d2, d3                            @add d5, d2, d3\n"
                    "vmul.f32   q2, q2, %q[vcoef]                      @mul q2, q2, vcoef\n"
                    "vst1.f32   {d4-d5}, [%[dr_out]]!                 @vst1 q2, dr_out\n"
                    "subs       %[num], #1                            @subs num, 1\n"
                    "bne        s2_ave_loop                           @bne num\n"
                    :[dr0] "+r" (dr0), [dr_out] "+r" (dr_out), [vcoef] "+w" (vcoef), [num] "+r" (num)
                    :"r" (dr0), "r" (dr_out), "r" (num), "w" (vcoef)
                    :"q0", "q1", "q2"
                    );
                }
            #endif //__aarch64__
                for (; w < w_even; w += 2) {
                    data_out_channel[w >> 1] = (r0[w] + r0[w + 1]) / 4.f;
                }
                for (; w < w_in; ++w) { // run 0 or 1 time
                    data_out_channel[w >> 1] = r0[w] / 4.f;
                }
            }
        }

    }
}

void pooling3x3s2_max(Tensor<float>& tensor_out, Tensor<float>& tensor_in, \
    PoolingType type, bool global, int kernel_w, int kernel_h, \
    int stride_w, int stride_h, int pad_w, int pad_h) {
    //todo
}

void pooling3x3s2_ave(Tensor<float>& tensor_out, Tensor<float>& tensor_in, \
    PoolingType type, bool global, int kernel_w, int kernel_h, \
    int stride_w, int stride_h, int pad_w, int pad_h) {
    //todo
}

}  //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE
