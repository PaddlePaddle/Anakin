#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

#define AKMAX(a, b) (a) > (b)? (a) : (b)

void conv_3x3s1_direct_bias_int8(int* dout, const signed char* din, \
                          const signed char* weights, const int* bias, bool flag_bias, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, Context* ctx);

void conv_3x3s1_direct_bias_relu_int8(int* dout, const signed char* din, \
                          const signed char* weights, const  int* bias, bool flag_bias, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, Context* ctx);

void conv_3x3s1_direct_bias_s_int8(int* dout, const signed char* din, \
                          const signed char* weights, const  int* bias, bool flag_bias, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, Context* ctx);

void conv_3x3s1_direct_bias_relu_s_int8(int* dout, const signed char* din, \
                          const signed char* weights, const  int* bias, bool flag_bias, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, Context* ctx);


void conv_3x3s1_direct_int8(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, \
                          Context* ctx, void* work_space, const void* idx_ptr){
    if (flag_relu){
        if (win > 8){
            conv_3x3s1_direct_bias_relu_int8(static_cast<int*>(dout), static_cast<const signed char*>(din), \
                    static_cast<const signed char*>(weights), \
                    static_cast<const int*>(bias), flag_bias, \
                    num, chout, hout, wout, chin, hin, win, group, kernel_w, kernel_h, \
                    stride_w, stride_h, dila_w, dila_h, pad_w, pad_h, ctx);
        }else{
            conv_3x3s1_direct_bias_relu_s_int8(static_cast<int*>(dout), static_cast<const signed char*>(din), \
                    static_cast<const signed char*>(weights), \
                    static_cast<const int*>(bias), flag_bias, \
                    num, chout, hout, wout, chin, hin, win, group, kernel_w, kernel_h, \
                    stride_w, stride_h, dila_w, dila_h, pad_w, pad_h, ctx);
        }
    }else{
        if (win > 8){
            conv_3x3s1_direct_bias_int8(static_cast<int*>(dout), static_cast<const signed char*>(din), \
                    static_cast<const signed char*>(weights), \
                    static_cast<const int*>(bias), flag_bias, \
                    num, chout, hout, wout, chin, hin, win, group, kernel_w, kernel_h, \
                    stride_w, stride_h, dila_w, dila_h, pad_w, pad_h, ctx);
        }else{
            conv_3x3s1_direct_bias_s_int8(static_cast<int*>(dout), static_cast<const signed char*>(din), \
                    static_cast<const signed char*>(weights), \
                    static_cast<const int*>(bias), flag_bias, \
                    num, chout, hout, wout, chin, hin, win, group, kernel_w, kernel_h, \
                    stride_w, stride_h, dila_w, dila_h, pad_w, pad_h, ctx);
        }

    }
}


#ifdef __aarch64__

template <typename Dtype>
inline void prefetch(const Dtype *din) {
    asm volatile(
    "PRFM PLDL1KEEP, [%[din]] \n"
    :
    : [din] "r"(din)
    : "memory");
}

void conv_3x3s1_direct_bias_int8(int* dout, const signed char* din, \
                          const signed char* weights, const  int* bias, bool flag_bias, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, Context* ctx) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    // printf("conv3x3s1_direct_int8 start \n");
    int threads = ctx->get_threads();
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const short unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, win * sizeof(signed char));

    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    int w_stride = chin * 9;

    int tile_w = (win + 7) >> 3;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(win - 7 - (cnt_col << 3));

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)(wout - ((cnt_col + 1) << 3));
    uint16x8_t vmask_result = vcgtq_u16(vdupq_n_u16(rst_remain), vld1q_u16(right_pad_rst));

    int8x8_t vzero = vdup_n_s8(0);
    int16x8_t vzero_16 = vdupq_n_s16(0);
    int zero[4] = {0, 0, 0, 0};

    int ch = ((chout >> 2) << 2);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * chin * size_in_channel;
        int *dout_batch = dout + n * chout * size_out_channel;
#pragma omp parallel for num_threads(threads)
        for (int c = 0; c < chout - 3; c += 4) {

            int* dout_c0 = dout_batch + c * size_out_channel;
            int* dout_c1 = dout_c0 + size_out_channel;
            int* dout_c2 = dout_c1 + size_out_channel;
            int* dout_c3 = dout_c2 + size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c0, &bias[c], 1, size_out_channel);
                fill_bias_int8(dout_c1, &bias[c + 1], 1, size_out_channel);
                fill_bias_int8(dout_c2, &bias[c + 2], 1, size_out_channel);
                fill_bias_int8(dout_c3, &bias[c + 3], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c0, zero, 1, size_out_channel);
                fill_bias_int8(dout_c1, zero, 1, size_out_channel);
                fill_bias_int8(dout_c2, zero, 1, size_out_channel);
                fill_bias_int8(dout_c3, zero, 1, size_out_channel);
            }

            const signed char* wc0 = weights + c * w_stride;
            const signed char* wc1 = wc0 + w_stride;
            const signed char* wc2 = wc1 + w_stride;
            const signed char* wc3 = wc2 + w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *din_ch = din_batch + i * size_in_channel;
                int *doutc0_ptr = dout_c0;
                int *doutc1_ptr = dout_c1;
                int *doutc2_ptr = dout_c2;
                int *doutc3_ptr = dout_c3;

                int *doutc0r0 = nullptr;
                int *doutc1r0 = nullptr;
                int *doutc2r0 = nullptr;
                int *doutc3r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    doutc1r0 = doutc1_ptr;
                    doutc2r0 = doutc2_ptr;
                    doutc3r0 = doutc3_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }
                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    //prefetch input
                    prefetch(doutc0r0);
                    prefetch(doutc1r0);
                    prefetch(doutc2r0);
                    prefetch(doutc3r0);

                    prefetch(din_ptr0);
                    prefetch(din_ptr1);
                    prefetch(din_ptr2);
                    prefetch(wc0);
                    prefetch(wc1);
                    prefetch(wc2);
                    prefetch(wc3);

                    //left
                    if (1){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//b = 8910111213
                        int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);
                        int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);

                        int8x8_t w01 = vdup_n_s8(wc0[1]);
                        int8x8_t w11 = vdup_n_s8(wc1[1]);
                        int8x8_t w21 = vdup_n_s8(wc2[1]);
                        int8x8_t w31 = vdup_n_s8(wc3[1]);

                        int8x8_t tmp00 = vext_s8(vzero, vinr0, 7); //c = -1 0 1 2
                        int8x8_t tmp10 = vext_s8(vzero, vinr1, 7); //c = -1 0 1 2
                        int8x8_t tmp20 = vext_s8(vzero, vinr2, 7); //c = -1 0 1 2

                        int8x8_t tmp01 = vext_s8(vinr0, vinr0_1, 1); //d = 1 2 3 4
                        int8x8_t tmp11 = vext_s8(vinr1, vinr1_1, 1); //d = 1 2 3 4
                        int8x8_t tmp21 = vext_s8(vinr2, vinr2_1, 1); //d = 1 2 3 4

                        int8x8_t w00 = vdup_n_s8(wc0[0]);
                        int8x8_t w10 = vdup_n_s8(wc1[0]);
                        int8x8_t w20 = vdup_n_s8(wc2[0]);
                        int8x8_t w30 = vdup_n_s8(wc3[0]);

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w01);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(vinr0, w11);// a = 0 1 2 3
                        int16x8_t vdata20 = vmull_s8(vinr0, w21);// a = 0 1 2 3
                        int16x8_t vdata30 = vmull_s8(vinr0, w31);// a = 0 1 2 3

                        int8x8_t w02 = vdup_n_s8(wc0[2]);
                        int8x8_t w12 = vdup_n_s8(wc1[2]);
                        int8x8_t w22 = vdup_n_s8(wc2[2]);
                        int8x8_t w32 = vdup_n_s8(wc3[2]);

                        vdata00 = vmlal_s8(vdata00, tmp00, w00);// c = -1 0 1 2
                        vdata10 = vmlal_s8(vdata10, tmp00, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp00, w20);// c = -1 0 1 2
                        vdata30 = vmlal_s8(vdata30, tmp00, w30);// c = -1 0 1 2

                        w00 = vdup_n_s8(wc0[3]);
                        w10 = vdup_n_s8(wc1[3]);
                        w20 = vdup_n_s8(wc2[3]);
                        w30 = vdup_n_s8(wc3[3]);

                        vdata00 = vmlal_s8(vdata00, tmp01, w02);// d = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp01, w12);// d = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp01, w22);// d = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp01, w32);// d = 1 2 3 4

                        //r1
                        w01 = vdup_n_s8(wc0[4]);
                        w11 = vdup_n_s8(wc1[4]);
                        w21 = vdup_n_s8(wc2[4]);
                        w31 = vdup_n_s8(wc3[4]);

                        vdata00 = vmlal_s8(vdata00, tmp10, w00);// c = -1 0 1 2
                        vdata10 = vmlal_s8(vdata10, tmp10, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp10, w20);// c = -1 0 1 2
                        vdata30 = vmlal_s8(vdata30, tmp10, w30);// c = -1 0 1 2

                        w02 = vdup_n_s8(wc0[5]);
                        w12 = vdup_n_s8(wc1[5]);
                        w22 = vdup_n_s8(wc2[5]);
                        w32 = vdup_n_s8(wc3[5]);

                        vdata00 = vmlal_s8(vdata00, vinr1, w01);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr1, w11);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr1, w21);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr1, w31);// a = 0 1 2 3

                        w00 = vdup_n_s8(wc0[6]);
                        w10 = vdup_n_s8(wc1[6]);
                        w20 = vdup_n_s8(wc2[6]);
                        w30 = vdup_n_s8(wc3[6]);

                        vdata00 = vmlal_s8(vdata00, tmp11, w02);// d = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp11, w12);// d = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp11, w22);// d = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp11, w32);// d = 1 2 3 4

                        //r2
                        w01 = vdup_n_s8(wc0[7]);
                        w11 = vdup_n_s8(wc1[7]);
                        w21 = vdup_n_s8(wc2[7]);
                        w31 = vdup_n_s8(wc3[7]);

                        vdata00 = vmlal_s8(vdata00, tmp20, w00);// c = -1 0 1 2
                        vdata10 = vmlal_s8(vdata10, tmp20, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp20, w20);// c = -1 0 1 2
                        vdata30 = vmlal_s8(vdata30, tmp20, w30);// c = -1 0 1 2

                        w02 = vdup_n_s8(wc0[8]);
                        w12 = vdup_n_s8(wc1[8]);
                        w22 = vdup_n_s8(wc2[8]);
                        w32 = vdup_n_s8(wc3[8]);

                        vdata00 = vmlal_s8(vdata00, vinr2, w01);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr2, w11);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr2, w21);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr2, w31);// a = 0 1 2 3

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr10 = vld1q_s32(doutc1r0);
                        int32x4_t outr20 = vld1q_s32(doutc2r0);
                        int32x4_t outr30 = vld1q_s32(doutc3r0);

                        int16x8_t vdata00_1 = vmull_s8(tmp21, w02);// d = 1 2 3 4
                        int16x8_t vdata10_1 = vmull_s8(tmp21, w12);// d = 1 2 3 4
                        int16x8_t vdata20_1 = vmull_s8(tmp21, w22);// d = 1 2 3 4
                        int16x8_t vdata30_1 = vmull_s8(tmp21, w32);// d = 1 2 3 4

                        // vdata00 = vmlal_s8(vdata00, tmp21, w02);// d = 1 2 3 4
                        // vdata10 = vmlal_s8(vdata10, tmp21, w12);// d = 1 2 3 4
                        // vdata20 = vmlal_s8(vdata20, tmp21, w22);// d = 1 2 3 4
                        // vdata30 = vmlal_s8(vdata30, tmp21, w32);// d = 1 2 3 4

                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);
                        int32x4_t outr11 = vld1q_s32(doutc1r0 + 4);
                        int32x4_t outr21 = vld1q_s32(doutc2r0 + 4);
                        int32x4_t outr31 = vld1q_s32(doutc3r0 + 4);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30));

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00_1));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10_1));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20_1));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30_1));

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00_1));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10_1));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20_1));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30_1));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc1r0, outr10);
                        vst1q_s32(doutc2r0, outr20);
                        vst1q_s32(doutc3r0, outr30);

                        din_ptr0 += 7;
                        din_ptr1 += 7;
                        din_ptr2 += 7;

                        vst1q_s32(doutc0r0 + 4, outr01);
                        vst1q_s32(doutc1r0 + 4, outr11);
                        vst1q_s32(doutc2r0 + 4, outr21);
                        vst1q_s32(doutc3r0 + 4, outr31);

                        doutc0r0 += 8;
                        doutc1r0 += 8;
                        doutc2r0 += 8;
                        doutc3r0 += 8;
                    }
                    for (int j = 0; j < cnt_col; j++){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//b = 8910111213
                        int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);
                        int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);

                        int8x8_t w00 = vdup_n_s8(wc0[0]);
                        int8x8_t w10 = vdup_n_s8(wc1[0]);
                        int8x8_t w20 = vdup_n_s8(wc2[0]);
                        int8x8_t w30 = vdup_n_s8(wc3[0]);

                        int8x8_t tmp00 = vext_s8(vinr0, vinr0_1, 1); //c = 1 2 3 4
                        int8x8_t tmp10 = vext_s8(vinr1, vinr1_1, 1); //c = 1 2 3 4
                        int8x8_t tmp20 = vext_s8(vinr2, vinr2_1, 1); //c = 1 2 3 4

                        int8x8_t tmp01 = vext_s8(vinr0, vinr0_1, 2); //d = 2 3 4 5
                        int8x8_t tmp11 = vext_s8(vinr1, vinr1_1, 2); //d = 2 3 4 5
                        int8x8_t tmp21 = vext_s8(vinr2, vinr2_1, 2); //d = 2 3 4 5

                        int8x8_t w01 = vdup_n_s8(wc0[1]);
                        int8x8_t w11 = vdup_n_s8(wc1[1]);
                        int8x8_t w21 = vdup_n_s8(wc2[1]);
                        int8x8_t w31 = vdup_n_s8(wc3[1]);

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w00);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(vinr0, w10);// a = 0 1 2 3
                        int16x8_t vdata20 = vmull_s8(vinr0, w20);// a = 0 1 2 3
                        int16x8_t vdata30 = vmull_s8(vinr0, w30);// a = 0 1 2 3

                        int8x8_t w02 = vdup_n_s8(wc0[2]);
                        int8x8_t w12 = vdup_n_s8(wc1[2]);
                        int8x8_t w22 = vdup_n_s8(wc2[2]);
                        int8x8_t w32 = vdup_n_s8(wc3[2]);

                        vdata00 = vmlal_s8(vdata00, tmp00, w01);//c = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp00, w11);//c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp00, w21);//c = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp00, w31);//c = 1 2 3 4

                        w00 = vdup_n_s8(wc0[3]);
                        w10 = vdup_n_s8(wc1[3]);
                        w20 = vdup_n_s8(wc2[3]);
                        w30 = vdup_n_s8(wc3[3]);

                        vdata00 = vmlal_s8(vdata00, tmp01, w02);//d = 2 3 4 5
                        vdata10 = vmlal_s8(vdata10, tmp01, w12);//d = 2 3 4 5
                        vdata20 = vmlal_s8(vdata20, tmp01, w22);//d = 2 3 4 5
                        vdata30 = vmlal_s8(vdata30, tmp01, w32);//d = 2 3 4 5

                        //r1
                        w01 = vdup_n_s8(wc0[4]);
                        w11 = vdup_n_s8(wc1[4]);
                        w21 = vdup_n_s8(wc2[4]);
                        w31 = vdup_n_s8(wc3[4]);

                        vdata00 = vmlal_s8(vdata00, vinr1, w00);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr1, w10);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr1, w20);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr1, w30);// a = 0 1 2 3

                        w02 = vdup_n_s8(wc0[5]);
                        w12 = vdup_n_s8(wc1[5]);
                        w22 = vdup_n_s8(wc2[5]);
                        w32 = vdup_n_s8(wc3[5]);

                        vdata00 = vmlal_s8(vdata00, tmp10, w01);//c = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp10, w11);//c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp10, w21);//c = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp10, w31);//c = 1 2 3 4

                        w00 = vdup_n_s8(wc0[6]);
                        w10 = vdup_n_s8(wc1[6]);
                        w20 = vdup_n_s8(wc2[6]);
                        w30 = vdup_n_s8(wc3[6]);

                        vdata00 = vmlal_s8(vdata00, tmp11, w02);//d = 2 3 4 5
                        vdata10 = vmlal_s8(vdata10, tmp11, w12);//d = 2 3 4 5
                        vdata20 = vmlal_s8(vdata20, tmp11, w22);//d = 2 3 4 5
                        vdata30 = vmlal_s8(vdata30, tmp11, w32);//d = 2 3 4 5

                        //r2
                        w01 = vdup_n_s8(wc0[7]);
                        w11 = vdup_n_s8(wc1[7]);
                        w21 = vdup_n_s8(wc2[7]);
                        w31 = vdup_n_s8(wc3[7]);

                        vdata00 = vmlal_s8(vdata00, vinr2, w00);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr2, w10);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr2, w20);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr2, w30);// a = 0 1 2 3

                        w02 = vdup_n_s8(wc0[8]);
                        w12 = vdup_n_s8(wc1[8]);
                        w22 = vdup_n_s8(wc2[8]);
                        w32 = vdup_n_s8(wc3[8]);

                        vdata00 = vmlal_s8(vdata00, tmp20, w01);//c = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp20, w11);//c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp20, w21);//c = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp20, w31);//c = 1 2 3 4

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr10 = vld1q_s32(doutc1r0);
                        int32x4_t outr20 = vld1q_s32(doutc2r0);
                        int32x4_t outr30 = vld1q_s32(doutc3r0);

                        // vdata00 = vmlal_s8(vdata00, tmp21, w02);//d = 2 3 4 5
                        // vdata10 = vmlal_s8(vdata10, tmp21, w12);//d = 2 3 4 5
                        // vdata20 = vmlal_s8(vdata20, tmp21, w22);//d = 2 3 4 5
                        // vdata30 = vmlal_s8(vdata30, tmp21, w32);//d = 2 3 4 5
                        int16x8_t vdata00_1 = vmull_s8(tmp21, w02);// d = 1 2 3 4
                        int16x8_t vdata10_1 = vmull_s8(tmp21, w12);// d = 1 2 3 4
                        int16x8_t vdata20_1 = vmull_s8(tmp21, w22);// d = 1 2 3 4
                        int16x8_t vdata30_1 = vmull_s8(tmp21, w32);// d = 1 2 3 4

                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);
                        int32x4_t outr11 = vld1q_s32(doutc1r0 + 4);
                        int32x4_t outr21 = vld1q_s32(doutc2r0 + 4);
                        int32x4_t outr31 = vld1q_s32(doutc3r0 + 4);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30));

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00_1));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10_1));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20_1));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30_1));

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00_1));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10_1));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20_1));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30_1));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc1r0, outr10);
                        vst1q_s32(doutc2r0, outr20);
                        vst1q_s32(doutc3r0, outr30);

                        din_ptr0 += 8;
                        din_ptr1 += 8;
                        din_ptr2 += 8;

                        vst1q_s32(doutc0r0 + 4, outr01);
                        vst1q_s32(doutc1r0 + 4, outr11);
                        vst1q_s32(doutc2r0 + 4, outr21);
                        vst1q_s32(doutc3r0 + 4, outr31);

                        doutc0r0 += 8;
                        doutc1r0 += 8;
                        doutc2r0 += 8;
                        doutc3r0 += 8;

                    }
                    //right
                    if (1){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//b = 8910111213
                        int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);
                        int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);

                        int8x8_t w00 = vdup_n_s8(wc0[0]);
                        int8x8_t w10 = vdup_n_s8(wc1[0]);
                        int8x8_t w20 = vdup_n_s8(wc2[0]);
                        int8x8_t w30 = vdup_n_s8(wc3[0]);

                        vinr0 = vbsl_s8(vmask_rp1, vinr0, vzero);
                        vinr0_1 = vbsl_s8(vmask_rp2, vinr0_1, vzero);

                        vinr1 = vbsl_s8(vmask_rp1, vinr1, vzero);
                        vinr1_1 = vbsl_s8(vmask_rp2, vinr1_1, vzero);

                        vinr2 = vbsl_s8(vmask_rp1, vinr2, vzero);
                        vinr2_1 = vbsl_s8(vmask_rp2, vinr2_1, vzero);

                        int8x8_t w01 = vdup_n_s8(wc0[1]);
                        int8x8_t w11 = vdup_n_s8(wc1[1]);
                        int8x8_t w21 = vdup_n_s8(wc2[1]);
                        int8x8_t w31 = vdup_n_s8(wc3[1]);

                        int8x8_t tmp00 = vext_s8(vinr0, vinr0_1, 1); //c = 1 2 3 4
                        int8x8_t tmp10 = vext_s8(vinr1, vinr1_1, 1); //c = 1 2 3 4
                        int8x8_t tmp20 = vext_s8(vinr2, vinr2_1, 1); //c = 1 2 3 4

                        int8x8_t tmp01 = vext_s8(vinr0, vinr0_1, 2); //d = 2 3 4 5
                        int8x8_t tmp11 = vext_s8(vinr1, vinr1_1, 2); //d = 2 3 4 5
                        int8x8_t tmp21 = vext_s8(vinr2, vinr2_1, 2); //d = 2 3 4 5

                        //r0
                        int8x8_t w02 = vdup_n_s8(wc0[2]);
                        int8x8_t w12 = vdup_n_s8(wc1[2]);
                        int8x8_t w22 = vdup_n_s8(wc2[2]);
                        int8x8_t w32 = vdup_n_s8(wc3[2]);

                        int16x8_t vdata00 = vmull_s8(vinr0, w00);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(vinr0, w10);// a = 0 1 2 3
                        int16x8_t vdata20 = vmull_s8(vinr0, w20);// a = 0 1 2 3
                        int16x8_t vdata30 = vmull_s8(vinr0, w30);// a = 0 1 2 3

                        w00 = vdup_n_s8(wc0[3]);
                        w10 = vdup_n_s8(wc1[3]);
                        w20 = vdup_n_s8(wc2[3]);
                        w30 = vdup_n_s8(wc3[3]);

                        vdata00 = vmlal_s8(vdata00, tmp00, w01);//c = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp00, w11);//c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp00, w21);//c = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp00, w31);//c = 1 2 3 4

                        w01 = vdup_n_s8(wc0[4]);
                        w11 = vdup_n_s8(wc1[4]);
                        w21 = vdup_n_s8(wc2[4]);
                        w31 = vdup_n_s8(wc3[4]);

                        vdata00 = vmlal_s8(vdata00, tmp01, w02);//d = 2 3 4 5
                        vdata10 = vmlal_s8(vdata10, tmp01, w12);//d = 2 3 4 5
                        vdata20 = vmlal_s8(vdata20, tmp01, w22);//d = 2 3 4 5
                        vdata30 = vmlal_s8(vdata30, tmp01, w32);//d = 2 3 4 5

                        //r1
                        w02 = vdup_n_s8(wc0[5]);
                        w12 = vdup_n_s8(wc1[5]);
                        w22 = vdup_n_s8(wc2[5]);
                        w32 = vdup_n_s8(wc3[5]);

                        vdata00 = vmlal_s8(vdata00, vinr1, w00);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr1, w10);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr1, w20);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr1, w30);// a = 0 1 2 3

                        w00 = vdup_n_s8(wc0[6]);
                        w10 = vdup_n_s8(wc1[6]);
                        w20 = vdup_n_s8(wc2[6]);
                        w30 = vdup_n_s8(wc3[6]);

                        vdata00 = vmlal_s8(vdata00, tmp10, w01);//c = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp10, w11);//c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp10, w21);//c = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp10, w31);//c = 1 2 3 4

                        w01 = vdup_n_s8(wc0[7]);
                        w11 = vdup_n_s8(wc1[7]);
                        w21 = vdup_n_s8(wc2[7]);
                        w31 = vdup_n_s8(wc3[7]);

                        vdata00 = vmlal_s8(vdata00, tmp11, w02);//d = 2 3 4 5
                        vdata10 = vmlal_s8(vdata10, tmp11, w12);//d = 2 3 4 5
                        vdata20 = vmlal_s8(vdata20, tmp11, w22);//d = 2 3 4 5
                        vdata30 = vmlal_s8(vdata30, tmp11, w32);//d = 2 3 4 5

                        //r2
                        w02 = vdup_n_s8(wc0[8]);
                        w12 = vdup_n_s8(wc1[8]);
                        w22 = vdup_n_s8(wc2[8]);
                        w32 = vdup_n_s8(wc3[8]);

                        vdata00 = vmlal_s8(vdata00, vinr2, w00);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr2, w10);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr2, w20);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr2, w30);// a = 0 1 2 3

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr10 = vld1q_s32(doutc1r0);
                        int32x4_t outr20 = vld1q_s32(doutc2r0);
                        int32x4_t outr30 = vld1q_s32(doutc3r0);

                        vdata00 = vmlal_s8(vdata00, tmp20, w01);//c = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp20, w11);//c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp20, w21);//c = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp20, w31);//c = 1 2 3 4

                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);
                        int32x4_t outr11 = vld1q_s32(doutc1r0 + 4);
                        int32x4_t outr21 = vld1q_s32(doutc2r0 + 4);
                        int32x4_t outr31 = vld1q_s32(doutc3r0 + 4);

                        // vdata00 = vmlal_s8(vdata00, tmp21, w02);//d = 2 3 4 5
                        // vdata10 = vmlal_s8(vdata10, tmp21, w12);//d = 2 3 4 5
                        // vdata20 = vmlal_s8(vdata20, tmp21, w22);//d = 2 3 4 5
                        // vdata30 = vmlal_s8(vdata30, tmp21, w32);//d = 2 3 4 5
                        int16x8_t vdata00_1 = vmull_s8(tmp21, w02);// d = 1 2 3 4
                        int16x8_t vdata10_1 = vmull_s8(tmp21, w12);// d = 1 2 3 4
                        int16x8_t vdata20_1 = vmull_s8(tmp21, w22);// d = 1 2 3 4
                        int16x8_t vdata30_1 = vmull_s8(tmp21, w32);// d = 1 2 3 4

                        vdata00 = vbslq_s16(vmask_result, vdata00, vzero_16);
                        vdata10 = vbslq_s16(vmask_result, vdata10, vzero_16);
                        vdata20 = vbslq_s16(vmask_result, vdata20, vzero_16);
                        vdata30 = vbslq_s16(vmask_result, vdata30, vzero_16);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30));

                        vdata00_1 = vbslq_s16(vmask_result, vdata00_1, vzero_16);
                        vdata10_1 = vbslq_s16(vmask_result, vdata10_1, vzero_16);
                        vdata20_1 = vbslq_s16(vmask_result, vdata20_1, vzero_16);
                        vdata30_1 = vbslq_s16(vmask_result, vdata30_1, vzero_16);

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00_1));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10_1));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20_1));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30_1));

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00_1));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10_1));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20_1));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30_1));


                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc1r0, outr10);
                        vst1q_s32(doutc2r0, outr20);
                        vst1q_s32(doutc3r0, outr30);

                        doutc0r0 += 4;
                        doutc1r0 += 4;
                        doutc2r0 += 4;
                        doutc3r0 += 4;

                        vst1q_s32(doutc0r0, outr01);
                        vst1q_s32(doutc1r0, outr11);
                        vst1q_s32(doutc2r0, outr21);
                        vst1q_s32(doutc3r0, outr31);

                    }
                    doutc0_ptr += wout;
                    doutc1_ptr += wout;
                    doutc2_ptr += wout;
                    doutc3_ptr += wout;
                }
                wc0 += 9;
                wc1 += 9;
                wc2 += 9;
                wc3 += 9;

            }

        } //end of processing output channel
        for (int c = ch; c < chout; ++c) {

            int* dout_c = dout_batch + c * size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c, &bias[c], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c, zero, 1, size_out_channel);
            }
            const signed char* wc0 = weights + c * w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *wc0_ptr = wc0 + i * 9;
                const signed char *din_ch = din_batch + i * size_in_channel;
                int *doutc0_ptr = dout_c;
                prefetch(wc0_ptr);

                int *doutc0r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                int8x8_t w00 = vdup_n_s8(wc0_ptr[0]);
                int8x8_t w01 = vdup_n_s8(wc0_ptr[1]);
                int8x8_t w02 = vdup_n_s8(wc0_ptr[2]);

                int8x8_t w10 = vdup_n_s8(wc0_ptr[3]);
                int8x8_t w11 = vdup_n_s8(wc0_ptr[4]);
                int8x8_t w12 = vdup_n_s8(wc0_ptr[5]);

                int8x8_t w20 = vdup_n_s8(wc0_ptr[6]);
                int8x8_t w21 = vdup_n_s8(wc0_ptr[7]);
                int8x8_t w22 = vdup_n_s8(wc0_ptr[8]);

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }

                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    //prefetch input
                    prefetch(doutc0r0);

                    prefetch(din_ptr0);
                    prefetch(din_ptr1);
                    prefetch(din_ptr2);

                    //left
                    if (1){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//b = 8910111213
                        int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);
                        int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);

                        int8x8_t tmp00 = vext_s8(vzero, vinr0, 7); //c = -1 0 1 2
                        int8x8_t tmp10 = vext_s8(vzero, vinr1, 7); //c = -1 0 1 2
                        int8x8_t tmp20 = vext_s8(vzero, vinr2, 7); //c = -1 0 1 2

                        int8x8_t tmp01 = vext_s8(vinr0, vinr0_1, 1); //d = 1 2 3 4
                        int8x8_t tmp11 = vext_s8(vinr1, vinr1_1, 1); //d = 1 2 3 4
                        int8x8_t tmp21 = vext_s8(vinr2, vinr2_1, 1); //d = 1 2 3 4

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w01);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(tmp00, w00);// c = -1 0 1 2
                        int16x8_t vdata20 = vmull_s8(tmp01, w02);// d = 1 2 3 4

                        din_ptr0 += 7;
                        din_ptr1 += 7;
                        din_ptr2 += 7;

                        //r1
                        vdata00 = vmlal_s8(vdata00, vinr1, w11);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp10, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp11, w12);// d = 1 2 3 4

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);

                        //r2
                        vdata00 = vmlal_s8(vdata00, vinr2, w21);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp20, w20);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp21, w22);// d = 1 2 3 4

                        prefetch(din_ptr0);
                        prefetch(din_ptr1);
                        prefetch(din_ptr2);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata10));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata10));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata20));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata20));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc0r0 + 4, outr01);

                        doutc0r0 += 8;
                    }
                    for (int j = 0; j < cnt_col; j++){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//b = 8910111213
                        int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);
                        int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);

                        int8x8_t tmp00 = vext_s8(vinr0, vinr0_1, 1); //c = 1 2 3 4
                        int8x8_t tmp10 = vext_s8(vinr1, vinr1_1, 1); //c = 1 2 3 4
                        int8x8_t tmp20 = vext_s8(vinr2, vinr2_1, 1);//c = 1 2 3 4

                        int8x8_t tmp01 = vext_s8(vinr0, vinr0_1, 2); //d = 2 3 4 5
                        int8x8_t tmp11 = vext_s8(vinr1, vinr1_1, 2); //d = 2 3 4 5
                        int8x8_t tmp21 = vext_s8(vinr2, vinr2_1, 2); //d = 2 3 4 5

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w00);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(tmp00, w01);// c = 1 2 3 4
                        int16x8_t vdata20 = vmull_s8(tmp01, w02);// d = 2 3 4 5

                        din_ptr0 += 8;
                        din_ptr1 += 8;
                        din_ptr2 += 8;

                        //r1
                        vdata00 = vmlal_s8(vdata00, vinr1, w10);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp10, w11);// c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp11, w12);// d = 2 3 4 5

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);

                        //r2
                        vdata00 = vmlal_s8(vdata00, vinr2, w20);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp20, w21);// c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp21, w22);// d = 2 3 4 5

                        prefetch(din_ptr0);
                        prefetch(din_ptr1);
                        prefetch(din_ptr2);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata10));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata10));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata20));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata20));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc0r0 + 4, outr01);

                        doutc0r0 += 8;
                    }
                    //right
                    if (1){

                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//b = 8910111213
                        int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);
                        int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);

                        vinr0 = vbsl_s8(vmask_rp1, vinr0, vzero);
                        vinr0_1 = vbsl_s8(vmask_rp2, vinr0_1, vzero);

                        vinr1 = vbsl_s8(vmask_rp1, vinr1, vzero);
                        vinr1_1 = vbsl_s8(vmask_rp2, vinr1_1, vzero);

                        vinr2 = vbsl_s8(vmask_rp1, vinr2, vzero);
                        vinr2_1 = vbsl_s8(vmask_rp2, vinr2_1, vzero);

                        int8x8_t tmp00 = vext_s8(vinr0, vinr0_1, 1); //c = 1 2 3 4
                        int8x8_t tmp10 = vext_s8(vinr1, vinr1_1, 1); //c = 1 2 3 4
                        int8x8_t tmp20 = vext_s8(vinr2, vinr2_1, 1);//c = 1 2 3 4

                        int8x8_t tmp01 = vext_s8(vinr0, vinr0_1, 2); //d = 2 3 4 5
                        int8x8_t tmp11 = vext_s8(vinr1, vinr1_1, 2); //d = 2 3 4 5
                        int8x8_t tmp21 = vext_s8(vinr2, vinr2_1, 2); //d = 2 3 4 5

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w00);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(tmp00, w01);// c = 1 2 3 4
                        int16x8_t vdata20 = vmull_s8(tmp01, w02);// d = 2 3 4 5

                        // din_ptr0 += 8;
                        // din_ptr1 += 8;
                        // din_ptr2 += 8;

                        //r1
                        vdata00 = vmlal_s8(vdata00, vinr1, w10);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp10, w11);// c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp11, w12);// d = 2 3 4 5

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);

                        //r2
                        vdata00 = vmlal_s8(vdata00, vinr2, w20);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp20, w21);// c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp21, w22);// d = 2 3 4 5

                        prefetch(din_ptr0);
                        prefetch(din_ptr1);
                        prefetch(din_ptr2);

                        vdata00 = vbslq_s16(vmask_result, vdata00, vzero_16);
                        vdata10 = vbslq_s16(vmask_result, vdata10, vzero_16);
                        vdata20 = vbslq_s16(vmask_result, vdata20, vzero_16);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata10));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata10));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata20));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata20));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc0r0 + 4, outr01);

                        // doutc0r0 += 8;
                    }
                    doutc0_ptr += wout;
                }
            }
        } // end of remain out channel

    } // end of processing batchs
}

void conv_3x3s1_direct_bias_s_int8(int* dout, const signed char* din, \
                          const signed char* weights, const  int* bias, bool flag_bias, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, Context* ctx){
    // printf("conv3x3s1_direct_int8 start \n");
    int threads = ctx->get_threads();
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const short unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, win * sizeof(signed char));

    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    int w_stride = chin * 9;

    unsigned int size_pad_right = (unsigned int)(win);

    uint8x8_t vmask_rp = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    unsigned int rst_remain = (unsigned int)(wout);
    uint16x8_t vmask_result = vcgtq_u16(vdupq_n_u16(rst_remain), vld1q_u16(right_pad_rst));

    int8x8_t vzero = vdup_n_s8(0);
    int16x8_t vzero_16 = vdupq_n_s16(0);
    int zero[4] = {0, 0, 0, 0};

    int ch = ((chout >> 2) << 2);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * chin * size_in_channel;
        int *dout_batch = dout + n * chout * size_out_channel;
#pragma omp parallel for num_threads(threads)
        for (int c = 0; c < chout - 3; c += 4) {

            int* dout_c0 = dout_batch + c * size_out_channel;
            int* dout_c1 = dout_c0 + size_out_channel;
            int* dout_c2 = dout_c1 + size_out_channel;
            int* dout_c3 = dout_c2 + size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c0, &bias[c], 1, size_out_channel);
                fill_bias_int8(dout_c1, &bias[c + 1], 1, size_out_channel);
                fill_bias_int8(dout_c2, &bias[c + 2], 1, size_out_channel);
                fill_bias_int8(dout_c3, &bias[c + 3], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c0, zero, 1, size_out_channel);
                fill_bias_int8(dout_c1, zero, 1, size_out_channel);
                fill_bias_int8(dout_c2, zero, 1, size_out_channel);
                fill_bias_int8(dout_c3, zero, 1, size_out_channel);
            }

            const signed char* wc0 = weights + c * w_stride;
            const signed char* wc1 = wc0 + w_stride;
            const signed char* wc2 = wc1 + w_stride;
            const signed char* wc3 = wc2 + w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *din_ch = din_batch + i * size_in_channel;
                int *doutc0_ptr = dout_c0;
                int *doutc1_ptr = dout_c1;
                int *doutc2_ptr = dout_c2;
                int *doutc3_ptr = dout_c3;

                int *doutc0r0 = nullptr;
                int *doutc1r0 = nullptr;
                int *doutc2r0 = nullptr;
                int *doutc3r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    doutc1r0 = doutc1_ptr;
                    doutc2r0 = doutc2_ptr;
                    doutc3r0 = doutc3_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }
                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    //prefetch input
                    prefetch(doutc0r0);
                    prefetch(doutc1r0);
                    prefetch(doutc2r0);
                    prefetch(doutc3r0);

                    prefetch(din_ptr0);
                    prefetch(din_ptr1);
                    prefetch(din_ptr2);
                    prefetch(wc0);
                    prefetch(wc1);
                    prefetch(wc2);
                    prefetch(wc3);

                    //left
                    if (1){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t w01 = vdup_n_s8(wc0[1]);
                        int8x8_t w11 = vdup_n_s8(wc1[1]);
                        int8x8_t w21 = vdup_n_s8(wc2[1]);
                        int8x8_t w31 = vdup_n_s8(wc3[1]);

                        vinr0 = vbsl_s8(vmask_rp, vinr0, vzero);
                        vinr1 = vbsl_s8(vmask_rp, vinr1, vzero);
                        vinr2 = vbsl_s8(vmask_rp, vinr2, vzero);

                        int8x8_t w00 = vdup_n_s8(wc0[0]);
                        int8x8_t w10 = vdup_n_s8(wc1[0]);
                        int8x8_t w20 = vdup_n_s8(wc2[0]);
                        int8x8_t w30 = vdup_n_s8(wc3[0]);

                        int8x8_t tmp00 = vext_s8(vzero, vinr0, 7); //c = -1 0 1 2
                        int8x8_t tmp10 = vext_s8(vzero, vinr1, 7); //c = -1 0 1 2
                        int8x8_t tmp20 = vext_s8(vzero, vinr2, 7); //c = -1 0 1 2

                        int8x8_t tmp01 = vext_s8(vinr0, vzero, 1); //d = 1 2 3 4
                        int8x8_t tmp11 = vext_s8(vinr1, vzero, 1); //d = 1 2 3 4
                        int8x8_t tmp21 = vext_s8(vinr2, vzero, 1); //d = 1 2 3 4

                        int8x8_t w02 = vdup_n_s8(wc0[2]);
                        int8x8_t w12 = vdup_n_s8(wc1[2]);
                        int8x8_t w22 = vdup_n_s8(wc2[2]);
                        int8x8_t w32 = vdup_n_s8(wc3[2]);

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w01);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(vinr0, w11);// a = 0 1 2 3
                        int16x8_t vdata20 = vmull_s8(vinr0, w21);// a = 0 1 2 3
                        int16x8_t vdata30 = vmull_s8(vinr0, w31);// a = 0 1 2 3

                        vdata00 = vmlal_s8(vdata00, tmp00, w00);// c = -1 0 1 2
                        vdata10 = vmlal_s8(vdata10, tmp00, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp00, w20);// c = -1 0 1 2
                        vdata30 = vmlal_s8(vdata30, tmp00, w30);// c = -1 0 1 2

                        w00 = vdup_n_s8(wc0[3]);
                        w10 = vdup_n_s8(wc1[3]);
                        w20 = vdup_n_s8(wc2[3]);
                        w30 = vdup_n_s8(wc3[3]);

                        vdata00 = vmlal_s8(vdata00, tmp01, w02);// d = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp01, w12);// d = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp01, w22);// d = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp01, w32);// d = 1 2 3 4

                        //r1
                        w01 = vdup_n_s8(wc0[4]);
                        w11 = vdup_n_s8(wc1[4]);
                        w21 = vdup_n_s8(wc2[4]);
                        w31 = vdup_n_s8(wc3[4]);

                        vdata00 = vmlal_s8(vdata00, tmp10, w00);// c = -1 0 1 2
                        vdata10 = vmlal_s8(vdata10, tmp10, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp10, w20);// c = -1 0 1 2
                        vdata30 = vmlal_s8(vdata30, tmp10, w30);// c = -1 0 1 2

                        w02 = vdup_n_s8(wc0[5]);
                        w12 = vdup_n_s8(wc1[5]);
                        w22 = vdup_n_s8(wc2[5]);
                        w32 = vdup_n_s8(wc3[5]);

                        vdata00 = vmlal_s8(vdata00, vinr1, w01);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr1, w11);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr1, w21);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr1, w31);// a = 0 1 2 3

                        w00 = vdup_n_s8(wc0[6]);
                        w10 = vdup_n_s8(wc1[6]);
                        w20 = vdup_n_s8(wc2[6]);
                        w30 = vdup_n_s8(wc3[6]);

                        vdata00 = vmlal_s8(vdata00, tmp11, w02);// d = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp11, w12);// d = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp11, w22);// d = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp11, w32);// d = 1 2 3 4

                        //r2
                        w01 = vdup_n_s8(wc0[7]);
                        w11 = vdup_n_s8(wc1[7]);
                        w21 = vdup_n_s8(wc2[7]);
                        w31 = vdup_n_s8(wc3[7]);

                        vdata00 = vmlal_s8(vdata00, tmp20, w00);// c = -1 0 1 2
                        vdata10 = vmlal_s8(vdata10, tmp20, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp20, w20);// c = -1 0 1 2
                        vdata30 = vmlal_s8(vdata30, tmp20, w30);// c = -1 0 1 2

                        w02 = vdup_n_s8(wc0[8]);
                        w12 = vdup_n_s8(wc1[8]);
                        w22 = vdup_n_s8(wc2[8]);
                        w32 = vdup_n_s8(wc3[8]);

                        vdata00 = vmlal_s8(vdata00, vinr2, w01);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr2, w11);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr2, w21);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr2, w31);// a = 0 1 2 3

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr10 = vld1q_s32(doutc1r0);
                        int32x4_t outr20 = vld1q_s32(doutc2r0);
                        int32x4_t outr30 = vld1q_s32(doutc3r0);

                        int16x8_t vdata00_1 = vmull_s8(tmp21, w02);// d = 1 2 3 4
                        int16x8_t vdata10_1 = vmull_s8(tmp21, w12);// d = 1 2 3 4
                        int16x8_t vdata20_1 = vmull_s8(tmp21, w22);// d = 1 2 3 4
                        int16x8_t vdata30_1 = vmull_s8(tmp21, w32);// d = 1 2 3 4

                        // vdata00 = vmlal_s8(vdata00, tmp21, w02);// d = 1 2 3 4
                        // vdata10 = vmlal_s8(vdata10, tmp21, w12);// d = 1 2 3 4
                        // vdata20 = vmlal_s8(vdata20, tmp21, w22);// d = 1 2 3 4
                        // vdata30 = vmlal_s8(vdata30, tmp21, w32);// d = 1 2 3 4

                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);
                        int32x4_t outr11 = vld1q_s32(doutc1r0 + 4);
                        int32x4_t outr21 = vld1q_s32(doutc2r0 + 4);
                        int32x4_t outr31 = vld1q_s32(doutc3r0 + 4);

                        vdata00 = vbslq_s16(vmask_result, vdata00, vzero_16);
                        vdata10 = vbslq_s16(vmask_result, vdata10, vzero_16);
                        vdata20 = vbslq_s16(vmask_result, vdata20, vzero_16);
                        vdata30 = vbslq_s16(vmask_result, vdata30, vzero_16);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30));

                        vdata00_1 = vbslq_s16(vmask_result, vdata00_1, vzero_16);
                        vdata10_1 = vbslq_s16(vmask_result, vdata10_1, vzero_16);
                        vdata20_1 = vbslq_s16(vmask_result, vdata20_1, vzero_16);
                        vdata30_1 = vbslq_s16(vmask_result, vdata30_1, vzero_16);

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00_1));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10_1));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20_1));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30_1));

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00_1));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10_1));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20_1));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30_1));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc1r0, outr10);
                        vst1q_s32(doutc2r0, outr20);
                        vst1q_s32(doutc3r0, outr30);

                        vst1q_s32(doutc0r0 + 4, outr01);
                        vst1q_s32(doutc1r0 + 4, outr11);
                        vst1q_s32(doutc2r0 + 4, outr21);
                        vst1q_s32(doutc3r0 + 4, outr31);
                    }
                    doutc0_ptr += wout;
                    doutc1_ptr += wout;
                    doutc2_ptr += wout;
                    doutc3_ptr += wout;
                }
                wc0 += 9;
                wc1 += 9;
                wc2 += 9;
                wc3 += 9;

            }

        } //end of processing output channel
        for (int c = ch; c < chout; ++c) {

            int* dout_c = dout_batch + c * size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c, &bias[c], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c, zero, 1, size_out_channel);
            }
            const signed char* wc0 = weights + c * w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *wc0_ptr = wc0 + i * 9;
                const signed char *din_ch = din_batch + i * size_in_channel;
                int *doutc0_ptr = dout_c;
                prefetch(wc0_ptr);

                int *doutc0r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                int8x8_t w00 = vdup_n_s8(wc0_ptr[0]);
                int8x8_t w01 = vdup_n_s8(wc0_ptr[1]);
                int8x8_t w02 = vdup_n_s8(wc0_ptr[2]);

                int8x8_t w10 = vdup_n_s8(wc0_ptr[3]);
                int8x8_t w11 = vdup_n_s8(wc0_ptr[4]);
                int8x8_t w12 = vdup_n_s8(wc0_ptr[5]);

                int8x8_t w20 = vdup_n_s8(wc0_ptr[6]);
                int8x8_t w21 = vdup_n_s8(wc0_ptr[7]);
                int8x8_t w22 = vdup_n_s8(wc0_ptr[8]);

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }

                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    //prefetch input
                    prefetch(doutc0r0);

                    prefetch(din_ptr0);
                    prefetch(din_ptr1);
                    prefetch(din_ptr2);

                    //left
                    if (1){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        vinr0 = vbsl_s8(vmask_rp, vinr0, vzero);
                        vinr1 = vbsl_s8(vmask_rp, vinr1, vzero);
                        vinr2 = vbsl_s8(vmask_rp, vinr2, vzero);

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);

                        int8x8_t tmp00 = vext_s8(vzero, vinr0, 7); //c = -1 0 1 2
                        int8x8_t tmp10 = vext_s8(vzero, vinr1, 7); //c = -1 0 1 2
                        int8x8_t tmp20 = vext_s8(vzero, vinr2, 7); //c = -1 0 1 2

                        int8x8_t tmp01 = vext_s8(vinr0, vzero, 1); //d = 1 2 3 4
                        int8x8_t tmp11 = vext_s8(vinr1, vzero, 1); //d = 1 2 3 4
                        int8x8_t tmp21 = vext_s8(vinr2, vzero, 1); //d = 1 2 3 4

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w01);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(tmp00, w00);// c = -1 0 1 2
                        int16x8_t vdata20 = vmull_s8(tmp01, w02);// d = 1 2 3 4

                        //r1
                        vdata00 = vmlal_s8(vdata00, vinr1, w11);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp10, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp11, w12);// d = 1 2 3 4

                        //r2
                        vdata00 = vmlal_s8(vdata00, vinr2, w21);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp20, w20);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp21, w22);// d = 1 2 3 4

                        vdata00 = vbslq_s16(vmask_result, vdata00, vzero_16);
                        vdata10 = vbslq_s16(vmask_result, vdata10, vzero_16);
                        vdata20 = vbslq_s16(vmask_result, vdata20, vzero_16);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata10));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata10));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata20));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata20));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc0r0 + 4, outr01);
                    }
                    doutc0_ptr += wout;
                }
            }
        } // end of remain out channel

    } // end of processing batchs
}

void conv_3x3s1_direct_bias_relu_int8(int* dout, const signed char* din, \
                          const signed char* weights, const  int* bias, bool flag_bias, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, Context* ctx){
    int threads = ctx->get_threads();
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const short unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, win * sizeof(signed char));

    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    int w_stride = chin * 9;

    int tile_w = (win + 7) >> 3;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(win - 7 - (cnt_col << 3));

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)(wout - ((cnt_col + 1) << 3));
    uint16x8_t vmask_result = vcgtq_u16(vdupq_n_u16(rst_remain), vld1q_u16(right_pad_rst));

    int8x8_t vzero = vdup_n_s8(0);
    int16x8_t vzero_16 = vdupq_n_s16(0);
    int zero[4] = {0, 0, 0, 0};

    int ch = ((chout >> 2) << 2);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * chin * size_in_channel;
        int *dout_batch = dout + n * chout * size_out_channel;
#pragma omp parallel for num_threads(threads)
        for (int c = 0; c < chout - 3; c += 4) {

            int* dout_c0 = dout_batch + c * size_out_channel;
            int* dout_c1 = dout_c0 + size_out_channel;
            int* dout_c2 = dout_c1 + size_out_channel;
            int* dout_c3 = dout_c2 + size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c0, &bias[c], 1, size_out_channel);
                fill_bias_int8(dout_c1, &bias[c + 1], 1, size_out_channel);
                fill_bias_int8(dout_c2, &bias[c + 2], 1, size_out_channel);
                fill_bias_int8(dout_c3, &bias[c + 3], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c0, zero, 1, size_out_channel);
                fill_bias_int8(dout_c1, zero, 1, size_out_channel);
                fill_bias_int8(dout_c2, zero, 1, size_out_channel);
                fill_bias_int8(dout_c3, zero, 1, size_out_channel);
            }

            const signed char* wc0 = weights + c * w_stride;
            const signed char* wc1 = wc0 + w_stride;
            const signed char* wc2 = wc1 + w_stride;
            const signed char* wc3 = wc2 + w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *din_ch = din_batch + i * size_in_channel;
                int *doutc0_ptr = dout_c0;
                int *doutc1_ptr = dout_c1;
                int *doutc2_ptr = dout_c2;
                int *doutc3_ptr = dout_c3;

                int *doutc0r0 = nullptr;
                int *doutc1r0 = nullptr;
                int *doutc2r0 = nullptr;
                int *doutc3r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    doutc1r0 = doutc1_ptr;
                    doutc2r0 = doutc2_ptr;
                    doutc3r0 = doutc3_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }
                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    //prefetch input
                    prefetch(doutc0r0);
                    prefetch(doutc1r0);
                    prefetch(doutc2r0);
                    prefetch(doutc3r0);

                    prefetch(din_ptr0);
                    prefetch(din_ptr1);
                    prefetch(din_ptr2);
                    prefetch(wc0);
                    prefetch(wc1);
                    prefetch(wc2);
                    prefetch(wc3);

                    //left
                    if (1){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//b = 8910111213
                        int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);
                        int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);

                        int8x8_t w01 = vdup_n_s8(wc0[1]);
                        int8x8_t w11 = vdup_n_s8(wc1[1]);
                        int8x8_t w21 = vdup_n_s8(wc2[1]);
                        int8x8_t w31 = vdup_n_s8(wc3[1]);

                        int8x8_t tmp00 = vext_s8(vzero, vinr0, 7); //c = -1 0 1 2
                        int8x8_t tmp10 = vext_s8(vzero, vinr1, 7); //c = -1 0 1 2
                        int8x8_t tmp20 = vext_s8(vzero, vinr2, 7); //c = -1 0 1 2

                        int8x8_t tmp01 = vext_s8(vinr0, vinr0_1, 1); //d = 1 2 3 4
                        int8x8_t tmp11 = vext_s8(vinr1, vinr1_1, 1); //d = 1 2 3 4
                        int8x8_t tmp21 = vext_s8(vinr2, vinr2_1, 1); //d = 1 2 3 4

                        int8x8_t w00 = vdup_n_s8(wc0[0]);
                        int8x8_t w10 = vdup_n_s8(wc1[0]);
                        int8x8_t w20 = vdup_n_s8(wc2[0]);
                        int8x8_t w30 = vdup_n_s8(wc3[0]);

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w01);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(vinr0, w11);// a = 0 1 2 3
                        int16x8_t vdata20 = vmull_s8(vinr0, w21);// a = 0 1 2 3
                        int16x8_t vdata30 = vmull_s8(vinr0, w31);// a = 0 1 2 3

                        int8x8_t w02 = vdup_n_s8(wc0[2]);
                        int8x8_t w12 = vdup_n_s8(wc1[2]);
                        int8x8_t w22 = vdup_n_s8(wc2[2]);
                        int8x8_t w32 = vdup_n_s8(wc3[2]);

                        vdata00 = vmlal_s8(vdata00, tmp00, w00);// c = -1 0 1 2
                        vdata10 = vmlal_s8(vdata10, tmp00, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp00, w20);// c = -1 0 1 2
                        vdata30 = vmlal_s8(vdata30, tmp00, w30);// c = -1 0 1 2

                        w00 = vdup_n_s8(wc0[3]);
                        w10 = vdup_n_s8(wc1[3]);
                        w20 = vdup_n_s8(wc2[3]);
                        w30 = vdup_n_s8(wc3[3]);

                        vdata00 = vmlal_s8(vdata00, tmp01, w02);// d = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp01, w12);// d = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp01, w22);// d = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp01, w32);// d = 1 2 3 4

                        //r1
                        w01 = vdup_n_s8(wc0[4]);
                        w11 = vdup_n_s8(wc1[4]);
                        w21 = vdup_n_s8(wc2[4]);
                        w31 = vdup_n_s8(wc3[4]);

                        vdata00 = vmlal_s8(vdata00, tmp10, w00);// c = -1 0 1 2
                        vdata10 = vmlal_s8(vdata10, tmp10, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp10, w20);// c = -1 0 1 2
                        vdata30 = vmlal_s8(vdata30, tmp10, w30);// c = -1 0 1 2

                        w02 = vdup_n_s8(wc0[5]);
                        w12 = vdup_n_s8(wc1[5]);
                        w22 = vdup_n_s8(wc2[5]);
                        w32 = vdup_n_s8(wc3[5]);

                        vdata00 = vmlal_s8(vdata00, vinr1, w01);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr1, w11);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr1, w21);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr1, w31);// a = 0 1 2 3

                        w00 = vdup_n_s8(wc0[6]);
                        w10 = vdup_n_s8(wc1[6]);
                        w20 = vdup_n_s8(wc2[6]);
                        w30 = vdup_n_s8(wc3[6]);

                        vdata00 = vmlal_s8(vdata00, tmp11, w02);// d = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp11, w12);// d = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp11, w22);// d = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp11, w32);// d = 1 2 3 4

                        //r2
                        w01 = vdup_n_s8(wc0[7]);
                        w11 = vdup_n_s8(wc1[7]);
                        w21 = vdup_n_s8(wc2[7]);
                        w31 = vdup_n_s8(wc3[7]);

                        vdata00 = vmlal_s8(vdata00, tmp20, w00);// c = -1 0 1 2
                        vdata10 = vmlal_s8(vdata10, tmp20, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp20, w20);// c = -1 0 1 2
                        vdata30 = vmlal_s8(vdata30, tmp20, w30);// c = -1 0 1 2

                        w02 = vdup_n_s8(wc0[8]);
                        w12 = vdup_n_s8(wc1[8]);
                        w22 = vdup_n_s8(wc2[8]);
                        w32 = vdup_n_s8(wc3[8]);

                        vdata00 = vmlal_s8(vdata00, vinr2, w01);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr2, w11);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr2, w21);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr2, w31);// a = 0 1 2 3

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr10 = vld1q_s32(doutc1r0);
                        int32x4_t outr20 = vld1q_s32(doutc2r0);
                        int32x4_t outr30 = vld1q_s32(doutc3r0);

                        int16x8_t vdata00_1 = vmull_s8(tmp21, w02);// d = 1 2 3 4
                        int16x8_t vdata10_1 = vmull_s8(tmp21, w12);// d = 1 2 3 4
                        int16x8_t vdata20_1 = vmull_s8(tmp21, w22);// d = 1 2 3 4
                        int16x8_t vdata30_1 = vmull_s8(tmp21, w32);// d = 1 2 3 4

                        // vdata00 = vmlal_s8(vdata00, tmp21, w32);// d = 1 2 3 4
                        // vdata10 = vmlal_s8(vdata10, tmp21, w32);// d = 1 2 3 4
                        // vdata20 = vmlal_s8(vdata20, tmp21, w32);// d = 1 2 3 4
                        // vdata30 = vmlal_s8(vdata30, tmp21, w32);// d = 1 2 3 4

                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);
                        int32x4_t outr11 = vld1q_s32(doutc1r0 + 4);
                        int32x4_t outr21 = vld1q_s32(doutc2r0 + 4);
                        int32x4_t outr31 = vld1q_s32(doutc3r0 + 4);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30));

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00_1));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10_1));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20_1));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30_1));

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00_1));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10_1));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20_1));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30_1));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc1r0, outr10);
                        vst1q_s32(doutc2r0, outr20);
                        vst1q_s32(doutc3r0, outr30);

                        din_ptr0 += 7;
                        din_ptr1 += 7;
                        din_ptr2 += 7;

                        vst1q_s32(doutc0r0 + 4, outr01);
                        vst1q_s32(doutc1r0 + 4, outr11);
                        vst1q_s32(doutc2r0 + 4, outr21);
                        vst1q_s32(doutc3r0 + 4, outr31);

                        doutc0r0 += 8;
                        doutc1r0 += 8;
                        doutc2r0 += 8;
                        doutc3r0 += 8;
                    }
                    for (int j = 0; j < cnt_col; j++){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//b = 8910111213
                        int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);
                        int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);

                        int8x8_t w00 = vdup_n_s8(wc0[0]);
                        int8x8_t w10 = vdup_n_s8(wc1[0]);
                        int8x8_t w20 = vdup_n_s8(wc2[0]);
                        int8x8_t w30 = vdup_n_s8(wc3[0]);

                        int8x8_t tmp00 = vext_s8(vinr0, vinr0_1, 1); //c = 1 2 3 4
                        int8x8_t tmp10 = vext_s8(vinr1, vinr1_1, 1); //c = 1 2 3 4
                        int8x8_t tmp20 = vext_s8(vinr2, vinr2_1, 1); //c = 1 2 3 4

                        int8x8_t tmp01 = vext_s8(vinr0, vinr0_1, 2); //d = 2 3 4 5
                        int8x8_t tmp11 = vext_s8(vinr1, vinr1_1, 2); //d = 2 3 4 5
                        int8x8_t tmp21 = vext_s8(vinr2, vinr2_1, 2); //d = 2 3 4 5

                        int8x8_t w01 = vdup_n_s8(wc0[1]);
                        int8x8_t w11 = vdup_n_s8(wc1[1]);
                        int8x8_t w21 = vdup_n_s8(wc2[1]);
                        int8x8_t w31 = vdup_n_s8(wc3[1]);

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w00);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(vinr0, w10);// a = 0 1 2 3
                        int16x8_t vdata20 = vmull_s8(vinr0, w20);// a = 0 1 2 3
                        int16x8_t vdata30 = vmull_s8(vinr0, w30);// a = 0 1 2 3

                        int8x8_t w02 = vdup_n_s8(wc0[2]);
                        int8x8_t w12 = vdup_n_s8(wc1[2]);
                        int8x8_t w22 = vdup_n_s8(wc2[2]);
                        int8x8_t w32 = vdup_n_s8(wc3[2]);

                        vdata00 = vmlal_s8(vdata00, tmp00, w01);//c = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp00, w11);//c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp00, w21);//c = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp00, w31);//c = 1 2 3 4

                        w00 = vdup_n_s8(wc0[3]);
                        w10 = vdup_n_s8(wc1[3]);
                        w20 = vdup_n_s8(wc2[3]);
                        w30 = vdup_n_s8(wc3[3]);

                        vdata00 = vmlal_s8(vdata00, tmp01, w02);//d = 2 3 4 5
                        vdata10 = vmlal_s8(vdata10, tmp01, w12);//d = 2 3 4 5
                        vdata20 = vmlal_s8(vdata20, tmp01, w22);//d = 2 3 4 5
                        vdata30 = vmlal_s8(vdata30, tmp01, w32);//d = 2 3 4 5

                        //r1
                        w01 = vdup_n_s8(wc0[4]);
                        w11 = vdup_n_s8(wc1[4]);
                        w21 = vdup_n_s8(wc2[4]);
                        w31 = vdup_n_s8(wc3[4]);

                        vdata00 = vmlal_s8(vdata00, vinr1, w00);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr1, w10);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr1, w20);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr1, w30);// a = 0 1 2 3

                        w02 = vdup_n_s8(wc0[5]);
                        w12 = vdup_n_s8(wc1[5]);
                        w22 = vdup_n_s8(wc2[5]);
                        w32 = vdup_n_s8(wc3[5]);

                        vdata00 = vmlal_s8(vdata00, tmp10, w01);//c = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp10, w11);//c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp10, w21);//c = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp10, w31);//c = 1 2 3 4

                        w00 = vdup_n_s8(wc0[6]);
                        w10 = vdup_n_s8(wc1[6]);
                        w20 = vdup_n_s8(wc2[6]);
                        w30 = vdup_n_s8(wc3[6]);

                        vdata00 = vmlal_s8(vdata00, tmp11, w02);//d = 2 3 4 5
                        vdata10 = vmlal_s8(vdata10, tmp11, w12);//d = 2 3 4 5
                        vdata20 = vmlal_s8(vdata20, tmp11, w22);//d = 2 3 4 5
                        vdata30 = vmlal_s8(vdata30, tmp11, w32);//d = 2 3 4 5

                        //r2
                        w01 = vdup_n_s8(wc0[7]);
                        w11 = vdup_n_s8(wc1[7]);
                        w21 = vdup_n_s8(wc2[7]);
                        w31 = vdup_n_s8(wc3[7]);

                        vdata00 = vmlal_s8(vdata00, vinr2, w00);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr2, w10);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr2, w20);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr2, w30);// a = 0 1 2 3

                        w02 = vdup_n_s8(wc0[8]);
                        w12 = vdup_n_s8(wc1[8]);
                        w22 = vdup_n_s8(wc2[8]);
                        w32 = vdup_n_s8(wc3[8]);

                        vdata00 = vmlal_s8(vdata00, tmp20, w01);//c = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp20, w11);//c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp20, w21);//c = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp20, w31);//c = 1 2 3 4

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr10 = vld1q_s32(doutc1r0);
                        int32x4_t outr20 = vld1q_s32(doutc2r0);
                        int32x4_t outr30 = vld1q_s32(doutc3r0);

                        // vdata00 = vmlal_s8(vdata00, tmp21, w02);//d = 2 3 4 5
                        // vdata10 = vmlal_s8(vdata10, tmp21, w12);//d = 2 3 4 5
                        // vdata20 = vmlal_s8(vdata20, tmp21, w22);//d = 2 3 4 5
                        // vdata30 = vmlal_s8(vdata30, tmp21, w32);//d = 2 3 4 5
                        int16x8_t vdata00_1 = vmull_s8(tmp21, w02);// d = 1 2 3 4
                        int16x8_t vdata10_1 = vmull_s8(tmp21, w12);// d = 1 2 3 4
                        int16x8_t vdata20_1 = vmull_s8(tmp21, w22);// d = 1 2 3 4
                        int16x8_t vdata30_1 = vmull_s8(tmp21, w32);// d = 1 2 3 4

                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);
                        int32x4_t outr11 = vld1q_s32(doutc1r0 + 4);
                        int32x4_t outr21 = vld1q_s32(doutc2r0 + 4);
                        int32x4_t outr31 = vld1q_s32(doutc3r0 + 4);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30));

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00_1));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10_1));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20_1));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30_1));

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00_1));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10_1));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20_1));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30_1));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc1r0, outr10);
                        vst1q_s32(doutc2r0, outr20);
                        vst1q_s32(doutc3r0, outr30);

                        din_ptr0 += 8;
                        din_ptr1 += 8;
                        din_ptr2 += 8;

                        vst1q_s32(doutc0r0 + 4, outr01);
                        vst1q_s32(doutc1r0 + 4, outr11);
                        vst1q_s32(doutc2r0 + 4, outr21);
                        vst1q_s32(doutc3r0 + 4, outr31);

                        doutc0r0 += 8;
                        doutc1r0 += 8;
                        doutc2r0 += 8;
                        doutc3r0 += 8;

                    }
                    //right
                    if (1){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//b = 8910111213
                        int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);
                        int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);

                        int8x8_t w00 = vdup_n_s8(wc0[0]);
                        int8x8_t w10 = vdup_n_s8(wc1[0]);
                        int8x8_t w20 = vdup_n_s8(wc2[0]);
                        int8x8_t w30 = vdup_n_s8(wc3[0]);

                        vinr0 = vbsl_s8(vmask_rp1, vinr0, vzero);
                        vinr0_1 = vbsl_s8(vmask_rp2, vinr0_1, vzero);

                        vinr1 = vbsl_s8(vmask_rp1, vinr1, vzero);
                        vinr1_1 = vbsl_s8(vmask_rp2, vinr1_1, vzero);

                        vinr2 = vbsl_s8(vmask_rp1, vinr2, vzero);
                        vinr2_1 = vbsl_s8(vmask_rp2, vinr2_1, vzero);

                        int8x8_t w01 = vdup_n_s8(wc0[1]);
                        int8x8_t w11 = vdup_n_s8(wc1[1]);
                        int8x8_t w21 = vdup_n_s8(wc2[1]);
                        int8x8_t w31 = vdup_n_s8(wc3[1]);

                        int8x8_t tmp00 = vext_s8(vinr0, vinr0_1, 1); //c = 1 2 3 4
                        int8x8_t tmp10 = vext_s8(vinr1, vinr1_1, 1); //c = 1 2 3 4
                        int8x8_t tmp20 = vext_s8(vinr2, vinr2_1, 1); //c = 1 2 3 4

                        int8x8_t tmp01 = vext_s8(vinr0, vinr0_1, 2); //d = 2 3 4 5
                        int8x8_t tmp11 = vext_s8(vinr1, vinr1_1, 2); //d = 2 3 4 5
                        int8x8_t tmp21 = vext_s8(vinr2, vinr2_1, 2); //d = 2 3 4 5

                        //r0
                        int8x8_t w02 = vdup_n_s8(wc0[2]);
                        int8x8_t w12 = vdup_n_s8(wc1[2]);
                        int8x8_t w22 = vdup_n_s8(wc2[2]);
                        int8x8_t w32 = vdup_n_s8(wc3[2]);

                        int16x8_t vdata00 = vmull_s8(vinr0, w00);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(vinr0, w10);// a = 0 1 2 3
                        int16x8_t vdata20 = vmull_s8(vinr0, w20);// a = 0 1 2 3
                        int16x8_t vdata30 = vmull_s8(vinr0, w30);// a = 0 1 2 3

                        w00 = vdup_n_s8(wc0[3]);
                        w10 = vdup_n_s8(wc1[3]);
                        w20 = vdup_n_s8(wc2[3]);
                        w30 = vdup_n_s8(wc3[3]);

                        vdata00 = vmlal_s8(vdata00, tmp00, w01);//c = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp00, w11);//c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp00, w21);//c = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp00, w31);//c = 1 2 3 4

                        w01 = vdup_n_s8(wc0[4]);
                        w11 = vdup_n_s8(wc1[4]);
                        w21 = vdup_n_s8(wc2[4]);
                        w31 = vdup_n_s8(wc3[4]);

                        vdata00 = vmlal_s8(vdata00, tmp01, w02);//d = 2 3 4 5
                        vdata10 = vmlal_s8(vdata10, tmp01, w12);//d = 2 3 4 5
                        vdata20 = vmlal_s8(vdata20, tmp01, w22);//d = 2 3 4 5
                        vdata30 = vmlal_s8(vdata30, tmp01, w32);//d = 2 3 4 5

                        //r1
                        w02 = vdup_n_s8(wc0[5]);
                        w12 = vdup_n_s8(wc1[5]);
                        w22 = vdup_n_s8(wc2[5]);
                        w32 = vdup_n_s8(wc3[5]);

                        vdata00 = vmlal_s8(vdata00, vinr1, w00);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr1, w10);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr1, w20);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr1, w30);// a = 0 1 2 3

                        w00 = vdup_n_s8(wc0[6]);
                        w10 = vdup_n_s8(wc1[6]);
                        w20 = vdup_n_s8(wc2[6]);
                        w30 = vdup_n_s8(wc3[6]);

                        vdata00 = vmlal_s8(vdata00, tmp10, w01);//c = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp10, w11);//c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp10, w21);//c = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp10, w31);//c = 1 2 3 4

                        w01 = vdup_n_s8(wc0[7]);
                        w11 = vdup_n_s8(wc1[7]);
                        w21 = vdup_n_s8(wc2[7]);
                        w31 = vdup_n_s8(wc3[7]);

                        vdata00 = vmlal_s8(vdata00, tmp11, w02);//d = 2 3 4 5
                        vdata10 = vmlal_s8(vdata10, tmp11, w12);//d = 2 3 4 5
                        vdata20 = vmlal_s8(vdata20, tmp11, w22);//d = 2 3 4 5
                        vdata30 = vmlal_s8(vdata30, tmp11, w32);//d = 2 3 4 5

                        //r2
                        w02 = vdup_n_s8(wc0[8]);
                        w12 = vdup_n_s8(wc1[8]);
                        w22 = vdup_n_s8(wc2[8]);
                        w32 = vdup_n_s8(wc3[8]);

                        vdata00 = vmlal_s8(vdata00, vinr2, w00);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr2, w10);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr2, w20);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr2, w30);// a = 0 1 2 3

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr10 = vld1q_s32(doutc1r0);
                        int32x4_t outr20 = vld1q_s32(doutc2r0);
                        int32x4_t outr30 = vld1q_s32(doutc3r0);

                        vdata00 = vmlal_s8(vdata00, tmp20, w01);//c = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp20, w11);//c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp20, w21);//c = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp20, w31);//c = 1 2 3 4

                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);
                        int32x4_t outr11 = vld1q_s32(doutc1r0 + 4);
                        int32x4_t outr21 = vld1q_s32(doutc2r0 + 4);
                        int32x4_t outr31 = vld1q_s32(doutc3r0 + 4);

                        // vdata00 = vmlal_s8(vdata00, tmp21, w02);//d = 2 3 4 5
                        // vdata10 = vmlal_s8(vdata10, tmp21, w12);//d = 2 3 4 5
                        // vdata20 = vmlal_s8(vdata20, tmp21, w22);//d = 2 3 4 5
                        // vdata30 = vmlal_s8(vdata30, tmp21, w32);//d = 2 3 4 5
                        int16x8_t vdata00_1 = vmull_s8(tmp21, w02);// d = 1 2 3 4
                        int16x8_t vdata10_1 = vmull_s8(tmp21, w12);// d = 1 2 3 4
                        int16x8_t vdata20_1 = vmull_s8(tmp21, w22);// d = 1 2 3 4
                        int16x8_t vdata30_1 = vmull_s8(tmp21, w32);// d = 1 2 3 4

                        vdata00 = vbslq_s16(vmask_result, vdata00, vzero_16);
                        vdata10 = vbslq_s16(vmask_result, vdata10, vzero_16);
                        vdata20 = vbslq_s16(vmask_result, vdata20, vzero_16);
                        vdata30 = vbslq_s16(vmask_result, vdata30, vzero_16);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30));

                        vdata00_1 = vbslq_s16(vmask_result, vdata00_1, vzero_16);
                        vdata10_1 = vbslq_s16(vmask_result, vdata10_1, vzero_16);
                        vdata20_1 = vbslq_s16(vmask_result, vdata20_1, vzero_16);
                        vdata30_1 = vbslq_s16(vmask_result, vdata30_1, vzero_16);

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00_1));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10_1));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20_1));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30_1));

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00_1));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10_1));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20_1));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30_1));


                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc1r0, outr10);
                        vst1q_s32(doutc2r0, outr20);
                        vst1q_s32(doutc3r0, outr30);

                        doutc0r0 += 4;
                        doutc1r0 += 4;
                        doutc2r0 += 4;
                        doutc3r0 += 4;

                        vst1q_s32(doutc0r0, outr01);
                        vst1q_s32(doutc1r0, outr11);
                        vst1q_s32(doutc2r0, outr21);
                        vst1q_s32(doutc3r0, outr31);

                    }
                    doutc0_ptr += wout;
                    doutc1_ptr += wout;
                    doutc2_ptr += wout;
                    doutc3_ptr += wout;
                }
                wc0 += 9;
                wc1 += 9;
                wc2 += 9;
                wc3 += 9;

            }
            //relu
            int *doutc0r0 = dout_c0;
            int *doutc1r0 = dout_c1;
            int *doutc2r0 = dout_c2;
            int *doutc3r0 = dout_c3;
            int cnt = size_out_channel / 8;
            int i = 0;
            if (cnt > 0){
                asm volatile(
                    "ldp q0, q1, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[doutc1r0]]      \n"         /* load r02, r03 to q2, q3 */
                    "ldp q4, q5, [%[doutc2r0]]      \n"         /* load r10, r11 to q4, q5 */
                    "ldp q6, q7, [%[doutc3r0]]      \n"         /* load r12, r13 to q6, q7 */
                    "movi v20.4s, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "smax   v0.4s, v0.4s, v20.4s  \n"         /*relu*/
                    "smax   v2.4s, v2.4s, v20.4s  \n"         /*relu*/
                    "smax   v4.4s, v4.4s, v20.4s  \n"         /*relu*/
                    "smax   v6.4s, v6.4s, v20.4s  \n"         /*relu*/
                    "smax   v1.4s, v1.4s, v20.4s  \n"         /*relu*/
                    "smax   v3.4s, v3.4s, v20.4s  \n"         /*relu*/
                    "str    q0, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q2, [%[doutc1r0]], #16 \n"         /* store c1r0*/
                    "str    q4, [%[doutc2r0]], #16 \n"         /* store c2r0*/
                    "str    q6, [%[doutc3r0]], #16 \n"         /* store c3r0*/
                    "smax   v5.4s, v5.4s, v20.4s  \n"         /*relu*/
                    "smax   v7.4s, v7.4s, v20.4s  \n"         /*relu*/
                    "str    q1, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q3, [%[doutc1r0]], #16 \n"         /* store c1r0*/
                    "prfm   pldl1keep, [%[doutc0r0]]                \n"
                    "prfm   pldl1keep, [%[doutc1r0]]                \n"
                    "str    q5, [%[doutc2r0]], #16 \n"         /* store c2r0*/
                    "str    q7, [%[doutc3r0]], #16 \n"         /* store c3r0*/
                    "prfm   pldl1keep, [%[doutc2r0]]                \n"
                    "prfm   pldl1keep, [%[doutc3r0]]                \n"
                    "ldp q0, q1, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[doutc1r0]]      \n"         /* load r02, r03 to q2, q3 */
                    "ldp q4, q5, [%[doutc2r0]]      \n"         /* load r10, r11 to q4, q5 */
                    "ldp q6, q7, [%[doutc3r0]]      \n"         /* load r12, r13 to q6, q7 */
                    "subs   %w[cnt], %w[cnt], #1    \n"         /* loop count -1*/
                    "bne    1b                      \n"         /* jump to main loop*/
                :[doutc0r0]"+r"(doutc0r0), [doutc1r0]"+r"(doutc1r0), \
                  [doutc2r0]"+r"(doutc2r0), [doutc3r0]"+r"(doutc3r0), [cnt] "+r"(cnt)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v20"
                );
            }
            i = i * 8;
            for (; i < size_out_channel; i++){
                *doutc0r0 = AKMAX(*doutc0r0, 0);
                *doutc1r0 = AKMAX(*doutc1r0, 0);
                *doutc2r0 = AKMAX(*doutc2r0, 0);
                *doutc3r0 = AKMAX(*doutc3r0, 0);
                doutc0r0++;
                doutc1r0++;
                doutc2r0++;
                doutc3r0++;
            }

        } //end of processing output channel
        for (int c = ch; c < chout; ++c) {

            int* dout_c = dout_batch + c * size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c, &bias[c], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c, zero, 1, size_out_channel);
            }
            const signed char* wc0 = weights + c * w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *wc0_ptr = wc0 + i * 9;
                const signed char *din_ch = din_batch + i * size_in_channel;
                int *doutc0_ptr = dout_c;
                prefetch(wc0_ptr);

                int *doutc0r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                int8x8_t w00 = vdup_n_s8(wc0_ptr[0]);
                int8x8_t w01 = vdup_n_s8(wc0_ptr[1]);
                int8x8_t w02 = vdup_n_s8(wc0_ptr[2]);

                int8x8_t w10 = vdup_n_s8(wc0_ptr[3]);
                int8x8_t w11 = vdup_n_s8(wc0_ptr[4]);
                int8x8_t w12 = vdup_n_s8(wc0_ptr[5]);

                int8x8_t w20 = vdup_n_s8(wc0_ptr[6]);
                int8x8_t w21 = vdup_n_s8(wc0_ptr[7]);
                int8x8_t w22 = vdup_n_s8(wc0_ptr[8]);

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }

                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    //prefetch input
                    prefetch(doutc0r0);

                    prefetch(din_ptr0);
                    prefetch(din_ptr1);
                    prefetch(din_ptr2);

                    //left
                    if (1){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//b = 8910111213
                        int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);
                        int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);

                        int8x8_t tmp00 = vext_s8(vzero, vinr0, 7); //c = -1 0 1 2
                        int8x8_t tmp10 = vext_s8(vzero, vinr1, 7); //c = -1 0 1 2
                        int8x8_t tmp20 = vext_s8(vzero, vinr2, 7); //c = -1 0 1 2

                        int8x8_t tmp01 = vext_s8(vinr0, vinr0_1, 1); //d = 1 2 3 4
                        int8x8_t tmp11 = vext_s8(vinr1, vinr1_1, 1); //d = 1 2 3 4
                        int8x8_t tmp21 = vext_s8(vinr2, vinr2_1, 1); //d = 1 2 3 4

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w01);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(tmp00, w00);// c = -1 0 1 2
                        int16x8_t vdata20 = vmull_s8(tmp01, w02);// d = 1 2 3 4

                        din_ptr0 += 7;
                        din_ptr1 += 7;
                        din_ptr2 += 7;

                        //r1
                        vdata00 = vmlal_s8(vdata00, vinr1, w11);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp10, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp11, w12);// d = 1 2 3 4

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);

                        //r2
                        vdata00 = vmlal_s8(vdata00, vinr2, w21);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp20, w20);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp21, w22);// d = 1 2 3 4

                        prefetch(din_ptr0);
                        prefetch(din_ptr1);
                        prefetch(din_ptr2);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata10));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata10));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata20));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata20));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc0r0 + 4, outr01);

                        doutc0r0 += 8;
                    }
                    for (int j = 0; j < cnt_col; j++){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//b = 8910111213
                        int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);
                        int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);

                        int8x8_t tmp00 = vext_s8(vinr0, vinr0_1, 1); //c = 1 2 3 4
                        int8x8_t tmp10 = vext_s8(vinr1, vinr1_1, 1); //c = 1 2 3 4
                        int8x8_t tmp20 = vext_s8(vinr2, vinr2_1, 1);//c = 1 2 3 4

                        int8x8_t tmp01 = vext_s8(vinr0, vinr0_1, 2); //d = 2 3 4 5
                        int8x8_t tmp11 = vext_s8(vinr1, vinr1_1, 2); //d = 2 3 4 5
                        int8x8_t tmp21 = vext_s8(vinr2, vinr2_1, 2); //d = 2 3 4 5

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w00);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(tmp00, w01);// c = 1 2 3 4
                        int16x8_t vdata20 = vmull_s8(tmp01, w02);// d = 2 3 4 5

                        din_ptr0 += 8;
                        din_ptr1 += 8;
                        din_ptr2 += 8;

                        //r1
                        vdata00 = vmlal_s8(vdata00, vinr1, w10);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp10, w11);// c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp11, w12);// d = 2 3 4 5

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);

                        //r2
                        vdata00 = vmlal_s8(vdata00, vinr2, w20);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp20, w21);// c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp21, w22);// d = 2 3 4 5

                        prefetch(din_ptr0);
                        prefetch(din_ptr1);
                        prefetch(din_ptr2);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata10));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata10));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata20));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata20));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc0r0 + 4, outr01);

                        doutc0r0 += 8;
                    }
                    //right
                    if (1){

                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//b = 8910111213
                        int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);
                        int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);

                        vinr0 = vbsl_s8(vmask_rp1, vinr0, vzero);
                        vinr0_1 = vbsl_s8(vmask_rp2, vinr0_1, vzero);

                        vinr1 = vbsl_s8(vmask_rp1, vinr1, vzero);
                        vinr1_1 = vbsl_s8(vmask_rp2, vinr1_1, vzero);

                        vinr2 = vbsl_s8(vmask_rp1, vinr2, vzero);
                        vinr2_1 = vbsl_s8(vmask_rp2, vinr2_1, vzero);

                        int8x8_t tmp00 = vext_s8(vinr0, vinr0_1, 1); //c = 1 2 3 4
                        int8x8_t tmp10 = vext_s8(vinr1, vinr1_1, 1); //c = 1 2 3 4
                        int8x8_t tmp20 = vext_s8(vinr2, vinr2_1, 1);//c = 1 2 3 4

                        int8x8_t tmp01 = vext_s8(vinr0, vinr0_1, 2); //d = 2 3 4 5
                        int8x8_t tmp11 = vext_s8(vinr1, vinr1_1, 2); //d = 2 3 4 5
                        int8x8_t tmp21 = vext_s8(vinr2, vinr2_1, 2); //d = 2 3 4 5

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w00);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(tmp00, w01);// c = 1 2 3 4
                        int16x8_t vdata20 = vmull_s8(tmp01, w02);// d = 2 3 4 5

                        // din_ptr0 += 8;
                        // din_ptr1 += 8;
                        // din_ptr2 += 8;

                        //r1
                        vdata00 = vmlal_s8(vdata00, vinr1, w10);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp10, w11);// c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp11, w12);// d = 2 3 4 5

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);

                        //r2
                        vdata00 = vmlal_s8(vdata00, vinr2, w20);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp20, w21);// c = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp21, w22);// d = 2 3 4 5

                        prefetch(din_ptr0);
                        prefetch(din_ptr1);
                        prefetch(din_ptr2);

                        vdata00 = vbslq_s16(vmask_result, vdata00, vzero_16);
                        vdata10 = vbslq_s16(vmask_result, vdata10, vzero_16);
                        vdata20 = vbslq_s16(vmask_result, vdata20, vzero_16);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata10));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata10));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata20));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata20));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc0r0 + 4, outr01);

                        doutc0r0 += 8;
                    }
                    doutc0_ptr += wout;
                }
            }
            //relu
            int *doutc0r0 = dout_c;
            int cnt = size_out_channel / 8;
            int i = 0;
           if (cnt > 0) {
                asm volatile(
                    "ldp q0, q1, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "movi v20.4s, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "smax   v0.4s, v0.4s, v20.4s  \n"         /*relu*/
                    "smax   v1.4s, v1.4s, v20.4s  \n"         /*relu*/
                    "subs   %w[cnt], %w[cnt], #1    \n"         /* loop count -1*/
                    "str    q0, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q1, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "prfm   pldl1keep, [%[doutc0r0]]                \n"
                    "ldp q0, q1, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "bne    1b                      \n"         /* jump to main loop*/
                :[doutc0r0]"+r"(doutc0r0), [cnt] "+r"(cnt)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v20"
                );
            }
            i = i * 8;
            for (; i < size_out_channel; i++){
                *doutc0r0 = AKMAX(*doutc0r0, 0);
                doutc0r0++;
            }
        } // end of remain out channel

    } // end of processing batchs
}

void conv_3x3s1_direct_bias_relu_s_int8(int* dout, const signed char* din, \
                          const signed char* weights, const int* bias, bool flag_bias, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, Context* ctx){
    // printf("conv3x3s1_direct_int8 start \n");
    int threads = ctx->get_threads();
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const short unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, win * sizeof(signed char));

    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    int w_stride = chin * 9;

    unsigned int size_pad_right = (unsigned int)(win);

    uint8x8_t vmask_rp = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    unsigned int rst_remain = (unsigned int)(wout);
    uint16x8_t vmask_result = vcgtq_u16(vdupq_n_u16(rst_remain), vld1q_u16(right_pad_rst));

    int8x8_t vzero = vdup_n_s8(0);
    int16x8_t vzero_16 = vdupq_n_s16(0);
    int zero[4] = {0, 0, 0, 0};

    int ch = ((chout >> 2) << 2);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * chin * size_in_channel;
        int *dout_batch = dout + n * chout * size_out_channel;
#pragma omp parallel for num_threads(threads)
        for (int c = 0; c < chout - 3; c += 4) {

            int* dout_c0 = dout_batch + c * size_out_channel;
            int* dout_c1 = dout_c0 + size_out_channel;
            int* dout_c2 = dout_c1 + size_out_channel;
            int* dout_c3 = dout_c2 + size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c0, &bias[c], 1, size_out_channel);
                fill_bias_int8(dout_c1, &bias[c + 1], 1, size_out_channel);
                fill_bias_int8(dout_c2, &bias[c + 2], 1, size_out_channel);
                fill_bias_int8(dout_c3, &bias[c + 3], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c0, zero, 1, size_out_channel);
                fill_bias_int8(dout_c1, zero, 1, size_out_channel);
                fill_bias_int8(dout_c2, zero, 1, size_out_channel);
                fill_bias_int8(dout_c3, zero, 1, size_out_channel);
            }

            const signed char* wc0 = weights + c * w_stride;
            const signed char* wc1 = wc0 + w_stride;
            const signed char* wc2 = wc1 + w_stride;
            const signed char* wc3 = wc2 + w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *din_ch = din_batch + i * size_in_channel;
                int *doutc0_ptr = dout_c0;
                int *doutc1_ptr = dout_c1;
                int *doutc2_ptr = dout_c2;
                int *doutc3_ptr = dout_c3;

                int *doutc0r0 = nullptr;
                int *doutc1r0 = nullptr;
                int *doutc2r0 = nullptr;
                int *doutc3r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    doutc1r0 = doutc1_ptr;
                    doutc2r0 = doutc2_ptr;
                    doutc3r0 = doutc3_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }
                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    //prefetch input
                    prefetch(doutc0r0);
                    prefetch(doutc1r0);
                    prefetch(doutc2r0);
                    prefetch(doutc3r0);

                    prefetch(din_ptr0);
                    prefetch(din_ptr1);
                    prefetch(din_ptr2);
                    prefetch(wc0);
                    prefetch(wc1);
                    prefetch(wc2);
                    prefetch(wc3);

                   //left
                    if (1){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        int8x8_t w01 = vdup_n_s8(wc0[1]);
                        int8x8_t w11 = vdup_n_s8(wc1[1]);
                        int8x8_t w21 = vdup_n_s8(wc2[1]);
                        int8x8_t w31 = vdup_n_s8(wc3[1]);

                        vinr0 = vbsl_s8(vmask_rp, vinr0, vzero);
                        vinr1 = vbsl_s8(vmask_rp, vinr1, vzero);
                        vinr2 = vbsl_s8(vmask_rp, vinr2, vzero);

                        int8x8_t w00 = vdup_n_s8(wc0[0]);
                        int8x8_t w10 = vdup_n_s8(wc1[0]);
                        int8x8_t w20 = vdup_n_s8(wc2[0]);
                        int8x8_t w30 = vdup_n_s8(wc3[0]);

                        int8x8_t tmp00 = vext_s8(vzero, vinr0, 7); //c = -1 0 1 2
                        int8x8_t tmp10 = vext_s8(vzero, vinr1, 7); //c = -1 0 1 2
                        int8x8_t tmp20 = vext_s8(vzero, vinr2, 7); //c = -1 0 1 2

                        int8x8_t tmp01 = vext_s8(vinr0, vzero, 1); //d = 1 2 3 4
                        int8x8_t tmp11 = vext_s8(vinr1, vzero, 1); //d = 1 2 3 4
                        int8x8_t tmp21 = vext_s8(vinr2, vzero, 1); //d = 1 2 3 4

                        int8x8_t w02 = vdup_n_s8(wc0[2]);
                        int8x8_t w12 = vdup_n_s8(wc1[2]);
                        int8x8_t w22 = vdup_n_s8(wc2[2]);
                        int8x8_t w32 = vdup_n_s8(wc3[2]);

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w01);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(vinr0, w11);// a = 0 1 2 3
                        int16x8_t vdata20 = vmull_s8(vinr0, w21);// a = 0 1 2 3
                        int16x8_t vdata30 = vmull_s8(vinr0, w31);// a = 0 1 2 3

                        vdata00 = vmlal_s8(vdata00, tmp00, w00);// c = -1 0 1 2
                        vdata10 = vmlal_s8(vdata10, tmp00, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp00, w20);// c = -1 0 1 2
                        vdata30 = vmlal_s8(vdata30, tmp00, w30);// c = -1 0 1 2

                        w00 = vdup_n_s8(wc0[3]);
                        w10 = vdup_n_s8(wc1[3]);
                        w20 = vdup_n_s8(wc2[3]);
                        w30 = vdup_n_s8(wc3[3]);

                        vdata00 = vmlal_s8(vdata00, tmp01, w02);// d = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp01, w12);// d = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp01, w22);// d = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp01, w32);// d = 1 2 3 4

                        //r1
                        w01 = vdup_n_s8(wc0[4]);
                        w11 = vdup_n_s8(wc1[4]);
                        w21 = vdup_n_s8(wc2[4]);
                        w31 = vdup_n_s8(wc3[4]);

                        vdata00 = vmlal_s8(vdata00, tmp10, w00);// c = -1 0 1 2
                        vdata10 = vmlal_s8(vdata10, tmp10, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp10, w20);// c = -1 0 1 2
                        vdata30 = vmlal_s8(vdata30, tmp10, w30);// c = -1 0 1 2

                        w02 = vdup_n_s8(wc0[5]);
                        w12 = vdup_n_s8(wc1[5]);
                        w22 = vdup_n_s8(wc2[5]);
                        w32 = vdup_n_s8(wc3[5]);

                        vdata00 = vmlal_s8(vdata00, vinr1, w01);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr1, w11);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr1, w21);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr1, w31);// a = 0 1 2 3

                        w00 = vdup_n_s8(wc0[6]);
                        w10 = vdup_n_s8(wc1[6]);
                        w20 = vdup_n_s8(wc2[6]);
                        w30 = vdup_n_s8(wc3[6]);

                        vdata00 = vmlal_s8(vdata00, tmp11, w02);// d = 1 2 3 4
                        vdata10 = vmlal_s8(vdata10, tmp11, w12);// d = 1 2 3 4
                        vdata20 = vmlal_s8(vdata20, tmp11, w22);// d = 1 2 3 4
                        vdata30 = vmlal_s8(vdata30, tmp11, w32);// d = 1 2 3 4

                        //r2
                        w01 = vdup_n_s8(wc0[7]);
                        w11 = vdup_n_s8(wc1[7]);
                        w21 = vdup_n_s8(wc2[7]);
                        w31 = vdup_n_s8(wc3[7]);

                        vdata00 = vmlal_s8(vdata00, tmp20, w00);// c = -1 0 1 2
                        vdata10 = vmlal_s8(vdata10, tmp20, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp20, w20);// c = -1 0 1 2
                        vdata30 = vmlal_s8(vdata30, tmp20, w30);// c = -1 0 1 2

                        w02 = vdup_n_s8(wc0[8]);
                        w12 = vdup_n_s8(wc1[8]);
                        w22 = vdup_n_s8(wc2[8]);
                        w32 = vdup_n_s8(wc3[8]);

                        vdata00 = vmlal_s8(vdata00, vinr2, w01);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, vinr2, w11);// a = 0 1 2 3
                        vdata20 = vmlal_s8(vdata20, vinr2, w21);// a = 0 1 2 3
                        vdata30 = vmlal_s8(vdata30, vinr2, w31);// a = 0 1 2 3

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr10 = vld1q_s32(doutc1r0);
                        int32x4_t outr20 = vld1q_s32(doutc2r0);
                        int32x4_t outr30 = vld1q_s32(doutc3r0);

                        int16x8_t vdata00_1 = vmull_s8(tmp21, w02);// d = 1 2 3 4
                        int16x8_t vdata10_1 = vmull_s8(tmp21, w12);// d = 1 2 3 4
                        int16x8_t vdata20_1 = vmull_s8(tmp21, w22);// d = 1 2 3 4
                        int16x8_t vdata30_1 = vmull_s8(tmp21, w32);// d = 1 2 3 4

                        // vdata00 = vmlal_s8(vdata00, tmp21, w02);// d = 1 2 3 4
                        // vdata10 = vmlal_s8(vdata10, tmp21, w12);// d = 1 2 3 4
                        // vdata20 = vmlal_s8(vdata20, tmp21, w22);// d = 1 2 3 4
                        // vdata30 = vmlal_s8(vdata30, tmp21, w32);// d = 1 2 3 4

                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);
                        int32x4_t outr11 = vld1q_s32(doutc1r0 + 4);
                        int32x4_t outr21 = vld1q_s32(doutc2r0 + 4);
                        int32x4_t outr31 = vld1q_s32(doutc3r0 + 4);

                        vdata00 = vbslq_s16(vmask_result, vdata00, vzero_16);
                        vdata10 = vbslq_s16(vmask_result, vdata10, vzero_16);
                        vdata20 = vbslq_s16(vmask_result, vdata20, vzero_16);
                        vdata30 = vbslq_s16(vmask_result, vdata30, vzero_16);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30));

                        vdata00_1 = vbslq_s16(vmask_result, vdata00_1, vzero_16);
                        vdata10_1 = vbslq_s16(vmask_result, vdata10_1, vzero_16);
                        vdata20_1 = vbslq_s16(vmask_result, vdata20_1, vzero_16);
                        vdata30_1 = vbslq_s16(vmask_result, vdata30_1, vzero_16);

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00_1));
                        outr10 = vaddw_s16(outr10, vget_low_s16(vdata10_1));
                        outr20 = vaddw_s16(outr20, vget_low_s16(vdata20_1));
                        outr30 = vaddw_s16(outr30, vget_low_s16(vdata30_1));

                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00_1));
                        outr11 = vaddw_s16(outr11, vget_high_s16(vdata10_1));
                        outr21 = vaddw_s16(outr21, vget_high_s16(vdata20_1));
                        outr31 = vaddw_s16(outr31, vget_high_s16(vdata30_1));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc1r0, outr10);
                        vst1q_s32(doutc2r0, outr20);
                        vst1q_s32(doutc3r0, outr30);

                        vst1q_s32(doutc0r0 + 4, outr01);
                        vst1q_s32(doutc1r0 + 4, outr11);
                        vst1q_s32(doutc2r0 + 4, outr21);
                        vst1q_s32(doutc3r0 + 4, outr31);
                    }
                    doutc0_ptr += wout;
                    doutc1_ptr += wout;
                    doutc2_ptr += wout;
                    doutc3_ptr += wout;
                }
                wc0 += 9;
                wc1 += 9;
                wc2 += 9;
                wc3 += 9;

            }
            //relu
            int *doutc0r0 = dout_c0;
            int *doutc1r0 = dout_c1;
            int *doutc2r0 = dout_c2;
            int *doutc3r0 = dout_c3;
            int cnt = size_out_channel / 8;
            int i = 0;
            if (cnt > 0) {
                asm volatile(
                    "ldp q0, q1, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[doutc1r0]]      \n"         /* load r02, r03 to q2, q3 */
                    "ldp q4, q5, [%[doutc2r0]]      \n"         /* load r10, r11 to q4, q5 */
                    "ldp q6, q7, [%[doutc3r0]]      \n"         /* load r12, r13 to q6, q7 */
                    "movi v20.4s, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "smax   v0.4s, v0.4s, v20.4s  \n"         /*relu*/
                    "smax   v2.4s, v2.4s, v20.4s  \n"         /*relu*/
                    "smax   v4.4s, v4.4s, v20.4s  \n"         /*relu*/
                    "smax   v6.4s, v6.4s, v20.4s  \n"         /*relu*/
                    "smax   v1.4s, v1.4s, v20.4s  \n"         /*relu*/
                    "smax   v3.4s, v3.4s, v20.4s  \n"         /*relu*/
                    "str    q0, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q2, [%[doutc1r0]], #16 \n"         /* store c1r0*/
                    "str    q4, [%[doutc2r0]], #16 \n"         /* store c2r0*/
                    "str    q6, [%[doutc3r0]], #16 \n"         /* store c3r0*/
                    "smax   v5.4s, v5.4s, v20.4s  \n"         /*relu*/
                    "smax   v7.4s, v7.4s, v20.4s  \n"         /*relu*/
                    "str    q1, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q3, [%[doutc1r0]], #16 \n"         /* store c1r0*/
                    "prfm   pldl1keep, [%[doutc0r0]]                \n"
                    "prfm   pldl1keep, [%[doutc1r0]]                \n"
                    "str    q5, [%[doutc2r0]], #16 \n"         /* store c2r0*/
                    "str    q7, [%[doutc3r0]], #16 \n"         /* store c3r0*/
                    "prfm   pldl1keep, [%[doutc2r0]]                \n"
                    "prfm   pldl1keep, [%[doutc3r0]]                \n"
                    "ldp q0, q1, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "ldp q2, q3, [%[doutc1r0]]      \n"         /* load r02, r03 to q2, q3 */
                    "ldp q4, q5, [%[doutc2r0]]      \n"         /* load r10, r11 to q4, q5 */
                    "ldp q6, q7, [%[doutc3r0]]      \n"         /* load r12, r13 to q6, q7 */
                    "subs   %w[cnt], %w[cnt], #1    \n"         /* loop count -1*/
                    "bne    1b                      \n"         /* jump to main loop*/
                :[doutc0r0]"+r"(doutc0r0), [doutc1r0]"+r"(doutc1r0), \
                  [doutc2r0]"+r"(doutc2r0), [doutc3r0]"+r"(doutc3r0), [cnt] "+r"(cnt)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v20"
                );
            }
            i = i * 8;
            for (; i < size_out_channel; i++){
                *doutc0r0 = AKMAX(*doutc0r0, 0);
                *doutc1r0 = AKMAX(*doutc1r0, 0);
                *doutc2r0 = AKMAX(*doutc2r0, 0);
                *doutc3r0 = AKMAX(*doutc3r0, 0);
                doutc0r0++;
                doutc1r0++;
                doutc2r0++;
                doutc3r0++;
            }

        } //end of processing output channel
        for (int c = ch; c < chout; ++c) {

            int* dout_c = dout_batch + c * size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c, &bias[c], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c, zero, 1, size_out_channel);
            }
            const signed char* wc0 = weights + c * w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *wc0_ptr = wc0 + i * 9;
                const signed char *din_ch = din_batch + i * size_in_channel;
                int *doutc0_ptr = dout_c;
                prefetch(wc0_ptr);

                int *doutc0r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                int8x8_t w00 = vdup_n_s8(wc0_ptr[0]);
                int8x8_t w01 = vdup_n_s8(wc0_ptr[1]);
                int8x8_t w02 = vdup_n_s8(wc0_ptr[2]);

                int8x8_t w10 = vdup_n_s8(wc0_ptr[3]);
                int8x8_t w11 = vdup_n_s8(wc0_ptr[4]);
                int8x8_t w12 = vdup_n_s8(wc0_ptr[5]);

                int8x8_t w20 = vdup_n_s8(wc0_ptr[6]);
                int8x8_t w21 = vdup_n_s8(wc0_ptr[7]);
                int8x8_t w22 = vdup_n_s8(wc0_ptr[8]);

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }

                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    //prefetch input
                    prefetch(doutc0r0);

                    prefetch(din_ptr0);
                    prefetch(din_ptr1);
                    prefetch(din_ptr2);

                    //left
                    if (1){
                        int8x8_t vinr0 = vld1_s8(din_ptr0);//a = 01234567
                        int8x8_t vinr1 = vld1_s8(din_ptr1);
                        int8x8_t vinr2 = vld1_s8(din_ptr2);

                        vinr0 = vbsl_s8(vmask_rp, vinr0, vzero);
                        vinr1 = vbsl_s8(vmask_rp, vinr1, vzero);
                        vinr2 = vbsl_s8(vmask_rp, vinr2, vzero);

                        int32x4_t outr00 = vld1q_s32(doutc0r0);
                        int32x4_t outr01 = vld1q_s32(doutc0r0 + 4);

                        int8x8_t tmp00 = vext_s8(vzero, vinr0, 7); //c = -1 0 1 2
                        int8x8_t tmp10 = vext_s8(vzero, vinr1, 7); //c = -1 0 1 2
                        int8x8_t tmp20 = vext_s8(vzero, vinr2, 7); //c = -1 0 1 2

                        int8x8_t tmp01 = vext_s8(vinr0, vzero, 1); //d = 1 2 3 4
                        int8x8_t tmp11 = vext_s8(vinr1, vzero, 1); //d = 1 2 3 4
                        int8x8_t tmp21 = vext_s8(vinr2, vzero, 1); //d = 1 2 3 4

                        //r0
                        int16x8_t vdata00 = vmull_s8(vinr0, w01);// a = 0 1 2 3
                        int16x8_t vdata10 = vmull_s8(tmp00, w00);// c = -1 0 1 2
                        int16x8_t vdata20 = vmull_s8(tmp01, w02);// d = 1 2 3 4

                        //r1
                        vdata00 = vmlal_s8(vdata00, vinr1, w11);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp10, w10);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp11, w12);// d = 1 2 3 4

                        //r2
                        vdata00 = vmlal_s8(vdata00, vinr2, w21);// a = 0 1 2 3
                        vdata10 = vmlal_s8(vdata10, tmp20, w20);// c = -1 0 1 2
                        vdata20 = vmlal_s8(vdata20, tmp21, w22);// d = 1 2 3 4

                        vdata00 = vbslq_s16(vmask_result, vdata00, vzero_16);
                        vdata10 = vbslq_s16(vmask_result, vdata10, vzero_16);
                        vdata20 = vbslq_s16(vmask_result, vdata20, vzero_16);

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata00));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata00));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata10));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata10));

                        outr00 = vaddw_s16(outr00, vget_low_s16(vdata20));
                        outr01 = vaddw_s16(outr01, vget_high_s16(vdata20));

                        vst1q_s32(doutc0r0, outr00);
                        vst1q_s32(doutc0r0 + 4, outr01);
                    }
                    doutc0_ptr += wout;
                }
            }

            //relu
            int *doutc0r0 = dout_c;
            int cnt = size_out_channel / 8;
            int i = 0;
            if (cnt > 0) {
                asm volatile(
                    "ldp q0, q1, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "movi v20.4s, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "smax   v0.4s, v0.4s, v20.4s  \n"         /*relu*/
                    "smax   v1.4s, v1.4s, v20.4s  \n"         /*relu*/
                    "subs   %w[cnt], %w[cnt], #1    \n"         /* loop count -1*/
                    "str    q0, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "str    q1, [%[doutc0r0]], #16 \n"         /* store c0r0*/
                    "prfm   pldl1keep, [%[doutc0r0]]                \n"
                    "ldp q0, q1, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "bne    1b                      \n"         /* jump to main loop*/
                :[doutc0r0]"+r"(doutc0r0), [cnt] "+r"(cnt)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v20"
                );
            }
            i = i * 8;
            for (; i < size_out_channel; i++){
                *doutc0r0 = AKMAX(*doutc0r0, 0);
                doutc0r0++;
            }
        } // end of remain out channel

    } // end of processing batchs
}

#else
void conv_3x3s1_direct_bias_int8(int* dout, const signed char* din, \
                          const signed char* weights, const int* bias, bool flag_bias, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, Context* ctx) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    //printf("conv3x3_dw start \n");
    int threads = ctx->get_threads();
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const short unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, win * sizeof(signed char));

    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    int w_stride = chin * 9;

    int tile_w = (win + 7) >> 3;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(win - 7 - (cnt_col << 3));

    // uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    // uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));

    unsigned int rst_remain = (unsigned int)(wout - ((cnt_col + 1) << 3));
    uint16x8_t vmask_result = vcgtq_u16(vdupq_n_u16(rst_remain), vld1q_u16(right_pad_rst));

    int8x8_t vzero = vdup_n_s8(0);
    int16x8_t vzero_16 = vdupq_n_s16(0);
    int zero[4] = {0, 0, 0, 0};

    unsigned char vmask[16];
    vst1q_u8(vmask, vmask_rp);

    short unsigned int rmask[8];
    vst1q_u16(rmask, vmask_result);

    int ch = ((chout >> 1) << 1);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * chin * size_in_channel;
        int *dout_batch = dout + n * chout * size_out_channel;
#pragma omp parallel for num_threads(threads)
        for (int c = 0; c < chout - 1; c += 2) {

            int* dout_c0 = dout_batch + c * size_out_channel;
            int* dout_c1 = dout_c0 + size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c0, &bias[c], 1, size_out_channel);
                fill_bias_int8(dout_c1, &bias[c + 1], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c0, zero, 1, size_out_channel);
                fill_bias_int8(dout_c1, zero, 1, size_out_channel);
            }

            const signed char* wc0 = weights + c * w_stride;
            const signed char* wc1 = wc0 + w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *din_ch = din_batch + i * size_in_channel;
                const signed char *wc0_ptr = wc0 + i * 9;
                const signed char *wc1_ptr = wc1 + i * 9;
                int *doutc0_ptr = dout_c0;
                int *doutc1_ptr = dout_c1;

                int *doutc0r0 = nullptr;
                int *doutc1r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wc0]]    \n"
                    "vld1.8    {d2-d3}, [%[wc1]]    \n"
                    :
                    :[wc0] "r" (wc0_ptr), [wc1] "r" (wc1_ptr)
                    : "memory"
                );

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    doutc1r0 = doutc1_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }
                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    int cnt = cnt_col;
                    asm volatile(
                    //left
                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"
                        "vdup.s8     d5, d0[1]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[1]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d4, d0[0]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[0]               @ d2 = w00, w00, w00, w00\n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"
                        "vmov.u32 d11, #0                   @ zero\n"
                        "vdup.s8     d6, d0[2]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[2]               @ d4 = w02, w02, w02, w02\n"

                        "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678

                        //r0
                        "vmull.s8 q12, d12, d5                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                        "vmull.s8 q13, d12, d8                 @ out1 = din0 * w01 \n" // q12 = d12 * w01

                        "vdup.s8     d5, d0[4]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[4]               @ d3 = w01, w01, w01, w01\n"

                        "vmlal.s8 q12, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vdup.s8     d4, d0[3]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[3]               @ d2 = w00, w00, w00, w00\n"
                        "pld [%[doutc0r0]]                @ preload data\n"
                        "pld [%[doutc1r0]]                @ preload data\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d0[5]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[5]               @ d4 = w02, w02, w02, w02\n"

                        //r1
                        "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d14, d15, #1          @ ext \n" //d11 = 12345678

                        "vmlal.s8 q12, d14, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d14, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01

                        "vdup.s8     d5, d0[7]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[7]               @ d3 = w01, w01, w01, w01\n"

                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"
                        "vld1.32 {d14-d15}, [%[doutc1r0]]!           @ load dout \n"

                        "vmlal.s8 q12, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vdup.s8     d4, d0[6]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[6]               @ d2 = w00, w00, w00, w00\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d1[0]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d3[0]               @ d4 = w02, w02, w02, w02\n"

                        //r2
                        "vext.8     d30, d11, d16, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d16, d17, #1          @ ext \n" //d11 = 12345678

                        "vmlal.s8 q12, d16, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d16, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01

                        "vld1.32 {d18-d19}, [%[doutc0r0]]           @ load dout \n"
                        "vld1.32 {d20-d21}, [%[doutc1r0]]           @ load dout \n"

                        "vmlal.s8 q12, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "sub %[doutc0r0], #16          @ sub \n"
                        "sub %[doutc1r0], #16          @ sub \n"
                        "add %[din_ptr0], #7                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "vmull.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmull.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "add %[din_ptr1], #7                   @add \n"
                        "add %[din_ptr2], #7                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d14-d15}, [%[doutc1r0]]!         @ store\n"
                        "cmp %[cnt], #1                                 \n"
                        "vst1.32 {d18-d19}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d20-d21}, [%[doutc1r0]]!         @ store\n"
                        "blt 1f                                         \n"
                    //mid
                        "2:                                             \n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"

                        "vdup.s8     d4, d0[0]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[0]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d5, d0[1]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[1]               @ d3 = w01, w01, w01, w01\n"

                        "vext.8     d30, d12, d13, #1         @ ext \n" //d30 = 12345678
                        "vext.8     d31, d12, d13, #2          @ ext \n" //31 = 23456789

                        "vdup.s8     d6, d0[2]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[2]               @ d4 = w02, w02, w02, w02\n"

                        //r0
                        "vmull.s8 q12, d12, d4                 @ out0 = din0 * w00 \n" // q12 = d12 * w00
                        "vmull.s8 q13, d12, d7                 @ out1 = din0 * w00 \n" // q12 = d12 * w00

                        "vdup.s8     d4, d0[3]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[3]               @ d2 = w00, w00, w00, w00\n"

                        "vmlal.s8 q12, d30, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01

                        "vdup.s8     d5, d0[4]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[4]               @ d3 = w01, w01, w01, w01\n"
                        "pld [%[doutc0r0]]                @ preload data\n"
                        "pld [%[doutc1r0]]                @ preload data\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d0[5]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[5]               @ d4 = w02, w02, w02, w02\n"

                        //r1
                        "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 23456789

                        "vmlal.s8 q12, d14, d4                 @ out0 += din0 * w00 \n"
                        "vmlal.s8 q13, d14, d7                 @ out0 += din0 * w00 \n"

                        "vdup.s8     d4, d0[6]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[6]               @ d2 = w00, w00, w00, w00\n"

                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"
                        "vld1.32 {d14-d15}, [%[doutc1r0]]!           @ load dout \n"

                        "vmlal.s8 q12, d30, d5                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vdup.s8     d5, d0[7]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[7]               @ d3 = w01, w01, w01, w01\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d1[0]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d3[0]               @ d4 = w02, w02, w02, w02\n"

                        //r2
                        "vext.8     d30, d16, d17, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d16, d17, #2          @ ext \n" //d11 = 23456789

                        "vmlal.s8 q12, d16, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d16, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w01

                        "vld1.32 {d18-d19}, [%[doutc0r0]]           @ load dout \n"
                        "vld1.32 {d20-d21}, [%[doutc1r0]]           @ load dout \n"

                        "vmlal.s8 q12, d30, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w00

                        "sub %[doutc0r0], #16          @ sub \n"
                        "sub %[doutc1r0], #16          @ sub \n"
                        "add %[din_ptr0], #8                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "vmull.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmull.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "add %[din_ptr1], #8                   @add \n"
                        "add %[din_ptr2], #8                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d14-d15}, [%[doutc1r0]]!         @ store\n"
                        "subs %[cnt], #1                                \n"
                        "vst1.32 {d18-d19}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d20-d21}, [%[doutc1r0]]!         @ store\n"
                        "bne  2b                                        \n"

                    //right
                        "1:                                             \n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"
                        "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                        "vdup.s8     d4, d0[0]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[0]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d5, d0[1]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[1]               @ d3 = w01, w01, w01, w01\n"

                        "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                        "vdup.s8     d6, d0[2]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[2]               @ d4 = w02, w02, w02, w02\n"

                        "vext.8     d30, d12, d13, #1         @ ext \n" //d30 = 12345678
                        "vext.8     d31, d12, d13, #2          @ ext \n" //31 = 23456789

                        //r0
                        "vmull.s8 q12, d12, d4                 @ out0 = din0 * w00 \n" // q12 = d12 * w00
                        "vmull.s8 q13, d12, d7                 @ out1 = din0 * w00 \n" // q12 = d12 * w00

                        "vdup.s8     d4, d0[3]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[3]               @ d2 = w00, w00, w00, w00\n"

                        "vmlal.s8 q12, d30, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01

                        "vdup.s8     d5, d0[4]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[4]               @ d3 = w01, w01, w01, w01\n"
                        "pld [%[doutc0r0]]                @ preload data\n"
                        "pld [%[doutc1r0]]                @ preload data\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d0[5]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[5]               @ d4 = w02, w02, w02, w02\n"

                        //r1
                        "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 23456789

                        "vmlal.s8 q12, d14, d4                 @ out0 += din0 * w00 \n"
                        "vmlal.s8 q13, d14, d7                 @ out0 += din0 * w00 \n"

                        "vdup.s8     d4, d0[6]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[6]               @ d2 = w00, w00, w00, w00\n"

                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"
                        "vld1.32 {d14-d15}, [%[doutc1r0]]!           @ load dout \n"

                        "vmlal.s8 q12, d30, d5                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vdup.s8     d5, d0[7]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[7]               @ d3 = w01, w01, w01, w01\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d1[0]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d3[0]               @ d4 = w02, w02, w02, w02\n"

                        //r2
                        "vext.8     d30, d16, d17, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d16, d17, #2          @ ext \n" //d11 = 23456789

                        "vmlal.s8 q12, d16, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d16, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w01

                        "vld1.16 {d22-d23}, [%[rs_mask]]     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vmov.u16 q14, #0                   @ zero\n"

                        "vmlal.s8 q12, d30, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w00

                        "vld1.32 {d18-d19}, [%[doutc0r0]]           @ load dout \n"
                        "vld1.32 {d20-d21}, [%[doutc1r0]]           @ load dout \n"

                        "vbif.16 q12, q14, q11                     @bif \n"
                        "vbif.16 q13, q14, q11                     @bif \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "vmull.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmull.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "sub %[doutc0r0], #16          @ sub \n"
                        "sub %[doutc1r0], #16          @ sub \n"

                        "vbif.16 q12, q14, q11                     @bif \n"
                        "vbif.16 q13, q14, q11                     @bif \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d14-d15}, [%[doutc1r0]]!         @ store\n"
                        "vst1.32 {d18-d19}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d20-d21}, [%[doutc1r0]]!         @ store\n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [doutc0r0] "+r"(doutc0r0), [doutc1r0] "+r"(doutc1r0), [cnt] "+r" (cnt)
                    :[mask] "r" (vmask), [rs_mask] "r" (rmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"

                    );

                    doutc0_ptr += wout;
                    doutc1_ptr += wout;
                }

            }

        } //end of processing output channel
        for (int c = ch; c < chout; ++c) {

            int* dout_c = dout_batch + c * size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c, &bias[c], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c, zero, 1, size_out_channel);
            }
            const signed char* wc0 = weights + c * w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *wei_ptr = wc0 + i * 9;
                const signed char *din_ch = din_batch + i * size_in_channel;
                int *doutc0_ptr = dout_c;

                int *doutc0r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }

                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }

                    int cnt = cnt_col;
                    asm volatile(
                    //left
                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"
                        "vdup.s8     d4, d0[0]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d5, d0[1]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d6, d0[2]               @ d4 = w02, w02, w02, w02\n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"
                        "vmov.u32 d11, #0                   @ zero\n"
                        "vdup.s8     d7, d0[3]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d8, d0[4]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d0[5]               @ d4 = w02, w02, w02, w02\n"

                        "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678

                        //r0
                        "vmull.s8 q12, d12, d5                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                        "vmull.s8 q13, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmull.s8 q14, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d18, d0[6]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d19, d0[7]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d20, d1[0]               @ d4 = w02, w02, w02, w02\n"

                        "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d14, d15, #1          @ ext \n" //d11 = 12345678

                        //r1
                        "vmlal.s8 q12, d14, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q14, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vext.8     d30, d11, d16, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d16, d17, #1          @ ext \n" //d11 = 12345678
                        "pld [%[doutc0r0]]                @ preload data\n"

                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"

                        //r2
                        "vmlal.s8 q12, d16, d19                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d18                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q14, d31, d20                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vld1.32 {d14-d15}, [%[doutc0r0]]           @ load dout \n"

                        "add %[din_ptr0], #7                   @add \n"
                        "add %[din_ptr1], #7                   @add \n"
                        "add %[din_ptr2], #7                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d25                 @addw \n" // out1 += vget_low_s16(out10)

                        "sub %[doutc0r0], #16          @ sub \n"

                        "vaddw.s16 q6, q6, d26                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d27                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q6, q6, d28                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d29                 @addw \n" // out1 += vget_low_s16(out10)

                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "cmp %[cnt], #1                                 \n"
                        "vst1.32 {d14-d15}, [%[doutc0r0]]!         @ store\n"
                        "blt 1f                                         \n"
                    //mid
                        "2:                                             \n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"

                        "pld [%[doutc0r0]]                @ preload data\n"

                        "vext.8     d30, d12, d13, #1         @ ext \n" //d30 = 12345678
                        "vext.8     d31, d12, d13, #2          @ ext \n" //31 = 23456789

                        //r0
                        "vmull.s8 q12, d12, d4                 @ out0 = din0 * w00 \n" // q12 = d12 * w00
                        "vmull.s8 q13, d30, d5                 @ out1 = din0 * w01 \n" // q12 = d12 * w00
                        "vmull.s8 q14, d31, d6                 @ out1 = din0 * w02 \n" // q12 = d12 * w00\

                        //r1
                        "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 23456789
                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"

                        "vmlal.s8 q12, d14, d7                 @ out0 += din0 * w00 \n"
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w01 \n"
                        "vmlal.s8 q14, d31, d9                 @ out0 += din0 * w02 \n"


                        //r2
                        "vext.8     d30, d16, d17, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d16, d17, #2          @ ext \n" //d11 = 23456789
                        "vld1.32 {d14-d15}, [%[doutc0r0]]           @ load dout \n"
                        "sub %[doutc0r0], #16          @ sub \n"

                        "vmlal.s8 q12, d16, d18                 @ out0 += din0 * w00 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d19                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q14, d31, d20                 @ out0 += din0 * w02 \n" // q12 += d10 * w01

                        "add %[din_ptr0], #8                   @add \n"
                        "add %[din_ptr1], #8                   @add \n"
                        "add %[din_ptr2], #8                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d25                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q6, q6, d26                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d27                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q6, q6, d28                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d29                 @addw \n" // out1 += vget_low_s16(out10)

                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "subs %[cnt], #1                                \n"
                        "vst1.32 {d14-d15}, [%[doutc0r0]]!         @ store\n"
                        "bne  2b                                        \n"

                    //right
                        "1:                                             \n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"
                        "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                        "pld [%[doutc0r0]]                @ preload data\n"

                        "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                        "vext.8     d30, d12, d13, #1         @ ext \n" //d30 = 12345678
                        "vext.8     d31, d12, d13, #2          @ ext \n" //31 = 23456789

                        "vld1.16 {d22-d23}, [%[rs_mask]]     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vmov.u16 q5, #0                   @ zero\n"

                        //r0
                        "vmull.s8 q12, d12, d4                 @ out0 = din0 * w00 \n" // q12 = d12 * w00
                        "vmull.s8 q13, d30, d5                 @ out1 = din0 * w01 \n" // q12 = d12 * w00
                        "vmull.s8 q14, d31, d6                 @ out1 = din0 * w02 \n" // q12 = d12 * w00\

                        //r1
                        "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 23456789
                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"

                        "vmlal.s8 q12, d14, d7                 @ out0 += din0 * w00 \n"
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w01 \n"
                        "vmlal.s8 q14, d31, d9                 @ out0 += din0 * w02 \n"

                        //r2
                        "vext.8     d30, d16, d17, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d16, d17, #2          @ ext \n" //d11 = 23456789
                        "vld1.32 {d14-d15}, [%[doutc0r0]]           @ load dout \n"
                        "sub %[doutc0r0], #16          @ sub \n"

                        "vmlal.s8 q12, d16, d18                 @ out0 += din0 * w00 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d19                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q14, d31, d20                 @ out0 += din0 * w02 \n" // q12 += d10 * w01

                        "vbif.16 q12, q5, q11                     @bif \n"
                        "vbif.16 q13, q5, q11                     @bif \n"
                        "vbif.16 q14, q5, q11                     @bif \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d25                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q6, q6, d26                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d27                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q6, q6, d28                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d29                 @addw \n" // out1 += vget_low_s16(out10)

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d14-d15}, [%[doutc0r0]]!         @ store\n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [doutc0r0] "+r"(doutc0r0), [cnt] "+r" (cnt)
                    :[mask] "r" (vmask), [rs_mask] "r" (rmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"
                    );
                    doutc0_ptr += wout;
                }

            }
        } // end of remain out channel

    } // end of processing batchs
}

void conv_3x3s1_direct_bias_s_int8(int* dout, const signed char* din, \
                          const signed char* weights, const int* bias, bool flag_bias, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, Context* ctx){
    //printf("conv3x3_dw start \n");
    int threads = ctx->get_threads();
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const short unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, win * sizeof(signed char));

    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    int w_stride = chin * 9;

    unsigned int size_pad_right = (unsigned int)(win);

    // uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    // uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    uint8x8_t vmask_rp = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));

    unsigned int rst_remain = (unsigned int)(wout);
    uint16x8_t vmask_result = vcgtq_u16(vdupq_n_u16(rst_remain), vld1q_u16(right_pad_rst));

    int8x8_t vzero = vdup_n_s8(0);
    int16x8_t vzero_16 = vdupq_n_s16(0);
    int zero[4] = {0, 0, 0, 0};

    unsigned char vmask[8];
    vst1_u8(vmask, vmask_rp);

    short unsigned int rmask[8];
    vst1q_u16(rmask, vmask_result);

    int ch = ((chout >> 1) << 1);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * chin * size_in_channel;
        int *dout_batch = dout + n * chout * size_out_channel;
#pragma omp parallel for num_threads(threads)
        for (int c = 0; c < chout - 1; c += 2) {

            int* dout_c0 = dout_batch + c * size_out_channel;
            int* dout_c1 = dout_c0 + size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c0, &bias[c], 1, size_out_channel);
                fill_bias_int8(dout_c1, &bias[c + 1], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c0, zero, 1, size_out_channel);
                fill_bias_int8(dout_c1, zero, 1, size_out_channel);
            }

            const signed char* wc0 = weights + c * w_stride;
            const signed char* wc1 = wc0 + w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *din_ch = din_batch + i * size_in_channel;
                const signed char *wc0_ptr = wc0 + i * 9;
                const signed char *wc1_ptr = wc1 + i * 9;
                int *doutc0_ptr = dout_c0;
                int *doutc1_ptr = dout_c1;

                int *doutc0r0 = nullptr;
                int *doutc1r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wc0]]    \n"
                    "vld1.8    {d2-d3}, [%[wc1]]    \n"
                    :
                    :[wc0] "r" (wc0_ptr), [wc1] "r" (wc1_ptr)
                    : "memory"
                );

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    doutc1r0 = doutc1_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }
                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    asm volatile(
                    //left
                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"
                        "vdup.s8     d5, d0[1]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[1]               @ d3 = w01, w01, w01, w01\n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"
                        "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                        "vdup.s8     d4, d0[0]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[0]               @ d2 = w00, w00, w00, w00\n"
                        "vmov.u32 d11, #0                   @ zero\n"

                        "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                        "vdup.s8     d6, d0[2]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[2]               @ d4 = w02, w02, w02, w02\n"

                        "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678

                        //r0
                        "vmull.s8 q12, d12, d5                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                        "vmull.s8 q13, d12, d8                 @ out1 = din0 * w01 \n" // q12 = d12 * w01

                        "vdup.s8     d5, d0[4]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[4]               @ d3 = w01, w01, w01, w01\n"

                        "vmlal.s8 q12, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vdup.s8     d4, d0[3]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[3]               @ d2 = w00, w00, w00, w00\n"
                        "pld [%[doutc0r0]]                @ preload data\n"
                        "pld [%[doutc1r0]]                @ preload data\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d0[5]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[5]               @ d4 = w02, w02, w02, w02\n"

                        //r1
                        "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d14, d15, #1          @ ext \n" //d11 = 12345678

                        "vmlal.s8 q12, d14, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d14, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01

                        "vdup.s8     d5, d0[7]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[7]               @ d3 = w01, w01, w01, w01\n"

                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"
                        "vld1.32 {d14-d15}, [%[doutc1r0]]!           @ load dout \n"

                        "vmlal.s8 q12, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vdup.s8     d4, d0[6]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[6]               @ d2 = w00, w00, w00, w00\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d1[0]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d3[0]               @ d4 = w02, w02, w02, w02\n"

                        //r2
                        "vext.8     d30, d11, d16, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d16, d17, #1          @ ext \n" //d11 = 12345678

                        "vmlal.s8 q12, d16, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d16, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01

                        "vld1.16 {d22-d23}, [%[rs_mask]]     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vmov.u16 q14, #0                   @ zero\n"

                        "vmlal.s8 q12, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vld1.32 {d18-d19}, [%[doutc0r0]]           @ load dout \n"
                        "vld1.32 {d20-d21}, [%[doutc1r0]]           @ load dout \n"
                        "vbif.16 q12, q14, q11                     @bif \n"
                        "vbif.16 q13, q14, q11                     @bif \n"
                        // "add %[din_ptr0], #7                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "vmull.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmull.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "sub %[doutc0r0], #16          @ sub \n"
                        "sub %[doutc1r0], #16          @ sub \n"

                        "vbif.16 q12, q14, q11                     @bif \n"
                        "vbif.16 q13, q14, q11                     @bif \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d14-d15}, [%[doutc1r0]]!         @ store\n"
                        "vst1.32 {d18-d19}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d20-d21}, [%[doutc1r0]]!         @ store\n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [doutc0r0] "+r"(doutc0r0), [doutc1r0] "+r"(doutc1r0)
                    :[mask] "r" (vmask), [rs_mask] "r" (rmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"

                    );

                    doutc0_ptr += wout;
                    doutc1_ptr += wout;
                }

            }

        } //end of processing output channel
        for (int c = ch; c < chout; ++c) {

            int* dout_c = dout_batch + c * size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c, &bias[c], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c, zero, 1, size_out_channel);
            }
            const signed char* wc0 = weights + c * w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *wei_ptr = wc0 + i * 9;
                const signed char *din_ch = din_batch + i * size_in_channel;
                int *doutc0_ptr = dout_c;

                int *doutc0r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }

                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    asm volatile(
                    //left
                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"
                        "vdup.s8     d4, d0[0]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d5, d0[1]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d6, d0[2]               @ d4 = w02, w02, w02, w02\n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"
                        "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vmov.u32 d11, #0                   @ zero\n"

                        "vdup.s8     d7, d0[3]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d8, d0[4]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d0[5]               @ d4 = w02, w02, w02, w02\n"

                        "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                        "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678

                        //r0
                        "vmull.s8 q12, d12, d5                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                        "vmull.s8 q13, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmull.s8 q14, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d18, d0[6]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d19, d0[7]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d20, d1[0]               @ d4 = w02, w02, w02, w02\n"

                        "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d14, d15, #1          @ ext \n" //d11 = 12345678

                        //r1
                        "vmlal.s8 q12, d14, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q14, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vext.8     d30, d11, d16, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d16, d17, #1          @ ext \n" //d11 = 12345678
                        "vld1.16 {d22-d23}, [%[rs_mask]]     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vmov.u16 q5, #0                   @ zero\n"

                        "pld [%[doutc0r0]]                @ preload data\n"


                        //r2
                        "vmlal.s8 q12, d16, d19                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d18                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q14, d31, d20                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"
                        "vld1.32 {d14-d15}, [%[doutc0r0]]           @ load dout \n"

                        "vbif.16 q12, q5, q11                     @bif \n"
                        "vbif.16 q13, q5, q11                     @bif \n"
                        "vbif.16 q14, q5, q11                     @bif \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d25                 @addw \n" // out1 += vget_low_s16(out10)

                        "sub %[doutc0r0], #16          @ sub \n"

                        "vaddw.s16 q6, q6, d26                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d27                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q6, q6, d28                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d29                 @addw \n" // out1 += vget_low_s16(out10)

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d14-d15}, [%[doutc0r0]]!         @ store\n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [doutc0r0] "+r"(doutc0r0)
                    :[mask] "r" (vmask), [rs_mask] "r" (rmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"
                    );
                    doutc0_ptr += wout;
                }

            }
        } // end of remain out channel

    } // end of processing batchs
}

void conv_3x3s1_direct_bias_relu_int8(int* dout, const signed char* din, \
                          const signed char* weights, const int* bias, bool flag_bias, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, Context* ctx){
    //printf("conv3x3_dw start \n");
    int threads = ctx->get_threads();
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const short unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, win * sizeof(signed char));

    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    int w_stride = chin * 9;

    int tile_w = (win + 7) >> 3;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(win - 7 - (cnt_col << 3));

    // uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    // uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));

    unsigned int rst_remain = (unsigned int)(wout - ((cnt_col + 1) << 3));
    uint16x8_t vmask_result = vcgtq_u16(vdupq_n_u16(rst_remain), vld1q_u16(right_pad_rst));

    int8x8_t vzero = vdup_n_s8(0);
    int16x8_t vzero_16 = vdupq_n_s16(0);
    int zero[4] = {0, 0, 0, 0};

    unsigned char vmask[16];
    vst1q_u8(vmask, vmask_rp);

    short unsigned int rmask[8];
    vst1q_u16(rmask, vmask_result);

    int ch = ((chout >> 1) << 1);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * chin * size_in_channel;
        int *dout_batch = dout + n * chout * size_out_channel;
#pragma omp parallel for num_threads(threads)
        for (int c = 0; c < chout - 1; c += 2) {

            int* dout_c0 = dout_batch + c * size_out_channel;
            int* dout_c1 = dout_c0 + size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c0, &bias[c], 1, size_out_channel);
                fill_bias_int8(dout_c1, &bias[c + 1], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c0, zero, 1, size_out_channel);
                fill_bias_int8(dout_c1, zero, 1, size_out_channel);
            }

            const signed char* wc0 = weights + c * w_stride;
            const signed char* wc1 = wc0 + w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *din_ch = din_batch + i * size_in_channel;
                const signed char *wc0_ptr = wc0 + i * 9;
                const signed char *wc1_ptr = wc1 + i * 9;
                int *doutc0_ptr = dout_c0;
                int *doutc1_ptr = dout_c1;

                int *doutc0r0 = nullptr;
                int *doutc1r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wc0]]    \n"
                    "vld1.8    {d2-d3}, [%[wc1]]    \n"
                    :
                    :[wc0] "r" (wc0_ptr), [wc1] "r" (wc1_ptr)
                    : "memory"
                );

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    doutc1r0 = doutc1_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }
                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    int cnt = cnt_col;
                    asm volatile(
                    //left
                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"
                        "vdup.s8     d5, d0[1]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[1]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d4, d0[0]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[0]               @ d2 = w00, w00, w00, w00\n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"
                        "vmov.u32 d11, #0                   @ zero\n"
                        "vdup.s8     d6, d0[2]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[2]               @ d4 = w02, w02, w02, w02\n"

                        "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678

                        //r0
                        "vmull.s8 q12, d12, d5                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                        "vmull.s8 q13, d12, d8                 @ out1 = din0 * w01 \n" // q12 = d12 * w01

                        "vdup.s8     d5, d0[4]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[4]               @ d3 = w01, w01, w01, w01\n"

                        "vmlal.s8 q12, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vdup.s8     d4, d0[3]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[3]               @ d2 = w00, w00, w00, w00\n"
                        "pld [%[doutc0r0]]                @ preload data\n"
                        "pld [%[doutc1r0]]                @ preload data\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d0[5]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[5]               @ d4 = w02, w02, w02, w02\n"

                        //r1
                        "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d14, d15, #1          @ ext \n" //d11 = 12345678

                        "vmlal.s8 q12, d14, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d14, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01

                        "vdup.s8     d5, d0[7]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[7]               @ d3 = w01, w01, w01, w01\n"

                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"
                        "vld1.32 {d14-d15}, [%[doutc1r0]]!           @ load dout \n"

                        "vmlal.s8 q12, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vdup.s8     d4, d0[6]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[6]               @ d2 = w00, w00, w00, w00\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d1[0]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d3[0]               @ d4 = w02, w02, w02, w02\n"

                        //r2
                        "vext.8     d30, d11, d16, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d16, d17, #1          @ ext \n" //d11 = 12345678

                        "vmlal.s8 q12, d16, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d16, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01

                        "vld1.32 {d18-d19}, [%[doutc0r0]]           @ load dout \n"
                        "vld1.32 {d20-d21}, [%[doutc1r0]]           @ load dout \n"

                        "vmlal.s8 q12, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "sub %[doutc0r0], #16          @ sub \n"
                        "sub %[doutc1r0], #16          @ sub \n"
                        "add %[din_ptr0], #7                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "vmull.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmull.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "add %[din_ptr1], #7                   @add \n"
                        "add %[din_ptr2], #7                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d14-d15}, [%[doutc1r0]]!         @ store\n"
                        "cmp %[cnt], #1                                 \n"
                        "vst1.32 {d18-d19}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d20-d21}, [%[doutc1r0]]!         @ store\n"
                        "blt 1f                                         \n"
                    //mid
                        "2:                                             \n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"

                        "vdup.s8     d4, d0[0]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[0]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d5, d0[1]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[1]               @ d3 = w01, w01, w01, w01\n"

                        "vext.8     d30, d12, d13, #1         @ ext \n" //d30 = 12345678
                        "vext.8     d31, d12, d13, #2          @ ext \n" //31 = 23456789

                        "vdup.s8     d6, d0[2]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[2]               @ d4 = w02, w02, w02, w02\n"

                        //r0
                        "vmull.s8 q12, d12, d4                 @ out0 = din0 * w00 \n" // q12 = d12 * w00
                        "vmull.s8 q13, d12, d7                 @ out1 = din0 * w00 \n" // q12 = d12 * w00

                        "vdup.s8     d4, d0[3]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[3]               @ d2 = w00, w00, w00, w00\n"

                        "vmlal.s8 q12, d30, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01

                        "vdup.s8     d5, d0[4]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[4]               @ d3 = w01, w01, w01, w01\n"
                        "pld [%[doutc0r0]]                @ preload data\n"
                        "pld [%[doutc1r0]]                @ preload data\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d0[5]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[5]               @ d4 = w02, w02, w02, w02\n"

                        //r1
                        "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 23456789

                        "vmlal.s8 q12, d14, d4                 @ out0 += din0 * w00 \n"
                        "vmlal.s8 q13, d14, d7                 @ out0 += din0 * w00 \n"

                        "vdup.s8     d4, d0[6]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[6]               @ d2 = w00, w00, w00, w00\n"

                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"
                        "vld1.32 {d14-d15}, [%[doutc1r0]]!           @ load dout \n"

                        "vmlal.s8 q12, d30, d5                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vdup.s8     d5, d0[7]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[7]               @ d3 = w01, w01, w01, w01\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d1[0]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d3[0]               @ d4 = w02, w02, w02, w02\n"

                        //r2
                        "vext.8     d30, d16, d17, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d16, d17, #2          @ ext \n" //d11 = 23456789

                        "vmlal.s8 q12, d16, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d16, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w01

                        "vld1.32 {d18-d19}, [%[doutc0r0]]           @ load dout \n"
                        "vld1.32 {d20-d21}, [%[doutc1r0]]           @ load dout \n"

                        "vmlal.s8 q12, d30, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w00

                        "sub %[doutc0r0], #16          @ sub \n"
                        "sub %[doutc1r0], #16          @ sub \n"
                        "add %[din_ptr0], #8                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "vmull.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmull.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "add %[din_ptr1], #8                   @add \n"
                        "add %[din_ptr2], #8                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d14-d15}, [%[doutc1r0]]!         @ store\n"
                        "subs %[cnt], #1                                \n"
                        "vst1.32 {d18-d19}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d20-d21}, [%[doutc1r0]]!         @ store\n"
                        "bne  2b                                        \n"

                    //right
                        "1:                                             \n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"
                        "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                        "vdup.s8     d4, d0[0]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[0]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d5, d0[1]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[1]               @ d3 = w01, w01, w01, w01\n"

                        "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                        "vdup.s8     d6, d0[2]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[2]               @ d4 = w02, w02, w02, w02\n"

                        "vext.8     d30, d12, d13, #1         @ ext \n" //d30 = 12345678
                        "vext.8     d31, d12, d13, #2          @ ext \n" //31 = 23456789

                        //r0
                        "vmull.s8 q12, d12, d4                 @ out0 = din0 * w00 \n" // q12 = d12 * w00
                        "vmull.s8 q13, d12, d7                 @ out1 = din0 * w00 \n" // q12 = d12 * w00

                        "vdup.s8     d4, d0[3]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[3]               @ d2 = w00, w00, w00, w00\n"

                        "vmlal.s8 q12, d30, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01

                        "vdup.s8     d5, d0[4]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[4]               @ d3 = w01, w01, w01, w01\n"
                        "pld [%[doutc0r0]]                @ preload data\n"
                        "pld [%[doutc1r0]]                @ preload data\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d0[5]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[5]               @ d4 = w02, w02, w02, w02\n"

                        //r1
                        "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 23456789

                        "vmlal.s8 q12, d14, d4                 @ out0 += din0 * w00 \n"
                        "vmlal.s8 q13, d14, d7                 @ out0 += din0 * w00 \n"

                        "vdup.s8     d4, d0[6]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[6]               @ d2 = w00, w00, w00, w00\n"

                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"
                        "vld1.32 {d14-d15}, [%[doutc1r0]]!           @ load dout \n"

                        "vmlal.s8 q12, d30, d5                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vdup.s8     d5, d0[7]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[7]               @ d3 = w01, w01, w01, w01\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d1[0]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d3[0]               @ d4 = w02, w02, w02, w02\n"

                        //r2
                        "vext.8     d30, d16, d17, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d16, d17, #2          @ ext \n" //d11 = 23456789

                        "vmlal.s8 q12, d16, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d16, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w01

                        "vld1.16 {d22-d23}, [%[rs_mask]]     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vmov.u16 q14, #0                   @ zero\n"

                        "vmlal.s8 q12, d30, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w00

                        "vld1.32 {d18-d19}, [%[doutc0r0]]           @ load dout \n"
                        "vld1.32 {d20-d21}, [%[doutc1r0]]           @ load dout \n"

                        "vbif.16 q12, q14, q11                     @bif \n"
                        "vbif.16 q13, q14, q11                     @bif \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "vmull.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmull.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "sub %[doutc0r0], #16          @ sub \n"
                        "sub %[doutc1r0], #16          @ sub \n"

                        "vbif.16 q12, q14, q11                     @bif \n"
                        "vbif.16 q13, q14, q11                     @bif \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d14-d15}, [%[doutc1r0]]!         @ store\n"
                        "vst1.32 {d18-d19}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d20-d21}, [%[doutc1r0]]!         @ store\n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [doutc0r0] "+r"(doutc0r0), [doutc1r0] "+r"(doutc1r0), [cnt] "+r" (cnt)
                    :[mask] "r" (vmask), [rs_mask] "r" (rmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"

                    );

                    doutc0_ptr += wout;
                    doutc1_ptr += wout;
                }

            }
            //relu
            int *doutc0r0 = dout_c0;
            int *doutc1r0 = dout_c1;
            int cnt = size_out_channel / 8;
            int i = cnt;
            if (cnt > 0){
                asm volatile(
                    "vld1.32 {d0-d3}, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "vld1.32 {d4-d7}, [%[doutc1r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "vmov.u32 q4, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "vmax.s32   q0, q0, q4  \n"         /*relu*/
                    "vmax.s32   q2, q2, q4 \n"         /*relu*/
                    "vmax.s32   q1, q1, q4  \n"         /*relu*/
                    "vmax.s32   q3, q3, q4  \n"         /*relu*/
                    "pld [%[doutc0r0]]      \n"
                    "pld [%[doutc1r0]]      \n"
                    "vst1.32    {d0-d1}, [%[doutc0r0]]! \n"         /* store c0r0*/
                    "vst1.32    {d4-d5}, [%[doutc1r0]]! \n"         /* store c1r0*/
                    "subs %[cnt], #1                \n"         /* loop count -1*/
                    "vst1.32    {d2-d3}, [%[doutc0r0]]! \n"         /* store c0r0*/
                    "vst1.32    {d6-d7}, [%[doutc1r0]]! \n"         /* store c1r0*/
                    "vld1.32 {d0-d3}, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "vld1.32 {d4-d7}, [%[doutc1r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "bne    1b                      \n"         /* jump to main loop*/
                :[doutc0r0]"+r"(doutc0r0), [doutc1r0]"+r"(doutc1r0), [cnt] "+r"(cnt)
                :
                : "q0", "q1", "q2", "q3", "q4"
                );
            }
            i = i * 8;
            for (; i < size_out_channel; i++){
                *doutc0r0 = AKMAX(*doutc0r0, 0);
                *doutc1r0 = AKMAX(*doutc1r0, 0);
                doutc0r0++;
                doutc1r0++;
            }

        } //end of processing output channel
        for (int c = ch; c < chout; ++c) {

            int* dout_c = dout_batch + c * size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c, &bias[c], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c, zero, 1, size_out_channel);
            }
            const signed char* wc0 = weights + c * w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *wei_ptr = wc0 + i * 9;
                const signed char *din_ch = din_batch + i * size_in_channel;
                int *doutc0_ptr = dout_c;

                int *doutc0r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }

                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }

                    int cnt = cnt_col;
                    asm volatile(
                    //left
                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"
                        "vdup.s8     d4, d0[0]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d5, d0[1]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d6, d0[2]               @ d4 = w02, w02, w02, w02\n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"
                        "vmov.u32 d11, #0                   @ zero\n"
                        "vdup.s8     d7, d0[3]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d8, d0[4]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d0[5]               @ d4 = w02, w02, w02, w02\n"

                        "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678

                        //r0
                        "vmull.s8 q12, d12, d5                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                        "vmull.s8 q13, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmull.s8 q14, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d18, d0[6]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d19, d0[7]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d20, d1[0]               @ d4 = w02, w02, w02, w02\n"

                        "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d14, d15, #1          @ ext \n" //d11 = 12345678

                        //r1
                        "vmlal.s8 q12, d14, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q14, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vext.8     d30, d11, d16, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d16, d17, #1          @ ext \n" //d11 = 12345678
                        "pld [%[doutc0r0]]                @ preload data\n"

                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"

                        //r2
                        "vmlal.s8 q12, d16, d19                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d18                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q14, d31, d20                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vld1.32 {d14-d15}, [%[doutc0r0]]           @ load dout \n"

                        "add %[din_ptr0], #7                   @add \n"
                        "add %[din_ptr1], #7                   @add \n"
                        "add %[din_ptr2], #7                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d25                 @addw \n" // out1 += vget_low_s16(out10)

                        "sub %[doutc0r0], #16          @ sub \n"

                        "vaddw.s16 q6, q6, d26                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d27                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q6, q6, d28                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d29                 @addw \n" // out1 += vget_low_s16(out10)

                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "cmp %[cnt], #1                                 \n"
                        "vst1.32 {d14-d15}, [%[doutc0r0]]!         @ store\n"
                        "blt 1f                                         \n"
                    //mid
                        "2:                                             \n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"

                        "pld [%[doutc0r0]]                @ preload data\n"

                        "vext.8     d30, d12, d13, #1         @ ext \n" //d30 = 12345678
                        "vext.8     d31, d12, d13, #2          @ ext \n" //31 = 23456789

                        //r0
                        "vmull.s8 q12, d12, d4                 @ out0 = din0 * w00 \n" // q12 = d12 * w00
                        "vmull.s8 q13, d30, d5                 @ out1 = din0 * w01 \n" // q12 = d12 * w00
                        "vmull.s8 q14, d31, d6                 @ out1 = din0 * w02 \n" // q12 = d12 * w00\

                        //r1
                        "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 23456789
                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"

                        "vmlal.s8 q12, d14, d7                 @ out0 += din0 * w00 \n"
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w01 \n"
                        "vmlal.s8 q14, d31, d9                 @ out0 += din0 * w02 \n"


                        //r2
                        "vext.8     d30, d16, d17, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d16, d17, #2          @ ext \n" //d11 = 23456789
                        "vld1.32 {d14-d15}, [%[doutc0r0]]           @ load dout \n"
                        "sub %[doutc0r0], #16          @ sub \n"

                        "vmlal.s8 q12, d16, d18                 @ out0 += din0 * w00 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d19                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q14, d31, d20                 @ out0 += din0 * w02 \n" // q12 += d10 * w01

                        "add %[din_ptr0], #8                   @add \n"
                        "add %[din_ptr1], #8                   @add \n"
                        "add %[din_ptr2], #8                   @add \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d25                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q6, q6, d26                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d27                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q6, q6, d28                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d29                 @addw \n" // out1 += vget_low_s16(out10)

                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "subs %[cnt], #1                                \n"
                        "vst1.32 {d14-d15}, [%[doutc0r0]]!         @ store\n"
                        "bne  2b                                        \n"

                    //right
                        "1:                                             \n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"
                        "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                        "pld [%[doutc0r0]]                @ preload data\n"

                        "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                        "vext.8     d30, d12, d13, #1         @ ext \n" //d30 = 12345678
                        "vext.8     d31, d12, d13, #2          @ ext \n" //31 = 23456789

                        "vld1.16 {d22-d23}, [%[rs_mask]]     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vmov.u16 q5, #0                   @ zero\n"

                        //r0
                        "vmull.s8 q12, d12, d4                 @ out0 = din0 * w00 \n" // q12 = d12 * w00
                        "vmull.s8 q13, d30, d5                 @ out1 = din0 * w01 \n" // q12 = d12 * w00
                        "vmull.s8 q14, d31, d6                 @ out1 = din0 * w02 \n" // q12 = d12 * w00\

                        //r1
                        "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 23456789
                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"

                        "vmlal.s8 q12, d14, d7                 @ out0 += din0 * w00 \n"
                        "vmlal.s8 q13, d30, d8                 @ out0 += din0 * w01 \n"
                        "vmlal.s8 q14, d31, d9                 @ out0 += din0 * w02 \n"

                        //r2
                        "vext.8     d30, d16, d17, #1     @ ext \n" //d10 = 12345678
                        "vext.8     d31, d16, d17, #2          @ ext \n" //d11 = 23456789
                        "vld1.32 {d14-d15}, [%[doutc0r0]]           @ load dout \n"
                        "sub %[doutc0r0], #16          @ sub \n"

                        "vmlal.s8 q12, d16, d18                 @ out0 += din0 * w00 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d19                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q14, d31, d20                 @ out0 += din0 * w02 \n" // q12 += d10 * w01

                        "vbif.16 q12, q5, q11                     @bif \n"
                        "vbif.16 q13, q5, q11                     @bif \n"
                        "vbif.16 q14, q5, q11                     @bif \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d25                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q6, q6, d26                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d27                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q6, q6, d28                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d29                 @addw \n" // out1 += vget_low_s16(out10)

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d14-d15}, [%[doutc0r0]]!         @ store\n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [doutc0r0] "+r"(doutc0r0), [cnt] "+r" (cnt)
                    :[mask] "r" (vmask), [rs_mask] "r" (rmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"
                    );
                    doutc0_ptr += wout;
                }

            }
            //relu
            int *doutc0r0 = dout_c;
            int cnt = size_out_channel / 8;
            int i = cnt;
            if (cnt > 0){
                asm volatile(
                    "vld1.32 {d0-d3}, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "vmov.u32 q2, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "vmax.s32   q0, q0, q2 \n"         /*relu*/
                    "vmax.s32   q1, q1, q2  \n"         /*relu*/
                    "subs %[cnt], #1                \n"         /* loop count -1*/
                    "vst1.32    {d0-d1}, [%[doutc0r0]]! \n"         /* store c0r0*/
                    "vst1.32    {d2-d3}, [%[doutc0r0]]! \n"         /* store c0r0*/
                    "pld [%[doutc0r0]]      \n"
                    "vld1.32 {d0-d3}, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "bne    1b                      \n"         /* jump to main loop*/
                :[doutc0r0]"+r"(doutc0r0), [cnt] "+r"(cnt)
                :
                : "q0", "q1", "q2"
                );
            }
            i = i * 8;
            for (; i < size_out_channel; i++){
                *doutc0r0 = AKMAX(*doutc0r0, 0);
                doutc0r0++;
            }
        } // end of remain out channel

    } // end of processing batchs
}

void conv_3x3s1_direct_bias_relu_s_int8(int* dout, const signed char* din, \
                          const signed char* weights, const int* bias, bool flag_bias, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          int group, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, Context* ctx){
    //printf("conv3x3_dw start \n");
    int threads = ctx->get_threads();
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const short unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, win * sizeof(signed char));

    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    int w_stride = chin * 9;

    unsigned int size_pad_right = (unsigned int)(win);

    // uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    // uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    uint8x8_t vmask_rp = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));

    unsigned int rst_remain = (unsigned int)(wout);
    uint16x8_t vmask_result = vcgtq_u16(vdupq_n_u16(rst_remain), vld1q_u16(right_pad_rst));

    int8x8_t vzero = vdup_n_s8(0);
    int16x8_t vzero_16 = vdupq_n_s16(0);
    int zero[4] = {0, 0, 0, 0};

    unsigned char vmask[8];
    vst1_u8(vmask, vmask_rp);

    short unsigned int rmask[8];
    vst1q_u16(rmask, vmask_result);

    int ch = ((chout >> 1) << 1);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * chin * size_in_channel;
        int *dout_batch = dout + n * chout * size_out_channel;
#pragma omp parallel for num_threads(threads)
        for (int c = 0; c < chout - 1; c += 2) {

            int* dout_c0 = dout_batch + c * size_out_channel;
            int* dout_c1 = dout_c0 + size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c0, &bias[c], 1, size_out_channel);
                fill_bias_int8(dout_c1, &bias[c + 1], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c0, zero, 1, size_out_channel);
                fill_bias_int8(dout_c1, zero, 1, size_out_channel);
            }

            const signed char* wc0 = weights + c * w_stride;
            const signed char* wc1 = wc0 + w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *din_ch = din_batch + i * size_in_channel;
                const signed char *wc0_ptr = wc0 + i * 9;
                const signed char *wc1_ptr = wc1 + i * 9;
                int *doutc0_ptr = dout_c0;
                int *doutc1_ptr = dout_c1;

                int *doutc0r0 = nullptr;
                int *doutc1r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wc0]]    \n"
                    "vld1.8    {d2-d3}, [%[wc1]]    \n"
                    :
                    :[wc0] "r" (wc0_ptr), [wc1] "r" (wc1_ptr)
                    : "memory"
                );

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    doutc1r0 = doutc1_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }
                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    asm volatile(
                    //left
                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"
                        "vdup.s8     d5, d0[1]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[1]               @ d3 = w01, w01, w01, w01\n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"
                        "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                        "vdup.s8     d4, d0[0]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[0]               @ d2 = w00, w00, w00, w00\n"
                        "vmov.u32 d11, #0                   @ zero\n"

                        "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                        "vdup.s8     d6, d0[2]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[2]               @ d4 = w02, w02, w02, w02\n"

                        "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678

                        //r0
                        "vmull.s8 q12, d12, d5                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                        "vmull.s8 q13, d12, d8                 @ out1 = din0 * w01 \n" // q12 = d12 * w01

                        "vdup.s8     d5, d0[4]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[4]               @ d3 = w01, w01, w01, w01\n"

                        "vmlal.s8 q12, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vdup.s8     d4, d0[3]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[3]               @ d2 = w00, w00, w00, w00\n"
                        "pld [%[doutc0r0]]                @ preload data\n"
                        "pld [%[doutc1r0]]                @ preload data\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d0[5]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d2[5]               @ d4 = w02, w02, w02, w02\n"

                        //r1
                        "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d14, d15, #1          @ ext \n" //d11 = 12345678

                        "vmlal.s8 q12, d14, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d14, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01

                        "vdup.s8     d5, d0[7]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d8, d2[7]               @ d3 = w01, w01, w01, w01\n"

                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"
                        "vld1.32 {d14-d15}, [%[doutc1r0]]!           @ load dout \n"

                        "vmlal.s8 q12, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vdup.s8     d4, d0[6]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d7, d2[6]               @ d2 = w00, w00, w00, w00\n"

                        "vmlal.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmlal.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d6, d1[0]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d3[0]               @ d4 = w02, w02, w02, w02\n"

                        //r2
                        "vext.8     d30, d11, d16, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d16, d17, #1          @ ext \n" //d11 = 12345678

                        "vmlal.s8 q12, d16, d5                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d16, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01

                        "vld1.16 {d22-d23}, [%[rs_mask]]     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vmov.u16 q14, #0                   @ zero\n"

                        "vmlal.s8 q12, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                        "vld1.32 {d18-d19}, [%[doutc0r0]]           @ load dout \n"
                        "vld1.32 {d20-d21}, [%[doutc1r0]]           @ load dout \n"
                        // "add %[din_ptr0], #7                   @add \n"
                        "vbif.16 q12, q14, q11                     @bif \n"
                        "vbif.16 q13, q14, q11                     @bif \n"
                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "vmull.s8 q12, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02
                        "vmull.s8 q13, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "sub %[doutc0r0], #16          @ sub \n"
                        "sub %[doutc1r0], #16          @ sub \n"

                        "vbif.16 q12, q14, q11                     @bif \n"
                        "vbif.16 q13, q14, q11                     @bif \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d26                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q9, q9, d25                 @addw \n" // out1_1 += vget_high_s16(out10)
                        "vaddw.s16 q10, q10, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d14-d15}, [%[doutc1r0]]!         @ store\n"
                        "vst1.32 {d18-d19}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d20-d21}, [%[doutc1r0]]!         @ store\n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [doutc0r0] "+r"(doutc0r0), [doutc1r0] "+r"(doutc1r0)
                    :[mask] "r" (vmask), [rs_mask] "r" (rmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"

                    );

                    doutc0_ptr += wout;
                    doutc1_ptr += wout;
                }

            }
            //relu
            int *doutc0r0 = dout_c0;
            int *doutc1r0 = dout_c1;
            int cnt = size_out_channel / 8;
            int i = cnt;
            if (cnt > 0){
                asm volatile(
                    "vld1.32 {d0-d3}, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "vld1.32 {d4-d7}, [%[doutc1r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "vmov.u32 q4, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "vmax.s32   q0, q0, q4  \n"         /*relu*/
                    "vmax.s32   q2, q2, q4 \n"         /*relu*/
                    "vmax.s32   q1, q1, q4  \n"         /*relu*/
                    "vmax.s32   q3, q3, q4  \n"         /*relu*/
                    "pld [%[doutc0r0]]      \n"
                    "pld [%[doutc1r0]]      \n"
                    "vst1.32    {d0-d1}, [%[doutc0r0]]! \n"         /* store c0r0*/
                    "vst1.32    {d4-d5}, [%[doutc1r0]]! \n"         /* store c1r0*/
                    "subs %[cnt], #1                \n"         /* loop count -1*/
                    "vst1.32    {d2-d3}, [%[doutc0r0]]! \n"         /* store c0r0*/
                    "vst1.32    {d6-d7}, [%[doutc1r0]]! \n"         /* store c1r0*/
                    "vld1.32 {d0-d3}, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "vld1.32 {d4-d7}, [%[doutc1r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "bne    1b                      \n"         /* jump to main loop*/
                :[doutc0r0]"+r"(doutc0r0), [doutc1r0]"+r"(doutc1r0), [cnt] "+r"(cnt)
                :
                : "q0", "q1", "q2", "q3", "q4"
                );
            }
            i = i * 8;
            for (; i < size_out_channel; i++){
                *doutc0r0 = AKMAX(*doutc0r0, 0);
                *doutc1r0 = AKMAX(*doutc1r0, 0);
                doutc0r0++;
                doutc1r0++;
            }

        } //end of processing output channel
        for (int c = ch; c < chout; ++c) {

            int* dout_c = dout_batch + c * size_out_channel;

            if (flag_bias) {
                fill_bias_int8(dout_c, &bias[c], 1, size_out_channel);
            } else {
                fill_bias_int8(dout_c, zero, 1, size_out_channel);
            }
            const signed char* wc0 = weights + c * w_stride;

            for (int i = 0; i < chin; ++i) {
                const signed char *wei_ptr = wc0 + i * 9;
                const signed char *din_ch = din_batch + i * size_in_channel;
                int *doutc0_ptr = dout_c;

                int *doutc0r0 = nullptr;

                const signed char *dr0 = din_ch;
                const signed char *dr1 = dr0 + win;
                const signed char *dr2 = dr1 + win;

                const signed char *din_ptr0 = nullptr;
                const signed char *din_ptr1 = nullptr;
                const signed char *din_ptr2 = nullptr;

                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );

                for (int h = 0; h < hin; h++){
                    din_ptr0 = dr0;
                    din_ptr1 = dr1;
                    din_ptr2 = dr2;

                    doutc0r0 = doutc0_ptr;
                    if (h == 0){
                        din_ptr0 = zero_ptr;
                        din_ptr1 = dr0;
                        din_ptr2 = dr1;
                    }else{
                        dr0 = dr1;
                        dr1 = dr2;
                        dr2 = dr1 + win;
                    }

                    //! process bottom pad
                    if (h + 2 > hin) {
                        din_ptr2 = zero_ptr;
                    }
                    asm volatile(
                    //left
                        "pld [%[din_ptr0]]                @ preload data\n"
                        "pld [%[din_ptr1]]                @ preload data\n"
                        "pld [%[din_ptr2]]                @ preload data\n"
                        "vdup.s8     d4, d0[0]               @ d3 = w01, w01, w01, w01\n"
                        "vdup.s8     d5, d0[1]               @ d2 = w00, w00, w00, w00\n"
                        "vdup.s8     d6, d0[2]               @ d4 = w02, w02, w02, w02\n"
                        "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load data \n"
                        "vld1.8 {d16-d17}, [%[din_ptr2]]    @ load data \n"
                        "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vmov.u32 d11, #0                   @ zero\n"

                        "vdup.s8     d7, d0[3]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d8, d0[4]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d9, d0[5]               @ d4 = w02, w02, w02, w02\n"

                        "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                        "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                        "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                        "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678

                        //r0
                        "vmull.s8 q12, d12, d5                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                        "vmull.s8 q13, d30, d4                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmull.s8 q14, d31, d6                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vdup.s8     d18, d0[6]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d19, d0[7]               @ d4 = w02, w02, w02, w02\n"
                        "vdup.s8     d20, d1[0]               @ d4 = w02, w02, w02, w02\n"

                        "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d14, d15, #1          @ ext \n" //d11 = 12345678

                        //r1
                        "vmlal.s8 q12, d14, d8                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d7                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q14, d31, d9                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        "vext.8     d30, d11, d16, #7     @ ext \n" //d10 = 00123456
                        "vext.8     d31, d16, d17, #1          @ ext \n" //d11 = 12345678
                        "vld1.16 {d22-d23}, [%[rs_mask]]     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                        "vmov.u16 q5, #0                   @ zero\n"

                        "pld [%[doutc0r0]]                @ preload data\n"


                        //r2
                        "vmlal.s8 q12, d16, d19                 @ out0 += din0 * w01 \n" // q12 += d10 * w01
                        "vmlal.s8 q13, d30, d18                 @ out0 += din0 * w00 \n" // q12 += d10 * w00
                        "vmlal.s8 q14, d31, d20                 @ out0 += din0 * w02 \n" // q12 += d10 * w02

                        //out0
                        "vld1.32 {d12-d13}, [%[doutc0r0]]!           @ load dout \n"
                        "vld1.32 {d14-d15}, [%[doutc0r0]]           @ load dout \n"

                        "vbif.16 q12, q5, q11                     @bif \n"
                        "vbif.16 q13, q5, q11                     @bif \n"
                        "vbif.16 q14, q5, q11                     @bif \n"

                        "vaddw.s16 q6, q6, d24                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d25                 @addw \n" // out1 += vget_low_s16(out10)

                        "sub %[doutc0r0], #16          @ sub \n"

                        "vaddw.s16 q6, q6, d26                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d27                 @addw \n" // out1 += vget_low_s16(out10)

                        "vaddw.s16 q6, q6, d28                 @addw \n" // out1 += vget_low_s16(out10)
                        "vaddw.s16 q7, q7, d29                 @addw \n" // out1 += vget_low_s16(out10)

                        "vst1.32 {d12-d13}, [%[doutc0r0]]!         @ store\n"
                        "vst1.32 {d14-d15}, [%[doutc0r0]]!         @ store\n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [doutc0r0] "+r"(doutc0r0)
                    :[mask] "r" (vmask), [rs_mask] "r" (rmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"
                    );
                    doutc0_ptr += wout;
                }

            }
            //relu
            int *doutc0r0 = dout_c;
            int cnt = size_out_channel / 8;
            int i = cnt;
            if (cnt > 0){
                asm volatile(
                    "vld1.32 {d0-d3}, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "vmov.u32 q2, #0                \n"         /* for relu */
                    "1:                             \n"         /* main loop*/
                    "vmax.s32   q0, q0, q2 \n"         /*relu*/
                    "vmax.s32   q1, q1, q2  \n"         /*relu*/
                    "subs %[cnt], #1                \n"         /* loop count -1*/
                    "vst1.32    {d0-d1}, [%[doutc0r0]]! \n"         /* store c0r0*/
                    "vst1.32    {d2-d3}, [%[doutc0r0]]! \n"         /* store c0r0*/
                    "pld [%[doutc0r0]]      \n"
                    "vld1.32 {d0-d3}, [%[doutc0r0]]      \n"         /* load r00, r01 to q0, q1 */
                    "bne    1b                      \n"         /* jump to main loop*/
                :[doutc0r0]"+r"(doutc0r0), [cnt] "+r"(cnt)
                :
                : "q0", "q1", "q2"
                );
            }
            i = i * 8;
            for (; i < size_out_channel; i++){
                *doutc0r0 = AKMAX(*doutc0r0, 0);
                doutc0r0++;
            }
        } // end of remain out channel

    } // end of processing batchs
}
#endif //__aarch64__

} //namespace lite

} //namespace saber

} //namespace anakin

#endif
