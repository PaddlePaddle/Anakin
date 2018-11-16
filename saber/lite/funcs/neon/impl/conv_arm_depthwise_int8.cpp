 #include "saber/lite/funcs/neon/impl/conv_arm_impl.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

void conv_depthwise_3x3s1p1_bias_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx);

//! for input width <= 8
void conv_depthwise_3x3s1p1_bias_s_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx);

void conv_depthwise_3x3s2p1_bias_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx);

//! for input width <= 8
void conv_depthwise_3x3s2p1_bias_s_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx);

void conv_depthwise_3x3s1p1_bias_relu_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx);

//! for input width <= 4
void conv_depthwise_3x3s1p1_bias_s_relu_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx);

void conv_depthwise_3x3s2p1_bias_relu_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx);

//! for input width <= 4
void conv_depthwise_3x3s2p1_bias_s_relu_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx);

void conv_depthwise_3x3_int8(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, \
                          Context* ctx, void* work_space, const void* idx_ptr) {

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
            if (w_in > 8) {
                conv_depthwise_3x3s1p1_bias_relu_int8(static_cast<int*>(dout), static_cast<const signed char*>(din), static_cast<const signed char*>(weights), \
                    static_cast<const int*>(bias), flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            } else {
                conv_depthwise_3x3s1p1_bias_s_relu_int8(static_cast<int*>(dout), static_cast<const signed char*>(din), static_cast<const signed char*>(weights), \
                    static_cast<const int*>(bias), flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }
        } else {
            if (w_in > 8) {
                conv_depthwise_3x3s1p1_bias_int8(static_cast<int*>(dout), static_cast<const signed char*>(din), static_cast<const signed char*>(weights), \
                    static_cast<const int*>(bias), flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }else {
                conv_depthwise_3x3s1p1_bias_s_int8(static_cast<int*>(dout), static_cast<const signed char*>(din), static_cast<const signed char*>(weights), \
                    static_cast<const int*>(bias), flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }

        }
    }
    else { //! stride = 2
        if (flag_relu) {
            if (w_in > 16){
                conv_depthwise_3x3s2p1_bias_relu_int8(static_cast<int*>(dout), static_cast<const signed char*>(din), static_cast<const signed char*>(weights), \
                    static_cast<const int*>(bias), flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }else{
                conv_depthwise_3x3s2p1_bias_s_relu_int8(static_cast<int*>(dout), static_cast<const signed char*>(din), static_cast<const signed char*>(weights), \
                    static_cast<const int*>(bias), flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }
        } else {
            if (w_in > 16){
                conv_depthwise_3x3s2p1_bias_int8(static_cast<int*>(dout), static_cast<const signed char*>(din), static_cast<const signed char*>(weights), \
                    static_cast<const int*>(bias), flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }else{
                conv_depthwise_3x3s2p1_bias_s_int8(static_cast<int*>(dout), static_cast<const signed char*>(din), static_cast<const signed char*>(weights), \
                    static_cast<const int*>(bias), flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out, ctx);
            }

        }
    }
}
/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias, width > 4
 */
#ifdef __aarch64__
template <typename Dtype>
inline void prefetch(const Dtype *din) {
    asm volatile(
    "PRFM PLDL1KEEP, [%[din]] \n"
    :
    : [din] "r"(din)
    : "memory");
}

//4line w_in > 8
void conv_depthwise_3x3s1p1_bias_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {

   // printf("3x3s1 mult height \n");
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 7) >> 3;
    int tile_h = (h_out + 3) >> 2;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(w_in - 7 - (cnt_col << 3));

    int size_pad_bottom = h_out % 4;

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);

    // printf("vmask_rp1: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3], \
    //     vmask_rp1[4], vmask_rp1[5], vmask_rp1[6], vmask_rp1[7]);
    // printf("vmask_rp2: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3], \
    //     vmask_rp2[4], vmask_rp2[5], vmask_rp2[6], vmask_rp2[7]);
    // printf("vmask_result: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_result1[0], vmask_result1[1], vmask_result1[2], vmask_result1[3], \
    //     vmask_result2[0], vmask_result2[1], vmask_result2[2], vmask_result2[3]);

    // printf("cnt_col: %d, rst_remain: %d, size_pad_right: %d, size_pad_bottom: %d \n", cnt_col, rst_remain, size_pad_right, size_pad_bottom);
     for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

            int *doutr0 = nullptr;
            int *doutr1 = nullptr;
            int *doutr2 = nullptr;
            int *doutr3 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;
            const signed char *dr3 = dr2 + w_in;
            const signed char *dr4 = dr3 + w_in;
            const signed char *dr5 = dr4 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;
            const signed char *din_ptr3 = nullptr;
            const signed char *din_ptr4 = nullptr;
            const signed char *din_ptr5 = nullptr;

            for (int i = 0; i < h_in; i += 4){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;
                din_ptr4 = dr4;
                din_ptr5 = dr5;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                doutr2 = doutr1 + w_out;
                doutr3 = doutr2 + w_out;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    din_ptr4 = dr3;
                    din_ptr5 = dr4;
                    dr0 = dr3;
                    dr1 = dr4;
                    dr2 = dr5;
                }else{
                    dr0 = dr4;
                    dr1 = dr5;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 5 > h_in) {
                    switch (i + 5 - h_in) {
                        case 5:
                            din_ptr1 = zero_ptr;
                        case 4:
                            din_ptr2 = zero_ptr;
                        case 3:
                            din_ptr3 = zero_ptr;
                        case 2:
                            din_ptr4 = zero_ptr;
                        case 1:
                            din_ptr5 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 4 > h_out) {
                    switch (i + 4 - h_out) {
                        case 3:
                            doutr1 = write_ptr;
                        case 2:
                            doutr2 = write_ptr;
                        case 1:
                            doutr3 = write_ptr;
                        default:
                            break;
                    }
                }
                //prefetch input
                prefetch(doutr0);
                prefetch(doutr1);
                prefetch(doutr2);
                prefetch(doutr3);

                prefetch(din_ptr0);
                prefetch(din_ptr1);
                prefetch(din_ptr2);
                prefetch(din_ptr3);
                prefetch(din_ptr4);
                prefetch(din_ptr5);

                //left
                //din data
                int8x8_t vinr0 = vld1_s8(din_ptr0);//01234567
                int8x8_t vinr1 = vld1_s8(din_ptr1);

                // printf("vinr1: %d, %d, %d, %d, %d, %d, %d, %d \n", vinr1[0], vinr1[1], vinr1[2], vinr1[3], vinr1[4], vinr1[5], vinr1[6], vinr1[7]);

                int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//01234567
                int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);

                int32x4_t voutr0 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                int32x4_t voutr0_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);

                int32x4_t voutr1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr1);
                int32x4_t voutr1_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr1);

                int32x4_t voutr2 = vdupq_n_s32(bias_val); //vld1q_f32(doutr2);
                int32x4_t voutr2_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr2);

                int32x4_t voutr3 = vdupq_n_s32(bias_val); //vld1q_f32(doutr3);
                int32x4_t voutr3_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr3);

                //r0 1234
                int16x8_t voutr00 = vmull_s8(vinr0, wr01);//01234567 * w11

                int8x8_t vtmp0 = vext_s8(vzero, vinr0, 7);//00123456
                int8x8_t vtmp1 = vext_s8(vinr0, vinr0_1, 1);//12345678

                int8x8_t vinr2 = vld1_s8(din_ptr2);
                int8x8_t vinr3 = vld1_s8(din_ptr3);//01234567

                voutr00 = vmlal_s8(voutr00, vtmp0, wr00);//r0 * w01

                int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);
                int8x8_t vinr3_1 = vld1_s8(din_ptr3 + 8);//01234567

                int8x8_t vinr4 = vld1_s8(din_ptr4);
                int8x8_t vinr5 = vld1_s8(din_ptr5);

                voutr00 = vmlal_s8(voutr00, vtmp1, wr02);//r0 * w01

                // printf("voutr0: %d, %d, %d, %d \n", voutr0[0], voutr0[1], voutr0[2], voutr0[3]);
                //r1
                din_ptr0 += 7;
                din_ptr1 += 7;
                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                int16x8_t voutr10 = vmull_s8(vinr1, wr01);//r0 * w01
                voutr00 = vmull_s8(vinr1, wr11);//r0 * w01

                vtmp0 = vext_s8(vzero, vinr1, 7);//00123456
                vtmp1 = vext_s8(vinr1, vinr1_1, 1);//12345678

                int8x8_t vinr4_1 = vld1_s8(din_ptr4 + 8);
                int8x8_t vinr5_1 = vld1_s8(din_ptr5 + 8);

                voutr10 = vmlal_s8(voutr10, vtmp0, wr00);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp0, wr10);//r0 * w01

                din_ptr2 += 7;
                din_ptr3 += 7;
                voutr10 = vmlal_s8(voutr10, vtmp1, wr02);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp1, wr12);//r0 * w01

                //r2
                din_ptr4 += 7;
                int16x8_t voutr20 = vmull_s8(vinr2, wr01);//r0 * w01
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));
                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                voutr10 = vmull_s8(vinr2, wr11);//r0 * w01
                voutr00 = vmull_s8(vinr2, wr21);//r0 * w01

                vtmp0 = vext_s8(vzero, vinr2, 7);//00123456
                vtmp1 = vext_s8(vinr2, vinr2_1, 1);//12345678

                voutr20 = vmlal_s8(voutr20, vtmp0, wr00);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp0, wr10);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp0, wr20);//r0 * w01

                din_ptr5 += 7;
                voutr20 = vmlal_s8(voutr20, vtmp1, wr02);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp1, wr12);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp1, wr22);//r0 * w01

                //r3
                vtmp0 = vext_s8(vzero, vinr3, 7);//00123456
                vtmp1 = vext_s8(vinr3, vinr3_1, 1);//12345678
                int16x8_t voutr30 = vmull_s8(vinr3, wr01);//r0 * w01

                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                voutr20 = vmull_s8(vinr3, wr11);//r0 * w01
                voutr10 = vmull_s8(vinr3, wr21);//r0 * w01

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr00);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp0, wr10);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp0, wr20);//r0 * w01

                voutr30 = vmlal_s8(voutr30, vtmp1, wr02);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp1, wr12);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp1, wr22);//r0 * w01

                //r4
                vtmp0 = vext_s8(vzero, vinr4, 7);//00123456
                vtmp1 = vext_s8(vinr4, vinr4_1, 1);//12345678
                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                voutr30 = vmull_s8(vinr4, wr11);//r0 * w01
                voutr20 = vmull_s8(vinr4, wr21);//r0 * w01

                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr10);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp0, wr20);//r0 * w01

                vst1q_s32(doutr0, voutr0);
                vst1q_s32(doutr0 + 4, voutr0_1);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr12);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp1, wr22);//r0 * w01

                doutr0 += 8;

                //r5
                vtmp0 = vext_s8(vzero, vinr5, 7);//00123456
                vtmp1 = vext_s8(vinr5, vinr5_1, 1);//12345678

                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));
                voutr30 = vmull_s8(vinr5, wr21);//r0 * w01

                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr20);//r0 * w01

                vst1q_s32(doutr1, voutr1);
                vst1q_s32(doutr1 + 4, voutr1_1);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr22);//r0 * w01

                doutr1 += 8;
                vst1q_s32(doutr2, voutr2);
                doutr2 += 4;

                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));

                vst1q_s32(doutr2, voutr2_1);

                vst1q_s32(doutr3, voutr3);
                doutr3 += 4;
                doutr2 += 4;

                vst1q_s32(doutr3, voutr3_1);
                doutr3 += 4;

                //mid col
                for (int j = 0; j < cnt_col; ++j) {
                    //din data
                    vinr0 = vld1_s8(din_ptr0);//01234567
                    vinr1 = vld1_s8(din_ptr1);

                    vinr0_1 = vld1_s8(din_ptr0 + 8);//01234567
                    vinr1_1 = vld1_s8(din_ptr1 + 8);

                    voutr0 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                    voutr0_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                    voutr1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr1);
                    voutr1_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                    voutr2 = vdupq_n_s32(bias_val); //vld1q_f32(doutr2);
                    voutr2_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                    voutr3 = vdupq_n_s32(bias_val); //vld1q_f32(doutr3);
                    voutr3_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);

                    //r0 1234
                    voutr00 = vmull_s8(vinr0, wr00);//01234567 * w10

                    vtmp0 = vext_s8(vinr0, vinr0_1, 1);//12345678
                    vtmp1 = vext_s8(vinr0, vinr0_1, 2);//23456789

                    vinr2 = vld1_s8(din_ptr2);
                    vinr3 = vld1_s8(din_ptr3);//01234567

                    voutr00 = vmlal_s8(voutr00, vtmp0, wr01);//r0 * w01

                    vinr2_1 = vld1_s8(din_ptr2 + 8);
                    vinr3_1 = vld1_s8(din_ptr3 + 8);//01234567

                    din_ptr0 += 8;
                    voutr00 = vmlal_s8(voutr00, vtmp1, wr02);//r0 * w01

                    //r1
                    vinr4 = vld1_s8(din_ptr4);
                    vinr5 = vld1_s8(din_ptr5);
                    voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                    voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                    voutr10 = vmull_s8(vinr1, wr00);//r0 * w01
                    voutr00 = vmull_s8(vinr1, wr10);//r0 * w01

                    vtmp0 = vext_s8(vinr1, vinr1_1, 1);//00123456
                    vtmp1 = vext_s8(vinr1, vinr1_1, 2);//12345678
                    vinr4_1 = vld1_s8(din_ptr4 + 8);
                    vinr5_1 = vld1_s8(din_ptr5 + 8);

                    voutr10 = vmlal_s8(voutr10, vtmp0, wr01);//r0 * w01
                    voutr00 = vmlal_s8(voutr00, vtmp0, wr11);//r0 * w01

                    din_ptr1 += 8;
                    voutr10 = vmlal_s8(voutr10, vtmp1, wr02);//r0 * w01
                    voutr00 = vmlal_s8(voutr00, vtmp1, wr12);//r0 * w01

                    //r2
                    vtmp0 = vext_s8(vinr2, vinr2_1, 1);//00123456
                    vtmp1 = vext_s8(vinr2, vinr2_1, 2);//12345678
                    voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                    voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));
                    voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                    voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                    voutr20 = vmull_s8(vinr2, wr00);//r0 * w01
                    voutr10 = vmull_s8(vinr2, wr10);//r0 * w01
                    voutr00 = vmull_s8(vinr2, wr20);//r0 * w01

                    din_ptr2 += 8;
                    din_ptr3 += 8;

                    voutr20 = vmlal_s8(voutr20, vtmp0, wr01);//r0 * w01
                    voutr10 = vmlal_s8(voutr10, vtmp0, wr11);//r0 * w01
                    voutr00 = vmlal_s8(voutr00, vtmp0, wr21);//r0 * w01

                    din_ptr4 += 8;
                    voutr20 = vmlal_s8(voutr20, vtmp1, wr02);//r0 * w01
                    voutr10 = vmlal_s8(voutr10, vtmp1, wr12);//r0 * w01
                    voutr00 = vmlal_s8(voutr00, vtmp1, wr22);//r0 * w01

                    //r3
                    vtmp0 = vext_s8(vinr3, vinr3_1, 1);//00123456
                    vtmp1 = vext_s8(vinr3, vinr3_1, 2);//12345678
                    voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                    voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));
                    voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                    voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                    voutr30 = vmull_s8(vinr3, wr00);//r0 * w01
                    voutr20 = vmull_s8(vinr3, wr10);//r0 * w01
                    voutr10 = vmull_s8(vinr3, wr20);//r0 * w01

                    voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                    voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                    voutr30 = vmlal_s8(voutr30, vtmp0, wr01);//r0 * w01
                    voutr20 = vmlal_s8(voutr20, vtmp0, wr11);//r0 * w01
                    voutr10 = vmlal_s8(voutr10, vtmp0, wr21);//r0 * w01

                    din_ptr5 += 8;
                    voutr30 = vmlal_s8(voutr30, vtmp1, wr02);//r0 * w01
                    voutr20 = vmlal_s8(voutr20, vtmp1, wr12);//r0 * w01
                    voutr10 = vmlal_s8(voutr10, vtmp1, wr22);//r0 * w01

                    //r4
                    vtmp0 = vext_s8(vinr4, vinr4_1, 1);//00123456
                    vtmp1 = vext_s8(vinr4, vinr4_1, 2);//12345678
                    voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                    voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));
                    voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                    voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));
                    voutr30 = vmull_s8(vinr4, wr10);//r0 * w01
                    voutr20 = vmull_s8(vinr4, wr20);//r0 * w01

                    voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                    voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                    voutr30 = vmlal_s8(voutr30, vtmp0, wr11);//r0 * w01
                    voutr20 = vmlal_s8(voutr20, vtmp0, wr21);//r0 * w01

                    vst1q_s32(doutr0, voutr0);
                    vst1q_s32(doutr0 + 4, voutr0_1);

                    voutr30 = vmlal_s8(voutr30, vtmp1, wr12);//r0 * w01
                    voutr20 = vmlal_s8(voutr20, vtmp1, wr22);//r0 * w01

                    doutr0 += 8;

                    //r5
                    vtmp0 = vext_s8(vinr5, vinr5_1, 1);//00123456
                    vtmp1 = vext_s8(vinr5, vinr5_1, 2);//12345678
                    voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                    voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));
                    voutr30 = vmull_s8(vinr5, wr20);//r0 * w01

                    voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                    voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                    voutr30 = vmlal_s8(voutr30, vtmp0, wr21);//r0 * w01

                    vst1q_s32(doutr1, voutr1);
                    vst1q_s32(doutr1 + 4, voutr1_1);

                    voutr30 = vmlal_s8(voutr30, vtmp1, wr22);//r0 * w01

                    doutr1 += 8;
                    vst1q_s32(doutr2, voutr2);
                    doutr2 += 4;

                    voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                    voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));

                    vst1q_s32(doutr2, voutr2_1);
                    doutr2 += 4;

                    vst1q_s32(doutr3, voutr3);
                    vst1q_s32(doutr3 + 4, voutr3_1);
                    doutr3 += 8;

                }
                //right
                //din data
                vinr0 = vld1_s8(din_ptr0);//01234567
                vinr1 = vld1_s8(din_ptr1);
                // printf("vinr0: %d, %d, %d, %d, %d, %d, %d, %d \n", vinr0[0], vinr0[1], vinr0[2], vinr0[3], vinr0[4], vinr0[5], vinr0[6], vinr0[7]);

                vinr0_1 = vld1_s8(din_ptr0 + 8);//01234567
                vinr1_1 = vld1_s8(din_ptr1 + 8);

                voutr0 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                voutr0_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                voutr1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr1);
                voutr1_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                voutr2 = vdupq_n_s32(bias_val); //vld1q_f32(doutr2);
                voutr2_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                voutr3 = vdupq_n_s32(bias_val); //vld1q_f32(doutr3);
                voutr3_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);

                vinr0 = vbsl_s8(vmask_rp1, vinr0, vzero);
                vinr1 = vbsl_s8(vmask_rp1, vinr1, vzero);
                // printf("vinr0: %d, %d, %d, %d, %d, %d, %d, %d \n", vinr0[0], vinr0[1], vinr0[2], vinr0[3], vinr0[4], vinr0[5], vinr0[6], vinr0[7]);

                vinr0_1 = vbsl_s8(vmask_rp2, vinr0_1, vzero);
                vinr1_1 = vbsl_s8(vmask_rp2, vinr1_1, vzero);

                // printf("vinr0_1: %d, %d, %d, %d, %d, %d, %d, %d \n", vinr0_1[0], vinr0_1[1], vinr0_1[2], vinr0_1[3], vinr0_1[4], vinr0_1[5], vinr0_1[6], vinr0_1[7]);

                //r0 1234
                voutr00 = vmull_s8(vinr0, wr00);//01234567 * w11

                vtmp0 = vext_s8(vinr0, vinr0_1, 1);//12345678
                vtmp1 = vext_s8(vinr0, vinr0_1, 2);//23456789
                vinr2 = vld1_s8(din_ptr2);
                vinr3 = vld1_s8(din_ptr3);//01234567

                voutr00 = vmlal_s8(voutr00, vtmp0, wr01);//r0 * w01
                vinr2_1 = vld1_s8(din_ptr2 + 8);
                vinr3_1 = vld1_s8(din_ptr3 + 8);//01234567
                int32x4_t voutr0_32 = vld1q_s32(doutr0);
                int32x4_t voutr1_32 = vld1q_s32(doutr1);

                voutr00 = vmlal_s8(voutr00, vtmp1, wr02);//r0 * w01

                //r1
                vinr2 = vbsl_s8(vmask_rp1, vinr2, vzero);
                vinr3 = vbsl_s8(vmask_rp1, vinr3, vzero);
                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                voutr10 = vmull_s8(vinr1, wr00);//r0 * w01
                voutr00 = vmull_s8(vinr1, wr10);//r0 * w01

                vtmp0 = vext_s8(vinr1, vinr1_1, 1);//00123456
                vtmp1 = vext_s8(vinr1, vinr1_1, 2);//12345678

                vinr2_1 = vbsl_s8(vmask_rp2, vinr2_1, vzero);
                vinr3_1 = vbsl_s8(vmask_rp2, vinr3_1, vzero);

                voutr10 = vmlal_s8(voutr10, vtmp0, wr01);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp0, wr11);//r0 * w01

                int32x4_t voutr2_32 = vld1q_s32(doutr2);
                int32x4_t voutr3_32 = vld1q_s32(doutr3);

                voutr10 = vmlal_s8(voutr10, vtmp1, wr02);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp1, wr12);//r0 * w01

                //r2
                vinr4 = vld1_s8(din_ptr4);
                vinr5 = vld1_s8(din_ptr5);
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));
                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                voutr20 = vmull_s8(vinr2, wr00);//r0 * w01
                voutr10 = vmull_s8(vinr2, wr10);//r0 * w01
                voutr00 = vmull_s8(vinr2, wr20);//r0 * w01

                vtmp0 = vext_s8(vinr2, vinr2_1, 1);//00123456
                vtmp1 = vext_s8(vinr2, vinr2_1, 2);//12345678

                vinr4_1 = vld1_s8(din_ptr4 + 8);
                vinr5_1 = vld1_s8(din_ptr5 + 8);

                voutr20 = vmlal_s8(voutr20, vtmp0, wr01);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp0, wr11);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp0, wr21);//r0 * w01

                int32x4_t voutr0_32_1 = vld1q_s32(doutr0 + 4);
                int32x4_t voutr1_32_1 = vld1q_s32(doutr1 + 4);

                voutr20 = vmlal_s8(voutr20, vtmp1, wr02);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp1, wr12);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp1, wr22);//r0 * w01

                //r3
                vtmp0 = vext_s8(vinr3, vinr3_1, 1);//00123456
                vtmp1 = vext_s8(vinr3, vinr3_1, 2);//12345678
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                voutr30 = vmull_s8(vinr3, wr00);//r0 * w01
                voutr20 = vmull_s8(vinr3, wr10);
                voutr10 = vmull_s8(vinr3, wr20);

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                vinr4 = vbsl_s8(vmask_rp1, vinr4, vzero);
                vinr5 = vbsl_s8(vmask_rp1, vinr5, vzero);

                // uint16x8_t vm_res = vmovl_u8(vmask_result);
                int32x4_t voutr2_32_1 = vld1q_s32(doutr2 + 4);
                int32x4_t voutr3_32_1 = vld1q_s32(doutr3 + 4);

                voutr30 = vmlal_s8(voutr30, vtmp0, wr01);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp0, wr11);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp0, wr21);//r0 * w01

                vinr4_1 = vbsl_s8(vmask_rp2, vinr4_1, vzero);
                vinr5_1 = vbsl_s8(vmask_rp2, vinr5_1, vzero);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr02);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp1, wr12);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp1, wr22);//r0 * w01

                //r4
                vtmp0 = vext_s8(vinr4, vinr4_1, 1);//00123456
                vtmp1 = vext_s8(vinr4, vinr4_1, 2);//12345678
                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                voutr30 = vmull_s8(vinr4, wr10);//r0 * w01
                voutr20 = vmull_s8(vinr4, wr20);//r0 * w01

                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr11);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp0, wr21);//r0 * w01

                voutr0 = vbslq_s32(vmask_result1, voutr0, voutr0_32);
                voutr0_1 = vbslq_s32(vmask_result2, voutr0_1, voutr0_32_1);
                // printf("voutr1_0: %d, %d, %d, %d \n", voutr1_0[0], voutr1_0[1], voutr1_0[2], voutr1_0[3]);
                // printf("voutr1_1: %d, %d, %d, %d \n", voutr1_1[0], voutr1_1[1], voutr1_1[2], voutr1_1[3]);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr12);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp1, wr22);//r0 * w01

                voutr1 = vbslq_s32(vmask_result1, voutr1, voutr1_32);
                voutr1_1 = vbslq_s32(vmask_result2, voutr1_1, voutr1_32_1);

                vst1q_s32(doutr0, voutr0);
                vst1q_s32(doutr0 + 4, voutr0_1);

                //r5
                vtmp0 = vext_s8(vinr5, vinr5_1, 1);//00123456
                vtmp1 = vext_s8(vinr5, vinr5_1, 2);//12345678
                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));

                voutr30 = vmull_s8(vinr5, wr20);//r0 * w01

                doutr0 += 8;
                dr3 = dr2 + w_in;
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr21);//r0 * w01

                dr4 = dr3 + w_in;
                dr5 = dr4 + w_in;
                vst1q_s32(doutr1, voutr1);
                vst1q_s32(doutr1 + 4, voutr1_1);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr22);//r0 * w01

                doutr1 += 8;

                voutr2 = vbslq_s32(vmask_result1, voutr2, voutr2_32);
                voutr2_1 = vbslq_s32(vmask_result2, voutr2_1, voutr2_32_1);

                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));

                dout_ptr = dout_ptr + 4 * w_out;
                voutr3 = vbslq_s32(vmask_result1, voutr3, voutr3_32);
                voutr3_1 = vbslq_s32(vmask_result2, voutr3_1, voutr3_32_1);

                vst1q_s32(doutr2, voutr2);
                vst1q_s32(doutr2 + 4, voutr2_1);
                doutr2 += 8;

                vst1q_s32(doutr3, voutr3);
                vst1q_s32(doutr3 + 4, voutr3_1);
                doutr3 += 8;
            }
        }
    }
}
//w_in <= 8
void conv_depthwise_3x3s1p1_bias_s_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {

   // printf("3x3s1 mult height \n");
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_h = (h_out + 3) >> 2;

    unsigned int size_pad_right = (unsigned int)(w_in);

    int size_pad_bottom = h_out % 4;

    uint8x8_t vmask_rp = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    unsigned int rst_remain = (unsigned int)w_out;
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

            int *doutr0 = nullptr;
            int *doutr1 = nullptr;
            int *doutr2 = nullptr;
            int *doutr3 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;
            const signed char *dr3 = dr2 + w_in;
            const signed char *dr4 = dr3 + w_in;
            const signed char *dr5 = dr4 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;
            const signed char *din_ptr3 = nullptr;
            const signed char *din_ptr4 = nullptr;
            const signed char *din_ptr5 = nullptr;

            for (int i = 0; i < h_in; i += 4){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;
                din_ptr4 = dr4;
                din_ptr5 = dr5;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                doutr2 = doutr1 + w_out;
                doutr3 = doutr2 + w_out;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    din_ptr4 = dr3;
                    din_ptr5 = dr4;
                    dr0 = dr3;
                    dr1 = dr4;
                    dr2 = dr5;
                }else{
                    dr0 = dr4;
                    dr1 = dr5;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 5 > h_in) {
                    switch (i + 5 - h_in) {
                        case 5:
                            din_ptr1 = zero_ptr;
                        case 4:
                            din_ptr2 = zero_ptr;
                        case 3:
                            din_ptr3 = zero_ptr;
                        case 2:
                            din_ptr4 = zero_ptr;
                        case 1:
                            din_ptr5 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 4 > h_out) {
                    switch (i + 4 - h_out) {
                        case 3:
                            doutr1 = write_ptr;
                        case 2:
                            doutr2 = write_ptr;
                        case 1:
                            doutr3 = write_ptr;
                        default:
                            break;
                    }
                }
                //prefetch input
                prefetch(doutr0);
                prefetch(doutr1);
                prefetch(doutr2);
                prefetch(doutr3);

                prefetch(din_ptr0);
                prefetch(din_ptr1);
                prefetch(din_ptr2);
                prefetch(din_ptr3);
                prefetch(din_ptr4);
                prefetch(din_ptr5);

                //din data
                int8x8_t vinr0 = vld1_s8(din_ptr0);//01234567
                int8x8_t vinr1 = vld1_s8(din_ptr1);

                int32x4_t voutr0 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                int32x4_t voutr0_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                int32x4_t voutr1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr1);
                int32x4_t voutr1_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                int32x4_t voutr2 = vdupq_n_s32(bias_val); //vld1q_f32(doutr2);
                int32x4_t voutr2_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                int32x4_t voutr3 = vdupq_n_s32(bias_val); //vld1q_f32(doutr3);
                int32x4_t voutr3_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);

                vinr0 = vbsl_s8(vmask_rp, vinr0, vzero);
                vinr1 = vbsl_s8(vmask_rp, vinr1, vzero);

                //r0 1234
                int16x8_t voutr00 = vmull_s8(vinr0, wr01);//01234567 * w01

                int8x8_t vtmp0 = vext_s8(vzero, vinr0, 7);//001234567
                int8x8_t vtmp1 = vext_s8(vinr0, vzero, 1);//12345670
                int8x8_t vinr2 = vld1_s8(din_ptr2);
                int8x8_t vinr3 = vld1_s8(din_ptr3);//01234567

                voutr00 = vmlal_s8(voutr00, vtmp0, wr00);//r0 * w01

                int32x4_t voutr0_32 = vld1q_s32(doutr0);
                int32x4_t voutr1_32 = vld1q_s32(doutr1);

                voutr00 = vmlal_s8(voutr00, vtmp1, wr02);//r0 * w01

                //r1
                vtmp0 = vext_s8(vzero, vinr1, 7);//00123456
                vtmp1 = vext_s8(vinr1, vzero, 1);//12345678
                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                int16x8_t voutr10 = vmull_s8(vinr1, wr01);//r0 * w01
                voutr00 = vmull_s8(vinr1, wr11);//r0 * w01

                vinr2 = vbsl_s8(vmask_rp, vinr2, vzero);
                vinr3 = vbsl_s8(vmask_rp, vinr3, vzero);
                int8x8_t vinr4 = vld1_s8(din_ptr4);
                int8x8_t vinr5 = vld1_s8(din_ptr5);

                voutr10 = vmlal_s8(voutr10, vtmp0, wr00);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp0, wr10);//r0 * w01

                int32x4_t voutr2_32 = vld1q_s32(doutr2);
                int32x4_t voutr3_32 = vld1q_s32(doutr3);

                voutr10 = vmlal_s8(voutr10, vtmp1, wr02);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp1, wr12);//r0 * w01

                //r2
                vtmp0 = vext_s8(vzero, vinr2, 7);//00123456
                vtmp1 = vext_s8(vinr2, vzero, 1);//12345678
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));
                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                int16x8_t voutr20 = vmull_s8(vinr2, wr01);//r0 * w01
                voutr10 = vmull_s8(vinr2, wr11);//r0 * w01
                voutr00 = vmull_s8(vinr2, wr21);//r0 * w01

                vinr4 = vbsl_s8(vmask_rp, vinr4, vzero);
                vinr5 = vbsl_s8(vmask_rp, vinr5, vzero);

                voutr20 = vmlal_s8(voutr20, vtmp0, wr00);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp0, wr10);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp0, wr20);//r0 * w01

                int32x4_t voutr0_32_1 = vld1q_s32(doutr0 + 4);
                int32x4_t voutr1_32_1 = vld1q_s32(doutr1 + 4);

                voutr20 = vmlal_s8(voutr20, vtmp1, wr02);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp1, wr12);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp1, wr22);//r0 * w01

                //r3
                vtmp0 = vext_s8(vzero, vinr3, 7);//00123456
                vtmp1 = vext_s8(vinr3, vzero, 1);//12345678
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                int16x8_t voutr30 = vmull_s8(vinr3, wr01);//r0 * w01
                voutr20 = vmull_s8(vinr3, wr11);
                voutr10 = vmull_s8(vinr3, wr21);

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                // uint16x8_t vm_res = vmovl_u8(vmask_result);
                int32x4_t voutr2_32_1 = vld1q_s32(doutr2 + 4);
                int32x4_t voutr3_32_1 = vld1q_s32(doutr3 + 4);

                voutr30 = vmlal_s8(voutr30, vtmp0, wr00);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp0, wr10);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp0, wr20);//r0 * w01

                voutr30 = vmlal_s8(voutr30, vtmp1, wr02);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp1, wr12);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp1, wr22);//r0 * w01

                //r4
                vtmp0 = vext_s8(vzero, vinr4, 7);//00123456
                vtmp1 = vext_s8(vinr4, vzero, 1);//12345678
                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                voutr30 = vmull_s8(vinr4, wr11);//r0 * w01
                voutr20 = vmull_s8(vinr4, wr21);//r0 * w01

                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr10);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp0, wr20);//r0 * w01

                voutr0 = vbslq_s32(vmask_result1, voutr0, voutr0_32);
                voutr0_1 = vbslq_s32(vmask_result2, voutr0_1, voutr0_32_1);
                // printf("voutr1_0: %d, %d, %d, %d \n", voutr1_0[0], voutr1_0[1], voutr1_0[2], voutr1_0[3]);
                // printf("voutr1_1: %d, %d, %d, %d \n", voutr1_1[0], voutr1_1[1], voutr1_1[2], voutr1_1[3]);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr12);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp1, wr22);//r0 * w01

                voutr1 = vbslq_s32(vmask_result1, voutr1, voutr1_32);
                voutr1_1 = vbslq_s32(vmask_result2, voutr1_1, voutr1_32_1);

                vst1q_s32(doutr0, voutr0);
                vst1q_s32(doutr0 + 4, voutr0_1);

                //r5
                vtmp0 = vext_s8(vzero, vinr5, 7);//00123456
                vtmp1 = vext_s8(vinr5, vzero, 1);//12345678
                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));

                voutr30 = vmull_s8(vinr5, wr21);//r0 * w01

                doutr0 += 8;
                dr3 = dr2 + w_in;
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr20);//r0 * w01

                dr4 = dr3 + w_in;
                dr5 = dr4 + w_in;
                vst1q_s32(doutr1, voutr1);
                vst1q_s32(doutr1 + 4, voutr1_1);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr22);//r0 * w01

                doutr1 += 8;

                voutr2 = vbslq_s32(vmask_result1, voutr2, voutr2_32);
                voutr2_1 = vbslq_s32(vmask_result2, voutr2_1, voutr2_32_1);

                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));

                dout_ptr = dout_ptr + 4 * w_out;
                vst1q_s32(doutr2, voutr2);
                vst1q_s32(doutr2 + 4, voutr2_1);

                voutr3 = vbslq_s32(vmask_result1, voutr3, voutr3_32);
                voutr3_1 = vbslq_s32(vmask_result2, voutr3_1, voutr3_32_1);
                doutr2 += 8;

                vst1q_s32(doutr3, voutr3);
                vst1q_s32(doutr3 + 4, voutr3_1);
                doutr3 += 8;
            }
        }
    }
}

//4line w_in > 16
void conv_depthwise_3x3s2p1_bias_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {

   // printf("3x3s2 mult height \n");
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_out;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 15) >> 4;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(w_in - 15 - (cnt_col << 4));
    if (size_pad_right == 17){
        size_pad_right = 0;
        cnt_col++;
    }

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    int8x8_t vzero = vdup_n_s8(0);
    // int32x4_t vzero_32 = vdupq_n_s32(0);

    // printf("vmask_rp1: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3], \
    //     vmask_rp1[4], vmask_rp1[5], vmask_rp1[6], vmask_rp1[7]);
    // printf("vmask_rp2: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3], \
    //     vmask_rp2[4], vmask_rp2[5], vmask_rp2[6], vmask_rp2[7]);
    // printf("vmask_result: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_result1[0], vmask_result1[1], vmask_result1[2], vmask_result1[3], \
    //     vmask_result2[0], vmask_result2[1], vmask_result2[2], vmask_result2[3]);

    // printf("cnt_col: %d, rst_remain: %d, size_pad_right: %d\n", cnt_col, rst_remain, size_pad_right);
     for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

            int *doutr0 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;

                doutr0 = dout_ptr;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr1 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr0 + w_in;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 2 > h_in) {
                    switch (i + 2 - h_in) {
                        case 2:
                            din_ptr1 = zero_ptr;
                        case 1:
                            din_ptr2 = zero_ptr;
                        default:
                            break;
                    }
                }
                //prefetch input
                prefetch(doutr0);

                prefetch(din_ptr0);
                prefetch(din_ptr1);
                prefetch(din_ptr2);

                //left
                int8x8x2_t vinr0 = vld2_s8(din_ptr0); //a = 0 2 4 6 8 10 12 14 b = 1 3 5 7 9 11 13 15
                int8x8x2_t vinr1 = vld2_s8(din_ptr1);
                int8x8x2_t vinr2 = vld2_s8(din_ptr2);

                int32x4_t voutr0 = vdupq_n_s32(bias_val);
                int32x4_t voutr0_1 = vdupq_n_s32(bias_val);

                int8x8_t tmp0 = vext_s8(vzero, vinr0.val[1], 7); //c = -1 1 3 5 7 9 11 13
                int8x8_t tmp1 = vext_s8(vzero, vinr1.val[1], 7); //c = -1 1 3 5 7 9 11 13
                int8x8_t tmp2 = vext_s8(vzero, vinr2.val[1], 7); //c = -1 1 3 5 7 9 11 13

                //r0
                int16x8_t voutr01 = vmull_s8(vinr0.val[0], wr01);//a = 0 2 4 6 8 10 12 14
                int16x8_t voutr02 = vmull_s8(vinr0.val[1], wr02);// b = 1 3 5 7 9 11 13 15
                int16x8_t voutr00 = vmull_s8(tmp0, wr00); //c = -1 1 3 5 7 9 11 13

                //r1
                voutr01 = vmlal_s8(voutr01, vinr1.val[0], wr11);//a = 0 2 4 6 8 10 12 14
                voutr02 = vmlal_s8(voutr02, vinr1.val[1], wr12);// b = 1 3 5 7 9 11 13 15
                voutr00 = vmlal_s8(voutr00, tmp1, wr10); //c = -1 1 3 5 7 9 11 13

                //r2
                voutr01 = vmlal_s8(voutr01, vinr2.val[0], wr21);//a = 0 2 4 6 8 10 12 14
                voutr02 = vmlal_s8(voutr02, vinr2.val[1], wr22);// b = 1 3 5 7 9 11 13 15
                voutr00 = vmlal_s8(voutr00, tmp2, wr20); //c = -1 1 3 5 7 9 11 13

                din_ptr0 += 15;
                din_ptr1 += 15;
                din_ptr2 += 15;

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr01));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr01));

                prefetch(din_ptr0);
                prefetch(din_ptr1);
                prefetch(din_ptr2);

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr02));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr02));

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                vst1q_s32(doutr0, voutr0);
                vst1q_s32(doutr0 + 4, voutr0_1);
                doutr0 += 8;

                //mid
                for (int i =0; i < cnt_col; i++){
                    vinr0 = vld2_s8(din_ptr0);//a = 0 2 4 6 8 10 12 14  b = 1 3 5 7 9 11 13 15
                    vinr1 = vld2_s8(din_ptr1);
                    vinr2 = vld2_s8(din_ptr2);

                    int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 16); // d = 16 17 18 19
                    int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 16);
                    int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 16);

                    voutr0 = vdupq_n_s32(bias_val); //
                    voutr0_1 = vdupq_n_s32(bias_val);

                    tmp0 = vext_s8(vinr0.val[0], vinr0_1, 1); // c = 2 4 6 8 10 12 14 16
                    tmp1 = vext_s8(vinr1.val[0], vinr1_1, 1); // c = 2 4 6 8 10 12 14 16
                    tmp2 = vext_s8(vinr2.val[0], vinr2_1, 1); // c = 2 4 6 8 10 12 14 16

                    //r0
                    voutr00 = vmull_s8(vinr0.val[0], wr00);//a = 0 2 4 6 8 10 12 14
                    voutr01 = vmull_s8(vinr0.val[1], wr01);//b = 1 3 5 7 9 11 13 15
                    voutr02 = vmull_s8(tmp0, wr02);//c = 2 4 6 8 10 12 14 16

                    //r1
                    voutr00 = vmlal_s8(voutr00, vinr1.val[0], wr10);//a = 0 2 4 6 8 10 12 14
                    voutr01 = vmlal_s8(voutr01, vinr1.val[1], wr11);//b = 1 3 5 7 9 11 13 15
                    voutr02 = vmlal_s8(voutr02, tmp1, wr12);//c = 2 4 6 8 10 12 14 16

                    //r2
                    voutr00 = vmlal_s8(voutr00, vinr2.val[0], wr20);//a = 0 2 4 6 8 10 12 14
                    voutr01 = vmlal_s8(voutr01, vinr2.val[1], wr21);//b = 1 3 5 7 9 11 13 15
                    voutr02 = vmlal_s8(voutr02, tmp2, wr22);//c = 2 4 6 8 10 12 14 16

                    din_ptr0 += 16;
                    din_ptr1 += 16;
                    din_ptr2 += 16;

                    voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                    voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                    prefetch(din_ptr0);
                    prefetch(din_ptr1);
                    prefetch(din_ptr2);

                    voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr01));
                    voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr01));

                    voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr02));
                    voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr02));

                    vst1q_s32(doutr0, voutr0);
                    vst1q_s32(doutr0 + 4, voutr0_1);

                    doutr0 += 8;

                }
                //right
                if (size_pad_right == 17){
                    continue;
                }
                prefetch(doutr0);
                vinr0 = vld2_s8(din_ptr0);//a = 0 2 4 6 8 10 12 14 b = 1 3 5 7 9 11 13 15
                vinr1 = vld2_s8(din_ptr1);
                vinr2 = vld2_s8(din_ptr2);
                voutr0 = vdupq_n_s32(bias_val);
                voutr0_1 = vdupq_n_s32(bias_val);

                vinr0.val[0] = vbsl_s8(vmask_rp1, vinr0.val[0], vzero);
                vinr0.val[1] = vbsl_s8(vmask_rp2, vinr0.val[1], vzero);

                vinr1.val[0] = vbsl_s8(vmask_rp1, vinr1.val[0], vzero);
                vinr1.val[1] = vbsl_s8(vmask_rp2, vinr1.val[1], vzero);

                vinr2.val[0] = vbsl_s8(vmask_rp1, vinr2.val[0], vzero);
                vinr2.val[1] = vbsl_s8(vmask_rp2, vinr2.val[1], vzero);

                tmp0 = vext_s8(vinr0.val[0], vzero, 1); // c = 2 4 6 8 10 12 14 0
                tmp1 = vext_s8(vinr1.val[0], vzero, 1); // c = 2 4 6 8 10 12 14 0
                tmp2 = vext_s8(vinr2.val[0], vzero, 1); // c = 2 4 6 8 10 12 14 0

                //r0
                voutr00 = vmull_s8(vinr0.val[0], wr00); //a = 0 2 4 6 8 10 12 14
                voutr01 = vmull_s8(vinr0.val[1], wr01); //b = 1 3 5 7 9 11 13 15
                voutr02 = vmull_s8(tmp0, wr02); //c = 2 4 6 8 10 12 14 0

                //r1
                voutr00 = vmlal_s8(voutr00, vinr1.val[0], wr10);//a = 0 2 4 6 8 10 12 14
                voutr01 = vmlal_s8(voutr01, vinr1.val[1], wr11);//b = 1 3 5 7 9 11 13 15
                voutr02 = vmlal_s8(voutr02, tmp1, wr12);//c = 2 4 6 8 10 12 14 16

                //r2
                voutr00 = vmlal_s8(voutr00, vinr2.val[0], wr20);//a = 0 2 4 6 8 10 12 14
                voutr01 = vmlal_s8(voutr01, vinr2.val[1], wr21);//b = 1 3 5 7 9 11 13 15
                voutr02 = vmlal_s8(voutr02, tmp2, wr22);//c = 2 4 6 8 10 12 14 16

                int32x4_t vdata0 = vld1q_s32(doutr0);
                int32x4_t vdata0_1 = vld1q_s32(doutr0 + 4);

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr01));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr01));

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr02));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr02));

                voutr0 = vbslq_s32(vmask_result1, voutr0, vdata0);
                voutr0_1 = vbslq_s32(vmask_result2, voutr0_1, vdata0_1);

                vst1q_s32(doutr0, voutr0);
                vst1q_s32(doutr0 + 4, voutr0_1);

                dout_ptr += w_out;

            }
        }
    }
}
//w_in <= 16
void conv_depthwise_3x3s2p1_bias_s_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {

    // printf("3x3s2 mult height \n");
    //! pad is done implicit
    // const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_out;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    unsigned int size_pad_right = (unsigned int)(w_in);

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)w_out;
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));
    // printf("vmask_rp1: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3], \
    //     vmask_rp1[4], vmask_rp1[5], vmask_rp1[6], vmask_rp1[7]);
    // printf("vmask_rp2: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3], \
    //     vmask_rp2[4], vmask_rp2[5], vmask_rp2[6], vmask_rp2[7]);
    // printf("vmask_result: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_result1[0], vmask_result1[1], vmask_result1[2], vmask_result1[3], \
    //     vmask_result2[0], vmask_result2[1], vmask_result2[2], vmask_result2[3]);

    // printf("rst_remain: %d, size_pad_right: %d\n", w_out, size_pad_right);

    int8x8_t vzero = vdup_n_s8(0);
    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

            int *doutr0 = nullptr;
            int *doutr1 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;
            const signed char *dr3 = dr2 + w_in;
            const signed char *dr4 = dr3 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;
            const signed char *din_ptr3 = nullptr;
            const signed char *din_ptr4 = nullptr;

            for (int i = 0; i < h_in; i += 4){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;
                din_ptr4 = dr4;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    din_ptr4 = dr3;
                    dr0 = dr3;
                    dr1 = dr4;
                }else{
                    dr0 = dr4;
                    dr1 = dr0 + w_in;
                }
                //! process bottom pad
                if (i + 4 > h_in) {
                    switch (i + 4 - h_in) {
                        case 4:
                            din_ptr1 = zero_ptr;
                        case 3:
                            din_ptr2 = zero_ptr;
                        case 2:
                            din_ptr3 = zero_ptr;
                        case 1:
                            din_ptr4 = zero_ptr;
                        default:
                            break;
                    }
                }

                if (i / 2 + 2 > h_out){
                    doutr1 = write_ptr;
                }
                //prefetch input
                prefetch(doutr0);
                prefetch(doutr1);

                prefetch(din_ptr0);
                prefetch(din_ptr1);
                prefetch(din_ptr2);
                prefetch(din_ptr3);
                prefetch(din_ptr4);

                //left
                int8x8x2_t vinr0 = vld2_s8(din_ptr0); //a = 0 2 4 6 8 10 12 14 b = 1 3 5 7 9 11 13 15
                int8x8x2_t vinr1 = vld2_s8(din_ptr1);
                int8x8x2_t vinr2 = vld2_s8(din_ptr2);
                int8x8x2_t vinr3 = vld2_s8(din_ptr3);
                int8x8x2_t vinr4 = vld2_s8(din_ptr4);

                int32x4_t voutr0 = vdupq_n_s32(bias_val);
                int32x4_t voutr0_1 = vdupq_n_s32(bias_val);
                int32x4_t voutr1 = vdupq_n_s32(bias_val);
                int32x4_t voutr1_1 = vdupq_n_s32(bias_val);

                vinr0.val[0] = vbsl_s8(vmask_rp1, vinr0.val[0], vzero);
                vinr0.val[1] = vbsl_s8(vmask_rp2, vinr0.val[1], vzero);

                vinr1.val[0] = vbsl_s8(vmask_rp1, vinr1.val[0], vzero);
                vinr1.val[1] = vbsl_s8(vmask_rp2, vinr1.val[1], vzero);

                vinr2.val[0] = vbsl_s8(vmask_rp1, vinr2.val[0], vzero);
                vinr2.val[1] = vbsl_s8(vmask_rp2, vinr2.val[1], vzero);

                vinr3.val[0] = vbsl_s8(vmask_rp1, vinr3.val[0], vzero);
                vinr3.val[1] = vbsl_s8(vmask_rp2, vinr3.val[1], vzero);

                vinr4.val[0] = vbsl_s8(vmask_rp1, vinr4.val[0], vzero);
                vinr4.val[1] = vbsl_s8(vmask_rp2, vinr4.val[1], vzero);

                int8x8_t tmp0 = vext_s8(vzero, vinr0.val[1], 7); //c = -1 1 3 5 7 9 11 13
                int8x8_t tmp1 = vext_s8(vzero, vinr1.val[1], 7); //c = -1 1 3 5 7 9 11 13
                int8x8_t tmp2 = vext_s8(vzero, vinr2.val[1], 7); //c = -1 1 3 5 7 9 11 13
                int8x8_t tmp3 = vext_s8(vzero, vinr3.val[1], 7); //c = -1 1 3 5 7 9 11 13
                int8x8_t tmp4 = vext_s8(vzero, vinr4.val[1], 7); //c = -1 1 3 5 7 9 11 13

                //r0
                int16x8_t voutr01 = vmull_s8(vinr0.val[0], wr01);//a = 0 2 4 6 8 10 12 14
                int16x8_t voutr02 = vmull_s8(vinr0.val[1], wr02);// b = 1 3 5 7 9 11 13 15
                int16x8_t voutr00 = vmull_s8(tmp0, wr00); //c = -1 1 3 5 7 9 11 13

                int16x8_t voutr11 = vmull_s8(vinr2.val[0], wr01);//a = 0 2 4 6 8 10 12 14
                int16x8_t voutr12 = vmull_s8(vinr2.val[1], wr02);// b = 1 3 5 7 9 11 13 15
                int16x8_t voutr10 = vmull_s8(tmp2, wr00); //c = -1 1 3 5 7 9 11 13

                //r1
                voutr01 = vmlal_s8(voutr01, vinr1.val[0], wr11);//a = 0 2 4 6 8 10 12 14
                voutr02 = vmlal_s8(voutr02, vinr1.val[1], wr12);// b = 1 3 5 7 9 11 13 15
                voutr00 = vmlal_s8(voutr00, tmp1, wr10); //c = -1 1 3 5 7 9 11 13

                voutr11 = vmlal_s8(voutr11, vinr3.val[0], wr11);//a = 0 2 4 6 8 10 12 14
                voutr12 = vmlal_s8(voutr12, vinr3.val[1], wr12);// b = 1 3 5 7 9 11 13 15
                voutr10 = vmlal_s8(voutr10, tmp3, wr10); //c = -1 1 3 5 7 9 11 13

                //r2
                voutr01 = vmlal_s8(voutr01, vinr2.val[0], wr21);//a = 0 2 4 6 8 10 12 14
                voutr02 = vmlal_s8(voutr02, vinr2.val[1], wr22);// b = 1 3 5 7 9 11 13 15
                voutr00 = vmlal_s8(voutr00, tmp2, wr20); //c = -1 1 3 5 7 9 11 13

                voutr11 = vmlal_s8(voutr11, vinr4.val[0], wr21);//a = 0 2 4 6 8 10 12 14
                voutr12 = vmlal_s8(voutr12, vinr4.val[1], wr22);// b = 1 3 5 7 9 11 13 15
                voutr10 = vmlal_s8(voutr10, tmp4, wr20); //c = -1 1 3 5 7 9 11 13

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr01));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr01));
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr11));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr11));

                int32x4_t vdata0 = vld1q_s32(doutr0);
                int32x4_t vdata0_1 = vld1q_s32(doutr0 + 4);

                int32x4_t vdata1 = vld1q_s32(doutr1);
                int32x4_t vdata1_1 = vld1q_s32(doutr1 + 4);

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr02));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr02));
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr12));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr12));

                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                dr4 = dr3 + w_in;

                voutr0 = vbslq_s32(vmask_result1, voutr0, vdata0);
                voutr0_1 = vbslq_s32(vmask_result2, voutr0_1, vdata0_1);
                voutr1 = vbslq_s32(vmask_result1, voutr1, vdata1);
                voutr1_1 = vbslq_s32(vmask_result2, voutr1_1, vdata1_1);

                vst1q_s32(doutr0, voutr0);
                vst1q_s32(doutr0 + 4, voutr0_1);
                vst1q_s32(doutr1, voutr1);
                vst1q_s32(doutr1 + 4, voutr1_1);

                dout_ptr += 2 * w_out;

            }
        }
    }
}

//relu
void conv_depthwise_3x3s1p1_bias_relu_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {

   // printf("3x3s1 mult height \n");
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 7) >> 3;
    int tile_h = (h_out + 3) >> 2;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(w_in - 7 - (cnt_col << 3));

    int size_pad_bottom = h_out % 4;

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

            int *doutr0 = nullptr;
            int *doutr1 = nullptr;
            int *doutr2 = nullptr;
            int *doutr3 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;
            const signed char *dr3 = dr2 + w_in;
            const signed char *dr4 = dr3 + w_in;
            const signed char *dr5 = dr4 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;
            const signed char *din_ptr3 = nullptr;
            const signed char *din_ptr4 = nullptr;
            const signed char *din_ptr5 = nullptr;

            for (int i = 0; i < h_in; i += 4){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;
                din_ptr4 = dr4;
                din_ptr5 = dr5;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                doutr2 = doutr1 + w_out;
                doutr3 = doutr2 + w_out;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    din_ptr4 = dr3;
                    din_ptr5 = dr4;
                    dr0 = dr3;
                    dr1 = dr4;
                    dr2 = dr5;
                }else{
                    dr0 = dr4;
                    dr1 = dr5;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 5 > h_in) {
                    switch (i + 5 - h_in) {
                        case 5:
                            din_ptr1 = zero_ptr;
                        case 4:
                            din_ptr2 = zero_ptr;
                        case 3:
                            din_ptr3 = zero_ptr;
                        case 2:
                            din_ptr4 = zero_ptr;
                        case 1:
                            din_ptr5 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 4 > h_out) {
                    switch (i + 4 - h_out) {
                        case 3:
                            doutr1 = write_ptr;
                        case 2:
                            doutr2 = write_ptr;
                        case 1:
                            doutr3 = write_ptr;
                        default:
                            break;
                    }
                }
                //prefetch input
                prefetch(doutr0);
                prefetch(doutr1);
                prefetch(doutr2);
                prefetch(doutr3);

                prefetch(din_ptr0);
                prefetch(din_ptr1);
                prefetch(din_ptr2);
                prefetch(din_ptr3);
                prefetch(din_ptr4);
                prefetch(din_ptr5);

                //left
                //din data
                int8x8_t vinr0 = vld1_s8(din_ptr0);//01234567
                int8x8_t vinr1 = vld1_s8(din_ptr1);

                // printf("vinr1: %d, %d, %d, %d, %d, %d, %d, %d \n", vinr1[0], vinr1[1], vinr1[2], vinr1[3], vinr1[4], vinr1[5], vinr1[6], vinr1[7]);

                int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 8);//01234567
                int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 8);

                int32x4_t voutr0 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                int32x4_t voutr0_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);

                int32x4_t voutr1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr1);
                int32x4_t voutr1_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr1);

                int32x4_t voutr2 = vdupq_n_s32(bias_val); //vld1q_f32(doutr2);
                int32x4_t voutr2_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr2);

                int32x4_t voutr3 = vdupq_n_s32(bias_val); //vld1q_f32(doutr3);
                int32x4_t voutr3_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr3);

                //r0 1234
                int16x8_t voutr00 = vmull_s8(vinr0, wr01);//01234567 * w11

                int8x8_t vtmp0 = vext_s8(vzero, vinr0, 7);//00123456
                int8x8_t vtmp1 = vext_s8(vinr0, vinr0_1, 1);//12345678

                int8x8_t vinr2 = vld1_s8(din_ptr2);
                int8x8_t vinr3 = vld1_s8(din_ptr3);//01234567

                voutr00 = vmlal_s8(voutr00, vtmp0, wr00);//r0 * w01

                int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 8);
                int8x8_t vinr3_1 = vld1_s8(din_ptr3 + 8);//01234567

                int8x8_t vinr4 = vld1_s8(din_ptr4);
                int8x8_t vinr5 = vld1_s8(din_ptr5);

                voutr00 = vmlal_s8(voutr00, vtmp1, wr02);//r0 * w01

                // printf("voutr0: %d, %d, %d, %d \n", voutr0[0], voutr0[1], voutr0[2], voutr0[3]);
                //r1
                din_ptr0 += 7;
                din_ptr1 += 7;
                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                int16x8_t voutr10 = vmull_s8(vinr1, wr01);//r0 * w01
                voutr00 = vmull_s8(vinr1, wr11);//r0 * w01

                vtmp0 = vext_s8(vzero, vinr1, 7);//00123456
                vtmp1 = vext_s8(vinr1, vinr1_1, 1);//12345678

                int8x8_t vinr4_1 = vld1_s8(din_ptr4 + 8);
                int8x8_t vinr5_1 = vld1_s8(din_ptr5 + 8);

                voutr10 = vmlal_s8(voutr10, vtmp0, wr00);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp0, wr10);//r0 * w01

                din_ptr2 += 7;
                din_ptr3 += 7;
                voutr10 = vmlal_s8(voutr10, vtmp1, wr02);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp1, wr12);//r0 * w01

                //r2
                din_ptr4 += 7;
                int16x8_t voutr20 = vmull_s8(vinr2, wr01);//r0 * w01
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));
                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                voutr10 = vmull_s8(vinr2, wr11);//r0 * w01
                voutr00 = vmull_s8(vinr2, wr21);//r0 * w01

                vtmp0 = vext_s8(vzero, vinr2, 7);//00123456
                vtmp1 = vext_s8(vinr2, vinr2_1, 1);//12345678

                voutr20 = vmlal_s8(voutr20, vtmp0, wr00);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp0, wr10);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp0, wr20);//r0 * w01

                din_ptr5 += 7;
                voutr20 = vmlal_s8(voutr20, vtmp1, wr02);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp1, wr12);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp1, wr22);//r0 * w01

                //r3
                vtmp0 = vext_s8(vzero, vinr3, 7);//00123456
                vtmp1 = vext_s8(vinr3, vinr3_1, 1);//12345678
                int16x8_t voutr30 = vmull_s8(vinr3, wr01);//r0 * w01

                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                voutr20 = vmull_s8(vinr3, wr11);//r0 * w01
                voutr10 = vmull_s8(vinr3, wr21);//r0 * w01

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr00);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp0, wr10);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp0, wr20);//r0 * w01

                voutr30 = vmlal_s8(voutr30, vtmp1, wr02);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp1, wr12);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp1, wr22);//r0 * w01

                //r4
                vtmp0 = vext_s8(vzero, vinr4, 7);//00123456
                vtmp1 = vext_s8(vinr4, vinr4_1, 1);//12345678
                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                voutr30 = vmull_s8(vinr4, wr11);//r0 * w01
                voutr20 = vmull_s8(vinr4, wr21);//r0 * w01

                voutr0 = vmaxq_s32(voutr0, vzero_32);
                voutr0_1 = vmaxq_s32(voutr0_1, vzero_32);

                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr10);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp0, wr20);//r0 * w01

                vst1q_s32(doutr0, voutr0);
                vst1q_s32(doutr0 + 4, voutr0_1);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr12);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp1, wr22);//r0 * w01

                doutr0 += 8;

                //r5
                vtmp0 = vext_s8(vzero, vinr5, 7);//00123456
                vtmp1 = vext_s8(vinr5, vinr5_1, 1);//12345678

                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));
                voutr30 = vmull_s8(vinr5, wr21);//r0 * w01

                voutr1 = vmaxq_s32(voutr1, vzero_32);
                voutr1_1 = vmaxq_s32(voutr1_1, vzero_32);

                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr20);//r0 * w01

                vst1q_s32(doutr1, voutr1);
                vst1q_s32(doutr1 + 4, voutr1_1);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr22);//r0 * w01

                doutr1 += 8;

                voutr2 = vmaxq_s32(voutr2, vzero_32);
                voutr2_1 = vmaxq_s32(voutr2_1, vzero_32);

                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));

                vst1q_s32(doutr2, voutr2);
                vst1q_s32(doutr2 + 4, voutr2_1);

                voutr3 = vmaxq_s32(voutr3, vzero_32);
                voutr3_1 = vmaxq_s32(voutr3_1, vzero_32);
                doutr2 += 8;

                vst1q_s32(doutr3, voutr3);
                vst1q_s32(doutr3 + 4, voutr3_1);
                doutr3 += 8;

                //mid col
                for (int j = 0; j < cnt_col; ++j) {
                    //din data
                    vinr0 = vld1_s8(din_ptr0);//01234567
                    vinr1 = vld1_s8(din_ptr1);

                    vinr0_1 = vld1_s8(din_ptr0 + 8);//01234567
                    vinr1_1 = vld1_s8(din_ptr1 + 8);

                    voutr0 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                    voutr0_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                    voutr1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr1);
                    voutr1_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                    voutr2 = vdupq_n_s32(bias_val); //vld1q_f32(doutr2);
                    voutr2_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                    voutr3 = vdupq_n_s32(bias_val); //vld1q_f32(doutr3);
                    voutr3_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);

                    //r0 1234
                    voutr00 = vmull_s8(vinr0, wr00);//01234567 * w10

                    vtmp0 = vext_s8(vinr0, vinr0_1, 1);//12345678
                    vtmp1 = vext_s8(vinr0, vinr0_1, 2);//23456789

                    vinr2 = vld1_s8(din_ptr2);
                    vinr3 = vld1_s8(din_ptr3);//01234567

                    voutr00 = vmlal_s8(voutr00, vtmp0, wr01);//r0 * w01

                    vinr2_1 = vld1_s8(din_ptr2 + 8);
                    vinr3_1 = vld1_s8(din_ptr3 + 8);//01234567

                    din_ptr0 += 8;
                    voutr00 = vmlal_s8(voutr00, vtmp1, wr02);//r0 * w01

                    //r1
                    vinr4 = vld1_s8(din_ptr4);
                    vinr5 = vld1_s8(din_ptr5);
                    voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                    voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                    voutr10 = vmull_s8(vinr1, wr00);//r0 * w01
                    voutr00 = vmull_s8(vinr1, wr10);//r0 * w01

                    vtmp0 = vext_s8(vinr1, vinr1_1, 1);//00123456
                    vtmp1 = vext_s8(vinr1, vinr1_1, 2);//12345678
                    vinr4_1 = vld1_s8(din_ptr4 + 8);
                    vinr5_1 = vld1_s8(din_ptr5 + 8);

                    voutr10 = vmlal_s8(voutr10, vtmp0, wr01);//r0 * w01
                    voutr00 = vmlal_s8(voutr00, vtmp0, wr11);//r0 * w01

                    din_ptr1 += 8;
                    voutr10 = vmlal_s8(voutr10, vtmp1, wr02);//r0 * w01
                    voutr00 = vmlal_s8(voutr00, vtmp1, wr12);//r0 * w01

                    //r2
                    vtmp0 = vext_s8(vinr2, vinr2_1, 1);//00123456
                    vtmp1 = vext_s8(vinr2, vinr2_1, 2);//12345678
                    voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                    voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));
                    voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                    voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                    voutr20 = vmull_s8(vinr2, wr00);//r0 * w01
                    voutr10 = vmull_s8(vinr2, wr10);//r0 * w01
                    voutr00 = vmull_s8(vinr2, wr20);//r0 * w01

                    din_ptr2 += 8;
                    din_ptr3 += 8;

                    voutr20 = vmlal_s8(voutr20, vtmp0, wr01);//r0 * w01
                    voutr10 = vmlal_s8(voutr10, vtmp0, wr11);//r0 * w01
                    voutr00 = vmlal_s8(voutr00, vtmp0, wr21);//r0 * w01

                    din_ptr4 += 8;
                    voutr20 = vmlal_s8(voutr20, vtmp1, wr02);//r0 * w01
                    voutr10 = vmlal_s8(voutr10, vtmp1, wr12);//r0 * w01
                    voutr00 = vmlal_s8(voutr00, vtmp1, wr22);//r0 * w01

                    //r3
                    vtmp0 = vext_s8(vinr3, vinr3_1, 1);//00123456
                    vtmp1 = vext_s8(vinr3, vinr3_1, 2);//12345678
                    voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                    voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));
                    voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                    voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                    voutr30 = vmull_s8(vinr3, wr00);//r0 * w01
                    voutr20 = vmull_s8(vinr3, wr10);//r0 * w01
                    voutr10 = vmull_s8(vinr3, wr20);//r0 * w01

                    voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                    voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                    voutr30 = vmlal_s8(voutr30, vtmp0, wr01);//r0 * w01
                    voutr20 = vmlal_s8(voutr20, vtmp0, wr11);//r0 * w01
                    voutr10 = vmlal_s8(voutr10, vtmp0, wr21);//r0 * w01

                    din_ptr5 += 8;
                    voutr30 = vmlal_s8(voutr30, vtmp1, wr02);//r0 * w01
                    voutr20 = vmlal_s8(voutr20, vtmp1, wr12);//r0 * w01
                    voutr10 = vmlal_s8(voutr10, vtmp1, wr22);//r0 * w01

                    //r4
                    vtmp0 = vext_s8(vinr4, vinr4_1, 1);//00123456
                    vtmp1 = vext_s8(vinr4, vinr4_1, 2);//12345678
                    voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                    voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));
                    voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                    voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));
                    voutr30 = vmull_s8(vinr4, wr10);//r0 * w01
                    voutr20 = vmull_s8(vinr4, wr20);//r0 * w01

                    voutr0 = vmaxq_s32(voutr0, vzero_32);
                    voutr0_1 = vmaxq_s32(voutr0_1, vzero_32);

                    voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                    voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                    voutr30 = vmlal_s8(voutr30, vtmp0, wr11);//r0 * w01
                    voutr20 = vmlal_s8(voutr20, vtmp0, wr21);//r0 * w01

                    vst1q_s32(doutr0, voutr0);
                    vst1q_s32(doutr0 + 4, voutr0_1);

                    voutr30 = vmlal_s8(voutr30, vtmp1, wr12);//r0 * w01
                    voutr20 = vmlal_s8(voutr20, vtmp1, wr22);//r0 * w01

                    doutr0 += 8;

                    //r5
                    vtmp0 = vext_s8(vinr5, vinr5_1, 1);//00123456
                    vtmp1 = vext_s8(vinr5, vinr5_1, 2);//12345678
                    voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                    voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));
                    voutr30 = vmull_s8(vinr5, wr20);//r0 * w01

                    voutr1 = vmaxq_s32(voutr1, vzero_32);
                    voutr1_1 = vmaxq_s32(voutr1_1, vzero_32);
                    voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                    voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                    voutr30 = vmlal_s8(voutr30, vtmp0, wr21);//r0 * w01

                    vst1q_s32(doutr1, voutr1);
                    vst1q_s32(doutr1 + 4, voutr1_1);

                    voutr30 = vmlal_s8(voutr30, vtmp1, wr22);//r0 * w01

                    doutr1 += 8;
                    voutr2 = vmaxq_s32(voutr2, vzero_32);
                    voutr2_1 = vmaxq_s32(voutr2_1, vzero_32);

                    voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                    voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));

                    vst1q_s32(doutr2, voutr2);
                    vst1q_s32(doutr2 + 4, voutr2_1);
                    voutr3 = vmaxq_s32(voutr3, vzero_32);
                    voutr3_1 = vmaxq_s32(voutr3_1, vzero_32);
                    doutr2 += 8;

                    vst1q_s32(doutr3, voutr3);
                    vst1q_s32(doutr3 + 4, voutr3_1);
                    doutr3 += 8;

                }
                //right
                //din data
                vinr0 = vld1_s8(din_ptr0);//01234567
                vinr1 = vld1_s8(din_ptr1);
                // printf("vinr0: %d, %d, %d, %d, %d, %d, %d, %d \n", vinr0[0], vinr0[1], vinr0[2], vinr0[3], vinr0[4], vinr0[5], vinr0[6], vinr0[7]);

                vinr0_1 = vld1_s8(din_ptr0 + 8);//01234567
                vinr1_1 = vld1_s8(din_ptr1 + 8);

                voutr0 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                voutr0_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                voutr1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr1);
                voutr1_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                voutr2 = vdupq_n_s32(bias_val); //vld1q_f32(doutr2);
                voutr2_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                voutr3 = vdupq_n_s32(bias_val); //vld1q_f32(doutr3);
                voutr3_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);

                vinr0 = vbsl_s8(vmask_rp1, vinr0, vzero);
                vinr1 = vbsl_s8(vmask_rp1, vinr1, vzero);
                // printf("vinr0: %d, %d, %d, %d, %d, %d, %d, %d \n", vinr0[0], vinr0[1], vinr0[2], vinr0[3], vinr0[4], vinr0[5], vinr0[6], vinr0[7]);

                vinr0_1 = vbsl_s8(vmask_rp2, vinr0_1, vzero);
                vinr1_1 = vbsl_s8(vmask_rp2, vinr1_1, vzero);

                // printf("vinr0_1: %d, %d, %d, %d, %d, %d, %d, %d \n", vinr0_1[0], vinr0_1[1], vinr0_1[2], vinr0_1[3], vinr0_1[4], vinr0_1[5], vinr0_1[6], vinr0_1[7]);

                //r0 1234
                voutr00 = vmull_s8(vinr0, wr00);//01234567 * w11

                vtmp0 = vext_s8(vinr0, vinr0_1, 1);//12345678
                vtmp1 = vext_s8(vinr0, vinr0_1, 2);//23456789
                vinr2 = vld1_s8(din_ptr2);
                vinr3 = vld1_s8(din_ptr3);//01234567

                voutr00 = vmlal_s8(voutr00, vtmp0, wr01);//r0 * w01
                vinr2_1 = vld1_s8(din_ptr2 + 8);
                vinr3_1 = vld1_s8(din_ptr3 + 8);//01234567
                int32x4_t voutr0_32 = vld1q_s32(doutr0);
                int32x4_t voutr1_32 = vld1q_s32(doutr1);

                voutr00 = vmlal_s8(voutr00, vtmp1, wr02);//r0 * w01

                //r1
                vinr2 = vbsl_s8(vmask_rp1, vinr2, vzero);
                vinr3 = vbsl_s8(vmask_rp1, vinr3, vzero);
                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                voutr10 = vmull_s8(vinr1, wr00);//r0 * w01
                voutr00 = vmull_s8(vinr1, wr10);//r0 * w01

                vtmp0 = vext_s8(vinr1, vinr1_1, 1);//00123456
                vtmp1 = vext_s8(vinr1, vinr1_1, 2);//12345678

                vinr2_1 = vbsl_s8(vmask_rp2, vinr2_1, vzero);
                vinr3_1 = vbsl_s8(vmask_rp2, vinr3_1, vzero);

                voutr10 = vmlal_s8(voutr10, vtmp0, wr01);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp0, wr11);//r0 * w01

                int32x4_t voutr2_32 = vld1q_s32(doutr2);
                int32x4_t voutr3_32 = vld1q_s32(doutr3);

                voutr10 = vmlal_s8(voutr10, vtmp1, wr02);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp1, wr12);//r0 * w01

                //r2
                vinr4 = vld1_s8(din_ptr4);
                vinr5 = vld1_s8(din_ptr5);
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));
                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                voutr20 = vmull_s8(vinr2, wr00);//r0 * w01
                voutr10 = vmull_s8(vinr2, wr10);//r0 * w01
                voutr00 = vmull_s8(vinr2, wr20);//r0 * w01

                vtmp0 = vext_s8(vinr2, vinr2_1, 1);//00123456
                vtmp1 = vext_s8(vinr2, vinr2_1, 2);//12345678

                vinr4_1 = vld1_s8(din_ptr4 + 8);
                vinr5_1 = vld1_s8(din_ptr5 + 8);

                voutr20 = vmlal_s8(voutr20, vtmp0, wr01);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp0, wr11);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp0, wr21);//r0 * w01

                int32x4_t voutr0_32_1 = vld1q_s32(doutr0 + 4);
                int32x4_t voutr1_32_1 = vld1q_s32(doutr1 + 4);

                voutr20 = vmlal_s8(voutr20, vtmp1, wr02);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp1, wr12);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp1, wr22);//r0 * w01

                //r3
                vtmp0 = vext_s8(vinr3, vinr3_1, 1);//00123456
                vtmp1 = vext_s8(vinr3, vinr3_1, 2);//12345678
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                voutr30 = vmull_s8(vinr3, wr00);//r0 * w01
                voutr20 = vmull_s8(vinr3, wr10);
                voutr10 = vmull_s8(vinr3, wr20);

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                vinr4 = vbsl_s8(vmask_rp1, vinr4, vzero);
                vinr5 = vbsl_s8(vmask_rp1, vinr5, vzero);

                // uint16x8_t vm_res = vmovl_u8(vmask_result);
                int32x4_t voutr2_32_1 = vld1q_s32(doutr2 + 4);
                int32x4_t voutr3_32_1 = vld1q_s32(doutr3 + 4);

                voutr30 = vmlal_s8(voutr30, vtmp0, wr01);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp0, wr11);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp0, wr21);//r0 * w01

                vinr4_1 = vbsl_s8(vmask_rp2, vinr4_1, vzero);
                vinr5_1 = vbsl_s8(vmask_rp2, vinr5_1, vzero);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr02);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp1, wr12);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp1, wr22);//r0 * w01

                //r4
                vtmp0 = vext_s8(vinr4, vinr4_1, 1);//00123456
                vtmp1 = vext_s8(vinr4, vinr4_1, 2);//12345678
                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                voutr30 = vmull_s8(vinr4, wr10);//r0 * w01
                voutr20 = vmull_s8(vinr4, wr20);//r0 * w01

                voutr0 = vmaxq_s32(voutr0, vzero_32);
                voutr0_1 = vmaxq_s32(voutr0_1, vzero_32);

                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr11);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp0, wr21);//r0 * w01

                voutr0 = vbslq_s32(vmask_result1, voutr0, voutr0_32);
                voutr0_1 = vbslq_s32(vmask_result2, voutr0_1, voutr0_32_1);
                // printf("voutr1_0: %d, %d, %d, %d \n", voutr1_0[0], voutr1_0[1], voutr1_0[2], voutr1_0[3]);
                // printf("voutr1_1: %d, %d, %d, %d \n", voutr1_1[0], voutr1_1[1], voutr1_1[2], voutr1_1[3]);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr12);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp1, wr22);//r0 * w01

                voutr1 = vmaxq_s32(voutr1, vzero_32);
                voutr1_1 = vmaxq_s32(voutr1_1, vzero_32);

                vst1q_s32(doutr0, voutr0);
                vst1q_s32(doutr0 + 4, voutr0_1);

                //r5
                vtmp0 = vext_s8(vinr5, vinr5_1, 1);//00123456
                vtmp1 = vext_s8(vinr5, vinr5_1, 2);//12345678
                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));

                voutr30 = vmull_s8(vinr5, wr20);//r0 * w01

                voutr1 = vbslq_s32(vmask_result1, voutr1, voutr1_32);
                voutr1_1 = vbslq_s32(vmask_result2, voutr1_1, voutr1_32_1);

                doutr0 += 8;
                dr3 = dr2 + w_in;
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr21);//r0 * w01

                dr4 = dr3 + w_in;
                dr5 = dr4 + w_in;
                vst1q_s32(doutr1, voutr1);
                vst1q_s32(doutr1 + 4, voutr1_1);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr22);//r0 * w01

                voutr2 = vmaxq_s32(voutr2, vzero_32);
                voutr2_1 = vmaxq_s32(voutr2_1, vzero_32);

                doutr1 += 8;

                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));

                voutr2 = vbslq_s32(vmask_result1, voutr2, voutr2_32);
                voutr2_1 = vbslq_s32(vmask_result2, voutr2_1, voutr2_32_1);

                dout_ptr = dout_ptr + 4 * w_out;

                voutr3 = vmaxq_s32(voutr3, vzero_32);
                voutr3_1 = vmaxq_s32(voutr3_1, vzero_32);

                vst1q_s32(doutr2, voutr2);
                vst1q_s32(doutr2 + 4, voutr2_1);

                voutr3 = vbslq_s32(vmask_result1, voutr3, voutr3_32);
                voutr3_1 = vbslq_s32(vmask_result2, voutr3_1, voutr3_32_1);

                doutr2 += 8;

                vst1q_s32(doutr3, voutr3);
                vst1q_s32(doutr3 + 4, voutr3_1);
                doutr3 += 8;
            }
        }
    }
}
//w_in <= 8
void conv_depthwise_3x3s1p1_bias_s_relu_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {

   // printf("3x3s1 mult height \n");
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_h = (h_out + 3) >> 2;

    unsigned int size_pad_right = (unsigned int)(w_in);

    int size_pad_bottom = h_out % 4;

    uint8x8_t vmask_rp = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    unsigned int rst_remain = (unsigned int)w_out;
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

            int *doutr0 = nullptr;
            int *doutr1 = nullptr;
            int *doutr2 = nullptr;
            int *doutr3 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;
            const signed char *dr3 = dr2 + w_in;
            const signed char *dr4 = dr3 + w_in;
            const signed char *dr5 = dr4 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;
            const signed char *din_ptr3 = nullptr;
            const signed char *din_ptr4 = nullptr;
            const signed char *din_ptr5 = nullptr;

            for (int i = 0; i < h_in; i += 4){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;
                din_ptr4 = dr4;
                din_ptr5 = dr5;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                doutr2 = doutr1 + w_out;
                doutr3 = doutr2 + w_out;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    din_ptr4 = dr3;
                    din_ptr5 = dr4;
                    dr0 = dr3;
                    dr1 = dr4;
                    dr2 = dr5;
                }else{
                    dr0 = dr4;
                    dr1 = dr5;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 5 > h_in) {
                    switch (i + 5 - h_in) {
                        case 5:
                            din_ptr1 = zero_ptr;
                        case 4:
                            din_ptr2 = zero_ptr;
                        case 3:
                            din_ptr3 = zero_ptr;
                        case 2:
                            din_ptr4 = zero_ptr;
                        case 1:
                            din_ptr5 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 4 > h_out) {
                    switch (i + 4 - h_out) {
                        case 3:
                            doutr1 = write_ptr;
                        case 2:
                            doutr2 = write_ptr;
                        case 1:
                            doutr3 = write_ptr;
                        default:
                            break;
                    }
                }
                //prefetch input
                prefetch(doutr0);
                prefetch(doutr1);
                prefetch(doutr2);
                prefetch(doutr3);

                prefetch(din_ptr0);
                prefetch(din_ptr1);
                prefetch(din_ptr2);
                prefetch(din_ptr3);
                prefetch(din_ptr4);
                prefetch(din_ptr5);

                //din data
                int8x8_t vinr0 = vld1_s8(din_ptr0);//01234567
                int8x8_t vinr1 = vld1_s8(din_ptr1);

                int32x4_t voutr0 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                int32x4_t voutr0_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                int32x4_t voutr1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr1);
                int32x4_t voutr1_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                int32x4_t voutr2 = vdupq_n_s32(bias_val); //vld1q_f32(doutr2);
                int32x4_t voutr2_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);
                int32x4_t voutr3 = vdupq_n_s32(bias_val); //vld1q_f32(doutr3);
                int32x4_t voutr3_1 = vdupq_n_s32(bias_val); //vld1q_f32(doutr0);

                vinr0 = vbsl_s8(vmask_rp, vinr0, vzero);
                vinr1 = vbsl_s8(vmask_rp, vinr1, vzero);

                //r0 1234
                int16x8_t voutr00 = vmull_s8(vinr0, wr01);//01234567 * w01

                int8x8_t vtmp0 = vext_s8(vzero, vinr0, 7);//001234567
                int8x8_t vtmp1 = vext_s8(vinr0, vzero, 1);//12345670
                int8x8_t vinr2 = vld1_s8(din_ptr2);
                int8x8_t vinr3 = vld1_s8(din_ptr3);//01234567

                voutr00 = vmlal_s8(voutr00, vtmp0, wr00);//r0 * w01

                int32x4_t voutr0_32 = vld1q_s32(doutr0);
                int32x4_t voutr1_32 = vld1q_s32(doutr1);

                voutr00 = vmlal_s8(voutr00, vtmp1, wr02);//r0 * w01

                //r1
                vtmp0 = vext_s8(vzero, vinr1, 7);//00123456
                vtmp1 = vext_s8(vinr1, vzero, 1);//12345678
                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                int16x8_t voutr10 = vmull_s8(vinr1, wr01);//r0 * w01
                voutr00 = vmull_s8(vinr1, wr11);//r0 * w01

                vinr2 = vbsl_s8(vmask_rp, vinr2, vzero);
                vinr3 = vbsl_s8(vmask_rp, vinr3, vzero);
                int8x8_t vinr4 = vld1_s8(din_ptr4);
                int8x8_t vinr5 = vld1_s8(din_ptr5);

                voutr10 = vmlal_s8(voutr10, vtmp0, wr00);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp0, wr10);//r0 * w01

                int32x4_t voutr2_32 = vld1q_s32(doutr2);
                int32x4_t voutr3_32 = vld1q_s32(doutr3);

                voutr10 = vmlal_s8(voutr10, vtmp1, wr02);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp1, wr12);//r0 * w01

                //r2
                vtmp0 = vext_s8(vzero, vinr2, 7);//00123456
                vtmp1 = vext_s8(vinr2, vzero, 1);//12345678
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));
                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                int16x8_t voutr20 = vmull_s8(vinr2, wr01);//r0 * w01
                voutr10 = vmull_s8(vinr2, wr11);//r0 * w01
                voutr00 = vmull_s8(vinr2, wr21);//r0 * w01

                vinr4 = vbsl_s8(vmask_rp, vinr4, vzero);
                vinr5 = vbsl_s8(vmask_rp, vinr5, vzero);

                voutr20 = vmlal_s8(voutr20, vtmp0, wr00);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp0, wr10);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp0, wr20);//r0 * w01

                int32x4_t voutr0_32_1 = vld1q_s32(doutr0 + 4);
                int32x4_t voutr1_32_1 = vld1q_s32(doutr1 + 4);

                voutr20 = vmlal_s8(voutr20, vtmp1, wr02);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp1, wr12);//r0 * w01
                voutr00 = vmlal_s8(voutr00, vtmp1, wr22);//r0 * w01

                //r3
                vtmp0 = vext_s8(vzero, vinr3, 7);//00123456
                vtmp1 = vext_s8(vinr3, vzero, 1);//12345678
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                int16x8_t voutr30 = vmull_s8(vinr3, wr01);//r0 * w01
                voutr20 = vmull_s8(vinr3, wr11);
                voutr10 = vmull_s8(vinr3, wr21);

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                // uint16x8_t vm_res = vmovl_u8(vmask_result);
                int32x4_t voutr2_32_1 = vld1q_s32(doutr2 + 4);
                int32x4_t voutr3_32_1 = vld1q_s32(doutr3 + 4);

                voutr30 = vmlal_s8(voutr30, vtmp0, wr00);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp0, wr10);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp0, wr20);//r0 * w01

                voutr0 = vmaxq_s32(voutr0, vzero_32);
                voutr0_1 = vmaxq_s32(voutr0_1, vzero_32);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr02);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp1, wr12);//r0 * w01
                voutr10 = vmlal_s8(voutr10, vtmp1, wr22);//r0 * w01

                //r4
                vtmp0 = vext_s8(vzero, vinr4, 7);//00123456
                vtmp1 = vext_s8(vinr4, vzero, 1);//12345678
                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                voutr30 = vmull_s8(vinr4, wr11);//r0 * w01
                voutr20 = vmull_s8(vinr4, wr21);//r0 * w01

                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr10);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp0, wr20);//r0 * w01

                voutr0 = vbslq_s32(vmask_result1, voutr0, voutr0_32);
                voutr0_1 = vbslq_s32(vmask_result2, voutr0_1, voutr0_32_1);

                voutr1 = vmaxq_s32(voutr1, vzero_32);
                voutr1_1 = vmaxq_s32(voutr1_1, vzero_32);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr12);//r0 * w01
                voutr20 = vmlal_s8(voutr20, vtmp1, wr22);//r0 * w01

                vst1q_s32(doutr0, voutr0);
                vst1q_s32(doutr0 + 4, voutr0_1);

                voutr1 = vbslq_s32(vmask_result1, voutr1, voutr1_32);
                voutr1_1 = vbslq_s32(vmask_result2, voutr1_1, voutr1_32_1);

                //r5
                vtmp0 = vext_s8(vzero, vinr5, 7);//00123456
                vtmp1 = vext_s8(vinr5, vzero, 1);//12345678
                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));

                voutr30 = vmull_s8(vinr5, wr21);//r0 * w01

                doutr0 += 8;
                dr3 = dr2 + w_in;
                voutr2 = vaddw_s16(voutr2, vget_low_s16(voutr20));
                voutr2_1 = vaddw_s16(voutr2_1, vget_high_s16(voutr20));

                voutr30 = vmlal_s8(voutr30, vtmp0, wr20);//r0 * w01

                dr4 = dr3 + w_in;
                dr5 = dr4 + w_in;

                vst1q_s32(doutr1, voutr1);
                vst1q_s32(doutr1 + 4, voutr1_1);

                voutr30 = vmlal_s8(voutr30, vtmp1, wr22);//r0 * w01

                voutr2 = vmaxq_s32(voutr2, vzero_32);
                voutr2_1 = vmaxq_s32(voutr2_1, vzero_32);

                doutr1 += 8;

                voutr3 = vaddw_s16(voutr3, vget_low_s16(voutr30));
                voutr3_1 = vaddw_s16(voutr3_1, vget_high_s16(voutr30));

                voutr2 = vbslq_s32(vmask_result1, voutr2, voutr2_32);
                voutr2_1 = vbslq_s32(vmask_result2, voutr2_1, voutr2_32_1);

                dout_ptr = dout_ptr + 4 * w_out;

                voutr3 = vmaxq_s32(voutr3, vzero_32);
                voutr3_1 = vmaxq_s32(voutr3_1, vzero_32);

                vst1q_s32(doutr2, voutr2);
                vst1q_s32(doutr2 + 4, voutr2_1);

                voutr3 = vbslq_s32(vmask_result1, voutr3, voutr3_32);
                voutr3_1 = vbslq_s32(vmask_result2, voutr3_1, voutr3_32_1);
                doutr2 += 8;

                vst1q_s32(doutr3, voutr3);
                vst1q_s32(doutr3 + 4, voutr3_1);
                doutr3 += 8;
            }
        }
    }
}

//1 line w_in > 16
void conv_depthwise_3x3s2p1_bias_relu_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {

   // printf("3x3s2 mult height \n");
    //! pad is done implicit
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_out;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 15) >> 4;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(w_in - 15 - (cnt_col << 4));
    if (size_pad_right == 17){
        size_pad_right = 0;
        cnt_col++;
    }

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);

    // printf("vmask_rp1: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3], \
    //     vmask_rp1[4], vmask_rp1[5], vmask_rp1[6], vmask_rp1[7]);
    // printf("vmask_rp2: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3], \
    //     vmask_rp2[4], vmask_rp2[5], vmask_rp2[6], vmask_rp2[7]);
    // printf("vmask_result: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_result1[0], vmask_result1[1], vmask_result1[2], vmask_result1[3], \
    //     vmask_result2[0], vmask_result2[1], vmask_result2[2], vmask_result2[3]);

    // printf("cnt_col: %d, rst_remain: %d, size_pad_right: %d\n", cnt_col, rst_remain, size_pad_right);
    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

            int *doutr0 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;

                doutr0 = dout_ptr;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr1 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr0 + w_in;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 2 > h_in) {
                    switch (i + 2 - h_in) {
                        case 2:
                            din_ptr1 = zero_ptr;
                        case 1:
                            din_ptr2 = zero_ptr;
                        default:
                            break;
                    }
                }
                //prefetch input
                prefetch(doutr0);

                prefetch(din_ptr0);
                prefetch(din_ptr1);
                prefetch(din_ptr2);

                //left
                int8x8x2_t vinr0 = vld2_s8(din_ptr0); //a = 0 2 4 6 8 10 12 14 b = 1 3 5 7 9 11 13 15
                int8x8x2_t vinr1 = vld2_s8(din_ptr1);
                int8x8x2_t vinr2 = vld2_s8(din_ptr2);

                int32x4_t voutr0 = vdupq_n_s32(bias_val);
                int32x4_t voutr0_1 = vdupq_n_s32(bias_val);

                int8x8_t tmp0 = vext_s8(vzero, vinr0.val[1], 7); //c = -1 1 3 5 7 9 11 13
                int8x8_t tmp1 = vext_s8(vzero, vinr1.val[1], 7); //c = -1 1 3 5 7 9 11 13
                int8x8_t tmp2 = vext_s8(vzero, vinr2.val[1], 7); //c = -1 1 3 5 7 9 11 13

                //r0
                int16x8_t voutr01 = vmull_s8(vinr0.val[0], wr01);//a = 0 2 4 6 8 10 12 14
                int16x8_t voutr02 = vmull_s8(vinr0.val[1], wr02);// b = 1 3 5 7 9 11 13 15
                int16x8_t voutr00 = vmull_s8(tmp0, wr00); //c = -1 1 3 5 7 9 11 13

                //r1
                voutr01 = vmlal_s8(voutr01, vinr1.val[0], wr11);//a = 0 2 4 6 8 10 12 14
                voutr02 = vmlal_s8(voutr02, vinr1.val[1], wr12);// b = 1 3 5 7 9 11 13 15
                voutr00 = vmlal_s8(voutr00, tmp1, wr10); //c = -1 1 3 5 7 9 11 13

                //r2
                voutr01 = vmlal_s8(voutr01, vinr2.val[0], wr21);//a = 0 2 4 6 8 10 12 14
                voutr02 = vmlal_s8(voutr02, vinr2.val[1], wr22);// b = 1 3 5 7 9 11 13 15
                voutr00 = vmlal_s8(voutr00, tmp2, wr20); //c = -1 1 3 5 7 9 11 13

                din_ptr0 += 15;
                din_ptr1 += 15;
                din_ptr2 += 15;

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr01));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr01));

                prefetch(din_ptr0);
                prefetch(din_ptr1);
                prefetch(din_ptr2);

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr02));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr02));

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                voutr0 = vmaxq_s32(voutr0, vzero_32);
                voutr0_1 = vmaxq_s32(voutr0_1, vzero_32);

                vst1q_s32(doutr0, voutr0);
                vst1q_s32(doutr0 + 4, voutr0_1);
                doutr0 += 8;

                //mid
                for (int i =0; i < cnt_col; i++){
                    vinr0 = vld2_s8(din_ptr0);//a = 0 2 4 6 8 10 12 14  b = 1 3 5 7 9 11 13 15
                    vinr1 = vld2_s8(din_ptr1);
                    vinr2 = vld2_s8(din_ptr2);

                    int8x8_t vinr0_1 = vld1_s8(din_ptr0 + 16); // d = 16 17 18 19
                    int8x8_t vinr1_1 = vld1_s8(din_ptr1 + 16);
                    int8x8_t vinr2_1 = vld1_s8(din_ptr2 + 16);

                    voutr0 = vdupq_n_s32(bias_val); //
                    voutr0_1 = vdupq_n_s32(bias_val);

                    tmp0 = vext_s8(vinr0.val[0], vinr0_1, 1); // c = 2 4 6 8 10 12 14 16
                    tmp1 = vext_s8(vinr1.val[0], vinr1_1, 1); // c = 2 4 6 8 10 12 14 16
                    tmp2 = vext_s8(vinr2.val[0], vinr2_1, 1); // c = 2 4 6 8 10 12 14 16

                    //r0
                    voutr00 = vmull_s8(vinr0.val[0], wr00);//a = 0 2 4 6 8 10 12 14
                    voutr01 = vmull_s8(vinr0.val[1], wr01);//b = 1 3 5 7 9 11 13 15
                    voutr02 = vmull_s8(tmp0, wr02);//c = 2 4 6 8 10 12 14 16

                    //r1
                    voutr00 = vmlal_s8(voutr00, vinr1.val[0], wr10);//a = 0 2 4 6 8 10 12 14
                    voutr01 = vmlal_s8(voutr01, vinr1.val[1], wr11);//b = 1 3 5 7 9 11 13 15
                    voutr02 = vmlal_s8(voutr02, tmp1, wr12);//c = 2 4 6 8 10 12 14 16

                    //r2
                    voutr00 = vmlal_s8(voutr00, vinr2.val[0], wr20);//a = 0 2 4 6 8 10 12 14
                    voutr01 = vmlal_s8(voutr01, vinr2.val[1], wr21);//b = 1 3 5 7 9 11 13 15
                    voutr02 = vmlal_s8(voutr02, tmp2, wr22);//c = 2 4 6 8 10 12 14 16

                    din_ptr0 += 16;
                    din_ptr1 += 16;
                    din_ptr2 += 16;

                    voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                    voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                    prefetch(din_ptr0);
                    prefetch(din_ptr1);
                    prefetch(din_ptr2);

                    voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr01));
                    voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr01));

                    voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr02));
                    voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr02));

                    voutr0 = vmaxq_s32(voutr0, vzero_32);
                    voutr0_1 = vmaxq_s32(voutr0_1, vzero_32);

                    vst1q_s32(doutr0, voutr0);
                    vst1q_s32(doutr0 + 4, voutr0_1);

                    doutr0 += 8;

                }
                //right
                if (size_pad_right == 17){
                    continue;
                }
                prefetch(doutr0);
                vinr0 = vld2_s8(din_ptr0);//a = 0 2 4 6 8 10 12 14 b = 1 3 5 7 9 11 13 15
                vinr1 = vld2_s8(din_ptr1);
                vinr2 = vld2_s8(din_ptr2);
                voutr0 = vdupq_n_s32(bias_val);
                voutr0_1 = vdupq_n_s32(bias_val);

                vinr0.val[0] = vbsl_s8(vmask_rp1, vinr0.val[0], vzero);
                vinr0.val[1] = vbsl_s8(vmask_rp2, vinr0.val[1], vzero);

                vinr1.val[0] = vbsl_s8(vmask_rp1, vinr1.val[0], vzero);
                vinr1.val[1] = vbsl_s8(vmask_rp2, vinr1.val[1], vzero);

                vinr2.val[0] = vbsl_s8(vmask_rp1, vinr2.val[0], vzero);
                vinr2.val[1] = vbsl_s8(vmask_rp2, vinr2.val[1], vzero);

                tmp0 = vext_s8(vinr0.val[0], vzero, 1); // c = 2 4 6 8 10 12 14 0
                tmp1 = vext_s8(vinr1.val[0], vzero, 1); // c = 2 4 6 8 10 12 14 0
                tmp2 = vext_s8(vinr2.val[0], vzero, 1); // c = 2 4 6 8 10 12 14 0

                //r0
                voutr00 = vmull_s8(vinr0.val[0], wr00); //a = 0 2 4 6 8 10 12 14
                voutr01 = vmull_s8(vinr0.val[1], wr01); //b = 1 3 5 7 9 11 13 15
                voutr02 = vmull_s8(tmp0, wr02); //c = 2 4 6 8 10 12 14 0

                //r1
                voutr00 = vmlal_s8(voutr00, vinr1.val[0], wr10);//a = 0 2 4 6 8 10 12 14
                voutr01 = vmlal_s8(voutr01, vinr1.val[1], wr11);//b = 1 3 5 7 9 11 13 15
                voutr02 = vmlal_s8(voutr02, tmp1, wr12);//c = 2 4 6 8 10 12 14 16

                //r2
                voutr00 = vmlal_s8(voutr00, vinr2.val[0], wr20);//a = 0 2 4 6 8 10 12 14
                voutr01 = vmlal_s8(voutr01, vinr2.val[1], wr21);//b = 1 3 5 7 9 11 13 15
                voutr02 = vmlal_s8(voutr02, tmp2, wr22);//c = 2 4 6 8 10 12 14 16

                int32x4_t vdata0 = vld1q_s32(doutr0);
                int32x4_t vdata0_1 = vld1q_s32(doutr0 + 4);

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr01));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr01));

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr02));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr02));

                voutr0 = vmaxq_s32(voutr0, vzero_32);
                voutr0_1 = vmaxq_s32(voutr0_1, vzero_32);

                voutr0 = vbslq_s32(vmask_result1, voutr0, vdata0);
                voutr0_1 = vbslq_s32(vmask_result2, voutr0_1, vdata0_1);

                vst1q_s32(doutr0, voutr0);
                vst1q_s32(doutr0 + 4, voutr0_1);

                dout_ptr += w_out;

            }
        }
    }
}
//w_in <= 16
void conv_depthwise_3x3s2p1_bias_s_relu_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {

    // printf("3x3s2 mult height \n");
    //! pad is done implicit
    // const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_out;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    unsigned int size_pad_right = (unsigned int)(w_in);

    uint8x8_t vmask_rp1 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned int rst_remain = (unsigned int)w_out;
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));
    // printf("vmask_rp1: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3], \
    //     vmask_rp1[4], vmask_rp1[5], vmask_rp1[6], vmask_rp1[7]);
    // printf("vmask_rp2: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3], \
    //     vmask_rp2[4], vmask_rp2[5], vmask_rp2[6], vmask_rp2[7]);
    // printf("vmask_result: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask_result1[0], vmask_result1[1], vmask_result1[2], vmask_result1[3], \
    //     vmask_result2[0], vmask_result2[1], vmask_result2[2], vmask_result2[3]);

    // printf("rst_remain: %d, size_pad_right: %d\n", w_out, size_pad_right);

    int8x8_t vzero = vdup_n_s8(0);
    int32x4_t vzero_32 = vdupq_n_s32(0);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

            int *doutr0 = nullptr;
            int *doutr1 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;
            const signed char *dr3 = dr2 + w_in;
            const signed char *dr4 = dr3 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;
            const signed char *din_ptr3 = nullptr;
            const signed char *din_ptr4 = nullptr;

            for (int i = 0; i < h_in; i += 4){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;
                din_ptr4 = dr4;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    din_ptr4 = dr3;
                    dr0 = dr3;
                    dr1 = dr4;
                }else{
                    dr0 = dr4;
                    dr1 = dr0 + w_in;
                }
                //! process bottom pad
                if (i + 4 > h_in) {
                    switch (i + 4 - h_in) {
                        case 4:
                            din_ptr1 = zero_ptr;
                        case 3:
                            din_ptr2 = zero_ptr;
                        case 2:
                            din_ptr3 = zero_ptr;
                        case 1:
                            din_ptr4 = zero_ptr;
                        default:
                            break;
                    }
                }

                if (i / 2 + 2 > h_out){
                    doutr1 = write_ptr;
                }
                //prefetch input
                prefetch(doutr0);
                prefetch(doutr1);

                prefetch(din_ptr0);
                prefetch(din_ptr1);
                prefetch(din_ptr2);
                prefetch(din_ptr3);
                prefetch(din_ptr4);

                //left
                int8x8x2_t vinr0 = vld2_s8(din_ptr0); //a = 0 2 4 6 8 10 12 14 b = 1 3 5 7 9 11 13 15
                int8x8x2_t vinr1 = vld2_s8(din_ptr1);
                int8x8x2_t vinr2 = vld2_s8(din_ptr2);
                int8x8x2_t vinr3 = vld2_s8(din_ptr3);
                int8x8x2_t vinr4 = vld2_s8(din_ptr4);

                int32x4_t voutr0 = vdupq_n_s32(bias_val);
                int32x4_t voutr0_1 = vdupq_n_s32(bias_val);
                int32x4_t voutr1 = vdupq_n_s32(bias_val);
                int32x4_t voutr1_1 = vdupq_n_s32(bias_val);

                vinr0.val[0] = vbsl_s8(vmask_rp1, vinr0.val[0], vzero);
                vinr0.val[1] = vbsl_s8(vmask_rp2, vinr0.val[1], vzero);

                vinr1.val[0] = vbsl_s8(vmask_rp1, vinr1.val[0], vzero);
                vinr1.val[1] = vbsl_s8(vmask_rp2, vinr1.val[1], vzero);

                vinr2.val[0] = vbsl_s8(vmask_rp1, vinr2.val[0], vzero);
                vinr2.val[1] = vbsl_s8(vmask_rp2, vinr2.val[1], vzero);

                vinr3.val[0] = vbsl_s8(vmask_rp1, vinr3.val[0], vzero);
                vinr3.val[1] = vbsl_s8(vmask_rp2, vinr3.val[1], vzero);

                vinr4.val[0] = vbsl_s8(vmask_rp1, vinr4.val[0], vzero);
                vinr4.val[1] = vbsl_s8(vmask_rp2, vinr4.val[1], vzero);

                int8x8_t tmp0 = vext_s8(vzero, vinr0.val[1], 7); //c = -1 1 3 5 7 9 11 13
                int8x8_t tmp1 = vext_s8(vzero, vinr1.val[1], 7); //c = -1 1 3 5 7 9 11 13
                int8x8_t tmp2 = vext_s8(vzero, vinr2.val[1], 7); //c = -1 1 3 5 7 9 11 13
                int8x8_t tmp3 = vext_s8(vzero, vinr3.val[1], 7); //c = -1 1 3 5 7 9 11 13
                int8x8_t tmp4 = vext_s8(vzero, vinr4.val[1], 7); //c = -1 1 3 5 7 9 11 13

                //r0
                int16x8_t voutr01 = vmull_s8(vinr0.val[0], wr01);//a = 0 2 4 6 8 10 12 14
                int16x8_t voutr02 = vmull_s8(vinr0.val[1], wr02);// b = 1 3 5 7 9 11 13 15
                int16x8_t voutr00 = vmull_s8(tmp0, wr00); //c = -1 1 3 5 7 9 11 13

                int16x8_t voutr11 = vmull_s8(vinr2.val[0], wr01);//a = 0 2 4 6 8 10 12 14
                int16x8_t voutr12 = vmull_s8(vinr2.val[1], wr02);// b = 1 3 5 7 9 11 13 15
                int16x8_t voutr10 = vmull_s8(tmp2, wr00); //c = -1 1 3 5 7 9 11 13

                //r1
                voutr01 = vmlal_s8(voutr01, vinr1.val[0], wr11);//a = 0 2 4 6 8 10 12 14
                voutr02 = vmlal_s8(voutr02, vinr1.val[1], wr12);// b = 1 3 5 7 9 11 13 15
                voutr00 = vmlal_s8(voutr00, tmp1, wr10); //c = -1 1 3 5 7 9 11 13

                voutr11 = vmlal_s8(voutr11, vinr3.val[0], wr11);//a = 0 2 4 6 8 10 12 14
                voutr12 = vmlal_s8(voutr12, vinr3.val[1], wr12);// b = 1 3 5 7 9 11 13 15
                voutr10 = vmlal_s8(voutr10, tmp3, wr10); //c = -1 1 3 5 7 9 11 13

                //r2
                voutr01 = vmlal_s8(voutr01, vinr2.val[0], wr21);//a = 0 2 4 6 8 10 12 14
                voutr02 = vmlal_s8(voutr02, vinr2.val[1], wr22);// b = 1 3 5 7 9 11 13 15
                voutr00 = vmlal_s8(voutr00, tmp2, wr20); //c = -1 1 3 5 7 9 11 13

                voutr11 = vmlal_s8(voutr11, vinr4.val[0], wr21);//a = 0 2 4 6 8 10 12 14
                voutr12 = vmlal_s8(voutr12, vinr4.val[1], wr22);// b = 1 3 5 7 9 11 13 15
                voutr10 = vmlal_s8(voutr10, tmp4, wr20); //c = -1 1 3 5 7 9 11 13

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr01));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr01));
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr11));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr11));

                int32x4_t vdata0 = vld1q_s32(doutr0);
                int32x4_t vdata0_1 = vld1q_s32(doutr0 + 4);

                int32x4_t vdata1 = vld1q_s32(doutr1);
                int32x4_t vdata1_1 = vld1q_s32(doutr1 + 4);

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr02));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr02));
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr12));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr12));

                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;

                voutr0 = vaddw_s16(voutr0, vget_low_s16(voutr00));
                voutr0_1 = vaddw_s16(voutr0_1, vget_high_s16(voutr00));
                voutr1 = vaddw_s16(voutr1, vget_low_s16(voutr10));
                voutr1_1 = vaddw_s16(voutr1_1, vget_high_s16(voutr10));

                dr4 = dr3 + w_in;

                voutr0 = vbslq_s32(vmask_result1, voutr0, vdata0);
                voutr0_1 = vbslq_s32(vmask_result2, voutr0_1, vdata0_1);
                voutr1 = vbslq_s32(vmask_result1, voutr1, vdata1);
                voutr1_1 = vbslq_s32(vmask_result2, voutr1_1, vdata1_1);

                voutr0 = vmaxq_s32(voutr0, vzero_32);
                voutr0_1 = vmaxq_s32(voutr0_1, vzero_32);
                voutr1 = vmaxq_s32(voutr1, vzero_32);
                voutr1_1 = vmaxq_s32(voutr1_1, vzero_32);

                vst1q_s32(doutr0, voutr0);
                vst1q_s32(doutr0 + 4, voutr0_1);
                vst1q_s32(doutr1, voutr1);
                vst1q_s32(doutr1 + 4, voutr1_1);

                dout_ptr += 2 * w_out;

            }
        }
    }
}

#else
//w_in > 8
void conv_depthwise_3x3s1p1_bias_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {
    // printf("3x3s1 mult height \n");
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 7) >> 3;
    int tile_h = (h_out + 1) >> 1;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(w_in - 7 - (cnt_col << 3));

    int size_pad_bottom = h_out % 2;

    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
    // uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned char vmask[16];
    vst1q_u8(vmask, vmask_rp);

    unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int *doutr0 = nullptr;
            int *doutr1 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;
            const signed char *dr3 = dr2 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;
            const signed char *din_ptr3 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                unsigned int* rst_mask = rmask;

                // printf("rst_msk: %d, %d, %d, %d, %d, %d, %d, %d \n", rst_mask[0], rst_mask[1], rst_mask[2], rst_mask[3], rst_mask[4], rst_mask[5],rst_mask[6],rst_mask[7]);
                // printf("vmask: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask[0], vmask[1], vmask[2], vmask[3], vmask[4], vmask[5],vmask[6],vmask[7]);
                // printf("vmask: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask[8], vmask[9], vmask[10], vmask[11], vmask[12], vmask[13],vmask[14],vmask[15]);

                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr3;
                    dr3 = dr2 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr3;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                }
                //! process bottom pad
                if (i + 3 > h_in) {
                    switch (i + 3 - h_in) {
                        case 3:
                            din_ptr1 = zero_ptr;
                        case 2:
                            din_ptr2 = zero_ptr;
                        case 1:
                            din_ptr3 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 2 > h_out) {
                    doutr1 = write_ptr;
                    // switch (i + 2 - h_out) {
                    //     case 1:
                    //         doutr1 = write_ptr;
                    //     default:
                    //         break;
                    // }
                }
                int cnt = cnt_col;
                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vmov.u32 d11, #0                   @ zero\n"
                    //out0
                    "vdup.32 q8, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                            @ and \n" //q9 = vbias
                    //out1
                    "vdup.32 q10, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                            @ and \n" //q9 = vbias

                    //r0
                    "vmull.s8 q12, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678

                    "vld1.8 {d12-d13}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vdup.s8     d5, d0[3]               @ d5 = w10, w10, w00, w00\n"
                    "vdup.s8     d6, d0[4]               @ d6 = w11, w11, w01, w01\n"

                    "vmlal.s8 q12, d30, d2                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                    "vdup.s8     d7, d0[5]               @ d7 = w12, w12\n"
                    "add %[din_ptr0], #7                   @add \n"
                    "add %[din_ptr1], #7                   @add \n"

                    "vmlal.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d3                 @ out1 = din1 * w01 \n" // q13 = d12 * w01
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d12, d6                 @ out0 = din1 * w11 \n" // q12 = d12 * w11

                    "vld1.8 {d12-d13}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vdup.s8     d8, d0[6]               @ d8 = w20, w00, w00, w00\n"
                    "vdup.s8     d9, d0[7]               @ d9 = w21, w01, w01, w01\n"
                    "vdup.s8     d10, d1[0]               @ d10 = w22, w02, w02, w02\n"

                    "vmlal.s8 q13, d30, d2                 @ out1 += din1 * w00 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d5                 @ out0 += din1 * w10 \n" // q12 += d10 * w00

                    "add %[din_ptr2], #7                   @add \n"
                    "add %[din_ptr3], #7                   @add \n"

                    "vmlal.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #1          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d14, d6                 @ out1 = din2 * w11 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d9                 @ out1 = din2 * w21 \n" // q13 = d12 * w01

                    "vmlal.s8 q13, d30, d5                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d8                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d12, d9                 @ out1 = din3 * w21 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"

                    "vmlal.s8 q13, d30, d8                 @ out1 += din3 * w20 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"

                    "vmlal.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "cmp %[cnt], #1                                 \n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"
                    "blt 1f                                         \n"

                //mid
                    "2:                                          \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    //out0
                    "vdup.32 q8, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                            @ and \n" //q9 = vbias
                    //out1
                    "vdup.32 q10, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                            @ and \n" //q9 = vbias

                     //r0
                    "vmull.s8 q12, d12, d2                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vext.8     d30, d12, d13, #1     @ ext \n" //d10 = 12345678
                    "vext.8     d31, d12, d13, #2          @ ext \n" //d11 = 23456789

                    "vld1.8 {d12-d13}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q12, d30, d3                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                    "add %[din_ptr0], #8                   @add \n"
                    "add %[din_ptr1], #8                   @add \n"

                    "vmlal.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8     d30, d12, d13, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          @ ext \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d2                 @ out1 = din1 * w01 \n" // q13 = d12 * w01
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d12, d5                 @ out0 = din1 * w11 \n" // q12 = d12 * w11

                    "vld1.8 {d12-d13}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d30, d3                 @ out1 += din1 * w00 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d6                 @ out0 += din1 * w10 \n" // q12 += d10 * w00

                    "add %[din_ptr2], #8                   @add \n"
                    "add %[din_ptr3], #8                   @add \n"

                    "vmlal.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d14, d5                 @ out1 = din2 * w11 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d8                 @ out1 = din2 * w21 \n" // q13 = d12 * w01

                    "vmlal.s8 q13, d30, d6                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d12, d13, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d12, d8                 @ out1 = din3 * w21 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"

                    "vmlal.s8 q13, d30, d9                 @ out1 += din3 * w20 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"

                    "vmlal.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "subs %[cnt], #1                                \n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"
                    "bne  2b                                        \n"
                //right
                    "1:                                          \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    //out0
                    "vdup.32 q8, %[bias]                 @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                 @ and \n" //q9 = vbias
                    //out1
                    "vdup.32 q10, %[bias]                @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                @ and \n" //q9 = vbias

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"
                    "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                     //r0
                    "vmull.s8 q12, d12, d2                 @ out0 = din0 * w00 \n" // q12 = d12 * w01
                    "vext.8 d30, d12, d13, #1               @ ext \n" //d10 = 12345678
                    "vext.8 d31, d12, d13, #2               @ ext \n" //d11 = 23456789

                    "vld1.8 {d12-d13}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                    "vmlal.s8 q12, d30, d3                 @ out0 += din0 * w01 \n" // q12 += d10 * w00

                    "vmlal.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8 d30, d14, d15, #1           @ ext \n" //d10 = 00123456
                    "vext.8 d31, d14, d15, #2          @ ext \n" //d11 = 12345678

                    "vmull.s8 q13, d14, d2                 @ out1 = din1 * w00 \n" // q13 = d12 * w01
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d14, d5                 @ out0 = din1 * w10 \n" // q12 = d12 * w11

                    "vld1.8 {d14-d15}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vbif.8 d12, d11, d28                 @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29                 @ bit select, deal with right pad\n"

                    "vmlal.s8 q13, d30, d3                 @ out1 += din1 * w01 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d6                 @ out0 += din1 * w11 \n" // q12 += d10 * w00

                    "vmlal.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8 d30, d12, d13, #1               @ ext \n" //d10 = 00123456
                    "vext.8 d31, d12, d13, #2               @ ext \n" //d11 = 12345678

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d12, d5                 @ out1 = din2 * w10 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d12, d8                 @ out1 = din2 * w20 \n" // q13 = d12 * w01

                    "vbif.8 d14, d11, d28                     @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29                     @ bit select, deal with right pad\n"

                    "vmlal.s8 q13, d30, d6                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vld1.32 {d28-d29}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d12-d13}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d14, d8                 @ out1 = din3 * w20 \n" // q13 = d12 * w01
                    "sub %[dout_ptr1], #16                  @ sub \n"
                    "vld1.32 {d14-d15}, [%[dout_ptr2]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d24-d25}, [%[dout_ptr2]]     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d30, d9                 @ out1 += din3 * w21 \n" // q13 += d10 * w00
                    "vbif q8, q14, q1                   @ bit select, deal with right pad\n"
                    "vbif q9, q6, q2                    @ bit select, deal with right pad\n"
                    "sub %[dout_ptr2], #16                  @ sub \n"

                    "vmlal.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"
                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vbif q10, q7, q1        @ bit select, deal with right pad\n"
                    "vbif q11, q12, q2       @ bit select, deal with right pad\n"

                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
                dout_ptr += 2 * w_out;
            }

        }
    }
}
//w_in <= 8
void conv_depthwise_3x3s1p1_bias_s_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {
    // printf("3x3s1 mult height \n");
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_h = (h_out + 1) >> 1;

    unsigned int size_pad_right = (unsigned int)(w_in);

    uint8x8_t vmask_rp = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    // uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned char vmask[8];
    vst1_u8(vmask, vmask_rp);

    unsigned int rst_remain = (unsigned int)w_out;
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int *doutr0 = nullptr;
            int *doutr1 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;
            const signed char *dr3 = dr2 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;
            const signed char *din_ptr3 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                unsigned int* rst_mask = rmask;

                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr3;
                    dr3 = dr2 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr3;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                }
                //! process bottom pad
                if (i + 3 > h_in) {
                    switch (i + 3 - h_in) {
                        case 3:
                            din_ptr1 = zero_ptr;
                        case 2:
                            din_ptr2 = zero_ptr;
                        case 1:
                            din_ptr3 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 2 > h_out) {
                    doutr1 = write_ptr;
                    // switch (i + 2 - h_out) {
                    //     case 1:
                    //         doutr1 = write_ptr;
                    //     default:
                    //         break;
                    // }
                }
                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"
                    "vld1.8 {d28}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d12}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d13}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"

                    "vmov.u32 d11, #0                   @ zero\n"
                    //out0
                    "vdup.32 q8, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                            @ and \n" //q9 = vbias

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d28        @ bit select, deal with right pad\n"
                    "vld1.8 {d14}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d15}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    //out1
                    "vdup.32 q10, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                            @ and \n" //q9 = vbias

                    //r0
                    "vmull.s8 q12, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vext.8 d30, d11, d12, #7           @ ext \n" //d10 = 00123456
                    "vext.8 d31, d12, d11, #1          @ ext \n" //d11 = 12345678

                    "vdup.s8 d5, d0[3]               @ d5 = w10, w10, w00, w00\n"
                    "vdup.s8 d6, d0[4]               @ d6 = w11, w11, w01, w01\n"

                    "vmlal.s8 q12, d30, d2                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                    "vdup.s8 d7, d0[5]               @ d7 = w12, w12\n"
                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d28        @ bit select, deal with right pad\n"

                    "vmlal.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8     d30, d11, d13, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d13, d11, #1          @ ext \n" //d11 = 12345678
                    "vmull.s8 q13, d13, d3                 @ out1 = din1 * w01 \n" // q13 = d12 * w01
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d13, d6                 @ out0 = din1 * w11 \n" // q12 = d12 * w11

                    "vdup.s8 d8, d0[6]               @ d8 = w20, w00, w00, w00\n"
                    "vdup.s8 d9, d0[7]               @ d9 = w21, w01, w01, w01\n"

                    "vmlal.s8 q13, d30, d2                 @ out1 += din1 * w00 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d5                 @ out0 += din1 * w10 \n" // q12 += d10 * w00

                    "vdup.s8 d10, d1[0]               @ d10 = w22, w02, w02, w02\n"
                    "vld1.32 {d28-d29}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d12-d13}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d11, #1          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d14, d6                 @ out1 = din2 * w11 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d9                 @ out1 = din2 * w21 \n" // q13 = d12 * w01

                    "sub %[dout_ptr1], #16                  @ sub \n"
                    "vmlal.s8 q13, d30, d5                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d8                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d11, d15, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d15, d11, #1          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d15, d9                 @ out1 = din3 * w21 \n" // q13 = d12 * w01

                    "vld1.32 {d6-d7}, [%[dout_ptr2]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d14-d15}, [%[dout_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d30, d8                 @ out1 += din3 * w20 \n" // q13 += d10 * w00

                    "vbif q8, q14, q1                   @ bit select, deal with right pad\n"
                    "vbif q9, q6, q2                    @ bit select, deal with right pad\n"

                    "vmlal.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "sub %[dout_ptr2], #16                  @ sub \n"

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"
                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vbif q10, q3, q1                   @ bit select, deal with right pad\n"
                    "vbif q11, q7, q2                    @ bit select, deal with right pad\n"

                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                     "q11", "q12", "q13", "q14", "q15"
                );
                dout_ptr += 2 * w_out;
            }

        }
    }
}

//1 line w_in > 16
void conv_depthwise_3x3s2p1_bias_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {

   // printf("3x3s2 mult height \n");
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_out;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 15) >> 4;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(w_in - 15 - (cnt_col << 4));
    if (size_pad_right == 17){
        size_pad_right = 0;
        cnt_col++;
    }

    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
    unsigned char vmask[16];
    vst1q_u8(vmask, vmask_rp);

    unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

     for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

            int *doutr0 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;

                doutr0 = dout_ptr;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr1 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr2 + w_in;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 2 > h_in) {
                    switch (i + 2 - h_in) {
                        case 2:
                            din_ptr1 = zero_ptr;
                        case 1:
                            din_ptr2 = zero_ptr;
                        default:
                            break;
                    }
                }
                unsigned int* rst_mask = rmask;
                int cnt = cnt_col;
                //prefetch input
                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6
                    "vmov.u32 d11, #0                   @ zero\n"

                    "vdup.s8     d5, d0[3]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d6, d0[4]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d7, d0[5]               @ d4 = w02, w02, w02, w02\n"

                    "vext.8  d18, d11, d13, #7     @ ext \n" //d16 = -1 1 3 5
                    "vext.8  d19, d11, d15, #7     @ ext \n" //d17 = -1 1 3 5
                    "vext.8  d20, d11, d17, #7     @ ext \n" //d18 = -1 1 3 5

                    //r0
                    "vmull.s8 q13, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vmull.s8 q14, d13, d4                 @ out1 = din0 * w02 \n" // q12 = d12 * w02
                    "vmull.s8 q15, d18, d2                 @ out2 = din0 * w00 \n" // q12 = d12 * w02

                    "vdup.s8 d8, d0[6]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8 d9, d0[7]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8 d10, d1[0]               @ d4 = w02, w02, w02, w02\n"

                    //r1
                    "vmlal.s8 q13, d14, d6                 @ out0 += din1 * w11 \n" // q12 = d12 * w11
                    "vmlal.s8 q14, d15, d7                 @ out1 += din1 * w12 \n" // q12 = d12 * w11
                    "vmlal.s8 q15, d19, d5                 @ out2 += din1 * w10 \n" // q12 = d12 * w11

                    //out0
                    "vdup.32 q11, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                            @ and \n" //q9 = vbias

                    //r2
                    "vmlal.s8 q13, d16, d9                 @ out0 += din1 * w21 \n" // q12 = d12 * w11
                    "vmlal.s8 q14, d17, d10                 @ out1 += din1 * w22 \n" // q12 = d12 * w11
                    "vmlal.s8 q15, d20, d8                 @ out2 += din1 * w20 \n" // q12 = d12 * w11

                    "add %[din_ptr0], #15                   @add \n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "add %[din_ptr1], #15                   @add \n"

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "add %[din_ptr2], #15                   @add \n"

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"
                    "cmp %[cnt], #1                                 \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    "blt 1f                                         \n"

                //mid
                    "2:                                              \n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]!    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]!    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]!    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6

                    "vld1.8 {d21}, [%[din_ptr0]]    @ load din00= 16 17\n" //d10 = 0 2 4 6
                    "vld1.8 {d22}, [%[din_ptr1]]    @ load din00= 16 17\n" //d12 = 0 2 4 6
                    "vld1.8 {d23}, [%[din_ptr2]]    @ load din00= 16 17\n" //d14 = 0 2 4 6

                    "vext.8  d18, d12, d21, #1     @ ext din00 = 2 4 6 8\n" //d16 = 2 4 6 8
                    "vext.8  d19, d14, d22, #1     @ ext \n" //d17 = 2 4 6 8
                    "vext.8  d20, d16, d23, #1     @ ext \n" //d18 = 2 4 6 8

                    //r0
                    "vmull.s8 q13, d12, d2                 @ out0 = din0 * w00 \n" // q12 = 0 2 4 6
                    "vmull.s8 q14, d13, d3                 @ out1 = din0 * w01 \n" // q12 = 1 3 5 7
                    "vmull.s8 q15, d18, d4                 @ out2 = din0 * w02 \n" // q12 = 2 4 6 8

                    //out0
                    "vdup.32 q11, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                            @ and \n" //q9 = vbias

                    //r1
                    "vmlal.s8 q13, d14, d5                 @ out0 += din1 * w10 \n" // q12 = 0 2 4 6
                    "vmlal.s8 q14, d15, d6                 @ out1 += din1 * w11 \n" // q12 = 1 3 5 7
                    "vmlal.s8 q15, d19, d7                 @ out2 += din1 * w12 \n" // q12 = 2 4 6 8

                    //r2
                    "vmlal.s8 q13, d16, d8                 @ out0 += din1 * w20 \n" // q12 = 0 2 4 6
                    "vmlal.s8 q14, d17, d9                 @ out1 += din1 * w21 \n" // q12 = 1 3 5 7
                    "vmlal.s8 q15, d20, d10                 @ out2 += din1 * w22 \n" // q12 = 2 4 6 8

                    // "add %[din_ptr0], #16                   @add \n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    // "add %[din_ptr1], #16                   @add \n"

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)
                    // "add %[din_ptr2], #16                   @add \n"

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"

                    "subs %[cnt], #1                                \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    "bne  2b                                        \n"
                //right
                    "1:                                              \n"
                    "cmp %[size_pad_right], #1                       \n"
                    "blt 3f                                         \n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]!    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]!    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]!    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6
                    "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    //out0
                    "vdup.32 q11, %[bias]                 @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                 @ and \n" //q9 = vbias

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                    "vext.8  d18, d12, d11, #1     @ ext din00 = 2 4 6 8\n" //d16 = -1 1 3 5
                    "vext.8  d19, d14, d11, #1     @ ext \n" //d17 = -1 1 3 5
                    "vext.8  d20, d16, d11, #1     @ ext \n" //d18 = -1 1 3 5

                    //r0
                    "vmull.s8 q13, d12, d2                 @ out0 = din0 * w00 \n" // q12 = 0 2 4 6
                    "vmull.s8 q14, d13, d3                 @ out1 = din0 * w01 \n" // q12 = 1 3 5 7
                    "vmull.s8 q15, d18, d4                 @ out2 = din0 * w02 \n" // q12 = 2 4 6 8

                    //r1
                    "vmlal.s8 q13, d14, d5                 @ out0 += din1 * w11 \n" // q12 = 0 2 4 6
                    "vmlal.s8 q14, d15, d6                 @ out1 += din1 * w12 \n" // q12 = 1 3 5 7
                    "vmlal.s8 q15, d19, d7                 @ out2 += din1 * w10 \n" // q12 = 2 4 6 8

                    "vld1.32 {d12-d13}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d14-d15}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    //r2
                    "vmlal.s8 q13, d16, d8                 @ out0 += din1 * w11 \n" // q12 = 0 2 4 6
                    "vmlal.s8 q14, d17, d9                 @ out1 += din1 * w12 \n" // q12 = 1 3 5 7
                    "vmlal.s8 q15, d20, d10                 @ out2 += din1 * w10 \n" // q12 = 2 4 6 8

                    "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "sub %[dout_ptr1], #16                  @ sub \n"

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vbif q11, q6, q1        @ bit select, deal with right pad\n"
                    "vbif q12, q7, q2       @ bit select, deal with right pad\n"

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    "3:                                             \n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [dout_ptr1] "+r"(doutr0), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), [size_pad_right] "r" (size_pad_right)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"
                );
                dout_ptr += w_out;

            }
        }
    }
}
//w_in <= 16
void conv_depthwise_3x3s2p1_bias_s_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {

    // printf("3x3s2 mult height \n");
    //! pad is done implicit
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_out;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    unsigned int size_pad_right = (unsigned int)(w_in);

    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
    unsigned char vmask[16];
    vst1q_u8(vmask, vmask_rp);

    unsigned int rst_remain = (unsigned int) w_out;
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

     for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

            int *doutr0 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;

                doutr0 = dout_ptr;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr1 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr2 + w_in;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 2 > h_in) {
                    switch (i + 2 - h_in) {
                        case 2:
                            din_ptr1 = zero_ptr;
                        case 1:
                            din_ptr2 = zero_ptr;
                        default:
                            break;
                    }
                }
                unsigned int* rst_mask = rmask;
                //prefetch input
                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6
                    "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vmov.u32 d11, #0                   @ zero\n"

                    "vdup.s8     d5, d0[3]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d6, d0[4]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d7, d0[5]               @ d4 = w02, w02, w02, w02\n"

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                    "vext.8  d18, d11, d13, #7     @ ext \n" //d16 = -1 1 3 5
                    "vext.8  d19, d11, d15, #7     @ ext \n" //d17 = -1 1 3 5
                    "vext.8  d20, d11, d17, #7     @ ext \n" //d18 = -1 1 3 5

                    "pld [%[dout_ptr1]]                @ preload data\n"

                    //r0
                    "vmull.s8 q13, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vmull.s8 q14, d13, d4                 @ out1 = din0 * w02 \n" // q12 = d12 * w02
                    "vmull.s8 q15, d18, d2                 @ out2 = din0 * w00 \n" // q12 = d12 * w02

                    "vdup.s8 d8, d0[6]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8 d9, d0[7]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8 d10, d1[0]               @ d4 = w02, w02, w02, w02\n"

                    //r1
                    "vmlal.s8 q13, d14, d6                 @ out0 += din1 * w11 \n" // q12 = d12 * w11
                    "vmlal.s8 q14, d15, d7                 @ out1 += din1 * w12 \n" // q12 = d12 * w11
                    "vmlal.s8 q15, d19, d5                 @ out2 += din1 * w10 \n" // q12 = d12 * w11

                    "vld1.32 {d12-d13}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d14-d15}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    //out0
                    "vdup.32 q11, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                            @ and \n" //q9 = vbias

                    //r2
                    "vmlal.s8 q13, d16, d9                 @ out0 += din1 * w21 \n" // q12 = d12 * w11
                    "vmlal.s8 q14, d17, d10                 @ out1 += din1 * w22 \n" // q12 = d12 * w11
                    "vmlal.s8 q15, d20, d8                 @ out2 += din1 * w20 \n" // q12 = d12 * w11

                    "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "sub %[dout_ptr1], #16                  @ sub \n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vbif q11, q6, q1        @ bit select, deal with right pad\n"
                    "vbif q12, q7, q2       @ bit select, deal with right pad\n"

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [dout_ptr1] "+r"(doutr0), [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), [size_pad_right] "r" (size_pad_right)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"
                );
                dout_ptr += w_out;

            }
        }
    }
}

//w_in > 8
void conv_depthwise_3x3s1p1_bias_relu_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {
    // printf("3x3s1 mult height \n");
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 7) >> 3;
    int tile_h = (h_out + 1) >> 1;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(w_in - 7 - (cnt_col << 3));

    int size_pad_bottom = h_out % 2;

    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
    // uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned char vmask[16];
    vst1q_u8(vmask, vmask_rp);

    unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int *doutr0 = nullptr;
            int *doutr1 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;
            const signed char *dr3 = dr2 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;
            const signed char *din_ptr3 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                unsigned int* rst_mask = rmask;

                // printf("rst_msk: %d, %d, %d, %d, %d, %d, %d, %d \n", rst_mask[0], rst_mask[1], rst_mask[2], rst_mask[3], rst_mask[4], rst_mask[5],rst_mask[6],rst_mask[7]);
                // printf("vmask: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask[0], vmask[1], vmask[2], vmask[3], vmask[4], vmask[5],vmask[6],vmask[7]);
                // printf("vmask: %d, %d, %d, %d, %d, %d, %d, %d \n", vmask[8], vmask[9], vmask[10], vmask[11], vmask[12], vmask[13],vmask[14],vmask[15]);

                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr3;
                    dr3 = dr2 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr3;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                }
                //! process bottom pad
                if (i + 3 > h_in) {
                    switch (i + 3 - h_in) {
                        case 3:
                            din_ptr1 = zero_ptr;
                        case 2:
                            din_ptr2 = zero_ptr;
                        case 1:
                            din_ptr3 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 2 > h_out) {
                    doutr1 = write_ptr;
                    // switch (i + 2 - h_out) {
                    //     case 1:
                    //         doutr1 = write_ptr;
                    //     default:
                    //         break;
                    // }
                }
                int cnt = cnt_col;
                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vmov.u32 d11, #0                   @ zero\n"
                    //out0
                    "vdup.32 q8, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                            @ and \n" //q9 = vbias
                    //out1
                    "vdup.32 q10, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                            @ and \n" //q9 = vbias

                    //r0
                    "vmull.s8 q12, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678

                    "vld1.8 {d12-d13}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vdup.s8     d5, d0[3]               @ d5 = w10, w10, w00, w00\n"
                    "vdup.s8     d6, d0[4]               @ d6 = w11, w11, w01, w01\n"

                    "vmlal.s8 q12, d30, d2                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                    "vdup.s8     d7, d0[5]               @ d7 = w12, w12\n"
                    "add %[din_ptr0], #7                   @add \n"
                    "add %[din_ptr1], #7                   @add \n"

                    "vmlal.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d3                 @ out1 = din1 * w01 \n" // q13 = d12 * w01
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d12, d6                 @ out0 = din1 * w11 \n" // q12 = d12 * w11

                    "vld1.8 {d12-d13}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vdup.s8     d8, d0[6]               @ d8 = w20, w00, w00, w00\n"
                    "vdup.s8     d9, d0[7]               @ d9 = w21, w01, w01, w01\n"
                    "vdup.s8     d10, d1[0]               @ d10 = w22, w02, w02, w02\n"

                    "vmlal.s8 q13, d30, d2                 @ out1 += din1 * w00 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d5                 @ out0 += din1 * w10 \n" // q12 += d10 * w00

                    "add %[din_ptr2], #7                   @add \n"
                    "add %[din_ptr3], #7                   @add \n"

                    "vmlal.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #1          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d14, d6                 @ out1 = din2 * w11 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d9                 @ out1 = din2 * w21 \n" // q13 = d12 * w01

                    "vmlal.s8 q13, d30, d5                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d8                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d11, d12, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          @ ext \n" //d11 = 12345678
                    "vmov.u32 q0, #0                         @ mov \n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d12, d9                 @ out1 = din3 * w21 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "vmax.s32 q8, q8, q0              @ max \n"
                    "vmax.s32 q9, q9, q0              @ max \n"

                    "vmlal.s8 q13, d30, d8                 @ out1 += din3 * w20 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"

                    "vmlal.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmax.s32 q10, q10, q0              @ max \n"
                    "vmax.s32 q11, q11, q0              @ max \n"

                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "cmp %[cnt], #1                                 \n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"
                    "blt 1f                                         \n"

                //mid
                    "2:                                          \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    //out0
                    "vdup.32 q8, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                            @ and \n" //q9 = vbias
                    //out1
                    "vdup.32 q10, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                            @ and \n" //q9 = vbias

                     //r0
                    "vmull.s8 q12, d12, d2                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vext.8     d30, d12, d13, #1     @ ext \n" //d10 = 12345678
                    "vext.8     d31, d12, d13, #2          @ ext \n" //d11 = 23456789

                    "vld1.8 {d12-d13}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q12, d30, d3                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                    "add %[din_ptr0], #8                   @add \n"
                    "add %[din_ptr1], #8                   @add \n"

                    "vmlal.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8     d30, d12, d13, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          @ ext \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d2                 @ out1 = din1 * w01 \n" // q13 = d12 * w01
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d12, d5                 @ out0 = din1 * w11 \n" // q12 = d12 * w11

                    "vld1.8 {d12-d13}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d30, d3                 @ out1 += din1 * w00 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d6                 @ out0 += din1 * w10 \n" // q12 += d10 * w00

                    "add %[din_ptr2], #8                   @add \n"
                    "add %[din_ptr3], #8                   @add \n"

                    "vmlal.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d14, d5                 @ out1 = din2 * w11 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d8                 @ out1 = din2 * w21 \n" // q13 = d12 * w01

                    "vmlal.s8 q13, d30, d6                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d12, d13, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d12, d8                 @ out1 = din3 * w21 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "vmax.s32 q8, q8, q0              @ max \n"
                    "vmax.s32 q9, q9, q0              @ max \n"

                    "vmlal.s8 q13, d30, d9                 @ out1 += din3 * w20 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"

                    "vmlal.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmax.s32 q10, q10, q0              @ max \n"
                    "vmax.s32 q11, q11, q0              @ max \n"

                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "subs %[cnt], #1                                \n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"
                    "bne  2b                                        \n"
                //right
                    "1:                                          \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    //out0
                    "vdup.32 q8, %[bias]                 @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                 @ and \n" //q9 = vbias
                    //out1
                    "vdup.32 q10, %[bias]                @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                @ and \n" //q9 = vbias

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"
                    "vld1.8 {d14-d15}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                     //r0
                    "vmull.s8 q12, d12, d2                 @ out0 = din0 * w00 \n" // q12 = d12 * w01
                    "vext.8 d30, d12, d13, #1               @ ext \n" //d10 = 12345678
                    "vext.8 d31, d12, d13, #2               @ ext \n" //d11 = 23456789

                    "vld1.8 {d12-d13}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                    "vmlal.s8 q12, d30, d3                 @ out0 += din0 * w01 \n" // q12 += d10 * w00

                    "vmlal.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8 d30, d14, d15, #1           @ ext \n" //d10 = 00123456
                    "vext.8 d31, d14, d15, #2          @ ext \n" //d11 = 12345678

                    "vmull.s8 q13, d14, d2                 @ out1 = din1 * w00 \n" // q13 = d12 * w01
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d14, d5                 @ out0 = din1 * w10 \n" // q12 = d12 * w11

                    "vld1.8 {d14-d15}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vbif.8 d12, d11, d28                 @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29                 @ bit select, deal with right pad\n"

                    "vmlal.s8 q13, d30, d3                 @ out1 += din1 * w01 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d6                 @ out0 += din1 * w11 \n" // q12 += d10 * w00

                    "vmlal.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8 d30, d12, d13, #1               @ ext \n" //d10 = 00123456
                    "vext.8 d31, d12, d13, #2               @ ext \n" //d11 = 12345678

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d12, d5                 @ out1 = din2 * w10 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d12, d8                 @ out1 = din2 * w20 \n" // q13 = d12 * w01

                    "vbif.8 d14, d11, d28                     @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29                     @ bit select, deal with right pad\n"

                    "vmlal.s8 q13, d30, d6                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vld1.32 {d28-d29}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d12-d13}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d14, d15, #1     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d14, d8                 @ out1 = din3 * w20 \n" // q13 = d12 * w01
                    "vld1.32 {d14-d15}, [%[dout_ptr2]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d24-d25}, [%[dout_ptr2]]     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vmax.s32 q8, q8, q0              @ max \n"
                    "vmax.s32 q9, q9, q0              @ max \n"

                    "vmlal.s8 q13, d30, d9                 @ out1 += din3 * w21 \n" // q13 += d10 * w00
                    "vbif q8, q14, q1                   @ bit select, deal with right pad\n"
                    "vbif q9, q6, q2                    @ bit select, deal with right pad\n"
                    "sub %[dout_ptr1], #16                  @ sub \n"
                    "sub %[dout_ptr2], #16                  @ sub \n"

                    "vmlal.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"
                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmax.s32 q10, q10, q0              @ max \n"
                    "vmax.s32 q11, q11, q0              @ max \n"

                    "vbif q10, q7, q1        @ bit select, deal with right pad\n"
                    "vbif q11, q12, q2       @ bit select, deal with right pad\n"

                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
                dout_ptr += 2 * w_out;
            }

        }
    }
}
//w_in <= 8
void conv_depthwise_3x3s1p1_bias_s_relu_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {
    // printf("3x3s1 mult height \n");
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_in;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_h = (h_out + 1) >> 1;

    unsigned int size_pad_right = (unsigned int)(w_in);

    uint8x8_t vmask_rp = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
    // uint8x8_t vmask_rp2 = vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
    unsigned char vmask[8];
    vst1_u8(vmask, vmask_rp);

    unsigned int rst_remain = (unsigned int)w_out;
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

    for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int *doutr0 = nullptr;
            int *doutr1 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;
            const signed char *dr3 = dr2 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;
            const signed char *din_ptr3 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                unsigned int* rst_mask = rmask;

                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    din_ptr3 = dr2;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr3;
                    dr3 = dr2 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr3;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                }
                //! process bottom pad
                if (i + 3 > h_in) {
                    switch (i + 3 - h_in) {
                        case 3:
                            din_ptr1 = zero_ptr;
                        case 2:
                            din_ptr2 = zero_ptr;
                        case 1:
                            din_ptr3 = zero_ptr;
                        default:
                            break;
                    }
                }
                //! process bottom remain
                if (i + 2 > h_out) {
                    doutr1 = write_ptr;
                    // switch (i + 2 - h_out) {
                    //     case 1:
                    //         doutr1 = write_ptr;
                    //     default:
                    //         break;
                    // }
                }
                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "pld [%[din_ptr3]]                @ preload data\n"
                    "vld1.8 {d28}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d12}, [%[din_ptr0]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d13}, [%[din_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"

                    "vmov.u32 d11, #0                   @ zero\n"
                    //out0
                    "vdup.32 q8, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q9, %[bias]                            @ and \n" //q9 = vbias

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d28        @ bit select, deal with right pad\n"
                    "vld1.8 {d14}, [%[din_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.8 {d15}, [%[din_ptr3]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    //out1
                    "vdup.32 q10, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q11, %[bias]                            @ and \n" //q9 = vbias

                    //r0
                    "vmull.s8 q12, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vext.8 d30, d11, d12, #7           @ ext \n" //d10 = 00123456
                    "vext.8 d31, d12, d11, #1          @ ext \n" //d11 = 12345678

                    "vdup.s8 d5, d0[3]               @ d5 = w10, w10, w00, w00\n"
                    "vdup.s8 d6, d0[4]               @ d6 = w11, w11, w01, w01\n"

                    "vmlal.s8 q12, d30, d2                 @ out0 += din0 * w00 \n" // q12 += d10 * w00

                    "vdup.s8 d7, d0[5]               @ d7 = w12, w12\n"
                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d28        @ bit select, deal with right pad\n"

                    "vmlal.s8 q12, d31, d4                 @ out0 += din0 * w02 \n" // q12 += d11 * w02

                    //r1
                    "vext.8     d30, d11, d13, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d13, d11, #1          @ ext \n" //d11 = 12345678
                    "vmull.s8 q13, d13, d3                 @ out1 = din1 * w01 \n" // q13 = d12 * w01
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d13, d6                 @ out0 = din1 * w11 \n" // q12 = d12 * w11

                    "vdup.s8 d8, d0[6]               @ d8 = w20, w00, w00, w00\n"
                    "vdup.s8 d9, d0[7]               @ d9 = w21, w01, w01, w01\n"

                    "vmlal.s8 q13, d30, d2                 @ out1 += din1 * w00 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d5                 @ out0 += din1 * w10 \n" // q12 += d10 * w00

                    "vdup.s8 d10, d1[0]               @ d10 = w22, w02, w02, w02\n"
                    "vld1.32 {d28-d29}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d12-d13}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d31, d4                 @ out1 += din1 * w02 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 @ out0 += din1 * w12 \n" // q12 += d10 * w00

                    //r2
                    "vext.8     d30, d11, d14, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d14, d11, #1          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d14, d6                 @ out1 = din2 * w11 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d9                 @ out1 = din2 * w21 \n" // q13 = d12 * w01

                    "sub %[dout_ptr1], #16                  @ sub \n"
                    "vmlal.s8 q13, d30, d5                 @ out1 += din2 * w10 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d8                 @ out0 += din2 * w20 \n" // q12 += d10 * w00

                    "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d31, d7                 @ out1 += din2 * w12 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d10                 @ out0 += din2 * w22 \n" // q12 += d10 * w00

                    //r3
                    "vext.8     d30, d11, d15, #7     @ ext \n" //d10 = 00123456
                    "vext.8     d31, d15, d11, #1          @ ext \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 @addw \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 @addw \n" // out0_1 += vget_high_s16(out00)

                    "vmull.s8 q13, d15, d9                 @ out1 = din3 * w21 \n" // q13 = d12 * w01

                    "vmov.u32 q0, #0                   @ zero\n"

                    "vld1.32 {d6-d7}, [%[dout_ptr2]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d14-d15}, [%[dout_ptr2]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vmlal.s8 q13, d30, d8                 @ out1 += din3 * w20 \n" // q13 += d10 * w00

                    "vmax.s32 q8, q8, q0                    @ max \n"
                    "vmax.s32 q9, q9, q0                    @ max \n"

                    "vmlal.s8 q13, d31, d10                 @ out1 += din3 * w22 \n" // q12 += d10 * w00

                    "sub %[dout_ptr2], #16                  @ sub \n"
                    "vbif q8, q14, q1                   @ bit select, deal with right pad\n"
                    "vbif q9, q6, q2                    @ bit select, deal with right pad\n"

                    "vaddw.s16 q10, q10, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vst1.32 {d16-d17}, [%[dout_ptr1]]!         @ store\n"
                    "vst1.32 {d18-d19}, [%[dout_ptr1]]!         @ store\n"

                    "vmax.s32 q10, q10, q0                    @ max \n"
                    "vmax.s32 q11, q11, q0                    @ max \n"

                    "vbif q10, q3, q1                   @ bit select, deal with right pad\n"
                    "vbif q11, q7, q2                    @ bit select, deal with right pad\n"

                    "vst1.32 {d20-d21}, [%[dout_ptr2]]!         @ store\n"
                    "vst1.32 {d22-d23}, [%[dout_ptr2]]!         @ store\n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
                dout_ptr += 2 * w_out;
            }

        }
    }
}

void conv_depthwise_3x3s2p1_bias_relu_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {

   // printf("3x3s2 mult height \n");
    //! pad is done implicit
    const char zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_out;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 15) >> 4;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(w_in - 15 - (cnt_col << 4));
    if (size_pad_right == 17){
        size_pad_right = 0;
        cnt_col++;
    }

    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
    unsigned char vmask[16];
    vst1q_u8(vmask, vmask_rp);

    unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

     for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

            int *doutr0 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;

                doutr0 = dout_ptr;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr1 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr2 + w_in;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 2 > h_in) {
                    switch (i + 2 - h_in) {
                        case 2:
                            din_ptr1 = zero_ptr;
                        case 1:
                            din_ptr2 = zero_ptr;
                        default:
                            break;
                    }
                }
                unsigned int* rst_mask = rmask;
                int cnt = cnt_col;
                //prefetch input
                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6
                    "vmov.u32 d11, #0                   @ zero\n"

                    "vdup.s8     d5, d0[3]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d6, d0[4]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d7, d0[5]               @ d4 = w02, w02, w02, w02\n"

                    "vext.8  d18, d11, d13, #7     @ ext \n" //d16 = -1 1 3 5
                    "vext.8  d19, d11, d15, #7     @ ext \n" //d17 = -1 1 3 5
                    "vext.8  d20, d11, d17, #7     @ ext \n" //d18 = -1 1 3 5

                    //r0
                    "vmull.s8 q13, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vmull.s8 q14, d13, d4                 @ out1 = din0 * w02 \n" // q12 = d12 * w02
                    "vmull.s8 q15, d18, d2                 @ out2 = din0 * w00 \n" // q12 = d12 * w02

                    "vdup.s8 d8, d0[6]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8 d9, d0[7]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8 d10, d1[0]               @ d4 = w02, w02, w02, w02\n"

                    //r1
                    "vmlal.s8 q13, d14, d6                 @ out0 += din1 * w11 \n" // q12 = d12 * w11
                    "vmlal.s8 q14, d15, d7                 @ out1 += din1 * w12 \n" // q12 = d12 * w11
                    "vmlal.s8 q15, d19, d5                 @ out2 += din1 * w10 \n" // q12 = d12 * w11

                    //out0
                    "vdup.32 q11, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                            @ and \n" //q9 = vbias

                    //r2
                    "vmlal.s8 q13, d16, d9                 @ out0 += din1 * w21 \n" // q12 = d12 * w11
                    "vmlal.s8 q14, d17, d10                 @ out1 += din1 * w22 \n" // q12 = d12 * w11
                    "vmlal.s8 q15, d20, d8                 @ out2 += din1 * w20 \n" // q12 = d12 * w11

                    "add %[din_ptr0], #15                   @add \n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vmov.u32 q8, #0                        @ max \n" //max
                    "add %[din_ptr1], #15                   @add \n"

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "add %[din_ptr2], #15                   @add \n"

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"

                    "vmax.s32 q11, q11, q8                      @ max\n"
                    "vmax.s32 q12, q12, q8                      @ max\n"

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"
                    "cmp %[cnt], #1                                 \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    "blt 1f                                         \n"

                //mid
                    "2:                                              \n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]!    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]!    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]!    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6

                    "vld1.8 {d21}, [%[din_ptr0]]    @ load din00= 16 17\n" //d10 = 0 2 4 6
                    "vld1.8 {d22}, [%[din_ptr1]]    @ load din00= 16 17\n" //d12 = 0 2 4 6
                    "vld1.8 {d23}, [%[din_ptr2]]    @ load din00= 16 17\n" //d14 = 0 2 4 6

                    "vext.8  d18, d12, d21, #1     @ ext din00 = 2 4 6 8\n" //d16 = 2 4 6 8
                    "vext.8  d19, d14, d22, #1     @ ext \n" //d17 = 2 4 6 8
                    "vext.8  d20, d16, d23, #1     @ ext \n" //d18 = 2 4 6 8

                    //r0
                    "vmull.s8 q13, d12, d2                 @ out0 = din0 * w00 \n" // q12 = 0 2 4 6
                    "vmull.s8 q14, d13, d3                 @ out1 = din0 * w01 \n" // q12 = 1 3 5 7
                    "vmull.s8 q15, d18, d4                 @ out2 = din0 * w02 \n" // q12 = 2 4 6 8

                    //out0
                    "vdup.32 q11, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                            @ and \n" //q9 = vbias

                    //r1
                    "vmlal.s8 q13, d14, d5                 @ out0 += din1 * w10 \n" // q12 = 0 2 4 6
                    "vmlal.s8 q14, d15, d6                 @ out1 += din1 * w11 \n" // q12 = 1 3 5 7
                    "vmlal.s8 q15, d19, d7                 @ out2 += din1 * w12 \n" // q12 = 2 4 6 8

                    //r2
                    "vmlal.s8 q13, d16, d8                 @ out0 += din1 * w20 \n" // q12 = 0 2 4 6
                    "vmlal.s8 q14, d17, d9                 @ out1 += din1 * w21 \n" // q12 = 1 3 5 7
                    "vmlal.s8 q15, d20, d10                 @ out2 += din1 * w22 \n" // q12 = 2 4 6 8

                    // "add %[din_ptr0], #16                   @add \n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    // "add %[din_ptr1], #16                   @add \n"
                    "vmov.u32 q8, #0                          @ mov \n"

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)
                    // "add %[din_ptr2], #16                   @add \n"

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"

                    "vmax.s32 q11, q11, q8                      @ max\n"
                    "vmax.s32 q12, q12, q8                      @ max\n"

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"

                    "subs %[cnt], #1                                \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    "bne  2b                                        \n"
                //right
                    "1:                                              \n"
                    "cmp %[size_pad_right], #1                       \n"
                    "blt 3f                                         \n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]!    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]!    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]!    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6
                    "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    //out0
                    "vdup.32 q11, %[bias]                 @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                 @ and \n" //q9 = vbias

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                    "vext.8  d18, d12, d11, #1     @ ext din00 = 2 4 6 8\n" //d16 = -1 1 3 5
                    "vext.8  d19, d14, d11, #1     @ ext \n" //d17 = -1 1 3 5
                    "vext.8  d20, d16, d11, #1     @ ext \n" //d18 = -1 1 3 5

                    //r0
                    "vmull.s8 q13, d12, d2                 @ out0 = din0 * w00 \n" // q12 = 0 2 4 6
                    "vmull.s8 q14, d13, d3                 @ out1 = din0 * w01 \n" // q12 = 1 3 5 7
                    "vmull.s8 q15, d18, d4                 @ out2 = din0 * w02 \n" // q12 = 2 4 6 8

                    //r1
                    "vmlal.s8 q13, d14, d5                 @ out0 += din1 * w11 \n" // q12 = 0 2 4 6
                    "vmlal.s8 q14, d15, d6                 @ out1 += din1 * w12 \n" // q12 = 1 3 5 7
                    "vmlal.s8 q15, d19, d7                 @ out2 += din1 * w10 \n" // q12 = 2 4 6 8

                    "vld1.32 {d12-d13}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d14-d15}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    //r2
                    "vmlal.s8 q13, d16, d8                 @ out0 += din1 * w11 \n" // q12 = 0 2 4 6
                    "vmlal.s8 q14, d17, d9                 @ out1 += din1 * w12 \n" // q12 = 1 3 5 7
                    "vmlal.s8 q15, d20, d10                 @ out2 += din1 * w10 \n" // q12 = 2 4 6 8

                    "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "sub %[dout_ptr1], #16                  @ sub \n"
                    "vmov.u32 q8, #0                         @mov \n"
                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmax.s32 q11, q11, q8                      @ max\n"
                    "vmax.s32 q12, q12, q8                      @ max\n"

                    "vbif q11, q6, q1        @ bit select, deal with right pad\n"
                    "vbif q12, q7, q2       @ bit select, deal with right pad\n"

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    "3:                                             \n"

                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [dout_ptr1] "+r"(doutr0), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), [size_pad_right] "r" (size_pad_right)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"
                );
                dout_ptr += w_out;

            }
        }
    }
}
//w_in <= 16
void conv_depthwise_3x3s2p1_bias_s_relu_int8(int* dout, const signed char* din, \
    const signed char* weights, const int* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out, Context* ctx) {

    // printf("3x3s2 mult height \n");
    //! pad is done implicit
    //! for 4x6 convolution window
    const unsigned char right_pad_idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    //printf("conv3x3_dw start \n");
    signed char* zero_ptr = static_cast<signed char*>(ctx->get_work_space());
    memset(zero_ptr, 0, w_in * sizeof(signed char));
    int* write_ptr = (int*)ctx->get_work_space() + w_out;
    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    unsigned int size_pad_right = (unsigned int)(w_in);

    uint8x16_t vmask_rp = vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
    unsigned char vmask[16];
    vst1q_u8(vmask, vmask_rp);

    unsigned int rst_remain = (unsigned int)w_out;
    uint32x4_t vmask_result1 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
    uint32x4_t vmask_result2 = vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

    unsigned int rmask[8];
    vst1q_u32(rmask, vmask_result1);
    vst1q_u32(rmask + 4, vmask_result2);

    // printf("size_pad_right: %d, rst_remain: %d \n", size_pad_right, rst_remain);

     for (int n = 0; n < num; ++n) {
        const signed char *din_batch = din + n * ch_in * size_in_channel;
        int *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c++) {
            int* dout_ptr = dout_batch + c * size_out_channel;

            const signed char* din_ch_ptr = din_batch + c * size_in_channel;

            int bias_val = flag_bias ? bias[c] : 0;

            const signed char* wei_ptr = weights + c * w_stride;

            int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
            int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
            int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

            int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
            int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
            int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

            int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
            int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
            int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

            int *doutr0 = nullptr;

            const signed char *dr0 = din_ch_ptr;
            const signed char *dr1 = dr0 + w_in;
            const signed char *dr2 = dr1 + w_in;

            const signed char *din_ptr0 = nullptr;
            const signed char *din_ptr1 = nullptr;
            const signed char *din_ptr2 = nullptr;

            for (int i = 0; i < h_in; i += 2){
                //! process top pad pad_h = 1
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;

                doutr0 = dout_ptr;
                if (i == 0){
                    din_ptr0 = zero_ptr;
                    din_ptr1 = dr0;
                    din_ptr2 = dr1;
                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr1 + w_in;
                }else{
                    dr0 = dr2;
                    dr1 = dr2 + w_in;
                    dr2 = dr1 + w_in;
                }
                //! process bottom pad
                if (i + 2 > h_in) {
                    switch (i + 2 - h_in) {
                        case 2:
                            din_ptr1 = zero_ptr;
                        case 1:
                            din_ptr2 = zero_ptr;
                        default:
                            break;
                    }
                }
                unsigned int* rst_mask = rmask;
                //prefetch input
                //store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                @ preload data\n"
                    "pld [%[din_ptr1]]                @ preload data\n"
                    "pld [%[din_ptr2]]                @ preload data\n"
                    "vdup.s8     d2, d0[0]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d3, d0[1]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d4, d0[2]               @ d4 = w02, w02, w02, w02\n"
                    "vld2.8 {d12-d13}, [%[din_ptr0]]    @ load din00= 0 2 4 6 8\n" //d10 = 0 2 4 6
                    "vld2.8 {d14-d15}, [%[din_ptr1]]    @ load din00= 0 2 4 6 8\n" //d12 = 0 2 4 6
                    "vld2.8 {d16-d17}, [%[din_ptr2]]    @ load din00= 0 2 4 6 8\n" //d14 = 0 2 4 6
                    "vld1.8 {d28-d29}, [%[mask]]        @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vmov.u32 d11, #0                   @ zero\n"

                    "vdup.s8     d5, d0[3]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8     d6, d0[4]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8     d7, d0[5]               @ d4 = w02, w02, w02, w02\n"

                    "vbif.8 d12, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d13, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d14, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d15, d11, d29        @ bit select, deal with right pad\n"

                    "vbif.8 d16, d11, d28        @ bit select, deal with right pad\n"
                    "vbif.8 d17, d11, d29        @ bit select, deal with right pad\n"

                    "vext.8  d18, d11, d13, #7     @ ext \n" //d16 = -1 1 3 5
                    "vext.8  d19, d11, d15, #7     @ ext \n" //d17 = -1 1 3 5
                    "vext.8  d20, d11, d17, #7     @ ext \n" //d18 = -1 1 3 5

                    "pld [%[dout_ptr1]]                @ preload data\n"

                    //r0
                    "vmull.s8 q13, d12, d3                 @ out0 = din0 * w01 \n" // q12 = d12 * w01
                    "vmull.s8 q14, d13, d4                 @ out1 = din0 * w02 \n" // q12 = d12 * w02
                    "vmull.s8 q15, d18, d2                 @ out2 = din0 * w00 \n" // q12 = d12 * w02

                    "vdup.s8 d8, d0[6]               @ d2 = w00, w00, w00, w00\n"
                    "vdup.s8 d9, d0[7]               @ d3 = w01, w01, w01, w01\n"
                    "vdup.s8 d10, d1[0]               @ d4 = w02, w02, w02, w02\n"

                    //r1
                    "vmlal.s8 q13, d14, d6                 @ out0 += din1 * w11 \n" // q12 = d12 * w11
                    "vmlal.s8 q14, d15, d7                 @ out1 += din1 * w12 \n" // q12 = d12 * w11
                    "vmlal.s8 q15, d19, d5                 @ out2 += din1 * w10 \n" // q12 = d12 * w11

                    "vld1.32 {d12-d13}, [%[dout_ptr1]]!    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d14-d15}, [%[dout_ptr1]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    //out0
                    "vdup.32 q11, %[bias]                            @ and \n" //q8 = vbias
                    "vdup.32 q12, %[bias]                            @ and \n" //q9 = vbias

                    //r2
                    "vmlal.s8 q13, d16, d9                 @ out0 += din1 * w21 \n" // q12 = d12 * w11
                    "vmlal.s8 q14, d17, d10                 @ out1 += din1 * w22 \n" // q12 = d12 * w11
                    "vmlal.s8 q15, d20, d8                 @ out2 += din1 * w20 \n" // q12 = d12 * w11

                    "vld1.32 {d2-d3}, [%[rs_mask]]!     @ load din00= 0 1 2 3 4 5 6 7 8 9\n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    @ load din00= 0 1 2 3 4 5 6 7 8 9\n"

                    "sub %[dout_ptr1], #16                  @ sub \n"

                    "vaddw.s16 q11, q11, d26                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d27                 @addw \n" // out1_1 += vget_high_s16(out10)
                    "vmov.u32 q8, #0                         @ mov \n"

                    "vaddw.s16 q11, q11, d28                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d29                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vaddw.s16 q11, q11, d30                 @addw \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q12, q12, d31                 @addw \n" // out1_1 += vget_high_s16(out10)

                    "vmax.s32 q11, q11, q8                      @ max\n"
                    "vmax.s32 q12, q12, q8                      @ max\n"

                    "vbif q11, q6, q1        @ bit select, deal with right pad\n"
                    "vbif q12, q7, q2       @ bit select, deal with right pad\n"

                    "vst1.32 {d22-d23}, [%[dout_ptr1]]!         @ store\n"
                    "vst1.32 {d24-d25}, [%[dout_ptr1]]!         @ store\n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2),\
                      [dout_ptr1] "+r"(doutr0), [bias] "+r" (bias_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), [size_pad_right] "r" (size_pad_right)
                    :"q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", \
                      "q12", "q13", "q14", "q15"
                );
                dout_ptr += w_out;

            }
        }
    }
}
#endif

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE
