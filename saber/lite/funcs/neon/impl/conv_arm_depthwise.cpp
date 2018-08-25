#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

void conv_depthwise_3x3s1p1_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out);

//! for input width <= 4
void conv_depthwise_3x3s1p1_bias_s(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out);

void conv_depthwise_3x3s2p1_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out);

//! for input width <= 4
void conv_depthwise_3x3s2p1_bias_s(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out);

void conv_depthwise_3x3s1p1_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out);

//! for input width <= 4
void conv_depthwise_3x3s1p1_bias_s_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out);

void conv_depthwise_3x3s2p1_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out);

//! for input width <= 4
void conv_depthwise_3x3s2p1_bias_s_relu(float* dout, const float* din, \
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
            if (w_in > 4) {
                conv_depthwise_3x3s1p1_bias_relu(dout, din, weights, bias, flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out);
            } else {
                conv_depthwise_3x3s1p1_bias_s_relu(dout, din, weights, bias, flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out);
            }
        } else {
            if (w_in > 4) {
                conv_depthwise_3x3s1p1_bias(dout, din, weights, bias, flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out);
            } else {
                conv_depthwise_3x3s1p1_bias_s(dout, din, weights, bias, flag_bias, \
                    num, ch_in, h_in, w_in, h_out, w_out);
            }

        }
    } else { //! stride = 2
        if (flag_relu) {
            if(w_in > 7){
                conv_depthwise_3x3s2p1_bias_relu(dout, din, weights, bias, flag_bias, \
                num, ch_in, h_in, w_in, h_out, w_out);
            }else{
                conv_depthwise_3x3s2p1_bias_s_relu(dout, din, weights, bias, flag_bias, \
                num, ch_in, h_in, w_in, h_out, w_out);
            }
        } else {
            if(w_in > 7){
                conv_depthwise_3x3s2p1_bias(dout, din, weights, bias, flag_bias, \
                num, ch_in, h_in, w_in, h_out, w_out);
            }else{
                conv_depthwise_3x3s2p1_bias_s(dout, din, weights, bias, flag_bias, \
                num, ch_in, h_in, w_in, h_out, w_out);
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
//4line
void conv_depthwise_3x3s1p1_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {

   // printf("3x3s1 mult height \n");
    //! pad is done implicit
    const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    //! for 4x6 convolution window
    const unsigned int right_pad_idx[8] = {5, 4, 3, 2, 1, 0, 0, 0};

    //printf("conv3x3_dw start \n");

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 3) >> 2;
    int tile_h = (h_in + 3) >> 2;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(1 + (tile_w << 2) - w_in);
    int size_pad_bottom = (unsigned int)(1 + (tile_h << 2) - h_in);

    uint32x4_t vmask_rp1 = vcgeq_u32(vld1q_u32(right_pad_idx), vdupq_n_u32(size_pad_right));
    uint32x4_t vmask_rp2 = vcgeq_u32(vld1q_u32(right_pad_idx + 4), vdupq_n_u32(size_pad_right));
    uint32x4_t vmask_result = vcgtq_u32(vld1q_u32(right_pad_idx), vdupq_n_u32(size_pad_right));

     for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c ++) {
            float* dout_ptr = dout_batch + c * size_out_channel;

            const float* din_ch_ptr = din_batch + c * size_in_channel;

            float bias_val = flag_bias ? bias[c] : 0.f;
        
            const float* wei_ptr = weights + c * w_stride;

            float32x4_t wr0 = vld1q_f32(wei_ptr);
            float32x4_t wr1 = vld1q_f32(wei_ptr + 3);
            float32x4_t wr2 = vld1q_f32(wei_ptr + 6);

            float *doutr0 = dout_ptr;
            float *doutr1 = doutr0 + w_out;
            float *doutr2 = doutr1 + w_out;
            float *doutr3 = doutr2 + w_out;

            const float *dr0 = din_ch_ptr;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;
            const float *dr3 = dr2 + w_in;
            const float *dr4 = dr3 + w_in;
            const float *dr5 = dr4 + w_in;

            const float *din_ptr0 = dr0;
            const float *din_ptr1 = dr1;
            const float *din_ptr2 = dr2;
            const float *din_ptr3 = dr3;
            const float *din_ptr4 = dr4;
            const float *din_ptr5 = dr5;

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

            float* ptr_zero = const_cast<float*>(zero);
            float32x4_t vzero = vdupq_n_f32(0.f);

            //top
            int h = 0;
            if(1){
                float32x4_t voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                float32x4_t voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                float32x4_t voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                float32x4_t voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                //din data
                float32x4_t vinr0 = vld1q_f32(din_ptr0);
                float32x4_t vinr0_1 = vld1q_f32(din_ptr0 + 4);
                float32x4_t vinr1 = vld1q_f32(din_ptr1);
                float32x4_t vinr1_1 = vld1q_f32(din_ptr1 + 4);
                float32x4_t vinr2 = vld1q_f32(din_ptr2);
                float32x4_t vinr2_1 = vld1q_f32(din_ptr2 + 4);

                //r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr1, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr0, wr0, 1);

                float32x4_t vinr3 = vld1q_f32(din_ptr3);
                float32x4_t vinr3_1 = vld1q_f32(din_ptr3 + 4);
                float32x4_t vinr4 = vld1q_f32(din_ptr4);
                float32x4_t vinr4_1 = vld1q_f32(din_ptr4 + 4);

                //r1
                voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr2, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr1, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr1, wr0, 1);

              //  float32x4_t vinr5 = vld1q_f32(din_ptr5);
              //  float32x4_t vinr5_1 = vld1q_f32(din_ptr5 + 4);

                //r2
                voutr3 = vmlaq_laneq_f32(voutr3, vinr2, wr0, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr2, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr1, 1);

                // r0, r1, r2 shift left 2345
                float32x4_t vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                float32x4_t vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                float32x4_t vtmp2 = vextq_f32(vinr2, vinr2_1, 1);

                //r3
                voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr1, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr2, 1);

                float32x4_t vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                float32x4_t vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
               // float32x4_t vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                //r4
                voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr2, 1);

                //r0
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp0, wr0, 2);

                din_ptr0 += 3;
                prefetch(din_ptr0);

                //r1
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp1, wr0, 2);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr2, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr1, 2);

                // r0, r1, r2 shift right 0123
                vtmp0 = vextq_f32(vzero, vinr0, 3);
                vtmp1 = vextq_f32(vzero, vinr1, 3);

                //r2
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp2, wr0, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr2, 2);

                din_ptr1 += 3;
                prefetch(din_ptr1);

                //r3
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr1, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr2, 2);

                vtmp2 = vextq_f32(vzero, vinr2, 3);
                vtmp3 = vextq_f32(vzero, vinr3, 3);

                //r4
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr2, 2);

                vtmp4 = vextq_f32(vzero, vinr4, 3);
                //vtmp5 = vextq_f32(vzero, vinr5, 3);

                 //r0
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr1, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp0, wr0, 0);

                din_ptr2 += 3;
                prefetch(din_ptr2);
                //r1
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp1, wr0, 0);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr2, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr1, 0);

                din_ptr3 += 3;
                prefetch(din_ptr3);
                //r2
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp2, wr0, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr1, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr2, 0);
               
                din_ptr4 += 3;
                prefetch(din_ptr4);
                //r3
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr1, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr2, 0);

              //  din_ptr5 += 3;
              //  prefetch(din_ptr5);

                //r4
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr2, 0);

                vst1q_f32(doutr0, voutr0);
                vst1q_f32(doutr1, voutr1);
                vst1q_f32(doutr2, voutr2);
                vst1q_f32(doutr3, voutr3);

                doutr0 += 4;
                doutr1 += 4;
                doutr2 += 4;
                doutr3 += 4;

                //mid col
                for (int j = 0; j < cnt_col; ++j) {
                    voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                    voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                    voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                   //din data
                    vinr0 = vld1q_f32(din_ptr0);
                    vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    vinr1 = vld1q_f32(din_ptr1);
                    vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    vinr2 = vld1q_f32(din_ptr2);
                    vinr2_1 = vld1q_f32(din_ptr2 + 4);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr0, wr0, 0);

                    vinr3 = vld1q_f32(din_ptr3);
                    vinr3_1 = vld1q_f32(din_ptr3 + 4);
                    vinr4 = vld1q_f32(din_ptr4);
                    vinr4_1 = vld1q_f32(din_ptr4 + 4);

                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr2, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr1, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr1, wr0, 0);

                   // vinr5 = vld1q_f32(din_ptr5);
                    //vinr5_1 = vld1q_f32(din_ptr5 + 4);

                    // r0, r1, r2 shift left 2345
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                    vtmp2 = vextq_f32(vinr2, vinr2_1, 1);

                    //r2 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr2, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr1, 0);
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr2, wr0, 0);

                    vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                    vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                  //  vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                    //r3 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr2, 0);
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr1, 0);

                     //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp0, wr0, 1);

                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr2, 0);
                

                    din_ptr0 += 4;
                    prefetch(din_ptr0);
                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr2, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr1, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp1, wr0, 1);

                    din_ptr1 += 4;
                    prefetch(din_ptr1);
                    //r2 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp2, wr0, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr2, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr1, 1);

                     // r0, r1, r2 shift left 3456
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 2);
                    vtmp2 = vextq_f32(vinr2, vinr2_1, 2);

                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr1, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr2, 1);
                    
                    din_ptr2 += 4;
                    prefetch(din_ptr2);

                    ///r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr1, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp0, wr0, 2);
                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr2, 1);

                    vtmp3 = vextq_f32(vinr3, vinr3_1, 2);
                    vtmp4 = vextq_f32(vinr4, vinr4_1, 2);
                   // vtmp5 = vextq_f32(vinr5, vinr5_1, 2);


                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr2, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr1, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp1, wr0, 2);
                    
                    din_ptr3 += 4;
                    prefetch(din_ptr3);
                    //r2 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp2, wr0, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr2, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr1, 2);
                   
                    din_ptr4 += 4;
                    prefetch(din_ptr4);
                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr1, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr2, 2);

                   // din_ptr5 += 4;
                   // prefetch(din_ptr5);
                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr2, 2);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);
                    vst1q_f32(doutr2, voutr2);
                    vst1q_f32(doutr3, voutr3);

                    doutr0 += 4;
                    doutr1 += 4;
                    doutr2 += 4;
                    doutr3 += 4;

                }
                //right
                voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                //din data
                vinr0 = vld1q_f32(din_ptr0);
                vinr0_1 = vld1q_f32(din_ptr0 + 4);
                vinr1 = vld1q_f32(din_ptr1);
                vinr1_1 = vld1q_f32(din_ptr1 + 4);
                vinr2 = vld1q_f32(din_ptr2);
                vinr2_1 = vld1q_f32(din_ptr2 + 4);

                vinr0 = vbslq_f32(vmask_rp1, vinr0, vzero);
                vinr0_1 = vbslq_f32(vmask_rp2, vinr0_1, vzero);

                vinr1 = vbslq_f32(vmask_rp1, vinr1, vzero);
                vinr1_1 = vbslq_f32(vmask_rp2, vinr1_1, vzero);

                vinr2 = vbslq_f32(vmask_rp1, vinr2, vzero);
                vinr2_1 = vbslq_f32(vmask_rp2, vinr2_1, vzero);

                vinr3 = vld1q_f32(din_ptr3);
                vinr3_1 = vld1q_f32(din_ptr3 + 4);
                vinr4 = vld1q_f32(din_ptr4);
                vinr4_1 = vld1q_f32(din_ptr4 + 4);
               // vinr5 = vld1q_f32(din_ptr5);
               // vinr5_1 = vld1q_f32(din_ptr5 + 4);

                //r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr1, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr0, wr0, 0);

                vinr3 = vbslq_f32(vmask_rp1, vinr3, vzero);
                vinr3_1 = vbslq_f32(vmask_rp2, vinr3_1, vzero);

                vinr4 = vbslq_f32(vmask_rp1, vinr4, vzero);
                vinr4_1 = vbslq_f32(vmask_rp2, vinr4_1, vzero);
                    
               // vinr5 = vbslq_f32(vmask_rp1, vinr5, vzero);
               // vinr5_1 = vbslq_f32(vmask_rp2, vinr5_1, vzero);

                //r1 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr2, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr1, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr1, wr0, 0);

                // r0, r1, r2 shift left 2345
                vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                //r2 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vinr2, wr0, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr2, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr1, 0);

                vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                //r3 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr1, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr2, 0);

                vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                //r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr1, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp0, wr0, 1);

                //r4 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr2, 0);
                    
                vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
               // vtmp5 = vextq_f32(vinr5, vinr5_1, 1);
                    
                //r1 1234
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp1, wr0, 1);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr2, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr1, 1);

                // r0, r1, r2 shift left 3456
                vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                //r2 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp2, wr0, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr1, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr2, 1);

                vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                //r3 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr1, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr2, 1);
             
                vtmp3 = vextq_f32(vinr3, vinr3_1, 2);

                ///r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp0, wr0, 2);

                //r4 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr2, 1);

                vtmp4 = vextq_f32(vinr4, vinr4_1, 2);
               // vtmp5 = vextq_f32(vinr5, vinr5_1, 2);
                
                //r1 1234
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp1, wr0, 2);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr2, 2); 
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr1, 2);
                
                //r2 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp2, wr0, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr2, 2);

                vinr0 = vld1q_f32(doutr0);
                vinr1 = vld1q_f32(doutr1);
                vinr2 = vld1q_f32(doutr2);
                vinr3 = vld1q_f32(doutr3);
                //r3 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr1, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr2, 2);

                dr0 = dr3;
                dr1 = dr4;
                dr2 = dr5;
                //r4 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr2, 2);

                voutr0 = vbslq_f32(vmask_result, voutr0, vinr0);
                voutr1 = vbslq_f32(vmask_result, voutr1, vinr1);
                voutr2 = vbslq_f32(vmask_result, voutr2, vinr2);
                voutr3 = vbslq_f32(vmask_result, voutr3, vinr3);


                vst1q_f32(doutr0, voutr0);
                vst1q_f32(doutr1, voutr1);
                vst1q_f32(doutr2, voutr2);
                vst1q_f32(doutr3, voutr3);

                dr3 = dr2 + w_in;
                dr4 = dr3 + w_in;
                dr5 = dr4 + w_in;
                dout_ptr = dout_ptr + 4 * w_out;
            }
            //mid
            for (h = 0; h < tile_h - 2; h++) {

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                doutr2 = doutr1 + w_out;
                doutr3 = doutr2 + w_out;

                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;
                din_ptr4 = dr4;
                din_ptr5 = dr5;


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
                
                float32x4_t voutr0 = vdupq_n_f32(bias_val);  //vld1q_f32(doutr0);
                float32x4_t voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                float32x4_t voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                float32x4_t voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                //din data
                float32x4_t vinr0 = vld1q_f32(din_ptr0);
                float32x4_t vinr0_1 = vld1q_f32(din_ptr0 + 4);
                float32x4_t vinr1 = vld1q_f32(din_ptr1);
                float32x4_t vinr1_1 = vld1q_f32(din_ptr1 + 4);
                float32x4_t vinr2 = vld1q_f32(din_ptr2);
                float32x4_t vinr2_1 = vld1q_f32(din_ptr2 + 4);

                //r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 1);

                float32x4_t vinr3 = vld1q_f32(din_ptr3);
                float32x4_t vinr3_1 = vld1q_f32(din_ptr3 + 4);
                float32x4_t vinr4 = vld1q_f32(din_ptr4);
                float32x4_t vinr4_1 = vld1q_f32(din_ptr4 + 4);

                //r1
                voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 1);

                float32x4_t vinr5 = vld1q_f32(din_ptr5);
                float32x4_t vinr5_1 = vld1q_f32(din_ptr5 + 4);


                //r2
                voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 1);
                voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 1);

                // r0, r1, r2 shift left 2345
                float32x4_t vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                float32x4_t vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                float32x4_t vtmp2 = vextq_f32(vinr2, vinr2_1, 1);

                //r3
                voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr0, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 1);

                //r0
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                float32x4_t vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                float32x4_t vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                float32x4_t vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                //r4
                voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr1, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 1);

                //r1
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);

                //r5
                voutr3 = vmlaq_laneq_f32(voutr3, vinr5, wr2, 1);

                din_ptr0 += 3;
                prefetch(din_ptr0);

                //r2
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);

                din_ptr1 += 3;
                prefetch(din_ptr1);
                //r3
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                // r0, r1, r2 shift right 0123
                vtmp0 = vextq_f32(vzero, vinr0, 3);
                vtmp1 = vextq_f32(vzero, vinr1, 3);
                vtmp2 = vextq_f32(vzero, vinr2, 3);

                //r4
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                 //r0
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 0);

                vtmp3 = vextq_f32(vzero, vinr3, 3);
                vtmp4 = vextq_f32(vzero, vinr4, 3);
                //r5
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 2);

                vtmp5 = vextq_f32(vzero, vinr5, 3);

                //r1
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 0);

                din_ptr2 += 3;
                prefetch(din_ptr2);

                //r2
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 0);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 0);
               
                din_ptr3 += 3;
                prefetch(din_ptr3);
                //r3
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 0);

                din_ptr4 += 3;
                prefetch(din_ptr4);
                //r4
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 0);

                din_ptr5 += 3;
                prefetch(din_ptr5);
                //r5
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 0);

                vst1q_f32(doutr0, voutr0);
                vst1q_f32(doutr1, voutr1);
                vst1q_f32(doutr2, voutr2);
                vst1q_f32(doutr3, voutr3);

                doutr0 += 4;
                doutr1 += 4;
                doutr2 += 4;
                doutr3 += 4;

                //mid col
                for (int j = 0; j < cnt_col; ++j) {
                    voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                    voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                    voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                   //din data
                    vinr0 = vld1q_f32(din_ptr0);
                    vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    vinr1 = vld1q_f32(din_ptr1);
                    vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    vinr2 = vld1q_f32(din_ptr2);
                    vinr2_1 = vld1q_f32(din_ptr2 + 4);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                    vinr3 = vld1q_f32(din_ptr3);
                    vinr3_1 = vld1q_f32(din_ptr3 + 4);
                    vinr4 = vld1q_f32(din_ptr4);
                    vinr4_1 = vld1q_f32(din_ptr4 + 4);
                    vinr5 = vld1q_f32(din_ptr5);
                    vinr5_1 = vld1q_f32(din_ptr5 + 4);


                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);

                    // r0, r1, r2 shift left 2345
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                    vtmp2 = vextq_f32(vinr2, vinr2_1, 1);

                    //r2 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 0);
                    
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                    vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                    vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr0, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);
                
                     //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                    din_ptr0 += 4;
                    prefetch(din_ptr0);

                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr1, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 0);

                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);

                    //r5 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr5, wr2, 0);

                     // r0, r1, r2 shift left 3456
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);

                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);

                    vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 2);

                    ///r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 1);

                    din_ptr1 += 4;
                    prefetch(din_ptr1);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                    //r5 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 1);

                    vtmp4 = vextq_f32(vinr4, vinr4_1, 2);
                    vtmp5 = vextq_f32(vinr5, vinr5_1, 2);

                    din_ptr2 += 4;
                    prefetch(din_ptr2);
                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    din_ptr3 += 4;
                    prefetch(din_ptr3);
                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                    din_ptr4 += 4;
                    prefetch(din_ptr4);
                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                    din_ptr5 += 4;
                    prefetch(din_ptr5);

                    //r5 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 2);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);
                    vst1q_f32(doutr2, voutr2);
                    vst1q_f32(doutr3, voutr3);

                    doutr0 += 4;
                    doutr1 += 4;
                    doutr2 += 4;
                    doutr3 += 4;

                }
                //right
                voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                //din data
                vinr0 = vld1q_f32(din_ptr0);
                vinr0_1 = vld1q_f32(din_ptr0 + 4);
                vinr1 = vld1q_f32(din_ptr1);
                vinr1_1 = vld1q_f32(din_ptr1 + 4);
                vinr2 = vld1q_f32(din_ptr2);
                vinr2_1 = vld1q_f32(din_ptr2 + 4);

                vinr0 = vbslq_f32(vmask_rp1, vinr0, vzero);
                vinr0_1 = vbslq_f32(vmask_rp2, vinr0_1, vzero);

                vinr1 = vbslq_f32(vmask_rp1, vinr1, vzero);
                vinr1_1 = vbslq_f32(vmask_rp2, vinr1_1, vzero);

                vinr2 = vbslq_f32(vmask_rp1, vinr2, vzero);
                vinr2_1 = vbslq_f32(vmask_rp2, vinr2_1, vzero);

                //r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                vinr3 = vld1q_f32(din_ptr3);
                vinr3_1 = vld1q_f32(din_ptr3 + 4);
                vinr4 = vld1q_f32(din_ptr4);
                vinr4_1 = vld1q_f32(din_ptr4 + 4);
                vinr5 = vld1q_f32(din_ptr5);
                vinr5_1 = vld1q_f32(din_ptr5 + 4);
                
                //r1 1234
                voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                vinr3 = vbslq_f32(vmask_rp1, vinr3, vzero);
                vinr3_1 = vbslq_f32(vmask_rp2, vinr3_1, vzero);

                vinr4 = vbslq_f32(vmask_rp1, vinr4, vzero);
                vinr4_1 = vbslq_f32(vmask_rp2, vinr4_1, vzero);
                    
                vinr5 = vbslq_f32(vmask_rp1, vinr5, vzero);
                vinr5_1 = vbslq_f32(vmask_rp2, vinr5_1, vzero);

                //r2 1234
                voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);
                
                // r0, r1, r2 shift left 2345
                vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
               
                //r3 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr0, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);

                //r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                //r4 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr1, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 0);

                //r1 1234
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                //r5 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vinr5, wr2, 0);

                // r0, r1, r2 shift left 3456
                vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                //r2 1234
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                //r3 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);
             
                ///r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                vtmp3 = vextq_f32(vinr3, vinr3_1, 2);

                //r4 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 1);
                
                //r1 1234
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);
                vtmp4 = vextq_f32(vinr4, vinr4_1, 2); 

               //r5 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 1);

                vtmp5 = vextq_f32(vinr5, vinr5_1, 2);
              
                //r2 1234
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                vinr0 = vld1q_f32(doutr0);
                vinr1 = vld1q_f32(doutr1);
                //r3 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);
                
                vinr2 = vld1q_f32(doutr2);
                vinr3 = vld1q_f32(doutr3);

                //r4 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                dr0 = dr4;
                dr1 = dr5;
                dr2 = dr1 + w_in;
                //r5 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 2);
                
                voutr0 = vbslq_f32(vmask_result, voutr0, vinr0);
                voutr1 = vbslq_f32(vmask_result, voutr1, vinr1);
                voutr2 = vbslq_f32(vmask_result, voutr2, vinr2);
                voutr3 = vbslq_f32(vmask_result, voutr3, vinr3);

                dr3 = dr2 + w_in;
                dr4 = dr3 + w_in;
                dr5 = dr4 + w_in;

                vst1q_f32(doutr0, voutr0);
                vst1q_f32(doutr1, voutr1);
                vst1q_f32(doutr2, voutr2);
                vst1q_f32(doutr3, voutr3);

                dout_ptr = dout_ptr + 4 * w_out;
            }
            //bottom
            if(1){
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                doutr2 = doutr1 + w_out;
                doutr3 = doutr2 + w_out;

                prefetch(doutr0);

                prefetch(din_ptr0);
                prefetch(din_ptr1);

                if (size_pad_bottom == 1){//only 4 line
                    din_ptr2 = dr2;
                    din_ptr3 = dr3;
                    din_ptr4 = dr4;
                    din_ptr5 = ptr_zero;
                    
                    prefetch(doutr1);
                    prefetch(doutr2);
                    prefetch(doutr3);

                    prefetch(din_ptr2);
                    prefetch(din_ptr3);
                    prefetch(din_ptr4);
                    prefetch(din_ptr5);

                    float32x4_t voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    float32x4_t voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                    float32x4_t voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                    float32x4_t voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                    //din data
                    float32x4_t vinr0 = vld1q_f32(din_ptr0);
                    float32x4_t vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    float32x4_t vinr1 = vld1q_f32(din_ptr1);
                    float32x4_t vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    float32x4_t vinr2 = vld1q_f32(din_ptr2);
                    float32x4_t vinr2_1 = vld1q_f32(din_ptr2 + 4);

                     //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 1);

                    float32x4_t vinr3 = vld1q_f32(din_ptr3);
                    float32x4_t vinr3_1 = vld1q_f32(din_ptr3 + 4);
                    float32x4_t vinr4 = vld1q_f32(din_ptr4);
                    float32x4_t vinr4_1 = vld1q_f32(din_ptr4 + 4);
                    float32x4_t vinr5 = vld1q_f32(din_ptr5);
                    float32x4_t vinr5_1 = vld1q_f32(din_ptr5 + 4);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 1);

                    // r0, r1, r2 shift left 2345
                    float32x4_t vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    float32x4_t vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                    //r2
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 1);

                    float32x4_t vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                    float32x4_t vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                    //r3
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr0, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 1);

                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    float32x4_t vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                    float32x4_t vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                    //r4
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr1, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 1);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                    // r0, r1, r2 shift right 0123
                    vtmp0 = vextq_f32(vzero, vinr0, 3);
                    vtmp1 = vextq_f32(vzero, vinr1, 3);

                    //r5
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr5, wr2, 1);

                    //r2
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    din_ptr0 += 3;
                    prefetch(din_ptr0);
                    //r3
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                    din_ptr1 += 3;
                    prefetch(din_ptr1);  
                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 0);
                    //r4
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                    vtmp2 = vextq_f32(vzero, vinr2, 3);
                    vtmp3 = vextq_f32(vzero, vinr3, 3);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 0);
                    //r5
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 2);

                    vtmp4 = vextq_f32(vzero, vinr4, 3);
                    vtmp5 = vextq_f32(vzero, vinr5, 3);

                    //r2
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 0);
               
                    din_ptr2 += 3;
                    prefetch(din_ptr2);
                    //r3
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 0);

                    din_ptr3 += 3;
                    prefetch(din_ptr3);
                    //r4
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 0);

                    din_ptr4 += 3;
                    prefetch(din_ptr4);

                   // din_ptr5 += 3;
                    prefetch(din_ptr5);
                    //r5
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 0);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);
                    vst1q_f32(doutr2, voutr2);
                    vst1q_f32(doutr3, voutr3);

                    doutr0 += 4;
                    doutr1 += 4;
                    doutr2 += 4;
                    doutr3 += 4;

                    //mid col
                    for (int j = 0; j < cnt_col; ++j) {
                        voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                        voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                        voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                        voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                        //din data
                        vinr0 = vld1q_f32(din_ptr0);
                        vinr0_1 = vld1q_f32(din_ptr0 + 4);
                        vinr1 = vld1q_f32(din_ptr1);
                        vinr1_1 = vld1q_f32(din_ptr1 + 4);
                        vinr2 = vld1q_f32(din_ptr2);
                        vinr2_1 = vld1q_f32(din_ptr2 + 4);

                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                        vinr3 = vld1q_f32(din_ptr3);
                        vinr3_1 = vld1q_f32(din_ptr3 + 4);
                        vinr4 = vld1q_f32(din_ptr4);
                        vinr4_1 = vld1q_f32(din_ptr4 + 4);
                        vinr5 = vld1q_f32(din_ptr5);
                        vinr5_1 = vld1q_f32(din_ptr5 + 4);


                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                        // r0, r1, r2 shift left 2345
                        vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                        vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                        //r2 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 0);
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);

                        vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                        vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                        //r3 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr0, 0);
                        voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 0);
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);

                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                        vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                        vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                        //r4 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr1, 0);
                        voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 0);

                        //r1 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);

                        din_ptr0 += 4;
                        prefetch(din_ptr0);
                        //r5 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vinr5, wr2, 0);
                    

                        //r2 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 1);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                        // r0, r1, r2 shift left 3456
                        vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                        vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                        //r3 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 1);
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 1);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);
                    
                        ///r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);
                        vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                        vtmp3 = vextq_f32(vinr3, vinr3_1, 2);

                        //r4 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 1);
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 1);

                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                        din_ptr1 += 4;
                        prefetch(din_ptr1);
                        //r5 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 1);

                        vtmp4 = vextq_f32(vinr4, vinr4_1, 2);
                        vtmp5 = vextq_f32(vinr5, vinr5_1, 2);

                        //r2 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                        din_ptr2 += 4;
                        prefetch(din_ptr2);
                        //r3 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 2);
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                        din_ptr3 += 4;
                        prefetch(din_ptr3);
                        //r4 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 2);
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                        din_ptr4 += 4;
                        prefetch(din_ptr4);
                        //r5 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 2);
                    
                       // din_ptr5 += 4;
                        prefetch(din_ptr5);

                        vst1q_f32(doutr0, voutr0);
                        vst1q_f32(doutr1, voutr1);
                        vst1q_f32(doutr2, voutr2);
                        vst1q_f32(doutr3, voutr3);

                        doutr0 += 4;
                        doutr1 += 4;
                        doutr2 += 4;
                        doutr3 += 4;

                    }
                    //right
                    voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                    voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                    voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                    //din data
                    vinr0 = vld1q_f32(din_ptr0);
                    vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    vinr1 = vld1q_f32(din_ptr1);
                    vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    vinr2 = vld1q_f32(din_ptr2);
                    vinr2_1 = vld1q_f32(din_ptr2 + 4);

                    vinr0 = vbslq_f32(vmask_rp1, vinr0, vzero);
                    vinr0_1 = vbslq_f32(vmask_rp2, vinr0_1, vzero);

                    vinr1 = vbslq_f32(vmask_rp1, vinr1, vzero);
                    vinr1_1 = vbslq_f32(vmask_rp2, vinr1_1, vzero);

                    vinr2 = vbslq_f32(vmask_rp1, vinr2, vzero);
                    vinr2_1 = vbslq_f32(vmask_rp2, vinr2_1, vzero);

                    vinr3 = vld1q_f32(din_ptr3);
                    vinr3_1 = vld1q_f32(din_ptr3 + 4);
                    vinr4 = vld1q_f32(din_ptr4);
                    vinr4_1 = vld1q_f32(din_ptr4 + 4);
                    vinr5 = vld1q_f32(din_ptr5);
                    vinr5_1 = vld1q_f32(din_ptr5 + 4);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                    vinr3 = vbslq_f32(vmask_rp1, vinr3, vzero);
                    vinr3_1 = vbslq_f32(vmask_rp2, vinr3_1, vzero);

                    vinr4 = vbslq_f32(vmask_rp1, vinr4, vzero);
                    vinr4_1 = vbslq_f32(vmask_rp2, vinr4_1, vzero);
                    
                    vinr5 = vbslq_f32(vmask_rp1, vinr5, vzero);
                    vinr5_1 = vbslq_f32(vmask_rp2, vinr5_1, vzero);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                     // r0, r1, r2 shift left 2345
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);
                    
                    vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr0, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                    vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                    vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr1, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 0);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                    // r0, r1, r2 shift left 3456
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                     //r5 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr5, wr2, 0);
                    

                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                    vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);
             
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 2);
                    ///r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 1);
                    
                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                    //r5 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 1);


                    vtmp4 = vextq_f32(vinr4, vinr4_1, 2);
                    vtmp5 = vextq_f32(vinr5, vinr5_1, 2);
                    
                
                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    vinr0 = vld1q_f32(doutr0);
                    vinr1 = vld1q_f32(doutr1);
                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                    vinr2 = vld1q_f32(doutr2);
                    vinr3 = vld1q_f32(doutr3);
                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                    voutr0 = vbslq_f32(vmask_result, voutr0, vinr0);
                    voutr1 = vbslq_f32(vmask_result, voutr1, vinr1);
                    //r5 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 2);
                
                    voutr2 = vbslq_f32(vmask_result, voutr2, vinr2);
                    voutr3 = vbslq_f32(vmask_result, voutr3, vinr3);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);
                    vst1q_f32(doutr2, voutr2);
                    vst1q_f32(doutr3, voutr3);

                }
                if (size_pad_bottom == 2){//only 3 line
                    din_ptr2 = dr2;
                    din_ptr3 = dr3;
                    din_ptr4 = ptr_zero;
                    
                    prefetch(doutr1);
                    prefetch(doutr2);

                    prefetch(din_ptr2);
                    prefetch(din_ptr3);
                    prefetch(din_ptr4);

                    float32x4_t voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    float32x4_t voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                    float32x4_t voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);

                    //din data
                    float32x4_t vinr0 = vld1q_f32(din_ptr0);
                    float32x4_t vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    float32x4_t vinr1 = vld1q_f32(din_ptr1);
                    float32x4_t vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    float32x4_t vinr2 = vld1q_f32(din_ptr2);
                    float32x4_t vinr2_1 = vld1q_f32(din_ptr2 + 4);

                     //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 1);

                    float32x4_t vinr3 = vld1q_f32(din_ptr3);
                    float32x4_t vinr3_1 = vld1q_f32(din_ptr3 + 4);
                    float32x4_t vinr4 = vld1q_f32(din_ptr4);
                    float32x4_t vinr4_1 = vld1q_f32(din_ptr4 + 4);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 1);

                    // r0, r1, r2 shift left 2345
                    float32x4_t vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    float32x4_t vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                    //r2
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 1);

                    float32x4_t vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                    float32x4_t vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                    //r3
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 1);
                    
                    float32x4_t vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    din_ptr0 += 3;
                    prefetch(din_ptr0);

                    //r4
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 1);


                    din_ptr1 += 3;
                    prefetch(din_ptr1);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                    // r0, r1, r2 shift right 0123
                    vtmp0 = vextq_f32(vzero, vinr0, 3);
                    vtmp1 = vextq_f32(vzero, vinr1, 3);

                    //r2
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);
                    
                    din_ptr2 += 3;
                    prefetch(din_ptr2);

                    vtmp2 = vextq_f32(vzero, vinr2, 3);

                    //r3
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 0);
                    vtmp3 = vextq_f32(vzero, vinr3, 3);

                    //r4
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                    vtmp4 = vextq_f32(vzero, vinr4, 3);
                    
                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 0);

                    din_ptr3 += 3;
                    prefetch(din_ptr3);
                    //r2
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 0);
               
                   // din_ptr4 += 3;
                    prefetch(din_ptr4);
                    //r3
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 0);

                    //r4
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 0);


                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);
                    vst1q_f32(doutr2, voutr2);

                    doutr0 += 4;
                    doutr1 += 4;
                    doutr2 += 4;

                    //mid col
                    for (int j = 0; j < cnt_col; ++j) {
                        voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                        voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                        voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);

                        //din data
                        vinr0 = vld1q_f32(din_ptr0);
                        vinr0_1 = vld1q_f32(din_ptr0 + 4);
                        vinr1 = vld1q_f32(din_ptr1);
                        vinr1_1 = vld1q_f32(din_ptr1 + 4);
                        vinr2 = vld1q_f32(din_ptr2);
                        vinr2_1 = vld1q_f32(din_ptr2 + 4);

                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                        vinr3 = vld1q_f32(din_ptr3);
                        vinr3_1 = vld1q_f32(din_ptr3 + 4);
                        vinr4 = vld1q_f32(din_ptr4);
                        vinr4_1 = vld1q_f32(din_ptr4 + 4);


                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);
                        
                        // r0, r1, r2 shift left 2345
                        vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                        vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                        //r2 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 0);
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);
  
                        vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                        vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                        //r3 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 0);
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);

                        vtmp4 = vextq_f32(vinr4, vinr4_1, 1);

                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                        din_ptr0 += 4;
                        prefetch(din_ptr0);

                        //r4 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 0);
                    
                        din_ptr1 += 4;
                        prefetch(din_ptr1);
                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                        // r0, r1, r2 shift left 3456
                        vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                        vtmp1 = vextq_f32(vinr1, vinr1_1, 2);
                        //r2 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 1);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                        din_ptr2 += 4;
                        prefetch(din_ptr2);
                        //r3 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 1);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);
                    
                        ///r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);
                        vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                        vtmp3 = vextq_f32(vinr3, vinr3_1, 2);
                        //r4 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 1);

                        vtmp4 = vextq_f32(vinr4, vinr4_1, 2);

                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                        din_ptr3 += 4;
                        prefetch(din_ptr3);

                        //r2 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                        prefetch(din_ptr4);
                        //r3 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                        //r4 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                       // din_ptr5 += 4;
                       // prefetch(din_ptr5);

                        vst1q_f32(doutr0, voutr0);
                        vst1q_f32(doutr1, voutr1);
                        vst1q_f32(doutr2, voutr2);

                        doutr0 += 4;
                        doutr1 += 4;
                        doutr2 += 4;

                    }
                    //right
                    voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                    voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);

                    //din data
                    vinr0 = vld1q_f32(din_ptr0);
                    vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    vinr1 = vld1q_f32(din_ptr1);
                    vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    vinr2 = vld1q_f32(din_ptr2);
                    vinr2_1 = vld1q_f32(din_ptr2 + 4);

                    vinr0 = vbslq_f32(vmask_rp1, vinr0, vzero);
                    vinr0_1 = vbslq_f32(vmask_rp2, vinr0_1, vzero);

                    vinr1 = vbslq_f32(vmask_rp1, vinr1, vzero);
                    vinr1_1 = vbslq_f32(vmask_rp2, vinr1_1, vzero);

                    vinr2 = vbslq_f32(vmask_rp1, vinr2, vzero);
                    vinr2_1 = vbslq_f32(vmask_rp2, vinr2_1, vzero);

                    vinr3 = vld1q_f32(din_ptr3);
                    vinr3_1 = vld1q_f32(din_ptr3 + 4);
                    vinr4 = vld1q_f32(din_ptr4);
                    vinr4_1 = vld1q_f32(din_ptr4 + 4);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                    vinr3 = vbslq_f32(vmask_rp1, vinr3, vzero);
                    vinr3_1 = vbslq_f32(vmask_rp2, vinr3_1, vzero);

                    vinr4 = vbslq_f32(vmask_rp1, vinr4, vzero);
                    vinr4_1 = vbslq_f32(vmask_rp2, vinr4_1, vzero);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                     // r0, r1, r2 shift left 2345
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);

                    vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                    //r3 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                    //r4 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 0);
                    
                    vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                
                    
                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                    // r0, r1, r2 shift left 3456
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                    vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                    //r3 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);
             
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 2);
                    ///r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);
                    //r4 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 1);

                    vtmp4 = vextq_f32(vinr4, vinr4_1, 2);
                
                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2); 

                
                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    vinr0 = vld1q_f32(doutr0);
                    vinr1 = vld1q_f32(doutr1);
                    vinr2 = vld1q_f32(doutr2);
                    //r3 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                    voutr0 = vbslq_f32(vmask_result, voutr0, vinr0);
                    //r4 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);
                
                    voutr1 = vbslq_f32(vmask_result, voutr1, vinr1);
                    voutr2 = vbslq_f32(vmask_result, voutr2, vinr2);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);
                    vst1q_f32(doutr2, voutr2);

                }
                if (size_pad_bottom == 3){//only 2 line
                    din_ptr2 = dr2;
                    din_ptr3 = ptr_zero;
                    
                    prefetch(doutr1);

                    prefetch(din_ptr2);

                    float32x4_t voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    float32x4_t voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);

                    //din data
                    float32x4_t vinr0 = vld1q_f32(din_ptr0);
                    float32x4_t vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    float32x4_t vinr1 = vld1q_f32(din_ptr1);
                    float32x4_t vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    float32x4_t vinr2 = vld1q_f32(din_ptr2);
                    float32x4_t vinr2_1 = vld1q_f32(din_ptr2 + 4);

                     //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 1);

                    float32x4_t vinr3 = vld1q_f32(din_ptr3);
                    float32x4_t vinr3_1 = vld1q_f32(din_ptr3 + 4);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 1);

                    // r0, r1, r2 shift left 2345
                    float32x4_t vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    float32x4_t vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                    //r2
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 1);

                    float32x4_t vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                    float32x4_t vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                    //r3
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 1);

                    din_ptr0 += 3;
                    prefetch(din_ptr0);
                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    din_ptr1 += 3;
                    prefetch(din_ptr1);
                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                    // r0, r1, r2 shift right 0123
                    vtmp0 = vextq_f32(vzero, vinr0, 3);
                    vtmp1 = vextq_f32(vzero, vinr1, 3);

                    //r2
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    din_ptr2 += 3;
                    prefetch(din_ptr2);
                    //r3
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                    vtmp2 = vextq_f32(vzero, vinr2, 3);
                    vtmp3 = vextq_f32(vzero, vinr3, 3);

                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 0);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 0);

                   // din_ptr3 += 3;
                    prefetch(din_ptr3);
                    //r2
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 0);
               
                    //r3
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 0);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);

                    doutr0 += 4;
                    doutr1 += 4;

                    //mid col
                    for (int j = 0; j < cnt_col; ++j) {
                        voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                        voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);

                        //din data
                        vinr0 = vld1q_f32(din_ptr0);
                        vinr0_1 = vld1q_f32(din_ptr0 + 4);
                        vinr1 = vld1q_f32(din_ptr1);
                        vinr1_1 = vld1q_f32(din_ptr1 + 4);
                        vinr2 = vld1q_f32(din_ptr2);
                        vinr2_1 = vld1q_f32(din_ptr2 + 4);

                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                        vinr3 = vld1q_f32(din_ptr3);
                        vinr3_1 = vld1q_f32(din_ptr3 + 4);

                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                        // r0, r1, r2 shift left 2345
                        vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                        vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                        //r2 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);

                        vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                        vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                        //r3 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);
                    
                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                        din_ptr0 += 4;
                        prefetch(din_ptr0);

                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);
                        
                        // r0, r1, r2 shift left 3456
                        vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                        vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                        //r2 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                        din_ptr1 += 4;
                        prefetch(din_ptr1);
                        vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                        //r3 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);
                        
                        vtmp3 = vextq_f32(vinr3, vinr3_1, 2);

                        ///r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                        din_ptr2 += 4;
                        prefetch(din_ptr2);
                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                       // din_ptr3 += 4;
                        prefetch(din_ptr3);
                        //r2 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                        //r3 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                        vst1q_f32(doutr0, voutr0);
                        vst1q_f32(doutr1, voutr1);

                        doutr0 += 4;
                        doutr1 += 4;

                    }
                    //right
                    voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);

                    //din data
                    vinr0 = vld1q_f32(din_ptr0);
                    vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    vinr1 = vld1q_f32(din_ptr1);
                    vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    vinr2 = vld1q_f32(din_ptr2);
                    vinr2_1 = vld1q_f32(din_ptr2 + 4);

                    vinr0 = vbslq_f32(vmask_rp1, vinr0, vzero);
                    vinr0_1 = vbslq_f32(vmask_rp2, vinr0_1, vzero);

                    vinr1 = vbslq_f32(vmask_rp1, vinr1, vzero);
                    vinr1_1 = vbslq_f32(vmask_rp2, vinr1_1, vzero);

                    vinr3 = vld1q_f32(din_ptr3);
                    vinr3_1 = vld1q_f32(din_ptr3 + 4);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                    vinr2 = vbslq_f32(vmask_rp1, vinr2, vzero);
                    vinr2_1 = vbslq_f32(vmask_rp2, vinr2_1, vzero);

                    vinr3 = vbslq_f32(vmask_rp1, vinr3, vzero);
                    vinr3_1 = vbslq_f32(vmask_rp2, vinr3_1, vzero);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                     // r0, r1, r2 shift left 2345
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                    //r2 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);

                    vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                    //r3 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);
                    
                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);
                    
                    // r0, r1, r2 shift left 3456
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 2);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                    vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                    //r2 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                    vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                    //r3 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);

                    vtmp3 = vextq_f32(vinr3, vinr3_1, 2);

                    ///r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2); 
                
                    vinr0 = vld1q_f32(doutr0);
                    vinr1 = vld1q_f32(doutr1);

                    //r2 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    //r3 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);
                
                    voutr0 = vbslq_f32(vmask_result, voutr0, vinr0);
                    voutr1 = vbslq_f32(vmask_result, voutr1, vinr1);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);

                }
                if (size_pad_bottom == 4){//only 1 line
                    din_ptr2 = ptr_zero;

                    prefetch(din_ptr2);

                    float32x4_t voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);

                    //din data
                    float32x4_t vinr0 = vld1q_f32(din_ptr0);
                    float32x4_t vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    float32x4_t vinr1 = vld1q_f32(din_ptr1);
                    float32x4_t vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    float32x4_t vinr2 = vld1q_f32(din_ptr2);
                    float32x4_t vinr2_1 = vld1q_f32(din_ptr2 + 4);
                     //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 1);

                    float32x4_t vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    //r1
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 1);

                    float32x4_t vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                    //r2
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 1);

                    // r0, r1, r2 shift left 2345
                    float32x4_t vtmp2 = vextq_f32(vinr2, vinr2_1, 1);

                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    vtmp0 = vextq_f32(vzero, vinr0, 3);
                    //r1
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                    vtmp1 = vextq_f32(vzero, vinr1, 3);
                    //r2
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    // r0, r1, r2 shift right 0123
                    vtmp2 = vextq_f32(vzero, vinr2, 3);

                    din_ptr0 += 3;
                    prefetch(din_ptr0);


                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 0);

                    din_ptr1 += 3;
                    prefetch(din_ptr1);
                    //r1
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 0);

                    //din_ptr2 += 3;
                    prefetch(din_ptr2);
                    //r2
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 0);

                    vst1q_f32(doutr0, voutr0);

                    doutr0 += 4;

                    //mid col
                    for (int j = 0; j < cnt_col; ++j) {
                        voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);

                        //din data
                        vinr0 = vld1q_f32(din_ptr0);
                        vinr0_1 = vld1q_f32(din_ptr0 + 4);
                        vinr1 = vld1q_f32(din_ptr1);
                        vinr1_1 = vld1q_f32(din_ptr1 + 4);
                        vinr2 = vld1q_f32(din_ptr2);
                        vinr2_1 = vld1q_f32(din_ptr2 + 4);


                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                        vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                        //r1 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                        vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                        //r2 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);

                        // r0, r1, r2 shift left 2345
                        vtmp2 = vextq_f32(vinr2, vinr2_1, 1);

                
                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                        vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                        //r1 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                        vtmp1 = vextq_f32(vinr1, vinr1_1, 2);
                        //r2 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);
                    
                        // r0, r1, r2 shift left 3456
                        vtmp2 = vextq_f32(vinr2, vinr2_1, 2);

                        din_ptr0 += 4;
                        prefetch(din_ptr0);

                        ///r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                        din_ptr1 += 4;
                        prefetch(din_ptr1);
                        //r1 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                      //  din_ptr2 += 4;
                        prefetch(din_ptr2);
                        //r2 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);


                        vst1q_f32(doutr0, voutr0);

                        doutr0 += 4;

                    }
                    //right
                    voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);

                    //din data
                    vinr0 = vld1q_f32(din_ptr0);
                    vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    vinr1 = vld1q_f32(din_ptr1);
                    vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    vinr2 = vld1q_f32(din_ptr2);
                    vinr2_1 = vld1q_f32(din_ptr2 + 4);

                    vinr0 = vbslq_f32(vmask_rp1, vinr0, vzero);
                    vinr0_1 = vbslq_f32(vmask_rp2, vinr0_1, vzero);

                    vinr1 = vbslq_f32(vmask_rp1, vinr1, vzero);
                    vinr1_1 = vbslq_f32(vmask_rp2, vinr1_1, vzero);

                    vinr2 = vbslq_f32(vmask_rp1, vinr2, vzero);
                    vinr2_1 = vbslq_f32(vmask_rp2, vinr2_1, vzero);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                    vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                    vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                    //r2 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);
                    
                     // r0, r1, r2 shift left 2345
                    vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                
                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);
                    
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                    vtmp1 = vextq_f32(vinr1, vinr1_1, 2);
                    //r2 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                    // r0, r1, r2 shift left 3456
                    vtmp2 = vextq_f32(vinr2, vinr2_1, 2);

                    vinr0 = vld1q_f32(doutr0);
                    ///r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2); 
                
                    //r2 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);
                    //r3 1234
                
                    voutr0 = vbslq_f32(vmask_result, voutr0, vinr0);

                    vst1q_f32(doutr0, voutr0);

                }
                
            }
        }
        
    }

}

#else

void conv_depthwise_3x3s1p1_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
   // printf("conv_depthwise_3x3s1p1_bias armv7 \n");
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
    //unsigned int tmp1[4];
    //vst1q_u32(tmp1, vmask_rp);
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
            int cnt = cnt_col;
            asm volatile(
            //! process left pad
            "pld [%[din0_ptr]]                @ preload data\n"
                    "pld [%[din1_ptr]]                      @ preload data\n"
                    "pld [%[din2_ptr]]                      @ preload data\n"

                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vld1.32  {d28-d30}, [%[din2_ptr]]!     @ load din r3\n"

                    "vmul.f32 q5, q10, %e[wr0][1]           @ mul weight 00, outr1\n"
                    "vmul.f32 q4, q10, %e[wr1][1]           @ mul weight 10, outr0\n"


                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q12, %e[wr1][1]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][1]           @ mul weight 20, outr0\n"

                    "vext.32  q8, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q14, %e[wr2][1]           @ mul weight 20, outr1\n"

                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"

                    "vext.32  q9, q14, q15, #1              @ shift left r3\n"
                    "vmov.u32 d31, #0 @ zero\n"
                    "vmla.f32 q4, q8,  %f[wr2][0]           @ mul weight 21, outr0\n"
                    "vmla.f32 q5, q8,  %f[wr1][0]           @ mul weight 11, outr1\n"
 
                    "vext.32  d12, d31, d20, #1             @ shift right r1\n"
                    "vext.32  d13, d20, d21, #1             @ shift right r1\n"
                    "vmla.f32 q5, q9,  %f[wr2][0]           @ mul weight 21, outr1\n"

                    "vext.32  d16, d31, d24, #1             @ shift right r0\n"
                    "vext.32  d17, d24, d25, #1             @ shift right r0\n"
                    "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"
                    "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"

                    "vext.32  d18, d31, d28, #1             @ shift right r0\n"
                    "vext.32  d19, d28, d29, #1             @ shift right r0\n"
                    "vmla.f32 q5, q8,  %e[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q8,  %e[wr2][0]           @ mul weight 22, outr0\n"

                    "vmla.f32 q5, q9,  %e[wr2][0]           @ mul weight 22, outr1\n"

                    "sub %[din0_ptr], #12                   @ 1pad + 2 float data overlap\n"
                    "sub %[din1_ptr], #12                   @ 1pad + 2 float data overlap\n"
                    "sub %[din2_ptr], #12                   @ 1pad + 2 float data overlap\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "pld [%[din0_ptr]]                      @ preload data\n"
                    "pld [%[din1_ptr]]                      @ preload data\n"
                    "pld [%[din2_ptr]]                      @ preload data\n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    

                    //! process mid cols
                    "cmp %[cnt], #1                         @ check whether has mid cols\n"
                    "blt  1f                                @ jump to main loop start point\n"
                    "2:                                     @ main loop start point\n"
                   // "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vld1.32  {d28-d30}, [%[din2_ptr]]!     @ load din r3\n"

                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmul.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                   // "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                   // "pld [%[din2_ptr], #192]                @ preload data\n"
                    "vext.32  q8, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q14, %e[wr2][0]           @ mul weight 20, outr1\n"

                    "vext.32  q9, q14, q15, #1              @ shift left r3\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n" 
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q8,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q8,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q8, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q9,  %e[wr2][1]           @ mul weight 21, outr1\n"

                    "vext.32  q9, q14, q15, #2              @ shift right r3\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"

                    "vmla.f32 q4, q8,  %f[wr2][0]           @ mul weight 22, outr0\n"
                    "vmla.f32 q5, q8,  %f[wr1][0]           @ mul weight 12, outr1\n"

                    "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "vmla.f32 q5, q9,  %f[wr2][0]           @ mul weight 22, outr1\n"

                    "sub %[din2_ptr], #8                    @ 2 float data overlap with previous data\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "pld [%[din0_ptr]]                      @ preload data\n"
                    "pld [%[din1_ptr]]                      @ preload data\n"
                    "pld [%[din2_ptr]]                      @ preload data\n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    2b                              @ jump to main loop start point\n"

                    //! process right pad
                    "1:                                     @ right pad entry\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                   // "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r2\n"
                    "vld1.32  {d28-d29}, [%[din2_ptr]]!     @ load din r3\n"

                    "vbif d21, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vbif d25, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vbif d29, d31, %e[mask]                @ bit select, deal with right pad\n"

                    "vld1.32  {d22}, [%[din0_ptr]]!     @ load din r3\n"
                    "vld1.32  {d26}, [%[din1_ptr]]!     @ load din r3\n"
                    "vld1.32  {d30}, [%[din2_ptr]]!     @ load din r3\n"

                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmul.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "vbif  d22, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vbif  d26, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vbif  d30, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"

                   // "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                   // "pld [%[din2_ptr], #192]                @ preload data\n"
                    "vext.32  q8, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q14, %e[wr2][0]           @ mul weight 20, outr1\n"

                    "vext.32  q9, q14, q15, #1              @ shift left r3\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q8,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q8,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q8, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q9,  %e[wr2][1]           @ mul weight 21, outr1\n"

                    "vext.32  q9, q14, q15, #2              @ shift right r3\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "pld [%[dout_ptr1]]                     @ preload data\n"
                    "vmla.f32 q5, q8,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q8,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "vmla.f32 q5, q9,  %f[wr2][0]           @ mul weight 22, outr1\n"

                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"


                    "vmvn.32  d24, d31                      @ \n"
                    "vmvn.32  d25, d31                      @ \n"
                    "vext.32  q13, q12, %q[mask], #3        @ shift mask right 1\n"
                    "vbif q8, q10, q13                      @ bit select\n"

                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"

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

                cnt = cnt_col;
                asm volatile(
                //! process left pad
                "pld [%[din0_ptr]]                             @ preload data\n"
                        "pld [%[din1_ptr]]                      @ preload data\n"
                        "pld [%[din2_ptr]]                      @ preload data\n"
                        "pld [%[din3_ptr]]                      @ preload data\n"

                        "vld1.32  {d16-d18}, [%[din0_ptr]]!    @ load din r0\n"
                        "vld1.32  {d20-d22}, [%[din1_ptr]]!    @ load din r1\n"
                        "vld1.32  {d24-d26}, [%[din2_ptr]]!    @ load din r2\n"
                        "vld1.32  {d28-d30}, [%[din3_ptr]]!    @ load din r3\n"

                        "vmul.f32 q4, q8, %e[wr0][1]   @mul weight 00, outr0\n"

                        "vext.32  q6, q8, q9, #1     @ shift left r0\n"
                        "vmul.f32 q5, q10, %e[wr0][1]  @mul weight 00, outr1\n"
                        "vmla.f32 q4, q10, %e[wr1][1]  @mul weight 10, outr0\n"

                        "pld [%[din0_ptr], #192]               @ preload data\n"
                        "vmla.f32 q5, q12, %e[wr1][1]  @mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][1]  @mul weight 20, outr0\n"

                        "pld [%[din1_ptr], #192]               @ preload data\n"
                        "pld [%[din2_ptr], #192]               @ preload data\n"
                        "vmla.f32 q5, q14, %e[wr2][1]  @mul weight 20, outr1\n"
                        
                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 01, outr0\n"

                        "vext.32  q6, q10, q11, #1   @ shift left r1\n"
                        "pld [%[din3_ptr], #192]               @ preload data\n"
                        "vmla.f32 q5, q6, %f[wr0][0]  @mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]  @mul weight 11, outr0\n"

                        "vext.32  q6, q12, q13, #1  @ shift left r2\n"
                        "vmov.u32 d31, #0 @ zero\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 21, outr0\n"

                        "vext.32  q6, q14, q15, #1  @ shift left r3\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 21, outr1\n"

                        "vext.32  d12, d31, d16, #1     @ shift right r0\n"
                        "vext.32  d13, d16, d17, #1     @ shift right r0\n"
                        "vext.32  d16, d31, d20, #1     @ shift right r1\n"
                        "vext.32  d17, d20, d21, #1     @ shift right r1\n"
                        "vmla.f32 q4, q6, %e[wr0][0]  @mul weight 02, outr0\n"

                        
                        "vext.32  d12, d31, d24, #1     @ shift right r0\n"
                        "vext.32  d13, d24, d25, #1     @ shift right r0\n"
                        "vmla.f32 q5, q8, %e[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q8, %e[wr1][0]  @mul weight 12, outr0\n"
                        
                        "vext.32  d16, d31, d28, #1     @ shift right r0\n"
                        "vext.32  d17, d28, d29, #1     @ shift right r0\n"
                        "vmla.f32 q5, q6,  %e[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][0] @mul weight 22, outr0\n"

                        "sub %[din0_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "vmla.f32 q5, q8,  %e[wr2][0] @mul weight 22, outr1\n"

                        "vadd.f32 q9, q4, %q[bias] @ add bias \n"
                        "sub %[din2_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "sub %[din3_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "vadd.f32 q8, q5, %q[bias] @ add bias \n"

                        "vst1.32  {d18-d19},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "vst1.32  {d16-d17}, [%[dout_ptr2]]!  @ store result, add pointer\n"

                        //! process mid cols
                        "cmp %[cnt], #1                             @ check whether has mid cols\n"
                        "blt  3f                                @ jump to main loop start point\n"
                        "2:                                     @ main loop start point\n"
                        "vld1.32  {d16-d18}, [%[din0_ptr]]!     @ load din r0\n"
                        "vld1.32  {d20-d22}, [%[din1_ptr]]!     @ load din r1\n"
                        "vld1.32  {d24-d26}, [%[din2_ptr]]!     @ load din r2\n"
                        "vld1.32  {d28-d30}, [%[din3_ptr]]!     @ load din r3\n"

                        "vext.32  q6, q8, q9, #1        @ shift left r0\n"
                        "vmul.f32 q4, q8, %e[wr0][0]            @ mul weight 00, outr0\n"

                        "pld [%[din0_ptr], #192]                @ preload data\n"
                        "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                        "vmla.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                        "pld [%[din1_ptr], #192]               @ preload data\n"
                        "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"


                        "pld [%[din2_ptr], #192]               @ preload data\n"
                        "pld [%[din3_ptr], #192]               @ preload data\n"
                        "vmla.f32 q5, q14, %e[wr2][0]   @ mul weight 20, outr1\n"

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
                        "vext.32  q8, q10, q11, #2      @ shift left r1\n"
                        "vmla.f32 q4, q6, %f[wr0][0]    @ mul weight 02, outr0\n"

                        "vext.32  q6, q12, q13, #2      @ shift left r2\n"
                        "vmla.f32 q5, q8, %f[wr0][0]    @ mul weight 02, outr1\n"
                        "vmla.f32 q4, q8, %f[wr1][0]    @ mul weight 12, outr0\n"

                        "vext.32  q8, q14, q15, #2  @ shift right r3\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 22, outr0\n"

                        "sub %[din0_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8 @ 2 float data overlap with previous data\n"
                        "vmla.f32 q5, q8,  %f[wr2][0] @mul weight 22, outr1\n"

                        "vadd.f32 q9, q4, %q[bias] @ add bias \n"
                        "sub %[din2_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din3_ptr], #8 @ 2 float data overlap with previous data\n"
                        "vadd.f32 q8, q5, %q[bias] @ add bias \n"

                        "vst1.32  {d18-d19},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "vst1.32  {d16-d17}, [%[dout_ptr2]]!  @ store result, add pointer\n"

                       
                        "subs %[cnt], #1 @ loop count minus 1\n"
                        "bne    2b                             @ jump to main loop start point\n"

                        //! process right pad
                        "3:                                    @ right pad entry\n"
                        "vmov.u32  d31, #0                     @ zero buf\n"
                        "vld1.32  {d16-d17}, [%[din0_ptr]]!    @ load din r0\n"
                        "vld1.32  {d20-d21}, [%[din1_ptr]]!    @ load din r1\n"
                        "vld1.32  {d24-d25}, [%[din2_ptr]]!    @ load din r2\n"
                        "vld1.32  {d28-d29}, [%[din3_ptr]]!    @ load din r3\n"

                        "vld1.32  {d18}, [%[din0_ptr]]!    @ load din r3\n"
                        "vld1.32  {d22}, [%[din1_ptr]]!    @ load din r3\n"
                        "vbif d17, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vbif d21, d31, %e[mask]               @ bit select, deal with right pad\n"
                        
                        "vbif d25, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vbif d29, d31, %e[mask]               @ bit select, deal with right pad\n"

                        "vmul.f32 q4, q8, %e[wr0][0]           @ mul weight 00, outr0\n"

                        "vbif  d18, d31, %f[mask]              @ bit select, deal with right pad\n"
                        "vbif  d22, d31, %f[mask]              @ bit select, deal with right pad\n"
                        "vmul.f32 q5, q10, %e[wr0][0]          @ mul weight 00, outr1\n"
                        "vmla.f32 q4, q10, %e[wr1][0]          @ mul weight 10, outr0\n"

                        "vext.32  q6, q8, q9, #1               @ shift left r0\n"
                        "vmla.f32 q5, q12, %e[wr1][0]          @ mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][0]          @ mul weight 20, outr0\n"

                        "vld1.32  {d26}, [%[din2_ptr]]!    @ load din r3\n"
                        "vld1.32  {d30}, [%[din3_ptr]]!    @ load din r3\n"
                        "vmla.f32 q5, q14, %e[wr2][0]          @ mul weight 20, outr1\n"
                        "vmla.f32 q4, q6, %e[wr0][1]           @ mul weight 01, outr0\n"

                        "vext.32  q6, q10, q11, #1             @ shift left r1\n"
                        "vbif  d26, d31, %f[mask]               @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q6, %e[wr0][1]           @ mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][1]           @ mul weight 11, outr0\n"

                        "vext.32  q6, q12, q13, #1  @ shift left r2\n"
                        "vbif  d30, d31, %f[mask] @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q6,  %e[wr1][1] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][1] @mul weight 21, outr0\n"

                        "vext.32  q6, q14, q15, #1  @ shift left r3\n"
                        "vmla.f32 q5, q6,  %e[wr2][1] @mul weight 21, outr1\n"

                        "vext.32  q6, q8, q9, #2     @ shift left r0\n"
                        "vext.32  q8, q10, q11, #2   @ shift left r1\n"
                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 02, outr0\n"

                        "vext.32  q6, q12, q13, #2  @ shift left r2\n"
                        "pld [%[dout_ptr1], #128]         @ preload data\n"
                        "vmla.f32 q5, q8, %f[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q8, %f[wr1][0]  @mul weight 12, outr0\n"

                        "vext.32  q8, q14, q15, #2  @ shift right r3\n"
                        "vld1.32  {d20-d21}, [%[dout_ptr1]]    @ load dout r0\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 22, outr0\n"

                        "vmvn.32  d22, d31 @ \n"
                        "vmvn.32  d23, d31 @ \n"
                        "vmla.f32 q5, q8,  %f[wr2][0] @mul weight 22, outr1\n"

                        "vext.32  q12, q11, %q[mask], #3                @ shift mask right 1\n"
                        "vadd.f32 q9, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q8, q5, %q[bias] @ add bias \n"
                        
                        "vbif q9, q10, q12                              @ bit select\n"

                        "vst1.32  {d16-d17}, [%[dout_ptr2]]!            @ store result, add pointer\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr1]]!            @ store result, add pointer\n"

                        "sub %[dout_ptr2], %[dout_ptr2], %[pad_right]   @ sub \n"
                        "sub %[dout_ptr1], %[dout_ptr1], %[pad_right]   @ sub \n"

                :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                            [pad_right] "+r" (right_pad_sub), [cnt] "+r"(cnt)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                        [bias] "w"(wbias), [mask] "w" (vmask_rp)
                :"q4", "q5", "q6", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
                );
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

            cnt = cnt_col;
            asm volatile(
            // process left pad
            "pld [%[din0_ptr]]                        @ preload data\n"
                    "pld [%[din1_ptr]]                @ preload data\n"
                    "pld [%[din2_ptr]]                @ preload data\n"

                    "vld1.32  {d16-d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vld1.32  {d20-d22}, [%[din1_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"
                    "vext.32  q6, q8, q9, #1                @ shift left r0\n"

                    "vmul.f32 q4, q8, %e[wr0][1]            @ mul weight 00, outr0\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vmul.f32 q5, q10, %e[wr0][1]           @ mul weight 00, outr1\n"
                    "vmla.f32 q4, q10, %e[wr1][1]           @ mul weight 10, outr0\n"

                    "vmla.f32 q5, q12, %e[wr1][1]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][1]           @ mul weight 20, outr0\n"
 
                    "pld [%[din2_ptr], #192]                @ preload data\n"
                    "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 01, outr0\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmov.u32 d31, #0 @ zero\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"

                    
                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 21, outr0\n"
          
                    "vext.32  d12, d31, d16, #1             @ shift right r0\n"
                    "vext.32  d13, d16, d17, #1             @ shift right r0\n"

                    "vext.32  d16, d31, d20, #1             @ shift right r1\n"
                    "vext.32  d17, d20, d21, #1             @ shift right r1\n"
                    "vmla.f32 q4, q6, %e[wr0][0]            @ mul weight 02, outr0\n"

                    "vext.32  d12, d31, d24, #1             @ shift right r2\n"
                    "vext.32  d13, d24, d25, #1             @ shift right r2\n"
                    "vmla.f32 q5, q8, %e[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q8, %e[wr1][0]            @ mul weight 12, outr0\n"

                    "sub %[din0_ptr], #12                   @ 1pad + 2 data overlap\n"
                    "sub %[din1_ptr], #12                   @ 1pad + 2 data overlap\n"
                    "vmla.f32 q5, q6,  %e[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][0]           @ mul weight 22, outr0\n"

                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"
                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"

                    
                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                    "beq    1f                              @ jump to next block\n"
                    "add %[din2_ptr], #12                   @ 1pad + 2 data overlap\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"


                    // process mid cols
                    "1:                                     @  header of bottom process\n"
                    "cmp %[cnt], #1                         @ check whether has mid cols\n"
                    "blt  4f                                @ jump to main loop start point\n"
                    "2:                                     @ main loop start point\n"
                    "vld1.32  {d16-d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vld1.32  {d20-d22}, [%[din1_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"
                    "vext.32  q6, q8, q9, #1                @ shift left r0\n"

                    "vmul.f32 q4, q8, %e[wr0][0]            @ mul weight 00, outr0\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmla.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "pld [%[din2_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "vmla.f32 q4, q6, %e[wr0][1]            @ mul weight 01, outr0\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q6, q8, q9, #2                @ shift left r0\n"
                    "vext.32  q8, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 02, outr0\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q8, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q8, %f[wr1][0]            @ mul weight 12, outr0\n"
                    
                    "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"

                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"
                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"

                  
                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                    "beq    3f                              @ jump to check point\n"
                    "add %[din2_ptr], #16                   @ point to 4 data ahead\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "3:                                     @ check point\n"
                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    2b                              @ jump to main loop start point\n"

                    // process right pad
                    "4:                                     @ right pad process\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                    "vld1.32  {d16-d17}, [%[din0_ptr]]!     @ load din r0\n"
                    "vld1.32  {d20-d21}, [%[din1_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"

                    "vbif d17, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vbif d21, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vbif d25, d31, %e[mask]                @ bit select, deal with right pad\n"

                    "vld1.32  {d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vld1.32  {d22}, [%[din1_ptr]]!     @ load din r1\n"

                    "vmul.f32 q4, q8, %e[wr0][0]            @ mul weight 00, outr0\n"

                    "vbif  d18, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vbif  d22, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmla.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "vext.32  q6, q8, q9, #1                @ shift left r0\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "vbif  d26, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vmla.f32 q4, q6, %e[wr0][1]            @ mul weight 01, outr0\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q6, q8, q9, #2                @ shift left r0\n"
                    "vext.32  q8, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 02, outr0\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "pld [%[dout_ptr1], #128]               @ preload data\n"
                    "vmla.f32 q5, q8, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q8, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "pld [%[dout_ptr2], #128]               @ preload data\n"
                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"
                    
                    "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"

                    "vmvn.32  d24, d31                      @ \n"
                    "vmvn.32  d25, d31                      @ \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"
                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                
                    "vext.32  q13, q12, %q[mask], #3        @ shift mask right 1\n"
                    "vbif q9, q11, q13                      @ bit select\n"
                    "vbif q8, q10, q13                      @ bit select\n"

                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    "beq    5f                              @ jump to end point\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "5:                                     @ end\n"

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
            //! end of processing bottom pad
        } // end of processing channels
    } // end of processing batchs
}
#endif

/**
 * \brief depthwise convolution kernel 3x3, stride 2
 */
#ifdef __aarch64__
//one line
#if 1
//w_in > 7
void conv_depthwise_3x3s2p1_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {

    int right_pad_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    int out_pad_idx[4] = {0, 1, 2, 3};
    int size_pad_bottom = h_out * 2 - h_in;

    int cnt_col = (w_out >> 2) - 1;
    int cnt_remain = (w_out % 4);
    int size_right_remain = w_in - (7 + cnt_col * 8);

    int size_right_pad = w_out * 2 - w_in;

    uint32x4_t vmask_rp1 = vcgtq_s32(vdupq_n_s32(size_right_remain), vld1q_s32(right_pad_idx));//0 2 4 6
    uint32x4_t vmask_rp2 = vcgtq_s32(vdupq_n_s32(size_right_remain), vld1q_s32(right_pad_idx + 4));//1 3 5 7
    uint32x4_t wmask = vcgtq_s32(vdupq_n_s32(cnt_remain), vld1q_s32(out_pad_idx));//0 1 2 3
   // printf("w_out %d, cnt_col: %d, remain: %d \n", w_out, cnt_col, size_right_remain);
    //printf("mask1: %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3]);
    //printf("mask2: %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3]);
    //printf("wmask: %d, %d, %d, %d \n", wmask[0], wmask[1], wmask[2], wmask[3]);
   // size_right_remain *= sizeof(float);

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

            float32x4_t vzero= vdupq_n_f32(0.f);

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
            float *doutr0_ptr = doutr0;

            //! top pad
            if(1){
               int cnt = cnt_col;

               //printf("cnt_col: %d, remain: %d \n", cnt_col, size_right_remain);
             //  printf("mask1: %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3]);
             //  printf("mask2: %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3]);
              // printf("wmask: %d, %d, %d, %d \n", wmask[0], wmask[1], wmask[2], wmask[3]);

                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "ext  v2.16b, %[vzero].16b, v1.16b, #12     \n" // v2 = {0,1,3,5}
                "ext  v6.16b, %[vzero].16b, v5.16b, #12    \n" // v6 = {0,1,3,5}

                "fmul v8.4s, v0.4s, %[w1].s[1]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w1].s[2]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w1].s[0]            \n" // v2 * w00

                "sub %[inptr0], %[inptr0], #4            \n"
                "sub %[inptr1], %[inptr1], #4             \n"

                "fmla v8.4s, v4.4s, %[w2].s[1]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w2].s[2]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w2].s[0]            \n" // v2 * w00

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "cmp %[cnt], #1                             \n"
                "blt 1f                                     \n"
                //mid
                "2:                                          \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "ld2  {v2.4s, v3.4s}, [%[inptr0]]      \n" //v2={8,10,12,14} v3={9,11,13,15}
                "ext  v6.16b, v0.16b, v2.16b, #4     \n" // v6 = {2,4,6,8}

                "fmul v8.4s, v0.4s, %[w1].s[0]            \n" // v0 * w00
                "fmul v9.4s, v1.4s, %[w1].s[1]            \n" // v1 * w01
                "fmla v10.4s, v6.4s, %[w1].s[2]            \n" // v6 * w02

                "ld2  {v6.4s, v7.4s}, [%[inptr1]]      \n" //v2={8,10,12,14} v3={9,11,13,15}
                "ext  v11.16b, v4.16b, v6.16b, #4     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v4.4s, %[w2].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w2].s[1]            \n" // v1 * w02
                "fmla v10.4s, v11.4s, %[w2].s[2]            \n" // v2 * w00

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"

                "subs %[cnt], %[cnt], #1                    \n"

                "st1 {v0.4s}, [%[outptr]], #16              \n"

                "bne  2b                                    \n"

                //right
                "1:                                          \n"
                "cmp %[remain], #1                           \n"
                "blt 4f                                     \n"
                "3:                                         \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias
                "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
               // "bif  v10.16b, %[vzero].16b, %[wmask].16b    \n" //pipei
                "ext  v2.16b, v0.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}
                "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei


                "fmul v8.4s, v0.4s, %[w1].s[0]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w1].s[1]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w1].s[2]            \n" // v2 * w00

                "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
                "ext  v6.16b, v4.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v4.4s, %[w2].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w2].s[1]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w2].s[2]            \n" // v2 * w00

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
                "bif  v0.16b, %[vzero].16b, %[wmask].16b    \n" //pipei

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "4:                                          \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [outptr] "+r"(doutr0_ptr), \
                  [vzero] "+w" (vzero), [w1] "+w" (wr1), [w2] "+w" (wr2), [cnt] "+r" (cnt), \
                  [mask1] "+w" (vmask_rp1), [mask2] "+w" (vmask_rp2), [wmask] "+w" (wmask), [vbias] "+w" (wbias)
                : [remain] "r" (cnt_remain)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
                );
            }

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 = doutr0 + w_out;
            //! mid
            for (int j = h_out - size_pad_bottom - 1; j > 0; j--){
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;

                doutr0_ptr = doutr0;

               int cnt = cnt_col;

                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"
                "prfm pldl1keep, [%[inptr2]]             \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "ext  v2.16b, %[vzero].16b, v1.16b, #12     \n" // v2 = {0,1,3,5}
                "ext  v6.16b, %[vzero].16b, v5.16b, #12    \n" // v6 = {0,1,3,5}

                "fmul v8.4s, v0.4s, %[w0].s[1]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w0].s[2]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w0].s[0]            \n" // v2 * w00

                "ld2  {v12.4s, v13.4s}, [%[inptr2]], #32    \n"

                "sub %[inptr0], %[inptr0], #4            \n"
                "sub %[inptr1], %[inptr1], #4             \n"

                "fmla v8.4s, v4.4s, %[w1].s[1]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[2]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w1].s[0]            \n" // v2 * w00

                "ext  v14.16b, %[vzero].16b, v13.16b, #12    \n" // v6 = {0,1,3,5}
                "sub %[inptr2], %[inptr2], #4             \n"

                "prfm pldl1keep, [%[inptr0]]             \n"

                "fmla v8.4s, v12.4s, %[w2].s[1]            \n" // v0 * w01
                "fmla v9.4s, v13.4s, %[w2].s[2]            \n" // v1 * w02
                "fmla v10.4s, v14.4s, %[w2].s[0]            \n" // v2 * w00

                "prfm pldl1keep, [%[inptr1]]             \n"
                "prfm pldl1keep, [%[inptr2]]             \n"

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "cmp %[cnt], #1                             \n"
                "blt 1f                                     \n"
                //mid
                "2:                                          \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "ld2  {v2.4s, v3.4s}, [%[inptr0]]      \n" //v2={8,10,12,14} v3={9,11,13,15}
                "ld2  {v6.4s, v7.4s}, [%[inptr1]]      \n" //v2={8,10,12,14} v3={9,11,13,15}
                "ld2  {v12.4s, v13.4s}, [%[inptr2]], #32    \n"
                "ext  v11.16b, v0.16b, v2.16b, #4     \n" // v6 = {2,4,6,8}

                "prfm pldl1keep, [%[inptr2]]             \n"

                "fmul v8.4s, v0.4s, %[w0].s[0]            \n" // v0 * w00
                "fmul v9.4s, v1.4s, %[w0].s[1]            \n" // v1 * w01
                "fmla v10.4s, v11.4s, %[w0].s[2]            \n" // v6 * w02

                "ext  v11.16b, v4.16b, v6.16b, #4     \n" // v6 = {2,4,6,8}
                "ld2  {v14.4s, v15.4s}, [%[inptr2]]    \n"

                "fmla v8.4s, v4.4s, %[w1].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[1]            \n" // v1 * w02
                "fmla v10.4s, v11.4s, %[w1].s[2]            \n" // v2 * w00

                "ext  v11.16b, v12.16b, v14.16b, #4     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v12.4s, %[w2].s[0]            \n" // v0 * w00
                "fmla v9.4s, v13.4s, %[w2].s[1]            \n" // v1 * w01
                "fmla v10.4s, v11.4s, %[w2].s[2]            \n" // v6 * w02

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"
                "prfm pldl1keep, [%[inptr2]]             \n"

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"

                "subs %[cnt], %[cnt], #1                    \n"

                "st1 {v0.4s}, [%[outptr]], #16              \n"

                "bne  2b                                    \n"

                //right
                "1:                                          \n"
                "cmp %[remain], #1                           \n"
                "blt 4f                                     \n"
                "3:                                         \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
               // "bif  v10.16b, %[vzero].16b, %[wmask].16b    \n" //pipei
                "ext  v2.16b, v0.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}
                "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei

                "ld2  {v12.4s, v13.4s}, [%[inptr2]], #32    \n"
                "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
                "ext  v6.16b, v4.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}

                "fmul v8.4s, v0.4s, %[w0].s[0]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w0].s[1]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w0].s[2]            \n" // v2 * w00

                "bif  v12.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v13.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                "fmla v8.4s, v4.4s, %[w1].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[1]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w1].s[2]            \n" // v2 * w00

                "ext  v14.16b, v12.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v12.4s, %[w2].s[0]            \n" // v0 * w01
                "fmla v9.4s, v13.4s, %[w2].s[1]            \n" // v1 * w02
                "fmla v10.4s, v14.4s, %[w2].s[2]            \n" // v2 * w00

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
                "bif  v0.16b, %[vzero].16b, %[wmask].16b    \n" //pipei

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "4:                                          \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [inptr2] "+r"(din2_ptr), [outptr] "+r"(doutr0_ptr), \
                  [vzero] "+w" (vzero), [w1] "+w" (wr1), [w2] "+w" (wr2), [cnt] "+r" (cnt), [w0] "+w" (wr0), \
                  [mask1] "+w" (vmask_rp1), [mask2] "+w" (vmask_rp2), [wmask] "+w" (wmask), [vbias] "+w" (wbias)
                : [remain] "r" (cnt_remain)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14"
                );
                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
                doutr0 = doutr0 + w_out;
            }

            if (size_pad_bottom){
               int cnt = cnt_col;
                din0_ptr = dr0;
                din1_ptr = dr1;

                doutr0_ptr = doutr0;

                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "ext  v2.16b, %[vzero].16b, v1.16b, #12     \n" // v2 = {0,1,3,5}
                "ext  v6.16b, %[vzero].16b, v5.16b, #12    \n" // v6 = {0,1,3,5}

                "fmul v8.4s, v0.4s, %[w0].s[1]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w0].s[2]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w0].s[0]            \n" // v2 * w00

                "sub %[inptr0], %[inptr0], #4            \n"
                "sub %[inptr1], %[inptr1], #4             \n"

                "fmla v8.4s, v4.4s, %[w1].s[1]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[2]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w1].s[0]            \n" // v2 * w00

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "cmp %[cnt], #1                             \n"
                "blt 1f                                     \n"
                //mid
                "2:                                          \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "ld2  {v2.4s, v3.4s}, [%[inptr0]]      \n" //v2={8,10,12,14} v3={9,11,13,15}
                "ext  v6.16b, v0.16b, v2.16b, #4     \n" // v6 = {2,4,6,8}

                "fmul v8.4s, v0.4s, %[w0].s[0]            \n" // v0 * w00
                "fmul v9.4s, v1.4s, %[w0].s[1]            \n" // v1 * w01
                "fmla v10.4s, v6.4s, %[w0].s[2]            \n" // v6 * w02

                "ld2  {v6.4s, v7.4s}, [%[inptr1]]      \n" //v2={8,10,12,14} v3={9,11,13,15}
                "ext  v11.16b, v4.16b, v6.16b, #4     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v4.4s, %[w1].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[1]            \n" // v1 * w02
                "fmla v10.4s, v11.4s, %[w1].s[2]            \n" // v2 * w00

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"

                "subs %[cnt], %[cnt], #1                    \n"

                "st1 {v0.4s}, [%[outptr]], #16              \n"

                "bne  2b                                    \n"

                //right
                "1:                                          \n"
                "cmp %[remain], #1                           \n"
                "blt 4f                                     \n"
                "3:                                         \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias
                "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
               // "bif  v10.16b, %[vzero].16b, %[wmask].16b    \n" //pipei
                "ext  v2.16b, v0.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}
                "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei


                "fmul v8.4s, v0.4s, %[w0].s[0]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w0].s[1]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w0].s[2]            \n" // v2 * w00

                "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
                "ext  v6.16b, v4.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v4.4s, %[w1].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[1]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w1].s[2]            \n" // v2 * w00

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
                "bif  v0.16b, %[vzero].16b, %[wmask].16b    \n" //pipei

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "4:                                          \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [outptr] "+r"(doutr0_ptr), \
                  [vzero] "+w" (vzero), [w0] "+w" (wr0), [w1] "+w" (wr1), [cnt] "+r" (cnt), \
                  [mask1] "+w" (vmask_rp1), [mask2] "+w" (vmask_rp2), [wmask] "+w" (wmask), [vbias] "+w" (wbias)
                : [remain] "r" (cnt_remain)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
                );
            }

        }
    }


}
#else
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
    uint32x4_t vmask_rp = vcgtq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(0));
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

            float32x4_t wbias;
            if (flag_bias) {
                wbias  = vdupq_n_f32(bias[i]);
            } else {
                wbias = vdupq_n_f32(0.f);
            }

            wr0 = vsetq_lane_f32(0.f, wr0, 3);
            wr1 = vsetq_lane_f32(0.f, wr1, 3);
            wr2 = vsetq_lane_f32(0.f, wr2, 3);

            const float *dr0 = din_channel;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;

            const float *din0_ptr = dr0;
            const float *din1_ptr = dr1;
            const float *din2_ptr = dr2;

            float *doutr0 = dout_channel;

            //! top pad
            float32x4_t vzero= vdupq_n_f32(0.f);

            prefetch(din0_ptr);
            prefetch(din1_ptr);

            float *doutr0_ptr = doutr0;
            // todo
            float32x4_t din0_1234 = vld1q_f32(din0_ptr);
            float32x4_t din1_1234 = vld1q_f32(din1_ptr);

            din0_ptr += 3;
            din1_ptr += 3;

            float32x4_t din0_0123 = vextq_f32(vzero, din0_1234, 3);
            float32x4_t din1_0123 = vextq_f32(vzero, din1_1234, 3);

            float32x4_t din0_2340 = vextq_f32(din0_1234, vzero, 1);
            float32x4_t din1_2340 = vextq_f32(din1_1234, vzero, 1);

            float32x4_t sum0 = vmulq_f32(din0_0123, wr1);
            float32x4_t sum1 = vmulq_f32(din0_2340, wr1);

            prefetch(din0_ptr);
            prefetch(din1_ptr);

            sum0 = vmlaq_f32(sum0, din1_0123, wr2);
            sum1 = vmlaq_f32(sum1, din1_2340, wr2);

            float32x2_t vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
            float32x2_t vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            float32x2_t vsum = vpadd_f32(vsum0, vsum1);

            vsum = vadd_f32(vsum, vget_low_f32(wbias));


            vst1_f32(doutr0_ptr, vsum);

            doutr0_ptr += 2;

            //mid
            int cnt = cnt_col;
            for (;cnt > 0; cnt--){
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);

                float32x4_t din0_5678 = vld1q_f32(din0_ptr + 4);
                float32x4_t din1_5678 = vld1q_f32(din1_ptr + 4);

                din0_ptr += 4;
                din1_ptr += 4;

                float32x4_t din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                float32x4_t din1_3456 = vextq_f32(din1_1234, din1_5678, 2);

                sum0 = vmulq_f32(din0_1234, wr1);
                sum1 = vmulq_f32(din0_3456, wr1);

                prefetch(din0_ptr);
                prefetch(din1_ptr);
                
                sum0 = vmlaq_f32(sum0, din1_1234, wr2);
                sum1 = vmlaq_f32(sum1, din1_3456, wr2);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                vst1_f32(doutr0_ptr, vsum);

                doutr0_ptr += 2;
            }
            //right
            din0_1234 = vld1q_f32(din0_ptr);
            din1_1234 = vld1q_f32(din1_ptr);

            float32x4_t din0_5678 = vld1q_f32(din0_ptr + 4);
            float32x4_t din1_5678 = vld1q_f32(din1_ptr + 4);

            uint32x4_t vone = vdupq_n_u32(-1);
            uint32x4_t vmask_rp_r2 = vextq_u32(vone, vmask_rp, 2);
            uint32x4_t vmask_rp_l2 = vextq_u32(vmask_rp, vone, 2);

            din0_1234 = vbslq_f32(vmask_rp_r2, din0_1234, vzero);
            din1_1234 = vbslq_f32(vmask_rp_r2, din1_1234, vzero);

            din0_5678 = vbslq_f32(vmask_rp_l2, din0_5678, vzero);
            din1_5678 = vbslq_f32(vmask_rp_l2, din1_5678, vzero);

            float32x4_t din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
            float32x4_t din1_3456 = vextq_f32(din1_1234, din1_5678, 2);

            sum0 = vmulq_f32(din0_1234, wr1);
            sum1 = vmulq_f32(din0_3456, wr1);

            sum0 = vmlaq_f32(sum0, din1_1234, wr2);
            sum1 = vmlaq_f32(sum1, din1_3456, wr2);

            float32x2_t vdata = vld1_f32(doutr0_ptr);

            vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
            vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            vsum = vpadd_f32(vsum0, vsum1);

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;

            vsum = vadd_f32(vsum, vget_low_f32(wbias));

            vsum = vbsl_f32(vget_low_u32(mask_w), vsum, vdata);

            vst1_f32(doutr0_ptr, vsum);

            doutr0 += w_out;

            //! process mid rows
            for (int j = h_out - size_pad_bottom - 1; j > 0; j--) {
                // todo
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                doutr0_ptr = doutr0;


                prefetch(din0_ptr);
                prefetch(din1_ptr);
                prefetch(din2_ptr);

                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                float32x4_t din2_1234 = vld1q_f32(din2_ptr);


                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                float32x4_t din0_2345 = vextq_f32(din0_1234, vzero, 1);

                din1_0123 = vextq_f32(vzero, din1_1234, 3);
                float32x4_t din1_2345 = vextq_f32(din1_1234, vzero, 1);

                float32x4_t din2_0123 = vextq_f32(vzero, din2_1234, 3);
                float32x4_t din2_2345 = vextq_f32(din2_1234, vzero, 1);

                sum0 = vmulq_f32(din0_0123, wr0);
                sum1 = vmulq_f32(din0_2345, wr0);

                sum0 =vmlaq_f32(sum0, din1_0123, wr1);
                sum1 =vmlaq_f32(sum1, din1_2345, wr1);

                sum0 =vmlaq_f32(sum0, din2_0123, wr2);
                sum1 =vmlaq_f32(sum1, din2_2345, wr2);

                din0_ptr += 3;
                din1_ptr += 3;
                din2_ptr += 3;

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                prefetch(din0_ptr);
                prefetch(din1_ptr);
                prefetch(din2_ptr);

                vst1_f32(doutr0_ptr, vsum);

                doutr0_ptr += 2;

                //mid
                cnt = cnt_col;
                for (;cnt > 0; cnt--){
                    din0_1234 = vld1q_f32(din0_ptr);
                    din1_1234 = vld1q_f32(din1_ptr);
                    din2_1234 = vld1q_f32(din2_ptr);

                    float32x4_t din0_5678 = vld1q_f32(din0_ptr + 4);
                    float32x4_t din1_5678 = vld1q_f32(din1_ptr + 4);
                    float32x4_t din2_5678 = vld1q_f32(din2_ptr + 4);

                    float32x4_t din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                    float32x4_t din1_3456 = vextq_f32(din1_1234, din1_5678, 2);
                    float32x4_t din2_3456 = vextq_f32(din2_1234, din2_5678, 2);

                    sum0 = vmulq_f32(din0_1234, wr0);
                    sum1 = vmulq_f32(din0_3456, wr0);

                    din0_ptr += 4;
                    din1_ptr += 4;

                    sum0 = vmlaq_f32(sum0, din1_1234, wr1);
                    sum1 = vmlaq_f32(sum1, din1_3456, wr1);

                    din2_ptr += 4;
                    prefetch(din0_ptr);

                    sum0 = vmlaq_f32(sum0, din2_1234, wr2);
                    sum1 = vmlaq_f32(sum1, din2_3456, wr2);

                    prefetch(din1_ptr);
                    prefetch(din2_ptr);

                    vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                    vsum = vpadd_f32(vsum0, vsum1);

                    vsum = vadd_f32(vsum, vget_low_f32(wbias));

                    vst1_f32(doutr0_ptr, vsum);

                    doutr0_ptr += 2;
                }
            //right
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                din2_1234 = vld1q_f32(din2_ptr);
                
                vmask_rp_r2 = vextq_u32(vone, vmask_rp, 2);
                vmask_rp_l2 = vextq_u32(vmask_rp, vone, 2);

                din0_5678 = vld1q_f32(din0_ptr + 4);
                din1_5678 = vld1q_f32(din1_ptr + 4);
                float32x4_t din2_5678 = vld1q_f32(din2_ptr + 4);

                din0_1234 = vbslq_f32(vmask_rp_r2, din0_1234, vzero);
                din1_1234 = vbslq_f32(vmask_rp_r2, din1_1234, vzero);
                din2_1234 = vbslq_f32(vmask_rp_r2, din2_1234, vzero);

                din0_5678 = vbslq_f32(vmask_rp_l2, din0_5678, vzero);
                din1_5678 = vbslq_f32(vmask_rp_l2, din1_5678, vzero);
                din2_5678 = vbslq_f32(vmask_rp_l2, din2_5678, vzero);

                sum0 = vmulq_f32(din0_1234, wr0);
          
                din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                din1_3456 = vextq_f32(din1_1234, din1_5678, 2);
                float32x4_t din2_3456 = vextq_f32(din2_1234, din2_5678, 2);

                sum1 = vmulq_f32(din0_3456, wr0);

                sum0 = vmlaq_f32(sum0, din1_1234, wr1);
                sum1 = vmlaq_f32(sum1, din1_3456, wr1);

                sum0 = vmlaq_f32(sum0, din2_1234, wr2);
                sum1 = vmlaq_f32(sum1, din2_3456, wr2);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                float32x2_t vdata = vld1_f32(doutr0_ptr);

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                vsum = vbsl_f32(vget_low_u32(mask_w), vsum, vdata);
            
                vst1_f32(doutr0_ptr, vsum);

                doutr0 += w_out;
            } // end of process mid rows

            // process bottom pad if needed
            if (size_pad_bottom) {
                din0_ptr = dr0;
                din1_ptr = dr1;
                doutr0_ptr = doutr0;

                prefetch(din0_ptr);
                prefetch(din1_ptr);

                // todo
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);


                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);

                float32x4_t din0_2345 = vextq_f32(din0_1234, vzero, 1);
                float32x4_t din1_2345 = vextq_f32(din1_1234, vzero, 1);

                sum0 = vmulq_f32(din0_0123, wr0);
                sum1 = vmulq_f32(din0_2345, wr0);

                sum0 =vmlaq_f32(sum0, din1_0123, wr1);
                sum1 =vmlaq_f32(sum1, din1_2345, wr1);

                din0_ptr += 3;
                din1_ptr += 3;

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);
                
                prefetch(din0_ptr);
                prefetch(din1_ptr);
                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                vst1_f32(doutr0_ptr, vsum);

                doutr0_ptr += 2;

                //mid
                cnt = cnt_col;
                for (;cnt > 0; cnt--){
                    din0_1234 = vld1q_f32(din0_ptr);
                    din1_1234 = vld1q_f32(din1_ptr);

                    float32x4_t din0_5678 = vld1q_f32(din0_ptr + 4);
                    float32x4_t din1_5678 = vld1q_f32(din1_ptr + 4);

                    float32x4_t din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                    float32x4_t din1_3456 = vextq_f32(din1_1234, din1_5678, 2);

                    sum0 = vmulq_f32(din0_1234, wr0);
                    sum1 = vmulq_f32(din0_3456, wr0);

                    din0_ptr += 4;
                    din1_ptr += 4;

                    sum0 = vmlaq_f32(sum0, din1_1234, wr1);
                    sum1 = vmlaq_f32(sum1, din1_3456, wr1);

                    prefetch(din0_ptr);
                    prefetch(din1_ptr);

                    vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                    vsum = vpadd_f32(vsum0, vsum1);

                    vsum = vadd_f32(vsum, vget_low_f32(wbias));

                    vst1_f32(doutr0_ptr, vsum);

                    doutr0_ptr += 2;
                }
            //right
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);

                vmask_rp_r2 = vextq_u32(vone, vmask_rp, 2);
                vmask_rp_l2 = vextq_u32(vmask_rp, vone, 2);

                din0_5678 = vld1q_f32(din0_ptr + 4);
                din1_5678 = vld1q_f32(din1_ptr + 4);

                din0_1234 = vbslq_f32(vmask_rp_r2, din0_1234, vzero);
                din1_1234 = vbslq_f32(vmask_rp_r2, din1_1234, vzero);
                
                din0_5678 = vbslq_f32(vmask_rp_l2, din0_5678, vzero);
                din1_5678 = vbslq_f32(vmask_rp_l2, din1_5678, vzero);

                sum0 = vmulq_f32(din0_1234, wr0);

                din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                din1_3456 = vextq_f32(din1_1234, din1_5678, 2);

                sum1 = vmulq_f32(din0_3456, wr0);
                sum0 = vmlaq_f32(sum0, din1_1234, wr1);

                sum1 = vmlaq_f32(sum1, din1_3456, wr1);

                float32x2_t vdata = vld1_f32(doutr0_ptr);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                vsum = vbsl_f32(vget_low_u32(mask_w), vsum, vdata);

                vst1_f32(doutr0_ptr, vsum);

            } // end of process bottom pad

        }
    }
}
#endif
#else

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
            int cnt = cnt_col;
            asm volatile(
            // process left pad
            "pld [%[din0_ptr], #128]                @ preload data\n"
                    "pld [%[din1_ptr], #128]                @ preload data\n"
                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r2\n"
                    "vmov.u32 q11, #0                       @ for left pad\n"

                    "vext.32 q7, q11, q10, #3               @ shift right 1 data\n"
                    "vext.32  q14, q10, q11, #1              @ shift left r1\n"
                    "vmul.f32 q8, q7, %q[wr1]               @ mul weight 1, out0\n"
                    "vmul.f32 q9, q14, %q[wr1]               @ mul weight 1, out1\n"

                    "vext.32 q7, q11, q12, #3               @ shift right 1 data\n"
                    "vext.32  q14, q12, q11, #1              @ shift left r2\n"
                    "vmla.f32 q8, q7, %q[wr2]               @ mul weight 2, out0\n"
                    "vmla.f32 q9, q14,  %q[wr2]              @ mul weight 2, out1\n"

                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "pld [%[din0_ptr], #192]                @ preload data\n"

                    "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                    "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                    "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                    "vadd.f32  d17, d16, %e[bias]           @ add bias \n"

                    "sub %[din0_ptr], #4                    @ 1pad + 2 float data overlap\n"
                    "sub %[din1_ptr], #4                    @ 1pad + 2 float data overlap\n"
                    "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                    // process mid cols
                    "cmp %[cnt], #1                         @ check whether has mid loop\n"
                    "blt  2f                                @ jump to rightpad\n"
                    "1:                                     @ main loop start point\n"
                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"

                    "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                    "vext.32  q14, q12, q13, #2              @ shift left r2\n"

                    "vmul.f32 q8, q10, %q[wr1]              @ mul weight 1, out0\n"
                    "vmul.f32 q9, q7, %q[wr1]               @ mul weight 1, outr1\n"

                    "vmla.f32 q8, q12, %q[wr2]              @ mul weight 2, outr0\n"
                    "vmla.f32 q9, q14, %q[wr2]               @ mul weight 2, outr1\n"


                    "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                    "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                    "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                    "vadd.f32  d17, d16, %e[bias]           @ add bias \n"

                    "sub %[din0_ptr], #8                    @ 1 float data overlap and 1 redundant\n"
                    "sub %[din1_ptr], #8                    @ 1 float data overlap and 1 redundant\n"
                    "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    1b                              @ jump to main loop start point\n"

                    //! process right pad
                    "2:                                     @ right pad entry\n"
                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r2\n"
                    "vld1.32  {d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d26}, [%[din1_ptr]]!     @ load din r1\n"

                    "vmov.u32  d31, #0                      @ zero buf\n"

                    "vbif d21, d31, %e[mask_din]            @ bit select, deal with right pad\n"
                    "vbif d22, d31, %f[mask_din]            @ bit select, deal with right pad\n"

                    "vmul.f32 q8, q10, %q[wr1]              @ mul weight 1, out0\n"
                    "vext.32  q7, q10, q11, #2              @ shift left r1\n"

                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vbif d25, d31, %e[mask_din]            @ bit select, deal with right pad\n"
                    "vbif d26, d31, %f[mask_din]            @ bit select, deal with right pad\n"

                    "vmul.f32 q9, q7, %q[wr1]               @ mul weight 1, out1\n"

                    "vext.32  q14, q12, q13, #2              @ shift left r1\n"
                    "pld [%[dout_ptr1], #64]                @ preload data\n"

                    "vmla.f32 q8, q12, %q[wr2]              @ mul weight 2, outr0\n"
                    "vmla.f32 q9, q14, %q[wr2]               @ mul weight 2, outr1\n"

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

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;

            //! process mid rows
            for (int j = h_out - size_pad_bottom - 1; j > 0; j--) {

                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;

                cnt = cnt_col;
                asm volatile(
                // process left pad
                "pld [%[din0_ptr], #128]                @ preload data\n"
                        "pld [%[din1_ptr], #128]                @ preload data\n"
                        "pld [%[din2_ptr], #128]                @ preload data\n"

                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0\n"
                        "vmov.u32 q11, #0                       @ for left pad\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1\n"
                        "vld1.32  {d28-d29}, [%[din2_ptr]]!     @ load din r2\n"

                        "vext.32 q7, q11, q10, #3               @ shift right 1 data\n"
                        "vext.32  q13, q10, q11, #1              @ shift left r0\n"
                        "pld [%[din0_ptr], #128]                @ preload data\n"
                        "vmul.f32 q8, q7, %q[wr0]               @ mul weight 00, outr0\n"
                        "vmul.f32 q9, q13, %q[wr0]               @ mul weight 00, outr1\n"

                        "vext.32 q7, q11, q12, #3               @ shift right 1 data\n"
                        "vext.32  q13, q12, q11, #1              @ shift left r1\n"
                        "pld [%[din1_ptr], #128]                @ preload data\n"
                        "vmla.f32 q8, q7, %q[wr1]               @ mul weight 10, outr0\n"
                        "vmla.f32 q9, q13,  %q[wr1]              @ mul weight 10, outr1\n"

                        "vext.32 q7, q11, q14, #3               @ shift right 1 data\n"
                        "vext.32  q13, q14, q11, #1              @ shift left r2\n"
                        "pld [%[din2_ptr], #128]                @ preload data\n"
                        "vmla.f32 q8, q7, %q[wr2]               @ mul weight 20, outr0\n"
                        "vmla.f32 q9, q13,  %q[wr2]              @ mul weight 20, outr1\n"

                        "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"

                        "sub %[din0_ptr], #4                    @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #4                    @ 1pad + 2 float data overlap\n"
                        "sub %[din2_ptr], #4                    @ 1pad + 2 float data overlap\n"
                        "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"


                        // process mid cols
                        "cmp %[cnt], #1                         @ check whether has mid loop\n"
                        "blt  2f                                @ jump to rightpad\n"
                        "1:                                     @ main loop start point\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r0\n"
                        "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r1\n"
                        "vld1.32  {d28-d30}, [%[din2_ptr]]!    @ load din r2\n"

                        "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                        "vext.32  q11, q12, q13, #2              @ shift left r2\n"
                        "vmul.f32 q8, q10, %q[wr0]              @ mul weight 00, outr0\n"
                        "vmul.f32 q9, q7, %q[wr0]               @ mul weight 00, outr1\n"

                        "vext.32  q7, q14, q15, #2      @ shift left r2\n"
                        "pld [%[din0_ptr], #128]                @ preload data\n"
                        "vmla.f32 q8, q12, %q[wr1]              @ mul weight 10, outr0\n"
                        "vmla.f32 q9, q11, %q[wr1]               @ mul weight 10, outr1\n"

                        "pld [%[din1_ptr], #128]                @ preload data\n"
                        "pld [%[din2_ptr], #128]                @ preload data\n"
                        "vmla.f32 q8, q14, %q[wr2]  @mul weight 10, outr0\n"
                        "vmla.f32 q9, q7, %q[wr2]  @mul weight 10, outr1\n"

                        "vpadd.f32 d22, d16, d17  @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19 @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23 @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"

                        "sub %[din0_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din2_ptr], #8 @ 2 float data overlap with previous data\n"
                        "vst1.32  {d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                        "subs %[cnt], #1 @ loop count minus 1\n"
                        "bne    1b                          @ jump to main loop start point\n"

                        // process right pad
                        "2:                                 @ right pad entry\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!    @ load din r1\n"
                        "vmov.u32  d31, #0                  @ zero buf\n"
                        "vld1.32  {d24-d26}, [%[din1_ptr]]!    @ load din r1\n"
                        "vld1.32  {d28-d30}, [%[din2_ptr]]!    @ load din r1\n"

                        "vbif d21, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vbif d22, d31, %f[mask_din] @ bit select, deal with right pad\n"

                        "vext.32  q7, q10, q11, #2      @ shift left r0\n"
                        "vbif d25, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vbif d26, d31, %f[mask_din] @ bit select, deal with right pad\n"
                        "vmul.f32 q8, q10, %q[wr0]  @mul weight 00, outr0\n"
                        "vmul.f32 q9, q7, %q[wr0]  @mul weight 00, outr1\n"

                        "vext.32  q7, q12, q13, #2      @ shift left r1\n"
                        "vbif d29, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vbif d30, d31, %f[mask_din] @ bit select, deal with right pad\n"
                        "vmla.f32 q8, q12, %q[wr1]  @mul weight 10, outr0\n"
                        "vmla.f32 q9, q7, %q[wr1]  @mul weight 10, outr1\n"

                        "pld [%[dout_ptr1], #64]         @ preload ouput data\n"
                        "vext.32  q7, q14, q15, #2      @ shift left r2\n"
                        "vmla.f32 q8, q14, %q[wr2]  @mul weight 20, outr0\n"
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

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
            } // end of process mid rows

            // process bottom pad if needed
            if (size_pad_bottom) {
                din0_ptr = dr0;
                din1_ptr = dr1;

                cnt = cnt_col;
                asm volatile(
                // process left pad
                "pld [%[din0_ptr], #128]               @ preload data\n"
                        "pld [%[din1_ptr], #128]               @ preload data\n"

                        "vld1.32  {d20-d21}, [%[din0_ptr]]!    @ load din r0\n"
                        "vmov.u32 q11, #0 @ for left pad\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!    @ load din r1\n"

                        "vext.32 q7, q11, q10, #3 @ shift right 1 data\n"
                        "vext.32  q13, q10, q11, #1   @ shift left r0\n"
                        "vmul.f32 q8, q7, %q[wr0]  @mul weight 00, outr0\n"
                        "vmul.f32 q9, q13, %q[wr0]    @mul weight 00, outr1\n"

                        "vext.32 q7, q11, q12, #3 @ shift right 1 data\n"
                        "vext.32  q13, q12, q11, #1   @ shift left r1\n"
                        "vmla.f32 q8, q7, %q[wr1]   @mul weight 10, outr0\n"
                        "vmla.f32 q9, q13,  %q[wr1]   @mul weight 10, outr1\n"

                        "vpadd.f32 d22, d16, d17  @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19  @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23  @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"

                        "sub %[din0_ptr], #4 @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #4 @ 1pad + 2 float data overlap\n"
                        "vst1.32  {d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                        // process mid cols
                        "cmp %[cnt], #1                             @ check whether has mid loop\n"
                        "blt  2f                                @ jump to rightpad\n"
                        "1:                                     @ main loop start point\n"
                        "pld [%[din0_ptr], #192]               @ preload data\n"
                        "pld [%[din1_ptr], #192]               @ preload data\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!    @ load din r0\n"
                        "vld1.32  {d24-d26}, [%[din1_ptr]]!    @ load din r1\n"

                        "vext.32  q7, q10, q11, #2      @ shift left r1\n"
                        "vmul.f32 q8, q10, %q[wr0]      @mul weight 00, outr0\n"
                        "vmul.f32 q9, q7, %q[wr0]  @mul weight 00, outr1\n"

                        "vext.32  q7, q12, q13, #2      @ shift left r2\n"
                        "vmla.f32 q8, q12, %q[wr1]  @mul weight 10, outr0\n"
                        "vmla.f32 q9, q7, %q[wr1]  @mul weight 10, outr1\n"

                        "vpadd.f32 d22, d16, d17  @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19 @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23 @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"

                        "sub %[din0_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8 @ 2 float data overlap with previous data\n"

                        "vst1.32  {d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                        "subs %[cnt], #1 @ loop count minus 1\n"
                        "bne    1b @ jump to main loop start point\n"

                        // process right pad
                        "2:             @ right pad entry\n"
                        "pld [%[din0_ptr], #192]               @ preload data\n"
                        "pld [%[din1_ptr], #192]               @ preload data\n"
                        "vld1.32  {d20-d21}, [%[din0_ptr]]!    @ load din r1\n"
                        "vld1.32  {d22}, [%[din0_ptr]]!    @ load din r1\n"
                        "vmov.u32  d31, #0 @ zero buf\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!    @ load din r1\n"
                        "vld1.32  {d26}, [%[din1_ptr]]!    @ load din r1\n"

                        "vbif d21, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vbif d22, d31, %f[mask_din] @ bit select, deal with right pad\n"

                        "vext.32  q7, q10, q11, #2      @ shift left r0\n"
                        "vmul.f32 q8, q10, %q[wr0]  @mul weight 00, outr0\n"
                        "vmul.f32 q9, q7, %q[wr0]  @mul weight 00, outr1\n"

                        "vbif d25, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vbif d26, d31, %f[mask_din] @ bit select, deal with right pad\n"

                        "pld [%[dout_ptr1], #64]         @ preload data\n"

                        "vext.32  q7, q12, q13, #2      @ shift left r1\n"
                        "vmla.f32 q8, q12, %q[wr1]  @mul weight 10, outr0\n"
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

            } // end of process bottom pad

        }
    }
}
#endif

#ifdef __aarch64__
//4line
void conv_depthwise_3x3s1p1_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {

   // printf("3x3s1 mult height \n");
    //! pad is done implicit
    const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    //! for 4x6 convolution window
    const unsigned int right_pad_idx[8] = {5, 4, 3, 2, 1, 0, 0, 0};

    //printf("conv3x3_dw start \n");

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = 9;

    int tile_w = (w_in + 3) >> 2;
    int tile_h = (h_in + 3) >> 2;
    int cnt_col = tile_w - 2;

    unsigned int size_pad_right = (unsigned int)(1 + (tile_w << 2) - w_in);
    int size_pad_bottom = (unsigned int)(1 + (tile_h << 2) - h_in);

    uint32x4_t vmask_rp1 = vcgeq_u32(vld1q_u32(right_pad_idx), vdupq_n_u32(size_pad_right));
    uint32x4_t vmask_rp2 = vcgeq_u32(vld1q_u32(right_pad_idx + 4), vdupq_n_u32(size_pad_right));
    uint32x4_t vmask_result = vcgtq_u32(vld1q_u32(right_pad_idx), vdupq_n_u32(size_pad_right));

     for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_in; c ++) {
            float* dout_ptr = dout_batch + c * size_out_channel;

            const float* din_ch_ptr = din_batch + c * size_in_channel;

            float bias_val = flag_bias ? bias[c] : 0.f;
        
            const float* wei_ptr = weights + c * w_stride;

            float32x4_t wr0 = vld1q_f32(wei_ptr);
            float32x4_t wr1 = vld1q_f32(wei_ptr + 3);
            float32x4_t wr2 = vld1q_f32(wei_ptr + 6);

            float *doutr0 = dout_ptr;
            float *doutr1 = doutr0 + w_out;
            float *doutr2 = doutr1 + w_out;
            float *doutr3 = doutr2 + w_out;

            const float *dr0 = din_ch_ptr;
            const float *dr1 = dr0 + w_in;
            const float *dr2 = dr1 + w_in;
            const float *dr3 = dr2 + w_in;
            const float *dr4 = dr3 + w_in;
            const float *dr5 = dr4 + w_in;

            const float *din_ptr0 = dr0;
            const float *din_ptr1 = dr1;
            const float *din_ptr2 = dr2;
            const float *din_ptr3 = dr3;
            const float *din_ptr4 = dr4;
            const float *din_ptr5 = dr5;

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

            float* ptr_zero = const_cast<float*>(zero);
            float32x4_t vzero = vdupq_n_f32(0.f);

            //top
            int h = 0;
            if(1){
                float32x4_t voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                float32x4_t voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                float32x4_t voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                float32x4_t voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                //din data
                float32x4_t vinr0 = vld1q_f32(din_ptr0);
                float32x4_t vinr0_1 = vld1q_f32(din_ptr0 + 4);
                float32x4_t vinr1 = vld1q_f32(din_ptr1);
                float32x4_t vinr1_1 = vld1q_f32(din_ptr1 + 4);
                float32x4_t vinr2 = vld1q_f32(din_ptr2);
                float32x4_t vinr2_1 = vld1q_f32(din_ptr2 + 4);

                //r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr1, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr0, wr0, 1);

                float32x4_t vinr3 = vld1q_f32(din_ptr3);
                float32x4_t vinr3_1 = vld1q_f32(din_ptr3 + 4);
                float32x4_t vinr4 = vld1q_f32(din_ptr4);
                float32x4_t vinr4_1 = vld1q_f32(din_ptr4 + 4);

                //r1
                voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr2, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr1, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr1, wr0, 1);

              //  float32x4_t vinr5 = vld1q_f32(din_ptr5);
              //  float32x4_t vinr5_1 = vld1q_f32(din_ptr5 + 4);

                //r2
                voutr3 = vmlaq_laneq_f32(voutr3, vinr2, wr0, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr2, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr1, 1);

                // r0, r1, r2 shift left 2345
                float32x4_t vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                float32x4_t vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                float32x4_t vtmp2 = vextq_f32(vinr2, vinr2_1, 1);

                //r3
                voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr1, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr2, 1);

                float32x4_t vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                float32x4_t vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
               // float32x4_t vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                //r4
                voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr2, 1);

                //r0
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp0, wr0, 2);

                din_ptr0 += 3;
                prefetch(din_ptr0);

                //r1
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp1, wr0, 2);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr2, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr1, 2);

                // r0, r1, r2 shift right 0123
                vtmp0 = vextq_f32(vzero, vinr0, 3);
                vtmp1 = vextq_f32(vzero, vinr1, 3);

                //r2
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp2, wr0, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr2, 2);

                din_ptr1 += 3;
                prefetch(din_ptr1);

                //r3
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr1, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr2, 2);

                vtmp2 = vextq_f32(vzero, vinr2, 3);
                vtmp3 = vextq_f32(vzero, vinr3, 3);

                //r4
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr2, 2);

                vtmp4 = vextq_f32(vzero, vinr4, 3);
                //vtmp5 = vextq_f32(vzero, vinr5, 3);

                 //r0
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr1, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp0, wr0, 0);

                din_ptr2 += 3;
                prefetch(din_ptr2);
                //r1
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp1, wr0, 0);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr2, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr1, 0);

                din_ptr3 += 3;
                prefetch(din_ptr3);
                //r2
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp2, wr0, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr1, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr2, 0);
               
                din_ptr4 += 3;
                prefetch(din_ptr4);
                //r3
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr1, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr2, 0);

              //  din_ptr5 += 3;
              //  prefetch(din_ptr5);
                //r4
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr2, 0);

                voutr0 = vmaxq_f32(voutr0, vzero);
                voutr1 = vmaxq_f32(voutr1, vzero);
                voutr2 = vmaxq_f32(voutr2, vzero);
                voutr3 = vmaxq_f32(voutr3, vzero);

                vst1q_f32(doutr0, voutr0);
                vst1q_f32(doutr1, voutr1);
                vst1q_f32(doutr2, voutr2);
                vst1q_f32(doutr3, voutr3);

                doutr0 += 4;
                doutr1 += 4;
                doutr2 += 4;
                doutr3 += 4;

                //mid col
                for (int j = 0; j < cnt_col; ++j) {
                    voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                    voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                    voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                   //din data
                    vinr0 = vld1q_f32(din_ptr0);
                    vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    vinr1 = vld1q_f32(din_ptr1);
                    vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    vinr2 = vld1q_f32(din_ptr2);
                    vinr2_1 = vld1q_f32(din_ptr2 + 4);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr0, wr0, 0);

                    vinr3 = vld1q_f32(din_ptr3);
                    vinr3_1 = vld1q_f32(din_ptr3 + 4);
                    vinr4 = vld1q_f32(din_ptr4);
                    vinr4_1 = vld1q_f32(din_ptr4 + 4);

                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr2, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr1, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr1, wr0, 0);

                   // vinr5 = vld1q_f32(din_ptr5);
                    //vinr5_1 = vld1q_f32(din_ptr5 + 4);

                    // r0, r1, r2 shift left 2345
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                    vtmp2 = vextq_f32(vinr2, vinr2_1, 1);

                    //r2 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr2, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr1, 0);
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr2, wr0, 0);

                    vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                    vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                  //  vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                    //r3 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr2, 0);
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr1, 0);

                     //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp0, wr0, 1);

                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr2, 0);
                

                    din_ptr0 += 4;
                    prefetch(din_ptr0);
                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr2, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr1, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp1, wr0, 1);

                    din_ptr1 += 4;
                    prefetch(din_ptr1);
                    //r2 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp2, wr0, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr2, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr1, 1);

                     // r0, r1, r2 shift left 3456
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 2);
                    vtmp2 = vextq_f32(vinr2, vinr2_1, 2);

                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr1, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr2, 1);
                    
                    din_ptr2 += 4;
                    prefetch(din_ptr2);

                    ///r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr1, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp0, wr0, 2);
                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr2, 1);

                    vtmp3 = vextq_f32(vinr3, vinr3_1, 2);
                    vtmp4 = vextq_f32(vinr4, vinr4_1, 2);
                   // vtmp5 = vextq_f32(vinr5, vinr5_1, 2);


                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr2, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr1, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp1, wr0, 2);
                    
                    din_ptr3 += 4;
                    prefetch(din_ptr3);
                    //r2 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp2, wr0, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr2, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr1, 2);
                   
                    din_ptr4 += 4;
                    prefetch(din_ptr4);
                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr1, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr2, 2);

                   // din_ptr5 += 4;
                   // prefetch(din_ptr5);
                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr2, 2);


                    voutr0 = vmaxq_f32(voutr0, vzero);
                    voutr1 = vmaxq_f32(voutr1, vzero);
                    voutr2 = vmaxq_f32(voutr2, vzero);
                    voutr3 = vmaxq_f32(voutr3, vzero);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);
                    vst1q_f32(doutr2, voutr2);
                    vst1q_f32(doutr3, voutr3);

                    doutr0 += 4;
                    doutr1 += 4;
                    doutr2 += 4;
                    doutr3 += 4;

                }
                //right
                voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                //din data
                vinr0 = vld1q_f32(din_ptr0);
                vinr0_1 = vld1q_f32(din_ptr0 + 4);
                vinr1 = vld1q_f32(din_ptr1);
                vinr1_1 = vld1q_f32(din_ptr1 + 4);
                vinr2 = vld1q_f32(din_ptr2);
                vinr2_1 = vld1q_f32(din_ptr2 + 4);

                vinr0 = vbslq_f32(vmask_rp1, vinr0, vzero);
                vinr0_1 = vbslq_f32(vmask_rp2, vinr0_1, vzero);

                vinr1 = vbslq_f32(vmask_rp1, vinr1, vzero);
                vinr1_1 = vbslq_f32(vmask_rp2, vinr1_1, vzero);

                vinr2 = vbslq_f32(vmask_rp1, vinr2, vzero);
                vinr2_1 = vbslq_f32(vmask_rp2, vinr2_1, vzero);

                vinr3 = vld1q_f32(din_ptr3);
                vinr3_1 = vld1q_f32(din_ptr3 + 4);
                vinr4 = vld1q_f32(din_ptr4);
                vinr4_1 = vld1q_f32(din_ptr4 + 4);
               // vinr5 = vld1q_f32(din_ptr5);
               // vinr5_1 = vld1q_f32(din_ptr5 + 4);

                //r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr1, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr0, wr0, 0);

                vinr3 = vbslq_f32(vmask_rp1, vinr3, vzero);
                vinr3_1 = vbslq_f32(vmask_rp2, vinr3_1, vzero);

                vinr4 = vbslq_f32(vmask_rp1, vinr4, vzero);
                vinr4_1 = vbslq_f32(vmask_rp2, vinr4_1, vzero);
                    
               // vinr5 = vbslq_f32(vmask_rp1, vinr5, vzero);
               // vinr5_1 = vbslq_f32(vmask_rp2, vinr5_1, vzero);

                //r1 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr2, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr1, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr1, wr0, 0);

                // r0, r1, r2 shift left 2345
                vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                //r2 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vinr2, wr0, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr2, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr1, 0);

                vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                //r3 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr1, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr2, 0);

                vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                //r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr1, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp0, wr0, 1);

                //r4 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr2, 0);
                    
                vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
               // vtmp5 = vextq_f32(vinr5, vinr5_1, 1);
                    
                //r1 1234
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp1, wr0, 1);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr2, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr1, 1);

                // r0, r1, r2 shift left 3456
                vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                //r2 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp2, wr0, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr1, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr2, 1);

                vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                //r3 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr1, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr2, 1);
             
                vtmp3 = vextq_f32(vinr3, vinr3_1, 2);

                ///r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp0, wr0, 2);

                //r4 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr2, 1);

                vtmp4 = vextq_f32(vinr4, vinr4_1, 2);
               // vtmp5 = vextq_f32(vinr5, vinr5_1, 2);
                
                //r1 1234
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp1, wr0, 2);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr2, 2); 
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr1, 2);
                
                //r2 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp2, wr0, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr2, 2);

                vinr0 = vld1q_f32(doutr0);
                vinr1 = vld1q_f32(doutr1);
                vinr2 = vld1q_f32(doutr2);
                vinr3 = vld1q_f32(doutr3);
                //r3 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr1, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr2, 2);

                dr0 = dr3;
                dr1 = dr4;
                dr2 = dr5;
                //r4 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr2, 2);

                voutr0 = vmaxq_f32(voutr0, vzero);
                voutr1 = vmaxq_f32(voutr1, vzero);
                voutr2 = vmaxq_f32(voutr2, vzero);
                voutr3 = vmaxq_f32(voutr3, vzero);

                voutr0 = vbslq_f32(vmask_result, voutr0, vinr0);
                voutr1 = vbslq_f32(vmask_result, voutr1, vinr1);
                voutr2 = vbslq_f32(vmask_result, voutr2, vinr2);
                voutr3 = vbslq_f32(vmask_result, voutr3, vinr3);


                vst1q_f32(doutr0, voutr0);
                vst1q_f32(doutr1, voutr1);
                vst1q_f32(doutr2, voutr2);
                vst1q_f32(doutr3, voutr3);

                dr3 = dr2 + w_in;
                dr4 = dr3 + w_in;
                dr5 = dr4 + w_in;
                dout_ptr = dout_ptr + 4 * w_out;
            }
            //mid
            for (h = 0; h < tile_h - 2; h++) {

                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                doutr2 = doutr1 + w_out;
                doutr3 = doutr2 + w_out;

                din_ptr0 = dr0;
                din_ptr1 = dr1;
                din_ptr2 = dr2;
                din_ptr3 = dr3;
                din_ptr4 = dr4;
                din_ptr5 = dr5;


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
                
                float32x4_t voutr0 = vdupq_n_f32(bias_val);  //vld1q_f32(doutr0);
                float32x4_t voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                float32x4_t voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                float32x4_t voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                //din data
                float32x4_t vinr0 = vld1q_f32(din_ptr0);
                float32x4_t vinr0_1 = vld1q_f32(din_ptr0 + 4);
                float32x4_t vinr1 = vld1q_f32(din_ptr1);
                float32x4_t vinr1_1 = vld1q_f32(din_ptr1 + 4);
                float32x4_t vinr2 = vld1q_f32(din_ptr2);
                float32x4_t vinr2_1 = vld1q_f32(din_ptr2 + 4);

                //r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 1);

                float32x4_t vinr3 = vld1q_f32(din_ptr3);
                float32x4_t vinr3_1 = vld1q_f32(din_ptr3 + 4);
                float32x4_t vinr4 = vld1q_f32(din_ptr4);
                float32x4_t vinr4_1 = vld1q_f32(din_ptr4 + 4);

                //r1
                voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 1);

                float32x4_t vinr5 = vld1q_f32(din_ptr5);
                float32x4_t vinr5_1 = vld1q_f32(din_ptr5 + 4);


                //r2
                voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 1);
                voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 1);

                // r0, r1, r2 shift left 2345
                float32x4_t vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                float32x4_t vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                float32x4_t vtmp2 = vextq_f32(vinr2, vinr2_1, 1);

                //r3
                voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr0, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 1);

                //r0
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                float32x4_t vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                float32x4_t vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                float32x4_t vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                //r4
                voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr1, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 1);

                //r1
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);

                //r5
                voutr3 = vmlaq_laneq_f32(voutr3, vinr5, wr2, 1);

                din_ptr0 += 3;
                prefetch(din_ptr0);

                //r2
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);

                din_ptr1 += 3;
                prefetch(din_ptr1);
                //r3
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                // r0, r1, r2 shift right 0123
                vtmp0 = vextq_f32(vzero, vinr0, 3);
                vtmp1 = vextq_f32(vzero, vinr1, 3);
                vtmp2 = vextq_f32(vzero, vinr2, 3);

                //r4
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                 //r0
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 0);

                vtmp3 = vextq_f32(vzero, vinr3, 3);
                vtmp4 = vextq_f32(vzero, vinr4, 3);
                //r5
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 2);

                vtmp5 = vextq_f32(vzero, vinr5, 3);

                //r1
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 0);

                din_ptr2 += 3;
                prefetch(din_ptr2);

                //r2
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 0);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 0);
               
                din_ptr3 += 3;
                prefetch(din_ptr3);
                //r3
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 0);

                din_ptr4 += 3;
                prefetch(din_ptr4);
                //r4
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 0);

                din_ptr5 += 3;
                prefetch(din_ptr5);
                //r5
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 0);

                voutr0 = vmaxq_f32(voutr0, vzero);
                voutr1 = vmaxq_f32(voutr1, vzero);
                voutr2 = vmaxq_f32(voutr2, vzero);
                voutr3 = vmaxq_f32(voutr3, vzero);

                vst1q_f32(doutr0, voutr0);
                vst1q_f32(doutr1, voutr1);
                vst1q_f32(doutr2, voutr2);
                vst1q_f32(doutr3, voutr3);

                doutr0 += 4;
                doutr1 += 4;
                doutr2 += 4;
                doutr3 += 4;

                //mid col
                for (int j = 0; j < cnt_col; ++j) {
                    voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                    voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                    voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                   //din data
                    vinr0 = vld1q_f32(din_ptr0);
                    vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    vinr1 = vld1q_f32(din_ptr1);
                    vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    vinr2 = vld1q_f32(din_ptr2);
                    vinr2_1 = vld1q_f32(din_ptr2 + 4);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                    vinr3 = vld1q_f32(din_ptr3);
                    vinr3_1 = vld1q_f32(din_ptr3 + 4);
                    vinr4 = vld1q_f32(din_ptr4);
                    vinr4_1 = vld1q_f32(din_ptr4 + 4);
                    vinr5 = vld1q_f32(din_ptr5);
                    vinr5_1 = vld1q_f32(din_ptr5 + 4);


                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);

                    // r0, r1, r2 shift left 2345
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                    vtmp2 = vextq_f32(vinr2, vinr2_1, 1);

                    //r2 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 0);
                    
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                    vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                    vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr0, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);
                
                     //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                    din_ptr0 += 4;
                    prefetch(din_ptr0);

                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr1, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 0);

                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);

                    //r5 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr5, wr2, 0);

                     // r0, r1, r2 shift left 3456
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);

                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);

                    vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 2);

                    ///r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 1);

                    din_ptr1 += 4;
                    prefetch(din_ptr1);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                    //r5 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 1);

                    vtmp4 = vextq_f32(vinr4, vinr4_1, 2);
                    vtmp5 = vextq_f32(vinr5, vinr5_1, 2);

                    din_ptr2 += 4;
                    prefetch(din_ptr2);
                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    din_ptr3 += 4;
                    prefetch(din_ptr3);
                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                    din_ptr4 += 4;
                    prefetch(din_ptr4);
                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                    din_ptr5 += 4;
                    prefetch(din_ptr5);

                    //r5 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 2);


                    voutr0 = vmaxq_f32(voutr0, vzero);
                    voutr1 = vmaxq_f32(voutr1, vzero);
                    voutr2 = vmaxq_f32(voutr2, vzero);
                    voutr3 = vmaxq_f32(voutr3, vzero);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);
                    vst1q_f32(doutr2, voutr2);
                    vst1q_f32(doutr3, voutr3);

                    doutr0 += 4;
                    doutr1 += 4;
                    doutr2 += 4;
                    doutr3 += 4;

                }
                //right
                voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                //din data
                vinr0 = vld1q_f32(din_ptr0);
                vinr0_1 = vld1q_f32(din_ptr0 + 4);
                vinr1 = vld1q_f32(din_ptr1);
                vinr1_1 = vld1q_f32(din_ptr1 + 4);
                vinr2 = vld1q_f32(din_ptr2);
                vinr2_1 = vld1q_f32(din_ptr2 + 4);

                vinr0 = vbslq_f32(vmask_rp1, vinr0, vzero);
                vinr0_1 = vbslq_f32(vmask_rp2, vinr0_1, vzero);

                vinr1 = vbslq_f32(vmask_rp1, vinr1, vzero);
                vinr1_1 = vbslq_f32(vmask_rp2, vinr1_1, vzero);

                vinr2 = vbslq_f32(vmask_rp1, vinr2, vzero);
                vinr2_1 = vbslq_f32(vmask_rp2, vinr2_1, vzero);

                //r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                vinr3 = vld1q_f32(din_ptr3);
                vinr3_1 = vld1q_f32(din_ptr3 + 4);
                vinr4 = vld1q_f32(din_ptr4);
                vinr4_1 = vld1q_f32(din_ptr4 + 4);
                vinr5 = vld1q_f32(din_ptr5);
                vinr5_1 = vld1q_f32(din_ptr5 + 4);
                
                //r1 1234
                voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                vinr3 = vbslq_f32(vmask_rp1, vinr3, vzero);
                vinr3_1 = vbslq_f32(vmask_rp2, vinr3_1, vzero);

                vinr4 = vbslq_f32(vmask_rp1, vinr4, vzero);
                vinr4_1 = vbslq_f32(vmask_rp2, vinr4_1, vzero);
                    
                vinr5 = vbslq_f32(vmask_rp1, vinr5, vzero);
                vinr5_1 = vbslq_f32(vmask_rp2, vinr5_1, vzero);

                //r2 1234
                voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);
                
                // r0, r1, r2 shift left 2345
                vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
               
                //r3 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr0, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 0);
                voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);

                //r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                //r4 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr1, 0);
                voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 0);

                //r1 1234
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                //r5 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vinr5, wr2, 0);

                // r0, r1, r2 shift left 3456
                vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                //r2 1234
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                //r3 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 1);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);
             
                ///r0 1234
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                vtmp3 = vextq_f32(vinr3, vinr3_1, 2);

                //r4 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 1);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 1);
                
                //r1 1234
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);
                vtmp4 = vextq_f32(vinr4, vinr4_1, 2); 

               //r5 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 1);

                vtmp5 = vextq_f32(vinr5, vinr5_1, 2);
              
                //r2 1234
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                vinr0 = vld1q_f32(doutr0);
                vinr1 = vld1q_f32(doutr1);
                //r3 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);
                
                vinr2 = vld1q_f32(doutr2);
                vinr3 = vld1q_f32(doutr3);

                //r4 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 2);
                voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                dr0 = dr4;
                dr1 = dr5;
                dr2 = dr1 + w_in;
                //r5 1234
                voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 2);


                voutr0 = vmaxq_f32(voutr0, vzero);
                voutr1 = vmaxq_f32(voutr1, vzero);
                voutr2 = vmaxq_f32(voutr2, vzero);
                voutr3 = vmaxq_f32(voutr3, vzero);
                
                voutr0 = vbslq_f32(vmask_result, voutr0, vinr0);
                voutr1 = vbslq_f32(vmask_result, voutr1, vinr1);
                voutr2 = vbslq_f32(vmask_result, voutr2, vinr2);
                voutr3 = vbslq_f32(vmask_result, voutr3, vinr3);



                dr3 = dr2 + w_in;
                dr4 = dr3 + w_in;
                dr5 = dr4 + w_in;

                vst1q_f32(doutr0, voutr0);
                vst1q_f32(doutr1, voutr1);
                vst1q_f32(doutr2, voutr2);
                vst1q_f32(doutr3, voutr3);

                dout_ptr = dout_ptr + 4 * w_out;
            }
            //bottom
            if(1){
                din_ptr0 = dr0;
                din_ptr1 = dr1;
                doutr0 = dout_ptr;
                doutr1 = doutr0 + w_out;
                doutr2 = doutr1 + w_out;
                doutr3 = doutr2 + w_out;

                prefetch(doutr0);

                prefetch(din_ptr0);
                prefetch(din_ptr1);

                if (size_pad_bottom == 1){//only 4 line
                    din_ptr2 = dr2;
                    din_ptr3 = dr3;
                    din_ptr4 = dr4;
                    din_ptr5 = ptr_zero;
                    
                    prefetch(doutr1);
                    prefetch(doutr2);
                    prefetch(doutr3);

                    prefetch(din_ptr2);
                    prefetch(din_ptr3);
                    prefetch(din_ptr4);
                    prefetch(din_ptr5);

                    float32x4_t voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    float32x4_t voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                    float32x4_t voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                    float32x4_t voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                    //din data
                    float32x4_t vinr0 = vld1q_f32(din_ptr0);
                    float32x4_t vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    float32x4_t vinr1 = vld1q_f32(din_ptr1);
                    float32x4_t vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    float32x4_t vinr2 = vld1q_f32(din_ptr2);
                    float32x4_t vinr2_1 = vld1q_f32(din_ptr2 + 4);

                     //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 1);

                    float32x4_t vinr3 = vld1q_f32(din_ptr3);
                    float32x4_t vinr3_1 = vld1q_f32(din_ptr3 + 4);
                    float32x4_t vinr4 = vld1q_f32(din_ptr4);
                    float32x4_t vinr4_1 = vld1q_f32(din_ptr4 + 4);
                    float32x4_t vinr5 = vld1q_f32(din_ptr5);
                    float32x4_t vinr5_1 = vld1q_f32(din_ptr5 + 4);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 1);

                    // r0, r1, r2 shift left 2345
                    float32x4_t vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    float32x4_t vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                    //r2
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 1);

                    float32x4_t vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                    float32x4_t vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                    //r3
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr0, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 1);

                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    float32x4_t vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                    float32x4_t vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                    //r4
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr1, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 1);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                    // r0, r1, r2 shift right 0123
                    vtmp0 = vextq_f32(vzero, vinr0, 3);
                    vtmp1 = vextq_f32(vzero, vinr1, 3);

                    //r5
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr5, wr2, 1);

                    //r2
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    din_ptr0 += 3;
                    prefetch(din_ptr0);
                    //r3
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                    din_ptr1 += 3;
                    prefetch(din_ptr1);  
                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 0);
                    //r4
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                    vtmp2 = vextq_f32(vzero, vinr2, 3);
                    vtmp3 = vextq_f32(vzero, vinr3, 3);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 0);
                    //r5
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 2);

                    vtmp4 = vextq_f32(vzero, vinr4, 3);
                    vtmp5 = vextq_f32(vzero, vinr5, 3);

                    //r2
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 0);
               
                    din_ptr2 += 3;
                    prefetch(din_ptr2);
                    //r3
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 0);

                    din_ptr3 += 3;
                    prefetch(din_ptr3);
                    //r4
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 0);

                    din_ptr4 += 3;
                    prefetch(din_ptr4);

                   // din_ptr5 += 3;
                    prefetch(din_ptr5);
                    //r5
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 0);

                    voutr0 = vmaxq_f32(voutr0, vzero);
                    voutr1 = vmaxq_f32(voutr1, vzero);
                    voutr2 = vmaxq_f32(voutr2, vzero);
                    voutr3 = vmaxq_f32(voutr3, vzero);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);
                    vst1q_f32(doutr2, voutr2);
                    vst1q_f32(doutr3, voutr3);

                    doutr0 += 4;
                    doutr1 += 4;
                    doutr2 += 4;
                    doutr3 += 4;

                    //mid col
                    for (int j = 0; j < cnt_col; ++j) {
                        voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                        voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                        voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                        voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                        //din data
                        vinr0 = vld1q_f32(din_ptr0);
                        vinr0_1 = vld1q_f32(din_ptr0 + 4);
                        vinr1 = vld1q_f32(din_ptr1);
                        vinr1_1 = vld1q_f32(din_ptr1 + 4);
                        vinr2 = vld1q_f32(din_ptr2);
                        vinr2_1 = vld1q_f32(din_ptr2 + 4);

                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                        vinr3 = vld1q_f32(din_ptr3);
                        vinr3_1 = vld1q_f32(din_ptr3 + 4);
                        vinr4 = vld1q_f32(din_ptr4);
                        vinr4_1 = vld1q_f32(din_ptr4 + 4);
                        vinr5 = vld1q_f32(din_ptr5);
                        vinr5_1 = vld1q_f32(din_ptr5 + 4);


                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                        // r0, r1, r2 shift left 2345
                        vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                        vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                        //r2 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 0);
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);

                        vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                        vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                        //r3 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr0, 0);
                        voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 0);
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);

                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                        vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                        vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                        //r4 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr1, 0);
                        voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 0);

                        //r1 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);

                        din_ptr0 += 4;
                        prefetch(din_ptr0);
                        //r5 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vinr5, wr2, 0);
                    

                        //r2 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 1);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                        // r0, r1, r2 shift left 3456
                        vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                        vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                        //r3 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 1);
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 1);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);
                    
                        ///r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);
                        vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                        vtmp3 = vextq_f32(vinr3, vinr3_1, 2);

                        //r4 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 1);
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 1);

                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                        din_ptr1 += 4;
                        prefetch(din_ptr1);
                        //r5 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 1);

                        vtmp4 = vextq_f32(vinr4, vinr4_1, 2);
                        vtmp5 = vextq_f32(vinr5, vinr5_1, 2);

                        //r2 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                        din_ptr2 += 4;
                        prefetch(din_ptr2);
                        //r3 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 2);
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                        din_ptr3 += 4;
                        prefetch(din_ptr3);
                        //r4 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 2);
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                        din_ptr4 += 4;
                        prefetch(din_ptr4);
                        //r5 1234
                        voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 2);
                    
                       // din_ptr5 += 4;
                        prefetch(din_ptr5);

                        voutr0 = vmaxq_f32(voutr0, vzero);
                        voutr1 = vmaxq_f32(voutr1, vzero);
                        voutr2 = vmaxq_f32(voutr2, vzero);
                        voutr3 = vmaxq_f32(voutr3, vzero);

                        vst1q_f32(doutr0, voutr0);
                        vst1q_f32(doutr1, voutr1);
                        vst1q_f32(doutr2, voutr2);
                        vst1q_f32(doutr3, voutr3);

                        doutr0 += 4;
                        doutr1 += 4;
                        doutr2 += 4;
                        doutr3 += 4;

                    }
                    //right
                    voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                    voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);
                    voutr3 = vdupq_n_f32(bias_val); //vld1q_f32(doutr3);

                    //din data
                    vinr0 = vld1q_f32(din_ptr0);
                    vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    vinr1 = vld1q_f32(din_ptr1);
                    vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    vinr2 = vld1q_f32(din_ptr2);
                    vinr2_1 = vld1q_f32(din_ptr2 + 4);

                    vinr0 = vbslq_f32(vmask_rp1, vinr0, vzero);
                    vinr0_1 = vbslq_f32(vmask_rp2, vinr0_1, vzero);

                    vinr1 = vbslq_f32(vmask_rp1, vinr1, vzero);
                    vinr1_1 = vbslq_f32(vmask_rp2, vinr1_1, vzero);

                    vinr2 = vbslq_f32(vmask_rp1, vinr2, vzero);
                    vinr2_1 = vbslq_f32(vmask_rp2, vinr2_1, vzero);

                    vinr3 = vld1q_f32(din_ptr3);
                    vinr3_1 = vld1q_f32(din_ptr3 + 4);
                    vinr4 = vld1q_f32(din_ptr4);
                    vinr4_1 = vld1q_f32(din_ptr4 + 4);
                    vinr5 = vld1q_f32(din_ptr5);
                    vinr5_1 = vld1q_f32(din_ptr5 + 4);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                    vinr3 = vbslq_f32(vmask_rp1, vinr3, vzero);
                    vinr3_1 = vbslq_f32(vmask_rp2, vinr3_1, vzero);

                    vinr4 = vbslq_f32(vmask_rp1, vinr4, vzero);
                    vinr4_1 = vbslq_f32(vmask_rp2, vinr4_1, vzero);
                    
                    vinr5 = vbslq_f32(vmask_rp1, vinr5, vzero);
                    vinr5_1 = vbslq_f32(vmask_rp2, vinr5_1, vzero);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                     // r0, r1, r2 shift left 2345
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);
                    
                    vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr3, wr0, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                    vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                    vtmp5 = vextq_f32(vinr5, vinr5_1, 1);

                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr4, wr1, 0);
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 0);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                    // r0, r1, r2 shift left 3456
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                     //r5 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vinr5, wr2, 0);
                    

                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                    vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);
             
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 2);
                    ///r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 1);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 1);
                    
                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                    //r5 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 1);


                    vtmp4 = vextq_f32(vinr4, vinr4_1, 2);
                    vtmp5 = vextq_f32(vinr5, vinr5_1, 2);
                    
                
                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    vinr0 = vld1q_f32(doutr0);
                    vinr1 = vld1q_f32(doutr1);
                    //r3 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp3, wr0, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                    vinr2 = vld1q_f32(doutr2);
                    vinr3 = vld1q_f32(doutr3);
                    //r4 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp4, wr1, 2);
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                    //r5 1234
                    voutr3 = vmlaq_laneq_f32(voutr3, vtmp5, wr2, 2);

                    voutr0 = vmaxq_f32(voutr0, vzero);
                    voutr1 = vmaxq_f32(voutr1, vzero);
                    voutr2 = vmaxq_f32(voutr2, vzero);
                    voutr3 = vmaxq_f32(voutr3, vzero);
                
                    voutr0 = vbslq_f32(vmask_result, voutr0, vinr0);
                    voutr1 = vbslq_f32(vmask_result, voutr1, vinr1);
                    voutr2 = vbslq_f32(vmask_result, voutr2, vinr2);
                    voutr3 = vbslq_f32(vmask_result, voutr3, vinr3);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);
                    vst1q_f32(doutr2, voutr2);
                    vst1q_f32(doutr3, voutr3);

                }
                if (size_pad_bottom == 2){//only 3 line
                    din_ptr2 = dr2;
                    din_ptr3 = dr3;
                    din_ptr4 = ptr_zero;
                    
                    prefetch(doutr1);
                    prefetch(doutr2);

                    prefetch(din_ptr2);
                    prefetch(din_ptr3);
                    prefetch(din_ptr4);

                    float32x4_t voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    float32x4_t voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                    float32x4_t voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);

                    //din data
                    float32x4_t vinr0 = vld1q_f32(din_ptr0);
                    float32x4_t vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    float32x4_t vinr1 = vld1q_f32(din_ptr1);
                    float32x4_t vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    float32x4_t vinr2 = vld1q_f32(din_ptr2);
                    float32x4_t vinr2_1 = vld1q_f32(din_ptr2 + 4);

                     //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 1);

                    float32x4_t vinr3 = vld1q_f32(din_ptr3);
                    float32x4_t vinr3_1 = vld1q_f32(din_ptr3 + 4);
                    float32x4_t vinr4 = vld1q_f32(din_ptr4);
                    float32x4_t vinr4_1 = vld1q_f32(din_ptr4 + 4);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 1);

                    // r0, r1, r2 shift left 2345
                    float32x4_t vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    float32x4_t vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                    //r2
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 1);

                    float32x4_t vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                    float32x4_t vtmp3 = vextq_f32(vinr3, vinr3_1, 1);
                    //r3
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 1);
                    
                    float32x4_t vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    din_ptr0 += 3;
                    prefetch(din_ptr0);

                    //r4
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 1);


                    din_ptr1 += 3;
                    prefetch(din_ptr1);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                    // r0, r1, r2 shift right 0123
                    vtmp0 = vextq_f32(vzero, vinr0, 3);
                    vtmp1 = vextq_f32(vzero, vinr1, 3);

                    //r2
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);
                    
                    din_ptr2 += 3;
                    prefetch(din_ptr2);

                    vtmp2 = vextq_f32(vzero, vinr2, 3);

                    //r3
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 0);
                    vtmp3 = vextq_f32(vzero, vinr3, 3);

                    //r4
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                    vtmp4 = vextq_f32(vzero, vinr4, 3);
                    
                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 0);

                    din_ptr3 += 3;
                    prefetch(din_ptr3);
                    //r2
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 0);
               
                   // din_ptr4 += 3;
                    prefetch(din_ptr4);
                    //r3
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 0);

                    //r4
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 0);


                    voutr0 = vmaxq_f32(voutr0, vzero);
                    voutr1 = vmaxq_f32(voutr1, vzero);
                    voutr2 = vmaxq_f32(voutr2, vzero);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);
                    vst1q_f32(doutr2, voutr2);

                    doutr0 += 4;
                    doutr1 += 4;
                    doutr2 += 4;

                    //mid col
                    for (int j = 0; j < cnt_col; ++j) {
                        voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                        voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                        voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);

                        //din data
                        vinr0 = vld1q_f32(din_ptr0);
                        vinr0_1 = vld1q_f32(din_ptr0 + 4);
                        vinr1 = vld1q_f32(din_ptr1);
                        vinr1_1 = vld1q_f32(din_ptr1 + 4);
                        vinr2 = vld1q_f32(din_ptr2);
                        vinr2_1 = vld1q_f32(din_ptr2 + 4);

                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                        vinr3 = vld1q_f32(din_ptr3);
                        vinr3_1 = vld1q_f32(din_ptr3 + 4);
                        vinr4 = vld1q_f32(din_ptr4);
                        vinr4_1 = vld1q_f32(din_ptr4 + 4);


                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);
                        
                        // r0, r1, r2 shift left 2345
                        vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                        vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                        //r2 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 0);
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);
  
                        vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                        vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                        //r3 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 0);
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);

                        vtmp4 = vextq_f32(vinr4, vinr4_1, 1);

                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                        din_ptr0 += 4;
                        prefetch(din_ptr0);

                        //r4 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 0);
                    
                        din_ptr1 += 4;
                        prefetch(din_ptr1);
                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                        // r0, r1, r2 shift left 3456
                        vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                        vtmp1 = vextq_f32(vinr1, vinr1_1, 2);
                        //r2 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 1);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                        din_ptr2 += 4;
                        prefetch(din_ptr2);
                        //r3 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 1);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);
                    
                        ///r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);
                        vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                        vtmp3 = vextq_f32(vinr3, vinr3_1, 2);
                        //r4 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 1);

                        vtmp4 = vextq_f32(vinr4, vinr4_1, 2);

                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                        din_ptr3 += 4;
                        prefetch(din_ptr3);

                        //r2 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                        prefetch(din_ptr4);
                        //r3 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                        //r4 1234
                        voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);

                       // din_ptr5 += 4;
                       // prefetch(din_ptr5);


                        voutr0 = vmaxq_f32(voutr0, vzero);
                        voutr1 = vmaxq_f32(voutr1, vzero);
                        voutr2 = vmaxq_f32(voutr2, vzero);

                        vst1q_f32(doutr0, voutr0);
                        vst1q_f32(doutr1, voutr1);
                        vst1q_f32(doutr2, voutr2);

                        doutr0 += 4;
                        doutr1 += 4;
                        doutr2 += 4;

                    }
                    //right
                    voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);
                    voutr2 = vdupq_n_f32(bias_val); //vld1q_f32(doutr2);

                    //din data
                    vinr0 = vld1q_f32(din_ptr0);
                    vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    vinr1 = vld1q_f32(din_ptr1);
                    vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    vinr2 = vld1q_f32(din_ptr2);
                    vinr2_1 = vld1q_f32(din_ptr2 + 4);

                    vinr0 = vbslq_f32(vmask_rp1, vinr0, vzero);
                    vinr0_1 = vbslq_f32(vmask_rp2, vinr0_1, vzero);

                    vinr1 = vbslq_f32(vmask_rp1, vinr1, vzero);
                    vinr1_1 = vbslq_f32(vmask_rp2, vinr1_1, vzero);

                    vinr2 = vbslq_f32(vmask_rp1, vinr2, vzero);
                    vinr2_1 = vbslq_f32(vmask_rp2, vinr2_1, vzero);

                    vinr3 = vld1q_f32(din_ptr3);
                    vinr3_1 = vld1q_f32(din_ptr3 + 4);
                    vinr4 = vld1q_f32(din_ptr4);
                    vinr4_1 = vld1q_f32(din_ptr4 + 4);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                    vinr3 = vbslq_f32(vmask_rp1, vinr3, vzero);
                    vinr3_1 = vbslq_f32(vmask_rp2, vinr3_1, vzero);

                    vinr4 = vbslq_f32(vmask_rp1, vinr4, vzero);
                    vinr4_1 = vbslq_f32(vmask_rp2, vinr4_1, vzero);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                     // r0, r1, r2 shift left 2345
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr2, wr0, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);

                    vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                    //r3 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr3, wr1, 0);
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                    //r4 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vinr4, wr2, 0);
                    
                    vtmp4 = vextq_f32(vinr4, vinr4_1, 1);
                
                    
                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                    // r0, r1, r2 shift left 3456
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                    vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                    //r3 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 1);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);
             
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 2);
                    ///r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);
                    //r4 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 1);

                    vtmp4 = vextq_f32(vinr4, vinr4_1, 2);
                
                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2); 

                
                    //r2 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp2, wr0, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    vinr0 = vld1q_f32(doutr0);
                    vinr1 = vld1q_f32(doutr1);
                    vinr2 = vld1q_f32(doutr2);
                    //r3 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp3, wr1, 2);
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                    //r4 1234
                    voutr2 = vmlaq_laneq_f32(voutr2, vtmp4, wr2, 2);
                
                    voutr0 = vmaxq_f32(voutr0, vzero);
                    voutr1 = vmaxq_f32(voutr1, vzero);
                    voutr2 = vmaxq_f32(voutr2, vzero);

                    voutr0 = vbslq_f32(vmask_result, voutr0, vinr0);
                    voutr1 = vbslq_f32(vmask_result, voutr1, vinr1);
                    voutr2 = vbslq_f32(vmask_result, voutr2, vinr2);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);
                    vst1q_f32(doutr2, voutr2);

                }
                if (size_pad_bottom == 3){//only 2 line
                    din_ptr2 = dr2;
                    din_ptr3 = ptr_zero;
                    
                    prefetch(doutr1);

                    prefetch(din_ptr2);

                    float32x4_t voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    float32x4_t voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);

                    //din data
                    float32x4_t vinr0 = vld1q_f32(din_ptr0);
                    float32x4_t vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    float32x4_t vinr1 = vld1q_f32(din_ptr1);
                    float32x4_t vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    float32x4_t vinr2 = vld1q_f32(din_ptr2);
                    float32x4_t vinr2_1 = vld1q_f32(din_ptr2 + 4);

                     //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 1);

                    float32x4_t vinr3 = vld1q_f32(din_ptr3);
                    float32x4_t vinr3_1 = vld1q_f32(din_ptr3 + 4);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 1);

                    // r0, r1, r2 shift left 2345
                    float32x4_t vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    float32x4_t vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                    //r2
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 1);

                    float32x4_t vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                    float32x4_t vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                    //r3
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 1);

                    din_ptr0 += 3;
                    prefetch(din_ptr0);
                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    din_ptr1 += 3;
                    prefetch(din_ptr1);
                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                    // r0, r1, r2 shift right 0123
                    vtmp0 = vextq_f32(vzero, vinr0, 3);
                    vtmp1 = vextq_f32(vzero, vinr1, 3);

                    //r2
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    din_ptr2 += 3;
                    prefetch(din_ptr2);
                    //r3
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                    vtmp2 = vextq_f32(vzero, vinr2, 3);
                    vtmp3 = vextq_f32(vzero, vinr3, 3);

                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 0);

                    //r1
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 0);

                   // din_ptr3 += 3;
                    prefetch(din_ptr3);
                    //r2
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 0);
               
                    //r3
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 0);

                    voutr0 = vmaxq_f32(voutr0, vzero);
                    voutr1 = vmaxq_f32(voutr1, vzero);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);

                    doutr0 += 4;
                    doutr1 += 4;

                    //mid col
                    for (int j = 0; j < cnt_col; ++j) {
                        voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                        voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);

                        //din data
                        vinr0 = vld1q_f32(din_ptr0);
                        vinr0_1 = vld1q_f32(din_ptr0 + 4);
                        vinr1 = vld1q_f32(din_ptr1);
                        vinr1_1 = vld1q_f32(din_ptr1 + 4);
                        vinr2 = vld1q_f32(din_ptr2);
                        vinr2_1 = vld1q_f32(din_ptr2 + 4);

                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                        vinr3 = vld1q_f32(din_ptr3);
                        vinr3_1 = vld1q_f32(din_ptr3 + 4);

                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                        // r0, r1, r2 shift left 2345
                        vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                        vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                        //r2 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);

                        vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                        vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                        //r3 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);
                    
                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                        din_ptr0 += 4;
                        prefetch(din_ptr0);

                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);
                        
                        // r0, r1, r2 shift left 3456
                        vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                        vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                        //r2 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                        din_ptr1 += 4;
                        prefetch(din_ptr1);
                        vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                        //r3 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);
                        
                        vtmp3 = vextq_f32(vinr3, vinr3_1, 2);

                        ///r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                        din_ptr2 += 4;
                        prefetch(din_ptr2);
                        //r1 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                       // din_ptr3 += 4;
                        prefetch(din_ptr3);
                        //r2 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                        //r3 1234
                        voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);

                        voutr0 = vmaxq_f32(voutr0, vzero);
                        voutr1 = vmaxq_f32(voutr1, vzero);

                        vst1q_f32(doutr0, voutr0);
                        vst1q_f32(doutr1, voutr1);

                        doutr0 += 4;
                        doutr1 += 4;

                    }
                    //right
                    voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);
                    voutr1 = vdupq_n_f32(bias_val); //vld1q_f32(doutr1);

                    //din data
                    vinr0 = vld1q_f32(din_ptr0);
                    vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    vinr1 = vld1q_f32(din_ptr1);
                    vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    vinr2 = vld1q_f32(din_ptr2);
                    vinr2_1 = vld1q_f32(din_ptr2 + 4);

                    vinr0 = vbslq_f32(vmask_rp1, vinr0, vzero);
                    vinr0_1 = vbslq_f32(vmask_rp2, vinr0_1, vzero);

                    vinr1 = vbslq_f32(vmask_rp1, vinr1, vzero);
                    vinr1_1 = vbslq_f32(vmask_rp2, vinr1_1, vzero);

                    vinr3 = vld1q_f32(din_ptr3);
                    vinr3_1 = vld1q_f32(din_ptr3 + 4);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                    vinr2 = vbslq_f32(vmask_rp1, vinr2, vzero);
                    vinr2_1 = vbslq_f32(vmask_rp2, vinr2_1, vzero);

                    vinr3 = vbslq_f32(vmask_rp1, vinr3, vzero);
                    vinr3_1 = vbslq_f32(vmask_rp2, vinr3_1, vzero);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr1, wr0, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                     // r0, r1, r2 shift left 2345
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    vtmp1 = vextq_f32(vinr1, vinr1_1, 1);

                    //r2 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr2, wr1, 0);
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);

                    vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                    vtmp3 = vextq_f32(vinr3, vinr3_1, 1);

                    //r3 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vinr3, wr2, 0);
                    
                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);
                    
                    // r0, r1, r2 shift left 3456
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 2);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                    vtmp1 = vextq_f32(vinr1, vinr1_1, 2);

                    //r2 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 1);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                    vtmp2 = vextq_f32(vinr2, vinr2_1, 2);
                    //r3 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 1);

                    vtmp3 = vextq_f32(vinr3, vinr3_1, 2);

                    ///r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    //r1 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp1, wr0, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2); 
                
                    vinr0 = vld1q_f32(doutr0);
                    vinr1 = vld1q_f32(doutr1);

                    //r2 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp2, wr1, 2);
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    //r3 1234
                    voutr1 = vmlaq_laneq_f32(voutr1, vtmp3, wr2, 2);
                    
                    voutr0 = vmaxq_f32(voutr0, vzero);
                    voutr1 = vmaxq_f32(voutr1, vzero);

                    voutr0 = vbslq_f32(vmask_result, voutr0, vinr0);
                    voutr1 = vbslq_f32(vmask_result, voutr1, vinr1);

                    vst1q_f32(doutr0, voutr0);
                    vst1q_f32(doutr1, voutr1);

                }
                if (size_pad_bottom == 4){//only 1 line
                    din_ptr2 = ptr_zero;

                    prefetch(din_ptr2);

                    float32x4_t voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);

                    //din data
                    float32x4_t vinr0 = vld1q_f32(din_ptr0);
                    float32x4_t vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    float32x4_t vinr1 = vld1q_f32(din_ptr1);
                    float32x4_t vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    float32x4_t vinr2 = vld1q_f32(din_ptr2);
                    float32x4_t vinr2_1 = vld1q_f32(din_ptr2 + 4);
                     //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 1);

                    float32x4_t vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    //r1
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 1);

                    float32x4_t vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                    //r2
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 1);

                    // r0, r1, r2 shift left 2345
                    float32x4_t vtmp2 = vextq_f32(vinr2, vinr2_1, 1);

                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    vtmp0 = vextq_f32(vzero, vinr0, 3);
                    //r1
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                    vtmp1 = vextq_f32(vzero, vinr1, 3);
                    //r2
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                    // r0, r1, r2 shift right 0123
                    vtmp2 = vextq_f32(vzero, vinr2, 3);

                    din_ptr0 += 3;
                    prefetch(din_ptr0);


                    //r0
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 0);

                    din_ptr1 += 3;
                    prefetch(din_ptr1);
                    //r1
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 0);

                    //din_ptr2 += 3;
                    prefetch(din_ptr2);
                    //r2
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 0);
                    
                    voutr0 = vmaxq_f32(voutr0, vzero);
                    vst1q_f32(doutr0, voutr0);

                    doutr0 += 4;

                    //mid col
                    for (int j = 0; j < cnt_col; ++j) {
                        voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);

                        //din data
                        vinr0 = vld1q_f32(din_ptr0);
                        vinr0_1 = vld1q_f32(din_ptr0 + 4);
                        vinr1 = vld1q_f32(din_ptr1);
                        vinr1_1 = vld1q_f32(din_ptr1 + 4);
                        vinr2 = vld1q_f32(din_ptr2);
                        vinr2_1 = vld1q_f32(din_ptr2 + 4);


                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                        vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                        //r1 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                        vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                        //r2 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);

                        // r0, r1, r2 shift left 2345
                        vtmp2 = vextq_f32(vinr2, vinr2_1, 1);

                
                        //r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);

                        vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                        //r1 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                        vtmp1 = vextq_f32(vinr1, vinr1_1, 2);
                        //r2 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);
                    
                        // r0, r1, r2 shift left 3456
                        vtmp2 = vextq_f32(vinr2, vinr2_1, 2);

                        din_ptr0 += 4;
                        prefetch(din_ptr0);

                        ///r0 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                        din_ptr1 += 4;
                        prefetch(din_ptr1);
                        //r1 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2);

                      //  din_ptr2 += 4;
                        prefetch(din_ptr2);
                        //r2 1234
                        voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);

                        voutr0 = vmaxq_f32(voutr0, vzero);
                        vst1q_f32(doutr0, voutr0);

                        doutr0 += 4;

                    }
                    //right
                    voutr0 = vdupq_n_f32(bias_val); //vld1q_f32(doutr0);

                    //din data
                    vinr0 = vld1q_f32(din_ptr0);
                    vinr0_1 = vld1q_f32(din_ptr0 + 4);
                    vinr1 = vld1q_f32(din_ptr1);
                    vinr1_1 = vld1q_f32(din_ptr1 + 4);
                    vinr2 = vld1q_f32(din_ptr2);
                    vinr2_1 = vld1q_f32(din_ptr2 + 4);

                    vinr0 = vbslq_f32(vmask_rp1, vinr0, vzero);
                    vinr0_1 = vbslq_f32(vmask_rp2, vinr0_1, vzero);

                    vinr1 = vbslq_f32(vmask_rp1, vinr1, vzero);
                    vinr1_1 = vbslq_f32(vmask_rp2, vinr1_1, vzero);

                    vinr2 = vbslq_f32(vmask_rp1, vinr2, vzero);
                    vinr2_1 = vbslq_f32(vmask_rp2, vinr2_1, vzero);

                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr0, wr0, 0);

                    vtmp0 = vextq_f32(vinr0, vinr0_1, 1);
                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr1, wr1, 0);

                    vtmp1 = vextq_f32(vinr1, vinr1_1, 1);
                    //r2 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vinr2, wr2, 0);
                    
                     // r0, r1, r2 shift left 2345
                    vtmp2 = vextq_f32(vinr2, vinr2_1, 1);
                
                    //r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 1);
                    
                    vtmp0 = vextq_f32(vinr0, vinr0_1, 2);
                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 1);

                    vtmp1 = vextq_f32(vinr1, vinr1_1, 2);
                    //r2 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 1);

                    // r0, r1, r2 shift left 3456
                    vtmp2 = vextq_f32(vinr2, vinr2_1, 2);

                    vinr0 = vld1q_f32(doutr0);
                    ///r0 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp0, wr0, 2);

                    //r1 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp1, wr1, 2); 
                
                    //r2 1234
                    voutr0 = vmlaq_laneq_f32(voutr0, vtmp2, wr2, 2);
                    //r3 1234
                    voutr0 = vmaxq_f32(voutr0, vzero);
                    voutr0 = vbslq_f32(vmask_result, voutr0, vinr0);

                    vst1q_f32(doutr0, voutr0);

                }
                
            }
        }
        
    }

}
#else

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

            int cnt = cnt_col;
            asm volatile(
            //! process left pad
            "pld [%[din0_ptr], #192]                @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "pld [%[din2_ptr], #192]                @ preload data\n"

                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r2\n"
                    "vld1.32  {d28-d29}, [%[din2_ptr]]!     @ load din r3\n"

                    "vld1.32  {d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vmul.f32 q5, q10, %e[wr0][1]           @ mul weight 00, outr1\n"
                    "vmul.f32 q4, q10, %e[wr1][1]           @ mul weight 10, outr0\n"

                    "vld1.32  {d26}, [%[din1_ptr]]!     @ load din r1\n"
                    "vld1.32  {d30}, [%[din2_ptr]]!     @ load din r1\n"
                    "vmla.f32 q5, q12, %e[wr1][1]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][1]           @ mul weight 20, outr0\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q14, %e[wr2][1]           @ mul weight 20, outr1\n"

                    "vext.32  q8, q12, q13, #1              @ shift left r2\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q14, q15, #1              @ shift left r3\n"
                    "pld [%[din2_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q8,  %f[wr1][0]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q8,  %f[wr2][0]           @ mul weight 21, outr0\n"

                    "vmov.u32 d31, #0 @ zero\n"
                    "vext.32  d16, d31, d20, #1             @ shift right r1\n"
                    "vext.32  d17, d20, d21, #1             @ shift right r1\n"
                    "vmla.f32 q5, q6,  %f[wr2][0]           @ mul weight 21, outr1\n"


                    "vext.32  d12, d31, d24, #1             @ shift right r0\n"
                    "vext.32  d13, d24, d25, #1             @ shift right r0\n"
                    "vmla.f32 q5, q8, %e[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q8, %e[wr1][0]            @ mul weight 12, outr0\n"


                    "vext.32  d16, d31, d28, #1             @ shift right r0\n"
                    "vext.32  d17, d28, d29, #1             @ shift right r0\n"
                    "vmla.f32 q5, q6,  %e[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][0]           @ mul weight 22, outr0\n"

                    "vmla.f32 q5, q8,  %e[wr2][0]           @ mul weight 22, outr1\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                    "vmax.f32 q8, q8, q4                    @ relu\n"
                    "vmax.f32 q9, q9, q4                    @ relu\n"

                    "sub %[din0_ptr], #12                   @ 1pad + 2 float data overlap\n"
                    "sub %[din1_ptr], #12                   @ 1pad + 2 float data overlap\n"
                    "sub %[din2_ptr], #12                   @ 1pad + 2 float data overlap\n"
                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    //! process mid cols
                    "cmp %[cnt], #1                         @ check whether has mid cols\n"
                    "blt  2f                                @ jump to right pad\n"
                    "1:                                     @ main loop start point\n"
                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r2\n"
                    "vld1.32  {d28-d29}, [%[din2_ptr]]!     @ load din r3\n"

                    "vld1.32  {d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmul.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "vld1.32  {d26}, [%[din1_ptr]]!     @ load din r1\n"
                    "vld1.32  {d30}, [%[din2_ptr]]!     @ load din r1\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q14, %e[wr2][0]           @ mul weight 20, outr1\n"

                    "vext.32  q8, q12, q13, #1              @ shift left r2\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q14, q15, #1              @ shift left r3\n"
                    "pld [%[din2_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q8,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q8,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q8, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q6,  %e[wr2][1]           @ mul weight 21, outr1\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q8, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q8, %f[wr1][0]            @ mul weight 12, outr0\n"


                    "vext.32  q8, q14, q15, #2              @ shift right r3\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "vmla.f32 q5, q8,  %f[wr2][0]           @ mul weight 22, outr1\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                    "vmax.f32 q8, q8, q4                    @ relu\n"
                    "vmax.f32 q9, q9, q4                    @ relu\n"

                    "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din2_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    1b                              @ jump to main loop start point\n"

                    //! process right pad
                    "2:                                     @ right pad entry\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r2\n"
                    "vld1.32  {d28-d29}, [%[din2_ptr]]!     @ load din r3\n"

                    "vld1.32  {d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d26}, [%[din1_ptr]]!     @ load din r1\n"

                    "vbif d21, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vbif d25, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vbif d29, d31, %e[mask]                @ bit select, deal with right pad\n"
                    
                    "vld1.32  {d30}, [%[din2_ptr]]!     @ load din r1\n"

                    "vbif  d22, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmul.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "vext.32  q6, q10, q11, #1              @ shift left r1\n"
                    "vbif  d26, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "vext.32  q8, q12, q13, #1              @ shift left r2\n"
                    "vbif  d30, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vmla.f32 q5, q14, %e[wr2][0]           @ mul weight 20, outr1\n"
                    
                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q6, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vext.32  q6, q14, q15, #1              @ shift left r3\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q8,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q8,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q8, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q6,  %e[wr2][1]           @ mul weight 21, outr1\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "pld [%[din2_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q8, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q8, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "vext.32  q8, q14, q15, #2              @ shift right r3\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "vmvn.32  d24, d31                      @ \n"
                    "vmvn.32  d25, d31                      @ \n"
                    "vmla.f32 q5, q8,  %f[wr2][0]           @ mul weight 22, outr1\n"

                    "pld [%[dout_ptr1], #128]               @ preload data\n"
                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                    "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                    "vmax.f32 q8, q8, q4                    @ relu\n"
                    "vmax.f32 q9, q9, q4                    @ relu\n"

                    "vext.32  q13, q12, %q[mask], #3        @ shift mask right 1\n"
                    "vbif q8, q10, q13                      @ bit select\n"

                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    
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

                cnt = cnt_col;
                asm volatile(
                //! process left pad
                "pld [%[din0_ptr], #192]               @ preload data\n"
                        "pld [%[din1_ptr], #192]               @ preload data\n"
                        "pld [%[din2_ptr], #192]               @ preload data\n"
                        "pld [%[din3_ptr], #192]               @ preload data\n"

                        "vld1.32  {d16-d17}, [%[din0_ptr]]!    @ load din r0\n"
                        "vld1.32  {d20-d21}, [%[din1_ptr]]!    @ load din r1\n"
                        "vld1.32  {d24-d25}, [%[din2_ptr]]!    @ load din r2\n"
                        "vld1.32  {d28-d29}, [%[din3_ptr]]!    @ load din r3\n"

                        "vld1.32  {d18}, [%[din0_ptr]]!    @ load din r0\n"

                        "vmul.f32 q4, q8, %e[wr0][1]   @mul weight 00, outr0\n"

                        "vld1.32  {d22}, [%[din1_ptr]]!    @ load din r0\n"
                        "vld1.32  {d26}, [%[din2_ptr]]!    @ load din r0\n"
                        "vmul.f32 q5, q10, %e[wr0][1]  @mul weight 00, outr1\n"
                        "vmla.f32 q4, q10, %e[wr1][1]  @mul weight 10, outr0\n"

                        "vext.32  q6, q8, q9, #1     @ shift left r0\n"
                        "vld1.32  {d30}, [%[din3_ptr]]!    @ load din r0\n"
                        "vmla.f32 q5, q12, %e[wr1][1]  @mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][1]  @mul weight 20, outr0\n"

                        "vmla.f32 q5, q14, %e[wr2][1]  @mul weight 20, outr1\n"

                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 01, outr0\n"

                        "vext.32  q6, q10, q11, #1   @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]  @mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]  @mul weight 11, outr0\n"

                        "vext.32  q6, q12, q13, #1  @ shift left r2\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 21, outr0\n"

                        "vext.32  q6, q14, q15, #1  @ shift left r3\n"
                        "vmov.u32 d31, #0 @ zero\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 21, outr1\n"

                        "vext.32  d12, d31, d16, #1     @ shift right r0\n"
                        "vext.32  d13, d16, d17, #1     @ shift right r0\n"
                        "vmla.f32 q4, q6, %e[wr0][0]  @mul weight 02, outr0\n"

                        "vext.32  d12, d31, d20, #1     @ shift right r1\n"
                        "vext.32  d13, d20, d21, #1     @ shift right r1\n"
                        "pld [%[din0_ptr], #192]               @ preload data\n"

                        "vmla.f32 q5, q6, %e[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][0]  @mul weight 12, outr0\n"

                        "vext.32  d12, d31, d24, #1     @ shift right r0\n"
                        "vext.32  d13, d24, d25, #1     @ shift right r0\n"
                        "pld [%[din1_ptr], #192]               @ preload data\n"

                        "vmla.f32 q5, q6,  %e[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][0] @mul weight 22, outr0\n"

                        "vext.32  d12, d31, d28, #1     @ shift right r0\n"
                        "vext.32  d13, d28, d29, #1     @ shift right r0\n"
                        "pld [%[din2_ptr], #192]               @ preload data\n"

                        "vmla.f32 q5, q6,  %e[wr2][0] @mul weight 22, outr1\n"

                        "pld [%[din3_ptr], #192]               @ preload data\n"

                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"
                        "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"

                        "sub %[din0_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #12 @ 1pad + 2 float data overlap\n"

                        "vmax.f32 q8, q8, q4                    @ relu\n"
                        "vmax.f32 q9, q9, q4                    @ relu\n"

                        "sub %[din2_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "sub %[din3_ptr], #12 @ 1pad + 2 float data overlap\n"

                        "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "vst1.32  {d18-d19},   [%[dout_ptr2]]!  @ store result, add pointer\n"

                        //! process mid cols
                        "cmp %[cnt], #1                             @ check whether has mid cols\n"
                        "blt  4f                                @ jump to right pad\n"
                        "3:                                     @ main loop start point\n"
                        "vld1.32  {d16-d17}, [%[din0_ptr]]!    @ load din r0\n"
                        "vld1.32  {d20-d21}, [%[din1_ptr]]!    @ load din r1\n"
                        "vld1.32  {d24-d25}, [%[din2_ptr]]!    @ load din r2\n"
                        "vld1.32  {d28-d29}, [%[din3_ptr]]!    @ load din r3\n"

                        "vld1.32  {d18}, [%[din0_ptr]]!    @ load din r0\n"

                        "vmul.f32 q4, q8, %e[wr0][0]    @ mul weight 00, outr0\n"
                         
                        "vld1.32  {d22}, [%[din1_ptr]]!    @ load din r0\n"
                        "vmul.f32 q5, q10, %e[wr0][0]   @ mul weight 00, outr1\n"
                        "vmla.f32 q4, q10, %e[wr1][0]   @ mul weight 10, outr0\n"

                        "vld1.32  {d26}, [%[din2_ptr]]!    @ load din r0\n"
                        "vld1.32  {d30}, [%[din3_ptr]]!    @ load din r0\n"
                        "vext.32  q6, q8, q9, #1        @ shift left r0\n"
                        "vmla.f32 q5, q12, %e[wr1][0]   @ mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][0]   @ mul weight 20, outr0\n"

                        "pld [%[din0_ptr], #192]               @ preload data\n"
                        "vmla.f32 q5, q14, %e[wr2][0]   @ mul weight 20, outr1\n"

                        "vmla.f32 q4, q6, %e[wr0][1]    @ mul weight 01, outr0\n"

                        "vext.32  q6, q10, q11, #1      @ shift left r1\n"
                        "pld [%[din1_ptr], #192]               @ preload data\n"
                        "vmla.f32 q5, q6, %e[wr0][1]    @ mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][1]    @ mul weight 11, outr0\n"

                        "vext.32  q6, q12, q13, #1      @ shift left r2\n"
                        "pld [%[din2_ptr], #192]               @ preload data\n"
                        "vmla.f32 q5, q6,  %e[wr1][1]   @ mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][1]   @ mul weight 21, outr0\n"

                        "vext.32  q6, q14, q15, #1      @ shift left r3\n"
                        "pld [%[din3_ptr], #192]               @ preload data\n"
                        "vmla.f32 q5, q6,  %e[wr2][1]   @ mul weight 21, outr1\n"

                        "vext.32  q6, q8, q9, #2        @ shift left r0\n"
                        "vext.32  q8, q10, q11, #2      @ shift left r1\n"
                        "vmla.f32 q4, q6, %f[wr0][0]    @ mul weight 02, outr0\n"

                        "vext.32  q6, q12, q13, #2      @ shift left r2\n"
                        "vmla.f32 q5, q8, %f[wr0][0]    @ mul weight 02, outr1\n"
                        "vmla.f32 q4, q8, %f[wr1][0]    @ mul weight 12, outr0\n"

                        "vext.32  q8, q14, q15, #2  @ shift right r3\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 22, outr0\n"

                        "vmla.f32 q5, q8,  %f[wr2][0] @mul weight 22, outr1\n"

                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                        "sub %[din0_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8 @ 2 float data overlap with previous data\n"

                        "vmax.f32 q8, q8, q4                    @ relu\n"
                        "vmax.f32 q9, q9, q4                    @ relu\n"

                        "sub %[din2_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din3_ptr], #8 @ 2 float data overlap with previous data\n"

                        "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "vst1.32  {d18-d19},   [%[dout_ptr2]]!  @ store result, add pointer\n"

                        "subs %[cnt], #1 @ loop count minus 1\n"
                        "bne    3b                             @ jump to main loop start point\n"

                        //! process right pad
                        "4:                                    @ right pad entry\n"
                        "vmov.u32  d31, #0                     @ zero buf\n"
                        "vld1.32  {d16-d17}, [%[din0_ptr]]!    @ load din r0\n"
                        "vld1.32  {d20-d21}, [%[din1_ptr]]!    @ load din r1\n"
                        "vld1.32  {d24-d25}, [%[din2_ptr]]!    @ load din r2\n"
                        "vld1.32  {d28-d29}, [%[din3_ptr]]!    @ load din r3\n"

                        "vbif d17, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vbif d21, d31, %e[mask]               @ bit select, deal with right pad\n"

                        "vld1.32  {d18}, [%[din0_ptr]]!    @ load din r0\n"
                        "vld1.32  {d22}, [%[din1_ptr]]!    @ load din r0\n"

                        "vmul.f32 q4, q8, %e[wr0][0]           @ mul weight 00, outr0\n"

                        "vbif d25, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vbif d29, d31, %e[mask]               @ bit select, deal with right pad\n"
                        "vmul.f32 q5, q10, %e[wr0][0]          @ mul weight 00, outr1\n"
                        "vmla.f32 q4, q10, %e[wr1][0]          @ mul weight 10, outr0\n"

                        "vld1.32  {d26}, [%[din2_ptr]]!    @ load din r0\n"
                        "vld1.32  {d30}, [%[din3_ptr]]!    @ load din r0\n"
                        "vmla.f32 q5, q12, %e[wr1][0]          @ mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][0]          @ mul weight 20, outr0\n"

                        "vbif  d18, d31, %f[mask]              @ bit select, deal with right pad\n"
                        "vbif  d22, d31, %f[mask]              @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q14, %e[wr2][0]          @ mul weight 20, outr1\n"

                        "vext.32  q6, q8, q9, #1               @ shift left r0\n"
                        "vbif  d26, d31, %f[mask] @ bit select, deal with right pad\n"
                        "vbif  d30, d31, %f[mask] @ bit select, deal with right pad\n"
                        "vmla.f32 q4, q6, %e[wr0][1]           @ mul weight 01, outr0\n"

                        "vext.32  q6, q10, q11, #1             @ shift left r1\n"
                        "pld [%[din0_ptr], #192]         @ preload data\n"
                        "vmla.f32 q5, q6, %e[wr0][1]           @ mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][1]           @ mul weight 11, outr0\n"

                        "vext.32  q6, q12, q13, #1  @ shift left r2\n"
                        "pld [%[din1_ptr], #192]         @ preload data\n"
                        "vmla.f32 q5, q6,  %e[wr1][1] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][1] @mul weight 21, outr0\n"

                        "vext.32  q6, q14, q15, #1  @ shift left r3\n"
                        "pld [%[din2_ptr], #192]         @ preload data\n"
                        "vmla.f32 q5, q6,  %e[wr2][1] @mul weight 21, outr1\n"

                        "vext.32  q6, q8, q9, #2     @ shift left r0\n"
                        "vext.32  q8, q10, q11, #2   @ shift left r1\n"
                        "pld [%[din3_ptr], #192]         @ preload data\n"
                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 02, outr0\n"

                        "vext.32  q6, q12, q13, #2  @ shift left r2\n"
                        "vmla.f32 q5, q8, %f[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q8, %f[wr1][0]  @mul weight 12, outr0\n"

                        "vext.32  q8, q14, q15, #2  @ shift right r3\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 22, outr0\n"

                        "pld [%[dout_ptr1], #128]         @ preload data\n"
                        "vmla.f32 q5, q8,  %f[wr2][0] @mul weight 22, outr1\n"

                        "vmvn.32  d22, d31 @ \n"
                        "vmvn.32  d23, d31 @ \n"
                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"

                        "vld1.32  {d20-d21}, [%[dout_ptr1]]    @ load dout r0\n"
                        "vext.32  q12, q11, %q[mask], #3 @ shift mask right 1\n"

                        "vmax.f32 q8, q8, q4                    @ relu\n"
                        "vmax.f32 q9, q9, q4                    @ relu\n"

                        "vbif q8, q10, q12 @ bit select\n"

                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!  @ store result, add pointer\n"
                        "vst1.32  {d16-d17}, [%[dout_ptr1]]!  @ store result, add pointer\n"

                        "sub %[dout_ptr2], %[dout_ptr2], %[pad_right] @ sub \n"
                        "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"

                :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                    [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                    [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                    [pad_right] "+r" (right_pad_sub), [cnt] "+r"(cnt)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask] "w" (vmask_rp)
                :"q4", "q5", "q6", "q8", "q9", \
                    "q10", "q11", "q12", "q13", "q14", "q15"
                );

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

            cnt = cnt_col;
            asm volatile(
            //! process left pad
            "pld [%[din0_ptr], #192]                        @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "pld [%[din2_ptr], #192]                @ preload data\n"

                    "vld1.32  {d16-d17}, [%[din0_ptr]]!     @ load din r0\n"
                    "vld1.32  {d20-d21}, [%[din1_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"

                    "vld1.32  {d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vld1.32  {d22}, [%[din1_ptr]]!     @ load din r0\n"
                   
                    "vmul.f32 q4, q8, %e[wr0][1]            @ mul weight 00, outr0\n"

                    //"vld1.32  {d26}, [%[din2_ptr]]!     @ load din r0\n"
                    "vmul.f32 q5, q10, %e[wr0][1]           @ mul weight 00, outr1\n"
                    "vmla.f32 q4, q10, %e[wr1][1]           @ mul weight 10, outr0\n"
                    
                    "vext.32  q6, q8, q9, #1                @ shift left r0\n"
                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q12, %e[wr1][1]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][1]           @ mul weight 20, outr0\n"

                    "vext.32  q14, q10, q11, #1              @ shift left r1\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 01, outr0\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmov.u32 d31, #0 @ zero\n"
                    "vmla.f32 q5, q14, %f[wr0][0]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q14, %f[wr1][0]            @ mul weight 11, outr0\n"

                    "vext.32  d28, d31, d16, #1             @ shift right r0\n"
                    "vext.32  d29, d16, d17, #1             @ shift right r0\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 21, outr0\n"

                    "vext.32  d12, d31, d20, #1             @ shift right r1\n"
                    "vext.32  d13, d20, d21, #1             @ shift right r1\n"
                    "vmla.f32 q4, q14, %e[wr0][0]            @ mul weight 02, outr0\n"

                    "vext.32  d28, d31, d24, #1             @ shift right r2\n"
                    "vext.32  d29, d24, d25, #1             @ shift right r2\n"
                    "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"

                    "pld [%[din2_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q14,  %e[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q14,  %e[wr2][0]           @ mul weight 22, outr0\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                    "vmax.f32 q8, q8, q4                    @ relu\n"
                    "vmax.f32 q9, q9, q4                    @ relu\n"

                    "sub %[din0_ptr], #12                   @ 1pad + 2 data overlap\n"
                    "sub %[din1_ptr], #12                   @ 1pad + 2 data overlap\n"

                    "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "beq    5f                              @ jump to next block\n"
                    "add %[din2_ptr], #12                   @ 1pad + 2 data overlap\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    // process mid cols
                    "5:                                     @ header of bottom process\n"
                    "cmp %[cnt], #1                         @ check whether has mid cols\n"
                    "blt  8f                                @ jump to right pad\n"
                    "6:                                     @ main loop start point\n"
                    "vld1.32  {d16-d17}, [%[din0_ptr]]!     @ load din r0\n"
                    "vld1.32  {d20-d21}, [%[din1_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"

                    "vld1.32  {d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vld1.32  {d22}, [%[din1_ptr]]!     @ load din r0\n"

                    "vmul.f32 q4, q8, %e[wr0][0]            @ mul weight 00, outr0\n"
                    
                   // "vld1.32  {d26}, [%[din2_ptr]]!     @ load din r0\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmla.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "vext.32  q6, q8, q9, #1                @ shift left r0\n"
                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "vext.32  q14, q10, q11, #1              @ shift left r1\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vmla.f32 q4, q6, %e[wr0][1]            @ mul weight 01, outr0\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "pld [%[din2_ptr], #192]                @ preload data\n"
                    "vmla.f32 q5, q14, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q14, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vext.32  q14, q8, q9, #2                @ shift left r0\n"
                    "vmla.f32 q5, q6,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q4, q14, %f[wr0][0]            @ mul weight 02, outr0\n"


                    "vext.32  q14, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"


                    "vmla.f32 q5, q14,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q14,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                    "vmax.f32 q8, q8, q4                    @ relu\n"
                    "vmax.f32 q9, q9, q4                    @ relu\n"

                    "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "beq    7f                              @ jump to check point\n"
                    "add %[din2_ptr], #16                   @ point to 4 data ahead\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "7:                                     @ check point\n"
                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    6b                              @ jump to main loop start point\n"

                    // process right pad
                    "8:                                     @ right pad process\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                    "vld1.32  {d16-d17}, [%[din0_ptr]]!     @ load din r0\n"
                    "vld1.32  {d20-d21}, [%[din1_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"

                    "vbif d17, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vbif d21, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vbif d25, d31, %e[mask]                @ bit select, deal with right pad\n"

                    "vld1.32  {d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vld1.32  {d22}, [%[din1_ptr]]!     @ load din r1\n"

                    "vmul.f32 q4, q8, %e[wr0][0]            @ mul weight 00, outr0\n"

                    "vbif  d18, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vbif  d22, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vmul.f32 q5, q10, %e[wr0][0]           @ mul weight 00, outr1\n"
                    "vmla.f32 q4, q10, %e[wr1][0]           @ mul weight 10, outr0\n"

                    "vext.32  q6, q8, q9, #1                @ shift left r0\n"
                    "vbif  d26, d31, %f[mask]               @ bit select, deal with right pad\n"
                    "vmla.f32 q5, q12, %e[wr1][0]           @ mul weight 10, outr1\n"
                    "vmla.f32 q4, q12, %e[wr2][0]           @ mul weight 20, outr0\n"

                    "vext.32  q14, q10, q11, #1              @ shift left r1\n"
                    "vmla.f32 q4, q6, %e[wr0][1]            @ mul weight 01, outr0\n"

                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q14, %e[wr0][1]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q14, %e[wr1][1]            @ mul weight 11, outr0\n"

                    "vext.32  q14, q8, q9, #2                @ shift left r0\n"
                    "vmla.f32 q5, q6,  %e[wr1][1]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][1]           @ mul weight 21, outr0\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q4, q14, %f[wr0][0]            @ mul weight 02, outr0\n"

                    "vext.32  q14, q12, q13, #2              @ shift left r2\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "pld [%[dout_ptr1], #128]               @ preload data\n"
                    "pld [%[dout_ptr2], #128]               @ preload data\n"

                    "vmla.f32 q5, q14,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q14,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                    "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"

                    "vmvn.32  d24, d31                      @ \n"
                    "vmvn.32  d25, d31                      @ \n"
                    "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                    "vmov.u32 q4, #0                        @ dump zero to q4 for relu\n"
                    "vext.32  q13, q12, %q[mask], #3        @ shift mask right 1\n"
                    "vmax.f32 q8, q8, q4                    @ relu\n"
                    "vmax.f32 q9, q9, q4                    @ relu\n"

                    "vbif q8, q10, q13                      @ bit select\n"
                    "vbif q9, q11, q13                      @ bit select\n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"

                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "beq    9f                              @ jump to end point\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "9:                                     @ end\n"

            :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                [din2_ptr] "+r"(din2_ptr), [pad_right] "+r"(right_pad_sub), \
                [bot_pad] "+r"(size_pad_bottom), [cnt] "+r"(cnt)
            :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                [bias] "w"(wbias), [mask] "w"(vmask_rp)
            :"q4", "q5", "q6", "q8", "q9", \
                "q10", "q11", "q12", "q13", "q14", "q15"
            );

            //! end of processing bottom pad
        } // end of processing channels
    } // end of processing batchs
}
#endif
/**
 * \brief depthwise convolution kernel 3x3, stride 2, with reulu
 */
#ifdef __aarch64__
#if 1
//w_in > 7
void conv_depthwise_3x3s2p1_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {

    int right_pad_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    int out_pad_idx[4] = {0, 1, 2, 3};
    int size_pad_bottom = h_out * 2 - h_in;

    int cnt_col = (w_out >> 2) - 1;
    int cnt_remain = (w_out % 4);
    int size_right_remain = w_in - (7 + cnt_col * 8);

    int size_right_pad = w_out * 2 - w_in;

    uint32x4_t vmask_rp1 = vcgtq_s32(vdupq_n_s32(size_right_remain), vld1q_s32(right_pad_idx));//0 2 4 6
    uint32x4_t vmask_rp2 = vcgtq_s32(vdupq_n_s32(size_right_remain), vld1q_s32(right_pad_idx + 4));//1 3 5 7
    uint32x4_t wmask = vcgtq_s32(vdupq_n_s32(cnt_remain), vld1q_s32(out_pad_idx));//0 1 2 3
   // printf("w_out %d, cnt_col: %d, remain: %d \n", w_out, cnt_col, size_right_remain);
    //printf("mask1: %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3]);
    //printf("mask2: %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3]);
    //printf("wmask: %d, %d, %d, %d \n", wmask[0], wmask[1], wmask[2], wmask[3]);
   // size_right_remain *= sizeof(float);

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

            float32x4_t vzero= vdupq_n_f32(0.f);

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
            float *doutr0_ptr = doutr0;

            //! top pad
            if(1){
               int cnt = cnt_col;

               //printf("cnt_col: %d, remain: %d \n", cnt_col, size_right_remain);
             //  printf("mask1: %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3]);
             //  printf("mask2: %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3]);
              // printf("wmask: %d, %d, %d, %d \n", wmask[0], wmask[1], wmask[2], wmask[3]);

                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "ext  v2.16b, %[vzero].16b, v1.16b, #12     \n" // v2 = {0,1,3,5}
                "ext  v6.16b, %[vzero].16b, v5.16b, #12    \n" // v6 = {0,1,3,5}

                "fmul v8.4s, v0.4s, %[w1].s[1]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w1].s[2]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w1].s[0]            \n" // v2 * w00

                "sub %[inptr0], %[inptr0], #4            \n"
                "sub %[inptr1], %[inptr1], #4             \n"

                "fmla v8.4s, v4.4s, %[w2].s[1]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w2].s[2]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w2].s[0]            \n" // v2 * w00

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
                "fmax  v0.4s, v0.4s, %[vzero].4s             \n"

                "cmp %[cnt], #1                             \n"
                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "blt 1f                                     \n"
                //mid
                "2:                                          \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "ld2  {v2.4s, v3.4s}, [%[inptr0]]      \n" //v2={8,10,12,14} v3={9,11,13,15}
                "ext  v6.16b, v0.16b, v2.16b, #4     \n" // v6 = {2,4,6,8}

                "fmul v8.4s, v0.4s, %[w1].s[0]            \n" // v0 * w00
                "fmul v9.4s, v1.4s, %[w1].s[1]            \n" // v1 * w01
                "fmla v10.4s, v6.4s, %[w1].s[2]            \n" // v6 * w02

                "ld2  {v6.4s, v7.4s}, [%[inptr1]]      \n" //v2={8,10,12,14} v3={9,11,13,15}
                "ext  v11.16b, v4.16b, v6.16b, #4     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v4.4s, %[w2].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w2].s[1]            \n" // v1 * w02
                "fmla v10.4s, v11.4s, %[w2].s[2]            \n" // v2 * w00

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"

                "subs %[cnt], %[cnt], #1                    \n"

                "fmax  v0.4s, v0.4s, %[vzero].4s             \n"
                "st1 {v0.4s}, [%[outptr]], #16              \n"

                "bne  2b                                    \n"

                //right
                "1:                                          \n"
                "cmp %[remain], #1                           \n"
                "blt 4f                                     \n"
                "3:                                         \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias
                "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
               // "bif  v10.16b, %[vzero].16b, %[wmask].16b    \n" //pipei
                "ext  v2.16b, v0.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}
                "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei


                "fmul v8.4s, v0.4s, %[w1].s[0]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w1].s[1]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w1].s[2]            \n" // v2 * w00

                "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
                "ext  v6.16b, v4.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v4.4s, %[w2].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w2].s[1]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w2].s[2]            \n" // v2 * w00

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
                "fmax  v0.4s, v0.4s, %[vzero].4s             \n"
                "bif  v0.16b, %[vzero].16b, %[wmask].16b    \n" //pipei

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "4:                                          \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [outptr] "+r"(doutr0_ptr), \
                  [vzero] "+w" (vzero), [w1] "+w" (wr1), [w2] "+w" (wr2), [cnt] "+r" (cnt), \
                  [mask1] "+w" (vmask_rp1), [mask2] "+w" (vmask_rp2), [wmask] "+w" (wmask), [vbias] "+w" (wbias)
                : [remain] "r" (cnt_remain)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
                );
            }

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 = doutr0 + w_out;
            //! mid
            for (int j = h_out - size_pad_bottom - 1; j > 0; j--){
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;

                doutr0_ptr = doutr0;

               int cnt = cnt_col;

                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"
                "prfm pldl1keep, [%[inptr2]]             \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "ext  v2.16b, %[vzero].16b, v1.16b, #12     \n" // v2 = {0,1,3,5}
                "ext  v6.16b, %[vzero].16b, v5.16b, #12    \n" // v6 = {0,1,3,5}

                "fmul v8.4s, v0.4s, %[w0].s[1]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w0].s[2]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w0].s[0]            \n" // v2 * w00

                "ld2  {v12.4s, v13.4s}, [%[inptr2]], #32    \n"

                "sub %[inptr0], %[inptr0], #4            \n"
                "sub %[inptr1], %[inptr1], #4             \n"

                "fmla v8.4s, v4.4s, %[w1].s[1]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[2]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w1].s[0]            \n" // v2 * w00

                "ext  v14.16b, %[vzero].16b, v13.16b, #12    \n" // v6 = {0,1,3,5}
                "sub %[inptr2], %[inptr2], #4             \n"

                "prfm pldl1keep, [%[inptr0]]             \n"

                "fmla v8.4s, v12.4s, %[w2].s[1]            \n" // v0 * w01
                "fmla v9.4s, v13.4s, %[w2].s[2]            \n" // v1 * w02
                "fmla v10.4s, v14.4s, %[w2].s[0]            \n" // v2 * w00

                "prfm pldl1keep, [%[inptr1]]             \n"
                "prfm pldl1keep, [%[inptr2]]             \n"

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
                "fmax  v0.4s, v0.4s, %[vzero].4s            \n"

                "cmp %[cnt], #1                             \n"
                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "blt 1f                                     \n"
                //mid
                "2:                                          \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "ld2  {v2.4s, v3.4s}, [%[inptr0]]      \n" //v2={8,10,12,14} v3={9,11,13,15}
                "ld2  {v6.4s, v7.4s}, [%[inptr1]]      \n" //v2={8,10,12,14} v3={9,11,13,15}
                "ld2  {v12.4s, v13.4s}, [%[inptr2]], #32    \n"
                "ext  v11.16b, v0.16b, v2.16b, #4     \n" // v6 = {2,4,6,8}

                "prfm pldl1keep, [%[inptr2]]             \n"

                "fmul v8.4s, v0.4s, %[w0].s[0]            \n" // v0 * w00
                "fmul v9.4s, v1.4s, %[w0].s[1]            \n" // v1 * w01
                "fmla v10.4s, v11.4s, %[w0].s[2]            \n" // v6 * w02

                "ext  v11.16b, v4.16b, v6.16b, #4     \n" // v6 = {2,4,6,8}
                "ld2  {v14.4s, v15.4s}, [%[inptr2]]    \n"

                "fmla v8.4s, v4.4s, %[w1].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[1]            \n" // v1 * w02
                "fmla v10.4s, v11.4s, %[w1].s[2]            \n" // v2 * w00

                "ext  v11.16b, v12.16b, v14.16b, #4     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v12.4s, %[w2].s[0]            \n" // v0 * w00
                "fmla v9.4s, v13.4s, %[w2].s[1]            \n" // v1 * w01
                "fmla v10.4s, v11.4s, %[w2].s[2]            \n" // v6 * w02

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"
                "prfm pldl1keep, [%[inptr2]]             \n"

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"

                "subs %[cnt], %[cnt], #1                    \n"
                "fmax  v0.4s, v0.4s, %[vzero].4s            \n"

                "st1 {v0.4s}, [%[outptr]], #16              \n"

                "bne  2b                                    \n"

                //right
                "1:                                          \n"
                "cmp %[remain], #1                           \n"
                "blt 4f                                     \n"
                "3:                                         \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
               // "bif  v10.16b, %[vzero].16b, %[wmask].16b    \n" //pipei
                "ext  v2.16b, v0.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}
                "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei

                "ld2  {v12.4s, v13.4s}, [%[inptr2]], #32    \n"
                "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
                "ext  v6.16b, v4.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}

                "fmul v8.4s, v0.4s, %[w0].s[0]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w0].s[1]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w0].s[2]            \n" // v2 * w00

                "bif  v12.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v13.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                "fmla v8.4s, v4.4s, %[w1].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[1]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w1].s[2]            \n" // v2 * w00

                "ext  v14.16b, v12.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v12.4s, %[w2].s[0]            \n" // v0 * w01
                "fmla v9.4s, v13.4s, %[w2].s[1]            \n" // v1 * w02
                "fmla v10.4s, v14.4s, %[w2].s[2]            \n" // v2 * w00

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
                "fmax  v0.4s, v0.4s, %[vzero].4s             \n"

                "bif  v0.16b, %[vzero].16b, %[wmask].16b    \n" //pipei

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "4:                                          \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [inptr2] "+r"(din2_ptr), [outptr] "+r"(doutr0_ptr), \
                  [vzero] "+w" (vzero), [w1] "+w" (wr1), [w2] "+w" (wr2), [cnt] "+r" (cnt), [w0] "+w" (wr0), \
                  [mask1] "+w" (vmask_rp1), [mask2] "+w" (vmask_rp2), [wmask] "+w" (wmask), [vbias] "+w" (wbias)
                : [remain] "r" (cnt_remain)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14"
                );
                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
                doutr0 = doutr0 + w_out;
            }

            if (size_pad_bottom){
               int cnt = cnt_col;
                din0_ptr = dr0;
                din1_ptr = dr1;

                doutr0_ptr = doutr0;

                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "ext  v2.16b, %[vzero].16b, v1.16b, #12     \n" // v2 = {0,1,3,5}
                "ext  v6.16b, %[vzero].16b, v5.16b, #12    \n" // v6 = {0,1,3,5}

                "fmul v8.4s, v0.4s, %[w0].s[1]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w0].s[2]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w0].s[0]            \n" // v2 * w00

                "sub %[inptr0], %[inptr0], #4            \n"
                "sub %[inptr1], %[inptr1], #4             \n"

                "fmla v8.4s, v4.4s, %[w1].s[1]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[2]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w1].s[0]            \n" // v2 * w00

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"

                "fmax  v0.4s, v0.4s, %[vzero].4s             \n"

                "cmp %[cnt], #1                             \n"
                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "blt 1f                                     \n"
                //mid
                "2:                                          \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "ld2  {v2.4s, v3.4s}, [%[inptr0]]      \n" //v2={8,10,12,14} v3={9,11,13,15}
                "ext  v6.16b, v0.16b, v2.16b, #4     \n" // v6 = {2,4,6,8}

                "fmul v8.4s, v0.4s, %[w0].s[0]            \n" // v0 * w00
                "fmul v9.4s, v1.4s, %[w0].s[1]            \n" // v1 * w01
                "fmla v10.4s, v6.4s, %[w0].s[2]            \n" // v6 * w02

                "ld2  {v6.4s, v7.4s}, [%[inptr1]]      \n" //v2={8,10,12,14} v3={9,11,13,15}
                "ext  v11.16b, v4.16b, v6.16b, #4     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v4.4s, %[w1].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[1]            \n" // v1 * w02
                "fmla v10.4s, v11.4s, %[w1].s[2]            \n" // v2 * w00

                "prfm pldl1keep, [%[inptr0]]             \n"
                "prfm pldl1keep, [%[inptr1]]             \n"

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"

                "subs %[cnt], %[cnt], #1                    \n"
                "fmax  v0.4s, v0.4s, %[vzero].4s             \n"

                "st1 {v0.4s}, [%[outptr]], #16              \n"

                "bne  2b                                    \n"

                //right
                "1:                                          \n"
                "cmp %[remain], #1                           \n"
                "blt 4f                                     \n"
                "3:                                         \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias
                "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
               // "bif  v10.16b, %[vzero].16b, %[wmask].16b    \n" //pipei
                "ext  v2.16b, v0.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}
                "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei


                "fmul v8.4s, v0.4s, %[w0].s[0]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w0].s[1]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w0].s[2]            \n" // v2 * w00

                "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
                "ext  v6.16b, v4.16b, %[vzero].16b, #4     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v4.4s, %[w1].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[1]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w1].s[2]            \n" // v2 * w00

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
                "fmax  v0.4s, v0.4s, %[vzero].4s             \n"
                "bif  v0.16b, %[vzero].16b, %[wmask].16b    \n" //pipei

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "4:                                          \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [outptr] "+r"(doutr0_ptr), \
                  [vzero] "+w" (vzero), [w0] "+w" (wr0), [w1] "+w" (wr1), [cnt] "+r" (cnt), \
                  [mask1] "+w" (vmask_rp1), [mask2] "+w" (vmask_rp2), [wmask] "+w" (wmask), [vbias] "+w" (wbias)
                : [remain] "r" (cnt_remain)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
                );
            }

        }
    }


}
#else
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
    if (size_right_remain == 0 || size_pad_right == 0) {
        right_pad_idx[0] = 1;
    }
    if (size_right_remain == 0) {
        right_pad_idx[1] = 1;
        if (size_pad_right == 0) {
            right_pad_idx[2] = 1;
        }
    }
    uint32x4_t vmask_rp = vcgtq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(0));
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

            prefetch(din0_ptr);
            prefetch(din1_ptr);
            //! top pad
            float32x4_t vzero= vdupq_n_f32(0.f);
            // todo
            float *doutr0_ptr = doutr0;

            float32x4_t din0_1234 = vld1q_f32(din0_ptr);
            float32x4_t din1_1234 = vld1q_f32(din1_ptr);

            float32x4_t din0_0123 = vextq_f32(vzero, din0_1234, 3);
            float32x4_t din1_0123 = vextq_f32(vzero, din1_1234, 3);

            float32x4_t din0_2340 = vextq_f32(din0_1234, vzero, 1);
            float32x4_t din1_2340 = vextq_f32(din1_1234, vzero, 1);

            float32x4_t sum0 = vmulq_f32(din0_0123, wr1);
            float32x4_t sum1 = vmulq_f32(din0_2340, wr1);

            sum0 = vmlaq_f32(sum0, din1_0123, wr2);
            sum1 = vmlaq_f32(sum1, din1_2340, wr2);

            float32x2_t vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
            float32x2_t vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            float32x2_t vsum = vpadd_f32(vsum0, vsum1);

            vsum = vadd_f32(vsum, vget_low_f32(wbias));

            din0_ptr += 3;
            din1_ptr += 3;

            vsum = vmax_f32(vsum, vget_low_f32(vzero));

            prefetch(din0_ptr);
            prefetch(din1_ptr);
            vst1_f32(doutr0_ptr, vsum);

            doutr0_ptr += 2;

            //mid
            int cnt = cnt_col;
            for (;cnt > 0; cnt--){
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                float32x4_t din0_5678 = vld1q_f32(din0_ptr + 4);
                float32x4_t din1_5678 = vld1q_f32(din1_ptr + 4);
                float32x4_t din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                float32x4_t din1_3456 = vextq_f32(din1_1234, din1_5678, 2);

                sum0 = vmulq_f32(din0_1234, wr1);
                sum1 = vmulq_f32(din0_3456, wr1);

                din0_ptr += 4;
                din1_ptr += 4;

                sum0 = vmlaq_f32(sum0, din1_1234, wr2);
                sum1 = vmlaq_f32(sum1, din1_3456, wr2);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                prefetch(din0_ptr);
                prefetch(din1_ptr);

                vsum = vmax_f32(vsum, vget_low_f32(vzero));

                vst1_f32(doutr0_ptr, vsum);
                doutr0_ptr += 2;
            }
            //right
            din0_1234 = vld1q_f32(din0_ptr);
            din1_1234 = vld1q_f32(din1_ptr);

            float32x4_t din0_5678 = vld1q_f32(din0_ptr + 4);
            float32x4_t din1_5678 = vld1q_f32(din1_ptr + 4);

            uint32x4_t vone = vdupq_n_u32(-1);
            uint32x4_t vmask_rp_r2 = vextq_u32(vone, vmask_rp, 2);
            uint32x4_t vmask_rp_l2 = vextq_u32(vmask_rp, vone, 2);

            din0_1234 = vbslq_f32(vmask_rp_r2, din0_1234, vzero);
            din1_1234 = vbslq_f32(vmask_rp_r2, din1_1234, vzero);

            din0_5678 = vbslq_f32(vmask_rp_l2, din0_5678, vzero);
            din1_5678 = vbslq_f32(vmask_rp_l2, din1_5678, vzero);

            sum0 = vmulq_f32(din0_1234, wr1);

            float32x4_t din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
            float32x4_t din1_3456 = vextq_f32(din1_1234, din1_5678, 2);

            sum1 = vmulq_f32(din0_3456, wr1);

            sum0 = vmlaq_f32(sum0, din1_1234, wr2);
            sum1 = vmlaq_f32(sum1, din1_3456, wr2);

            vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
            vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            vsum = vpadd_f32(vsum0, vsum1);

            float32x2_t vdata = vld1_f32(doutr0_ptr);

            vsum = vadd_f32(vsum, vget_low_f32(wbias));

            vsum = vmax_f32(vsum, vget_low_f32(vzero));

            vsum = vbsl_f32(vget_low_u32(mask_w), vsum, vdata);

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;

            vst1_f32(doutr0_ptr, vsum);

            doutr0 += w_out;

            //! process mid rows
            for (int j = h_out - size_pad_bottom - 1; j > 0; j--) {
                // todo
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                doutr0_ptr = doutr0;

                prefetch(din0_ptr);
                prefetch(din1_ptr);
                prefetch(din2_ptr);

                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                float32x4_t din2_1234 = vld1q_f32(din2_ptr);


                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);
                float32x4_t din2_0123 = vextq_f32(vzero, din2_1234, 3);

                float32x4_t din0_2345 = vextq_f32(din0_1234, vzero, 1);
                float32x4_t din1_2345 = vextq_f32(din1_1234, vzero, 1);
                float32x4_t din2_2345 = vextq_f32(din2_1234, vzero, 1);

                sum0 = vmulq_f32(din0_0123, wr0);
                sum1 = vmulq_f32(din0_2345, wr0);

                din0_ptr += 3;
                din1_ptr += 3;

                sum0 =vmlaq_f32(sum0, din1_0123, wr1);
                sum1 =vmlaq_f32(sum1, din1_2345, wr1);

                din2_ptr += 3;
                prefetch(din0_ptr);

                sum0 =vmlaq_f32(sum0, din2_0123, wr2);
                sum1 =vmlaq_f32(sum1, din2_2345, wr2);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));

                vsum = vpadd_f32(vsum0, vsum1);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                vsum = vmax_f32(vsum, vget_low_f32(vzero));

                prefetch(din1_ptr);
                prefetch(din2_ptr);

                vst1_f32(doutr0_ptr, vsum);

                doutr0_ptr += 2;

                //mid
                cnt = cnt_col;
                for (;cnt > 0; cnt--){
                    din0_1234 = vld1q_f32(din0_ptr);
                    din1_1234 = vld1q_f32(din1_ptr);
                    din2_1234 = vld1q_f32(din2_ptr);

                    float32x4_t din0_5678 = vld1q_f32(din0_ptr + 4);
                    float32x4_t din1_5678 = vld1q_f32(din1_ptr + 4);
                    float32x4_t din2_5678 = vld1q_f32(din2_ptr + 4);

                    float32x4_t din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                    float32x4_t din1_3456 = vextq_f32(din1_1234, din1_5678, 2);
                    float32x4_t din2_3456 = vextq_f32(din2_1234, din2_5678, 2);

                    sum0 = vmulq_f32(din0_1234, wr0);
                    sum1 = vmulq_f32(din0_3456, wr0);

                    din0_ptr += 4;
                    din1_ptr += 4;

                    sum0 = vmlaq_f32(sum0, din1_1234, wr1);
                    sum1 = vmlaq_f32(sum1, din1_3456, wr1);
                    
                    din2_ptr += 4;
                    prefetch(din0_ptr);

                    sum0 = vmlaq_f32(sum0, din2_1234, wr2);
                    sum1 = vmlaq_f32(sum1, din2_3456, wr2);

                    vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                    vsum = vpadd_f32(vsum0, vsum1);

                    vsum = vadd_f32(vsum, vget_low_f32(wbias));

                    vsum = vmax_f32(vsum, vget_low_f32(vzero));
                    
                    prefetch(din1_ptr);
                    prefetch(din2_ptr);

                    vst1_f32(doutr0_ptr, vsum);

                    doutr0_ptr += 2;
                }
            //right
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                din2_1234 = vld1q_f32(din2_ptr);
                
                din0_5678 = vld1q_f32(din0_ptr + 4);
                din1_5678 = vld1q_f32(din1_ptr + 4);
                float32x4_t din2_5678 = vld1q_f32(din2_ptr + 4);

                vmask_rp_r2 = vextq_u32(vone, vmask_rp, 2);
                vmask_rp_l2 = vextq_u32(vmask_rp, vone, 2);

                din0_1234 = vbslq_f32(vmask_rp_r2, din0_1234, vzero);
                din1_1234 = vbslq_f32(vmask_rp_r2, din1_1234, vzero);
                din2_1234 = vbslq_f32(vmask_rp_r2, din2_1234, vzero);

                din0_5678 = vbslq_f32(vmask_rp_l2, din0_5678, vzero);
                din1_5678 = vbslq_f32(vmask_rp_l2, din1_5678, vzero);
                din2_5678 = vbslq_f32(vmask_rp_l2, din2_5678, vzero);

                sum0 = vmulq_f32(din0_1234, wr0);

                din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                din1_3456 = vextq_f32(din1_1234, din1_5678, 2);
                float32x4_t din2_3456 = vextq_f32(din2_1234, din2_5678, 2);

                sum1 = vmulq_f32(din0_3456, wr0);
                sum0 = vmlaq_f32(sum0, din1_1234, wr1);

                sum1 = vmlaq_f32(sum1, din1_3456, wr1);
                sum0 = vmlaq_f32(sum0, din2_1234, wr2);

                sum1 = vmlaq_f32(sum1, din2_3456, wr2);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                float32x2_t vdata = vld1_f32(doutr0_ptr);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                vsum = vmax_f32(vsum, vget_low_f32(vzero));

                vsum = vbsl_f32(vget_low_u32(mask_w), vsum, vdata);
            
                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;

                vst1_f32(doutr0_ptr, vsum);

                doutr0 += w_out;
            } // end of process mid rows

            // process bottom pad if needed
            if (size_pad_bottom) {
                din0_ptr = dr0;
                din1_ptr = dr1;
                doutr0_ptr = doutr0;

                prefetch(din0_ptr);
                prefetch(din1_ptr);
                // todo
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);


                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);

                float32x4_t din0_2345 = vextq_f32(din0_1234, vzero, 1);
                float32x4_t din1_2345 = vextq_f32(din1_1234, vzero, 1);

                sum0 = vmulq_f32(din0_0123, wr0);
                sum1 = vmulq_f32(din0_2345, wr0);

                sum0 =vmlaq_f32(sum0, din1_0123, wr1);
                sum1 =vmlaq_f32(sum1, din1_2345, wr1);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                din0_ptr += 3;
                din1_ptr += 3;

                vsum = vmax_f32(vsum, vget_low_f32(vzero));

                prefetch(din0_ptr);
                prefetch(din1_ptr);

                vst1_f32(doutr0_ptr, vsum);

                doutr0_ptr += 2;

                //mid
                cnt = cnt_col;
                for (;cnt > 0; cnt--){
                    din0_1234 = vld1q_f32(din0_ptr);
                    din1_1234 = vld1q_f32(din1_ptr);

                    float32x4_t din0_5678 = vld1q_f32(din0_ptr + 4);
                    float32x4_t din1_5678 = vld1q_f32(din1_ptr + 4);

                    float32x4_t din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                    float32x4_t din1_3456 = vextq_f32(din1_1234, din1_5678, 2);

                    sum0 = vmulq_f32(din0_1234, wr0);

                    sum1 = vmulq_f32(din0_3456, wr0);

                    sum0 = vmlaq_f32(sum0, din1_1234, wr1);
                    sum1 = vmlaq_f32(sum1, din1_3456, wr1);

                    vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                    vsum = vpadd_f32(vsum0, vsum1);

                    vsum = vadd_f32(vsum, vget_low_f32(wbias));

                    din0_ptr += 4;
                    din1_ptr += 4;
                    
                    vsum = vmax_f32(vsum, vget_low_f32(vzero));

                    prefetch(din0_ptr);
                    prefetch(din1_ptr);

                    vst1_f32(doutr0_ptr, vsum);

                    doutr0_ptr += 2;
                }
            //right
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                din0_5678 = vld1q_f32(din0_ptr + 4);
                din1_5678 = vld1q_f32(din1_ptr + 4);

                vmask_rp_r2 = vextq_u32(vone, vmask_rp, 2);
                vmask_rp_l2 = vextq_u32(vmask_rp, vone, 2);

                din0_1234 = vbslq_f32(vmask_rp_r2, din0_1234, vzero);
                din1_1234 = vbslq_f32(vmask_rp_r2, din1_1234, vzero);

                din0_5678 = vbslq_f32(vmask_rp_l2, din0_5678, vzero);
                din1_5678 = vbslq_f32(vmask_rp_l2, din1_5678, vzero);

                sum0 = vmulq_f32(din0_1234, wr0);

                din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                din1_3456 = vextq_f32(din1_1234, din1_5678, 2);

                sum1 = vmulq_f32(din0_3456, wr0);
                sum0 = vmlaq_f32(sum0, din1_1234, wr1);

                sum1 = vmlaq_f32(sum1, din1_3456, wr1);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                float32x2_t vdata = vld1_f32(doutr0_ptr);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                vsum = vmax_f32(vsum, vget_low_f32(vzero));

                vsum = vbsl_f32(vget_low_u32(mask_w), vsum, vdata);

                vst1_f32(doutr0_ptr, vsum);

            } // end of process bottom pad

        }
    }
}
#endif
#else

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
            int cnt = cnt_col;
            asm volatile(
            // process left pad
            "pld [%[din0_ptr], #128]                @ preload data\n"
                    "pld [%[din1_ptr], #128]                @ preload data\n"
                    "vmov.u32 q11, #0                       @ for left pad\n"
                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r2\n"

                    "vext.32 q7, q11, q10, #3               @ shift right 1 data\n"
                    "vext.32  q14, q10, q11, #1              @ shift left r1\n"
                    "vmul.f32 q8, q7, %q[wr1]               @ mul weight 1, out0\n"

                    "vext.32 q7, q11, q12, #3               @ shift right 1 data\n"
                    "vmul.f32 q9, q14, %q[wr1]               @ mul weight 1, out1\n"

                    "vext.32  q14, q12, q11, #1              @ shift left r2\n"
                    "vmla.f32 q8, q7, %q[wr2]               @ mul weight 2, out0\n"

                    "vmla.f32 q9, q14,  %q[wr2]              @ mul weight 2, out1\n"

                    "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                    "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                    "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                    "vadd.f32  d17, d16, %e[bias]           @ add bias \n"
                    "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                    "sub %[din0_ptr], #4                    @ 1pad + 2 float data overlap\n"
                    "sub %[din1_ptr], #4                    @ 1pad + 2 float data overlap\n"

                    "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                    // process mid cols
                    "cmp %[cnt], #1                         @ check whether has mid loop\n"
                    "blt  2f                                @ jump to rightpad\n"
                    "1:                                     @ main loop start point\n"
                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"
                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r2\n"
                    "vld1.32  {d26}, [%[din1_ptr]]!     @ load din r2\n"

                    "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                    "vext.32  q14, q12, q13, #2              @ shift left r2\n"
                    "vmul.f32 q8, q10, %q[wr1]              @ mul weight 1, out0\n"
                    "vmul.f32 q9, q7, %q[wr1]               @ mul weight 1, outr1\n"

                    "vmla.f32 q8, q12, %q[wr2]              @ mul weight 2, outr0\n"
                    "vmla.f32 q9, q14, %q[wr2]               @ mul weight 2, outr1\n"

                    "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                    "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                    "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                    "vadd.f32  d17, d16, %e[bias]           @ add bias \n"
                    "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                    "sub %[din0_ptr], #8                    @ 1 float data overlap and 1 redundant\n"
                    "sub %[din1_ptr], #8                    @ 1 float data overlap and 1 redundant\n"
                    "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    1b                              @ jump to main loop start point\n"

                    //! process right pad
                    "2:                                     @ right pad entry\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                    "pld [%[din0_ptr], #192]                @ preload data\n"
                    "pld [%[din1_ptr], #192]                @ preload data\n"

                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r2\n"
                    "vld1.32  {d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d26}, [%[din1_ptr]]!     @ load din r2\n"

                    "vbif d21, d31, %e[mask_din]            @ bit select, deal with right pad\n"
                    "vbif d22, d31, %f[mask_din]            @ bit select, deal with right pad\n"
                    "vbif d25, d31, %e[mask_din]            @ bit select, deal with right pad\n"
                    "vbif d26, d31, %f[mask_din]            @ bit select, deal with right pad\n"

                    "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                    "vext.32  q14, q12, q13, #2              @ shift left r1\n"
                    "vmul.f32 q8, q10, %q[wr1]              @ mul weight 1, out0\n"
                    "vmul.f32 q9, q7, %q[wr1]               @ mul weight 1, out1\n"

                    "pld [%[dout_ptr1], #64]                @ preload data\n"
                    "vmla.f32 q8, q12, %q[wr2]              @ mul weight 2, outr0\n"
                    "vmla.f32 q9, q14, %q[wr2]               @ mul weight 2, outr1\n"

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

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            //! end of top pad

            //! process mid rows
            for (int j = h_out - size_pad_bottom - 1; j > 0; j--) {

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
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1\n"
                        "vld1.32  {d28-d29}, [%[din2_ptr]]!     @ load din r2\n"

                        "vext.32 q7, q11, q10, #3               @ shift right 1 data\n"
                        "vext.32  q13, q10, q11, #1              @ shift left r0\n"
                        "vmul.f32 q8, q7, %q[wr0]               @ mul weight 00, outr0\n"
                        "vmul.f32 q9, q13, %q[wr0]               @ mul weight 00, outr1\n"

                        "vext.32 q7, q11, q12, #3               @ shift right 1 data\n"
                        "vext.32  q13, q12, q11, #1              @ shift left r1\n"
                        "vmla.f32 q8, q7, %q[wr1]               @ mul weight 10, outr0\n"
                        "vmla.f32 q9, q13,  %q[wr1]              @ mul weight 10, outr1\n"

                        "vext.32 q7, q11, q14, #3               @ shift right 1 data\n"
                        "vext.32  q13, q14, q11, #1              @ shift left r2\n"
                        "vmla.f32 q8, q7, %q[wr2]               @ mul weight 20, outr0\n"
                        "vmla.f32 q9, q13,  %q[wr2]              @ mul weight 20, outr1\n"

                        "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"
                        "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                        "sub %[din0_ptr], #4                    @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #4                    @ 1pad + 2 float data overlap\n"
                        "sub %[din2_ptr], #4                    @ 1pad + 2 float data overlap\n"

                        "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                        // process mid cols
                        "cmp %[cnt], #1                         @ check whether has mid loop\n"
                        "blt  4f                                @ jump to rightpad\n"
                        "3:                                     @ main loop start point\n"
                        "pld [%[din0_ptr], #192]                @ preload data\n"
                        "pld [%[din1_ptr], #192]                @ preload data\n"
                        "pld [%[din2_ptr], #192]                @ preload data\n"
                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0\n"
                        "vld1.32  {d22}, [%[din0_ptr]]!     @ load din r0\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1\n"
                        "vld1.32  {d26}, [%[din1_ptr]]!     @ load din r0\n"
                        "vld1.32  {d28-d29}, [%[din2_ptr]]!     @ load din r2\n"
                        "vld1.32  {d30}, [%[din2_ptr]]!     @ load din r0\n"

                        "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                        "vext.32  q11, q12, q13, #2              @ shift left r2\n"
                        "vmul.f32 q8, q10, %q[wr0]              @ mul weight 00, outr0\n"
                        "vmul.f32 q9, q7, %q[wr0]               @ mul weight 00, outr1\n"

                        "vext.32  q7, q14, q15, #2      @ shift left r2\n"
                        "vmla.f32 q8, q12, %q[wr1]              @ mul weight 10, outr0\n"
                        "vmla.f32 q9, q11, %q[wr1]               @ mul weight 10, outr1\n"

                        "vmla.f32 q8, q14, %q[wr2]  @mul weight 10, outr0\n"
                        "vmla.f32 q9, q7, %q[wr2]  @mul weight 10, outr1\n"

                        "vpadd.f32 d22, d16, d17  @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19 @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23 @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]  @ add bias \n"
                        "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                        "sub %[din0_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din2_ptr], #8 @ 2 float data overlap with previous data\n"
                        "vst1.32  {d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                        "subs %[cnt], #1 @ loop count minus 1\n"
                        "bne    3b                              @ jump to main loop start point\n"

                        // process right pad
                        "4:                                     @ right pad entry\n"
                        "vmov.u32  d31, #0 @ zero buf\n"
                        "pld [%[din0_ptr], #192]               @ preload data\n"
                        "pld [%[din1_ptr], #192]               @ preload data\n"

                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1\n"
                        "vld1.32  {d28-d29}, [%[din2_ptr]]!     @ load din r2\n"
                        "vld1.32  {d22}, [%[din0_ptr]]!     @ load din r0\n"
                        "vld1.32  {d26}, [%[din1_ptr]]!     @ load din r0\n"
                        "vld1.32  {d30}, [%[din2_ptr]]!     @ load din r0\n"

                        "vbif d21, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vbif d22, d31, %f[mask_din] @ bit select, deal with right pad\n"

                        "vbif d25, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vbif d26, d31, %f[mask_din] @ bit select, deal with right pad\n"

                        "vbif d29, d31, %e[mask_din] @ bit select, deal with right pad\n"
                        "vbif d30, d31, %f[mask_din] @ bit select, deal with right pad\n"

                        "vext.32  q7, q10, q11, #2      @ shift left r0\n"
                        "vext.32  q11, q12, q13, #2      @ shift left r1\n"
                        "vmul.f32 q8, q10, %q[wr0]  @mul weight 00, outr0\n"
                        "vmul.f32 q9, q7, %q[wr0]  @mul weight 00, outr1\n"

                        "vext.32  q7, q14, q15, #2      @ shift left r2\n"
                        "pld [%[dout_ptr1], #64]         @ preload ouput data\n"
                        "vmla.f32 q8, q12, %q[wr1]  @mul weight 10, outr0\n"
                        "vmla.f32 q9, q11, %q[wr1]  @mul weight 10, outr1\n"

                        "vld1.32  {d20}, [%[dout_ptr1]]    @ load dout\n"
                        "vmla.f32 q8, q14, %q[wr2]  @mul weight 20, outr0\n"
                        "vmla.f32 q9, q7, %q[wr2]  @mul weight 20, outr1\n"

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


                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
            } // end of process mid rows

            //! process bottom pad if needed
            if (size_pad_bottom) {
                din0_ptr = dr0;
                din1_ptr = dr1;

                cnt = cnt_col;
                asm volatile(
                // process left pad
                "pld [%[din0_ptr], #128]               @ preload data\n"
                        "pld [%[din1_ptr], #128]                @ preload data\n"

                        "vmov.u32 q11, #0 @ for left pad\n"
                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1\n"

                        "vext.32 q7, q11, q10, #3               @ shift right 1 data\n"
                        "vext.32  q14, q10, q11, #1              @ shift left r0\n"
                        "vmul.f32 q8, q7, %q[wr0]               @ mul weight 00, outr0\n"
                        "vmul.f32 q9, q14, %q[wr0]               @ mul weight 00, outr1\n"

                        "vext.32 q7, q11, q12, #3               @ shift right 1 data\n"
                        "vext.32  q14, q12, q11, #1              @ shift left r1\n"
                        "vmla.f32 q8, q7, %q[wr1]               @ mul weight 10, outr0\n"
                        "vmla.f32 q9, q14,  %q[wr1]              @ mul weight 10, outr1\n"

                        "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]           @ add bias \n"
                        "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                        "sub %[din0_ptr], #4                    @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #4                    @ 1pad + 2 float data overlap\n"
                        "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                        // process mid cols
                        "cmp %[cnt], #1                         @ check whether has mid loop\n"
                        "blt  1f                                @ jump to rightpad\n"
                        "2:                                     @ main loop start point\n"
                        "pld [%[din0_ptr], #192]                @ preload data\n"
                        "pld [%[din1_ptr], #192]                @ preload data\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r0\n"
                        "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r1\n"

                        "vext.32  q7, q10, q11, #2              @ shift left r1\n"
                        "vext.32  q14, q12, q13, #2              @ shift left r2\n"
                        "vmul.f32 q8, q10, %q[wr0]              @ mul weight 00, outr0\n"
                        "vmul.f32 q9, q7, %q[wr0]               @ mul weight 00, outr1\n"

                        "vmla.f32 q8, q12, %q[wr1]              @ mul weight 10, outr0\n"
                        "vmla.f32 q9, q14, %q[wr1]               @ mul weight 10, outr1\n"

                        "vpadd.f32 d22, d16, d17                @ pair add of out0 \n"
                        "vpadd.f32 d23, d18, d19                @ pair add of out1 \n"
                        "vpadd.f32 d16, d22, d23                @ get finnal out0,1\n"

                        "vadd.f32  d17, d16, %e[bias]           @ add bias \n"
                        "vmax.f32  d17, d17, %e[vzero]          @ relu\n"

                        "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"
                        "vst1.32  {d17},   [%[dout_ptr1]]!      @ store result, add pointer\n"

                        "subs %[cnt], #1                        @ loop count minus 1\n"
                        "bne    2b                              @ jump to main loop start point\n"

                        // process right pad
                        "1:                                     @ right pad entry\n"
                        "vmov.u32  d31, #0                      @ zero buf\n"
                        "pld [%[din0_ptr], #192]                @ preload data\n"
                        "pld [%[din1_ptr], #192]                @ preload data\n"
                        "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                        "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r1\n"

                        "vbif d21, d31, %e[mask_din]            @ bit select, deal with right pad\n"
                        "vbif d22, d31, %f[mask_din]            @ bit select, deal with right pad\n"

                        "vbif d25, d31, %e[mask_din]            @ bit select, deal with right pad\n"
                        "vbif d26, d31, %f[mask_din]            @ bit select, deal with right pad\n"

                        "vext.32  q7, q10, q11, #2              @ shift left r0\n"
                        "vext.32  q14, q12, q13, #2              @ shift left r1\n"

                        "vmul.f32 q8, q10, %q[wr0]              @ mul weight 00, outr0\n"
                        "vmul.f32 q9, q7, %q[wr0]               @ mul weight 00, outr1\n"


                        "pld [%[dout_ptr1], #64]                @ preload data\n"

                        "vmla.f32 q8, q12, %q[wr1]              @ mul weight 10, outr0\n"
                        "vmla.f32 q9, q14, %q[wr1]               @ mul weight 10, outr1\n"

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

            } //! end of process bottom pad
        }
    }
}
#endif
/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias, width <= 4
 */
#ifdef __aarch64__
void conv_depthwise_3x3s1p1_bias_s(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    //! for 4x6 convolution window
    const int right_pad_idx[4] = {3, 2, 1, 0};

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    int tile_h = (h_in + 1) >> 1;

    int size_pad_right = 4 - w_in;
    int size_pad_bottom = 1 + (tile_h << 1) - h_in;

    uint32x4_t vmask_rp = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(size_pad_right));
    int right_pad_sub = size_pad_right * sizeof(float);

    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
    const float* ptr_zero = zero;

    int flag_hin1 = int(h_in == 1);

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

            if (h_in == 2) {
                din2_ptr = ptr_zero;
            }
            if (h_in == 1) {
                din1_ptr = ptr_zero;
            }

            //! deal with top pad
            int h = 0;
            //! process
            float32x4_t vzero = vdupq_n_f32(0.f);
            // todo
            float32x4_t din0_1234 = vld1q_f32(din0_ptr);
            float32x4_t din1_1234 = vld1q_f32(din1_ptr);
            float32x4_t din2_1234 = vld1q_f32(din2_ptr);

            if(vgetq_lane_u32(vmask_rp, 0) == 0){
                din0_1234 = vdupq_n_f32(0.f);
                din1_1234 = vdupq_n_f32(0.f);
                din2_1234 = vdupq_n_f32(0.f);
            }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 1);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 1);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 1);
                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
            }else if(vgetq_lane_u32(vmask_rp, 2) == 0){

                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
            }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
            }

            /*
            printf("wr0: %f, %f, %f \n", vgetq_lane_f32(wr0, 0), vgetq_lane_f32(wr0, 1), vgetq_lane_f32(wr0, 2));
            printf("wr1: %f, %f, %f \n", vgetq_lane_f32(wr1, 0), vgetq_lane_f32(wr1, 1), vgetq_lane_f32(wr1, 2));
            printf("wr2: %f, %f, %f \n", vgetq_lane_f32(wr2, 0), vgetq_lane_f32(wr2, 1), vgetq_lane_f32(wr2, 2));

            printf("din0_1234: %f, %f, %f, %f \n", vgetq_lane_f32(din0_1234, 0), vgetq_lane_f32(din0_1234, 1), vgetq_lane_f32(din0_1234, 2), vgetq_lane_f32(din0_1234, 3));
            printf("din1_1234: %f, %f, %f, %f \n", vgetq_lane_f32(din1_1234, 0), vgetq_lane_f32(din1_1234, 1), vgetq_lane_f32(din1_1234, 2), vgetq_lane_f32(din1_1234, 3));
           
           */  
            float32x4_t sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 1);
            float32x4_t sum1 = vmulq_lane_f32(din0_1234, vget_low_f32(wr1), 1);

            sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 1);
            sum1 = vmlaq_lane_f32(sum1, din1_1234, vget_low_f32(wr2), 1);

            sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 1);

            float32x4_t din0_2340 = vextq_f32(din0_1234, vzero, 1);
            float32x4_t din1_2340 = vextq_f32(din1_1234, vzero, 1);
            float32x4_t din2_2340 = vextq_f32(din2_1234, vzero, 1);
            //printf("din0_2340: %f, %f, %f, %f \n", vgetq_lane_f32(din0_2340, 0), vgetq_lane_f32(din0_2340, 1), vgetq_lane_f32(din0_2340, 2), vgetq_lane_f32(din0_2340, 3));
            //printf("din1_2340: %f, %f, %f, %f \n", vgetq_lane_f32(din1_2340, 0), vgetq_lane_f32(din1_2340, 1), vgetq_lane_f32(din1_2340, 2), vgetq_lane_f32(din1_2340, 3));
            
            sum0 = vmlaq_lane_f32(sum0, din0_2340, vget_high_f32(wr0), 0);
            sum1 = vmlaq_lane_f32(sum1, din0_2340, vget_high_f32(wr1), 0);

            sum0 = vmlaq_lane_f32(sum0, din1_2340, vget_high_f32(wr1), 0);
            sum1 = vmlaq_lane_f32(sum1, din1_2340, vget_high_f32(wr2), 0);

            sum0 = vmlaq_lane_f32(sum0, din2_2340, vget_high_f32(wr2), 0);

            float32x4_t din0_0123 = vextq_f32(vzero, din0_1234, 3);
            float32x4_t din1_0123 = vextq_f32(vzero, din1_1234, 3);
            float32x4_t din2_0123 = vextq_f32(vzero, din2_1234, 3);

            //printf("din0_0123: %f, %f, %f, %f \n", vgetq_lane_f32(din0_0123, 0), vgetq_lane_f32(din0_0123, 1), vgetq_lane_f32(din0_0123, 2), vgetq_lane_f32(din0_0123, 3));
            //printf("din1_0123: %f, %f, %f, %f \n", vgetq_lane_f32(din1_0123, 0), vgetq_lane_f32(din1_0123, 1), vgetq_lane_f32(din1_0123, 2), vgetq_lane_f32(din1_0123, 3));
            
            sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);
            sum1 = vmlaq_lane_f32(sum1, din0_0123, vget_low_f32(wr1), 0);

            sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);
            sum1 = vmlaq_lane_f32(sum1, din1_0123, vget_low_f32(wr2), 0);

            sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);

            float32x4_t dout0_1234 = vld1q_f32(doutr0);
            float32x4_t dout1_1234 = vld1q_f32(doutr1);

            sum0 = vaddq_f32(sum0, wbias);
            sum1 = vaddq_f32(sum1, wbias);

            //printf("sum1: %f, %f, %f, %f \n", vgetq_lane_f32(sum0, 0), vgetq_lane_f32(sum0, 1), vgetq_lane_f32(sum0, 2), vgetq_lane_f32(sum0, 3));
            if(vgetq_lane_u32(vmask_rp, 0) == 0){
                //null
            }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout1_1234, 0);
            }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout1_1234, 0);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout0_1234, 1);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout1_1234, 1);
                
                //dout0_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(dout0_1234));
                //dout1_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(dout1_1234));
            }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                /*
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout1_1234, 0);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout0_1234, 1);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout1_1234, 1);
                */
                dout0_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(dout0_1234));
                dout1_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(dout1_1234));

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), dout0_1234, 2);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), dout1_1234, 2);
            }else{
                /*
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout1_1234, 0);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout0_1234, 1);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout1_1234, 1);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), dout0_1234, 2);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), dout1_1234, 2);
                */
                //dout0_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                //dout1_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(sum0));

                dout0_1234 = sum1;
                dout1_1234 = sum0;
            }

            //printf("dout1: %f, %f, %f, %f \n", vgetq_lane_f32(dout1_1234, 0), vgetq_lane_f32(dout1_1234, 1), vgetq_lane_f32(dout1_1234, 2), vgetq_lane_f32(dout1_1234, 3));
            vst1q_f32(doutr0, dout0_1234);//sum1
            if(!flag_hin1){//h > 1
                vst1q_f32(doutr1, dout1_1234);//sum0
            }

            //! after process, increase pointer
            doutr0 = doutr1 + w_out;
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

                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                din2_1234 = vld1q_f32(din2_ptr);
                float32x4_t din3_1234 = vld1q_f32(din3_ptr);


                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                din0_1234 = vdupq_n_f32(0.f);
                din1_1234 = vdupq_n_f32(0.f);
                din2_1234 = vdupq_n_f32(0.f);
                din3_1234 = vdupq_n_f32(0.f);
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 1);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 1);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 1);
                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);

                din3_1234 = vsetq_lane_f32(0.f, din3_1234, 1);
                din3_1234 = vsetq_lane_f32(0.f, din3_1234, 2);
                din3_1234 = vsetq_lane_f32(0.f, din3_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);

                din3_1234 = vsetq_lane_f32(0.f, din3_1234, 2);
                din3_1234 = vsetq_lane_f32(0.f, din3_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);

                din3_1234 = vsetq_lane_f32(0.f, din3_1234, 3);
                }

                sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 1);

                sum1 = vmulq_lane_f32(din1_1234, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 1);

                sum1 = vmlaq_lane_f32(sum1, din2_1234, vget_low_f32(wr1), 1);
                sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 1);

                sum1 = vmlaq_lane_f32(sum1, din3_1234, vget_low_f32(wr2), 1);

                din0_2340 = vextq_f32(din0_1234, vzero, 1);
                din1_2340 = vextq_f32(din1_1234, vzero, 1);
                din2_2340 = vextq_f32(din2_1234, vzero, 1);
                float32x4_t din3_2340 = vextq_f32(din3_1234, vzero, 1);

                sum0 = vmlaq_lane_f32(sum0, din0_2340, vget_high_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_2340, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_2340, vget_high_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_2340, vget_high_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_2340, vget_high_f32(wr2), 0);

                sum1 = vmlaq_lane_f32(sum1, din3_2340, vget_high_f32(wr2), 0);

                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);
                din2_0123 = vextq_f32(vzero, din2_1234, 3);
                float32x4_t din3_0123 = vextq_f32(vzero, din3_1234, 3);

                sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_0123, vget_low_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_0123, vget_low_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);

                sum1 = vmlaq_lane_f32(sum1, din3_0123, vget_low_f32(wr2), 0);

                float32x4_t dout0_1234 = vld1q_f32(doutr0);
                float32x4_t dout1_1234 = vld1q_f32(doutr1);

                sum0 = vaddq_f32(sum0, wbias);
                sum1 = vaddq_f32(sum1, wbias);

                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                //null
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                /*
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout0_1234, 1);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout1_1234, 1);
                */
                dout0_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(dout0_1234));
                dout1_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(dout1_1234));
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                /*
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout0_1234, 1);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout1_1234, 1);
                */
                dout0_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(dout0_1234));
                dout1_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(dout1_1234));

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), dout0_1234, 2);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), dout1_1234, 2);
                }else{
                /*
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout0_1234, 1);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout1_1234, 1);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), dout0_1234, 2);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), dout1_1234, 2);
                */
                //dout0_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                //dout1_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(sum1));

                dout0_1234 = sum0;
                dout1_1234 = sum1;
                }

                vst1q_f32(doutr0, dout0_1234);
                vst1q_f32(doutr1, dout1_1234);

                doutr0 = doutr1 + w_out;
                doutr1 = doutr0 + w_out;
                dr0 = dr2;
                dr1 = dr3;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
            } //! end of processing mid rows

            //! deal with bottom pad
            if (h_in > 2) {
                din0_ptr = dr0;
                din1_ptr = dr1;
                if (size_pad_bottom == 2){
                    din2_ptr = ptr_zero;
                } else {
                    din2_ptr = dr2;
                }

                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                din2_1234 = vld1q_f32(din2_ptr);


                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                din0_1234 = vdupq_n_f32(0.f);
                din1_1234 = vdupq_n_f32(0.f);
                din2_1234 = vdupq_n_f32(0.f);
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 1);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 1);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 1);
                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }

                sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 1);

                sum1 = vmulq_lane_f32(din1_1234, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 1);

                sum1 = vmlaq_lane_f32(sum1, din2_1234, vget_low_f32(wr1), 1);
                sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 1);

                din0_2340 = vextq_f32(din0_1234, vzero, 1);
                din1_2340 = vextq_f32(din1_1234, vzero, 1);
                din2_2340 = vextq_f32(din2_1234, vzero, 1);

                sum0 = vmlaq_lane_f32(sum0, din0_2340, vget_high_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_2340, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_2340, vget_high_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_2340, vget_high_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_2340, vget_high_f32(wr2), 0);

                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);
                din2_0123 = vextq_f32(vzero, din2_1234, 3);

                sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_0123, vget_low_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_0123, vget_low_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);

                float32x4_t dout0_1234 = vld1q_f32(doutr0);
                float32x4_t dout1_1234 = vld1q_f32(doutr1);

                sum0 = vaddq_f32(sum0, wbias);
                sum1 = vaddq_f32(sum1, wbias);

                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                //null
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                /*
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout0_1234, 1);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout1_1234, 1);
                */
                dout0_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(dout0_1234));
                dout1_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(dout1_1234));
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                /*
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout0_1234, 1);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout1_1234, 1);
                */
                dout0_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(dout0_1234));
                dout1_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(dout1_1234));

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), dout0_1234, 2);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), dout1_1234, 2);
                }else{
                /*
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout0_1234, 1);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout1_1234, 1);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), dout0_1234, 2);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), dout1_1234, 2);
                */
                //dout0_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                //dout1_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(sum1));

                dout0_1234 = sum0;
                dout1_1234 = sum1;
                }

                vst1q_f32(doutr0, dout0_1234);
                if(size_pad_bottom != 2)
                    vst1q_f32(doutr1, dout1_1234);
            }
            //! end of processing bottom pad
        } // end of processing channels
    } // end of processing batchs
}

#else

void conv_depthwise_3x3s1p1_bias_s(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    //! for 4x6 convolution window
    const int right_pad_idx[4] = {3, 2, 1, 0};

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    int tile_h = (h_in + 1) >> 1;

    int size_pad_right = 4 - w_in;
    int size_pad_bottom = 1 + (tile_h << 1) - h_in;

    uint32x4_t vmask_rp = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(size_pad_right));
    int right_pad_sub = size_pad_right * sizeof(float);

    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
    const float* ptr_zero = zero;

    int flag_hin1 = int(h_in == 1);

    //printf("size_pad_right: %d, right_pad_sub: %d, cnt_col: %d\n", size_pad_right, right_pad_sub, cnt_col);
    //unsigned int tmp1[4];
    //vst1q_u32(tmp1, vmask_rp);
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

            if (h_in == 2) {
                din2_ptr = ptr_zero;
            }
            if (h_in == 1) {
                din1_ptr = ptr_zero;
            }

            //! deal with top pad
            int h = 0;
            //! process

            asm volatile(
            //! process left right
            "pld [%[din0_ptr]]                @ preload data\n"
                    "pld [%[din1_ptr]]                @ preload data\n"
                    "pld [%[din2_ptr]]                @ preload data\n"

                    "vmov.u32 q15, #0 @ zero\n"

                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1, q10\n"
                    "vld1.32  {d22-d23}, [%[din1_ptr]]!     @ load din r2, q11\n"
                    "vld1.32  {d24-d25}, [%[din2_ptr]]!     @ load din r3, q12\n"

                    "vbif q10, q15, %q[mask]                @ bit select, deal with right pad\n"
                    "vbif q11, q15, %q[mask]                @ bit select, deal with right pad\n"
                    "vbif q12, q15, %q[mask]                @ bit select, deal with right pad\n"

                    "vmul.f32 q4, q10, %e[wr1][1]           @ mul weight 10, outr0\n"
                    "vmul.f32 q5, q10, %e[wr0][1]           @ mul weight 00, outr1\n"

                    "vext.32  q6, q10, q15, #1              @ shift left r1\n"
                    "vmla.f32 q4, q11, %e[wr2][1]           @ mul weight 20, outr0\n"
                    "vmla.f32 q5, q11, %e[wr1][1]           @ mul weight 10, outr1\n"

                    "vmla.f32 q5, q12, %e[wr2][1]           @ mul weight 20, outr1\n"

                    "vext.32  q8, q11, q15, #1              @ shift left r2\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"

                    "vext.32  q6, q12, q15, #1              @ shift left r3\n"
                    "vmla.f32 q4, q8,  %f[wr2][0]           @ mul weight 21, outr0\n"
                    "vmla.f32 q5, q8,  %f[wr1][0]           @ mul weight 11, outr1\n"

                    "vext.32  q8, q15, q10, #3              @ shift right r1\n"
                    "vmla.f32 q5, q6,  %f[wr2][0]           @ mul weight 21, outr1\n"

                    "vext.32  q6, q15, q11, #3              @ shift right r2\n"
                    "pld [%[dout_ptr1]]                @ preload data\n"
                    "vmla.f32 q4, q8, %e[wr1][0]            @ mul weight 12, outr0\n"
                    "vmla.f32 q5, q8, %e[wr0][0]            @ mul weight 02, outr1\n"

                    "vext.32  q8, q15, q12, #3              @ shift right r3\n"
                    "pld [%[dout_ptr2]]                @ preload data\n"
                    "vmla.f32 q4, q6,  %e[wr2][0]           @ mul weight 22, outr0\n"
                    "vmla.f32 q5, q6,  %e[wr1][0]           @ mul weight 12, outr1\n"

                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                    "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"
                    "vmla.f32 q5, q8,  %e[wr2][0]           @ mul weight 22, outr1\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vbif q8, q10, %q[mask]                 @ bit select\n"
                    "vbif q9, q11, %q[mask]                 @ bit select\n"

                    "sub %[din0_ptr], %[din0_ptr],   %[pad_right] @ sub \n"
                    "sub %[din1_ptr], %[din1_ptr],   %[pad_right] @ sub \n"
                    "sub %[din2_ptr], %[din2_ptr],   %[pad_right] @ sub \n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"

                    "cmp %[flag1],  #1                      @ check if hin = 1\n"
                    "beq    1f                              @ jump to next block\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "sub %[dout_ptr2], %[dout_ptr2], %[pad_right] @ sub \n"
                    "1:                                     @ end top\n"

                    "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"


            :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [pad_right] "+r" (right_pad_sub)
            :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                        [bias] "w"(wbias), [mask] "w" (vmask_rp), [flag1] "r" (flag_hin1)
            :"q4", "q5", "q6", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
            );

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

                asm volatile(
                //! process left pad
                "pld [%[din0_ptr]]               @ preload data\n"
                        "pld [%[din1_ptr]]               @ preload data\n"
                        "pld [%[din2_ptr]]               @ preload data\n"
                        "pld [%[din3_ptr]]               @ preload data\n"

                        "vmov.u32 q15, #0 @ zero\n"

                        "vld1.32  {d16-d17}, [%[din0_ptr]]!    @ load din r0, q8\n"
                        "vld1.32  {d18-d19}, [%[din1_ptr]]!    @ load din r1, q9\n"
                        "vld1.32  {d20-d21}, [%[din2_ptr]]!    @ load din r2, q10\n"
                        "vld1.32  {d22-d23}, [%[din3_ptr]]!    @ load din r3, q11\n"

                        "vbif q8, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vbif q9, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vbif q10, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vbif q11, q15, %q[mask]                @ bit select, deal with right pad\n"

                        "vmul.f32 q4, q8, %e[wr0][1]   @mul weight 00, outr0\n"

                        "vmul.f32 q5, q9, %e[wr0][1]  @mul weight 00, outr1\n"
                        "vmla.f32 q4, q9, %e[wr1][1]  @mul weight 10, outr0\n"

                        "vext.32  q6, q8, q15, #1     @ shift left r0\n"
                        "vmla.f32 q5, q10, %e[wr1][1]  @mul weight 10, outr1\n"
                        "vmla.f32 q4, q10, %e[wr2][1]  @mul weight 20, outr0\n"

                        "vext.32  q14, q9, q15, #1   @ shift left r1\n"
                        "vmla.f32 q5, q11, %e[wr2][1]  @mul weight 20, outr1\n"

                        "vext.32  q13, q10, q15, #1  @ shift left r2\n"
                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 01, outr0\n"

                        "vext.32  q6, q11, q15, #1  @ shift left r3\n"
                        "vmla.f32 q5, q14, %f[wr0][0]  @mul weight 01, outr1\n"
                        "vmla.f32 q4, q14, %f[wr1][0]  @mul weight 11, outr0\n"

                        "vext.32  q14, q15, q8, #3              @ shift right r0\n"
                        "vmla.f32 q5, q13,  %f[wr1][0] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q13,  %f[wr2][0] @mul weight 21, outr0\n"

                        "vext.32  q13, q15, q9, #3              @ shift right r1\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 21, outr1\n"

                        "vext.32  q6, q15, q10, #3              @ shift right r2\n"
                        "vmla.f32 q4, q14, %e[wr0][0]  @mul weight 02, outr0\n"

                        "vext.32  q14, q15, q11, #3              @ shift right r3\n"
                        "vmla.f32 q5, q13, %e[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q13, %e[wr1][0]  @mul weight 12, outr0\n"
                        
                        "pld [%[dout_ptr1]]                @ preload data\n"
                        "pld [%[dout_ptr2]]                @ preload data\n"
                        "vmla.f32 q5, q6,  %e[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][0] @mul weight 22, outr0\n"

                        "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                        "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"
                        "vmla.f32 q5, q14,  %e[wr2][0] @mul weight 22, outr1\n"

                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "sub %[din0_ptr], %[din0_ptr],   %[pad_right] @ sub \n"
                        "sub %[din1_ptr], %[din1_ptr],   %[pad_right] @ sub \n"

                        "vbif q8, q10, %q[mask]                 @ bit select\n"
                        "vbif q9, q11, %q[mask]                 @ bit select\n"

                        "sub %[din2_ptr], %[din2_ptr],   %[pad_right] @ sub \n"
                        "sub %[din3_ptr], %[din3_ptr],   %[pad_right] @ sub \n"
                        "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                        "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"
                        "sub %[dout_ptr2], %[dout_ptr2], %[pad_right] @ sub \n"


                :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                            [pad_right] "+r" (right_pad_sub)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                        [bias] "w"(wbias), [mask] "w" (vmask_rp)
                :"q4", "q5", "q6", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
                );

                doutr0 += w_out;
                doutr1 = doutr0 + w_out;
                dr0 = dr2;
                dr1 = dr3;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
            } //! end of processing mid rows

            //! deal with bottom pad
            if (h_in > 2) {
                din0_ptr = dr0;
                din1_ptr = dr1;
                if (size_pad_bottom == 2){
                    din2_ptr = ptr_zero;
                } else {
                    din2_ptr = dr2;
                }

                asm volatile(
                // process left pad
                "pld [%[din0_ptr]]                        @ preload data\n"
                        "pld [%[din1_ptr]]                        @ preload data\n"
                        "pld [%[din2_ptr]]                        @ preload data\n"

                        "vmov.u32 q15, #0 @ zero\n"

                        "vld1.32  {d16-d17}, [%[din0_ptr]]!     @ load din r0, q8\n"
                        "vld1.32  {d18-d19}, [%[din1_ptr]]!     @ load din r1, q9\n"
                        "vld1.32  {d20-d21}, [%[din2_ptr]]      @ load din r2, q10\n"

                        "vbif q8, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vbif q9, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vbif q10, q15, %q[mask]                @ bit select, deal with right pad\n"

                        "vmul.f32 q4, q8, %e[wr0][1]            @ mul weight 00, outr0\n"

                        "vext.32  q6, q8, q15, #1               @ shift left r0\n"
                        "vmla.f32 q4, q9, %e[wr1][1]           @ mul weight 10, outr0\n"
                        "vmul.f32 q5, q9, %e[wr0][1]           @ mul weight 00, outr1\n"

                        "vext.32  q13, q9, q15, #1               @ shift left r1\n"
                        "vmla.f32 q4, q10, %e[wr2][1]           @ mul weight 20, outr0\n"
                        "vmla.f32 q5, q10, %e[wr1][1]           @ mul weight 10, outr1\n"

                        "vext.32  q14, q10, q15, #1              @ shift left r2\n"
                        "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 01, outr0\n"

                        "vext.32  q6, q15, q8, #3               @ shift right r0\n"
                        "vmla.f32 q5, q13, %f[wr0][0]            @ mul weight 01, outr1\n"
                        "vmla.f32 q4, q13, %f[wr1][0]            @ mul weight 11, outr0\n"

                        "vext.32  q13, q15, q9, #3               @ shift right r1\n"
                        "vmla.f32 q5, q14,  %f[wr1][0]           @ mul weight 11, outr1\n"
                        "vmla.f32 q4, q14,  %f[wr2][0]           @ mul weight 21, outr0\n"

                        "vext.32  q14, q15, q10, #3              @ shift right r2\n"
                        "vmla.f32 q4, q6, %e[wr0][0]            @ mul weight 02, outr0\n"
                        
                        "pld [%[dout_ptr1]]                @ preload data\n"
                        "pld [%[dout_ptr2]]                @ preload data\n"
                        "vmla.f32 q4, q13, %e[wr1][0]            @ mul weight 12, outr0\n"
                        "vmla.f32 q5, q13, %e[wr0][0]            @ mul weight 02, outr1\n"
                        
                        "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                        "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"
                        "vmla.f32 q4, q14,  %e[wr2][0]           @ mul weight 22, outr0\n"
                        "vmla.f32 q5, q14,  %e[wr1][0]           @ mul weight 12, outr1\n"

                        "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                        "vbif q8, q10, %q[mask]                 @ bit select\n"
                        "vbif q9, q11, %q[mask]                 @ bit select\n"

                        "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                        "beq    1f                              @ jump to next block\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                        "1:                                     @\n"  
                        "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"

                :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [pad_right] "+r"(right_pad_sub), \
                            [bot_pad] "+r"(size_pad_bottom)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                        [bias] "w"(wbias), [mask] "w"(vmask_rp)
                //, [test] "r"(data_test_ptr)
                :"q4", "q5", "q6", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
                );

            }
            //! end of processing bottom pad
        } // end of processing channels
    } // end of processing batchs
}
#endif
/**
 * \brief depthwise convolution kernel 3x3, stride 2, width <= 4
 */
#ifdef __aarch64__
#if 1  //w <= 7
void conv_depthwise_3x3s2p1_bias_s(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {

    int right_pad_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    int out_pad_idx[4] = {0, 1, 2, 3};
    int size_pad_bottom = h_out * 2 - h_in;

    int size_right_remain = w_in;

    int size_right_pad = w_out * 2 - w_in;

    int cnt_remain = w_out;

    uint32x4_t vmask_rp1 = vcgtq_s32(vdupq_n_s32(size_right_remain), vld1q_s32(right_pad_idx));//0 2 4 6
    uint32x4_t vmask_rp2 = vcgtq_s32(vdupq_n_s32(size_right_remain), vld1q_s32(right_pad_idx + 4));//1 3 5 7
    uint32x4_t wmask = vcgtq_s32(vdupq_n_s32(cnt_remain), vld1q_s32(out_pad_idx));//0 1 2 3
    //printf("w_in %d, remain: %d \n", w_in, size_right_remain);
   // printf("mask1: %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3]);
    //printf("mask2: %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3]);
    //printf("wmask: %d, %d, %d, %d \n", wmask[0], wmask[1], wmask[2], wmask[3]);
   // size_right_remain *= sizeof(float);

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

            float32x4_t vzero= vdupq_n_f32(0.f);

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
            float *doutr0_ptr = doutr0;

            //! top pad
            if(1){
               //printf("cnt_col: %d, remain: %d \n", cnt_col, size_right_remain);
             //  printf("mask1: %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3]);
             //  printf("mask2: %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3]);
              // printf("wmask: %d, %d, %d, %d \n", wmask[0], wmask[1], wmask[2], wmask[3]);

                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias
                "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
               // "bif  v10.16b, %[vzero].16b, %[wmask].16b    \n" //pipei
                "ext  v2.16b, %[vzero].16b, v1.16b, #12     \n" // v6 = {0, 1, 3, 5}
                "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei


                "fmul v8.4s, v0.4s, %[w1].s[1]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w1].s[2]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w1].s[0]            \n" // v2 * w00

                "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
                "ext  v6.16b, %[vzero].16b, v5.16b, #12     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v4.4s, %[w2].s[1]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w2].s[2]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w2].s[0]            \n" // v2 * w00

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
              //  "fmax  v0.4s, v0.4s, %[vzero].4s             \n"
                "bif  v0.16b, %[vzero].16b, %[wmask].16b    \n" //pipei

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [outptr] "+r"(doutr0_ptr), \
                  [vzero] "+w" (vzero), [w1] "+w" (wr1), [w2] "+w" (wr2), \
                  [mask1] "+w" (vmask_rp1), [mask2] "+w" (vmask_rp2), [wmask] "+w" (wmask), [vbias] "+w" (wbias)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
                );
            }

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 = doutr0 + w_out;
            //! mid
            for (int j = h_out - size_pad_bottom - 1; j > 0; j--){
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;

                doutr0_ptr = doutr0;


                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
               // "bif  v10.16b, %[vzero].16b, %[wmask].16b    \n" //pipei
                "ext  v2.16b, %[vzero].16b, v1.16b, #12     \n" // v6 = {0,1,3,5}
                "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei

                "ld2  {v12.4s, v13.4s}, [%[inptr2]], #32    \n"
                "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
                "ext  v6.16b, %[vzero].16b, v5.16b, #12     \n" // v6 = {2,4,6,8}

                "fmul v8.4s, v0.4s, %[w0].s[1]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w0].s[2]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w0].s[0]            \n" // v2 * w00

                "bif  v12.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v13.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                "fmla v8.4s, v4.4s, %[w1].s[1]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[2]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w1].s[0]            \n" // v2 * w00

                "ext  v14.16b, %[vzero].16b, v13.16b, #12     \n" // v6 = {0,1,3,5}

                "fmla v8.4s, v12.4s, %[w2].s[1]            \n" // v0 * w01
                "fmla v9.4s, v13.4s, %[w2].s[2]            \n" // v1 * w02
                "fmla v10.4s, v14.4s, %[w2].s[0]            \n" // v2 * w00

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
               // "fmax  v0.4s, v0.4s, %[vzero].4s             \n"

                "bif  v0.16b, %[vzero].16b, %[wmask].16b    \n" //pipei

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [inptr2] "+r"(din2_ptr), [outptr] "+r"(doutr0_ptr), \
                  [vzero] "+w" (vzero), [w1] "+w" (wr1), [w2] "+w" (wr2), [w0] "+w" (wr0), \
                  [mask1] "+w" (vmask_rp1), [mask2] "+w" (vmask_rp2), [wmask] "+w" (wmask), [vbias] "+w" (wbias)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14"
                );
                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
                doutr0 = doutr0 + w_out;
            }

            if (size_pad_bottom){
                din0_ptr = dr0;
                din1_ptr = dr1;

                doutr0_ptr = doutr0;

                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias
                "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
               // "bif  v10.16b, %[vzero].16b, %[wmask].16b    \n" //pipei
                "ext  v2.16b, %[vzero].16b, v1.16b, #12    \n" // v6 = {2,4,6,8}
                "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei


                "fmul v8.4s, v0.4s, %[w0].s[0]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w0].s[1]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w0].s[2]            \n" // v2 * w00

                "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
                "ext  v6.16b, %[vzero].16b, v5.16b, #12     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v4.4s, %[w1].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[1]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w1].s[2]            \n" // v2 * w00

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
             //   "fmax  v0.4s, v0.4s, %[vzero].4s             \n"
                "bif  v0.16b, %[vzero].16b, %[wmask].16b    \n" //pipei

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "4:                                          \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [outptr] "+r"(doutr0_ptr), \
                  [vzero] "+w" (vzero), [w0] "+w" (wr0), [w1] "+w" (wr1), \
                  [mask1] "+w" (vmask_rp1), [mask2] "+w" (vmask_rp2), [wmask] "+w" (wmask), [vbias] "+w" (wbias)
                : [remain] "r" (cnt_remain)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
                );
            }

        }
    }


}
#else
void conv_depthwise_3x3s2p1_bias_s(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s2 depthwise convolution, pad 1 is done implicit

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    const int right_pad_idx[4] = {3, 2, 1, 0};
    int size_pad_right = 4 - w_in;
    int h_even = ((h_in >> 1) << 1);
    int size_pad_bottom = (h_even == h_in) ? 0 : 1;//4 --0, 5 --1
    int tile_h = h_even >> 1;

    uint32x4_t vmask_rp = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(size_pad_right));
    int right_pad_sub = size_pad_right * sizeof(float);
    uint32x4_t vmask_w = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(0));//1111
    if(w_in < 3)
        vmask_w = vsetq_lane_u32(0, vmask_w, 1);//1011

    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
    const float* ptr_zero = zero;

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
            float *doutr0_ptr = doutr0;

            //! top pad
            if(h_in == 1){
                din1_ptr = ptr_zero;
            }

            float32x4_t vzero = vdupq_n_f32(0.f);
            // todo
            float32x4_t din0_1234 = vld1q_f32(din0_ptr);
            float32x4_t din1_1234 = vld1q_f32(din1_ptr);

            if(vgetq_lane_u32(vmask_rp, 0) == 0){
                din0_1234 = vdupq_n_f32(0.f);
                din1_1234 = vdupq_n_f32(0.f);
            }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 1);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 1);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
            }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
            }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
            }

            float32x4_t sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr1), 1);
            sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr2), 1);

            float32x4_t din0_0123 = vextq_f32(vzero, din0_1234, 3);
            float32x4_t din1_0123 = vextq_f32(vzero, din1_1234, 3);
            sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr1), 0);
            sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr2), 0);

            float32x4_t din0_2340 = vextq_f32(din0_1234, vzero, 1);
            float32x4_t din1_2340 = vextq_f32(din1_1234, vzero, 1);
            sum0 = vmlaq_lane_f32(sum0, din0_2340, vget_high_f32(wr1), 0);
            sum0 = vmlaq_lane_f32(sum0, din1_2340, vget_high_f32(wr2), 0);

            sum0 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), sum0, 1);

            float32x2_t sum = vadd_f32(vget_low_f32(sum0), vget_low_f32(wbias));

            float32x2_t dout0_12 = vld1_f32(doutr0_ptr);

            if(vgetq_lane_u32(vmask_w, 1) == 0){//10
                sum = vset_lane_f32(vget_lane_f32(dout0_12, 1), sum, 1);
            }

            vst1_f32(doutr0_ptr, sum);

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 = doutr0 + w_out;

            //! process mid rows
            for (int j = tile_h - 1; j > 0; j--) {
            // todo
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                doutr0_ptr = doutr0;

                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                float32x4_t din2_1234 = vld1q_f32(din2_ptr);

                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                    din0_1234 = vdupq_n_f32(0.f);
                    din1_1234 = vdupq_n_f32(0.f);
                    din2_1234 = vdupq_n_f32(0.f);
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 1);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 1);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 1);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }

                sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 1);
                sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 1);

                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);
                float32x4_t din2_0123 = vextq_f32(vzero, din2_1234, 3);
                sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);

                din0_2340 = vextq_f32(din0_1234, vzero, 1);
                din1_2340 = vextq_f32(din1_1234, vzero, 1);
                float32x4_t din2_2340 = vextq_f32(din2_1234, vzero, 1);
                sum0 = vmlaq_lane_f32(sum0, din0_2340, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_2340, vget_high_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_2340, vget_high_f32(wr2), 0);

                sum0 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), sum0, 1);

                sum = vadd_f32(vget_low_f32(sum0), vget_low_f32(wbias));

                dout0_12 = vld1_f32(doutr0_ptr);

                if(vgetq_lane_u32(vmask_w, 1) == 0){//10
                    sum = vset_lane_f32(vget_lane_f32(dout0_12, 1), sum, 1);
                }

                vst1_f32(doutr0_ptr, sum);

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
                doutr0 += w_out;
            } // end of process mid rows

            // process bottom pad if needed
            if (size_pad_bottom) {
                din0_ptr = dr0;
                din1_ptr = dr1;
                doutr0_ptr = doutr0;

                // todo
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);

                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                    din0_1234 = vdupq_n_f32(0.f);
                    din1_1234 = vdupq_n_f32(0.f);
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 1);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 1);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
                }

                sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 1);

                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);
                sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);

                din0_2340 = vextq_f32(din0_1234, vzero, 1);
                din1_2340 = vextq_f32(din1_1234, vzero, 1);
                sum0 = vmlaq_lane_f32(sum0, din0_2340, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_2340, vget_high_f32(wr1), 0);

                sum0 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), sum0, 1);

                sum = vadd_f32(vget_low_f32(sum0), vget_low_f32(wbias));

                dout0_12 = vld1_f32(doutr0_ptr);

                if(vgetq_lane_u32(vmask_w, 1) == 0){//10
                    sum = vset_lane_f32(vget_lane_f32(dout0_12, 1), sum, 1);
                }

                vst1_f32(doutr0_ptr, sum);

            } // end of process bottom pad
        }
    }
}
#endif
#else
void conv_depthwise_3x3s2p1_bias_s(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s2 depthwise convolution, pad 1 is done implicit

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    const int right_pad_idx[4] = {3, 2, 1, 0};
    int size_pad_right = 4 - w_in;
    int h_even = ((h_in >> 1) << 1);
    int size_pad_bottom = (h_even == h_in) ? 0 : 1;//4 --0, 5 --1
    int tile_h = h_even >> 1;

    uint32x4_t vmask_rp = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(size_pad_right));
    int right_pad_sub = size_pad_right * sizeof(float);
    uint32x4_t vmask_w = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(0));
    if(w_in < 3)
        vmask_w = vsetq_lane_u32(0, vmask_w, 1);//1101

    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
    const float* ptr_zero = zero;

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
            float *doutr0_ptr = doutr0;

            //! top pad
            if(h_in == 1){
                din1_ptr = ptr_zero;
            }

            asm volatile(
            // process  pad
            "pld [%[din0_ptr], #128]                @ preload data\n"
                    "pld [%[din1_ptr], #128]                @ preload data\n"

                    "vmov.u32  q15, #0                      @ zero buf\n"

                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0 1234\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1 1234\n"
                    "vbif q10, q15, %q[mask_din]            @ bit select, deal with right pad\n"
                    "vbif q12, q15, %q[mask_din]            @ bit select, deal with right pad\n"

                    "vext.32 q6, q15, q10, #3              @ shift right 1, 0123\n"
                    "vext.32 q7, q15, q12, #3              @ shift right 1, 0123\n"
                    "vmul.f32 q8, q10, %e[wr1][1]           @ mul weight 11, 1234\n"
                    "vmul.f32 q9, q12, %e[wr2][1]           @ mul weight 21, 1234\n"

                    "vext.32 q10, q10, q15, #1              @ shift left 1, 2340\n"
                    "vext.32 q12, q12, q15, #1              @ shift left 1, 2340\n"

                    "vmla.f32 q8, q6, %e[wr1][0]           @ mla weight 10, 0123\n"
                    "vmla.f32 q9, q7, %e[wr2][0]           @ mla weight 20, 0123\n"
                    
                    "pld [%[dout_ptr0]]                     @ preload data\n"
                    "vmla.f32 q8, q10, %f[wr1][0]           @ mla weight 12, 2340\n"
                    "vmla.f32 q9, q12, %f[wr2][0]           @ mla weight 22, 2340\n"

                    "vld1.32  {d20}, [%[dout_ptr0]]         @ load dout\n"
                    "vadd.f32 q6, q8, q9                    @ add \n"

                    "vmov.f32  s25, s26                     @mov \n"

                    "vadd.f32  d12, d12, %e[bias]           @ add bias \n"

                    //"vst1.32  {d17},   [%[tmp_ptr]]! \n"
                    "vbif d12, d20, %e[mask_w]              @ bit select\n"

                    //"vst1.32  {d17},   [%[tmp_ptr]] \n"
                    "vst1.32  {d12}, [%[dout_ptr0]]!        @ store result, add pointer\n"
            // "sub %[dout_ptr0], %[dout_ptr0], %[pad_right] @ sub \n"

            :[dout_ptr0] "+r"(doutr0_ptr), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr)
            :[wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask_din] "w" (vmask_rp), [mask_w]  "w" (vmask_w) //, \
                    //[pad_right] "r" (size_right_remain)
            :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 = doutr0 + w_out;

            //! process mid rows
            for (int j = tile_h - 1; j > 0; j--) {

                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                doutr0_ptr = doutr0;

                asm volatile(
                // process  pad
                "pld [%[din0_ptr], #128]                @ preload data\n"
                        "pld [%[din1_ptr], #128]                @ preload data\n"
                        "pld [%[din2_ptr], #128]                @ preload data\n"

                        "vmov.u32  q15, #0                      @ zero buf\n"

                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0 1234\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1 1234\n"
                        "vld1.32  {d28-d29}, [%[din2_ptr]]!     @ load din r1 1234\n"
                        "vbif q10, q15, %q[mask_din]            @ bit select, deal with right pad\n"
                        "vbif q12, q15, %q[mask_din]            @ bit select, deal with right pad\n"
                        "vbif q14, q15, %q[mask_din]            @ bit select, deal with right pad\n"


                        "vext.f32 q6, q15, q10, #3              @ shift right 1, 0123\n"
                        "vext.f32 q7, q15, q12, #3              @ shift right 1, 0123\n"
                        "vext.f32 q8, q15, q14, #3              @ shift right 1, 0123\n"

                        "vmul.f32 q9, q10, %e[wr0][1]           @ mul weight 01, 1234\n"
                        "vmul.f32 q11, q12, %e[wr1][1]           @ mul weight 11, 1234\n"
                        "vmul.f32 q13, q14, %e[wr2][1]           @ mul weight 21, 1234\n"

                        "vext.f32 q10, q10, q15, #1              @ shift left 1, 2340\n"
                        "vext.f32 q12, q12, q15, #1              @ shift left 1, 2340\n"
                        "vext.f32 q14, q14, q15, #1              @ shift left 1, 2340\n"

                        "vmla.f32 q9, q6, %e[wr0][0]           @ mul weight 00, 0123\n"
                        "vmla.f32 q11, q7, %e[wr1][0]           @ mul weight 10, 0123\n"
                        "vmla.f32 q13, q8, %e[wr2][0]           @ mul weight 20, 0123\n"

                        "pld [%[dout_ptr0]]                    @ preload data\n"
                        "vmla.f32 q9, q10, %f[wr0][0]           @ mul weight 12, 2340\n"
                        "vmla.f32 q11, q12, %f[wr1][0]           @ mul weight 22, 2340\n"
                        "vmla.f32 q13, q14, %f[wr2][0]           @ mul weight 22, 2340\n"

                        "vld1.32  {d20}, [%[dout_ptr0]]         @ load dout\n"
                        "vadd.f32  q8, q9, q11                   @add\n"
                        "vadd.f32  q6, q8, q13                   @add\n"

                        "vmov.f32  s25, s26                     @ mov \n"
                        "vadd.f32  d12, d12, %e[bias]           @ add bias \n"

                        "vbif d12, d20, %e[mask_w]              @ bit select\n"

                        "vst1.32  {d12}, [%[dout_ptr0]]!        @ store result, add pointer\n"

                :[dout_ptr0] "+r"(doutr0_ptr), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr), [din2_ptr] "+r"(din2_ptr)
                :[wr1] "w"(wr1), [wr2] "w"(wr2), [wr0] "w" (wr0), \
                    [bias] "w"(wbias), [mask_din] "w" (vmask_rp), [mask_w] "w" (vmask_w) //, \
                    //[pad_right] "r" (size_right_remain)
                :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );


                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
                doutr0 = doutr0 + w_out;
            } // end of process mid rows
            // process bottom pad if needed
            if (size_pad_bottom) {
                din0_ptr = dr0;
                din1_ptr = dr1;

                asm volatile(
                // process  pad
                "pld [%[din0_ptr], #128]                @ preload data\n"
                        "pld [%[din1_ptr], #128]                @ preload data\n"

                        "vmov.u32  q15, #0                      @ zero buf\n"

                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0 1234\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1 1234\n"
                        "vbif q10, q15, %q[mask_din]            @ bit select, deal with right pad\n"
                        "vbif q12, q15, %q[mask_din]            @ bit select, deal with right pad\n"


                        "vext.f32 q6, q15, q10, #3              @ shift right 1, 0123\n"
                        "vext.f32 q7, q15, q12, #3              @ shift right 1, 0123\n"

                        "vmul.f32 q8, q10, %e[wr1][1]           @ mul weight 11, 1234\n"
                        "vmul.f32 q9, q12, %e[wr2][1]           @ mul weight 21, 1234\n"

                        "vext.f32 q10, q10, q15, #1              @ shift left 1, 2340\n"
                        "vext.f32 q12, q12, q15, #1              @ shift left 1, 2340\n"

                        "vmla.f32 q8, q6, %e[wr1][0]           @ mul weight 10, 0123\n"
                        "vmla.f32 q9, q7, %e[wr2][0]           @ mul weight 20, 0123\n"
                        
                        "pld [%[dout_ptr0]]                    @ preload data\n"
                        "vmla.f32 q8, q10, %f[wr1][0]           @ mul weight 12, 2340\n"
                        "vmla.f32 q9, q12, %f[wr2][0]           @ mul weight 22, 2340\n"

                        "vld1.32  {d20}, [%[dout_ptr0]]         @ load dout\n"
                        "vadd.f32 q6, q8, q9                    @ add \n"

                        "vmov.f32  s25, s26                     @ mov \n"
                        "vadd.f32  d12, d12, %e[bias]           @ add bias \n"

                        //"vst1.32  {d17},   [%[tmp_ptr]]! \n"
                        "vbif d12, d20, %e[mask_w]              @ bit select\n"

                        "vst1.32  {d12}, [%[dout_ptr0]]!        @ store result, add pointer\n"

                :[dout_ptr0] "+r"(doutr0_ptr), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr)
                :[wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask_din] "w" (vmask_rp), [mask_w]  "w" (vmask_w) //, \
                    //[pad_right] "r" (size_right_remain)
                :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
            } // end of process bottom pad
        }
    }
}
#endif
/**
 * \brief depthwise convolution, kernel size 3x3, stride 1, pad 1, with bias, width <= 4
 */
#ifdef __aarch64__
void conv_depthwise_3x3s1p1_bias_s_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    //! for 4x6 convolution window
    const int right_pad_idx[4] = {3, 2, 1, 0};

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    int tile_h = (h_in + 1) >> 1;

    int size_pad_right = 4 - w_in;
    int size_pad_bottom = 1 + (tile_h << 1) - h_in;

    uint32x4_t vmask_rp = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(size_pad_right));
    int right_pad_sub = size_pad_right * sizeof(float);

    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
    const float* ptr_zero = zero;

    int flag_hin1 = int(h_in == 1);

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

            if (h_in == 2) {
                din2_ptr = ptr_zero;
            }
            if (h_in == 1) {
                din1_ptr = ptr_zero;
            }

            //! deal with top pad
            int h = 0;
            //! process
            float32x4_t vzero = vdupq_n_f32(0.f);
            // todo
            float32x4_t din0_1234 = vld1q_f32(din0_ptr);
            float32x4_t din1_1234 = vld1q_f32(din1_ptr);
            float32x4_t din2_1234 = vld1q_f32(din2_ptr);

            if(vgetq_lane_u32(vmask_rp, 0) == 0){
                din0_1234 = vdupq_n_f32(0.f);
                din1_1234 = vdupq_n_f32(0.f);
                din2_1234 = vdupq_n_f32(0.f);
            }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 1);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 1);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 1);
                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
            }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
            }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
            }

            float32x4_t sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 1);
            float32x4_t sum1 = vmulq_lane_f32(din0_1234, vget_low_f32(wr1), 1);

            sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 1);
            sum1 = vmlaq_lane_f32(sum1, din1_1234, vget_low_f32(wr2), 1);

            sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 1);

            float32x4_t din0_2340 = vextq_f32(din0_1234, vzero, 1);
            float32x4_t din1_2340 = vextq_f32(din1_1234, vzero, 1);
            float32x4_t din2_2340 = vextq_f32(din2_1234, vzero, 1);

            sum0 = vmlaq_lane_f32(sum0, din0_2340, vget_high_f32(wr0), 0);
            sum1 = vmlaq_lane_f32(sum1, din0_2340, vget_high_f32(wr1), 0);

            sum0 = vmlaq_lane_f32(sum0, din1_2340, vget_high_f32(wr1), 0);
            sum1 = vmlaq_lane_f32(sum1, din1_2340, vget_high_f32(wr2), 0);

            sum0 = vmlaq_lane_f32(sum0, din2_2340, vget_high_f32(wr2), 0);

            float32x4_t din0_0123 = vextq_f32(vzero, din0_1234, 3);
            float32x4_t din1_0123 = vextq_f32(vzero, din1_1234, 3);
            float32x4_t din2_0123 = vextq_f32(vzero, din2_1234, 3);

            sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);
            sum1 = vmlaq_lane_f32(sum1, din0_0123, vget_low_f32(wr1), 0);

            sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);
            sum1 = vmlaq_lane_f32(sum1, din1_0123, vget_low_f32(wr2), 0);

            sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);

            float32x4_t dout0_1234 = vld1q_f32(doutr0);
            float32x4_t dout1_1234 = vld1q_f32(doutr1);

            sum0 = vaddq_f32(sum0, wbias);
            sum1 = vaddq_f32(sum1, wbias);

            sum0 = vmaxq_f32(sum0, vzero);
            sum1 = vmaxq_f32(sum1, vzero);

            if(vgetq_lane_u32(vmask_rp, 0) == 0){
                //null
            }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout1_1234, 0);
            }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                /*
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout1_1234, 0);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout0_1234, 1);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout1_1234, 1);
                */
                dout0_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(dout0_1234));
                dout1_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(dout1_1234));
            }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                /*
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout1_1234, 0);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout0_1234, 1);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout1_1234, 1);
                */
                dout0_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(dout0_1234));
                dout1_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(dout1_1234));

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), dout0_1234, 2);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), dout1_1234, 2);
            }else{
                /*
                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout0_1234, 0);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout1_1234, 0);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout0_1234, 1);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout1_1234, 1);

                dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), dout0_1234, 2);
                dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), dout1_1234, 2);
                */
                //dout0_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                //dout1_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(sum0));

                dout0_1234 = sum1;
                dout1_1234 = sum0;
            }

            vst1q_f32(doutr0, dout0_1234);//sum1
            if(!flag_hin1){//h > 1
                vst1q_f32(doutr1, dout1_1234);//sum0
            }

            //! after process, increase pointer
            doutr0 = doutr1 + w_out;
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

                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                din2_1234 = vld1q_f32(din2_ptr);
                float32x4_t din3_1234 = vld1q_f32(din3_ptr);


                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                    din0_1234 = vdupq_n_f32(0.f);
                    din1_1234 = vdupq_n_f32(0.f);
                    din2_1234 = vdupq_n_f32(0.f);
                    din3_1234 = vdupq_n_f32(0.f);
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 1);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 1);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 1);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);

                    din3_1234 = vsetq_lane_f32(0.f, din3_1234, 1);
                    din3_1234 = vsetq_lane_f32(0.f, din3_1234, 2);
                    din3_1234 = vsetq_lane_f32(0.f, din3_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);

                    din3_1234 = vsetq_lane_f32(0.f, din3_1234, 2);
                    din3_1234 = vsetq_lane_f32(0.f, din3_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);

                    din3_1234 = vsetq_lane_f32(0.f, din3_1234, 3);
                }

                sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 1);

                sum1 = vmulq_lane_f32(din1_1234, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 1);

                sum1 = vmlaq_lane_f32(sum1, din2_1234, vget_low_f32(wr1), 1);
                sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 1);

                sum1 = vmlaq_lane_f32(sum1, din3_1234, vget_low_f32(wr2), 1);

                din0_2340 = vextq_f32(din0_1234, vzero, 1);
                din1_2340 = vextq_f32(din1_1234, vzero, 1);
                din2_2340 = vextq_f32(din2_1234, vzero, 1);
                float32x4_t din3_2340 = vextq_f32(din3_1234, vzero, 1);

                sum0 = vmlaq_lane_f32(sum0, din0_2340, vget_high_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_2340, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_2340, vget_high_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_2340, vget_high_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_2340, vget_high_f32(wr2), 0);

                sum1 = vmlaq_lane_f32(sum1, din3_2340, vget_high_f32(wr2), 0);

                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);
                din2_0123 = vextq_f32(vzero, din2_1234, 3);
                float32x4_t din3_0123 = vextq_f32(vzero, din3_1234, 3);

                sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_0123, vget_low_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_0123, vget_low_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);

                sum1 = vmlaq_lane_f32(sum1, din3_0123, vget_low_f32(wr2), 0);

                float32x4_t dout0_1234 = vld1q_f32(doutr0);
                float32x4_t dout1_1234 = vld1q_f32(doutr1);

                sum0 = vaddq_f32(sum0, wbias);
                sum1 = vaddq_f32(sum1, wbias);

                sum0 = vmaxq_f32(sum0, vzero);
                sum1 = vmaxq_f32(sum1, vzero);

                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                //null
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                /*
                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);

                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout0_1234, 1);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout1_1234, 1);
                */
                    dout0_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(dout0_1234));
                    dout1_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(dout1_1234));
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                /*
                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);

                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout0_1234, 1);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout1_1234, 1);
                */
                    dout0_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(dout0_1234));
                    dout1_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(dout1_1234));

                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), dout0_1234, 2);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), dout1_1234, 2);
                }else{
                /*
                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);

                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout0_1234, 1);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout1_1234, 1);

                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), dout0_1234, 2);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), dout1_1234, 2);
                */
                    //dout0_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    //dout1_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(sum1));

                    dout0_1234 = sum0;
                    dout1_1234 = sum1;
                }

                vst1q_f32(doutr0, dout0_1234);
                vst1q_f32(doutr1, dout1_1234);

                doutr0 = doutr1 + w_out;
                doutr1 = doutr0 + w_out;
                dr0 = dr2;
                dr1 = dr3;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
            } //! end of processing mid rows

            //! deal with bottom pad
            if (h_in > 2) {
                din0_ptr = dr0;
                din1_ptr = dr1;
                if (size_pad_bottom == 2){
                    din2_ptr = ptr_zero;
                } else {
                    din2_ptr = dr2;
                }

                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                din2_1234 = vld1q_f32(din2_ptr);


                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                    din0_1234 = vdupq_n_f32(0.f);
                    din1_1234 = vdupq_n_f32(0.f);
                    din2_1234 = vdupq_n_f32(0.f);
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 1);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 1);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 1);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }

                sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 1);

                sum1 = vmulq_lane_f32(din1_1234, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 1);

                sum1 = vmlaq_lane_f32(sum1, din2_1234, vget_low_f32(wr1), 1);
                sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 1);

                din0_2340 = vextq_f32(din0_1234, vzero, 1);
                din1_2340 = vextq_f32(din1_1234, vzero, 1);
                din2_2340 = vextq_f32(din2_1234, vzero, 1);

                sum0 = vmlaq_lane_f32(sum0, din0_2340, vget_high_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_2340, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_2340, vget_high_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_2340, vget_high_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_2340, vget_high_f32(wr2), 0);

                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);
                din2_0123 = vextq_f32(vzero, din2_1234, 3);

                sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_0123, vget_low_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_0123, vget_low_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);

                float32x4_t dout0_1234 = vld1q_f32(doutr0);
                float32x4_t dout1_1234 = vld1q_f32(doutr1);

                sum0 = vaddq_f32(sum0, wbias);
                sum1 = vaddq_f32(sum1, wbias);

                sum0 = vmaxq_f32(sum0, vzero);
                sum1 = vmaxq_f32(sum1, vzero);

                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                //null
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                /*
                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);

                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout0_1234, 1);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout1_1234, 1);
                */
                    dout0_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(dout0_1234));
                    dout1_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(dout1_1234));
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                /*
                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);

                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout0_1234, 1);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout1_1234, 1);
                */
                    dout0_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(dout0_1234));
                    dout1_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(dout1_1234));

                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), dout0_1234, 2);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), dout1_1234, 2);
                }else{
                /*
                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), dout0_1234, 0);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), dout1_1234, 0);

                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 1), dout0_1234, 1);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 1), dout1_1234, 1);

                    dout0_1234 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), dout0_1234, 2);
                    dout1_1234 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), dout1_1234, 2);
                */
                    //dout0_1234 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    //dout1_1234 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(sum1));

                    dout0_1234 = sum0;
                    dout1_1234 = sum1;
                }

                vst1q_f32(doutr0, dout0_1234);
                if(size_pad_bottom != 2)
                    vst1q_f32(doutr1, dout1_1234);
            }
            //! end of processing bottom pad
        } // end of processing channels
    } // end of processing batchs
}

#else
void conv_depthwise_3x3s1p1_bias_s_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    //! for 4x6 convolution window
    const int right_pad_idx[4] = {3, 2, 1, 0};

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    int tile_h = (h_in + 1) >> 1;

    int size_pad_right = 4 - w_in;
    int size_pad_bottom = 1 + (tile_h << 1) - h_in;

    uint32x4_t vmask_rp = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(size_pad_right));
    int right_pad_sub = size_pad_right * sizeof(float);

    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
    const float* ptr_zero = zero;

    int flag_hin1 = int(h_in == 1);

    //printf("size_pad_right: %d, right_pad_sub: %d, cnt_col: %d\n", size_pad_right, right_pad_sub, cnt_col);
    //unsigned int tmp1[4];
    //vst1q_u32(tmp1, vmask_rp);
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

            if (h_in == 2) {
                din2_ptr = ptr_zero;
            }
            if (h_in == 1) {
                din1_ptr = ptr_zero;
            }

            //! deal with top pad
            int h = 0;
            //! process
            asm volatile(
            //! process left right
            "pld [%[din0_ptr]]                @ preload data\n"
                    "pld [%[din1_ptr]]                @ preload data\n"
                    "pld [%[din2_ptr]]                @ preload data\n"

                    "vmov.u32 q15, #0 @ zero\n"

                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r1, q10\n"
                    "vld1.32  {d22-d23}, [%[din1_ptr]]!     @ load din r2, q11\n"
                    "vld1.32  {d24-d25}, [%[din2_ptr]]!     @ load din r3, q12\n"

                    "vbif q10, q15, %q[mask]                @ bit select, deal with right pad\n"
                    "vbif q11, q15, %q[mask]                @ bit select, deal with right pad\n"
                    "vbif q12, q15, %q[mask]                @ bit select, deal with right pad\n"

                    "vmul.f32 q4, q10, %e[wr1][1]           @ mul weight 10, outr0\n"
                    "vmul.f32 q5, q10, %e[wr0][1]           @ mul weight 00, outr1\n"

                    "vext.32  q6, q10, q15, #1              @ shift left r1\n"
                    "vmla.f32 q4, q11, %e[wr2][1]           @ mul weight 20, outr0\n"
                    "vmla.f32 q5, q11, %e[wr1][1]           @ mul weight 10, outr1\n"

                    "vext.32  q13, q11, q15, #1              @ shift left r2\n"
                    "vmla.f32 q5, q12, %e[wr2][1]           @ mul weight 20, outr1\n"

                    "vext.32  q14, q12, q15, #1              @ shift left r3\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"

                    "vext.32  q6, q15, q10, #3              @ shift right r1\n"
                    "vmla.f32 q4, q13,  %f[wr2][0]           @ mul weight 21, outr0\n"
                    "vmla.f32 q5, q13,  %f[wr1][0]           @ mul weight 11, outr1\n"

                    "vext.32  q13, q15, q11, #3              @ shift right r2\n"
                    "vmla.f32 q5, q14,  %f[wr2][0]           @ mul weight 21, outr1\n"

                    "vext.32  q14, q15, q12, #3              @ shift right r3\n"
                    "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"
                    "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"
                    
                    "pld [%[dout_ptr1]]                @ preload data\n"
                    "vmla.f32 q4, q13,  %e[wr2][0]           @ mul weight 22, outr0\n"
                    "vmla.f32 q5, q13,  %e[wr1][0]           @ mul weight 12, outr1\n"

                    "pld [%[dout_ptr2]]                @ preload data\n"
                    "vmla.f32 q5, q14,  %e[wr2][0]           @ mul weight 22, outr1\n"

                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                    "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vmax.f32 q8, q15                       @ relu \n"
                    "vmax.f32 q9, q15                       @ relu \n"

                    "vbif q8, q10, %q[mask]                 @ bit select\n"
                    "vbif q9, q11, %q[mask]                 @ bit select\n"

                    "sub %[din0_ptr], %[din0_ptr],   %[pad_right] @ sub \n"
                    "sub %[din1_ptr], %[din1_ptr],   %[pad_right] @ sub \n"
                    "sub %[din2_ptr], %[din2_ptr],   %[pad_right] @ sub \n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"

                    "cmp %[flag1],  #1                      @ check if hin = 1\n"
                    "beq    1f                              @ jump to next block\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "sub %[dout_ptr2], %[dout_ptr2], %[pad_right] @ sub \n"
                    "1:                                     @ end top\n"
                    "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"


            :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [pad_right] "+r" (right_pad_sub)
            :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                        [bias] "w"(wbias), [mask] "w" (vmask_rp), [flag1] "r" (flag_hin1)
            :"q4", "q5", "q6", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
            );

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

                asm volatile(
                //! process left pad
                "pld [%[din0_ptr]]               @ preload data\n"
                        "pld [%[din1_ptr]]               @ preload data\n"
                        "pld [%[din2_ptr]]               @ preload data\n"
                        "pld [%[din3_ptr]]               @ preload data\n"

                        "vmov.u32 q15, #0 @ zero\n"

                        "vld1.32  {d16-d17}, [%[din0_ptr]]!    @ load din r0, q8\n"
                        "vld1.32  {d18-d19}, [%[din1_ptr]]!    @ load din r1, q9\n"
                        "vld1.32  {d20-d21}, [%[din2_ptr]]!    @ load din r2, q10\n"
                        "vld1.32  {d22-d23}, [%[din3_ptr]]!    @ load din r3, q11\n"

                        "vbif q8, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vbif q9, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vbif q10, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vbif q11, q15, %q[mask]                @ bit select, deal with right pad\n"

                        "vmul.f32 q4, q8, %e[wr0][1]   @mul weight 00, outr0\n"

                        "vext.32  q6, q8, q15, #1     @ shift left r0\n"
                        "vmul.f32 q5, q9, %e[wr0][1]  @mul weight 00, outr1\n"
                        "vmla.f32 q4, q9, %e[wr1][1]  @mul weight 10, outr0\n"

                        "vext.32  q13, q9, q15, #1   @ shift left r1\n"
                        "vmla.f32 q5, q10, %e[wr1][1]  @mul weight 10, outr1\n"
                        "vmla.f32 q4, q10, %e[wr2][1]  @mul weight 20, outr0\n"

                        "vext.32  q14, q10, q15, #1  @ shift left r2\n"
                        "vmla.f32 q5, q11, %e[wr2][1]  @mul weight 20, outr1\n"

                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 01, outr0\n"

                        "vext.32  q6, q11, q15, #1  @ shift left r3\n"
                        "vmla.f32 q5, q13, %f[wr0][0]  @mul weight 01, outr1\n"
                        "vmla.f32 q4, q13, %f[wr1][0]  @mul weight 11, outr0\n"

                        "vext.32  q13, q15, q8, #3              @ shift right r0\n"
                        "vmla.f32 q5, q14,  %f[wr1][0] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q14,  %f[wr2][0] @mul weight 21, outr0\n"

                        "vext.32  q14, q15, q9, #3              @ shift right r1\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 21, outr1\n"

                        "vext.32  q6, q15, q10, #3              @ shift right r2\n"
                        "vmla.f32 q4, q13, %e[wr0][0]  @mul weight 02, outr0\n"

                        "vext.32  q13, q15, q11, #3              @ shift right r3\n"
                        "vmla.f32 q5, q14, %e[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q14, %e[wr1][0]  @mul weight 12, outr0\n"
                        
                        "pld [%[dout_ptr1]]               @ preload data\n"
                        "vmla.f32 q5, q6,  %e[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][0] @mul weight 22, outr0\n"
                        
                        "pld [%[dout_ptr2]]               @ preload data\n"
                        "vmla.f32 q5, q13,  %e[wr2][0] @mul weight 22, outr1\n"

                        "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                        "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"

                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "sub %[din0_ptr], %[din0_ptr],   %[pad_right] @ sub \n"
                        "vmax.f32 q8, q15             @ relu \n"
                        "vmax.f32 q9, q15             @ relu \n"

                        "vbif q8, q10, %q[mask]                 @ bit select\n"
                        "vbif q9, q11, %q[mask]                 @ bit select\n"

                        "sub %[din1_ptr], %[din1_ptr],   %[pad_right] @ sub \n"
                        "sub %[din2_ptr], %[din2_ptr],   %[pad_right] @ sub \n"
                        "sub %[din3_ptr], %[din3_ptr],   %[pad_right] @ sub \n"

                        "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                        "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"
                        "sub %[dout_ptr2], %[dout_ptr2], %[pad_right] @ sub \n"


                :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                            [pad_right] "+r" (right_pad_sub)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                        [bias] "w"(wbias), [mask] "w" (vmask_rp)
                :"q4", "q5", "q6", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
                );

                doutr0 += w_out;
                doutr1 = doutr0 + w_out;
                dr0 = dr2;
                dr1 = dr3;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
            } //! end of processing mid rows

            //! deal with bottom pad
            if (h_in > 2) {
                din0_ptr = dr0;
                din1_ptr = dr1;
                if (size_pad_bottom == 2){
                    din2_ptr = ptr_zero;
                } else {
                    din2_ptr = dr2;
                }
                asm volatile(
                // process left pad
                "pld [%[din0_ptr]]                        @ preload data\n"
                        "pld [%[din1_ptr]]                        @ preload data\n"
                        "pld [%[din2_ptr]]                        @ preload data\n"

                        "vmov.u32 q15, #0 @ zero\n"

                        "vld1.32  {d16-d17}, [%[din0_ptr]]!     @ load din r0, q8\n"
                        "vld1.32  {d18-d19}, [%[din1_ptr]]!     @ load din r1, q9\n"
                        "vld1.32  {d20-d21}, [%[din2_ptr]]      @ load din r2, q10\n"

                        "vbif q8, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vbif q9, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vbif q10, q15, %q[mask]                @ bit select, deal with right pad\n"

                        "vmul.f32 q4, q8, %e[wr0][1]            @ mul weight 00, outr0\n"

                        "vext.32  q6, q8, q15, #1               @ shift left r0\n"
                        "vmla.f32 q4, q9, %e[wr1][1]           @ mul weight 10, outr0\n"
                        "vmul.f32 q5, q9, %e[wr0][1]           @ mul weight 00, outr1\n"

                        "vext.32  q13, q9, q15, #1               @ shift left r1\n"
                        "vmla.f32 q4, q10, %e[wr2][1]           @ mul weight 20, outr0\n"
                        "vmla.f32 q5, q10, %e[wr1][1]           @ mul weight 10, outr1\n"

                        "vext.32  q14, q10, q15, #1              @ shift left r2\n"
                        "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 01, outr0\n"

                        "vext.32  q6, q15, q8, #3               @ shift right r0\n"
                        "vmla.f32 q5, q13, %f[wr0][0]            @ mul weight 01, outr1\n"
                        "vmla.f32 q4, q13, %f[wr1][0]            @ mul weight 11, outr0\n"

                        "vext.32  q13, q15, q9, #3               @ shift right r1\n"
                        "vmla.f32 q5, q14,  %f[wr1][0]           @ mul weight 11, outr1\n"
                        "vmla.f32 q4, q14,  %f[wr2][0]           @ mul weight 21, outr0\n"

                        "vext.32  q14, q15, q10, #3              @ shift right r2\n"
                        "vmla.f32 q4, q6, %e[wr0][0]            @ mul weight 02, outr0\n"
                        
                        "pld [%[dout_ptr1]]                        @ preload data\n"
                        "vmla.f32 q4, q13, %e[wr1][0]            @ mul weight 12, outr0\n"
                        "vmla.f32 q5, q13, %e[wr0][0]            @ mul weight 02, outr1\n"
                        
                        "pld [%[dout_ptr2]]                        @ preload data\n"
                        "vmla.f32 q4, q14,  %e[wr2][0]           @ mul weight 22, outr0\n"
                        "vmla.f32 q5, q14,  %e[wr1][0]           @ mul weight 12, outr1\n"

                        "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                        "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"

                        "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                        "vmax.f32 q8, q15                       @ relu \n"
                        "vmax.f32 q9, q15                       @ relu \n"

                        "vbif q8, q10, %q[mask]                 @ bit select\n"
                        "vbif q9, q11, %q[mask]                 @ bit select\n"

                        "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                        "beq    1f                              @ jump to next block\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                        "1:                                     @\n"

                :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [pad_right] "+r"(right_pad_sub), \
                            [bot_pad] "+r"(size_pad_bottom)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                        [bias] "w"(wbias), [mask] "w"(vmask_rp)
                //, [test] "r"(data_test_ptr)
                :"q4", "q5", "q6", "q8", "q9", \
                        "q10", "q11", "q12", "q13", "q14", "q15"
                );
            }
            //! end of processing bottom pad
        } // end of processing channels
    } // end of processing batchs
}
#endif

/**
 * \brief depthwise convolution kernel 3x3, stride 2, width <= 4
 */
#ifdef __aarch64__
#if 1  //w <= 7
void conv_depthwise_3x3s2p1_bias_s_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {

    int right_pad_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    int out_pad_idx[4] = {0, 1, 2, 3};
    int size_pad_bottom = h_out * 2 - h_in;

    int size_right_remain = w_in;

    int size_right_pad = w_out * 2 - w_in;

    int cnt_remain = w_out;

    uint32x4_t vmask_rp1 = vcgtq_s32(vdupq_n_s32(size_right_remain), vld1q_s32(right_pad_idx));//0 2 4 6
    uint32x4_t vmask_rp2 = vcgtq_s32(vdupq_n_s32(size_right_remain), vld1q_s32(right_pad_idx + 4));//1 3 5 7
    uint32x4_t wmask = vcgtq_s32(vdupq_n_s32(cnt_remain), vld1q_s32(out_pad_idx));//0 1 2 3
    //printf("w_in %d, remain: %d \n", w_in, size_right_remain);
   // printf("mask1: %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3]);
    //printf("mask2: %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3]);
    //printf("wmask: %d, %d, %d, %d \n", wmask[0], wmask[1], wmask[2], wmask[3]);
   // size_right_remain *= sizeof(float);

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

            float32x4_t vzero= vdupq_n_f32(0.f);

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
            float *doutr0_ptr = doutr0;

            //! top pad
            if(1){
               //printf("cnt_col: %d, remain: %d \n", cnt_col, size_right_remain);
             //  printf("mask1: %d, %d, %d, %d \n", vmask_rp1[0], vmask_rp1[1], vmask_rp1[2], vmask_rp1[3]);
             //  printf("mask2: %d, %d, %d, %d \n", vmask_rp2[0], vmask_rp2[1], vmask_rp2[2], vmask_rp2[3]);
              // printf("wmask: %d, %d, %d, %d \n", wmask[0], wmask[1], wmask[2], wmask[3]);

                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias
                "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
               // "bif  v10.16b, %[vzero].16b, %[wmask].16b    \n" //pipei
                "ext  v2.16b, %[vzero].16b, v1.16b, #12     \n" // v6 = {0, 1, 3, 5}
                "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei


                "fmul v8.4s, v0.4s, %[w1].s[1]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w1].s[2]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w1].s[0]            \n" // v2 * w00

                "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
                "ext  v6.16b, %[vzero].16b, v5.16b, #12     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v4.4s, %[w2].s[1]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w2].s[2]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w2].s[0]            \n" // v2 * w00

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
                "fmax  v0.4s, v0.4s, %[vzero].4s             \n"
                "bif  v0.16b, %[vzero].16b, %[wmask].16b    \n" //pipei

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [outptr] "+r"(doutr0_ptr), \
                  [vzero] "+w" (vzero), [w1] "+w" (wr1), [w2] "+w" (wr2), \
                  [mask1] "+w" (vmask_rp1), [mask2] "+w" (vmask_rp2), [wmask] "+w" (wmask), [vbias] "+w" (wbias)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
                );
            }

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 = doutr0 + w_out;
            //! mid
            for (int j = h_out - size_pad_bottom - 1; j > 0; j--){
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;

                doutr0_ptr = doutr0;


                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias

                "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
               // "bif  v10.16b, %[vzero].16b, %[wmask].16b    \n" //pipei
                "ext  v2.16b, %[vzero].16b, v1.16b, #12     \n" // v6 = {0,1,3,5}
                "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei

                "ld2  {v12.4s, v13.4s}, [%[inptr2]], #32    \n"
                "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
                "ext  v6.16b, %[vzero].16b, v5.16b, #12     \n" // v6 = {2,4,6,8}

                "fmul v8.4s, v0.4s, %[w0].s[1]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w0].s[2]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w0].s[0]            \n" // v2 * w00

                "bif  v12.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v13.16b, %[vzero].16b, %[mask2].16b    \n" //pipei

                "fmla v8.4s, v4.4s, %[w1].s[1]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[2]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w1].s[0]            \n" // v2 * w00

                "ext  v14.16b, %[vzero].16b, v13.16b, #12     \n" // v6 = {0,1,3,5}

                "fmla v8.4s, v12.4s, %[w2].s[1]            \n" // v0 * w01
                "fmla v9.4s, v13.4s, %[w2].s[2]            \n" // v1 * w02
                "fmla v10.4s, v14.4s, %[w2].s[0]            \n" // v2 * w00

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
                "fmax  v0.4s, v0.4s, %[vzero].4s             \n"

                "bif  v0.16b, %[vzero].16b, %[wmask].16b    \n" //pipei

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [inptr2] "+r"(din2_ptr), [outptr] "+r"(doutr0_ptr), \
                  [vzero] "+w" (vzero), [w1] "+w" (wr1), [w2] "+w" (wr2), [w0] "+w" (wr0), \
                  [mask1] "+w" (vmask_rp1), [mask2] "+w" (vmask_rp2), [wmask] "+w" (wmask), [vbias] "+w" (wbias)
                :
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14"
                );
                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
                doutr0 = doutr0 + w_out;
            }

            if (size_pad_bottom){
                din0_ptr = dr0;
                din1_ptr = dr1;

                doutr0_ptr = doutr0;

                asm volatile (
                //top
                // Load up 12 elements (3 vectors) from each of 8 sources.
                "0:                                      \n"
                "ld2  {v0.4s, v1.4s}, [%[inptr0]], #32    \n" //v0={0,2,4,6} v1={1,3,5,7}
                "ld2  {v4.4s, v5.4s}, [%[inptr1]], #32    \n"
                "and  v10.16b, %[vbias].16b, %[vbias].16b  \n" //v10 = vbias
                "bif  v0.16b, %[vzero].16b, %[mask1].16b    \n" //pipei
                "bif  v1.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
               // "bif  v10.16b, %[vzero].16b, %[wmask].16b    \n" //pipei
                "ext  v2.16b, %[vzero].16b, v1.16b, #12    \n" // v6 = {2,4,6,8}
                "bif  v4.16b, %[vzero].16b, %[mask1].16b    \n" //pipei


                "fmul v8.4s, v0.4s, %[w0].s[0]            \n" // v0 * w01
                "fmul v9.4s, v1.4s, %[w0].s[1]            \n" // v1 * w02
                "fmla v10.4s, v2.4s, %[w0].s[2]            \n" // v2 * w00

                "bif  v5.16b, %[vzero].16b, %[mask2].16b    \n" //pipei
                "ext  v6.16b, %[vzero].16b, v5.16b, #12     \n" // v6 = {2,4,6,8}

                "fmla v8.4s, v4.4s, %[w1].s[0]            \n" // v0 * w01
                "fmla v9.4s, v5.4s, %[w1].s[1]            \n" // v1 * w02
                "fmla v10.4s, v6.4s, %[w1].s[2]            \n" // v2 * w00

                "fadd v0.4s, v8.4s, v9.4s                  \n"
                "fadd v0.4s, v0.4s, v10.4s                  \n"
                "fmax  v0.4s, v0.4s, %[vzero].4s             \n"
                "bif  v0.16b, %[vzero].16b, %[wmask].16b    \n" //pipei

                "st1 {v0.4s}, [%[outptr]], #16              \n"
                "4:                                          \n"
                : [inptr0] "+r"(din0_ptr), [inptr1] "+r"(din1_ptr), [outptr] "+r"(doutr0_ptr), \
                  [vzero] "+w" (vzero), [w0] "+w" (wr0), [w1] "+w" (wr1), \
                  [mask1] "+w" (vmask_rp1), [mask2] "+w" (vmask_rp2), [wmask] "+w" (wmask), [vbias] "+w" (wbias)
                : [remain] "r" (cnt_remain)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"
                );
            }

        }
    }


}
#else
void conv_depthwise_3x3s2p1_bias_s_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s2 depthwise convolution, pad 1 is done implicit

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    const int right_pad_idx[4] = {3, 2, 1, 0};
    int size_pad_right = 4 - w_in;
    int h_even = ((h_in >> 1) << 1);
    int size_pad_bottom = (h_even == h_in) ? 0 : 1;//4 --0, 5 --1
    int tile_h = h_even >> 1;

    uint32x4_t vmask_rp = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(size_pad_right));
    int right_pad_sub = size_pad_right * sizeof(float);
    uint32x4_t vmask_w = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(0));//1111
    if(w_in < 3)
        vmask_w = vsetq_lane_u32(0, vmask_w, 1);//1011

    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
    const float* ptr_zero = zero;

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
            float *doutr0_ptr = doutr0;

            //! top pad
            if(h_in == 1){
                din1_ptr = ptr_zero;
            }
            float32x4_t vzero = vdupq_n_f32(0.f);
            // todo
            float32x4_t din0_1234 = vld1q_f32(din0_ptr);
            float32x4_t din1_1234 = vld1q_f32(din1_ptr);

            if(vgetq_lane_u32(vmask_rp, 0) == 0){
                din0_1234 = vdupq_n_f32(0.f);
                din1_1234 = vdupq_n_f32(0.f);
            }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 1);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 1);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
            }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
            }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
            }

            float32x4_t sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr1), 1);
            sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr2), 1);

            float32x4_t din0_0123 = vextq_f32(vzero, din0_1234, 3);
            float32x4_t din1_0123 = vextq_f32(vzero, din1_1234, 3);
            sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr1), 0);
            sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr2), 0);

            float32x4_t din0_2340 = vextq_f32(din0_1234, vzero, 1);
            float32x4_t din1_2340 = vextq_f32(din1_1234, vzero, 1);
            sum0 = vmlaq_lane_f32(sum0, din0_2340, vget_high_f32(wr1), 0);
            sum0 = vmlaq_lane_f32(sum0, din1_2340, vget_high_f32(wr2), 0);

            sum0 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), sum0, 1);

            float32x2_t sum = vadd_f32(vget_low_f32(sum0), vget_low_f32(wbias));
            sum = vmax_f32(sum, vget_low_f32(vzero));

            float32x2_t dout0_12 = vld1_f32(doutr0_ptr);

            if(vgetq_lane_u32(vmask_w, 1) == 0){//10
                sum = vset_lane_f32(vget_lane_f32(dout0_12, 1), sum, 1);
            }

            vst1_f32(doutr0_ptr, sum);

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 = doutr0 + w_out;

            //! process mid rows
            for (int j = tile_h - 1; j > 0; j--) {
            // todo
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                doutr0_ptr = doutr0;

                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                float32x4_t din2_1234 = vld1q_f32(din2_ptr);

                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                    din0_1234 = vdupq_n_f32(0.f);
                    din1_1234 = vdupq_n_f32(0.f);
                    din2_1234 = vdupq_n_f32(0.f);
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 1);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 1);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 1);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);

                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }

                sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 1);
                sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 1);

                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);
                float32x4_t din2_0123 = vextq_f32(vzero, din2_1234, 3);
                sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);

                din0_2340 = vextq_f32(din0_1234, vzero, 1);
                din1_2340 = vextq_f32(din1_1234, vzero, 1);
                float32x4_t din2_2340 = vextq_f32(din2_1234, vzero, 1);
                sum0 = vmlaq_lane_f32(sum0, din0_2340, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_2340, vget_high_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_2340, vget_high_f32(wr2), 0);

                sum0 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), sum0, 1);

                sum = vadd_f32(vget_low_f32(sum0), vget_low_f32(wbias));

                sum = vmax_f32(sum, vget_low_f32(vzero));

                dout0_12 = vld1_f32(doutr0_ptr);

                if(vgetq_lane_u32(vmask_w, 1) == 0){//10
                    sum = vset_lane_f32(vget_lane_f32(dout0_12, 1), sum, 1);
                }

                vst1_f32(doutr0_ptr, sum);

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
                doutr0 += w_out;
            } // end of process mid rows

            // process bottom pad if needed
            if (size_pad_bottom) {
                din0_ptr = dr0;
                din1_ptr = dr1;
                doutr0_ptr = doutr0;

                // todo
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);

                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                    din0_1234 = vdupq_n_f32(0.f);
                    din1_1234 = vdupq_n_f32(0.f);
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 1);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 1);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);

                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
                }

                sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 1);

                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);
                sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);

                din0_2340 = vextq_f32(din0_1234, vzero, 1);
                din1_2340 = vextq_f32(din1_1234, vzero, 1);
                sum0 = vmlaq_lane_f32(sum0, din0_2340, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_2340, vget_high_f32(wr1), 0);

                sum0 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), sum0, 1);

                sum = vadd_f32(vget_low_f32(sum0), vget_low_f32(wbias));

                sum = vmax_f32(sum, vget_low_f32(vzero));

                dout0_12 = vld1_f32(doutr0_ptr);

                if(vgetq_lane_u32(vmask_w, 1) == 0){//10
                    sum = vset_lane_f32(vget_lane_f32(dout0_12, 1), sum, 1);
                }

                vst1_f32(doutr0_ptr, sum);

            } // end of process bottom pad
        }
    }
}
#endif
#else
void conv_depthwise_3x3s2p1_bias_s_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s2 depthwise convolution, pad 1 is done implicit

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    const int right_pad_idx[4] = {3, 2, 1, 0};
    int size_pad_right = 4 - w_in;
    int h_even = ((h_in >> 1) << 1);
    int size_pad_bottom = (h_even == h_in) ? 0 : 1;//4 --0, 5 --1
    int tile_h = h_even >> 1;

    uint32x4_t vmask_rp = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(size_pad_right));
    int right_pad_sub = size_pad_right * sizeof(float);
    uint32x4_t vmask_w = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(0));
    if(w_in < 3)
        vmask_w = vsetq_lane_u32(0, vmask_w, 1);//01

    const float zero[4] = {0.f, 0.f, 0.f, 0.f};
    const float* ptr_zero = zero;

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
            float *doutr0_ptr = doutr0;

            //! top pad
            if(h_in == 1){
                din1_ptr = ptr_zero;
            }

            asm volatile(
            // process  pad
            "pld [%[din0_ptr], #128]                @ preload data\n"
                    "pld [%[din1_ptr], #128]                @ preload data\n"

                    "vmov.u32  q15, #0                      @ zero buf\n"

                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0 1234\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1 1234\n"
                    "vbif q10, q15, %q[mask_din]            @ bit select, deal with right pad\n"
                    "vbif q12, q15, %q[mask_din]            @ bit select, deal with right pad\n"

                    "vext.f32 q6, q15, q10, #3              @ shift right 1, 0123\n"
                    "vext.f32 q7, q15, q12, #3              @ shift right 1, 0123\n"

                    "vmul.f32 q8, q10, %e[wr1][1]           @ mul weight 11, 1234\n"
                    "vmul.f32 q9, q12, %e[wr2][1]           @ mul weight 21, 1234\n"

                    "vext.f32 q10, q10, q15, #1              @ shift left 1, 2340\n"
                    "vext.f32 q12, q12, q15, #1              @ shift left 1, 2340\n"

                    "vmla.f32 q8, q6, %e[wr1][0]           @ mul weight 10, 0123\n"
                    "vmla.f32 q9, q7, %e[wr2][0]           @ mul weight 20, 0123\n"

                    "pld [%[dout_ptr0]]                @ preload data\n"
                    "vmla.f32 q8, q10, %f[wr1][0]           @ mul weight 12, 2340\n"
                    "vmla.f32 q9, q12, %f[wr2][0]           @ mul weight 22, 2340\n"

                    "vld1.32  {d20}, [%[dout_ptr0]]         @ load dout\n"
                    "vadd.f32 q6, q8, q9                    @ add \n"

                    "vmov.f32  s25, s26                     @ mov \n"
                    "vadd.f32  d12, d12, %e[bias]           @ add bias \n"

                    "vmax.f32 d12, d30                       @max \n"

                    //"vst1.32  {d17},   [%[tmp_ptr]]! \n"
                    "vbif d12, d20, %e[mask_w]              @ bit select\n"

                    //"vst1.32  {d17},   [%[tmp_ptr]] \n"

                    "vst1.32  {d12}, [%[dout_ptr0]]!        @ store result, add pointer\n"

            :[dout_ptr0] "+r"(doutr0_ptr), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr)
            :[wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask_din] "w" (vmask_rp), [mask_w]  "w" (vmask_w) //, \
                    //[pad_right] "r" (size_right_remain)
            :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
            );

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 = doutr0 + w_out;

            //! process mid rows
            for (int j = tile_h - 1; j > 0; j--) {

                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                doutr0_ptr = doutr0;
                asm volatile(
                // process  pad
                "pld [%[din0_ptr], #128]                @ preload data\n"
                        "pld [%[din1_ptr], #128]                @ preload data\n"
                        "pld [%[din2_ptr], #128]                @ preload data\n"

                        "vmov.u32  q15, #0                      @ zero buf\n"

                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0 1234\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1 1234\n"
                        "vld1.32  {d28-d29}, [%[din2_ptr]]!     @ load din r1 1234\n"

                        "vbif q10, q15, %q[mask_din]            @ bit select, deal with right pad\n"
                        "vbif q12, q15, %q[mask_din]            @ bit select, deal with right pad\n"
                        "vbif q14, q15, %q[mask_din]            @ bit select, deal with right pad\n"


                        "vext.f32 q6, q15, q10, #3              @ shift right 1, 0123\n"
                        "vext.f32 q7, q15, q12, #3              @ shift right 1, 0123\n"
                        "vext.f32 q8, q15, q14, #3              @ shift right 1, 0123\n"

                        "vmul.f32 q9, q10, %e[wr0][1]           @ mul weight 01, 1234\n"
                        "vmul.f32 q11, q12, %e[wr1][1]           @ mul weight 11, 1234\n"
                        "vmul.f32 q13, q14, %e[wr2][1]           @ mul weight 21, 1234\n"

                        "vext.f32 q10, q10, q15, #1              @ shift left 1, 2340\n"
                        "vext.f32 q12, q12, q15, #1              @ shift left 1, 2340\n"
                        "vext.f32 q14, q14, q15, #1              @ shift left 1, 2340\n"

                        "vmla.f32 q9, q6, %e[wr0][0]           @ mul weight 00, 0123\n"
                        "vmla.f32 q11, q7, %e[wr1][0]           @ mul weight 10, 0123\n"
                        "vmla.f32 q13, q8, %e[wr2][0]           @ mul weight 20, 0123\n"

                        "pld [%[dout_ptr0]]                     @ preload data\n"
                        "vmla.f32 q9, q10, %f[wr0][0]           @ mul weight 12, 2340\n"
                        "vmla.f32 q11, q12, %f[wr1][0]           @ mul weight 22, 2340\n"
                        "vmla.f32 q13, q14, %f[wr2][0]           @ mul weight 22, 2340\n"

                        "vld1.32  {d20}, [%[dout_ptr0]]         @ load dout\n"
                        "vadd.f32  q8, q9, q11                   @add\n"
                        "vadd.f32  q6, q8, q13                   @add\n"

                        "vmov.f32  s25, s26                     @ mov \n"
                        "vadd.f32  d12, d12, %e[bias]           @ add bias \n"

                        "vmax.f32 d12, d30                            @max \n"

                        //"vst1.32  {d17},   [%[tmp_ptr]]! \n"
                        "vbif d12, d20, %e[mask_w]              @ bit select\n"

                        //"vst1.32  {d17},   [%[tmp_ptr]] \n"

                        "vst1.32  {d12}, [%[dout_ptr0]]!        @ store result, add pointer\n"

                :[dout_ptr0] "+r"(doutr0_ptr), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr), [din2_ptr] "+r"(din2_ptr)
                :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask_din] "w" (vmask_rp), [mask_w] "w" (vmask_w) //, \
                    //[pad_right] "r" (size_right_remain)
                :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
                doutr0 += w_out;
            } // end of process mid rows

            // process bottom pad if needed
            if (size_pad_bottom) {
                din0_ptr = dr0;
                din1_ptr = dr1;

                asm volatile(
                // process  pad
                "pld [%[din0_ptr], #128]                @ preload data\n"
                        "pld [%[din1_ptr], #128]                @ preload data\n"

                        "vmov.u32  q15, #0                      @ zero buf\n"

                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0 1234\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1 1234\n"
                        "vbif q10, q15, %q[mask_din]            @ bit select, deal with right pad\n"
                        "vbif q12, q15, %q[mask_din]            @ bit select, deal with right pad\n"

                        "vext.f32 q6, q15, q10, #3              @ shift right 1, 0123\n"
                        "vext.f32 q7, q15, q12, #3              @ shift right 1, 0123\n"

                        "vmul.f32 q8, q10, %e[wr1][1]           @ mul weight 11, 1234\n"
                        "vmul.f32 q9, q12, %e[wr2][1]           @ mul weight 21, 1234\n"

                        "vext.f32 q10, q10, q15, #1              @ shift left 1, 2340\n"
                        "vext.f32 q12, q12, q15, #1              @ shift left 1, 2340\n"

                        "vmla.f32 q8, q6, %e[wr1][0]           @ mul weight 10, 0123\n"
                        "vmla.f32 q9, q7, %e[wr2][0]           @ mul weight 20, 0123\n"

                        "pld [%[dout_ptr0], #128]                @ preload data\n"
                        "vmla.f32 q8, q10, %f[wr1][0]           @ mul weight 12, 2340\n"
                        "vmla.f32 q9, q12, %f[wr2][0]           @ mul weight 22, 2340\n"

                        "vld1.32  {d20}, [%[dout_ptr0]]         @ load dout\n"
                        "vadd.f32 q6, q8, q9                    @ add \n"

                        "vmov.f32  s25, s26                     @ mov \n"
                        "vadd.f32  d12, d12, %e[bias]           @ add bias \n"

                        "vmax.f32 d12, d30                      @ max \n"

                        //"vst1.32  {d17},   [%[tmp_ptr]]! \n"
                        "vbif d12, d20, %e[mask_w]              @ bit select\n"

                        //"vst1.32  {d17},   [%[tmp_ptr]] \n"

                        "vst1.32  {d12}, [%[dout_ptr0]]!        @ store result, add pointer\n"

                :[dout_ptr0] "+r"(doutr0_ptr), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr)
                :[wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask_din] "w" (vmask_rp), [mask_w]  "w" (vmask_w) //, \
                    //[pad_right] "r" (size_right_remain)
                :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );

            } // end of process bottom pad
        }
    }
}
#endif
} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE
