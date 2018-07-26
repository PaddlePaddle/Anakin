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
            if(w_in > 4){
                conv_depthwise_3x3s2p1_bias_relu(dout, din, weights, bias, flag_bias, \
                num, ch_in, h_in, w_in, h_out, w_out);
            }else{
                conv_depthwise_3x3s2p1_bias_s_relu(dout, din, weights, bias, flag_bias, \
                num, ch_in, h_in, w_in, h_out, w_out);
            }
        } else {
            if(w_in > 4){
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
void conv_depthwise_3x3s1p1_bias(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
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

    uint32x4_t vmask_rp = vcgeq_u32(vld1q_u32(right_pad_idx), vdupq_n_u32(size_pad_right));
    int right_pad_sub = (size_pad_right - 1) * sizeof(float);

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
#ifdef __aarch64__
            // todo
            //printf("din0: %x, din1: %x, din2: %x \n", din0_ptr, din1_ptr, din2_ptr);
            float *doutr0_ptr = doutr0;
            float *doutr1_ptr = doutr1;

         //   printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
            //left
            float32x4_t vzero = vdupq_n_f32(0.f);
            
            float32x4_t din0 = vld1q_f32(din0_ptr);
            float32x4_t din1 = vld1q_f32(din1_ptr);
            float32x4_t din2 = vld1q_f32(din2_ptr);
            //1234
            //0: 1234
            float32x4_t sum1 = vmulq_lane_f32(din0, vget_low_f32(wr1), 1);
            //1: 1234
            float32x4_t sum0 = vmulq_lane_f32(din0, vget_low_f32(wr0), 1);

            float32x4_t din0_1 =  vld1q_f32(din0_ptr + 4);
            float32x4_t din1_1 =  vld1q_f32(din1_ptr + 4);
            float32x4_t din2_1 =  vld1q_f32(din2_ptr + 4);
            sum0 = vmlaq_lane_f32(sum0, din1, vget_low_f32(wr1), 1);
            sum1 = vmlaq_lane_f32(sum1, din1, vget_low_f32(wr2), 1);

            float32x4_t din0_2345 = vextq_f32(din0, din0_1, 1);
            float32x4_t din1_2345 = vextq_f32(din1, din1_1, 1);
            float32x4_t din2_2345 = vextq_f32(din2, din2_1, 1);

            sum0 = vmlaq_lane_f32(sum0, din2, vget_low_f32(wr2), 1);

            //2345
            float32x4_t din0_0123 = vextq_f32(vzero, din0, 3);
            sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_high_f32(wr0), 0);
            sum1 = vmlaq_lane_f32(sum1, din0_2345, vget_high_f32(wr1), 0);

            float32x4_t din1_0123 = vextq_f32(vzero, din1, 3);
            float32x4_t din2_0123 = vextq_f32(vzero, din2, 3);
            sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_high_f32(wr1), 0);
            sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_high_f32(wr2), 0);

            sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_high_f32(wr2), 0);
 
            //0123
            sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);
            sum1 = vmlaq_lane_f32(sum1, din0_0123, vget_low_f32(wr1), 0);

            sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);
            sum1 = vmlaq_lane_f32(sum1, din1_0123, vget_low_f32(wr2), 0);

            sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);

            sum0 = vaddq_f32(sum0, wbias);
            sum1 = vaddq_f32(sum1, wbias);

            din0_ptr += 3;
            din1_ptr += 3;
            din2_ptr += 3;
            //store data
            vst1q_f32(doutr0_ptr, sum1);
            vst1q_f32(doutr1_ptr, sum0);

           // printf("vdata0: %f, %f, %f, %f \n", vgetq_lane_f32(sum1, 0), vgetq_lane_f32(sum1, 1), vgetq_lane_f32(sum1, 2), vgetq_lane_f32(sum1, 3));
            
            //printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
           
            doutr0_ptr += 4;
            doutr1_ptr += 4;

            int cnt = cnt_col;

            //mid
          //  printf("din0: %x, din1: %x, din2: %x \n", din0_ptr, din1_ptr, din2_ptr);
           // printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
            for(; cnt > 0; cnt--){
              //  printf("cnt: %d \n", cnt);
               // printf("din0: %x, din1: %x, din2: %x \n", din0_ptr, din1_ptr, din2_ptr);
               // printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);

                din0 = vld1q_f32(din0_ptr);
                din1 = vld1q_f32(din1_ptr);
                din2 = vld1q_f32(din2_ptr);


                //float tmp1[4];
                //vst1q_f32(tmp1, din0);
                //printf("din0: %.2f, %.2f, %.2f, %.2f\n", tmp1[0], tmp1[1], tmp1[2], tmp1[3]);
                //1234
                //0: 1234
                sum1 = vmulq_lane_f32(din0, vget_low_f32(wr1), 0);
                 //1: 1234
                sum0 = vmulq_lane_f32(din0, vget_low_f32(wr0), 0);


                din0_1 =  vld1q_f32(din0_ptr + 4);
                din1_1 =  vld1q_f32(din1_ptr + 4);
                din2_1 =  vld1q_f32(din2_ptr + 4);

                sum0 = vmlaq_lane_f32(sum0, din1, vget_low_f32(wr1), 0);
                sum1 = vmlaq_lane_f32(sum1, din1, vget_low_f32(wr2), 0);

                sum0 = vmlaq_lane_f32(sum0, din2, vget_low_f32(wr2), 0);

                //2345
                din0_2345 = vextq_f32(din0, din0_1, 1);
                din1_2345 = vextq_f32(din1, din1_1, 1);
                sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_low_f32(wr0), 1);
                sum1 = vmlaq_lane_f32(sum1, din0_2345, vget_low_f32(wr1), 1);

                din2_2345 = vextq_f32(din2, din2_1, 1);
                sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_low_f32(wr1), 1);
                sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_low_f32(wr2), 1);

                float32x4_t din0_3456 = vextq_f32(din0, din0_1, 2);
                float32x4_t din1_3456 = vextq_f32(din1, din1_1, 2);
                sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_low_f32(wr2), 1);

                 //3456
                float32x4_t din2_3456 = vextq_f32(din2, din2_1, 2);
                sum0 = vmlaq_lane_f32(sum0, din0_3456, vget_high_f32(wr0), 0);
                sum1 = vmlaq_lane_f32(sum1, din0_3456, vget_high_f32(wr1), 0);

                sum0 = vmlaq_lane_f32(sum0, din1_3456, vget_high_f32(wr1), 0);
                sum1 = vmlaq_lane_f32(sum1, din1_3456, vget_high_f32(wr2), 0);

                sum0 = vmlaq_lane_f32(sum0, din2_3456, vget_high_f32(wr2), 0);

                sum0 = vaddq_f32(sum0, wbias);
                sum1 = vaddq_f32(sum1, wbias);


                din0_ptr += 4;
                din1_ptr += 4;
                din2_ptr += 4;
                //float tmp1[4];
                //vst1q_f32(tmp1, sum0);
                //printf("sum0: %.2f, %.2f, %.2f, %.2f\n", tmp1[0], tmp1[1], tmp1[2], tmp1[3]);
                //store data
                vst1q_f32(doutr0_ptr, sum1);
                vst1q_f32(doutr1_ptr, sum0);
               // printf("vdata1: %f, %f, %f, %f \n", vgetq_lane_f32(sum0, 0), vgetq_lane_f32(sum0, 1), vgetq_lane_f32(sum0, 2), vgetq_lane_f32(sum0, 3));
                //printf("vdata1: %.2f, %.2f, %.2f, %.2f\n", doutr1_ptr[0], doutr1_ptr[1], doutr1_ptr[2], doutr1_ptr[3]);


                doutr0_ptr += 4;
                doutr1_ptr += 4;
            }

            //right
            //printf("din0: %x, din1: %x, din2: %x \n", din0_ptr, din1_ptr, din2_ptr);
            //printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
            din0 = vld1q_f32(din0_ptr);
            din1 = vld1q_f32(din1_ptr);
            din2 = vld1q_f32(din2_ptr);
            //1234
          // printf("vmask_rp: %d, %d, %d, %d \n", vgetq_lane_u32(vmask_rp, 0), vgetq_lane_u32(vmask_rp, 1), vgetq_lane_u32(vmask_rp, 2), vgetq_lane_u32(vmask_rp, 3));
            //0: 1234
            /*
            float tmp[4];
            int tmp1[4];
            vst1q_u32(tmp1, vmask_rp);
            printf("vmask_rp: %d, %d, %d, %d \n", tmp1[0], tmp1[1], tmp1[2], tmp1[3]);
            vst1q_u32(tmp1, vmask_rp_r2);
            printf("vmask_rp_r2: %d, %d, %d, %d \n", tmp1[0], tmp1[1], tmp1[2], tmp1[3]);
            vst1q_f32(tmp, din0);
            printf("din0: %.2f, %.2f, %.2f, %.2f \n", tmp[0], tmp[1], tmp[2], tmp[3]);
            */
            uint32x4_t vone = vdupq_n_u32(-1);
            uint32x4_t vmask_rp_r2 = vextq_u32(vone, vmask_rp, 2);

            din0_1 =  vld1q_f32(din0_ptr + 4);
            din1_1 =  vld1q_f32(din1_ptr + 4);
            din2_1 =  vld1q_f32(din2_ptr + 4);

            din0 = vbslq_f32(vmask_rp_r2, din0, vzero);
            din1 = vbslq_f32(vmask_rp_r2, din1, vzero);
            din2 = vbslq_f32(vmask_rp_r2, din2, vzero);
           
           
            sum1 = vmulq_lane_f32(din0, vget_low_f32(wr1), 0);
            //1: 1234
            sum0 = vmulq_lane_f32(din0, vget_low_f32(wr0), 0);


            uint32x4_t vmask_rp_l2 = vextq_u32(vmask_rp, vone, 2);
            din0_1 = vbslq_f32(vmask_rp_l2, din0_1, vzero);
            din1_1 = vbslq_f32(vmask_rp_l2, din1_1, vzero);
            din2_1 = vbslq_f32(vmask_rp_l2, din2_1, vzero);

            sum0 = vmlaq_lane_f32(sum0, din1, vget_low_f32(wr1), 0);
            sum1 = vmlaq_lane_f32(sum1, din1, vget_low_f32(wr2), 0);

            
            din0_2345 = vextq_f32(din0, din0_1, 1);
            din1_2345 = vextq_f32(din1, din1_1, 1);
            din2_2345 = vextq_f32(din2, din2_1, 1);

            sum0 = vmlaq_lane_f32(sum0, din2, vget_low_f32(wr2), 0);

            //1: 2345

            float32x4_t din0_3456 = vextq_f32(din0, din0_1, 2);
            float32x4_t din1_3456 = vextq_f32(din1, din1_1, 2);
            float32x4_t din2_3456 = vextq_f32(din2, din2_1, 2);
            sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_low_f32(wr0), 1);
            sum1 = vmlaq_lane_f32(sum1, din0_2345, vget_low_f32(wr1), 1);

            sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_low_f32(wr1), 1);
            sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_low_f32(wr2), 1);

            sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_low_f32(wr2), 1);

            //3456

           // printf("din0_3456: %f, %f, %f, %f \n", vgetq_lane_f32(din0_3456, 0), vgetq_lane_f32(din0_3456, 1), vgetq_lane_f32(din0_3456, 2), vgetq_lane_f32(din0_3456, 3));

            sum0 = vmlaq_lane_f32(sum0, din0_3456, vget_high_f32(wr0), 0);
            sum1 = vmlaq_lane_f32(sum1, din0_3456, vget_high_f32(wr1), 0);

            float32x4_t vdata0 = vld1q_f32(doutr0_ptr);
            float32x4_t vdata1 = vld1q_f32(doutr1_ptr);
            sum0 = vmlaq_lane_f32(sum0, din1_3456, vget_high_f32(wr1), 0);
            sum1 = vmlaq_lane_f32(sum1, din1_3456, vget_high_f32(wr2), 0);

            uint32x4_t vmask_rp1 = vextq_u32(vone, vmask_rp, 3);//1000

            sum0 = vmlaq_lane_f32(sum0, din2_3456, vget_high_f32(wr2), 0);

            sum1 = vaddq_f32(sum1, wbias);
            sum0 = vaddq_f32(sum0, wbias);


            vdata0 = vbslq_f32(vmask_rp1, sum1, vdata0);
            vdata1 = vbslq_f32(vmask_rp1, sum0, vdata1);
            
            //store data
            vst1q_f32(doutr0_ptr, vdata0);
            vst1q_f32(doutr1_ptr, vdata1);
           // printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
            //printf("vdata0: %f, %f, %f, %f \n", vgetq_lane_f32(vdata0, 0), vgetq_lane_f32(vdata0, 1), vgetq_lane_f32(vdata0, 2), vgetq_lane_f32(vdata0, 3));
           //printf("vdata1: %f, %f, %f, %f \n", vgetq_lane_f32(vdata1, 0), vgetq_lane_f32(vdata1, 1), vgetq_lane_f32(vdata1, 2), vgetq_lane_f32(vdata1, 3));


#else

#endif //__aarch64__

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
                doutr0_ptr = doutr0;
                doutr1_ptr = doutr1;

            //printf("din0: %x, din1: %x, din2: %x \n", din0_ptr, din1_ptr, din2_ptr);
           // printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);

                float32x4_t din0_1234 = vld1q_f32(din0_ptr);
                float32x4_t din1_1234 = vld1q_f32(din1_ptr);
                float32x4_t din2_1234 = vld1q_f32(din2_ptr);
                float32x4_t din3_1234 = vld1q_f32(din3_ptr);


                //left
                sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 1);

                sum1 = vmulq_lane_f32(din1_1234, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 1);

                float32x4_t din0_5678 =  vld1q_f32(din0_ptr + 4);
                float32x4_t din1_5678 =  vld1q_f32(din1_ptr + 4);
                float32x4_t din2_5678 =  vld1q_f32(din2_ptr + 4);
                float32x4_t din3_5678 =  vld1q_f32(din3_ptr + 4);

                sum1 = vmlaq_lane_f32(sum1, din2_1234, vget_low_f32(wr1), 1);
                sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 1);

                din0_2345 = vextq_f32(din0_1234, din0_5678, 1);
                din1_2345 = vextq_f32(din1_1234, din1_5678, 1);
                din2_2345 = vextq_f32(din2_1234, din2_5678, 1);
                float32x4_t din3_2345 = vextq_f32(din3_1234, din3_5678, 1);

                sum1 = vmlaq_lane_f32(sum1, din3_1234, vget_low_f32(wr2), 1);


                sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_high_f32(wr0), 0);

                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);

                sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_high_f32(wr1), 0);

                din2_0123 = vextq_f32(vzero, din2_1234, 3);
                float32x4_t din3_0123 = vextq_f32(vzero, din3_1234, 3);

                sum1 = vmlaq_lane_f32(sum1, din2_2345, vget_high_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_high_f32(wr2), 0);

                sum1 = vmlaq_lane_f32(sum1, din3_2345, vget_high_f32(wr2), 0);


                sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_0123, vget_low_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_0123, vget_low_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);

                sum1 = vmlaq_lane_f32(sum1, din3_0123, vget_low_f32(wr2), 0);

                din0_ptr += 3;
                din1_ptr += 3;

                sum0 = vaddq_f32(sum0, wbias);
                sum1 = vaddq_f32(sum1, wbias);

                din2_ptr += 3;
                din3_ptr += 3;
                //store data
                vst1q_f32(doutr0_ptr, sum0);
                vst1q_f32(doutr1_ptr, sum1);

                doutr0_ptr += 4;
                doutr1_ptr += 4;

                cnt = cnt_col;
                for(;cnt > 0; cnt--){

                    din0_1234 = vld1q_f32(din0_ptr);
                    din1_1234 = vld1q_f32(din1_ptr);
                    din2_1234 = vld1q_f32(din2_ptr);
                    din3_1234 = vld1q_f32(din3_ptr);


                    //left
                    sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 0);

                    sum1 = vmulq_lane_f32(din1_1234, vget_low_f32(wr0), 0);
                    sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 0);

                    din0_5678 =  vld1q_f32(din0_ptr + 4);
                    din1_5678 =  vld1q_f32(din1_ptr + 4);
                    din2_5678 =  vld1q_f32(din2_ptr + 4);
                    din3_5678 =  vld1q_f32(din3_ptr + 4);

                    sum1 = vmlaq_lane_f32(sum1, din2_1234, vget_low_f32(wr1), 0);
                    sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 0);

                    din0_2345 = vextq_f32(din0_1234, din0_5678, 1);
                    din1_2345 = vextq_f32(din1_1234, din1_5678, 1);
                    sum1 = vmlaq_lane_f32(sum1, din3_1234, vget_low_f32(wr2), 0);


                    din2_2345 = vextq_f32(din2_1234, din2_5678, 1);
                    din3_2345 = vextq_f32(din3_1234, din3_5678, 1);
                    sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_low_f32(wr0), 1);

                    din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                    din1_3456 = vextq_f32(din1_1234, din1_5678, 2);
                    sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_low_f32(wr0), 1);
                    sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_low_f32(wr1), 1);

                    din2_3456 = vextq_f32(din2_1234, din2_5678, 2);
                    float32x4_t din3_3456 = vextq_f32(din3_1234, din3_5678, 2);
                    sum1 = vmlaq_lane_f32(sum1, din2_2345, vget_low_f32(wr1), 1);
                    sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_low_f32(wr2), 1);

                    sum1 = vmlaq_lane_f32(sum1, din3_2345, vget_low_f32(wr2), 1);


                    sum0 = vmlaq_lane_f32(sum0, din0_3456, vget_high_f32(wr0), 0);

                    sum1 = vmlaq_lane_f32(sum1, din1_3456, vget_high_f32(wr0), 0);
                    sum0 = vmlaq_lane_f32(sum0, din1_3456, vget_high_f32(wr1), 0);

                    sum1 = vmlaq_lane_f32(sum1, din2_3456, vget_high_f32(wr1), 0);
                    sum0 = vmlaq_lane_f32(sum0, din2_3456, vget_high_f32(wr2), 0);

                    din0_ptr += 4;
                    din1_ptr += 4;
                    sum1 = vmlaq_lane_f32(sum1, din3_3456, vget_high_f32(wr2), 0);

                    din2_ptr += 4;
                    din3_ptr += 4;
                    sum0 = vaddq_f32(sum0, wbias);
                    sum1 = vaddq_f32(sum1, wbias);

                    vst1q_f32(doutr0_ptr, sum0);
                    vst1q_f32(doutr1_ptr, sum1);

                    doutr0_ptr += 4;
                    doutr1_ptr += 4;
                }

                //right
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                din2_1234 = vld1q_f32(din2_ptr);
                din3_1234 = vld1q_f32(din3_ptr);

                //1234
                //0: 1234
                vmask_rp_r2 = vextq_u32(vone, vmask_rp, 2);
                
                din0_5678 =  vld1q_f32(din0_ptr + 4);
                din1_5678 =  vld1q_f32(din1_ptr + 4);
                din2_5678 =  vld1q_f32(din2_ptr + 4);
                din3_5678 =  vld1q_f32(din3_ptr + 4);

                din0_1234 = vbslq_f32(vmask_rp_r2, din0_1234, vzero);
                din1_1234 = vbslq_f32(vmask_rp_r2, din1_1234, vzero);
                din2_1234 = vbslq_f32(vmask_rp_r2, din2_1234, vzero);
                din3_1234 = vbslq_f32(vmask_rp_r2, din3_1234, vzero);

              
                sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 0);

                vmask_rp_l2 = vextq_u32(vmask_rp, vone, 2);

                sum1 = vmulq_lane_f32(din1_1234, vget_low_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 0);

                din0_5678 = vbslq_f32(vmask_rp_l2, din0_5678, vzero);
                din1_5678 = vbslq_f32(vmask_rp_l2, din1_5678, vzero);
                din2_5678 = vbslq_f32(vmask_rp_l2, din2_5678, vzero);
                din3_5678 = vbslq_f32(vmask_rp_l2, din3_5678, vzero);
                sum1 = vmlaq_lane_f32(sum1, din2_1234, vget_low_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 0);

                din0_2345 = vextq_f32(din0_1234, din0_5678, 1);
                din1_2345 = vextq_f32(din1_1234, din1_5678, 1);
                din2_2345 = vextq_f32(din2_1234, din2_5678, 1);
                din3_2345 = vextq_f32(din3_1234, din3_5678, 1);
                sum1 = vmlaq_lane_f32(sum1, din3_1234, vget_low_f32(wr2), 0);

                //0: 1234
            
                din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                din1_3456 = vextq_f32(din1_1234, din1_5678, 2);
                sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_low_f32(wr0), 1);

                din2_3456 = vextq_f32(din2_1234, din2_5678, 2);
                float32x4_t din3_3456 = vextq_f32(din3_1234, din3_5678, 2);
                sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_low_f32(wr1), 1);

                sum1 = vmlaq_lane_f32(sum1, din2_2345, vget_low_f32(wr1), 1);
                sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_low_f32(wr2), 1);

                sum1 = vmlaq_lane_f32(sum1, din3_2345, vget_low_f32(wr2), 1);


                sum0 = vmlaq_lane_f32(sum0, din0_3456, vget_high_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_3456, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_3456, vget_high_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_3456, vget_high_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_3456, vget_high_f32(wr2), 0);

                vmask_rp1 = vextq_u32(vone, vmask_rp, 3);//1000
                vdata0 = vld1q_f32(doutr0_ptr);
                vdata1 = vld1q_f32(doutr1_ptr);

                sum1 = vmlaq_lane_f32(sum1, din3_3456, vget_high_f32(wr2), 0);


                dr0 = dr2;
                dr1 = dr3;
                sum0 = vaddq_f32(sum0, wbias);
                sum1 = vaddq_f32(sum1, wbias);

                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;

                vdata0 = vbslq_f32(vmask_rp1, sum0, vdata0);
                vdata1 = vbslq_f32(vmask_rp1, sum1, vdata1);

                //store data
                vst1q_f32(doutr0_ptr, vdata0);
                vst1q_f32(doutr1_ptr, vdata1);

                doutr0 = doutr1 + w_out;
                doutr1 = doutr0 + w_out;
            } //! end of processing mid rows

            //! deal with bottom pad
            din0_ptr = dr0;
            din1_ptr = dr1;
            doutr0_ptr = doutr0;
            doutr1_ptr = doutr1;
            if (size_pad_bottom == 2){
                din2_ptr = ptr_zero;
            } else {
                din2_ptr = dr2;
            }
#ifdef __aarch64__
            // todo
            //left
            din0 = vld1q_f32(din0_ptr);
            din1 = vld1q_f32(din1_ptr);
            din2 = vld1q_f32(din2_ptr);

            //0: 1234
            sum0 = vmulq_lane_f32(din0, vget_low_f32(wr0), 1);
            //1: 1234
            sum1 = vmulq_lane_f32(din1, vget_low_f32(wr0), 1);


            din0_1 = vld1q_f32(din0_ptr + 4);
            din1_1 = vld1q_f32(din1_ptr + 4);
            din2_1 = vld1q_f32(din2_ptr + 4);

            sum0 = vmlaq_lane_f32(sum0, din1, vget_low_f32(wr1), 1);
            sum1 = vmlaq_lane_f32(sum1, din2, vget_low_f32(wr1), 1);

            din0_2345 = vextq_f32(din0, din0_1, 1);
            din1_2345 = vextq_f32(din1, din1_1, 1);
            sum0 = vmlaq_lane_f32(sum0, din2, vget_low_f32(wr2), 1);

            din2_2345 = vextq_f32(din2, din2_1, 1);
            din0_0123 = vextq_f32(vzero, din0, 3);
            //2345
            sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_high_f32(wr0), 0);

            din1_0123 = vextq_f32(vzero, din1, 3);
            din2_0123 = vextq_f32(vzero, din2, 3);
            sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_high_f32(wr0), 0);
            sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_high_f32(wr1), 0);

            sum1 = vmlaq_lane_f32(sum1, din2_2345, vget_high_f32(wr1), 0);
            sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_high_f32(wr2), 0);
 
            //0123
            sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);

            sum1 = vmlaq_lane_f32(sum1, din1_0123, vget_low_f32(wr0), 0);
            sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);

            sum1 = vmlaq_lane_f32(sum1, din2_0123, vget_low_f32(wr1), 0);
            sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);


            din0_ptr += 3;
            din1_ptr += 3;
            sum0 = vaddq_f32(sum0, wbias);
            sum1 = vaddq_f32(sum1, wbias);

           
            //store data
            if(size_pad_bottom != 2){
                vst1q_f32(doutr1_ptr, sum1);
                din2_ptr += 3;
                doutr1_ptr += 4;
            }
            vst1q_f32(doutr0_ptr, sum0);

            doutr0_ptr += 4;

            cnt = cnt_col;

            //mid
            for(; cnt > 0; cnt--){
                din0 = vld1q_f32(din0_ptr);
                din1 = vld1q_f32(din1_ptr);
                din2 = vld1q_f32(din2_ptr);

                //1234
                //0: 1234
                sum0 = vmulq_lane_f32(din0, vget_low_f32(wr0), 0);
                 //1: 1234
                sum1 = vmulq_lane_f32(din1, vget_low_f32(wr0), 0);

                din0_1 =  vld1q_f32(din0_ptr + 4);
                din1_1 =  vld1q_f32(din1_ptr + 4);
                din2_1 =  vld1q_f32(din2_ptr + 4);

                sum0 = vmlaq_lane_f32(sum0, din1, vget_low_f32(wr1), 0);
                sum1 = vmlaq_lane_f32(sum1, din2, vget_low_f32(wr1), 0);

                sum0 = vmlaq_lane_f32(sum0, din2, vget_low_f32(wr2), 0);

                //2345
                din0_2345 = vextq_f32(din0, din0_1, 1);
                din1_2345 = vextq_f32(din1, din1_1, 1);
                din2_2345 = vextq_f32(din2, din2_1, 1);
                sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_low_f32(wr0), 1);

                float32x4_t din0_3456 = vextq_f32(din0, din0_1, 2);
                sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_low_f32(wr1), 1);

                float32x4_t din1_3456 = vextq_f32(din1, din1_1, 2);
                float32x4_t din2_3456 = vextq_f32(din2, din2_1, 2);
                sum1 = vmlaq_lane_f32(sum1, din2_2345, vget_low_f32(wr1), 1);
                sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_low_f32(wr2), 1);

                 //3456
                sum0 = vmlaq_lane_f32(sum0, din0_3456, vget_high_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_3456, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_3456, vget_high_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_3456, vget_high_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_3456, vget_high_f32(wr2), 0);

                din0_ptr += 4;
                din1_ptr += 4;
                sum0 = vaddq_f32(sum0, wbias);
                sum1 = vaddq_f32(sum1, wbias);

                //store data
                if(size_pad_bottom != 2){
                    vst1q_f32(doutr1_ptr, sum1);
                    din2_ptr += 4;
                    doutr1_ptr += 4;
                }
                vst1q_f32(doutr0_ptr, sum0);

                doutr0_ptr += 4;
            }

            //right
            // printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
            din0 = vld1q_f32(din0_ptr);
            din1 = vld1q_f32(din1_ptr);
            din2 = vld1q_f32(din2_ptr);

            //1234
            //0: 1234

            vmask_rp_r2 = vextq_u32(vone, vmask_rp, 2);
            din0_1 =  vld1q_f32(din0_ptr + 4);
            din1_1 =  vld1q_f32(din1_ptr + 4);
            din2_1 =  vld1q_f32(din2_ptr + 4);

            din0 = vbslq_f32(vmask_rp_r2, din0, vzero);
            din1 = vbslq_f32(vmask_rp_r2, din1, vzero);
            din2 = vbslq_f32(vmask_rp_r2, din2, vzero);

           
            sum0 = vmulq_lane_f32(din0, vget_low_f32(wr0), 0);
            //1: 1234
            vmask_rp_l2 = vextq_u32(vmask_rp, vone, 2);
            sum1 = vmulq_lane_f32(din1, vget_low_f32(wr0), 0);
            sum0 = vmlaq_lane_f32(sum0, din1, vget_low_f32(wr1), 0);

            din0_1 = vbslq_f32(vmask_rp_l2, din0_1, vzero);
            din1_1 = vbslq_f32(vmask_rp_l2, din1_1, vzero);
            din2_1 = vbslq_f32(vmask_rp_l2, din2_1, vzero);
            sum1 = vmlaq_lane_f32(sum1, din2, vget_low_f32(wr1), 0);
            sum0 = vmlaq_lane_f32(sum0, din2, vget_low_f32(wr2), 0);

            //1: 2345
            din0_2345 = vextq_f32(din0, din0_1, 1);
            din1_2345 = vextq_f32(din1, din1_1, 1);
            din2_2345 = vextq_f32(din2, din2_1, 1);
            din0_3456 = vextq_f32(din0, din0_1, 2);

            sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_low_f32(wr0), 1);

            sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_low_f32(wr0), 1);
            sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_low_f32(wr1), 1);

            din1_3456 = vextq_f32(din1, din1_1, 2);
            din2_3456 = vextq_f32(din2, din2_1, 2);
            sum1 = vmlaq_lane_f32(sum1, din2_2345, vget_low_f32(wr1), 1);
            sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_low_f32(wr2), 1);

            //3456

            sum0 = vmlaq_lane_f32(sum0, din0_3456, vget_high_f32(wr0), 0);

            sum1 = vmlaq_lane_f32(sum1, din1_3456, vget_high_f32(wr0), 0);
            sum0 = vmlaq_lane_f32(sum0, din1_3456, vget_high_f32(wr1), 0);

            sum1 = vmlaq_lane_f32(sum1, din2_3456, vget_high_f32(wr1), 0);
            sum0 = vmlaq_lane_f32(sum0, din2_3456, vget_high_f32(wr2), 0);

            sum0 = vaddq_f32(sum0, wbias);
            sum1 = vaddq_f32(sum1, wbias);


            vdata0 = vld1q_f32(doutr0_ptr);
            vmask_rp1 = vextq_u32(vone, vmask_rp, 3);//1000
            vdata0 = vbslq_f32(vmask_rp1, sum0, vdata0);
            
            vst1q_f32(doutr0_ptr, vdata0);
            //store data
            if(size_pad_bottom != 2){
              //  vdata1 = vld1q_f32(doutr1_ptr);
                vst1q_f32(doutr1_ptr, sum1);
                //doutr1_ptr += 4;
            }
            //printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
           // printf("vdata0: %f, %f, %f, %f \n", vgetq_lane_f32(vdata0, 0), vgetq_lane_f32(vdata0, 1), vgetq_lane_f32(vdata0, 2), vgetq_lane_f32(vdata0, 3));
            //printf("vdata1: %f, %f, %f, %f \n", vgetq_lane_f32(vdata1, 0), vgetq_lane_f32(vdata1, 1), vgetq_lane_f32(vdata1, 2), vgetq_lane_f32(vdata1, 3));


#else

#endif
            //! end of processing bottom pad
        } // end of processing channels
    } // end of processing batchs
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
#ifdef __aarch64__
            // todo
#else
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

                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"

                    "vext.32  q9, q14, q15, #1              @ shift left r3\n"
                    "vmov.u32 d31, #0 @ zero\n"
                    "vmla.f32 q5, q8,  %f[wr1][0]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q8,  %f[wr2][0]           @ mul weight 21, outr0\n"
 
                    "vext.32  d12, d31, d20, #1             @ shift right r1\n"
                    "vext.32  d13, d20, d21, #1             @ shift right r1\n"
                    "vmla.f32 q5, q9,  %f[wr2][0]           @ mul weight 21, outr1\n"

                    "vext.32  d16, d31, d24, #1             @ shift right r0\n"
                    "vext.32  d17, d24, d25, #1             @ shift right r0\n"
                    "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"

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
                    "1:                                     @ main loop start point\n"
                   // "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
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

                    "vmla.f32 q5, q8,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q8,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "vmla.f32 q5, q9,  %f[wr2][0]           @ mul weight 22, outr1\n"

                    "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din2_ptr], #8                    @ 2 float data overlap with previous data\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "pld [%[din0_ptr]]                      @ preload data\n"
                    "pld [%[din1_ptr]]                      @ preload data\n"
                    "pld [%[din2_ptr]]                      @ preload data\n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    1b                              @ jump to main loop start point\n"

                    //! process right pad
                    "1:                                     @ right pad entry\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                   // "pld [%[din0_ptr], #192]                @ preload data\n"
                    "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din1_ptr]]!     @ load din r2\n"
                    "vld1.32  {d28-d30}, [%[din2_ptr]]!     @ load din r3\n"

                    "vbif d21, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vbif d25, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vbif d29, d31, %e[mask]                @ bit select, deal with right pad\n"

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
                        "vmla.f32 q4, q6, %e[wr0][0]  @mul weight 02, outr0\n"

                        
                        "vext.32  d12, d31, d20, #1     @ shift right r1\n"
                        "vext.32  d13, d20, d21, #1     @ shift right r1\n"
                        "vmla.f32 q5, q6, %e[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][0]  @mul weight 12, outr0\n"
                        
                        "vext.32  d12, d31, d24, #1     @ shift right r0\n"
                        "vext.32  d13, d24, d25, #1     @ shift right r0\n"
                       
                        "vmla.f32 q5, q6,  %e[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][0] @mul weight 22, outr0\n"

                        "vext.32  d12, d31, d28, #1     @ shift right r0\n"
                        "vext.32  d13, d28, d29, #1     @ shift right r0\n"
                        "sub %[din0_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "sub %[din1_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "vmla.f32 q5, q6,  %e[wr2][0] @mul weight 22, outr1\n"

                        "sub %[din2_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "sub %[din3_ptr], #12 @ 1pad + 2 float data overlap\n"
                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!  @ store result, add pointer\n"

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
                        "vmla.f32 q4, q6, %f[wr0][0]    @ mul weight 02, outr0\n"

                        "vext.32  q6, q10, q11, #2      @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]    @ mul weight 02, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]    @ mul weight 12, outr0\n"

                        "vext.32  q6, q12, q13, #2      @ shift left r2\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 22, outr0\n"

                        "vext.32  q6, q14, q15, #2  @ shift right r3\n"
                        "sub %[din0_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din1_ptr], #8 @ 2 float data overlap with previous data\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 22, outr1\n"

                        "sub %[din2_ptr], #8 @ 2 float data overlap with previous data\n"
                        "sub %[din3_ptr], #8 @ 2 float data overlap with previous data\n"
                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!  @ store result, add pointer\n"

                       
                        "subs %[cnt], #1 @ loop count minus 1\n"
                        "bne    2b                             @ jump to main loop start point\n"

                        //! process right pad
                        "3:                                    @ right pad entry\n"
                        "vmov.u32  d31, #0                     @ zero buf\n"
                        "vld1.32  {d16-d18}, [%[din0_ptr]]!    @ load din r0\n"
                        "vld1.32  {d20-d22}, [%[din1_ptr]]!    @ load din r1\n"
                        "vld1.32  {d24-d26}, [%[din2_ptr]]!    @ load din r2\n"
                        "vld1.32  {d28-d30}, [%[din3_ptr]]!    @ load din r3\n"

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
                        "vbif  d26, d31, %f[mask]               @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q12, %e[wr1][0]          @ mul weight 10, outr1\n"
                        "vmla.f32 q4, q12, %e[wr2][0]          @ mul weight 20, outr0\n"

                        "vmla.f32 q5, q14, %e[wr2][0]          @ mul weight 20, outr1\n"

                        "vmla.f32 q4, q6, %e[wr0][1]           @ mul weight 01, outr0\n"

                        "vext.32  q6, q10, q11, #1             @ shift left r1\n"
                        "vbif  d30, d31, %f[mask] @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q6, %e[wr0][1]           @ mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][1]           @ mul weight 11, outr0\n"

                        "vext.32  q6, q12, q13, #1  @ shift left r2\n"
                        "vmla.f32 q5, q6,  %e[wr1][1] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][1] @mul weight 21, outr0\n"

                        "vext.32  q6, q14, q15, #1  @ shift left r3\n"
                        "vmla.f32 q5, q6,  %e[wr2][1] @mul weight 21, outr1\n"

                        "vext.32  q6, q8, q9, #2     @ shift left r0\n"
                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 02, outr0\n"

                        "vext.32  q6, q10, q11, #2   @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]  @mul weight 12, outr0\n"

                        "vext.32  q6, q12, q13, #2  @ shift left r2\n"
                        "pld [%[dout_ptr1], #128]         @ preload data\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 22, outr0\n"

                        "vext.32  q6, q14, q15, #2  @ shift right r3\n"
                        "vld1.32  {d20-d21}, [%[dout_ptr1]]    @ load dout r0\n"
                        "vmvn.32  d22, d31 @ \n"
                        "vmvn.32  d23, d31 @ \n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 22, outr1\n"

                        "vext.32  q12, q11, %q[mask], #3                @ shift mask right 1\n"
                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"
                        
                        "vbif q8, q10, q12                              @ bit select\n"

                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!            @ store result, add pointer\n"
                        "vst1.32  {d16-d17}, [%[dout_ptr1]]!            @ store result, add pointer\n"

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
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"

                    "vmov.u32 d31, #0 @ zero\n"
                    "vext.32  q6, q12, q13, #1              @ shift left r2\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 11, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 21, outr0\n"
          
                    "vext.32  d12, d31, d16, #1             @ shift right r0\n"
                    "vext.32  d13, d16, d17, #1             @ shift right r0\n"
                    "vmla.f32 q4, q6, %e[wr0][0]            @ mul weight 02, outr0\n"

                    "vext.32  d12, d31, d20, #1             @ shift right r1\n"
                    "vext.32  d13, d20, d21, #1             @ shift right r1\n"
                    "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"

                    "vext.32  d12, d31, d24, #1             @ shift right r2\n"
                    "vext.32  d13, d24, d25, #1             @ shift right r2\n"
                    "sub %[din0_ptr], #12                   @ 1pad + 2 data overlap\n"
                    "sub %[din1_ptr], #12                   @ 1pad + 2 data overlap\n"
                    "vmla.f32 q5, q6,  %e[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %e[wr2][0]           @ mul weight 22, outr0\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                    
                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
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
                    "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 02, outr0\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"
                    
                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                    "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"

                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                  
                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                    "beq    3f                              @ jump to check point\n"
                    "add %[din2_ptr], #16                   @ point to 4 data ahead\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                    "3:                                     @ check point\n"
                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    2b                              @ jump to main loop start point\n"

                    // process right pad
                    "4:                                     @ right pad process\n"
                    "vmov.u32  d31, #0                      @ zero buf\n"
                    "vld1.32  {d16-d18}, [%[din0_ptr]]!     @ load din r0\n"
                    "vld1.32  {d20-d22}, [%[din1_ptr]]!     @ load din r1\n"
                    "vld1.32  {d24-d26}, [%[din2_ptr]]      @ load din r2\n"

                    "vbif d17, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vbif d21, d31, %e[mask]                @ bit select, deal with right pad\n"
                    "vbif d25, d31, %e[mask]                @ bit select, deal with right pad\n"

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
                    "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 02, outr0\n"

                    "vext.32  q6, q10, q11, #2              @ shift left r1\n"
                    "pld [%[dout_ptr1], #128]               @ preload data\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 02, outr1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 12, outr0\n"

                    "vext.32  q6, q12, q13, #2              @ shift left r2\n"
                    "pld [%[dout_ptr2], #128]               @ preload data\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 12, outr1\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 22, outr0\n"
                    
                    "vmvn.32  d24, d31                      @ \n"
                    "vmvn.32  d25, d31                      @ \n"
                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                    "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"

                
                    "vext.32  q13, q12, %q[mask], #3        @ shift mask right 1\n"
                    "vbif q8, q10, q13                      @ bit select\n"
                    "vbif q9, q11, q13                      @ bit select\n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"

                    "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
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
#endif //__aarch64__
            //! end of processing bottom pad
        } // end of processing channels
    } // end of processing batchs
}
#endif

/**
 * \brief depthwise convolution kernel 3x3, stride 2
 */
#ifdef __aarch64__
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
            float32x4_t vzero= vdupq_n_f32(0.f);
            // todo
            float *doutr0_ptr = doutr0;

            float32x4_t din0_1234 = vld1q_f32(din0_ptr);
            float32x4_t din1_1234 = vld1q_f32(din1_ptr);

            float32x4_t din0_0123 = vextq_f32(vzero, din0_1234, 3);
            float32x4_t din1_0123 = vextq_f32(vzero, din1_1234, 3);

            float32x4_t sum0 = vmulq_f32(din0_0123, wr1);
            sum0 = vmlaq_f32(sum0, din1_0123, wr2);

            float32x4_t din0_2340 = vextq_f32(din0_1234, vzero, 1);
            float32x4_t din1_2340 = vextq_f32(din1_1234, vzero, 1);

            float32x4_t sum1 = vmulq_f32(din0_2340, wr1);
            sum1 = vmlaq_f32(sum1, din1_2340, wr2);

            float32x2_t vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
            float32x2_t vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            float32x2_t vsum = vpadd_f32(vsum0, vsum1);

            vsum = vadd_f32(vsum, vget_low_f32(wbias));

            vst1_f32(doutr0_ptr, vsum);

            din0_ptr += 3;
            din1_ptr += 3;
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
                sum0 = vmlaq_f32(sum0, din1_1234, wr2);
                sum1 = vmulq_f32(din0_3456, wr1);
                sum1 = vmlaq_f32(sum1, din1_3456, wr2);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                vst1_f32(doutr0_ptr, vsum);

                din0_ptr += 4;
                din1_ptr += 4;
                doutr0_ptr += 2;
            }
            //right
            din0_1234 = vld1q_f32(din0_ptr);
            din1_1234 = vld1q_f32(din1_ptr);

            if(vgetq_lane_u32(mask_rp, 0) == 0){
                din0_1234 = vcombine_f32(vget_low_f32(din0_1234), vget_low_f32(vzero));
                din1_1234 = vcombine_f32(vget_low_f32(din1_1234), vget_low_f32(vzero));
            }else if(vgetq_lane_u32(mask_rp, 1) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
            }

            float32x4_t din0_5678 = vld1q_f32(din0_ptr + 4);
            float32x4_t din1_5678 = vld1q_f32(din1_ptr + 4);

            sum0 = vmulq_f32(din0_1234, wr1);
            sum0 = vmlaq_f32(sum0, din1_1234, wr2);

            if(vgetq_lane_u32(mask_rp, 2) == 0){
                din0_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din0_5678));
                din1_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din1_5678));
            }else if(vgetq_lane_u32(mask_rp, 3) == 0){
                din0_5678 = vsetq_lane_f32(0.f, din0_5678, 1);
                din1_5678 = vsetq_lane_f32(0.f, din1_5678, 1);
            }
            float32x4_t din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
            float32x4_t din1_3456 = vextq_f32(din1_1234, din1_5678, 2);

            sum1 = vmulq_f32(din0_3456, wr1);
            sum1 = vmlaq_f32(sum1, din1_3456, wr2);

            vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
            vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            vsum = vpadd_f32(vsum0, vsum1);

            float32x2_t vdata = vld1_f32(doutr0_ptr);

            vsum = vadd_f32(vsum, vget_low_f32(wbias));

            if(vgetq_lane_u32(mask_w, 0) == 0){//00
                //null
            }else if(vgetq_lane_u32(mask_w, 1) == 0){//10
                vsum = vset_lane_f32(vget_lane_f32(vdata, 1), vsum, 1);
            }

            vst1_f32(doutr0_ptr, vsum);
#else
            
#endif //__aarch64__

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 += w_out;

            //! process mid rows
            for (int j = h_out - size_pad_bottom - 1; j > 0; j--) {
#ifdef __aarch64__
                // todo
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                doutr0_ptr = doutr0;

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
                sum0 =vmlaq_f32(sum0, din1_0123, wr1);
                sum0 =vmlaq_f32(sum0, din2_0123, wr2);

                sum1 = vmulq_f32(din0_2345, wr0);
                sum1 =vmlaq_f32(sum1, din1_2345, wr1);
                sum1 =vmlaq_f32(sum1, din2_2345, wr2);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                vst1_f32(doutr0_ptr, vsum);


                din0_ptr += 3;
                din1_ptr += 3;
                din2_ptr += 3;
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
                    sum0 = vmlaq_f32(sum0, din1_1234, wr1);
                    sum0 = vmlaq_f32(sum0, din2_1234, wr2);

                    sum1 = vmulq_f32(din0_3456, wr0);
                    sum1 = vmlaq_f32(sum1, din1_3456, wr1);
                    sum1 = vmlaq_f32(sum1, din2_3456, wr2);

                    vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                    vsum = vpadd_f32(vsum0, vsum1);

                    vsum = vadd_f32(vsum, vget_low_f32(wbias));

                    vst1_f32(doutr0_ptr, vsum);

                    din0_ptr += 4;
                    din1_ptr += 4;
                    din2_ptr += 4;
                    doutr0_ptr += 2;
                }
            //right
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                din2_1234 = vld1q_f32(din2_ptr);

                if(vgetq_lane_u32(mask_rp, 0) == 0){
                    din0_1234 = vcombine_f32(vget_low_f32(din0_1234), vget_low_f32(vzero));
                    din1_1234 = vcombine_f32(vget_low_f32(din1_1234), vget_low_f32(vzero));
                    din2_1234 = vcombine_f32(vget_low_f32(din2_1234), vget_low_f32(vzero));
                }else if(vgetq_lane_u32(mask_rp, 1) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }
                din0_5678 = vld1q_f32(din0_ptr + 4);
                din1_5678 = vld1q_f32(din1_ptr + 4);
                float32x4_t din2_5678 = vld1q_f32(din2_ptr + 4);

                sum0 = vmulq_f32(din0_1234, wr0);
                sum0 = vmlaq_f32(sum0, din1_1234, wr1);
                sum0 = vmlaq_f32(sum0, din2_1234, wr2);

                if(vgetq_lane_u32(mask_rp, 2) == 0){
                    din0_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din0_5678));
                    din1_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din1_5678));
                    din2_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din2_5678));
                }else if(vgetq_lane_u32(mask_rp, 3) == 0){
                    din0_5678 = vsetq_lane_f32(0.f, din0_5678, 1);
                    din1_5678 = vsetq_lane_f32(0.f, din1_5678, 1);
                    din2_5678 = vsetq_lane_f32(0.f, din2_5678, 1);
                }
                din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                din1_3456 = vextq_f32(din1_1234, din1_5678, 2);
                float32x4_t din2_3456 = vextq_f32(din2_1234, din2_5678, 2);

                sum1 = vmulq_f32(din0_3456, wr0);
                sum1 = vmlaq_f32(sum1, din1_3456, wr1);
                sum1 = vmlaq_f32(sum1, din2_3456, wr2);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                float32x2_t vdata = vld1_f32(doutr0_ptr);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                if(vgetq_lane_u32(mask_w, 0) == 0){//00
                    //null
                }else if(vgetq_lane_u32(mask_w, 1) == 0){//10
                 vsum = vset_lane_f32(vget_lane_f32(vdata, 1), vsum, 1);
                }

                vst1_f32(doutr0_ptr, vsum);

#else

#endif //__aarch64__

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
#ifdef __aarch64__
                // todo
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);


                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);

                float32x4_t din0_2345 = vextq_f32(din0_1234, vzero, 1);
                float32x4_t din1_2345 = vextq_f32(din1_1234, vzero, 1);

                sum0 = vmulq_f32(din0_0123, wr0);
                sum0 =vmlaq_f32(sum0, din1_0123, wr1);

                sum1 = vmulq_f32(din0_2345, wr0);
                sum1 =vmlaq_f32(sum1, din1_2345, wr1);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                vst1_f32(doutr0_ptr, vsum);

                din0_ptr += 3;
                din1_ptr += 3;
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
                    sum0 = vmlaq_f32(sum0, din1_1234, wr1);

                    sum1 = vmulq_f32(din0_3456, wr0);
                    sum1 = vmlaq_f32(sum1, din1_3456, wr1);

                    vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                    vsum = vpadd_f32(vsum0, vsum1);

                    vsum = vadd_f32(vsum, vget_low_f32(wbias));

                    vst1_f32(doutr0_ptr, vsum);

                    din0_ptr += 4;
                    din1_ptr += 4;
                    doutr0_ptr += 2;
                }
            //right
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);

                if(vgetq_lane_u32(mask_rp, 0) == 0){
                    din0_1234 = vcombine_f32(vget_low_f32(din0_1234), vget_low_f32(vzero));
                    din1_1234 = vcombine_f32(vget_low_f32(din1_1234), vget_low_f32(vzero));
                }else if(vgetq_lane_u32(mask_rp, 1) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
                }
                din0_5678 = vld1q_f32(din0_ptr + 4);
                din1_5678 = vld1q_f32(din1_ptr + 4);

                sum0 = vmulq_f32(din0_1234, wr0);
                sum0 = vmlaq_f32(sum0, din1_1234, wr1);

                if(vgetq_lane_u32(mask_rp, 2) == 0){
                    din0_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din0_5678));
                    din1_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din1_5678));
                }else if(vgetq_lane_u32(mask_rp, 3) == 0){
                    din0_5678 = vsetq_lane_f32(0.f, din0_5678, 1);
                    din1_5678 = vsetq_lane_f32(0.f, din1_5678, 1);
                }
                din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                din1_3456 = vextq_f32(din1_1234, din1_5678, 2);

                sum1 = vmulq_f32(din0_3456, wr0);
                sum1 = vmlaq_f32(sum1, din1_3456, wr1);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                float32x2_t vdata = vld1_f32(doutr0_ptr);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));

                if(vgetq_lane_u32(mask_w, 0) == 0){//00
                    //null
                }else if(vgetq_lane_u32(mask_w, 1) == 0){//10
                 vsum = vset_lane_f32(vget_lane_f32(vdata, 1), vsum, 1);
                }

                vst1_f32(doutr0_ptr, vsum);

#else

#endif //__aarch64__
            } // end of process bottom pad

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
                    "cmp %[cnt], #1                         @ check whether has mid loop\n"
                    "blt  2f                                @ jump to rightpad\n"
                    "1:                                     @ main loop start point\n"

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

                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    1b                              @ jump to main loop start point\n"

                    //! process right pad
                    "2:                                     @ right pad entry\n"
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
                        "blt  1f                                @ jump to rightpad\n"
                        "1:                                     @ main loop start point\n"
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
                        "bne    1b                          @ jump to main loop start point\n"

                        // process right pad
                        "1:                                 @ right pad entry\n"
                        "vmov.u32  d31, #0                  @ zero buf\n"
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
                        "blt  2f                                @ jump to rightpad\n"
                        "1:                                     @ main loop start point\n"
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
                        "bne    1b @ jump to main loop start point\n"

                        // process right pad
                        "2:             @ right pad entry\n"
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
#endif

#ifdef __aarch64__
void conv_depthwise_3x3s1p1_bias_relu(float* dout, const float* din, \
    const float* weights, const float* bias, bool flag_bias, \
    const int num, const int ch_in, const int h_in, const int w_in, \
    const int h_out, const int w_out) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    //printf("conv_depthwise_3x3s1p1_bias armv8 \n");
    const float zero[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
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

    uint32x4_t vmask_rp = vcgeq_u32(vld1q_u32(right_pad_idx), vdupq_n_u32(size_pad_right));
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
#ifdef __aarch64__
            // todo
            //printf("din0: %x, din1: %x, din2: %x \n", din0_ptr, din1_ptr, din2_ptr);
            float *doutr0_ptr = doutr0;
            float *doutr1_ptr = doutr1;

         //   printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
            //left
            float32x4_t vzero = vdupq_n_f32(0.f);
            
            float32x4_t din0 = vld1q_f32(din0_ptr);
            float32x4_t din1 = vld1q_f32(din1_ptr);
            float32x4_t din2 = vld1q_f32(din2_ptr);
            float32x4_t din0_1 =  vld1q_f32(din0_ptr + 4);
            float32x4_t din1_1 =  vld1q_f32(din1_ptr + 4);
            float32x4_t din2_1 =  vld1q_f32(din2_ptr + 4);
            //1234
            //0: 1234
            float32x4_t sum1 = vmulq_lane_f32(din0, vget_low_f32(wr1), 1);
            //1: 1234
            float32x4_t sum0 = vmulq_lane_f32(din0, vget_low_f32(wr0), 1);

            sum0 = vmlaq_lane_f32(sum0, din1, vget_low_f32(wr1), 1);
            sum1 = vmlaq_lane_f32(sum1, din1, vget_low_f32(wr2), 1);

            sum0 = vmlaq_lane_f32(sum0, din2, vget_low_f32(wr2), 1);

            //2345
            float32x4_t din0_2345 = vextq_f32(din0, din0_1, 1);
            sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_high_f32(wr0), 0);
            sum1 = vmlaq_lane_f32(sum1, din0_2345, vget_high_f32(wr1), 0);

            float32x4_t din1_2345 = vextq_f32(din1, din1_1, 1);
            sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_high_f32(wr1), 0);
            sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_high_f32(wr2), 0);

            float32x4_t din2_2345 = vextq_f32(din2, din2_1, 1);
            sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_high_f32(wr2), 0);
 
            //0123
            float32x4_t din0_0123 = vextq_f32(vzero, din0, 3);
            sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);
            sum1 = vmlaq_lane_f32(sum1, din0_0123, vget_low_f32(wr1), 0);

            float32x4_t din1_0123 = vextq_f32(vzero, din1, 3);
            sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);
            sum1 = vmlaq_lane_f32(sum1, din1_0123, vget_low_f32(wr2), 0);

            float32x4_t din2_0123 = vextq_f32(vzero, din2, 3);
            sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);

            sum0 = vaddq_f32(sum0, wbias);
            sum1 = vaddq_f32(sum1, wbias);

            sum0 = vmaxq_f32(sum0, vzero);
            sum1 = vmaxq_f32(sum1, vzero);
            //store data
            vst1q_f32(doutr0_ptr, sum1);
            vst1q_f32(doutr1_ptr, sum0);

           // printf("vdata0: %f, %f, %f, %f \n", vgetq_lane_f32(sum1, 0), vgetq_lane_f32(sum1, 1), vgetq_lane_f32(sum1, 2), vgetq_lane_f32(sum1, 3));
            
            //printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
            din0_ptr += 3;
            din1_ptr += 3;
            din2_ptr += 3;
            doutr0_ptr += 4;
            doutr1_ptr += 4;

            int cnt = cnt_col;

            //mid
          //  printf("din0: %x, din1: %x, din2: %x \n", din0_ptr, din1_ptr, din2_ptr);
           // printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
            for(; cnt > 0; cnt--){
              //  printf("cnt: %d \n", cnt);
               // printf("din0: %x, din1: %x, din2: %x \n", din0_ptr, din1_ptr, din2_ptr);
               // printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);

                din0 = vld1q_f32(din0_ptr);
                din1 = vld1q_f32(din1_ptr);
                din2 = vld1q_f32(din2_ptr);

                din0_1 =  vld1q_f32(din0_ptr + 4);
                din1_1 =  vld1q_f32(din1_ptr + 4);
                din2_1 =  vld1q_f32(din2_ptr + 4);

                //float tmp1[4];
                //vst1q_f32(tmp1, din0);
                //printf("din0: %.2f, %.2f, %.2f, %.2f\n", tmp1[0], tmp1[1], tmp1[2], tmp1[3]);
                //1234
                //0: 1234
                sum1 = vmulq_lane_f32(din0, vget_low_f32(wr1), 0);
                 //1: 1234
                sum0 = vmulq_lane_f32(din0, vget_low_f32(wr0), 0);

                sum0 = vmlaq_lane_f32(sum0, din1, vget_low_f32(wr1), 0);
                sum1 = vmlaq_lane_f32(sum1, din1, vget_low_f32(wr2), 0);

                sum0 = vmlaq_lane_f32(sum0, din2, vget_low_f32(wr2), 0);

                //2345
                din0_2345 = vextq_f32(din0, din0_1, 1);
                sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_low_f32(wr0), 1);
                sum1 = vmlaq_lane_f32(sum1, din0_2345, vget_low_f32(wr1), 1);

                din1_2345 = vextq_f32(din1, din1_1, 1);
                sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_low_f32(wr1), 1);
                sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_low_f32(wr2), 1);

                din2_2345 = vextq_f32(din2, din2_1, 1);
                sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_low_f32(wr2), 1);

                 //3456
                float32x4_t din0_3456 = vextq_f32(din0, din0_1, 2);
                sum0 = vmlaq_lane_f32(sum0, din0_3456, vget_high_f32(wr0), 0);
                sum1 = vmlaq_lane_f32(sum1, din0_3456, vget_high_f32(wr1), 0);

                float32x4_t din1_3456 = vextq_f32(din1, din1_1, 2);
                sum0 = vmlaq_lane_f32(sum0, din1_3456, vget_high_f32(wr1), 0);
                sum1 = vmlaq_lane_f32(sum1, din1_3456, vget_high_f32(wr2), 0);

                float32x4_t din2_3456 = vextq_f32(din2, din2_1, 2);
                sum0 = vmlaq_lane_f32(sum0, din2_3456, vget_high_f32(wr2), 0);

                sum0 = vaddq_f32(sum0, wbias);
                sum1 = vaddq_f32(sum1, wbias);

                sum0 = vmaxq_f32(sum0, vzero);
                sum1 = vmaxq_f32(sum1, vzero);


                //float tmp1[4];
                //vst1q_f32(tmp1, sum0);
                //printf("sum0: %.2f, %.2f, %.2f, %.2f\n", tmp1[0], tmp1[1], tmp1[2], tmp1[3]);
                //store data
                vst1q_f32(doutr0_ptr, sum1);
                vst1q_f32(doutr1_ptr, sum0);
               // printf("vdata1: %f, %f, %f, %f \n", vgetq_lane_f32(sum0, 0), vgetq_lane_f32(sum0, 1), vgetq_lane_f32(sum0, 2), vgetq_lane_f32(sum0, 3));
                //printf("vdata1: %.2f, %.2f, %.2f, %.2f\n", doutr1_ptr[0], doutr1_ptr[1], doutr1_ptr[2], doutr1_ptr[3]);


                din0_ptr += 4;
                din1_ptr += 4;
                din2_ptr += 4;
                doutr0_ptr += 4;
                doutr1_ptr += 4;
            }

            //right
            //printf("din0: %x, din1: %x, din2: %x \n", din0_ptr, din1_ptr, din2_ptr);
            //printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
            din0 = vld1q_f32(din0_ptr);
            din1 = vld1q_f32(din1_ptr);
            din2 = vld1q_f32(din2_ptr);
            din0_1 =  vld1q_f32(din0_ptr + 4);
            din1_1 =  vld1q_f32(din1_ptr + 4);
            din2_1 =  vld1q_f32(din2_ptr + 4);
            //1234
          // printf("vmask_rp: %d, %d, %d, %d \n", vgetq_lane_u32(vmask_rp, 0), vgetq_lane_u32(vmask_rp, 1), vgetq_lane_u32(vmask_rp, 2), vgetq_lane_u32(vmask_rp, 3));
            //0: 1234
            /*
            float tmp[4];
            int tmp1[4];
            vst1q_u32(tmp1, vmask_rp);
            printf("vmask_rp: %d, %d, %d, %d \n", tmp1[0], tmp1[1], tmp1[2], tmp1[3]);
            vst1q_u32(tmp1, vmask_rp_r2);
            printf("vmask_rp_r2: %d, %d, %d, %d \n", tmp1[0], tmp1[1], tmp1[2], tmp1[3]);
            vst1q_f32(tmp, din0);
            printf("din0: %.2f, %.2f, %.2f, %.2f \n", tmp[0], tmp[1], tmp[2], tmp[3]);
            */
            uint32x4_t vone = vdupq_n_u32(-1);
            uint32x4_t vmask_rp_r2 = vextq_u32(vone, vmask_rp, 2);

            din0 = vbslq_f32(vmask_rp_r2, din0, vzero);
            din1 = vbslq_f32(vmask_rp_r2, din1, vzero);
            din2 = vbslq_f32(vmask_rp_r2, din2, vzero);
           
           /* if(vgetq_lane_u32(vmask_rp, 0) == 0){
                din0 = vsetq_lane_f32(0.f, din0, 2);
                din1 = vsetq_lane_f32(0.f, din1, 2);
                din2 = vsetq_lane_f32(0.f, din2, 2);
            }
            if(vgetq_lane_u32(vmask_rp, 1) == 0){
                din0 = vsetq_lane_f32(0.f, din0, 3);
                din1 = vsetq_lane_f32(0.f, din1, 3);
                din2 = vsetq_lane_f32(0.f, din2, 3);
            }
            */
            sum1 = vmulq_lane_f32(din0, vget_low_f32(wr1), 0);
            //1: 1234
            sum0 = vmulq_lane_f32(din0, vget_low_f32(wr0), 0);

            sum0 = vmlaq_lane_f32(sum0, din1, vget_low_f32(wr1), 0);
            sum1 = vmlaq_lane_f32(sum1, din1, vget_low_f32(wr2), 0);

            sum0 = vmlaq_lane_f32(sum0, din2, vget_low_f32(wr2), 0);

            //1: 2345
            uint32x4_t vmask_rp_l2 = vextq_u32(vmask_rp, vone, 2);
            din0_1 = vbslq_f32(vmask_rp_l2, din0_1, vzero);
            din1_1 = vbslq_f32(vmask_rp_l2, din1_1, vzero);
            din2_1 = vbslq_f32(vmask_rp_l2, din2_1, vzero);
            /*
            if(vgetq_lane_u32(vmask_rp, 2) == 0){
                din0_1 = vsetq_lane_f32(0.f, din0_1, 0);
                din1_1 = vsetq_lane_f32(0.f, din1_1, 0);
                din2_1 = vsetq_lane_f32(0.f, din2_1, 0);
            }
            if(vgetq_lane_u32(vmask_rp, 3) == 0){
                din0_1 = vsetq_lane_f32(0.f, din0_1, 1);
                din1_1 = vsetq_lane_f32(0.f, din1_1, 1);
                din2_1 = vsetq_lane_f32(0.f, din2_1, 1);
            }
            */
            din0_2345 = vextq_f32(din0, din0_1, 1);
            din1_2345 = vextq_f32(din1, din1_1, 1);
            din2_2345 = vextq_f32(din2, din2_1, 1);

            sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_low_f32(wr0), 1);
            sum1 = vmlaq_lane_f32(sum1, din0_2345, vget_low_f32(wr1), 1);

            sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_low_f32(wr1), 1);
            sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_low_f32(wr2), 1);

            sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_low_f32(wr2), 1);

            //3456
            float32x4_t din0_3456 = vextq_f32(din0, din0_1, 2);
            float32x4_t din1_3456 = vextq_f32(din1, din1_1, 2);
            float32x4_t din2_3456 = vextq_f32(din2, din2_1, 2);

           // printf("din0_3456: %f, %f, %f, %f \n", vgetq_lane_f32(din0_3456, 0), vgetq_lane_f32(din0_3456, 1), vgetq_lane_f32(din0_3456, 2), vgetq_lane_f32(din0_3456, 3));

            sum0 = vmlaq_lane_f32(sum0, din0_3456, vget_high_f32(wr0), 0);
            sum1 = vmlaq_lane_f32(sum1, din0_3456, vget_high_f32(wr1), 0);

            sum0 = vmlaq_lane_f32(sum0, din1_3456, vget_high_f32(wr1), 0);
            sum1 = vmlaq_lane_f32(sum1, din1_3456, vget_high_f32(wr2), 0);

            sum0 = vmlaq_lane_f32(sum0, din2_3456, vget_high_f32(wr2), 0);

            sum0 = vaddq_f32(sum0, wbias);
            sum1 = vaddq_f32(sum1, wbias);

            sum0 = vmaxq_f32(sum0, vzero);
            sum1 = vmaxq_f32(sum1, vzero);

            float32x4_t vdata0 = vld1q_f32(doutr0_ptr);
            float32x4_t vdata1 = vld1q_f32(doutr1_ptr);

            uint32x4_t vmask_rp1 = vextq_u32(vone, vmask_rp, 3);//1000
            vdata0 = vbslq_f32(vmask_rp1, sum1, vdata0);
            vdata1 = vbslq_f32(vmask_rp1, sum0, vdata1);
            /*
            if(vgetq_lane_u32(vmask_rp, 0) == 0){// 0000
                vdata0 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), vdata0, 0);
                vdata1 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), vdata1, 0);
            }else if(vgetq_lane_u32(vmask_rp, 1) == 0){//1000
                vdata0 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(vdata0));
                vdata1 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(vdata1));
            }else if(vgetq_lane_u32(vmask_rp, 2) == 0){//1100
                vdata0 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(vdata0));
                vdata1 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(vdata1));
                vdata0 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), vdata0, 2);
                vdata1 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), vdata1, 2);
            }else if(vgetq_lane_u32(vmask_rp, 3) == 0){//1110
                vdata0 = sum1;
                vdata1 = sum0;
            }
            */

            //store data
            vst1q_f32(doutr0_ptr, vdata0);
            vst1q_f32(doutr1_ptr, vdata1);
           // printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
            //printf("vdata0: %f, %f, %f, %f \n", vgetq_lane_f32(vdata0, 0), vgetq_lane_f32(vdata0, 1), vgetq_lane_f32(vdata0, 2), vgetq_lane_f32(vdata0, 3));
           //printf("vdata1: %f, %f, %f, %f \n", vgetq_lane_f32(vdata1, 0), vgetq_lane_f32(vdata1, 1), vgetq_lane_f32(vdata1, 2), vgetq_lane_f32(vdata1, 3));


#else

#endif //__aarch64__

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
                doutr0_ptr = doutr0;
                doutr1_ptr = doutr1;

            //printf("din0: %x, din1: %x, din2: %x \n", din0_ptr, din1_ptr, din2_ptr);
           // printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);

                float32x4_t din0_1234 = vld1q_f32(din0_ptr);
                float32x4_t din1_1234 = vld1q_f32(din1_ptr);
                float32x4_t din2_1234 = vld1q_f32(din2_ptr);
                float32x4_t din3_1234 = vld1q_f32(din3_ptr);

                float32x4_t din0_5678 =  vld1q_f32(din0_ptr + 4);
                float32x4_t din1_5678 =  vld1q_f32(din1_ptr + 4);
                float32x4_t din2_5678 =  vld1q_f32(din2_ptr + 4);
                float32x4_t din3_5678 =  vld1q_f32(din3_ptr + 4);

                //left
                sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 1);

                sum1 = vmulq_lane_f32(din1_1234, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 1);

                sum1 = vmlaq_lane_f32(sum1, din2_1234, vget_low_f32(wr1), 1);
                sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 1);

                sum1 = vmlaq_lane_f32(sum1, din3_1234, vget_low_f32(wr2), 1);

                din0_2345 = vextq_f32(din0_1234, din0_5678, 1);
                din1_2345 = vextq_f32(din1_1234, din1_5678, 1);
                din2_2345 = vextq_f32(din2_1234, din2_5678, 1);
                float32x4_t din3_2345 = vextq_f32(din3_1234, din3_5678, 1);

                sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_high_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_high_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_2345, vget_high_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_high_f32(wr2), 0);

                sum1 = vmlaq_lane_f32(sum1, din3_2345, vget_high_f32(wr2), 0);

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

                sum0 = vaddq_f32(sum0, wbias);
                sum1 = vaddq_f32(sum1, wbias);

                sum0 = vmaxq_f32(sum0, vzero);
                sum1 = vmaxq_f32(sum1, vzero);

                //store data
                vst1q_f32(doutr0_ptr, sum0);
                vst1q_f32(doutr1_ptr, sum1);

                din0_ptr += 3;
                din1_ptr += 3;
                din2_ptr += 3;
                din3_ptr += 3;
                doutr0_ptr += 4;
                doutr1_ptr += 4;

                cnt = cnt_col;
                for(;cnt > 0; cnt--){

                    din0_1234 = vld1q_f32(din0_ptr);
                    din1_1234 = vld1q_f32(din1_ptr);
                    din2_1234 = vld1q_f32(din2_ptr);
                    din3_1234 = vld1q_f32(din3_ptr);

                    din0_5678 =  vld1q_f32(din0_ptr + 4);
                    din1_5678 =  vld1q_f32(din1_ptr + 4);
                    din2_5678 =  vld1q_f32(din2_ptr + 4);
                    din3_5678 =  vld1q_f32(din3_ptr + 4);

                    //left
                    sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 0);

                    sum1 = vmulq_lane_f32(din1_1234, vget_low_f32(wr0), 0);
                    sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 0);

                    sum1 = vmlaq_lane_f32(sum1, din2_1234, vget_low_f32(wr1), 0);
                    sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 0);

                    sum1 = vmlaq_lane_f32(sum1, din3_1234, vget_low_f32(wr2), 0);

                    din0_2345 = vextq_f32(din0_1234, din0_5678, 1);
                    din1_2345 = vextq_f32(din1_1234, din1_5678, 1);
                    din2_2345 = vextq_f32(din2_1234, din2_5678, 1);
                    din3_2345 = vextq_f32(din3_1234, din3_5678, 1);

                    sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_low_f32(wr0), 1);

                    sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_low_f32(wr0), 1);
                    sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_low_f32(wr1), 1);

                    sum1 = vmlaq_lane_f32(sum1, din2_2345, vget_low_f32(wr1), 1);
                    sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_low_f32(wr2), 1);

                    sum1 = vmlaq_lane_f32(sum1, din3_2345, vget_low_f32(wr2), 1);

                    din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                    din1_3456 = vextq_f32(din1_1234, din1_5678, 2);
                    din2_3456 = vextq_f32(din2_1234, din2_5678, 2);
                    float32x4_t din3_3456 = vextq_f32(din3_1234, din3_5678, 2);

                    sum0 = vmlaq_lane_f32(sum0, din0_3456, vget_high_f32(wr0), 0);

                    sum1 = vmlaq_lane_f32(sum1, din1_3456, vget_high_f32(wr0), 0);
                    sum0 = vmlaq_lane_f32(sum0, din1_3456, vget_high_f32(wr1), 0);

                    sum1 = vmlaq_lane_f32(sum1, din2_3456, vget_high_f32(wr1), 0);
                    sum0 = vmlaq_lane_f32(sum0, din2_3456, vget_high_f32(wr2), 0);

                    sum1 = vmlaq_lane_f32(sum1, din3_3456, vget_high_f32(wr2), 0);

                    sum0 = vaddq_f32(sum0, wbias);
                    sum1 = vaddq_f32(sum1, wbias);

                    sum0 = vmaxq_f32(sum0, vzero);
                    sum1 = vmaxq_f32(sum1, vzero);

                    vst1q_f32(doutr0_ptr, sum0);
                    vst1q_f32(doutr1_ptr, sum1);

                    din0_ptr += 4;
                    din1_ptr += 4;
                    din2_ptr += 4;
                    din3_ptr += 4;
                    doutr0_ptr += 4;
                    doutr1_ptr += 4;
                }

                //right
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                din2_1234 = vld1q_f32(din2_ptr);
                din3_1234 = vld1q_f32(din3_ptr);

                din0_5678 =  vld1q_f32(din0_ptr + 4);
                din1_5678 =  vld1q_f32(din1_ptr + 4);
                din2_5678 =  vld1q_f32(din2_ptr + 4);
                din3_5678 =  vld1q_f32(din3_ptr + 4);
                //1234
                //0: 1234
                vmask_rp_r2 = vextq_u32(vone, vmask_rp, 2);
                din0_1234 = vbslq_f32(vmask_rp_r2, din0_1234, vzero);
                din1_1234 = vbslq_f32(vmask_rp_r2, din1_1234, vzero);
                din2_1234 = vbslq_f32(vmask_rp_r2, din2_1234, vzero);
                din3_1234 = vbslq_f32(vmask_rp_r2, din3_1234, vzero);
                /*
                if(vgetq_lane_u32(vmask_rp, 0) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 2);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 2);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 2);
                    din3_1234 = vsetq_lane_f32(0.f, din3_1234, 2);
                 }
                if(vgetq_lane_u32(vmask_rp, 1) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                    din3_1234 = vsetq_lane_f32(0.f, din3_1234, 3);
                }
                */
                sum0 = vmulq_lane_f32(din0_1234, vget_low_f32(wr0), 0);

                sum1 = vmulq_lane_f32(din1_1234, vget_low_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_1234, vget_low_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_1234, vget_low_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_1234, vget_low_f32(wr2), 0);

                sum1 = vmlaq_lane_f32(sum1, din3_1234, vget_low_f32(wr2), 0);

                //0: 1234
                vmask_rp_l2 = vextq_u32(vmask_rp, vone, 2);
                din0_5678 = vbslq_f32(vmask_rp_l2, din0_5678, vzero);
                din1_5678 = vbslq_f32(vmask_rp_l2, din1_5678, vzero);
                din2_5678 = vbslq_f32(vmask_rp_l2, din2_5678, vzero);
                din3_5678 = vbslq_f32(vmask_rp_l2, din3_5678, vzero);
                /*
                if(vgetq_lane_u32(vmask_rp, 2) == 0){
                    din0_5678 = vsetq_lane_f32(0.f, din0_5678, 0);
                    din1_5678 = vsetq_lane_f32(0.f, din1_5678, 0);
                    din2_5678 = vsetq_lane_f32(0.f, din2_5678, 0);
                    din3_5678 = vsetq_lane_f32(0.f, din3_5678, 0);
                 }
                if(vgetq_lane_u32(vmask_rp, 3) == 0){
                    din0_5678 = vsetq_lane_f32(0.f, din0_5678, 1);
                    din1_5678 = vsetq_lane_f32(0.f, din1_5678, 1);
                    din2_5678 = vsetq_lane_f32(0.f, din2_5678, 1);
                    din3_5678 = vsetq_lane_f32(0.f, din3_5678, 1);
                }
                */
                din0_2345 = vextq_f32(din0_1234, din0_5678, 1);
                din1_2345 = vextq_f32(din1_1234, din1_5678, 1);
                din2_2345 = vextq_f32(din2_1234, din2_5678, 1);
                din3_2345 = vextq_f32(din3_1234, din3_5678, 1);

                sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_low_f32(wr0), 1);

                sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_low_f32(wr1), 1);

                sum1 = vmlaq_lane_f32(sum1, din2_2345, vget_low_f32(wr1), 1);
                sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_low_f32(wr2), 1);

                sum1 = vmlaq_lane_f32(sum1, din3_2345, vget_low_f32(wr2), 1);

                din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                din1_3456 = vextq_f32(din1_1234, din1_5678, 2);
                din2_3456 = vextq_f32(din2_1234, din2_5678, 2);
                float32x4_t din3_3456 = vextq_f32(din3_1234, din3_5678, 2);

                sum0 = vmlaq_lane_f32(sum0, din0_3456, vget_high_f32(wr0), 0);

                sum1 = vmlaq_lane_f32(sum1, din1_3456, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_3456, vget_high_f32(wr1), 0);

                sum1 = vmlaq_lane_f32(sum1, din2_3456, vget_high_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_3456, vget_high_f32(wr2), 0);

                sum1 = vmlaq_lane_f32(sum1, din3_3456, vget_high_f32(wr2), 0);

                sum0 = vaddq_f32(sum0, wbias);
                sum1 = vaddq_f32(sum1, wbias);

                sum0 = vmaxq_f32(sum0, vzero);
                sum1 = vmaxq_f32(sum1, vzero);

                vdata0 = vld1q_f32(doutr0_ptr);
                vdata1 = vld1q_f32(doutr1_ptr);

                vmask_rp1 = vextq_u32(vone, vmask_rp, 3);//1000
                vdata0 = vbslq_f32(vmask_rp1, sum0, vdata0);
                vdata1 = vbslq_f32(vmask_rp1, sum1, vdata1);

                /*
                if(vgetq_lane_u32(vmask_rp, 0) == 0){// 0000
                    vdata0 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), vdata0, 0);
                    vdata1 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), vdata1, 0);
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){//1000
                    vdata0 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(vdata0));
                    vdata1 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(vdata1));
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){//1100
                    vdata0 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(vdata0));
                    vdata1 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(vdata1));
                    vdata0 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), vdata0, 2);
                    vdata1 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), vdata1, 2);
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){//1110
                    vdata0 = sum0;
                    vdata1 = sum1;
                }
                */
                //store data
                vst1q_f32(doutr0_ptr, vdata0);
                vst1q_f32(doutr1_ptr, vdata1);

                doutr0 = doutr1 + w_out;
                doutr1 = doutr0 + w_out;
                dr0 = dr2;
                dr1 = dr3;
                dr2 = dr1 + w_in;
                dr3 = dr2 + w_in;
            } //! end of processing mid rows

            //! deal with bottom pad
            din0_ptr = dr0;
            din1_ptr = dr1;
            doutr0_ptr = doutr0;
            doutr1_ptr = doutr1;
            if (size_pad_bottom == 2){
                din2_ptr = ptr_zero;
            } else {
                din2_ptr = dr2;
            }
#ifdef __aarch64__
            // todo
            //left
            din0 = vld1q_f32(din0_ptr);
            din1 = vld1q_f32(din1_ptr);
            din2 = vld1q_f32(din2_ptr);

            din0_1 = vld1q_f32(din0_ptr + 4);
            din1_1 = vld1q_f32(din1_ptr + 4);
            din2_1 = vld1q_f32(din2_ptr + 4);

            //0: 1234
            sum0 = vmulq_lane_f32(din0, vget_low_f32(wr0), 1);
            //1: 1234
            sum1 = vmulq_lane_f32(din1, vget_low_f32(wr0), 1);

            sum0 = vmlaq_lane_f32(sum0, din1, vget_low_f32(wr1), 1);
            sum1 = vmlaq_lane_f32(sum1, din2, vget_low_f32(wr1), 1);

            sum0 = vmlaq_lane_f32(sum0, din2, vget_low_f32(wr2), 1);

            //2345
            din0_2345 = vextq_f32(din0, din0_1, 1);
            sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_high_f32(wr0), 0);

            din1_2345 = vextq_f32(din1, din1_1, 1);
            sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_high_f32(wr0), 0);
            sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_high_f32(wr1), 0);

            din2_2345 = vextq_f32(din2, din2_1, 1);
            sum1 = vmlaq_lane_f32(sum1, din2_2345, vget_high_f32(wr1), 0);
            sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_high_f32(wr2), 0);
 
            //0123
            din0_0123 = vextq_f32(vzero, din0, 3);
            sum0 = vmlaq_lane_f32(sum0, din0_0123, vget_low_f32(wr0), 0);

            din1_0123 = vextq_f32(vzero, din1, 3);
            sum1 = vmlaq_lane_f32(sum1, din1_0123, vget_low_f32(wr0), 0);
            sum0 = vmlaq_lane_f32(sum0, din1_0123, vget_low_f32(wr1), 0);

            din2_0123 = vextq_f32(vzero, din2, 3);
            sum1 = vmlaq_lane_f32(sum1, din2_0123, vget_low_f32(wr1), 0);
            sum0 = vmlaq_lane_f32(sum0, din2_0123, vget_low_f32(wr2), 0);


            sum0 = vaddq_f32(sum0, wbias);
            sum1 = vaddq_f32(sum1, wbias);

            sum0 = vmaxq_f32(sum0, vzero);
            sum1 = vmaxq_f32(sum1, vzero);

            // printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
           
            //store data
            if(size_pad_bottom != 2){
                vst1q_f32(doutr1_ptr, sum1);
                din2_ptr += 3;
                doutr1_ptr += 4;
            }
            vst1q_f32(doutr0_ptr, sum0);

            din0_ptr += 3;
            din1_ptr += 3;
            doutr0_ptr += 4;

            cnt = cnt_col;

            //mid
            for(; cnt > 0; cnt--){
                din0 = vld1q_f32(din0_ptr);
                din1 = vld1q_f32(din1_ptr);
                din2 = vld1q_f32(din2_ptr);

                din0_1 =  vld1q_f32(din0_ptr + 4);
                din1_1 =  vld1q_f32(din1_ptr + 4);
                din2_1 =  vld1q_f32(din2_ptr + 4);
                //1234
                //0: 1234
                sum0 = vmulq_lane_f32(din0, vget_low_f32(wr0), 0);
                 //1: 1234
                sum1 = vmulq_lane_f32(din1, vget_low_f32(wr0), 0);

                sum0 = vmlaq_lane_f32(sum0, din1, vget_low_f32(wr1), 0);
                sum1 = vmlaq_lane_f32(sum1, din2, vget_low_f32(wr1), 0);

                sum0 = vmlaq_lane_f32(sum0, din2, vget_low_f32(wr2), 0);

                //2345
                din0_2345 = vextq_f32(din0, din0_1, 1);
                sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_low_f32(wr0), 1);

                din1_2345 = vextq_f32(din1, din1_1, 1);
                sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_low_f32(wr0), 1);
                sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_low_f32(wr1), 1);

                din2_2345 = vextq_f32(din2, din2_1, 1);
                sum1 = vmlaq_lane_f32(sum1, din2_2345, vget_low_f32(wr1), 1);
                sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_low_f32(wr2), 1);

                 //3456
                float32x4_t din0_3456 = vextq_f32(din0, din0_1, 2);
                sum0 = vmlaq_lane_f32(sum0, din0_3456, vget_high_f32(wr0), 0);

                float32x4_t din1_3456 = vextq_f32(din1, din1_1, 2);
                sum1 = vmlaq_lane_f32(sum1, din1_3456, vget_high_f32(wr0), 0);
                sum0 = vmlaq_lane_f32(sum0, din1_3456, vget_high_f32(wr1), 0);

                float32x4_t din2_3456 = vextq_f32(din2, din2_1, 2);
                sum1 = vmlaq_lane_f32(sum1, din2_3456, vget_high_f32(wr1), 0);
                sum0 = vmlaq_lane_f32(sum0, din2_3456, vget_high_f32(wr2), 0);

                sum0 = vaddq_f32(sum0, wbias);
                sum1 = vaddq_f32(sum1, wbias);

                sum0 = vmaxq_f32(sum0, vzero);
                sum1 = vmaxq_f32(sum1, vzero);

                //store data
                if(size_pad_bottom != 2){
                    vst1q_f32(doutr1_ptr, sum1);
                    din2_ptr += 4;
                    doutr1_ptr += 4;
                }
                vst1q_f32(doutr0_ptr, sum0);

                din0_ptr += 4;
                din1_ptr += 4;

                doutr0_ptr += 4;
            }

            //right
            // printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
            din0 = vld1q_f32(din0_ptr);
            din1 = vld1q_f32(din1_ptr);
            din2 = vld1q_f32(din2_ptr);

            din0_1 =  vld1q_f32(din0_ptr + 4);
            din1_1 =  vld1q_f32(din1_ptr + 4);
            din2_1 =  vld1q_f32(din2_ptr + 4);
            //1234
            //0: 1234

            vmask_rp_r2 = vextq_u32(vone, vmask_rp, 2);
            din0 = vbslq_f32(vmask_rp_r2, din0, vzero);
            din1 = vbslq_f32(vmask_rp_r2, din1, vzero);
            din2 = vbslq_f32(vmask_rp_r2, din2, vzero);
           /*     
            if(vgetq_lane_u32(vmask_rp, 0) == 0){
                din0 = vsetq_lane_f32(0.f, din0, 2);
                din1 = vsetq_lane_f32(0.f, din1, 2);
                din2 = vsetq_lane_f32(0.f, din2, 2);
            }
            if(vgetq_lane_u32(vmask_rp, 1) == 0){
                din0 = vsetq_lane_f32(0.f, din0, 3);
                din1 = vsetq_lane_f32(0.f, din1, 3);
                din2 = vsetq_lane_f32(0.f, din2, 3);
            }
            */
            sum0 = vmulq_lane_f32(din0, vget_low_f32(wr0), 0);
            //1: 1234
            sum1 = vmulq_lane_f32(din1, vget_low_f32(wr0), 0);
            sum0 = vmlaq_lane_f32(sum0, din1, vget_low_f32(wr1), 0);

            sum1 = vmlaq_lane_f32(sum1, din2, vget_low_f32(wr1), 0);
            sum0 = vmlaq_lane_f32(sum0, din2, vget_low_f32(wr2), 0);

            //1: 2345
            vmask_rp_l2 = vextq_u32(vmask_rp, vone, 2);
            din0_1 = vbslq_f32(vmask_rp_l2, din0_1, vzero);
            din1_1 = vbslq_f32(vmask_rp_l2, din1_1, vzero);
            din2_1 = vbslq_f32(vmask_rp_l2, din2_1, vzero);
            /*
            if(vgetq_lane_u32(vmask_rp, 2) == 0){
                din0_1 = vsetq_lane_f32(0.f, din0_1, 0);
                din1_1 = vsetq_lane_f32(0.f, din1_1, 0);
                din2_1 = vsetq_lane_f32(0.f, din2_1, 0);
            }
            if(vgetq_lane_u32(vmask_rp, 3) == 0){
                din0_1 = vsetq_lane_f32(0.f, din0_1, 1);
                din1_1 = vsetq_lane_f32(0.f, din1_1, 1);
                din2_1 = vsetq_lane_f32(0.f, din2_1, 1);
            }
            */
            din0_2345 = vextq_f32(din0, din0_1, 1);
            din1_2345 = vextq_f32(din1, din1_1, 1);
            din2_2345 = vextq_f32(din2, din2_1, 1);

            sum0 = vmlaq_lane_f32(sum0, din0_2345, vget_low_f32(wr0), 1);

            sum1 = vmlaq_lane_f32(sum1, din1_2345, vget_low_f32(wr0), 1);
            sum0 = vmlaq_lane_f32(sum0, din1_2345, vget_low_f32(wr1), 1);

            sum1 = vmlaq_lane_f32(sum1, din2_2345, vget_low_f32(wr1), 1);
            sum0 = vmlaq_lane_f32(sum0, din2_2345, vget_low_f32(wr2), 1);

            //3456
            din0_3456 = vextq_f32(din0, din0_1, 2);
            din1_3456 = vextq_f32(din1, din1_1, 2);
            din2_3456 = vextq_f32(din2, din2_1, 2);

            sum0 = vmlaq_lane_f32(sum0, din0_3456, vget_high_f32(wr0), 0);

            sum1 = vmlaq_lane_f32(sum1, din1_3456, vget_high_f32(wr0), 0);
            sum0 = vmlaq_lane_f32(sum0, din1_3456, vget_high_f32(wr1), 0);

            sum1 = vmlaq_lane_f32(sum1, din2_3456, vget_high_f32(wr1), 0);
            sum0 = vmlaq_lane_f32(sum0, din2_3456, vget_high_f32(wr2), 0);

            sum0 = vaddq_f32(sum0, wbias);
            sum1 = vaddq_f32(sum1, wbias);

            sum0 = vmaxq_f32(sum0, vzero);
            sum1 = vmaxq_f32(sum1, vzero);


            vdata0 = vld1q_f32(doutr0_ptr);
/*
            if(vgetq_lane_u32(vmask_rp, 0) == 0){// 0000
                vdata0 = vsetq_lane_f32(vgetq_lane_f32(sum0, 0), vdata0, 0);
            }else if(vgetq_lane_u32(vmask_rp, 1) == 0){//1000
                vdata0 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(vdata0));
            }else if(vgetq_lane_u32(vmask_rp, 2) == 0){//1100
                vdata0 = vcombine_f32(vget_low_f32(sum0), vget_high_f32(vdata0));
                vdata0 = vsetq_lane_f32(vgetq_lane_f32(sum0, 2), vdata0, 2);
            }else if(vgetq_lane_u32(vmask_rp, 3) == 0){//1110
                vdata0 = sum0;
            }
            */
            vmask_rp1 = vextq_u32(vone, vmask_rp, 3);//1000
            vdata0 = vbslq_f32(vmask_rp1, sum0, vdata0);
            
            vst1q_f32(doutr0_ptr, vdata0);
            //store data
            if(size_pad_bottom != 2){
              //  vdata1 = vld1q_f32(doutr1_ptr);
                /*
                if(vgetq_lane_u32(vmask_rp, 0) == 0){// 0000
                    vdata1 = vsetq_lane_f32(vgetq_lane_f32(sum1, 0), vdata1, 0);
                }else if(vgetq_lane_u32(vmask_rp, 1) == 0){//1000
                    vdata1 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(vdata1));
                }else if(vgetq_lane_u32(vmask_rp, 2) == 0){//1100
                    vdata1 = vcombine_f32(vget_low_f32(sum1), vget_high_f32(vdata1));
                    vdata1 = vsetq_lane_f32(vgetq_lane_f32(sum1, 2), vdata1, 2);
                }else if(vgetq_lane_u32(vmask_rp, 3) == 0){//1110
                    vdata1 = sum1;
                }
                */
                vst1q_f32(doutr1_ptr, sum1);
                //doutr1_ptr += 4;
            }
            //printf("dout0: %x, dout1: %x \n", doutr0_ptr, doutr1_ptr);
           // printf("vdata0: %f, %f, %f, %f \n", vgetq_lane_f32(vdata0, 0), vgetq_lane_f32(vdata0, 1), vgetq_lane_f32(vdata0, 2), vgetq_lane_f32(vdata0, 3));
            //printf("vdata1: %f, %f, %f, %f \n", vgetq_lane_f32(vdata1, 0), vgetq_lane_f32(vdata1, 1), vgetq_lane_f32(vdata1, 2), vgetq_lane_f32(vdata1, 3));


#else

#endif
            //! end of processing bottom pad
        } // end of processing channels
    } // end of processing batchs
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
                    "blt  2f                                @ jump to right pad\n"
                    "1:                                     @ main loop start point\n"
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
                    "bne    1b                              @ jump to main loop start point\n"

                    //! process right pad
                    "2:                                     @ right pad entry\n"
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
                        "blt  4f                                @ jump to right pad\n"
                        "3:                                     @ main loop start point\n"
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
                        "bne    3b                             @ jump to main loop start point\n"

                        //! process right pad
                        "4:                                    @ right pad entry\n"
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
                    "beq    5f                              @ jump to next block\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "add %[din2_ptr], #12                   @ 1pad + 2 data overlap\n"

                    // process mid cols
                    "5:                                     @ header of bottom process\n"
                    "cmp %[cnt], #1                         @ check whether has mid cols\n"
                    "blt  8f                                @ jump to right pad\n"
                    "6:                                     @ main loop start point\n"
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
                    "beq    7f                              @ jump to check point\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "add %[din2_ptr], #16                   @ point to 4 data ahead\n"

                    "7:                                     @ check point\n"
                    "subs %[cnt], #1                        @ loop count minus 1\n"
                    "bne    6b                              @ jump to main loop start point\n"

                    // process right pad
                    "8:                                     @ right pad process\n"
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
#endif //__aarch64__
            //! end of processing bottom pad
        } // end of processing channels
    } // end of processing batchs
}
#endif
/**
 * \brief depthwise convolution kernel 3x3, stride 2, with reulu
 */
#ifdef __aarch64__
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
            float32x4_t vzero= vdupq_n_f32(0.f);
            // todo
            float *doutr0_ptr = doutr0;

            float32x4_t din0_1234 = vld1q_f32(din0_ptr);
            float32x4_t din1_1234 = vld1q_f32(din1_ptr);

            float32x4_t din0_0123 = vextq_f32(vzero, din0_1234, 3);
            float32x4_t din1_0123 = vextq_f32(vzero, din1_1234, 3);

            float32x4_t sum0 = vmulq_f32(din0_0123, wr1);
            sum0 = vmlaq_f32(sum0, din1_0123, wr2);

            float32x4_t din0_2340 = vextq_f32(din0_1234, vzero, 1);
            float32x4_t din1_2340 = vextq_f32(din1_1234, vzero, 1);

            float32x4_t sum1 = vmulq_f32(din0_2340, wr1);
            sum1 = vmlaq_f32(sum1, din1_2340, wr2);

            float32x2_t vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
            float32x2_t vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            float32x2_t vsum = vpadd_f32(vsum0, vsum1);

            vsum = vadd_f32(vsum, vget_low_f32(wbias));
            vsum = vmax_f32(vsum, vget_low_f32(vzero));

            vst1_f32(doutr0_ptr, vsum);

            din0_ptr += 3;
            din1_ptr += 3;
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
                sum0 = vmlaq_f32(sum0, din1_1234, wr2);
                sum1 = vmulq_f32(din0_3456, wr1);
                sum1 = vmlaq_f32(sum1, din1_3456, wr2);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));
                vsum = vmax_f32(vsum, vget_low_f32(vzero));

                vst1_f32(doutr0_ptr, vsum);

                din0_ptr += 4;
                din1_ptr += 4;
                doutr0_ptr += 2;
            }
            //right
            din0_1234 = vld1q_f32(din0_ptr);
            din1_1234 = vld1q_f32(din1_ptr);

            if(vgetq_lane_u32(mask_rp, 0) == 0){
                din0_1234 = vcombine_f32(vget_low_f32(din0_1234), vget_low_f32(vzero));
                din1_1234 = vcombine_f32(vget_low_f32(din1_1234), vget_low_f32(vzero));
            }else if(vgetq_lane_u32(mask_rp, 1) == 0){
                din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);
                din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
            }

            float32x4_t din0_5678 = vld1q_f32(din0_ptr + 4);
            float32x4_t din1_5678 = vld1q_f32(din1_ptr + 4);

            sum0 = vmulq_f32(din0_1234, wr1);
            sum0 = vmlaq_f32(sum0, din1_1234, wr2);

            if(vgetq_lane_u32(mask_rp, 2) == 0){
                din0_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din0_5678));
                din1_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din1_5678));
            }else if(vgetq_lane_u32(mask_rp, 3) == 0){
                din0_5678 = vsetq_lane_f32(0.f, din0_5678, 1);
                din1_5678 = vsetq_lane_f32(0.f, din1_5678, 1);
            }
            float32x4_t din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
            float32x4_t din1_3456 = vextq_f32(din1_1234, din1_5678, 2);

            sum1 = vmulq_f32(din0_3456, wr1);
            sum1 = vmlaq_f32(sum1, din1_3456, wr2);

            vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
            vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
            vsum = vpadd_f32(vsum0, vsum1);

            float32x2_t vdata = vld1_f32(doutr0_ptr);

            vsum = vadd_f32(vsum, vget_low_f32(wbias));

            vsum = vmax_f32(vsum, vget_low_f32(vzero));

            if(vgetq_lane_u32(mask_w, 0) == 0){//00
                //null
            }else if(vgetq_lane_u32(mask_w, 1) == 0){//10
                vsum = vset_lane_f32(vget_lane_f32(vdata, 1), vsum, 1);
            }

            vst1_f32(doutr0_ptr, vsum);
#else
            
#endif //__aarch64__

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 += w_out;

            //! process mid rows
            for (int j = h_out - size_pad_bottom - 1; j > 0; j--) {
#ifdef __aarch64__
                // todo
                din0_ptr = dr0;
                din1_ptr = dr1;
                din2_ptr = dr2;
                doutr0_ptr = doutr0;

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
                sum0 =vmlaq_f32(sum0, din1_0123, wr1);
                sum0 =vmlaq_f32(sum0, din2_0123, wr2);

                sum1 = vmulq_f32(din0_2345, wr0);
                sum1 =vmlaq_f32(sum1, din1_2345, wr1);
                sum1 =vmlaq_f32(sum1, din2_2345, wr2);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));


                vsum = vmax_f32(vsum, vget_low_f32(vzero));

                vst1_f32(doutr0_ptr, vsum);


                din0_ptr += 3;
                din1_ptr += 3;
                din2_ptr += 3;
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
                    sum0 = vmlaq_f32(sum0, din1_1234, wr1);
                    sum0 = vmlaq_f32(sum0, din2_1234, wr2);

                    sum1 = vmulq_f32(din0_3456, wr0);
                    sum1 = vmlaq_f32(sum1, din1_3456, wr1);
                    sum1 = vmlaq_f32(sum1, din2_3456, wr2);

                    vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                    vsum = vpadd_f32(vsum0, vsum1);

                    vsum = vadd_f32(vsum, vget_low_f32(wbias));


                    vsum = vmax_f32(vsum, vget_low_f32(vzero));

                    vst1_f32(doutr0_ptr, vsum);

                    din0_ptr += 4;
                    din1_ptr += 4;
                    din2_ptr += 4;
                    doutr0_ptr += 2;
                }
            //right
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);
                din2_1234 = vld1q_f32(din2_ptr);

                if(vgetq_lane_u32(mask_rp, 0) == 0){
                    din0_1234 = vcombine_f32(vget_low_f32(din0_1234), vget_low_f32(vzero));
                    din1_1234 = vcombine_f32(vget_low_f32(din1_1234), vget_low_f32(vzero));
                    din2_1234 = vcombine_f32(vget_low_f32(din2_1234), vget_low_f32(vzero));
                }else if(vgetq_lane_u32(mask_rp, 1) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
                    din2_1234 = vsetq_lane_f32(0.f, din2_1234, 3);
                }
                din0_5678 = vld1q_f32(din0_ptr + 4);
                din1_5678 = vld1q_f32(din1_ptr + 4);
                float32x4_t din2_5678 = vld1q_f32(din2_ptr + 4);

                sum0 = vmulq_f32(din0_1234, wr0);
                sum0 = vmlaq_f32(sum0, din1_1234, wr1);
                sum0 = vmlaq_f32(sum0, din2_1234, wr2);

                if(vgetq_lane_u32(mask_rp, 2) == 0){
                    din0_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din0_5678));
                    din1_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din1_5678));
                    din2_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din2_5678));
                }else if(vgetq_lane_u32(mask_rp, 3) == 0){
                    din0_5678 = vsetq_lane_f32(0.f, din0_5678, 1);
                    din1_5678 = vsetq_lane_f32(0.f, din1_5678, 1);
                    din2_5678 = vsetq_lane_f32(0.f, din2_5678, 1);
                }
                din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                din1_3456 = vextq_f32(din1_1234, din1_5678, 2);
                float32x4_t din2_3456 = vextq_f32(din2_1234, din2_5678, 2);

                sum1 = vmulq_f32(din0_3456, wr0);
                sum1 = vmlaq_f32(sum1, din1_3456, wr1);
                sum1 = vmlaq_f32(sum1, din2_3456, wr2);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                float32x2_t vdata = vld1_f32(doutr0_ptr);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));


                vsum = vmax_f32(vsum, vget_low_f32(vzero));

                if(vgetq_lane_u32(mask_w, 0) == 0){//00
                    //null
                }else if(vgetq_lane_u32(mask_w, 1) == 0){//10
                 vsum = vset_lane_f32(vget_lane_f32(vdata, 1), vsum, 1);
                }

                vst1_f32(doutr0_ptr, vsum);

#else

#endif //__aarch64__

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
#ifdef __aarch64__
                // todo
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);


                din0_0123 = vextq_f32(vzero, din0_1234, 3);
                din1_0123 = vextq_f32(vzero, din1_1234, 3);

                float32x4_t din0_2345 = vextq_f32(din0_1234, vzero, 1);
                float32x4_t din1_2345 = vextq_f32(din1_1234, vzero, 1);

                sum0 = vmulq_f32(din0_0123, wr0);
                sum0 =vmlaq_f32(sum0, din1_0123, wr1);

                sum1 = vmulq_f32(din0_2345, wr0);
                sum1 =vmlaq_f32(sum1, din1_2345, wr1);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));


                vsum = vmax_f32(vsum, vget_low_f32(vzero));

                vst1_f32(doutr0_ptr, vsum);

                din0_ptr += 3;
                din1_ptr += 3;
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
                    sum0 = vmlaq_f32(sum0, din1_1234, wr1);

                    sum1 = vmulq_f32(din0_3456, wr0);
                    sum1 = vmlaq_f32(sum1, din1_3456, wr1);

                    vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                    vsum = vpadd_f32(vsum0, vsum1);

                    vsum = vadd_f32(vsum, vget_low_f32(wbias));


                    vsum = vmax_f32(vsum, vget_low_f32(vzero));

                    vst1_f32(doutr0_ptr, vsum);

                    din0_ptr += 4;
                    din1_ptr += 4;
                    doutr0_ptr += 2;
                }
            //right
                din0_1234 = vld1q_f32(din0_ptr);
                din1_1234 = vld1q_f32(din1_ptr);

                if(vgetq_lane_u32(mask_rp, 0) == 0){
                    din0_1234 = vcombine_f32(vget_low_f32(din0_1234), vget_low_f32(vzero));
                    din1_1234 = vcombine_f32(vget_low_f32(din1_1234), vget_low_f32(vzero));
                }else if(vgetq_lane_u32(mask_rp, 1) == 0){
                    din0_1234 = vsetq_lane_f32(0.f, din0_1234, 3);
                    din1_1234 = vsetq_lane_f32(0.f, din1_1234, 3);
                }
                din0_5678 = vld1q_f32(din0_ptr + 4);
                din1_5678 = vld1q_f32(din1_ptr + 4);

                sum0 = vmulq_f32(din0_1234, wr0);
                sum0 = vmlaq_f32(sum0, din1_1234, wr1);

                if(vgetq_lane_u32(mask_rp, 2) == 0){
                    din0_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din0_5678));
                    din1_5678 = vcombine_f32(vget_low_f32(vzero), vget_high_f32(din1_5678));
                }else if(vgetq_lane_u32(mask_rp, 3) == 0){
                    din0_5678 = vsetq_lane_f32(0.f, din0_5678, 1);
                    din1_5678 = vsetq_lane_f32(0.f, din1_5678, 1);
                }
                din0_3456 = vextq_f32(din0_1234, din0_5678, 2);
                din1_3456 = vextq_f32(din1_1234, din1_5678, 2);

                sum1 = vmulq_f32(din0_3456, wr0);
                sum1 = vmlaq_f32(sum1, din1_3456, wr1);

                vsum0 = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                vsum1 = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                vsum = vpadd_f32(vsum0, vsum1);

                float32x2_t vdata = vld1_f32(doutr0_ptr);

                vsum = vadd_f32(vsum, vget_low_f32(wbias));


                vsum = vmax_f32(vsum, vget_low_f32(vzero));

                if(vgetq_lane_u32(mask_w, 0) == 0){//00
                    //null
                }else if(vgetq_lane_u32(mask_w, 1) == 0){//10
                 vsum = vset_lane_f32(vget_lane_f32(vdata, 1), vsum, 1);
                }

                vst1_f32(doutr0_ptr, vsum);

#else

#endif //__aarch64__
            } // end of process bottom pad

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
                    "blt  2f                                @ jump to rightpad\n"
                    "1:                                     @ main loop start point\n"
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
                    "bne    1b                              @ jump to main loop start point\n"

                    //! process right pad
                    "2:                                     @ right pad entry\n"
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
                        "blt  4f                                @ jump to rightpad\n"
                        "3:                                     @ main loop start point\n"
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
                        "bne    3b                              @ jump to main loop start point\n"

                        // process right pad
                        "4:                                     @ right pad entry\n"
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
                        "blt  1f                                @ jump to rightpad\n"
                        "1:                                     @ main loop start point\n"
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
                        "bne    1b                              @ jump to main loop start point\n"

                        // process right pad
                        "1:                                     @ right pad entry\n"
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
#ifdef __aarch64__
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

#endif //__aarch64__

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
#ifdef __aarch64__
            // todo
#else
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
                    "vmul.f32 q4, q10, %e[wr1][1]           @ mul weight 10, outr0\n"
                    "vmul.f32 q5, q10, %e[wr0][1]           @ mul weight 00, outr1\n"

                    "vbif q11, q15, %q[mask]                @ bit select, deal with right pad\n"
                    "vmla.f32 q4, q11, %e[wr2][1]           @ mul weight 20, outr0\n"
                    "vmla.f32 q5, q11, %e[wr1][1]           @ mul weight 10, outr1\n"

                    "vbif q12, q15, %q[mask]                @ bit select, deal with right pad\n"
                    "vmla.f32 q5, q12, %e[wr2][1]           @ mul weight 20, outr1\n"

                    "vext.32  q6, q10, q15, #1              @ shift left r1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"

                    "vext.32  q6, q11, q15, #1              @ shift left r2\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 21, outr0\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 11, outr1\n"

                    "vext.32  q6, q12, q15, #1              @ shift left r3\n"
                    "vmla.f32 q5, q6,  %f[wr2][0]           @ mul weight 21, outr1\n"

                    "vext.32  q6, q15, q10, #3              @ shift right r1\n"
                    "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"
                    "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"

                    "vext.32  q6, q15, q11, #3              @ shift right r2\n"
                    "vmla.f32 q4, q6,  %e[wr2][0]           @ mul weight 22, outr0\n"
                    "vmla.f32 q5, q6,  %e[wr1][0]           @ mul weight 12, outr1\n"

                    "vext.32  q6, q15, q12, #3              @ shift right r3\n"
                    "vmla.f32 q5, q6,  %e[wr2][0]           @ mul weight 22, outr1\n"

                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                    "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vbif q8, q10, %q[mask]                 @ bit select\n"
                    "vbif q9, q11, %q[mask]                 @ bit select\n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"

                    "cmp %[flag1],  #1                      @ check if hin = 1\n"
                    "beq    1f                              @ jump to next block\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "1:                                     @ end top\n"

                    "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"
                    "sub %[dout_ptr2], %[dout_ptr2], %[pad_right] @ sub \n"

                    "sub %[din0_ptr], %[din0_ptr],   %[pad_right] @ sub \n"
                    "sub %[din1_ptr], %[din1_ptr],   %[pad_right] @ sub \n"
                    "sub %[din2_ptr], %[din2_ptr],   %[pad_right] @ sub \n"

            :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [pad_right] "+r" (right_pad_sub)
            :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                        [bias] "w"(wbias), [mask] "w" (vmask_rp), [flag1] "r" (flag_hin1)
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
                        "vmul.f32 q4, q8, %e[wr0][1]   @mul weight 00, outr0\n"

                        "vbif q9, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vmul.f32 q5, q9, %e[wr0][1]  @mul weight 00, outr1\n"
                        "vmla.f32 q4, q9, %e[wr1][1]  @mul weight 10, outr0\n"

                        "vbif q10, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q10, %e[wr1][1]  @mul weight 10, outr1\n"
                        "vmla.f32 q4, q10, %e[wr2][1]  @mul weight 20, outr0\n"

                        "vbif q11, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q11, %e[wr2][1]  @mul weight 20, outr1\n"

                        "vext.32  q6, q8, q15, #1     @ shift left r0\n"
                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 01, outr0\n"

                        "vext.32  q6, q9, q15, #1   @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]  @mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]  @mul weight 11, outr0\n"

                        "vext.32  q6, q10, q15, #1  @ shift left r2\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 21, outr0\n"

                        "vext.32  q6, q11, q15, #1  @ shift left r3\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 21, outr1\n"

                        "vext.32  q6, q15, q8, #3              @ shift right r0\n"
                        "vmla.f32 q4, q6, %e[wr0][0]  @mul weight 02, outr0\n"

                        "vext.32  q6, q15, q9, #3              @ shift right r1\n"
                        "vmla.f32 q5, q6, %e[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][0]  @mul weight 12, outr0\n"

                        "vext.32  q6, q15, q10, #3              @ shift right r2\n"
                        "vmla.f32 q5, q6,  %e[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][0] @mul weight 22, outr0\n"

                        "vext.32  q6, q15, q11, #3              @ shift right r3\n"
                        "vmla.f32 q5, q6,  %e[wr2][0] @mul weight 22, outr1\n"

                        "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                        "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"

                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "vbif q8, q10, %q[mask]                 @ bit select\n"
                        "vbif q9, q11, %q[mask]                 @ bit select\n"

                        "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                        "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"
                        "sub %[dout_ptr2], %[dout_ptr2], %[pad_right] @ sub \n"

                        "sub %[din0_ptr], %[din0_ptr],   %[pad_right] @ sub \n"
                        "sub %[din1_ptr], %[din1_ptr],   %[pad_right] @ sub \n"
                        "sub %[din2_ptr], %[din2_ptr],   %[pad_right] @ sub \n"
                        "sub %[din3_ptr], %[din3_ptr],   %[pad_right] @ sub \n"

                :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                            [pad_right] "+r" (right_pad_sub)
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
            if (h_in > 2) {
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
                        "vmul.f32 q4, q8, %e[wr0][1]            @ mul weight 00, outr0\n"

                        "vbif q9, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vmla.f32 q4, q9, %e[wr1][1]           @ mul weight 10, outr0\n"
                        "vmul.f32 q5, q9, %e[wr0][1]           @ mul weight 00, outr1\n"

                        "vbif q10, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vmla.f32 q4, q10, %e[wr2][1]           @ mul weight 20, outr0\n"
                        "vmla.f32 q5, q10, %e[wr1][1]           @ mul weight 10, outr1\n"

                        "vext.32  q6, q8, q15, #1               @ shift left r0\n"
                        "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 01, outr0\n"

                        "vext.32  q6, q9, q15, #1               @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"

                        "vext.32  q6, q10, q15, #1              @ shift left r2\n"
                        "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 21, outr0\n"

                        "vext.32  q6, q15, q8, #3               @ shift right r0\n"
                        "vmla.f32 q4, q6, %e[wr0][0]            @ mul weight 02, outr0\n"

                        "vext.32  q6, q15, q9, #3               @ shift right r1\n"
                        "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"
                        "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"

                        "vext.32  q6, q15, q10, #3              @ shift right r2\n"
                        "vmla.f32 q4, q6,  %e[wr2][0]           @ mul weight 22, outr0\n"
                        "vmla.f32 q5, q6,  %e[wr1][0]           @ mul weight 12, outr1\n"

                        "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"

                        "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                        "vbif q8, q10, %q[mask]                 @ bit select\n"

                        "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                        "beq    1f                              @ jump to next block\n"
                        "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"
                        "vbif q9, q11, %q[mask]                 @ bit select\n"
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
#endif //__aarch64__
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
#ifdef __aarch64__
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
#else
            
#endif //__aarch64__

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 = doutr0 + w_out;

            //! process mid rows
            for (int j = tile_h - 1; j > 0; j--) {
#ifdef __aarch64__
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
#else

#endif //__aarch64__

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
#ifdef __aarch64__
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
#else
                
#endif //__aarch64__
            } // end of process bottom pad
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

#ifdef __aarch64__
            // todo

#else
            asm volatile(
            // process  pad
            "pld [%[din0_ptr], #128]                @ preload data\n"
                    "pld [%[din1_ptr], #128]                @ preload data\n"

                    "vmov.u32  q15, #0                      @ zero buf\n"

                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0 1234\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1 1234\n"
                    "vbif q10, q15, %q[mask_din]            @ bit select, deal with right pad\n"
                    "vbif q12, q15, %q[mask_din]            @ bit select, deal with right pad\n"


                    "vmul.f32 q8, q10, %e[wr1][1]           @ mul weight 11, 1234\n"
                    "vext.32 q6, q15, q10, #3              @ shift right 1, 0123\n"
                    "vext.32 q7, q15, q12, #3              @ shift right 1, 0123\n"

                    "vmul.f32 q9, q12, %e[wr2][1]           @ mul weight 21, 1234\n"
                    "vext.32 q10, q10, q15, #1              @ shift left 1, 2340\n"
                    "vext.32 q12, q12, q15, #1              @ shift left 1, 2340\n"

                    "vmla.f32 q8, q6, %e[wr1][0]           @ mla weight 10, 0123\n"
                    "vmla.f32 q9, q7, %e[wr2][0]           @ mla weight 20, 0123\n"

                    "vmla.f32 q8, q10, %f[wr1][0]           @ mla weight 12, 2340\n"
                    "vmla.f32 q9, q12, %f[wr2][0]           @ mla weight 22, 2340\n"

                    "vadd.f32 q6, q8, q9                    @ add \n"

                    "vld1.32  {d20}, [%[dout_ptr0]]         @ load dout\n"

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
            :"q6", "q7", "q8", "q10", "q12", "q15"
            );

#endif //__aarch64__

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 = doutr0 + w_out;

            //! process mid rows
            for (int j = tile_h - 1; j > 0; j--) {
#ifdef __aarch64__
                // todo
#else
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


                        "vmul.f32 q9, q10, %e[wr0][1]           @ mul weight 01, 1234\n"
                        "vext.f32 q6, q15, q10, #3              @ shift right 1, 0123\n"

                        "vmul.f32 q11, q12, %e[wr1][1]           @ mul weight 11, 1234\n"
                        "vext.f32 q7, q15, q12, #3              @ shift right 1, 0123\n"

                        "vmul.f32 q13, q14, %e[wr2][1]           @ mul weight 21, 1234\n"
                        "vext.f32 q8, q15, q14, #3              @ shift right 1, 0123\n"

                        "vmla.f32 q9, q6, %e[wr0][0]           @ mul weight 00, 0123\n"
                        "vext.f32 q10, q10, q15, #1              @ shift left 1, 2340\n"

                        "vmla.f32 q11, q7, %e[wr1][0]           @ mul weight 10, 0123\n"
                        "vext.f32 q12, q12, q15, #1              @ shift left 1, 2340\n"

                        "vmla.f32 q13, q8, %e[wr2][0]           @ mul weight 20, 0123\n"
                        "vext.f32 q14, q14, q15, #1              @ shift left 1, 2340\n"

                        "vmla.f32 q9, q10, %f[wr0][0]           @ mul weight 12, 2340\n"
                        "vmla.f32 q11, q12, %f[wr1][0]           @ mul weight 22, 2340\n"
                        "vmla.f32 q13, q14, %f[wr2][0]           @ mul weight 22, 2340\n"

                        "vld1.32  {d20}, [%[dout_ptr0]]         @ load dout\n"
                        "vadd.f32  q8, q9, q11                   @add\n"
                        "vadd.f32  q6, q8, q13                   @add\n"

                        "vmov.f32  s25, s26                     @ mov \n"
                        "vadd.f32  d12, d12, %e[bias]           @ add bias \n"

                        //"vst1.32  {d17},   [%[tmp_ptr]]! \n"
                        "vbif d12, d20, %e[mask_w]              @ bit select\n"

                        //"vst1.32  {d17},   [%[tmp_ptr]] \n"

                        "vst1.32  {d12}, [%[dout_ptr0]]!        @ store result, add pointer\n"

                :[dout_ptr0] "+r"(doutr0_ptr), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr), [din2_ptr] "+r"(din2_ptr)
                :[wr1] "w"(wr1), [wr2] "w"(wr2), [wr0] "w" (wr0), \
                    [bias] "w"(wbias), [mask_din] "w" (vmask_rp), [mask_w] "w" (vmask_w) //, \
                    //[pad_right] "r" (size_right_remain)
                :"q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );

#endif //__aarch64__

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
                doutr0 = doutr0 + w_out;
            } // end of process mid rows
            // process bottom pad if needed
            if (size_pad_bottom) {
                din0_ptr = dr0;
                din1_ptr = dr1;

#ifdef __aarch64__
                // todo
#else
                asm volatile(
                // process  pad
                "pld [%[din0_ptr], #128]                @ preload data\n"
                        "pld [%[din1_ptr], #128]                @ preload data\n"

                        "vmov.u32  q15, #0                      @ zero buf\n"

                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0 1234\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1 1234\n"
                        "vbif q10, q15, %q[mask_din]            @ bit select, deal with right pad\n"
                        "vbif q12, q15, %q[mask_din]            @ bit select, deal with right pad\n"


                        "vmul.f32 q8, q10, %e[wr1][1]           @ mul weight 11, 1234\n"
                        "vext.f32 q6, q15, q10, #3              @ shift right 1, 0123\n"
                        "vext.f32 q7, q15, q12, #3              @ shift right 1, 0123\n"

                        "vmul.f32 q9, q12, %e[wr2][1]           @ mul weight 21, 1234\n"
                        "vext.f32 q10, q10, q15, #1              @ shift left 1, 2340\n"
                        "vext.f32 q12, q12, q15, #1              @ shift left 1, 2340\n"

                        "vmla.f32 q8, q6, %e[wr1][0]           @ mul weight 10, 0123\n"
                        "vmla.f32 q9, q7, %e[wr2][0]           @ mul weight 20, 0123\n"

                        "vmla.f32 q8, q10, %f[wr1][0]           @ mul weight 12, 2340\n"
                        "vmla.f32 q9, q12, %f[wr2][0]           @ mul weight 22, 2340\n"

                        "vadd.f32 q6, q8, q9                    @ add \n"
                        "vld1.32  {d20}, [%[dout_ptr0]]         @ load dout\n"

                        "vmov.f32  s25, s26                     @ mov \n"
                        "vadd.f32  d12, d12, %e[bias]           @ add bias \n"

                        //"vst1.32  {d17},   [%[tmp_ptr]]! \n"
                        "vbif d12, d20, %e[mask_w]              @ bit select\n"

                        //"vst1.32  {d17},   [%[tmp_ptr]] \n"

                        "vst1.32  {d12}, [%[dout_ptr0]]!        @ store result, add pointer\n"

                :[dout_ptr0] "+r"(doutr0_ptr), [din0_ptr] "+r"(din0_ptr), \
                    [din1_ptr] "+r"(din1_ptr)
                :[wr1] "w"(wr1), [wr2] "w"(wr2), \
                    [bias] "w"(wbias), [mask_din] "w" (vmask_rp), [mask_w]  "w" (vmask_w) //, \
                    //[pad_right] "r" (size_right_remain)
                :"q6", "q7", "q8", "q10", "q12", "q15"
                );
#endif //__aarch64__
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
#ifdef __aarch64__
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

#endif //__aarch64__

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
#ifdef __aarch64__
            // todo
#else
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
                    "vmul.f32 q4, q10, %e[wr1][1]           @ mul weight 10, outr0\n"
                    "vmul.f32 q5, q10, %e[wr0][1]           @ mul weight 00, outr1\n"

                    "vbif q11, q15, %q[mask]                @ bit select, deal with right pad\n"
                    "vmla.f32 q4, q11, %e[wr2][1]           @ mul weight 20, outr0\n"
                    "vmla.f32 q5, q11, %e[wr1][1]           @ mul weight 10, outr1\n"

                    "vbif q12, q15, %q[mask]                @ bit select, deal with right pad\n"
                    "vmla.f32 q5, q12, %e[wr2][1]           @ mul weight 20, outr1\n"

                    "vext.32  q6, q10, q15, #1              @ shift left r1\n"
                    "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"
                    "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"

                    "vext.32  q6, q11, q15, #1              @ shift left r2\n"
                    "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 21, outr0\n"
                    "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 11, outr1\n"

                    "vext.32  q6, q12, q15, #1              @ shift left r3\n"
                    "vmla.f32 q5, q6,  %f[wr2][0]           @ mul weight 21, outr1\n"

                    "vext.32  q6, q15, q10, #3              @ shift right r1\n"
                    "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"
                    "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"

                    "vext.32  q6, q15, q11, #3              @ shift right r2\n"
                    "vmla.f32 q4, q6,  %e[wr2][0]           @ mul weight 22, outr0\n"
                    "vmla.f32 q5, q6,  %e[wr1][0]           @ mul weight 12, outr1\n"

                    "vext.32  q6, q15, q12, #3              @ shift right r3\n"
                    "vmla.f32 q5, q6,  %e[wr2][0]           @ mul weight 22, outr1\n"

                    "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                    "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"

                    "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                    "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                    "vmax.f32 q8, q15                       @ relu \n"
                    "vmax.f32 q9, q15                       @ relu \n"

                    "vbif q8, q10, %q[mask]                 @ bit select\n"
                    "vbif q9, q11, %q[mask]                 @ bit select\n"

                    "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"

                    "cmp %[flag1],  #1                      @ check if hin = 1\n"
                    "beq    1f                              @ jump to next block\n"
                    "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"
                    "1:                                     @ end top\n"

                    "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"
                    "sub %[dout_ptr2], %[dout_ptr2], %[pad_right] @ sub \n"

                    "sub %[din0_ptr], %[din0_ptr],   %[pad_right] @ sub \n"
                    "sub %[din1_ptr], %[din1_ptr],   %[pad_right] @ sub \n"
                    "sub %[din2_ptr], %[din2_ptr],   %[pad_right] @ sub \n"

            :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [pad_right] "+r" (right_pad_sub)
            :[wr0] "w"(wr0), [wr1] "w"(wr1), [wr2] "w"(wr2), \
                        [bias] "w"(wbias), [mask] "w" (vmask_rp), [flag1] "r" (flag_hin1)
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
                        "vmul.f32 q4, q8, %e[wr0][1]   @mul weight 00, outr0\n"

                        "vbif q9, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vmul.f32 q5, q9, %e[wr0][1]  @mul weight 00, outr1\n"
                        "vmla.f32 q4, q9, %e[wr1][1]  @mul weight 10, outr0\n"

                        "vbif q10, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q10, %e[wr1][1]  @mul weight 10, outr1\n"
                        "vmla.f32 q4, q10, %e[wr2][1]  @mul weight 20, outr0\n"

                        "vbif q11, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vmla.f32 q5, q11, %e[wr2][1]  @mul weight 20, outr1\n"

                        "vext.32  q6, q8, q15, #1     @ shift left r0\n"
                        "vmla.f32 q4, q6, %f[wr0][0]  @mul weight 01, outr0\n"

                        "vext.32  q6, q9, q15, #1   @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]  @mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]  @mul weight 11, outr0\n"

                        "vext.32  q6, q10, q15, #1  @ shift left r2\n"
                        "vmla.f32 q5, q6,  %f[wr1][0] @mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0] @mul weight 21, outr0\n"

                        "vext.32  q6, q11, q15, #1  @ shift left r3\n"
                        "vmla.f32 q5, q6,  %f[wr2][0] @mul weight 21, outr1\n"

                        "vext.32  q6, q15, q8, #3              @ shift right r0\n"
                        "vmla.f32 q4, q6, %e[wr0][0]  @mul weight 02, outr0\n"

                        "vext.32  q6, q15, q9, #3              @ shift right r1\n"
                        "vmla.f32 q5, q6, %e[wr0][0]  @mul weight 02, outr1\n"
                        "vmla.f32 q4, q6, %e[wr1][0]  @mul weight 12, outr0\n"

                        "vext.32  q6, q15, q10, #3              @ shift right r2\n"
                        "vmla.f32 q5, q6,  %e[wr1][0] @mul weight 12, outr1\n"
                        "vmla.f32 q4, q6,  %e[wr2][0] @mul weight 22, outr0\n"

                        "vext.32  q6, q15, q11, #3              @ shift right r3\n"
                        "vmla.f32 q5, q6,  %e[wr2][0] @mul weight 22, outr1\n"

                        "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"
                        "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"

                        "vadd.f32 q8, q4, %q[bias] @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias] @ add bias \n"

                        "vmax.f32 q8, q15             @ relu \n"
                        "vmax.f32 q9, q15             @ relu \n"

                        "vbif q8, q10, %q[mask]                 @ bit select\n"
                        "vbif q9, q11, %q[mask]                 @ bit select\n"

                        "vst1.32  {d16-d17}, [%[dout_ptr1]]!    @ store result, add pointer\n"
                        "vst1.32  {d18-d19}, [%[dout_ptr2]]!    @ store result, add pointer\n"

                        "sub %[dout_ptr1], %[dout_ptr1], %[pad_right] @ sub \n"
                        "sub %[dout_ptr2], %[dout_ptr2], %[pad_right] @ sub \n"

                        "sub %[din0_ptr], %[din0_ptr],   %[pad_right] @ sub \n"
                        "sub %[din1_ptr], %[din1_ptr],   %[pad_right] @ sub \n"
                        "sub %[din2_ptr], %[din2_ptr],   %[pad_right] @ sub \n"
                        "sub %[din3_ptr], %[din3_ptr],   %[pad_right] @ sub \n"

                :[dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), \
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                            [pad_right] "+r" (right_pad_sub)
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
            if (h_in > 2) {
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
                        "vmul.f32 q4, q8, %e[wr0][1]            @ mul weight 00, outr0\n"

                        "vbif q9, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vmla.f32 q4, q9, %e[wr1][1]           @ mul weight 10, outr0\n"
                        "vmul.f32 q5, q9, %e[wr0][1]           @ mul weight 00, outr1\n"

                        "vbif q10, q15, %q[mask]                @ bit select, deal with right pad\n"
                        "vmla.f32 q4, q10, %e[wr2][1]           @ mul weight 20, outr0\n"
                        "vmla.f32 q5, q10, %e[wr1][1]           @ mul weight 10, outr1\n"

                        "vext.32  q6, q8, q15, #1               @ shift left r0\n"
                        "vmla.f32 q4, q6, %f[wr0][0]            @ mul weight 01, outr0\n"

                        "vext.32  q6, q9, q15, #1               @ shift left r1\n"
                        "vmla.f32 q5, q6, %f[wr0][0]            @ mul weight 01, outr1\n"
                        "vmla.f32 q4, q6, %f[wr1][0]            @ mul weight 11, outr0\n"

                        "vext.32  q6, q10, q15, #1              @ shift left r2\n"
                        "vmla.f32 q5, q6,  %f[wr1][0]           @ mul weight 11, outr1\n"
                        "vmla.f32 q4, q6,  %f[wr2][0]           @ mul weight 21, outr0\n"

                        "vext.32  q6, q15, q8, #3               @ shift right r0\n"
                        "vmla.f32 q4, q6, %e[wr0][0]            @ mul weight 02, outr0\n"

                        "vext.32  q6, q15, q9, #3               @ shift right r1\n"
                        "vmla.f32 q4, q6, %e[wr1][0]            @ mul weight 12, outr0\n"
                        "vmla.f32 q5, q6, %e[wr0][0]            @ mul weight 02, outr1\n"

                        "vext.32  q6, q15, q10, #3              @ shift right r2\n"
                        "vmla.f32 q4, q6,  %e[wr2][0]           @ mul weight 22, outr0\n"
                        "vmla.f32 q5, q6,  %e[wr1][0]           @ mul weight 12, outr1\n"

                        "vld1.32  {d20-d21}, [%[dout_ptr1]]     @ load dout r0\n"

                        "vadd.f32 q8, q4, %q[bias]              @ add bias \n"
                        "vadd.f32 q9, q5, %q[bias]              @ add bias \n"

                        "vmax.f32 q8, q15                       @ relu \n"
                        "vmax.f32 q9, q15                       @ relu \n"

                        "vbif q8, q10, %q[mask]                 @ bit select\n"

                        "vst1.32  {d16-d17},   [%[dout_ptr1]]!  @ store result, add pointer\n"
                        "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                        "beq    1f                              @ jump to next block\n"
                        "vld1.32  {d22-d23}, [%[dout_ptr2]]     @ load dout r1\n"
                        "vbif q9, q11, %q[mask]                 @ bit select\n"
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
#endif //__aarch64__
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
#ifdef __aarch64__
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
#else
            
#endif //__aarch64__

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 = doutr0 + w_out;

            //! process mid rows
            for (int j = tile_h - 1; j > 0; j--) {
#ifdef __aarch64__
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
#else

#endif //__aarch64__

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
#ifdef __aarch64__
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
#else
                
#endif //__aarch64__
            } // end of process bottom pad
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
#ifdef __aarch64__
            // todo

#else
            asm volatile(
            // process  pad
            "pld [%[din0_ptr], #128]                @ preload data\n"
                    "pld [%[din1_ptr], #128]                @ preload data\n"

                    "vmov.u32  q15, #0                      @ zero buf\n"

                    "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0 1234\n"
                    "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1 1234\n"
                    "vbif q10, q15, %q[mask_din]            @ bit select, deal with right pad\n"
                    "vbif q12, q15, %q[mask_din]            @ bit select, deal with right pad\n"


                    "vmul.f32 q8, q10, %e[wr1][1]           @ mul weight 11, 1234\n"
                    "vext.f32 q6, q15, q10, #3              @ shift right 1, 0123\n"
                    "vext.f32 q7, q15, q12, #3              @ shift right 1, 0123\n"

                    "vmul.f32 q9, q12, %e[wr2][1]           @ mul weight 21, 1234\n"
                    "vext.f32 q10, q10, q15, #1              @ shift left 1, 2340\n"
                    "vext.f32 q12, q12, q15, #1              @ shift left 1, 2340\n"

                    "vmla.f32 q8, q6, %e[wr1][0]           @ mul weight 10, 0123\n"
                    "vmla.f32 q9, q7, %e[wr2][0]           @ mul weight 20, 0123\n"

                    "vmla.f32 q8, q10, %f[wr1][0]           @ mul weight 12, 2340\n"
                    "vmla.f32 q9, q12, %f[wr2][0]           @ mul weight 22, 2340\n"

                    "vadd.f32 q6, q8, q9                    @ add \n"
                    "vld1.32  {d20}, [%[dout_ptr0]]         @ load dout\n"

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
            :"q6", "q7", "q8", "q10", "q12", "q15"
            );

#endif //__aarch64__

            dr0 = dr1;
            dr1 = dr2;
            dr2 = dr1 + w_in;
            doutr0 = doutr0 + w_out;

            //! process mid rows
            for (int j = tile_h - 1; j > 0; j--) {
#ifdef __aarch64__
                // todo
#else
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


                        "vmul.f32 q9, q10, %e[wr0][1]           @ mul weight 01, 1234\n"
                        "vext.f32 q6, q15, q10, #3              @ shift right 1, 0123\n"

                        "vmul.f32 q11, q12, %e[wr1][1]           @ mul weight 11, 1234\n"
                        "vext.f32 q7, q15, q12, #3              @ shift right 1, 0123\n"

                        "vmul.f32 q13, q14, %e[wr2][1]           @ mul weight 21, 1234\n"
                        "vext.f32 q8, q15, q14, #3              @ shift right 1, 0123\n"

                        "vmla.f32 q9, q6, %e[wr0][0]           @ mul weight 00, 0123\n"
                        "vext.f32 q10, q10, q15, #1              @ shift left 1, 2340\n"

                        "vmla.f32 q11, q7, %e[wr1][0]           @ mul weight 10, 0123\n"
                        "vext.f32 q12, q12, q15, #1              @ shift left 1, 2340\n"

                        "vmla.f32 q13, q8, %e[wr2][0]           @ mul weight 20, 0123\n"
                        "vext.f32 q14, q14, q15, #1              @ shift left 1, 2340\n"

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

#endif //__aarch64__

                dr0 = dr2;
                dr1 = dr0 + w_in;
                dr2 = dr1 + w_in;
                doutr0 += w_out;
            } // end of process mid rows

            // process bottom pad if needed
            if (size_pad_bottom) {
                din0_ptr = dr0;
                din1_ptr = dr1;
#ifdef __aarch64__
                // todo
#else
                asm volatile(
                // process  pad
                "pld [%[din0_ptr], #128]                @ preload data\n"
                        "pld [%[din1_ptr], #128]                @ preload data\n"

                        "vmov.u32  q15, #0                      @ zero buf\n"

                        "vld1.32  {d20-d21}, [%[din0_ptr]]!     @ load din r0 1234\n"
                        "vld1.32  {d24-d25}, [%[din1_ptr]]!     @ load din r1 1234\n"
                        "vbif q10, q15, %q[mask_din]            @ bit select, deal with right pad\n"
                        "vbif q12, q15, %q[mask_din]            @ bit select, deal with right pad\n"


                        "vmul.f32 q8, q10, %e[wr1][1]           @ mul weight 11, 1234\n"
                        "vext.f32 q6, q15, q10, #3              @ shift right 1, 0123\n"
                        "vext.f32 q7, q15, q12, #3              @ shift right 1, 0123\n"

                        "vmul.f32 q9, q12, %e[wr2][1]           @ mul weight 21, 1234\n"
                        "vext.f32 q10, q10, q15, #1              @ shift left 1, 2340\n"
                        "vext.f32 q12, q12, q15, #1              @ shift left 1, 2340\n"

                        "vmla.f32 q8, q6, %e[wr1][0]           @ mul weight 10, 0123\n"
                        "vmla.f32 q9, q7, %e[wr2][0]           @ mul weight 20, 0123\n"

                        "vmla.f32 q8, q10, %f[wr1][0]           @ mul weight 12, 2340\n"
                        "vmla.f32 q9, q12, %f[wr2][0]           @ mul weight 22, 2340\n"

                        "vadd.f32 q6, q8, q9                    @ add \n"
                        "vld1.32  {d20}, [%[dout_ptr0]]         @ load dout\n"

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
                :"q6", "q7", "q8", "q10", "q12", "q15"
                );
#endif //__aarch64__
            } // end of process bottom pad
        }
    }
}
#endif
} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE
