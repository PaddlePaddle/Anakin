
#include "saber/funcs/impl/arm/saber_activation.h"
#include "saber/funcs/impl/arm/impl/neon_mathfun.h"
namespace anakin{
namespace saber {

template <>
SaberStatus SaberActivation<ARM, AK_FLOAT>::dispatch(
        const std::vector<Tensor<ARM>*>& inputs,
        std::vector<Tensor<ARM>*>& outputs,
        ActivationParam<ARM> &param) {
   
    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    float* ptr_out = (float*)outputs[0]->mutable_data();
    const float* ptr_in = (const float*)inputs[0]->data();
    int size = inputs[0]->valid_size();
    int csize= size / (channel * num);
    int threads = 1;
    this->_ctx->get_mode(threads);
    //multi threads
    int nums_per_thread = size / threads;
    int remain = size - threads * nums_per_thread;
    //openmp 16
    int neon_loop_cnt = nums_per_thread >> 4;
    int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
    //deal with 4 data
    int neon_loop_cnt_dim4 = nums_per_thread >> 2;
    int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);
    float32x4_t vzero = vdupq_n_f32(0.f);
    float coef = param.coef;
    float slope = param.negative_slope;
    bool channel_shared = param.prelu_param.channel_shared;
    float* slopes_ptr = nullptr; 
    switch (param.active){
        //x > 0 ? x :0
        case Active_relu:
            #pragma omp parallel for
            for (int i = 0; i < threads; ++i) {
                const float* ptr_in_thread = ptr_in + i * nums_per_thread;
                float* ptr_out_thread = ptr_out + i * nums_per_thread;
                int cnt = neon_loop_cnt;
#ifdef __aarch64__
                for (int num = 0; num < neon_loop_cnt; num++){
                    float32x4_t vr0 = vld1q_f32(ptr_in_thread);
                   // ptr_in_thread+=4;
                    float32x4_t vr1 = vld1q_f32(ptr_in_thread + 4);
                   // ptr_in_thread+=4;
                    float32x4_t vr2 = vld1q_f32(ptr_in_thread + 8);
                   // ptr_in_thread+=4;
                    float32x4_t vr3 = vld1q_f32(ptr_in_thread + 12);
                    //ptr_in_thread+=4;
                    ptr_in_thread += 16;
                    vr0 = vmaxq_f32(vr0, vzero);
                    vr1 = vmaxq_f32(vr1, vzero);
                    vr2 = vmaxq_f32(vr2, vzero);
                    vr3 = vmaxq_f32(vr3, vzero);
                    vst1q_f32(ptr_out_thread, vr0);
                    //ptr_out_thread+=4;
                    vst1q_f32(ptr_out_thread + 4, vr1);
                   // ptr_out_thread+=4;
                    vst1q_f32(ptr_out_thread + 8, vr2);
                   // ptr_out_thread+=4;
                    vst1q_f32(ptr_out_thread + 12, vr3);
                    //ptr_out_thread+=4;
                    ptr_out_thread += 16;
                }      
#else
                if (cnt > 0) {
                    asm volatile (
                    "1:                                     @ loop header\n"
                            "vld1.32  {d0-d1}, [%[din]]!            @ load din 0\n"
                            "vld1.32  {d2-d3}, [%[din]]!            @ load din 0\n"
                            "vld1.32  {d4-d5}, [%[din]]!            @ load din 0\n"
                            "vld1.32  {d6-d7}, [%[din]]!            @ load din 0\n"

                            "vmax.f32 q8, q0, %q[vzero]             @ relu\n"
                            "vmax.f32 q9, q1, %q[vzero]             @ relu\n"
                            "vmax.f32 q10, q2, %q[vzero]            @ relu\n"
                            "vmax.f32 q11, q3, %q[vzero]            @ relu\n"

                            "vst1.32  {d16-d17}, [%[dout]]!         @ store result, add pointer\n"
                            "pld [%[din]]                           @ preload data\n"
                            "vst1.32  {d18-d19}, [%[dout]]!         @ store result, add pointer\n"
                            "pld [%[din], #128]                     @ preload data\n"
                            "vst1.32  {d20-d21}, [%[dout]]!         @ store result, add pointer\n"
                            "pld [%[din], #256]                     @ preload data\n"
                            "vst1.32  {d22-d23}, [%[dout]]!         @ store result, add pointer\n"
                            "pld [%[din], #384]                     @ preload data\n"

                            "subs %[cnt], #1                        @ loop count minus 1\n"
                            "bne    1b                              @ jump to main loop start point\n"
                    :[dout] "+r"(ptr_out_thread), [din] "+r"(ptr_in_thread), [cnt] "+r"(cnt)
                    :[vzero] "w" (vzero)
                    :"q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
                    );
                }
#endif
                for (int j = 0; j < neon_loop_remain; j++) {
                    ptr_out_thread[0] = ptr_in_thread[0] > 0.f ? ptr_in_thread[0] : 0.f;
                    ptr_in_thread++;
                    ptr_out_thread++;
                }
            }
            ptr_out = ptr_out + threads * nums_per_thread;
            ptr_in = ptr_in + threads * nums_per_thread;
            for (int i = 0; i < remain; i++) {
                ptr_out[0] = ptr_in[0] > 0.f ? ptr_in[0] : 0.f;
                ptr_in++;
                ptr_out++;
            }
            break;

        // x > 0 ? x : 0;
        // x < threshold ? x : threshold
        case Active_clipped_relu:
            //coef = param.coef;
             #pragma omp parallel for
            for (int i = 0; i < threads; ++i) {
                const float* ptr_in_thread = ptr_in + i * nums_per_thread;
                float* ptr_out_thread = ptr_out + i * nums_per_thread;
                int cnt = neon_loop_cnt;
                float32x4_t vthreshold = vdupq_n_f32(coef);
#ifdef __aarch64__
                for (int num = 0; num < neon_loop_cnt; num++){
                    float32x4_t vr0 = vld1q_f32(ptr_in_thread);
                    float32x4_t vr1 = vld1q_f32(ptr_in_thread + 4);
                    float32x4_t vr2 = vld1q_f32(ptr_in_thread + 8);
                    float32x4_t vr3 = vld1q_f32(ptr_in_thread + 12);
                    ptr_in_thread += 16;

                    vr0 = vmaxq_f32(vr0,vzero);
                    vr1 = vmaxq_f32(vr1,vzero);
                    vr2 = vmaxq_f32(vr2,vzero);
                    vr3 = vmaxq_f32(vr3,vzero);
                    
                    uint32x4_t vmask0 = vcgeq_f32(vr0, vthreshold);
                    uint32x4_t vmask1 = vcgeq_f32(vr1, vthreshold);
                    uint32x4_t vmask2 = vcgeq_f32(vr2, vthreshold);
                    uint32x4_t vmask3 = vcgeq_f32(vr3, vthreshold);
                    
                    float32x4_t vout0 =vbslq_f32(vmask0, vthreshold, vr0);
                    float32x4_t vout1 =vbslq_f32(vmask1, vthreshold, vr1);
                    float32x4_t vout2 =vbslq_f32(vmask2, vthreshold, vr2);
                    float32x4_t vout3 =vbslq_f32(vmask3, vthreshold, vr3);


                    vst1q_f32(ptr_out_thread, vout0);
                    vst1q_f32(ptr_out_thread + 4, vout1);
                    vst1q_f32(ptr_out_thread + 8, vout2);
                    vst1q_f32(ptr_out_thread + 12, vout3);
                    //ptr_out_thread+=4;
                    ptr_out_thread += 16;
                }      
#else
                if (cnt > 0) {
                    asm volatile (
                    "3:                                     @ loop header\n"
                            "vld1.32  {d0-d1}, [%[din]]!            @ load din 0\n"
                            "vld1.32  {d2-d3}, [%[din]]!            @ load din 0\n"
                            "vld1.32  {d4-d5}, [%[din]]!            @ load din 0\n"
                            "vld1.32  {d6-d7}, [%[din]]!            @ load din 0\n"

                            "vmax.f32 q8, q0, %q[vzero]             @ relu\n"
                            "vmax.f32 q9, q1, %q[vzero]             @ relu\n"
                            "vmax.f32 q10, q2, %q[vzero]            @ relu\n"
                            "vmax.f32 q11, q3, %q[vzero]            @ relu\n"

                            "vcgt.f32  q0, q8, %q[vthreshold]        @ v0 > threshold\n"
                            "vcgt.f32  q1, q9, %q[vthreshold]        @ v0 > threshold\n"
                            "vcgt.f32  q2, q10, %q[vthreshold]        @ v0 > threshold\n"
                            "vcgt.f32  q3, q11, %q[vthreshold]        @ v0 > threshold\n"

                            "vbit.f32 q8, %q[vthreshold], q0        @ \n"
                            "vbit.f32 q9, %q[vthreshold], q1        @ \n"
                            "vbit.f32 q10, %q[vthreshold], q2        @ \n"
                            "vbit.f32 q11, %q[vthreshold], q3        @ \n"

                            "vst1.32  {d16-d17}, [%[dout]]!         @ store result, add pointer\n"
                            "pld [%[din]]                           @ preload data\n"
                            "vst1.32  {d18-d19}, [%[dout]]!         @ store result, add pointer\n"
                            "pld [%[din], #128]                     @ preload data\n"
                            "vst1.32  {d20-d21}, [%[dout]]!         @ store result, add pointer\n"
                            "pld [%[din], #256]                     @ preload data\n"
                            "vst1.32  {d22-d23}, [%[dout]]!         @ store result, add pointer\n"
                            "pld [%[din], #384]                     @ preload data\n"

                            "subs %[cnt], #1                        @ loop count minus 1\n"
                            "bne    3b                              @ jump to main loop start point\n"
                    :[dout] "+r"(ptr_out_thread), [din] "+r"(ptr_in_thread), [cnt] "+r"(cnt)
                    :[vzero] "w" (vzero), [vthreshold] "w" (vthreshold)
                    :"q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
                    );
                }
#endif
                for (int j = 0; j < neon_loop_remain; j++) {
                    ptr_out_thread[0] = ptr_in_thread[0] > 0.f ? (ptr_in_thread[0] > coef ? coef : ptr_in_thread[0])  : 0.f;
                    ptr_in_thread++;
                    ptr_out_thread++;
                }
            }
            ptr_out = ptr_out + threads * nums_per_thread;
            ptr_in = ptr_in + threads * nums_per_thread;
            for (int i = 0; i < remain; i++) {
                ptr_out[0] = ptr_in[0] > 0.f ? (ptr_in[0] > coef ? coef : ptr_in[0])  : 0.f;
                ptr_in++;
                ptr_out++;
            }
            break;
        //sigmoid: 1/(exp(-x) + 1)
        case Active_sigmoid:
            #pragma omp parallel for
            for (int i = 0; i < threads; i++) {
                float32x4_t exp_vec = vdupq_n_f32(0.0f);
                float32x4_t recip  = vdupq_n_f32(0.0f);
                const float* ptr_in_thread = ptr_in + i * nums_per_thread;
                float* ptr_out_thread = ptr_out + i * nums_per_thread;
                for (int j = 0; j < neon_loop_cnt_dim4; j++ ) {
                    exp_vec = exp_ps(vnegq_f32(vld1q_f32(ptr_in_thread)));
                    exp_vec = vaddq_f32(exp_vec, vdupq_n_f32(1.0f));
                    recip = vrecpeq_f32(exp_vec);
                    recip = vmulq_f32 (vrecpsq_f32 (exp_vec, recip), recip);
                    recip = vmulq_f32 (vrecpsq_f32 (exp_vec, recip), recip);
                    vst1q_f32(ptr_out_thread, recip);
                    ptr_out_thread += 4;
                    ptr_in_thread += 4;
                }
                for (int j = 0; j < neon_loop_remain_dim4; j++){
                   ptr_out_thread[0] = 1 / (1 + exp(-ptr_in_thread[0]));
                   ptr_in_thread++;
                   ptr_out_thread++;
                }
            }
            ptr_out = ptr_out + threads * nums_per_thread;
            ptr_in = ptr_in + threads * nums_per_thread;
            for (int i = 0; i < remain; i++) {
                ptr_out[0] =  1/(1+exp(-ptr_in[0]));
                ptr_in++;
                ptr_out++;
            }
            break;

        // tanh : (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        case Active_tanh:
            //LOG(INFO) << "Active_tanh";
            #pragma omp parallel for
            for (int i = 0; i < threads; ++i) {
                float32x4_t vtwo = vdupq_n_f32(2.0f);
                float32x4_t vone = vdupq_n_f32(1.0f);
                const float* ptr_in_thread = ptr_in + i * nums_per_thread;
                float* ptr_out_thread = ptr_out + i * nums_per_thread;
                int cnt4 = neon_loop_cnt_dim4;
                int remain4 = size;
                cnt4 = cnt4 < 5 ? cnt4 : 0;
                remain4 = cnt4  == 0 ? remain4 : neon_loop_remain_dim4;
                for (int j = 0; j < cnt4; j++) {
                    float32x4_t vdin = vld1q_f32(ptr_in_thread);
                    float32x4_t vsum = vmulq_f32(vdin, vtwo);
                    float32x4_t vexp_sum = exp_ps(vsum);
                    float32x4_t vadd_sum = vaddq_f32(vexp_sum, vone);
                    float32x4_t vrecip = div_ps(vtwo, vadd_sum);
                    float32x4_t vout = vsubq_f32(vone, vrecip);
                    vst1q_f32(ptr_out_thread, vout);
                    ptr_out_thread += 4;
                    ptr_in_thread += 4;
                }
                for(int j = 0; j < remain4; j++){
                    ptr_out_thread[0] = 1.0 - 2.0 / (1.0 + exp(2.0 * ptr_in_thread[0]));
                    //(exp(ptr_in_thread[0]) - exp(-ptr_in_thread[0])) / (exp(ptr_in_thread[0]) + exp(-ptr_in_thread[0]));
                    ptr_in_thread++;
                    ptr_out_thread++;
                }
            }
            ptr_out = ptr_out + threads * nums_per_thread;
            ptr_in = ptr_in + threads * nums_per_thread;
            for (int j = 0; j < remain; ++j) {
                ptr_out[0] = 1.0 - 2.0 / (1.0 + exp(2.0 * ptr_in[0]));//(exp(ptr_in[0]) - exp(-ptr_in[0])) / (exp(ptr_in[0]) + exp(-ptr_in[0]));
                ptr_in++;
                ptr_out++;
            }
           break;
        
        // stanh : b * \frac{e^{a * x} - e^{-a * x}}{e^{a * x} + e^{-a * x}}
        case Active_stanh:
            #pragma omp parallel for
            for (int i = 0; i < threads; ++i) {
                float32x4_t vcoef = vdupq_n_f32(coef);
                float32x4_t vslope = vdupq_n_f32(slope);
                float32x4_t vtwo = vdupq_n_f32(2.0f);
                float32x4_t vone = vdupq_n_f32(1.0f);
                const float* ptr_in_thread = ptr_in + i * nums_per_thread;
                float* ptr_out_thread = ptr_out + i * nums_per_thread;
                int cnt4 = neon_loop_cnt_dim4;
                int remain4 = size;
                cnt4 = cnt4 < 10 ? cnt4 : 0;
                remain4 = cnt4  == 0 ? remain4 : neon_loop_remain_dim4;
                for (int j = 0; j < cnt4; j++) {
                    float32x4_t vdin = vld1q_f32(ptr_in_thread);
                    float32x4_t vmul_sum = vmulq_f32(vdin, vslope);
                    float32x4_t vsum = vmulq_f32(vmul_sum, vtwo);
                    float32x4_t vexp_sum = exp_ps(vsum);
                    float32x4_t vadd_sum = vaddq_f32(vexp_sum, vone);
                    float32x4_t vrecip = div_ps(vtwo, vadd_sum);
                    float32x4_t vout = vsubq_f32(vone, vrecip);
                    vout = vmulq_f32(vout, vcoef);
                    vst1q_f32(ptr_out_thread, vout);
                    ptr_out_thread += 4;
                    ptr_in_thread += 4;
                }
                for(int j = 0; j < remain4; j++){
                    float din = ptr_in_thread[0] * slope;
                    ptr_out_thread[0] = coef * (1.0 - 2.0 / (1.0 + exp(2.0 * din)));
                    ptr_in_thread++;
                    ptr_out_thread++;
                }
            }
            ptr_out = ptr_out + threads * nums_per_thread;
            ptr_in = ptr_in + threads * nums_per_thread;
            for (int j = 0; j < remain; ++j) {
                float din = ptr_in[0] * slope;
                ptr_out[0] = coef * (1.0 - 2.0 / (1.0 + exp(2.0 * din)));
                ptr_in++;
                ptr_out++;
            }
            break;
        
        //prelu: x > 0 ? x : slope[c] * x
        case Active_prelu:
            slopes_ptr = (float*)param.prelu_param.slope->data();
            for (int n = 0; n < num; n++){
                const float* data_in_batch = ptr_in + n * channel * csize;
                float* data_out_batch = ptr_out + n * channel * csize;
#pragma omp parallel for
                for (int c = 0; c < channel; c++){
                    const float* data_in_channel = data_in_batch + c * csize;
                    float* data_out_channel = data_out_batch + c * csize;
                    float slope_val = channel_shared ? slopes_ptr[0] : slopes_ptr[c];
                    float32x4_t vzero = vdupq_n_f32(0.f);
                    float32x4_t vslope = vdupq_n_f32(slope_val);
                    int dim4 = csize >> 2;
                    int dim4_remain = csize - (dim4 * 4);
#ifdef __aarch64__
                    for (int i = 0; i < dim4; i++){
                        float32x4_t vr0 = vld1q_f32(data_in_channel);
                        uint32x4_t vmask = vcltq_f32(vr0, vzero);//vr0 <= vzero
                        float32x4_t vout = vmulq_f32(vr0, vslope);//vr0 * vslope
                        float32x4_t vout_sel = vbslq_f32(vmask, vout, vr0);
                        vst1q_f32(data_out_channel, vout_sel);
                        data_in_channel += 4;
                        data_out_channel += 4;
                    }
#else
                    int cnt = dim4;
                    if (dim4 > 0){
                        asm volatile(
                                "2:                                            @main loop\n"
                                     "vld1.f32   {d0-d1}, [%[ptr_in]]!              @load q1\n"
                                     "vclt.f32   q1, q0, %q[vzero]                   @vcle q0 <= vzero\n"
                                     "vmul.f32   q2, q0, %q[vslope]                  @vmul q0 * vslope\n"
                                     "vbit.32    q0, q2, q1                          @vbit q0, q2, q1\n"
                                     "subs       %[cnt], #1                          @subs nn, 1\n"
                                     "vst1.f32   {d0-d1}, [%[ptr_out]]!                 @store data\n"
                                     "bne        2b                                   @bne nn\n"
                                     :[ptr_in] "+r" (data_in_channel), [cnt] "+r" (cnt), \
                                     [ptr_out] "+r" (data_out_channel)
                                     :[vzero] "w" (vzero), [vslope] "w" (vslope)
                                     :"q0", "q1", "q2"
                                     );
                    }
#endif //__aarch64__
                    for (int i = 0 ; i < dim4_remain ; i++) {
                        data_out_channel[0] = data_in_channel[0] > 0 ? data_in_channel[0] : data_in_channel[0] * slope_val;
                        data_in_channel++;
                        data_out_channel++;
                    }
                }
            }
            break;
        
        //elu:  x > 0 ? x : coef * (exp(x) - 1)
        case Active_elu:
            #pragma omp parallel for
            for (int i = 0; i < threads; ++i) {
                const float* ptr_in_thread = ptr_in + i * nums_per_thread;
                float* ptr_out_thread = ptr_out + i * nums_per_thread;
                int cnt = neon_loop_cnt;
                float32x4_t vone = vdupq_n_f32(1.0f);
                float32x4_t vcoef = vdupq_n_f32(coef);
                for (int num = 0; num < neon_loop_cnt; num++){
                    float32x4_t vr0 = vld1q_f32(ptr_in_thread);
                   // ptr_in_thread+=4;
                    float32x4_t vr1 = vld1q_f32(ptr_in_thread + 4);
                   // ptr_in_thread+=4;
                    float32x4_t vr2 = vld1q_f32(ptr_in_thread + 8);
                   // ptr_in_thread+=4;
                    float32x4_t vr3 = vld1q_f32(ptr_in_thread + 12);
                    //ptr_in_thread+=4;
                    ptr_in_thread += 16;

                    float32x4_t vsum0 = exp_ps(vr0);
                    float32x4_t vsum1 = exp_ps(vr1);
                    float32x4_t vsum2 = exp_ps(vr2);
                    float32x4_t vsum3 = exp_ps(vr3);
                    uint32x4_t vmask0 = vcgeq_f32(vr0, vzero);
                    uint32x4_t vmask1 = vcgeq_f32(vr1, vzero);
                    uint32x4_t vmask2 = vcgeq_f32(vr2, vzero);
                    uint32x4_t vmask3 = vcgeq_f32(vr3, vzero);
                    vsum0 = vsubq_f32(vsum0, vone);
                    vsum1 = vsubq_f32(vsum1, vone);
                    vsum2 = vsubq_f32(vsum2, vone);
                    vsum3 = vsubq_f32(vsum3, vone);

                    vsum0 = vmulq_f32(vsum0, vcoef);
                    vsum1 = vmulq_f32(vsum1, vcoef);
                    vsum2 = vmulq_f32(vsum2, vcoef);
                    vsum3 = vmulq_f32(vsum3, vcoef);


                    
                    float32x4_t vout0 =vbslq_f32(vmask0, vr0, vsum0);
                    float32x4_t vout1 =vbslq_f32(vmask1,  vr1, vsum1);
                    float32x4_t vout2 =vbslq_f32(vmask2,  vr2, vsum2);
                    float32x4_t vout3 =vbslq_f32(vmask3,  vr3, vsum3);

                    vst1q_f32(ptr_out_thread, vout0);
                    //ptr_out_thread+=4;
                    vst1q_f32(ptr_out_thread + 4, vout1);
                   // ptr_out_thread+=4;
                    vst1q_f32(ptr_out_thread + 8, vout2);
                   // ptr_out_thread+=4;
                    vst1q_f32(ptr_out_thread + 12, vout3);
                    //ptr_out_thread+=4;
                    ptr_out_thread += 16;
                }      

                for (int j = 0; j < neon_loop_remain; j++) {
                    ptr_out_thread[0] = ptr_in_thread[0] > 0.f ? ptr_in_thread[0] : coef * (exp(ptr_in_thread[0]) - 1);
                    ptr_in_thread++;
                    ptr_out_thread++;
                }
            }
            ptr_out = ptr_out + threads * nums_per_thread;
            ptr_in = ptr_in + threads * nums_per_thread;
            for (int i = 0; i < remain; i++) {
                ptr_out[0] = ptr_in[0] > 0.f ? ptr_in[0] : coef * (exp(ptr_in[0]) - 1);
                ptr_in++;
                ptr_out++;
            }
            break;
        default:
            return SaberUnKownError;
    }
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberActivation, ActivationParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberActivation, ActivationParam, ARM, AK_INT8);
}
} // namespace anakin
