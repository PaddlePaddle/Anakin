#include "saber/lite/funcs/saber_activation.h"
#include "saber/lite/funcs/neon/impl/neon_mathfun.h"
namespace anakin{

namespace saber{

namespace lite{

//SaberActivation::SaberActivation(ActiveType type, float neg_slop) {
//    _type = type;
//    _neg_slop = neg_slop;
//}

SaberActivation::SaberActivation(const ParamBase *param) {
    _param = (ActivationParam*)param;
    this->_flag_param = true;
}

//SaberStatus SaberActivation::load_param(ActiveType type, float neg_slop) {
//    _type = type;
//    _neg_slop = neg_slop;
//    return SaberSuccess;
//}

SaberStatus SaberActivation::load_param(const ParamBase *param) {
    _param = (ActivationParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberActivation::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                                  std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {
    if (!this->_flag_param) {
        printf("load activation param first\n");
        return SaberNotInitialized;
    }
    outputs[0]->set_shape(inputs[0]->valid_shape());
    return SaberSuccess;
}

SaberStatus SaberActivation::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                  std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {
    if (!this->_flag_param) {
        printf("load activation param first\n");
        return SaberNotInitialized;
    }
    this->_ctx = &ctx;
    this->_flag_init = true;
    return SaberSuccess;
}

SaberStatus SaberActivation::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                      std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_init) {
        printf("init activation first\n");
        return SaberNotInitialized;
    }
    float* ptr_out = outputs[0]->mutable_data();
    const float* ptr_in = inputs[0]->data();
    int size = inputs[0]->valid_size();
    int threads = 1;
    this->_ctx->get_mode(threads);
    int nums_per_thread = size / threads;
    int remain = size - threads * nums_per_thread;
    int neon_loop_cnt = nums_per_thread >> 4;
    int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
    int neon_loop_cnt_dim4 = nums_per_thread >> 2;
    int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);
    float32x4_t vzero = vdupq_n_f32(0.f);
    switch (_param->_act_type){
        case Active_relu:
            #pragma omp parallel for
            for (int i = 0; i < threads; ++i) {
                const float* ptr_in_thread = ptr_in + i * nums_per_thread;
                float* ptr_out_thread = ptr_out + i * nums_per_thread;
                int cnt = neon_loop_cnt;
#ifdef __aarch64__
                for(int num=0;num<neon_loop_cnt;++num){
                    float32x4_t vr0 = vld1q_f32(ptr_in_thread);
                    ptr_in_thread+=4;
                    float32x4_t vr1 = vld1q_f32(ptr_in_thread);
                    ptr_in_thread+=4;
                    float32x4_t vr2 = vld1q_f32(ptr_in_thread);
                    ptr_in_thread+=4;
                    float32x4_t vr3 = vld1q_f32(ptr_in_thread);
                    ptr_in_thread+=4;
                    vr0=vmaxq_f32(vr0,vzero);
                    vr1=vmaxq_f32(vr1,vzero);
                    vr2=vmaxq_f32(vr2,vzero);
                    vr3=vmaxq_f32(vr3,vzero);
                    vst1q_f32(ptr_out_thread,vr0);
                    ptr_out_thread+=4;
                    vst1q_f32(ptr_out_thread,vr1);
                    ptr_out_thread+=4;
                    vst1q_f32(ptr_out_thread,vr2);
                    ptr_out_thread+=4;
                    vst1q_f32(ptr_out_thread,vr3);
                    ptr_out_thread+=4;
                }
                
#else
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
#endif
                for (int j = 0; j < neon_loop_remain; ++j) {
                    ptr_out_thread[0] = ptr_in_thread[0] > 0.f? ptr_in_thread[0] : 0.f;
                    ptr_in_thread++;
                    ptr_out_thread++;
                }
            }
            ptr_out = ptr_out+threads * nums_per_thread;
            ptr_in = ptr_in+threads * nums_per_thread;
            for (int j = 0; j < remain; ++j) {
                ptr_out[0] = ptr_in[0] > 0.f? ptr_in[0] : 0.f;
                ptr_in++;
                ptr_out++;
            }
            return SaberSuccess;
            
            
        case Active_sigmoid:
            #pragma omp parallel for
            for (int i = 0; i < threads; ++i) {
                float32x4_t exp_vec = vdupq_n_f32(0.0f);
                float32x4_t recip  = vdupq_n_f32(0.0f);
                const float* ptr_in_thread = ptr_in + i * nums_per_thread;
                float* ptr_out_thread = ptr_out + i * nums_per_thread;
                for (int k=0; k<neon_loop_cnt_dim4; ++k ) {
                    exp_vec=exp_ps(vnegq_f32(vld1q_f32(ptr_in_thread)));
                    exp_vec = vaddq_f32(exp_vec, vdupq_n_f32(1.0f));
                    recip = vrecpeq_f32(exp_vec);
                    recip = vmulq_f32 (vrecpsq_f32 (exp_vec, recip), recip);
                    recip = vmulq_f32 (vrecpsq_f32 (exp_vec, recip), recip);
                    vst1q_f32(ptr_out_thread, recip);
                    ptr_out_thread+=4;
                    ptr_in_thread+=4;
                }
                for(int j=0;j<neon_loop_remain_dim4;++j){
                   ptr_out_thread[0]=1/(1+exp(-ptr_in_thread[0]));
                    ptr_in_thread++;
                    ptr_out_thread++;
                }
            }
            ptr_out = ptr_out+threads * nums_per_thread;
            ptr_in = ptr_in+threads * nums_per_thread;
            for (int j = 0; j < remain; ++j) {
                ptr_out[0] =  1/(1+exp(-ptr_in[0]));
                ptr_in++;
                ptr_out++;
            }

            return SaberSuccess;
        case Active_tanh:
            #pragma omp parallel for
            for (int i = 0; i < threads; ++i) {
                float32x4_t exp_plus_vec = vdupq_n_f32(0.0f);
                float32x4_t exp_minus_vec = vdupq_n_f32(0.0f);
                float32x4_t exp_sum_vec = vdupq_n_f32(0.0f);
                float32x4_t exp_diff_vec = vdupq_n_f32(0.0f);
                float32x4_t recip  = vdupq_n_f32(0.0f);
                const float* ptr_in_thread = ptr_in + i * nums_per_thread;
                float* ptr_out_thread = ptr_out + i * nums_per_thread;
                for (int k=0; k<neon_loop_cnt_dim4; ++k ) {
                    exp_plus_vec=exp_ps(vld1q_f32(ptr_in_thread));
                    exp_minus_vec=exp_ps(vnegq_f32(vld1q_f32(ptr_in_thread)));
                    exp_sum_vec=vaddq_f32(exp_plus_vec,exp_minus_vec);
                    exp_diff_vec=vsubq_f32(exp_plus_vec,exp_minus_vec);
                    recip = div_ps(exp_diff_vec,exp_sum_vec);
                    vst1q_f32(ptr_out_thread, recip);
                    ptr_out_thread+=4;
                    ptr_in_thread+=4;
                }
                for(int j=0;j<neon_loop_remain_dim4;++j){
                    ptr_out_thread[0]=(exp(ptr_in_thread[0])-exp(-ptr_in_thread[0]))/(exp(ptr_in_thread[0])+exp(-ptr_in_thread[0]));
                    ptr_in_thread++;
                    ptr_out_thread++;
                }
            }
            ptr_out = ptr_out+threads * nums_per_thread;
            ptr_in = ptr_in+threads * nums_per_thread;
            for (int j = 0; j < remain; ++j) {
                ptr_out[0] = (exp(ptr_in[0])-exp(-ptr_in[0]))/(exp(ptr_in[0])+exp(-ptr_in[0]));
                ptr_in++;
                ptr_out++;
            }
            return SaberSuccess;
        default:
            return SaberUnKownError;
    }
}

} //namespace lite

} //namespace saber

} //namespace anakin
