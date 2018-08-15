#include "saber/lite/funcs/saber_activation.h"
#include "saber/lite/funcs/neon/impl/neon_mathfun.h"
#include "saber/lite/net/saber_factory_lite.h"
namespace anakin{

namespace saber{

namespace lite{

void act_relu(const float* din, float* dout, int n, int c, int h, int w, const ActivationParam* _param, int threads) {
    int size = n * c * h * w;
    int nums_per_thread = size / threads;
    int remain = size - threads * nums_per_thread;
    int neon_loop_cnt = nums_per_thread >> 4;
    int neon_loop_remain = nums_per_thread - (neon_loop_cnt << 4);
    float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        const float* ptr_in_thread = din + i * nums_per_thread;
        float* ptr_out_thread = dout + i * nums_per_thread;
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
        for (int j = 0; j < neon_loop_remain; ++j) {
            ptr_out_thread[0] = ptr_in_thread[0] > 0.f? ptr_in_thread[0] : 0.f;
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    float* out_ptr_remain = dout + threads * nums_per_thread;
    const float* in_ptr_remain = din + threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        out_ptr_remain[0] = in_ptr_remain[0] > 0.f? in_ptr_remain[0] : 0.f;
        in_ptr_remain++;
        out_ptr_remain++;
    }
}

void act_prelu(const float* din, float* dout, int num, int channel, int h, int w, const ActivationParam* _param, int threads) {
    int csize = w * h;
    for (int n = 0; n < num; n++){
        const float* data_in_batch = din + n * channel * csize;
        float* data_out_batch = dout + n * channel * csize;
#pragma omp parallel for
        for (int c = 0; c < channel; c++){
            const float* data_in_channel = data_in_batch + c * csize;
            float* data_out_channel = data_out_batch + c * csize;
            float slope = _param->_prelu_channel_shared ? _param->_prelu_weights[0] : _param->_prelu_weights[c];

            float32x4_t vzero = vdupq_n_f32(0.f);
            float32x4_t vslope = vdupq_n_f32(slope);
            int dim4 = csize >> 2;
            int dim4_remain = csize - (dim4 * 4);
#ifdef __aarch64__
            for (int i=0;i< dim4; ++i){
                float32x4_t vr0 = vld1q_f32(data_in_channel);
                uint32x4_t vmask = vcltq_f32(vr0, vzero);//vr0 <= vzero
                float32x4_t vout = vmulq_f32(vr0, vslope);//vr0 * vslope
                float32x4_t vout_sel = vbslq_f32(vmask, vout, vr0);
                vst1q_f32(data_out_channel, vout_sel);
                data_in_channel += 4;
                data_out_channel += 4;
            }
#else
            if (dim4 > 0){
                        asm volatile(
                                     "prelu_loop:                                    @main loop\n"
                                     "vld1.f32   {d0-d1}, [%[ptr_in]]!              @load q1\n"
                                     "vclt.f32   q1, q0, %q[vzero]                   @vcle q0 <= vzero\n"
                                     "vmul.f32   q2, q0, %q[vslope]                  @vmul q0 * vslope\n"
                                     "vbit.32    q0, q2, q1                          @vbit q0, q2, q1\n"
                                     "subs       %[cnt], #1                          @subs nn, 1\n"
                                     "vst1.f32   {d0-d1}, [%[dout]]!                 @store data\n"
                                     "bne        prelu_loop                          @bne nn\n"
                                     :[ptr_in] "+r" (data_in_channel), [cnt] "+r" (dim4), \
                                     [dout] "+r" (data_out_channel)
                                     :[vzero] "w" (vzero), [vslope] "w" (vslope)
                                     :"q0", "q1", "q2"
                                     );
                    }
#endif //__aarch64__
            for (int i=0 ; i < dim4_remain ; ++i) {
                if (data_in_channel[0] < 0){
                    data_out_channel[0] = data_in_channel[0] * slope;
                } else {
                    data_out_channel[0] = data_in_channel[0];
                }
                data_in_channel++;
                data_out_channel++;
            }
        }
    }
}

void act_sigmoid(const float* din, float* dout, int n, int c, int h, int w, const ActivationParam* _param, int threads) {
    int size = n * c * h * w;
    int nums_per_thread = size / threads;
    int remain = size - threads * nums_per_thread;
    int neon_loop_cnt_dim4 = nums_per_thread >> 2;
    int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);

    float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        float32x4_t exp_vec = vdupq_n_f32(0.0f);
        float32x4_t recip  = vdupq_n_f32(0.0f);
        const float* ptr_in_thread = din + i * nums_per_thread;
        float* ptr_out_thread = dout + i * nums_per_thread;
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
            ptr_out_thread[0] = 1.f / ( 1 + expf(-ptr_in_thread[0]));
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    float* ptr_out = dout + threads * nums_per_thread;
    const float* ptr_in = din + threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        ptr_out[0] =  1.f / (1 + expf(-ptr_in[0]));
        ptr_in++;
        ptr_out++;
    }
}


void act_tanh(const float* din, float* dout, int n, int c, int h, int w, const ActivationParam* _param, int threads) {
    int size = n * c * h * w;
    int nums_per_thread = size / threads;
    int remain = size - threads * nums_per_thread;
    int neon_loop_cnt_dim4 = nums_per_thread >> 2;
    int neon_loop_remain_dim4 = nums_per_thread - (neon_loop_cnt_dim4 << 2);
#pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        float32x4_t exp_plus_vec = vdupq_n_f32(0.0f);
        float32x4_t exp_minus_vec = vdupq_n_f32(0.0f);
        float32x4_t exp_sum_vec = vdupq_n_f32(0.0f);
        float32x4_t exp_diff_vec = vdupq_n_f32(0.0f);
        float32x4_t recip  = vdupq_n_f32(0.0f);
        const float* ptr_in_thread = din + i * nums_per_thread;
        float* ptr_out_thread = dout + i * nums_per_thread;
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
            ptr_out_thread[0]=(expf(ptr_in_thread[0]) - expf(-ptr_in_thread[0])) / (expf(ptr_in_thread[0]) + expf(-ptr_in_thread[0]));
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    float* ptr_out = dout + threads * nums_per_thread;
    const float* ptr_in = din + threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        ptr_out[0] = (expf(ptr_in[0]) - expf(-ptr_in[0])) / (expf(ptr_in[0]) + expf(-ptr_in[0]));
        ptr_in++;
        ptr_out++;
    }
}

SaberActivation::SaberActivation(const ParamBase *param) {
    _param = (ActivationParam*)param;
    this->_flag_param = true;
}

SaberActivation::~SaberActivation() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberActivation::load_param(const ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (ActivationParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberStatus SaberActivation::load_param(std::istream &stream, const float *weights) {
    int type;
    float neg_slop;
    float coef;
    int channel_share;
    int w_offset;
    stream >> type >> neg_slop >> coef >> channel_share >> w_offset;
    ActiveType atype = static_cast<ActiveType>(type);
    _param = new ActivationParam(atype, neg_slop, coef, channel_share > 0, weights + w_offset);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#if 0
SaberStatus SaberActivation::load_param(FILE *fp, const float *weights) {
    int type;
    float neg_slop;
    float coef;
    int channel_share;
    int w_offset;
    fscanf(fp, "%d %f %f %d %d\n", &type, &neg_slop, &coef, &channel_share, &w_offset);
    ActiveType atype = static_cast<ActiveType>(type);
    _param = new ActivationParam(atype, neg_slop, coef, channel_share > 0, weights + w_offset);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#endif
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

    switch (this->_param->_act_type) {
        case Active_relu:
            _impl = act_relu;
            break;
        case Active_sigmoid:
            _impl = act_sigmoid;
            break;
        case Active_tanh:
            _impl = act_tanh;
            break;
        case Active_prelu:
            _impl = act_prelu;
            break;
        default:
            return SaberUnImplError;
    }

    this->_flag_init = true;
    return SaberSuccess;
}

SaberStatus SaberActivation::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                      std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_init) {
        printf("init activation first\n");
        return SaberNotInitialized;
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    float* ptr_out = outputs[0]->mutable_data();
    const float* ptr_in = inputs[0]->data();
    int threads = 1;
    this->_ctx->get_mode(threads);

    _impl(ptr_in, ptr_out, num, channel, height, width, this->_param, threads);

#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    printf("activation: %d, time: %f\n", (int)this->_param->_act_type, ts);
    OpTimer::add_timer("activation", ts);
    OpTimer::add_timer("total", ts);
#endif
    return SaberSuccess;
}

REGISTER_LAYER_CLASS(SaberActivation);

} //namespace lite

} //namespace saber

} //namespace anakin
