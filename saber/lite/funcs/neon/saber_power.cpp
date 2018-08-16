#include "saber/lite/funcs/saber_power.h"
#include "saber/lite/funcs/neon/impl/neon_mathfun.h"
#include "saber/lite/net/saber_factory_lite.h"
#ifdef USE_ARM_PLACE

namespace anakin {

namespace saber {

namespace lite{

SaberPower::SaberPower(const ParamBase *param) {
    _param = (const PowerParam*)param;
    this->_flag_param = true;
}

SaberStatus SaberPower::load_param(const ParamBase *param) {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
        this->_flag_create_param = false;
    }
    _param = (const PowerParam*)param;
    this->_flag_param = true;
    return SaberSuccess;
}

SaberPower::~SaberPower() {
    if (this->_flag_create_param) {
        delete _param;
        _param = nullptr;
    }
}

SaberStatus SaberPower::load_param(std::istream &stream, const float *weights) {
    float scale;
    float shift;
    float power;
    stream >> scale >> shift >> power;
    _param = new PowerParam(scale, shift, power);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#if 0
SaberStatus SaberPower::load_param(FILE *fp, const float *weights) {
    float scale;
    float shift;
    float power;
    fscanf(fp, "%f,%f,%f\n", &scale, &shift, &power);
    _param = new PowerParam(scale, shift, power);
    this->_flag_create_param = true;
    this->_flag_param = true;
    return SaberSuccess;
}
#endif
SaberStatus SaberPower::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_param) {
        printf("load power param first\n");
        return SaberNotInitialized;
    }

    outputs[0]->set_shape(inputs[0]->valid_shape());

    return SaberSuccess;
}

//template <>
SaberStatus SaberPower::init(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                               std::vector<Tensor<CPU, AK_FLOAT> *> &outputs, Context &ctx) {
    if (!this->_flag_param) {
        printf("load power param first\n");
        return SaberNotInitialized;
    }

    this->_ctx = &ctx;

    this->_flag_init = true;

    return SaberSuccess;
}

//template <>
SaberStatus SaberPower::dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *> &inputs,
                                   std::vector<Tensor<CPU, AK_FLOAT> *> &outputs) {

    if (!this->_flag_init) {
        printf("init power first\n");
        return SaberNotInitialized;
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start();
#endif

    float scale=_param->_scale;
    float shift=_param->_shift;
    float power=_param->_power;
    bool _do_power = true;
    bool _do_scale = true;
    bool _do_shift = true;
    if (fabsf(power - 1.f) < 1e-6f) {
        _do_power = false;
    }
    if (fabsf(scale - 1.f) < 1e-6f) {
        _do_scale = false;
    }
    if (fabsf(shift - 0.f) < 1e-6f) {
        bool _do_shift = false;
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
    float32x4_t vscale = vdupq_n_f32(scale);
    float32x4_t vshift=vdupq_n_f32(shift);
    float32x4_t vpower=vdupq_n_f32(power);
#pragma omp parallel for
    for (int i = 0; i < threads; ++i) {
        const float* ptr_in_thread = ptr_in + i * nums_per_thread;
        float* ptr_out_thread = ptr_out + i * nums_per_thread;
        int cnt = neon_loop_cnt;
        for(int num=0;num<neon_loop_cnt;++num){
            float32x4_t vr0 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vr1 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vr2 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            float32x4_t vr3 = vld1q_f32(ptr_in_thread);
            ptr_in_thread+=4;
            if(_do_scale){
                vr0=vmulq_f32(vr0,vscale);
                vr1=vmulq_f32(vr1,vscale);
                vr2=vmulq_f32(vr2,vscale);
                vr3=vmulq_f32(vr3,vscale);
            }
            if(_do_shift){
                vr0=vaddq_f32(vr0,vshift);
                vr1=vaddq_f32(vr1,vshift);
                vr2=vaddq_f32(vr2,vshift);
                vr3=vaddq_f32(vr3,vshift);
            }
            if(_do_power){
                vr0=pow_ps(vr0,vpower);
                vr1=pow_ps(vr1,vpower);
                vr2=pow_ps(vr2,vpower);
                vr3=pow_ps(vr3,vpower);
            }
            vst1q_f32(ptr_out_thread,vr0);
            ptr_out_thread+=4;
            vst1q_f32(ptr_out_thread,vr1);
            ptr_out_thread+=4;
            vst1q_f32(ptr_out_thread,vr2);
            ptr_out_thread+=4;
            vst1q_f32(ptr_out_thread,vr3);
            ptr_out_thread+=4;
        }
        for (int j = 0; j < neon_loop_remain; ++j) {
            ptr_out_thread[0] = std::pow((ptr_in_thread[0]*scale+shift),power);
            ptr_in_thread++;
            ptr_out_thread++;
        }
    }
    ptr_out = ptr_out+threads * nums_per_thread;
    ptr_in = ptr_in+threads * nums_per_thread;
    for (int j = 0; j < remain; ++j) {
        ptr_out[0] = std::pow((ptr_in[0]*scale+shift),power);
        ptr_in++;
        ptr_out++;
    }

#ifdef ENABLE_OP_TIMER
    this->_timer.end();
    float ts = this->_timer.get_average_ms();
    printf("power %s: time: %f\n", this->_op_name.c_str(), ts);
    OpTimer::add_timer("power", ts);
    OpTimer::add_timer("total", ts);
#endif

    return SaberSuccess;
}
REGISTER_LAYER_CLASS(SaberPower);
} //namespace lite

} //namespace saber

} // namespace anakin

#endif //USE_ARM_PLACE
