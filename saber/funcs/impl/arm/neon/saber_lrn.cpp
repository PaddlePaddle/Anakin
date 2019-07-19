#include "saber/funcs/impl/arm/saber_lrn.h"
#include "saber/funcs/impl/arm/neon/impl/neon_mathfun.h"

namespace anakin{

namespace saber{
void compute_across_channels(const float* din, float* dout, int num, int channel, int h, int w, \
    int pre_pad, int post_pad, float alpha, float beta, float k){
    int channel_size = h * w;
    int cnt = channel_size / 4;
    int remain = channel_size % 4;
    float32x4_t k_val = vdupq_n_f32(k);
    float32x4_t alpha_val = vdupq_n_f32(alpha);
    float32x4_t beta_val = vdupq_n_f32(-beta);
    for (int n = 0; n < num; n++){
        const float* din_ptr = din + n * channel * channel_size;
        float* dout_ptr = dout + n * channel * channel_size;
        for (int c = 0; c < channel; c++){
            const float* din_ch_ptr = din_ptr + c * channel_size;
            float* dout_ch_ptr = dout_ptr + c * channel_size;
            int cs = (c - pre_pad) < 0 ? 0: (c - pre_pad);
            int ce = (c + post_pad) >= channel? channel: (c + pre_pad + 1);
            for (int i = 0; i < cnt; i++){
                int idx = i * 4;
                float32x4_t sum = vdupq_n_f32(0.f);
                float32x4_t din = vld1q_f32(din_ch_ptr);
                for (int k = cs; k < ce; k++){
                    float32x4_t v0 = vld1q_f32(&din_ptr[k * channel_size + idx]);
                    sum = vmlaq_f32(sum, v0, v0);
                }
                sum = vmulq_f32(sum, alpha_val);
                sum = vaddq_f32(sum, k_val);
                float32x4_t res0 = pow_ps(sum, beta_val);
                float32x4_t res1 = vmulq_f32(din, res0);
                vst1q_f32(dout_ch_ptr, res1);
                dout_ch_ptr += 4;
                din_ch_ptr += 4;
            }
            int idx = cnt * 4;
            for (int i = 0; i < remain; i++){
                float sum = 0.0;
                for (int k = cs; k < ce; k++){
                    sum += din_ptr[k * channel_size + idx] * din_ptr[k * channel_size + idx];
                }
                sum = k + sum * alpha;
                dout_ch_ptr[0] = din_ch_ptr[0] * pow(sum, -beta);
                dout_ch_ptr++;
                din_ch_ptr++;
                idx++;
            }
        }
    }
}

void compute_within_channels(const float* din, float* dout, int num, int channel, int h, int w, \
    int pre_pad, int post_pad, float alpha, float beta, float k){
    printf("iit does not implement \n");
}

template <>
SaberStatus SaberLrn<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        LrnParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif

    int num = inputs[0]->num();
    int ch_in = inputs[0]->channel();
    int w_in = inputs[0]->width();
    int h_in = inputs[0]->height();
    const float* din_ptr = static_cast<const float*>(inputs[0]->data());
    float* dout_ptr = static_cast<float*>(outputs[0]->mutable_data());

    if (param.norm_region == ACROSS_CHANNELS){
        compute_across_channels(din_ptr, dout_ptr, num, ch_in, h_in, w_in, _pre_pad, _post_pad, \
            param.alpha, param.beta, param.k);
    }else if (param.norm_region == WITHIN_CHANNEL){
        compute_within_channels(din_ptr, dout_ptr, num, ch_in, h_in, w_in, _pre_pad, _post_pad, \
            param.alpha, param.beta, param.k);
    }else{
        LOG(ERROR) << "ERROR: Other Lrn norm_region should be replace by other ops.\n";
        return SaberNotInitialized;
    }

#ifdef ENABLE_OP_TIMER
   this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Lrn : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("lrn", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberLrn, LrnParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberLrn, LrnParam, ARM, AK_INT8);
//template class SaberLrn<ARM, AK::FLOAT>;

} //namespace anakin

} //namespace anakin
