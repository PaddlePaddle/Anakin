#include "saber/funcs/impl/arm/saber_scale.h"

namespace anakin{

namespace saber{
void scale_compute_kernel(const float* din, float* dout, int outer_dim, int scale_dim, int inner_dim, \
     bool bias_flag, const float* scale_data, const float* bias_data){
    int cnt = inner_dim >> 4;
    int remain = inner_dim % 16;
    int size = inner_dim * scale_dim;
    for (int n = 0; n < outer_dim; n++){
        const float* din_ptr_n  = din + n * size;
        float* dout_ptr_n = dout + n * size;
#pragma omp parallel for
        for (int i = 0; i < scale_dim; i++){
            const float* din_ptr  = din_ptr_n + i * inner_dim;
            float* dout_ptr = dout_ptr_n + i * inner_dim;
            float32x4_t vscale = vdupq_n_f32(scale_data[i]);
            float bias = bias_flag ? bias_data[i] : 0.f;
            float32x4_t vbias = vdupq_n_f32(bias);
            for (int j = 0; j < cnt; j++){
                float32x4_t din0 = vld1q_f32(din_ptr);
                float32x4_t din1 = vld1q_f32(din_ptr + 4);
                float32x4_t din2 = vld1q_f32(din_ptr + 8);
                float32x4_t din3 = vld1q_f32(din_ptr + 12);

                float32x4_t vsum1 = vmlaq_f32(vbias, din0, vscale);
                float32x4_t vsum2 = vmlaq_f32(vbias, din1, vscale);
                float32x4_t vsum3 = vmlaq_f32(vbias, din2, vscale);
                float32x4_t vsum4 = vmlaq_f32(vbias, din3, vscale);

                din_ptr += 16;
                vst1q_f32(dout_ptr, vsum1);
                vst1q_f32(dout_ptr + 4, vsum2);
                vst1q_f32(dout_ptr + 8, vsum3);
                vst1q_f32(dout_ptr + 12, vsum4);

                dout_ptr += 16;
            }
            for (int j = 0; j < remain; j++){
                *dout_ptr = *din_ptr * scale_data[i] + bias;
                dout_ptr++;
                din_ptr++;
            }
        }
    }
}
void scale_compute_kernel_one_bias(const float* din, float* dout, int outer_dim, int scale_dim, int inner_dim, \
    const float* scale_data, const float* bias_data){
    int cnt = scale_dim >> 4;
    int remain = scale_dim % 16;
    for (int n = 0; n < outer_dim; n++){
        const float* din_ptr_n  = din + n * scale_dim;
        float* dout_ptr_n = dout + n * scale_dim;
#pragma omp parallel for
        for (int i = 0; i < cnt; i++){
            int tmp = i << 4;
            const float* din_ptr  = din_ptr_n + tmp;
            const float* scale_ptr = scale_data + tmp;
            const float* bias_ptr = bias_data + tmp;
            float* dout_ptr = dout_ptr_n + tmp;

            float32x4_t din0 = vld1q_f32(din_ptr);
            float32x4_t vscale0 = vld1q_f32(scale_ptr);
            float32x4_t vbias0 = vld1q_f32(bias_ptr);

            float32x4_t din1 = vld1q_f32(din_ptr + 4);
            float32x4_t vscale1 = vld1q_f32(scale_ptr + 4);
            float32x4_t vbias1 = vld1q_f32(bias_ptr + 4);

            float32x4_t din2 = vld1q_f32(din_ptr + 8);
            float32x4_t vscale2 = vld1q_f32(scale_ptr + 8);
            float32x4_t vbias2 = vld1q_f32(bias_ptr + 8);

            float32x4_t vsum1 = vmlaq_f32(vbias0, din0, vscale0);
            float32x4_t vsum2 = vmlaq_f32(vbias1, din1, vscale1);

            float32x4_t din3 = vld1q_f32(din_ptr + 12);
            float32x4_t vscale3 = vld1q_f32(scale_ptr + 12);
            float32x4_t vbias3 = vld1q_f32(bias_ptr + 12);

            vst1q_f32(dout_ptr, vsum1);
            vst1q_f32(dout_ptr + 4, vsum2);

            float32x4_t vsum3 = vmlaq_f32(vbias2, din2, vscale2);
            float32x4_t vsum4 = vmlaq_f32(vbias3, din3, vscale3);

            vst1q_f32(dout_ptr + 8, vsum3);
            vst1q_f32(dout_ptr + 12, vsum4);
        }
        int tmp = cnt << 4;
        const float* din_ptr  = din_ptr_n + tmp;
        float* dout_ptr = dout_ptr_n + tmp;
        float* scale_ptr = scale_data + tmp;
        const float* bias_ptr = bias_data + tmp;
        for (int j = 0; j < remain; j++){
            *dout_ptr = *din_ptr * (*scale_ptr) + (*bias_ptr);
            dout_ptr++;
            din_ptr++;
            scale_ptr++;
            bias_ptr++;
        }
    }
}
void scale_compute_kernel_one(const float* din, float* dout, int outer_dim, int scale_dim, int inner_dim, \
    const float* scale_data, const float* bias_data){
    int cnt = scale_dim >> 4;
    int remain = scale_dim % 16;
    for (int n = 0; n < outer_dim; n++){
        const float* din_ptr_n  = din + n * scale_dim;
        float* dout_ptr_n = dout + n * scale_dim;
#pragma omp parallel for
        for (int i = 0; i < cnt; i++){
            int tmp = i << 4;
            const float* din_ptr  = din_ptr_n + tmp;
            const float* scale_ptr = scale_data + tmp;
            float* dout_ptr = dout_ptr_n + tmp;

            float32x4_t din0 = vld1q_f32(din_ptr);
            float32x4_t vscale0 = vld1q_f32(scale_ptr);
            float32x4_t din1 = vld1q_f32(din_ptr + 4);
            float32x4_t vscale1 = vld1q_f32(scale_ptr + 4);
            float32x4_t din2 = vld1q_f32(din_ptr + 8);
            float32x4_t vscale2 = vld1q_f32(scale_ptr + 8);
            float32x4_t din3 = vld1q_f32(din_ptr + 12);
            float32x4_t vscale3 = vld1q_f32(scale_ptr + 12);

            float32x4_t vsum1 = vmulq_f32(din0, vscale0);
            float32x4_t vsum2 = vmulq_f32(din1, vscale1);
            float32x4_t vsum3 = vmulq_f32(din2, vscale2);
            float32x4_t vsum4 = vmulq_f32(din3, vscale3);

            vst1q_f32(dout_ptr, vsum1);
            vst1q_f32(dout_ptr + 4, vsum2);
            vst1q_f32(dout_ptr + 8, vsum3);
            vst1q_f32(dout_ptr + 12, vsum4);
        }
        int tmp = cnt << 4;
        const float* din_ptr  = din_ptr_n + tmp;
        float* dout_ptr = dout_ptr_n + tmp;
        float* scale_ptr = scale_data + tmp;
        for (int j = 0; j < remain; j++){
            *dout_ptr = *din_ptr * (*scale_ptr);
            dout_ptr++;
            din_ptr++;
            scale_ptr++;
        }
    }
}
void scale_global_compute_kernel(const float* din, float* dout, int num, int ch, int w, int h, float scale, float bias){
    int size = w * h;
    int cnt = size >> 2;
    int remain = size % 4;
    float32x4_t vscale = vdupq_n_f32(scale);
    //float32x4_t vbias = vdupq_f32(bias);
    for (int i = 0; i < num; i++){
        const float* din_ptr  = din + i * ch * size;
        float* dout_ptr = dout + i * ch * size;
#pragma omp parallel for
        for (int c = 0; c < ch; c++){
            const float* din_ch_ptr = din_ptr + c * size;
            float* dout_ch_ptr = dout_ptr + c * size;
            for (int j = 0; j < cnt; j++){
                float32x4_t din0 = vld1q_f32(din_ch_ptr);
                float32x4_t vsum = vdupq_n_f32(bias);
                vsum = vmlaq_f32(vsum, din0, vscale);
                vst1q_f32(dout_ch_ptr, vsum);
                dout_ch_ptr += 4;
                din_ch_ptr += 4;
            }
            for (int j = 0; j < remain; j++){
                *dout_ch_ptr = *din_ch_ptr * scale + bias;
                dout_ch_ptr++;
                din_ch_ptr++;
            }
        }
    }

}
template <>
SaberStatus SaberScale<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        ScaleParam<ARM> &param) {

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
    if (_scale_dim > 1 || inputs.size() > 1) {
        const float* scale_data = inputs.size() > 1 ? static_cast<const float*>(inputs[1]->data()) \
          : param.scale_w.data();
        const float* bias_data = param.bias_term ? param.scale_b.data() : nullptr;
        bool bias_flag = param.bias_term;
        if (_inner_dim == 1){
            if (bias_flag){
                scale_compute_kernel_one_bias(din_ptr, dout_ptr, _outer_dim, _scale_dim, \
                    _inner_dim, scale_data, bias_data);
            }else{
                scale_compute_kernel_one(din_ptr, dout_ptr, _outer_dim, _scale_dim,\
                    _inner_dim, scale_data, bias_data);
            }
        }else{
            scale_compute_kernel(din_ptr, dout_ptr, _outer_dim, _scale_dim, \
                _inner_dim, bias_flag, scale_data, bias_data);
        }
    } else {
        float scale = param.scale_w[0];
        float bias = 0;
        if (param.bias_term) {
            bias = param.scale_b[0];
        }
        scale_global_compute_kernel(din_ptr, dout_ptr, num, ch_in, w_in, h_in, scale, bias);
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Scale : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 2.0f * num * ch_in * h_in * w_in;
    ops.ts = ts;
    OpTimer::add_timer("Scale", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberScale, ScaleParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberScale, ScaleParam, ARM, AK_INT8);

} //namespace anakin

} //namespace anakin
