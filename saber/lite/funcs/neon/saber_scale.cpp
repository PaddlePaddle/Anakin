#include "saber/lite/funcs/saber_scale.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

   SaberScale::SaberScale(const ParamBase* param){
      _param = (const ScaleParam*)param;
      this->_flag_param = true;
    }

    SaberStatus SaberScale::load_param(const ParamBase* param){
      _param = (const ScaleParam*)param;
      this->_flag_param = true;
      return SaberSuccess;
    }

    SaberStatus SaberScale::compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) {
      if (!this->_flag_param) {
          printf("load scale param first\n");
          return SaberNotInitialized;
      }
      Shape output_shape = (inputs[0]->valid_shape());
      return outputs[0]->set_shape(output_shape);
    }

    SaberStatus SaberScale::init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) {
      _inner_dim = inputs[0]->count(_param->axis + _param->num_axes, inputs[0]->shape().dims());
      _scale_dim = inputs[0]->count(_param->axis, _param->axis + _param->num_axes);
      if (inputs.size() == 1) {
          LCHECK_EQ(_scale_dim, _param->scale_w.size(), "scale dim not valid");
      }
      const int count = inputs[0]->valid_size();

      if (inputs.size() > 1) {
          _scale_dim = inputs[1]->valid_size();
          _inner_dim = count / _scale_dim;
      }

      return SaberSuccess;
    }

    void scale_compute_kernel(const float* din, float* dout, int num, int ch, int w, int h, float* scale_data, float* bias_data){
    int size = w * h;
    int cnt = size >> 4;
    int remain = size % 16;
    for(int i = 0; i < num; i++) {
        const float* din_ptr  = din + i * ch * size;
        float* dout_ptr = dout + i * ch * size;
        float* scale_ptr = scale_data;
       // float* bias_ptr = bias_data + i * ch;
#pragma omp parallel for
        for(int c = 0; c < ch; c++){
            const float* din_ch_ptr = din_ptr + c * size;
            float* dout_ch_ptr = dout_ptr + c * size;
            float32x4_t vscale = vdupq_n_f32(scale_ptr[c]);
            float bias = bias_data == NULL ? 0.f : bias_data[c];
            float32x4_t vbias = vdupq_n_f32(bias);
            for(int j = 0; j < cnt; j++){

                float32x4_t din0 = vld1q_f32(din_ch_ptr);
                float32x4_t din1 = vld1q_f32(din_ch_ptr + 4);
                float32x4_t din2 = vld1q_f32(din_ch_ptr + 8);
                float32x4_t din3 = vld1q_f32(din_ch_ptr + 12);

                float32x4_t vsum1 = vmlaq_f32(vbias, din0, vscale);
                float32x4_t vsum2 = vmlaq_f32(vbias, din1, vscale);
                float32x4_t vsum3 = vmlaq_f32(vbias, din2, vscale);
                float32x4_t vsum4 = vmlaq_f32(vbias, din3, vscale);

                vst1q_f32(dout_ch_ptr, vsum1);
                vst1q_f32(dout_ch_ptr + 4, vsum2);
                vst1q_f32(dout_ch_ptr + 8, vsum3);
                vst1q_f32(dout_ch_ptr + 12, vsum4);

                dout_ch_ptr += 16;
                din_ch_ptr += 16;
            }
            for(int j = 0; j < remain; j++){
                *dout_ch_ptr = *din_ch_ptr * scale_ptr[c] + bias;
                dout_ch_ptr++;
                din_ch_ptr++;
            }
        }
    }

}
void scale_global_compute_kernel(const float* din, float* dout, int num, int ch, int w, int h, float scale, float bias){
    int size = w * h;
    int cnt = size >> 2;
    int remain = size % 4;
    float32x4_t vscale = vdupq_n_f32(scale);
    //float32x4_t vbias = vdupq_f32(bias);
    for(int i = 0; i < num; i++){
        const float* din_ptr  = din + i * ch * size;
        float* dout_ptr = dout + i * ch * size;
#pragma omp parallel for
        for(int c = 0; c < ch; c++){
            const float* din_ch_ptr = din_ptr + c * size;
            float* dout_ch_ptr = dout_ptr + c * size;
            for(int j = 0; j < cnt; j++){
                float32x4_t din0 = vld1q_f32(din_ch_ptr);
                float32x4_t vsum = vdupq_n_f32(bias);
                vsum = vmlaq_f32(vsum, din0, vscale);
                vst1q_f32(dout_ch_ptr, vsum);
                dout_ch_ptr += 4;
                din_ch_ptr += 4;
            }
            for(int j = 0; j < remain; j++){
                *dout_ch_ptr = *din_ch_ptr * scale + bias;
                dout_ch_ptr++;
                din_ch_ptr++;
            }
        }
    }

}

    SaberStatus SaberScale::dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT>*>& outputs){
      int num = inputs[0]->num();
      int ch_in = inputs[0]->channel();
      int w_in = inputs[0]->width();
      int h_in = inputs[0]->height();
      const float* din_ptr = inputs[0]->data();
      float* dout_ptr = outputs[0]->mutable_data();
      if (_scale_dim > 1 || inputs.size() > 1) {
          float* scale_data = inputs.size() > 1 ? inputs[1]->data() : &_param->scale_w[0];
          float* bias_data = _param->bias_term ? &_param->scale_b[0] : NULL;
          scale_compute_kernel(din_ptr, dout_ptr, num, ch_in, w_in, h_in, scale_data, bias_data);
        } else {
            float scale = _param->scale_w[0];
            float bias = 0;
            if (_param->bias_term) {
                bias = _param->scale_b[0];
            }
           scale_global_compute_kernel(din_ptr, dout_ptr, num, ch_in, w_in, h_in, scale, bias);
        }
        return SaberSuccess;
    }

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

