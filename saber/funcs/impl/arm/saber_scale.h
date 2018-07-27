/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#ifndef ANAKIN_SABER_FUNCS_ARM_SABER_SCALE_H
#define ANAKIN_SABER_FUNCS_ARM_SABER_SCALE_H

#include "saber/funcs/impl/impl_scale.h"
#include "saber/core/tensor.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberScale<ARM, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>:\
    public ImplBase<
        Tensor<ARM, inDtype, LayOutType_in>,
        Tensor<ARM, outDtype, LayOutType_out>,
        Tensor<ARM, OpDtype, LayOutType_op>,
        ScaleParam<Tensor<ARM, OpDtype, LayOutType_op>>> {
public:
    typedef Tensor<ARM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<ARM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<ARM, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberScale() = default;
    ~SaberScale() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                      std::vector<DataTensor_out*>& outputs,
                      ScaleParam<OpTensor> &param, Context<ARM> &ctx) override {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                        std::vector<DataTensor_out*>& outputs,
                        ScaleParam<OpTensor> &param, Context<ARM> &ctx) override {

        _inner_dim = inputs[0]->count(param.axis + param.num_axes, inputs[0]->shape().dims());
        _scale_dim = inputs[0]->count(param.axis, param.axis + param.num_axes);
        if (inputs.size() == 1) {
            CHECK_EQ(_scale_dim, param.scale_w.size()) << "scale dim not valid";
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
    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          ScaleParam<OpTensor> &param) override {

//        LOG(ERROR) << "scale param, axis=" << param.axis << ", " << param.bias_term << ", num_axis: " << param.num_axes << ", bias term:" << param.bias_term;
//        LOG(ERROR) << "scale_w:";
//        for (int i = 0; i < param.scale_w.size(); ++i) {
//            printf("%.2f ", param.scale_w[i]);
//        }
//        printf("\n");
//                LOG(ERROR) << "scale_b:";
//        for (int i = 0; i < param.scale_b.size(); ++i) {
//            printf("%.2f ", param.scale_b[i]);
//        }
//        printf("\n");

        int num = inputs[0]->num();
        int ch_in = inputs[0]->channel();
        int w_in = inputs[0]->width();
        int h_in = inputs[0]->height();
        const float* din_ptr = inputs[0]->data();
        float* dout_ptr = outputs[0]->mutable_data();
        if (_scale_dim > 1 || inputs.size() > 1) {
            float* scale_data = inputs.size() > 1 ? inputs[1]->data() : &param.scale_w[0];
            float* bias_data = param.bias_term ? &param.scale_b[0] : NULL;
            scale_compute_kernel(din_ptr, dout_ptr, num, ch_in, w_in, h_in, scale_data, bias_data);
        } else {
            float scale = param.scale_w[0];
            float bias = 0;
            if (param.bias_term) {
                bias = param.scale_b[0];
            }
           scale_global_compute_kernel(din_ptr, dout_ptr, num, ch_in, w_in, h_in, scale, bias);
        }
        return SaberSuccess;
    }
private:
    int _scale_dim;
    int _inner_dim;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_ARM_SABER_SCALE_H
