/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.
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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_H

#include "saber/lite/funcs/saber_conv.h"
#include "saber/lite/funcs/saber_pooling.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class SaberConvActPooling2D : public SaberConv2D, public SaberPooling {
public:
    SaberConvActPooling2D() {
        _vtensor_tmp.push_back(&_tensor_tmp);
    }

    SaberConvActPooling2D(int weights_size, int num_output, int group, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, \
        ActiveType type, bool flag_relu, const float* weights, const float* bias, \
        PoolingType pool_type, bool flag_global, int pool_kw, int pool_kh, \
        int pool_stride_w, int pool_stride_h, int pool_pad_w, int pool_pad_h) {

        _vtensor_tmp.push_back(&_tensor_tmp);

        if (flag_relu) {
            _flag_relu = true;
            LCHECK_EQ(type, Active_relu, "active type must be relu");
        } else {
            _flag_relu = false;
        }
        SaberConv2D::set_activation(_flag_relu);
        SaberConv2D::SaberConv2D(weights_size, num_output, group, kw, kh, stride_w, stride_h, \
            pad_w, pad_h, dila_w, dila_h, flag_bias, weights, bias);
        SaberPooling::SaberPooling(pool_type, flag_global, pool_kw, pool_kh, \
            pool_stride_w, pool_stride_h, pool_pad_w, pool_pad_h);
    }

    SaberStatus load_param(int weights_size, int num_output, int group, int kw, int kh, \
        int stride_w, int stride_h, int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, \
        ActiveType type, bool flag_relu, const float* weights, const float* bias, \
        PoolingType type, bool flag_global, int kernel_w, int kernel_h, \
        int stride_w, int stride_h, int pad_w, int pad_h) {

        if (flag_relu) {
            _flag_relu = true;
            LCHECK_EQ(type, Active_relu, "active type must be relu");
        } else {
            _flag_relu = false;
        }
        SaberConv2D::set_activation(_flag_relu);
        SaberStatus state = SaberConv2D::load_param(weights_size, num_output, group, kw, kh, \
            stride_w, stride_h, pad_w, pad_h, dila_w, dila_h, flag_bias, weights, bias);
        if (state != SaberSuccess) {
            return state;
        }
        return SaberPooling::load_param(pool_type, flag_global, pool_kw, pool_kh, \
            pool_stride_w, pool_stride_h, pool_pad_w, pool_pad_h);
    }

    ~SaberConvActPooling2D() {}

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override {
        SaberStatus state = SaberConv2D::compute_output_shape(inputs, _vtensor_tmp);
        if (state != SaberSuccess) {
            return state;
        }
        _tensor_tmp.re_alloc(_tensor_tmp.valid_shape());
        return SaberPooling::compute_output_shape(_vtensor_tmp, outputs);
    }

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) override {
        SaberConv2D::set_activation(_flag_relu);
        _tensor_tmp.re_alloc(_tensor_tmp.valid_shape());
        SaberStatus state = SaberConv2D::init(inputs, _vtensor_tmp, ctx);
        if (state != SaberSuccess) {
            return state;
        }
        return SaberPooling::init(_vtensor_tmp, outputs, ctx);
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT> *>& outputs) override {
        SaberStatus state = SaberConv2D::dispatch(inputs, _vtensor_tmp);
        if (state != SaberSuccess) {
            return state;
        }
        return SaberPooling::dispatch(_vtensor_tmp, outputs);
    }

private:
    bool _flag_relu;
    Tensor<CPU, AK_FLOAT> _tensor_tmp;
    std::vector<Tensor<CPU, AK_FLOAT> *> _vtensor_tmp;
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_H
