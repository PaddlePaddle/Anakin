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
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class SaberConvAct2D : public SaberConv2D {
public:
    SaberConvAct2D() {}

    SaberConvAct2D(int weights_size, int num_output, int group, int kw, int kh, int stride_w, int stride_h, \
        int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, ActiveType type, bool flag_relu, \
        const float* weights, const float* bias) {

        if (flag_relu) {
            LCHECK_EQ(type, Active_relu, "active type must be relu");
            _flag_relu = true;
        } else {
            _flag_relu = false;
        }
        SaberConv2D::set_activation(_flag_relu);
        SaberConv2D::SaberConv2D(weights_size, num_output, group, kw, kh, stride_w, stride_h, \
            pad_w, pad_h, dila_w, dila_h, flag_bias, weights, bias);
    }

    SaberStatus load_param(int weights_size, int num_output, int group, int kw, int kh, int stride_w, int stride_h, \
        int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, ActiveType type, bool flag_relu, \
        const float* weights, const float* bias) {

        if (flag_relu) {
            LCHECK_EQ(type, Active_relu, "active type must be relu");
            _flag_relu = true;
        } else {
            _flag_relu = false;
        }
        SaberConv2D::set_activation(_flag_relu);
        return SaberConv2D::load_param(weights_size, num_output, group, kw, kh, stride_w, stride_h, \
            pad_w, pad_h, dila_w, dila_h, flag_bias, weights, bias);
    }

    ~SaberConvAct2D() {}

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs)  override{
        return SaberConv2D::compute_output_shape(inputs, outputs);
    }

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx)  override{
        SaberConv2D::set_activation(_flag_relu);
        return SaberConv2D::init(inputs, outputs, ctx);
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT> *>& outputs)  override{
        return SaberConv2D::dispatch(inputs, outputs);
    }
private:
    bool _flag_relu;
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_H
