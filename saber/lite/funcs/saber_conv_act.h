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

#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/saber_conv.h"

namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class SaberConvAct2D {
public:
    SaberConvAct2D() {
        _conv_op = new SaberConv2D;
    }

    SaberConvAct2D(int weights_size, int num_output, int group, int kw, int kh, int stride_w, int stride_h, \
        int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, ActiveType type, \
        const float* weights, const float* bias) {

        LCHECK_EQ(type, Active_relu, "active type must be relu");
        _conv_op = new SaberConv2D(weights_size, num_output, group, kw, kh, stride_w, stride_h, \
            pad_w, pad_h, dila_w, dila_h, flag_bias, weights, bias);
    }

    SaberStatus load_param(int weights_size, int num_output, int group, int kw, int kh, int stride_w, int stride_h, \
        int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, ActiveType type, \
        const float* weights, const float* bias) {

        LCHECK_EQ(type, Active_relu, "active type must be relu");
        _conv_op->set_activation(true);
        return _conv_op->load_param(weights_size, num_output, group, kw, kh, stride_w, stride_h, \
            pad_w, pad_h, dila_w, dila_h, flag_bias, weights, bias);

    }

    ~SaberConvAct2D() {
        delete _conv_op;
    }

    SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) {
        return _conv_op->compute_output_shape(inputs, outputs);
    }

    SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) {
        _conv_op->set_activation(true);
        return _conv_op->init(inputs, outputs, ctx);
    }

    SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT> *>& outputs) {
        return _conv_op->dispatch(inputs, outputs);
    }

private:
    SaberConv2D* _conv_op;
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_H
