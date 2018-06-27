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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_CONV_BATCHNORM_SCALE_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_CONV_BATCHNORM_SCALE_H

#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/saber_conv.h"
#include "saber/lite/funcs/utils_arm.h"

namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class SaberConvBatchnormScale {
public:
    SaberConvBatchnormScale() {
        _conv_op = new SaberConv2D;
    }

    SaberConvBatchnormScale(int weights_size, int num_output, int group, int kw, int kh, int stride_w, int stride_h, \
        int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, \
        float bn_scale, float bn_eps, std::vector<float> bn_mean, std::vector<float> bn_variance, \
        std::vector<float> scale_w, std::vector<float> scale_b, bool scale_bias_term, \
        const float* weights, const float* bias) {

        int ch = weights_size / (num_output * kw * kh);
        update_weights(_new_weights, _new_bias, weights, bias, \
            num_output, ch, kh, kw, flag_bias, \
            bn_scale, bn_eps, bn_mean, bn_variance, \
            scale_w, scale_b, scale_bias_term);

        _conv_op = new SaberConv2D(weights_size, num_output, group, kw, kh, stride_w, stride_h, \
            pad_w, pad_h, dila_w, dila_h, true, _new_weights.data(), _new_bias.data());
    }

    SaberStatus load_param(int weights_size, int num_output, int group, int kw, int kh, int stride_w, int stride_h, \
        int pad_w, int pad_h, int dila_w, int dila_h, bool flag_bias, \
        float bn_scale, float bn_eps, std::vector<float> bn_mean, std::vector<float> bn_variance, \
        std::vector<float> scale_w, std::vector<float> scale_b, bool scale_bias_term, \
        const float* weights, const float* bias) {

        int ch = weights_size / (num_output * kw * kh);
        update_weights(_new_weights, _new_bias, weights, bias, \
            num_output, ch, kh, kw, flag_bias, \
            bn_scale, bn_eps, bn_mean, bn_variance, \
            scale_w, scale_b, scale_bias_term);

        return _conv_op->load_param(weights_size, num_output, group, kw, kh, stride_w, stride_h, \
            pad_w, pad_h, dila_w, dila_h, true, _new_weights.data(), _new_bias.data());

    }

    ~SaberConvBatchnormScale() {
        delete _conv_op;
    }

    SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) {
        return _conv_op->compute_output_shape(inputs, outputs);
    }

    SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx) {
        _conv_op->set_activation(false);
        return _conv_op->init(inputs, outputs, ctx);
    }

    SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT> *>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT> *>& outputs) {
        return _conv_op->dispatch(inputs, outputs);
    }

private:
    SaberConv2D* _conv_op;
    Tensor<CPU, AK_FLOAT> _new_weights;
    Tensor<CPU, AK_FLOAT> _new_bias;
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_CONV_BATCHNORM_SCALE_H
