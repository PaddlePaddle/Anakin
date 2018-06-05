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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_CONV_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_CONV_H

#include "saber/saber_funcs_param.h"
#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/neon/impl/sgemm_arm.h"

namespace anakin{

namespace saber{

namespace lite{

typedef void (*conv_func)(Tensor<float>& tensor_out, Tensor<float>& tensor_in, \
    const float* weights, const float* bias, \
    int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
    int pad_w, int pad_h, bool flag_bias, bool flag_relu, Sgemm& gemmer, void* work_space);


template <typename Dtype>
class SaberConv2D {
public:
    SaberConv2D() {
        _impl = nullptr;
        _workspace_fwd_sizes = 0;
        _is_trans_weights = false;
        _flag_relu = false;
        _bias_term = true;
        _workspace_data = std::make_shared<Tensor<Dtype>>();
        _weights_trans = std::make_shared<Tensor<Dtype>>();
    }

    ~SaberConv2D() {}

    SaberStatus compute_output_shape(const std::vector<Tensor<Dtype>*>& inputs,
                                     std::vector<Tensor<Dtype>*>& outputs,
                                     ConvParam<Tensor<Dtype>> &param) {
        Shape output_shape = inputs[0]->valid_shape();
        CHECK_EQ(input[0]->valid_shape().size(), 4) << "using reshape2d to reshape a 1d conv?";

        // append the $n and $c/$k, output: N * K * P * Q
        int num_idx = inputs[0]->num_index();
        int channel_idx = inputs[0]->channel_index();
        int height_idx = inputs[0]->height_index();
        int width_idx = inputs[0]->width_index();

        output_shape[num_idx] = inputs[0]->num(); // N
        output_shape[channel_idx] = param.weight()->num(); // K

        int input_dim = inputs[0]->height(); // P
        int kernel_exten = param.dilation_h * (param.weight()->height() - 1) + 1;
        int output_dim = (input_dim + 2 * param.pad_h - kernel_exten)
                         / param.stride_h + 1;

        output_shape[height_idx] = output_dim;

        input_dim = inputs[0]->width(); // Q
        kernel_exten = param.dilation_w * (param.weight()->width() - 1) + 1;
        output_dim = (input_dim + 2 * param.pad_w - kernel_exten)
                     / param.stride_w + 1;

        output_shape[width_idx] = output_dim;

        return outputs[0]->set_shape(output_shape);
    }

    SaberStatus init(const std::vector<Tensor<Dtype>*>& inputs,
                             std::vector<Tensor<Dtype>*>& outputs,
                             ConvParam<Tensor<Dtype>> &conv_param, Context &ctx);

    SaberStatus create(const std::vector<Tensor<Dtype>*>& inputs,
                               std::vector<Tensor<Dtype>*>& outputs,
                               ConvParam<Tensor<Dtype>> &conv_param, Context &ctx);

    SaberStatus dispatch(const std::vector<Tensor<Dtype>*>& inputs,
                                 std::vector<Tensor<Dtype>*>& outputs,
                                 ConvParam<Tensor<Dtype>> &conv_param);

    SaberStatus set_activation(bool flag) {
        _flag_relu = flag;
        return SaberSuccess;
    }

private:
    Context _ctx;
    conv_func _impl{nullptr};
    Sgemm _gemmer;
    bool _flag_relu{false};
    bool _is_trans_weights{false};
    bool _bias_term{true};
    int _kw;
    int _kh;
    size_t _workspace_fwd_sizes{0};
    std::shared_ptr<Tensor<Dtype>> _workspace_data{nullptr};
    std::shared_ptr<Tensor<Dtype>> _weights_trans{nullptr};
};


} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_CONV_H
