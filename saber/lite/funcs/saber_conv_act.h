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

#include "saber/saber_funcs_param.h"
#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/saber_conv.h"

namespace anakin{

namespace saber{

namespace lite{

template <typename Dtype>
class SaberConv2DAct {
public:
    SaberConv2DAct() {
        _conv_op = new SaberConv2D<Dtype>;
    }

    ~SaberConv2DAct() {
        delete _conv_op;
    }

    SaberStatus compute_output_shape(const std::vector<Tensor<Dtype>*>& inputs,
                                     std::vector<Tensor<Dtype>*>& outputs,
                                     ConvActiveParam<Tensor<Dtype>> &param) {
        return _conv_op->compute_output_shape(inputs, outputs, param.conv_param);
    }

    SaberStatus init(const std::vector<Tensor<Dtype>*>& inputs,
                             std::vector<Tensor<Dtype>*>& outputs,
                             ConvActiveParam<Tensor<Dtype>> &param, Context &ctx) {
        return _conv_op->init(inputs, outputs, param.conv_param, ctx);
    }

    SaberStatus create(const std::vector<Tensor<Dtype> *>& inputs,
                               std::vector<Tensor<Dtype> *>& outputs,
                               ConvActiveParam<Tensor<Dtype>> &param, Context &ctx) {
        _conv_op->set_activation(true);
        return _conv_op->create(inputs, outputs, param.conv_param, ctx);
    }

    SaberStatus dispatch(const std::vector<Tensor<Dtype> *>& inputs,
                                 std::vector<Tensor<Dtype> *>& outputs,
                                 ConvActiveParam<Tensor<Dtype>> &param) {
        return _conv_op->dispatch(inputs, outputs, param.conv_param);
    }

private:
    SaberConv2D<Dtype>* _conv_op;
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_CONV_ACT_H
