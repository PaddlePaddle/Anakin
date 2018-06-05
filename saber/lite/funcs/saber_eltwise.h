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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_ELTWISE_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_ELTWISE_H

#include "saber/saber_funcs_param.h"
#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

typedef void (*eltwise_func)(const float* din_a, \
    const float* din_b, float* dout, const int size, std::vector<float> coef);

template <typename Dtype>
class SaberEltwise {
public:
    SaberEltwise() {}
    ~SaberEltwise() {}

    SaberStatus compute_output_shape(const std::vector<Tensor<Dtype>*>& inputs,
                                     std::vector<Tensor<Dtype>*>& outputs,
                                     EltwiseParam<Tensor<Dtype>> &param) {
        for (int i = 1; i < inputs.size(); ++i) {
            CHECK_EQ(input[0]->num(), input[i]->num());
            CHECK_EQ(input[0]->channel(), input[i]->channel());
            CHECK_EQ(input[0]->height(), input[i]->height());
            CHECK_EQ(input[0]->width(), input[i]->width());
        }

        Shape output_shape = inputs[0]->valid_shape();
        return outputs[0]->set_shape(output_shape);
    }

    SaberStatus init(const std::vector<Tensor<Dtype>*>& inputs,
                             std::vector<Tensor<Dtype>*>& outputs,
                             EltwiseParam<Tensor<Dtype>> &param, Context &ctx) {
        return create(inputs, outputs, param, ctx);
    }

    SaberStatus create(const std::vector<Tensor<Dtype>*>& inputs,\
                               std::vector<Tensor<Dtype>*>& outputs,\
                               EltwiseParam<Tensor<Dtype>> &param, \
                               Context &ctx);

    SaberStatus dispatch(const std::vector<Tensor<Dtype>*>& inputs, \
                                 std::vector<Tensor<Dtype>*>& outputs, \
                                 EltwiseParam<Tensor<Dtype>> &param);

private:
    Context _ctx;
    eltwise_func _impl{nullptr};
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_ELTWISE_H
