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

#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

typedef void (*eltwise_func)(const float* din_a, \
    const float* din_b, float* dout, const int size, std::vector<float> coef);

//template <typename Dtype>
class SaberEltwise {
public:
    SaberEltwise() {}
    SaberEltwise(EltwiseType type, std::vector<float> coef);

    SaberStatus load_param(EltwiseType type, std::vector<float> coef);

    ~SaberEltwise() {}

    SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs);

    SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx);

    SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
                                 std::vector<Tensor<CPU, AK_FLOAT>*>& outputs);

private:
    Context _ctx;
    EltwiseType _type;
    std::vector<float> _coef;
    eltwise_func _impl{nullptr};
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_ELTWISE_H
