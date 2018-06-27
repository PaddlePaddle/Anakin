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
#ifndef ANAKIN_SABER_FUNCS_ARM_IMPL_UTILS_ARM_H
#define ANAKIN_SABER_FUNCS_ARM_IMPL_UTILS_ARM_H

#include "saber/lite/core/common_lite.h"
#include "saber/lite/core/tensor_lite.h"
namespace anakin{

namespace saber{

namespace lite{

void update_weights(Tensor<CPU, AK_FLOAT>& new_weight, Tensor<CPU, AK_FLOAT>& new_bias, \
    const float* weights, const float* bias, int num, int ch, int kh, int kw, bool conv_bias_term, \
    float batchnorm_scale, float batchnorm_eps, \
    std::vector<float> batchnorm_mean, std::vector<float> batchnorm_variance, \
    std::vector<float> scale_w, std::vector<float> scale_b, bool scale_bias_term);

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_ARM_IMPL_UTILS_ARM_H
