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
#ifndef ANAKIN_SABER_LITE_FUNCS_SABER_FC_H
#define ANAKIN_SABER_LITE_FUNCS_SABER_FC_H

#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

#ifdef USE_ARM_PLACE

#include "saber/lite/funcs/neon/impl/sgemm_arm.h"

namespace anakin{

namespace saber{

namespace lite{

//! input size: 1xk
//! output size: 1xn
//! weights size: nxk
//! bias size: 1xn
//template <typename Dtype>
class SaberFc {
public:
    SaberFc() {}

    SaberFc(int axis, int num_output, bool flag_trans, bool flag_bias, \
        const float* weights, const float* bias);

    SaberStatus load_param(int axis, int num_output, bool flag_trans, bool flag_bias, \
        const float* weights, const float* bias);

    ~SaberFc() {}

    SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs);

    SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
        std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context &ctx);

    SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
        std::vector<Tensor<CPU, AK_FLOAT>*>& outputs);


private:
    Context _ctx;
    Sgemm _gemmer;
    int _m;
    int _k;
    int _n;

    int _axis;
    int _num_output;
    bool _bias_term{true};
    bool _flag_trans{false};
    const float* _weights{nullptr};
    const float* _bias{nullptr};
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_SABER_FC_H
