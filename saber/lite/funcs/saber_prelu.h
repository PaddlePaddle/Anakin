/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
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
#ifndef ANAKIN_SABER_LITE_FUNCS_NEON_SABER_PRELU_H
#define ANAKIN_SABER_LITE_FUNCS_NEON_SABER_PRELU_H

#include "saber/lite/funcs/op_base.h"
#if 0
#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

namespace lite{

//template <typename Dtype>
class SaberPrelu : public OpBase {

public:

    SaberPrelu() {}

    SaberPrelu(bool flag_shared, const float* weights);

    SaberStatus load_param(bool flag_shared, const float* weights);

    ~SaberPrelu() {}

    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                     std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
        std::vector<Tensor<CPU, AK_FLOAT>*>& outputs,  Context &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs, \
        std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) override;

private:

    bool _flag_shared;
    const float* _weights{nullptr};
};

} //namespace lite

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE
#endif
#endif //ANAKIN_SABER_LITE_FUNCS_NEON_SABER_PRELU_H
