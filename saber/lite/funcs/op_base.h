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

#ifndef ANAKIN_SABER_LITE_FUNCS_OP_BASE_H
#define ANAKIN_SABER_LITE_FUNCS_OP_BASE_H

#include "saber/lite/core/common_lite.h"
#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"

namespace anakin{

namespace saber{

namespace lite{

class OpBase {
public:
    OpBase(){}
    virtual ~OpBase(){}
    OpBase(const char* parm_name);
    virtual SaberStatus load_param(const char* param_name) {
        return SaberUnImplError;
    }
    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) {
        return SaberUnImplError;
    }

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context& ctx) {
        return SaberUnImplError;
    }
    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) {
        return SaberUnImplError;
    }

protected:
    Context* _ctx;
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_FUNCS_OP_BASE_H
