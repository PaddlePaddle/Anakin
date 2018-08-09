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
#include "saber/lite/funcs/op_param.h"
#include "saber/lite/funcs/timer_lite.h"

namespace anakin{

namespace saber{

namespace lite{

class OpBase {
public:
    OpBase(){}
    virtual ~OpBase(){}
    OpBase(const ParamBase* param) {}
    virtual SaberStatus load_param(const ParamBase* param) = 0;
    virtual SaberStatus load_param(FILE* fp, const float* weights) = 0;
    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) = 0;

    virtual SaberStatus init(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                             std::vector<Tensor<CPU, AK_FLOAT>*>& outputs, Context& ctx) = 0;
    virtual SaberStatus dispatch(const std::vector<Tensor<CPU, AK_FLOAT>*>& inputs,
                                 std::vector<Tensor<CPU, AK_FLOAT>*>& outputs) = 0;

    void set_op_name(const char* name){_op_name = name;}
    const char* get_op_name() { return _op_name;}

protected:
    const char* _op_name;
    Context* _ctx;
    bool _flag_param{false};
    bool _flag_init{false};
    bool _flag_create_param{false};
#ifdef ENABLE_OP_TIMER
    SaberTimer _timer;
#endif
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_FUNCS_OP_BASE_H
