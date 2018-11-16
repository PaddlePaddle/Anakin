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
    OpBase(ParamBase* param) {}
    virtual SaberStatus load_param(ParamBase* param) = 0;
    virtual SaberStatus load_param(std::istream& stream, const float* weights) = 0;
    virtual SaberStatus compute_output_shape(const std::vector<Tensor<CPU>*>& inputs,
                                 std::vector<Tensor<CPU>*>& outputs) = 0;

    virtual SaberStatus init(const std::vector<Tensor<CPU>*>& inputs,
                             std::vector<Tensor<CPU>*>& outputs, Context& ctx) = 0;
    virtual SaberStatus dispatch(const std::vector<Tensor<CPU>*>& inputs,
                                 std::vector<Tensor<CPU>*>& outputs) = 0;
    virtual SaberStatus set_op_precision(DataType ptype) = 0;
    DataType get_op_precision() { return _precision_type; }
    void set_op_name(const char* name){_op_name = name;}
    const char* get_op_name() { return _op_name.c_str();}
protected:
    Context* _ctx;
    bool _flag_param{false};
    bool _flag_init{false};
    bool _flag_create_param{false};
    std::string _op_name;
    DataType _precision_type{AK_FLOAT};
#if defined(ENABLE_OP_TIMER) || defined(ENABLE_DEBUG)
    SaberTimer _timer;
#endif
};

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_FUNCS_OP_BASE_H
