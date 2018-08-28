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
#include "anakin_config.h"
#include <vector>
#include "saber/core/tensor.h"
#include "saber_funcs_param.h"
#include "saber_types.h"
#include "saber/funcs/impl/impl_base.h"
namespace anakin{

namespace saber{

#define DEFINE_OP_CLASS(class_name, param_name) \
template <typename TargetType, \
    DataType OpDtype = AK_FLOAT> \
class Saber##class_name : public ImplBase< \
    TargetType, OpDtype,\
    param_name <TargetType> > {}; \
\
template <typename TargetType, \
    DataType OpDtype = AK_FLOAT> \
class Vender##class_name : public ImplBase< \
    TargetType, OpDtype,\
    param_name <TargetType> > {};

#define DEFINE_OP_TEMPLATE(op_name, op_param, op_target, op_dtype) \
template<> SaberStatus op_name<op_target, op_dtype>::create( \
        const std::vector<Tensor<op_target> *>& inputs, \
        std::vector<Tensor<op_target> *>& outputs, op_param<op_target> &param, \
        Context<op_target> &ctx) {return SaberUnImplError;} \
template<> SaberStatus op_name<op_target, op_dtype>::init( \
        const std::vector<Tensor<op_target> *>& inputs, \
        std::vector<Tensor<op_target> *>& outputs, op_param<op_target> &param, \
        Context<op_target> &ctx) {return SaberUnImplError;} \
template<> SaberStatus op_name<op_target, op_dtype>::dispatch( \
        const std::vector<Tensor<op_target> *>& inputs, \
        std::vector<Tensor<op_target> *>& outputs, op_param<op_target> &param \
        ) {return SaberUnImplError;}
}
}
