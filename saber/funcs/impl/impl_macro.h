/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.
   Modifications (c) 2018 Advanced Micro Devices, Inc.

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
    DataType OpDtype = AK_FLOAT, \
    DataType inDtype = AK_FLOAT, \
    DataType outDtype = AK_FLOAT, \
    typename LayOutType_op = NCHW, \
    typename LayOutType_in = NCHW, \
    typename LayOutType_out = NCHW> \
class Saber##class_name : public ImplBase< \
    Tensor<TargetType, inDtype, LayOutType_in>, \
    Tensor<TargetType, outDtype, LayOutType_out>, \
    Tensor<TargetType, OpDtype, LayOutType_op>, \
    param_name <Tensor<TargetType, OpDtype, LayOutType_op> > > {}; \
\
template <typename TargetType, \
    DataType OpDtype = AK_FLOAT, \
    DataType inDtype = AK_FLOAT, \
    DataType outDtype = AK_FLOAT, \
    typename LayOutType_op = NCHW, \
    typename LayOutType_in = NCHW, \
    typename LayOutType_out = NCHW> \
class Vender##class_name : public ImplBase< \
    Tensor<TargetType, inDtype, LayOutType_in>, \
    Tensor<TargetType, outDtype, LayOutType_out>, \
    Tensor<TargetType, OpDtype, LayOutType_op>, \
    param_name <Tensor<TargetType, OpDtype, LayOutType_op> > > {}; 

}
}
