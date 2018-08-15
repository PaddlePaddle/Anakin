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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ACTIVATION_H
#define ANAKIN_SABER_FUNCS_IMPL_ACTIVATION_H

#include "saber/funcs/impl/impl_macro.h"
namespace anakin{

namespace saber{

DEFINE_OP_CLASS(Activation, ActivationParam);
//template <typename TargetType, \
//    DataType OpDtype = AK_FLOAT,> \
//class SaberActivation : public ImplBase<TargetType, OpDtype, ActivationParam <TargetType> > {}; \
//\
//template <typename TargetType, \
//    DataType OpDtype = AK_FLOAT,> \
//class VenderActivation : public ImplBase< \
//    TargetType, OpDtype,\
//    ActivationParam <TargetType> > {};



}
}

#endif //ANAKIN_SABER_FUNCS_IMPL_ACTIVATION_H
