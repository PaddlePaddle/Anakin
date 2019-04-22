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
#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_SGEMV_ARM_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_SGEMV_ARM_H

#include "saber/core/common.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

// fixme now only support transA = false

SaberStatus sgemv(const float* A, const float* x, float* y, bool transA, int M, int N, \
    bool is_bias = false, const float* bias = nullptr, bool is_relu = false);


} //namespace saber

} //namespace anakin

#endif // USE_ARM_PLACE

#endif // ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_SGEMV_ARM_H
