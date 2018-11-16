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
#ifndef ANAKIN_SABER_FUNCS_ARM_IMPL_SGEMV_ARM_INT8_H
#define ANAKIN_SABER_FUNCS_ARM_IMPL_SGEMV_ARM_INT8_H

#include "saber/lite/core/common_lite.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

// fixme now only support transA = false
void sgemv_int8(const bool transA, const int M, const int N, \
    const signed char* A, const signed char* x, int* y);

void sgemv_relu_int8(const bool transA, const int M, const int N, \
    const signed char* A, const signed char* x, int* y);

void sgemv_bias_int8(const bool transA, const int M, const int N, \
    const signed char* A, const signed char* x, int* y, const int* bias);

void sgemv_bias_relu_int8(const bool transA, const int M, const int N, \
    const signed char* A, const signed char* x, int* y, const int* bias);

} //namespace lite

} //namespace saber

} //namespace anakin

#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_FUNCS_ARM_IMPL_SGEMV_ARM_INT8_H
