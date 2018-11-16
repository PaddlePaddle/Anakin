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
#ifndef ANAKIN_SABER_FUNCS_ARM_IMPL_SGEMM_PREPACKED_INT8_H
#define ANAKIN_SABER_FUNCS_ARM_IMPL_SGEMM_PREPACKED_INT8_H

#include "saber/lite/core/context_lite.h"
namespace anakin{

namespace saber{

namespace lite{

#ifdef __aarch64__
//const int HBLOCK = 4;
//const int WBLOCK = 16;
inline int get_hblock_int8(ARMArch arch) {
    return 4;
}
#else
//const int HBLOCK = 4;
//const int WBLOCK = 8;
inline int get_hblock_int8(ARMArch arch) {
    return 4;
}
#endif// __aarch64__

void prepackA_int8(char* out, const char* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax, bool is_trans, Context* ctx);

void sgemm_prepack_int8(const char* A_packed, const char* B, const int* bias, int* C, \
    int M, int N, int K, bool is_bias, bool is_relu, bool is_transB, Context* ctx);

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_ARM_IMPL_SGEMM_PREPACKED_INT8_H
