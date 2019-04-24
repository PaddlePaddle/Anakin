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
#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_SGEMM_PREPACKED_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_SGEMM_PREPACKED_H

#include "saber/core/context.h"
namespace anakin{

namespace saber{

#ifdef __aarch64__
const int MBLOCK = 8;
const int NBLOCK = 12;
const int KBLOCK = 4;
inline int get_hblock(ARMArch arch) {
    return MBLOCK;
}
#else
const int MBLOCK_A73 = 4;
const int MBLOCK_OTH = 6;
const int NBLOCK = 8;
const int KBLOCK = 4;
inline int get_hblock(ARMArch arch) {
    if (arch == A73) {
        return MBLOCK_A73;
    } else {
        return MBLOCK_OTH;
    }
}
#endif// __aarch64__

void prepackA(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax, bool is_trans, Context<ARM>* ctx);

void prepackA(Tensor<ARM>& tout, const Tensor<ARM>& tin, \
    int m, int k, int group, bool is_trans, Context<ARM>* ctx);

void sgemm_prepack(const float* A_packed, const float* B, const float* bias, float* C, \
    int M, int N, int K, \
    bool is_bias, bool is_relu, bool is_transB, Context<ARM>* ctx);

} //namespace saber

} //namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_SGEMM_PREPACKED_H
