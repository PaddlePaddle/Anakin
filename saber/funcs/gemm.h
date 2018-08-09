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

#ifndef ANAKIN_SABER_FUNCS_GEMM_H
#define ANAKIN_SABER_FUNCS_GEMM_H

#include "anakin_config.h"
#include "saber/core/context.h"
#include "saber/saber_types.h"

namespace anakin {
namespace saber {

template<typename TargetType,
        ImplEnum impl,
        typename inDtype,
        typename outDtype = inDtype>
class Gemm {
    // Row major gemm
public:
    Gemm() = default;
    ~Gemm() {}

    SaberStatus init(const bool trans_A, const bool trans_B,
                     const int m, const int n, const int k,
                     Context<TargetType> ctx) {
        return SaberUnImplError;
    }

    SaberStatus dispatch(const outDtype alpha, const outDtype beta,
                         const inDtype* a, const inDtype* b,
                         outDtype* c) {
        return SaberUnImplError;
    }

private:
    Context<TargetType> _ctx;
};

template<typename TargetType,
        ImplEnum impl,
        typename inDtype,
        typename outDtype = inDtype>
class Gemv {
    // Row major gemm
public:
    Gemv() = default;
    ~Gemv() {}

    SaberStatus init(const bool trans_A, const int m, const int n,
                     const int incx, const int incy,
                     Context<TargetType> ctx) {
        return SaberUnImplError;
    }

    SaberStatus dispatch(const outDtype alpha, const outDtype beta,
                         const inDtype* a, const inDtype* b,
                         outDtype* c) {
        return SaberUnImplError;
    }

private:
    Context<TargetType> _ctx;
};

}
}

#ifdef USE_CUDA
#include "saber/funcs/impl/cuda/vender_gemm.h"
#include "saber/funcs/impl/cuda/saber_gemm.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/vender_gemm.h"
#endif

#endif