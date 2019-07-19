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
 *
 * This implementation is based on https://github.com/ARM-software/ComputeLibrary/
 *
 * Copyright (c) 2017-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_SGEMM_ARM_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_SGEMM_ARM_H

#include "saber/core/common.h"
#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{


typedef void (*load_data)(float* out, const float* in, const int ldin, const int m0, \
    const int mmax, const int k0, const int kmax);

class Sgemm {

public:

    Sgemm();
    ~Sgemm();
    SaberStatus init(unsigned int L1_cache, unsigned int L2_cache, unsigned int M, unsigned int N, \
        unsigned int K, bool trA, bool trB, int thread_num = 1, void* work_space = nullptr);

    //! Actually execute the GEMM.
    SaberStatus operator()(const float *A, const int lda, const float *B, const int ldb, \
        float *C, const int ldc, const float alpha, const float beta, bool flag_relu = false);

private:

    unsigned int _M;
    unsigned int _NN;
    unsigned int _K;

    bool _trA;
    bool _trB;

    unsigned int _k_block{0};
    unsigned int _x_block{0};
    unsigned int _Mround{0};

    unsigned int _loop_count{0};
    unsigned int _cblock_size{0};
    int _thread_num{1};

    void* _work_space_ptr{nullptr};
    void* _align_ptr{nullptr};
    bool _has_mem{false};

    size_t _work_size{0};
    size_t _a_worksize{0};
    size_t _b_worksize{0};
    load_data _load_a;
    load_data _load_b;

    bool _init_flag{false};
};

} //namespace saber

} //namespace anakin

#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_SGEMM_ARM_H
