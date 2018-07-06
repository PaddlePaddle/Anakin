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
#ifndef ANAKIN_SABER_FUNCS_ARM_IMPL_SGEMM_ARM_H
#define ANAKIN_SABER_FUNCS_ARM_IMPL_SGEMM_ARM_H

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
    void init(unsigned int L1_cache, unsigned int L2_cache, unsigned int M, unsigned int N, \
        unsigned int K, bool trA, bool trB, int thread_num = 1);

    //! Actually execute the GEMM.
    void operator()(const float *A, const int lda, const float *B, const int ldb, \
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

#endif //ANAKIN_SABER_FUNCS_ARM_IMPL_SGEMM_ARM_H
