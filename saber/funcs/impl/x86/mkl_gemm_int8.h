/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_MKL_GEMM_INT8_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_MKL_GEMM_INT8_H

#include "mkl.h"

#include "saber/saber_types.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/funcs/impl/x86/anakin_thread.h"

namespace anakin {
namespace saber {

template<DataType op_dtype>
class MKLGEMM {

public:
    typedef typename DataTrait<X86, op_dtype>::Dtype OP_DType;

    MKLGEMM()
        : omp_max_thread(anakin_get_max_threads())
    {}

    ~MKLGEMM() {}

    SaberStatus init(const void* mem_b,
                     const void* mem_oc,
                     void** handle,
                     const char oc_mode,
                     const size_t m,
                     const size_t n,
                     const size_t k,
                     const int8_t oa,
                     const int8_t ob,
                     const bool s8_a,
                     const bool pack_b,
                     const bool trans_a,
                     const bool trans_b,
                     const float beta,
                     const float alpha,
                     const size_t lda,
                     const size_t ldb,
                     const size_t ldc);

    SaberStatus execute(const void* handle,const int m,
                        const void* a_matrix,
                        void* c_matrix, const void* b_matrix=nullptr);

    SaberStatus release(void* handle);

    void* pack_mem(const void* mem_in,
                   const bool pack_b,
                   const bool trans,
                   const size_t m,
                   const size_t n,
                   const size_t k,
                   const size_t stride,
                   const float alpha);

    SaberStatus execute(const void* a_matrix,
                        const void* b_matrix,
                        const void* oc_matrix,
                        void* c_matrix,
                        const bool s8_a,
                        const size_t m,
                        const size_t n,
                        const size_t k,
                        const int8_t oa,
                        const int8_t ob,
                        const size_t lda,
                        const size_t ldb,
                        const size_t ldc,
                        const bool pack_b,
                        const bool trans_a,
                        const bool trans_b,
                        const float beta,
                        const float alpha,
                        const char oc_mode);

private:
    size_t omp_max_thread;

    struct gemm_param {
        const void* matrix_b{nullptr};
        const void* matrix_oc{nullptr};
        void* packed_mem{nullptr};
        void* oc_mem_s8a{nullptr};
        char oc_mode{' '};
        char s8a_oc_mode{' '};
        size_t m{0};
        size_t n{0};
        size_t k{0};
        size_t lda{0};
        size_t ldb{0};
        size_t ldc{0};
        size_t oa{0};
        int8_t ob{0};
        bool s8_a{false};
        bool pack_b{false};
        bool trans_a{false};
        bool trans_b{false};
        float beta{0.f};
        float alpha{0.f};
    };

    SaberStatus init_check(const void* mem_b,
                           const void* mem_oc,
                           const size_t oa,
                           const int8_t ob,
                           const char oc_mode);

    SaberStatus execute_check(const void* mem_a,
                              const void* mem_b,
                              const void* mem_oc,
                              void* mem_c,
                              const size_t oa,
                              const int8_t ob,
                              const char oc_mode);

    SaberStatus mem_a_s82u8(const int8_t* src, size_t length);

    void* mem_oc_s8a_compute(void* handle);

    SaberStatus add_mem_oc_s8a(bool a_s82u8, char oc_mode, const void* in,
                               void* out, size_t m, size_t n);
    void add_mem_oc_s8a(char oc_mode, const int* oc_mem, const void* b_in,int8_t ob,
                   size_t m, size_t k, size_t n, float alpha, bool trans_b);
    Tensor<X86> _inner_u8_matrix_a;
    Tensor<X86> _inner_c_offset;
};

template<DataType op_dtype>
SaberStatus MKLGEMM<op_dtype>::init_check(const void* mem_b,
        const void* mem_oc,
        const size_t oa,
        const int8_t ob,
        const char oc_mode) {
    if (mem_b == nullptr || mem_oc == nullptr) {
        LOG(ERROR) << "wrong empty pointer !";
        return SaberInvalidValue;
    }

    if (oc_mode != 'F' &&
            oc_mode != 'C' &&
            oc_mode != 'R') {
        LOG(ERROR) << "wrong mem_oc mode !";
        return SaberInvalidValue;
    }

    if (op_dtype == AK_FLOAT && (oa != 0 || ob != 0)) {
        LOG(ERROR) << "don't support offset a,b for float op!";
        return SaberInvalidValue;
    }

    return SaberSuccess;
};

template<DataType op_dtype>
SaberStatus MKLGEMM<op_dtype>::execute_check(const void* mem_a,
        const void* mem_b,
        const void* mem_oc,
        void* mem_c,
        const size_t oa,
        const int8_t ob,
        const char oc_mode) {
    if (mem_a == nullptr ||
            mem_b == nullptr ||
            mem_c == nullptr ||
            mem_oc == nullptr) {
        LOG(FATAL) << "wrong empty pointer !";
        return SaberInvalidValue;
    }

    if (oc_mode != 'F' &&
            oc_mode != 'C' &&
            oc_mode != 'R') {
        LOG(FATAL) << "wrong mem_oc mode !";
        return SaberInvalidValue;
    }

    if (op_dtype == AK_FLOAT && (oa != 0 || ob != 0)) {
        LOG(FATAL) << "don't support offset a,b for float op!";
        return SaberInvalidValue;
    }

    return SaberSuccess;
};

template<DataType op_dtype>
SaberStatus MKLGEMM<op_dtype>::init(const void* mem_b,
                                    const void* mem_oc,
                                    void** handle,
                                    const char oc_mode,
                                    const size_t m,
                                    const size_t n,
                                    const size_t k,
                                    const int8_t oa,
                                    const int8_t ob,
                                    const bool s8_a,
                                    const bool pack_b,
                                    const bool trans_a,
                                    const bool trans_b,
                                    const float beta,
                                    const float alpha,
                                    const size_t lda,
                                    const size_t ldb,
                                    const size_t ldc) {
    auto status = init_check(mem_b, mem_oc, oa, ob, oc_mode);

    if (status != SaberSuccess) {
        return status;
    }

    auto args = new gemm_param;

    args->s8_a = op_dtype == AK_INT8 ? s8_a : false;
    args->oc_mode = oc_mode;
    args->s8a_oc_mode = args->oc_mode == 'C' ? 'C' : 'R';
    args->m = m;
    args->n = n;
    args->k = k;
    args->oa = oa;
    args->ob = ob;
    args->lda = lda;
    args->ldb = ldb;
    args->ldc = ldc;
    args->pack_b = pack_b;
    args->trans_a = trans_a;
    args->trans_b = trans_b;
    args->beta = beta;
    args->alpha = alpha;

    args->matrix_b = mem_b;
    args->matrix_oc = mem_oc;
    args->packed_mem = nullptr;
    args->oc_mem_s8a = nullptr;

    if (args->pack_b) {
        args->packed_mem = pack_mem(args->matrix_b, true, args->trans_b,
                                    args->m, args->n, args->k, args->ldb, args->alpha);
    }
    if (args->s8_a){
        _inner_u8_matrix_a.re_alloc(Shape({1,1,m,k}), AK_UINT8);
        _inner_c_offset.re_alloc(Shape({1,1,1,n}),AK_INT32);
    }

    args->oc_mem_s8a = mem_oc_s8a_compute(args);

    *handle = args;
    args = nullptr;
    return SaberSuccess;
};

template<DataType op_dtype>
SaberStatus MKLGEMM<op_dtype>::release(void* handle) {
    auto args = static_cast<gemm_param*>(handle);

    if (args->packed_mem) {
        free(args->packed_mem);
        args->packed_mem = nullptr;
    }

    if (args->oc_mem_s8a) {
        free(args->oc_mem_s8a);
        args->oc_mem_s8a = nullptr;
    }

    delete (args);
    return SaberSuccess;
}

}
}


#endif //ANAKIN_MKL_GEMM_INT8_H
