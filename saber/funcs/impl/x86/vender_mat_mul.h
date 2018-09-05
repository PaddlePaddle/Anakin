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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_MAT_MUL_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_MAT_MUL_H

#include "saber/funcs/impl/impl_mat_mul.h"
#include "mkl.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberMatMul<X86, OpDtype>: public ImplBase<X86, OpDtype, MatMulParam<X86> > {

public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberMatMul() {}

    ~SaberMatMul() {}

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86> *>& outputs,
                             MatMulParam<X86> &param,
                             Context<X86> &ctx) {
        this->_ctx = &ctx;

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86> *>& outputs,
                               MatMulParam<X86> &param,
                               Context<X86> &ctx) {
        M = param._m;
        N = param._n;
        K = param._k;
        batch = param._b;

        //row major.
        layout = CblasRowMajor;

        //matrix A whether to tranpose.
        //matrix A has size M by K.
        if (param._is_transpose_X) {
            transa = CblasTrans;
            if (layout == CblasRowMajor) {
                //A has changed its shape at mat_mul.h
                lda = M;
            }else {
                lda = K;
            }
        }else {
            transa = CblasNoTrans;
            if (layout == CblasRowMajor) {
                lda = K; 
            }else {
                lda = M;
            }
        }

        //matrix B whether to transpose.
        //matrix B has size K by N.
        if (param._is_transpose_Y) {
            transb = CblasTrans;
            if (layout == CblasRowMajor) {
                ldb = K;
            }else {
                ldb = N;
            }
        }else {
            transb = CblasNoTrans;
            if (layout == CblasRowMajor) {
                ldb = N;
            }else {
                ldb = K;
            }
        }

        if (layout == CblasRowMajor) {
            ldc = N;
        }else {
            ldc = M;
        }

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<X86> *>& inputs,
                                 std::vector<Tensor<X86> *>& outputs,
                                 MatMulParam<X86>  &param);

private:
    CBLAS_LAYOUT layout; //CblasRowMajor or CblasColMajor
    CBLAS_TRANSPOSE transa; //matrix A whether to tranpose.
    CBLAS_TRANSPOSE transb; //matrix B whether to tranpose.
    int batch;
    int M;
    int N;
    int K;
    int lda; //matrix A leading dimention.
    int ldb; //matrix B leading dimention.
    int ldc; //matrix C leading dimention.
    float alpha{1.0f};
    float beta{0.0f};
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_X86_MAT_MUL_H