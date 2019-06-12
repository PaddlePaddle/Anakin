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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_ALIGNED_MAT_MUL_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_ALIGNED_MAT_MUL_H

#include "saber/funcs/impl/impl_aligned_mat_mul.h"
#include "sass_funcs.h"
#include "saber/funcs/impl/cuda/vender_batch_gemm.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberAlignedMatMul<NV, OpDtype>: public ImplBase<NV, OpDtype, AlignedMatMulParam<NV> > {

public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberAlignedMatMul() {}

    ~SaberAlignedMatMul() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             AlignedMatMulParam<NV> &param,
                             Context<NV> &ctx) {
        this->_ctx = &ctx;
        int batch = inputs[0]->get_seq_offset()[0].size() - 1;
        _vender_batch_gemm.init(param.is_transpose_X,
                           param.is_transpose_Y,
                           batch,
                           *(this->_ctx));
        

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               AlignedMatMulParam<NV> &param,
                               Context<NV> &ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                                 std::vector<Tensor<NV> *>& outputs,
                                 AlignedMatMulParam<NV>  &param);

private:

    std::function<void(const int, const int, const int,
                    const float, const float*, const float,
                    const float*, float*, cudaStream_t)> _kernel;
    BatchGemm<NV, VENDER_IMPL, float, float> _vender_batch_gemm;
    static int id;
    
};

} //namespace saber.

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_ALIGNED_MAT_MUL_H
