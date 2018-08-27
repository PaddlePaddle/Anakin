/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_FUNCS_CUDA_SABER_TRANSPOSE_H
#define ANAKIN_SABER_FUNCS_CUDA_SABER_TRANSPOSE_H

#include "saber/funcs/impl/impl_transpose.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberTranspose<NV, OpDtype>:
    public ImplBase<NV, OpDtype, TransposeParam<NV>> {

public:
    typedef Tensor<NV> DataTensor_in;
    typedef Tensor<NV> DataTensor_out;
    typedef Tensor<NV> OpTensor;

    typedef typename DataTrait<NV, OpDtype>::Dtype InDataType;
    typedef typename DataTrait<NV, OpDtype>::Dtype OutDataType;
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberTranspose() = default;

    ~SaberTranspose() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             TransposeParam<NV> &param,
                             Context<NV> &ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param,ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               TransposeParam<NV> &param,
                               Context<NV> &ctx) {
        if (!(&ctx == this->_ctx)) {
            this->_ctx = &ctx;
        }
        // do nothing
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 TransposeParam<NV> &param);


};
template class SaberTranspose<NV, AK_FLOAT>;
} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_FUNCS_CUDA_SABER_TRANSPOSE_H
