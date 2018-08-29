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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ELTWISE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ELTWISE_H

#include "saber/funcs/impl/impl_eltwise.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberEltwise<NV, OpDtype>:
public ImplBase<
        NV, OpDtype,
        EltwiseParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;


    SaberEltwise() {}

    ~SaberEltwise() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                         std::vector<Tensor<NV> *>& outputs,
                         EltwiseParam<NV> &param,
                         Context<NV> &ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                           std::vector<Tensor<NV> *>& outputs,
                           EltwiseParam<NV> &param,
                           Context<NV> &ctx) {
        this->_ctx = &ctx;
        if ((param.operation == Eltwise_max) && (outputs.size() == 1)) {
            _max_idx.reshape(inputs[0]->shape());
        }
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             EltwiseParam<NV> &param) override;

private:
    Tensor<NV> _max_idx;
};

template class SaberEltwise<NV, AK_FLOAT>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ELTWISE_H
