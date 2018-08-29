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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LRN_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LRN_H

#include "saber/funcs/impl/impl_lrn.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberLrn<X86, OpDtype>: public ImplBase<X86, OpDtype, LrnParam<X86> > {

public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberLrn() {}
    ~SaberLrn() {}

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86> *>& outputs,
                             LrnParam<X86> &param,
                             Context<X86> &ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86> *>& outputs,
                               LrnParam<X86> &crop_param,
                               Context<X86> &ctx) {
        N = inputs[0]->num();
        C = inputs[0]->channel();
        H = inputs[0]->height();
        W = inputs[0]->width();
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<X86> *>& inputs,
                                 std::vector<Tensor<X86> *>& outputs,
                                 LrnParam<X86> &param);

private:
    int N;
    int C;
    int H;
    int W;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_LRN_H