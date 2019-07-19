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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_TOPK_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_TOPK_POOLING_H

#include "saber/funcs/impl/impl_topk_pooling.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberTopKPooling<X86, OpDtype> :
    public ImplBase<
        X86, OpDtype,
        TopKPoolingParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberTopKPooling() {}

    ~SaberTopKPooling() {}

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             TopKPoolingParam<X86> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               TopKPoolingParam<X86> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 TopKPoolingParam<X86> &param) override;

private:
   SaberStatus get_topk(std::vector<OpDataType>& src, int top_k, int real_k, OpDataType* dst);

};

}
}
#endif
