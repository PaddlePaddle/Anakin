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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CRF_DECODING_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CRF_DECODING_H

#include "saber/funcs/impl/impl_crf_decoding.h"
#include "saber/saber_funcs_param.h"

namespace anakin{
namespace saber {

template <DataType OpDtype>
class SaberCrfDecoding<X86, OpDtype> : public ImplBase<
        X86, OpDtype,
        CrfDecodingParam<X86> >
{
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberCrfDecoding() = default;

    ~SaberCrfDecoding() {}

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86> *>& outputs,
                             CrfDecodingParam<X86> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86> *>& outputs,
                               CrfDecodingParam<X86> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86> *>& inputs,
                                 std::vector<Tensor<X86> *>& outputs,
                                 CrfDecodingParam<X86> &param) override;
private:
    Tensor<X86> _alpha;
    Tensor<X86> _track;
    Tensor<X86> _trans;
    Tensor<X86> _emis;
    int _aligned_tag_num;
};
}
}

#endif
