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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_COL2IM_DECONV_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_COL2IM_DECONV_H

#include "saber/core/tensor.h"
#include "saber_funcs_param.h"
#include "saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/gemm.h"

namespace anakin {
namespace saber {

template<DataType OpDtype = AK_FLOAT>
class SaberCol2ImDeconv : public ImplBase<
        X86, OpDtype, ConvParam <X86>> {

    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
public:

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             ConvParam<X86> &param, Context<X86>&ctx) override;
    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               ConvParam<X86> &param, Context<X86>&ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86> *>& inputs,
                                 std::vector<Tensor < X86>*>& outputs,
                                 ConvParam <X86> &param) override;

private:
    int _m;
    int _n;
    int _k;
    Tensor<X86> workspace_tensor;
    Gemm<X86, VENDER_IMPL, OpDataType> _gemm;
};

}
}
#endif