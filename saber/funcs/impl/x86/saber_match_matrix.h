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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_MATCH_MATRIX_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_MATCH_MATRIX_H

#include "saber/funcs/impl/impl_match_matrix.h"
#include "saber/funcs/gemm.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberMatchMatrix<X86, OpDtype> :
    public ImplBase<
        X86, OpDtype,
        MatchMatrixParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberMatchMatrix() {}

    ~SaberMatchMatrix() {}

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             MatchMatrixParam<X86> &param,
                             Context<X86> &ctx) override;

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               MatchMatrixParam<X86> &param,
                               Context<X86> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 MatchMatrixParam<X86> &param) override;
private:
    Tensor<X86> _input_l_transform;
    Tensor<X86> _input_l_transform_reorganize;
    Tensor<X86> _output_tmp;
    Gemm<X86, VENDER_IMPL, float> _gemm_l_transform;
    Gemm<X86, VENDER_IMPL, float> _gemm_r_transform;

};

}
}
#endif
