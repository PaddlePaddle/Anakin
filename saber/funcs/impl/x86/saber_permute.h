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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_PERMUTE_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_PERMUTE_H

#include "saber/funcs/impl/impl_permute.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberPermute<X86, OpDtype>:\
    public ImplBase<
        X86,
        OpDtype,
        PermuteParam<X86>> {

public:

    SaberPermute() {}
    ~SaberPermute() {}

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             PermuteParam<X86> &param,
                             Context<X86> &ctx);
    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               PermuteParam<X86> &param,
                               Context<X86> &ctx);
    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 PermuteParam<X86> &param);

private:
    int _num_axes;
    bool _need_permute;
    std::vector<int> _order_dims;
    Tensor<X86> _permute_order;
    Tensor<X86> _in_steps;
    Tensor<X86> _out_steps;
    Tensor<X86> _out_valid_shape;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_PERMUTE_H
