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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PERMUTE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PERMUTE_H

#include "saber/funcs/impl/impl_permute.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberPermute<NV, OpDtype>:\
    public ImplBase<
        NV,
        OpDtype,
        PermuteParam<NV>> {

public:

    SaberPermute() {}

    ~SaberPermute() {}

    virtual SaberStatus init(const std::vector<Tensor<NV>*>& inputs,
                             std::vector<Tensor<NV>*>& outputs,
                             PermuteParam<NV> &param,
                             Context<NV> &ctx);
    virtual SaberStatus create(const std::vector<Tensor<NV>*>& inputs,
                               std::vector<Tensor<NV>*>& outputs,
                               PermuteParam<NV> &param,
                               Context<NV> &ctx);
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                                 std::vector<Tensor<NV>*>& outputs,
                                 PermuteParam<NV> &param);

private:
    int _num_axes;
    bool _need_permute;
    std::vector<int> _order_dims;
    Tensor<NV> _permute_order;
    Tensor<NV> _in_steps;
    Tensor<NV> _out_steps;
    Tensor<NV> _out_valid_shape;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PERMUTE_H
