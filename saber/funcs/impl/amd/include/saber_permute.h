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
#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_PERMUTE_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_PERMUTE_H

#include "saber/funcs/base.h"
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/core/impl/amd/utils/amd_profiler.h"
#include "saber/funcs/impl/impl_permute.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberPermute<AMD, OpDtype> : public ImplBase<AMD, OpDtype, PermuteParam<AMD>> {
public:
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;

    SaberPermute() {}

    ~SaberPermute() {}

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         PermuteParam<AMD>& param,
         Context<AMD>& ctx) override;

    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           PermuteParam<AMD>& param,
           Context<AMD>& ctx) override;

    virtual SaberStatus dispatch(
            const std::vector<Tensor<AMD>*>& inputs,
            std::vector<Tensor<AMD>*>& outputs,
            PermuteParam<AMD>& param) override;

private:
    int _num_axes;
    bool _need_permute;
    std::vector<int> _order_dims;
    Tensor<AMD> _permute_order;
    Tensor<AMD> _in_steps;
    Tensor<AMD> _out_steps;
    Tensor<AMD> _out_valid_shape;
    AMDKernelPtr _kernel_ptr;
};

} // namespace saber
} // namespace anakin
#endif
