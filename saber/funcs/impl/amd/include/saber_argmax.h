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
#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_ARGMAX_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_ARGMAX_H

#include "saber/funcs/base.h"
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/core/impl/amd/utils/amd_profiler.h"
#include "saber/funcs/impl/impl_argmax.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"
#include <unordered_map>

#define TOP_1 0
#define BLOCK_TOP_1 1
#define TOP_1_BIG 2
#define TOP_1_CHANNEL 3
#define TOPK_CHANNEL 4
#define TOPK_HEAP_SHARED 5

namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberArgmax<AMD, OpDtype> : public ImplBase<AMD, OpDtype, ArgmaxParam<AMD>> {
public:
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;

    SaberArgmax() {}

    ~SaberArgmax() {}

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         ArgmaxParam<AMD>& param,
         Context<AMD>& ctx) override;

    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           ArgmaxParam<AMD>& param,
           Context<AMD>& ctx) override;

    virtual SaberStatus dispatch(
            const std::vector<Tensor<AMD>*>& inputs,
            std::vector<Tensor<AMD>*>& outputs,
            ArgmaxParam<AMD>& param) override;

private:
    Tensor<AMD> _group_max_value;
    Tensor<AMD> _group_max_index;
    std::unordered_map<int, size_t> _globalWorkSize;
    std::unordered_map<int, size_t> _localWorkSize;
    std::unordered_map<int, AMDKernelPtr> _kernel_map;
};

} // namespace saber
} // namespace anakin
#endif