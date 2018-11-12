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
#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_NORMALIZE_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_NORMALIZE_H

#include "saber/funcs/base.h"
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/funcs/impl/impl_normalize.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"
#include <unordered_map>

namespace anakin {
namespace saber {

#define NORM_NO_ACROSS_SPATIAL 0
#define REDUCE_ADD_ATOMIC 1
#define GPU_POW_REVERSE 2
#define NORM_WITH_SCALE 3
#define NORM 4

template <DataType OpDtype>
class SaberNormalize<AMD, OpDtype> : public ImplBase<AMD, OpDtype, NormalizeParam<AMD>> {
public:
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;

    SaberNormalize() {}

    ~SaberNormalize() {}

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         NormalizeParam<AMD>& param,
         Context<AMD>& ctx) override;

    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           NormalizeParam<AMD>& param,
           Context<AMD>& ctx) override;

    virtual SaberStatus dispatch(
            const std::vector<Tensor<AMD>*>& inputs,
            std::vector<Tensor<AMD>*>& outputs,
            NormalizeParam<AMD>& param) override;

private:
    Tensor<AMD> _norm_reduce;
    int _size;
    int _norm_size;
    int _compute_size;
    int _batchs;
    int _channels;
    int _dims;
    int _channel_stride;
    Tensor<AMD> _input_stride;
    Tensor<AMD> _output_stride;
    Tensor<AMD> _valid_shape;
    bool _is_continue_buf{true};
    std::unordered_map<int, AMDKernelPtr> _kernel_map;
};

} // namespace saber
} // namespace anakin
#endif
