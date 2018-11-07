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

#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_POOLING_WITH_INDEX_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_POOLING_WITH_INDEX_H

#include "saber/funcs/impl/impl_pooling_with_index.h"
#include "saber/funcs/base.h"
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/core/impl/amd/utils/amd_profiler.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"

#define AMD_NUM_THREADS 256

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberPoolingWithIndex<AMD, OpDtype> : public ImplBase<AMD, OpDtype, PoolingParam<AMD>> {

public:
    typedef typename DataTrait<AMD, OpDtype>::Dtype dtype;
    typedef TargetWrapper<AMD> API;
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;

    SaberPoolingWithIndex() {}

    ~SaberPoolingWithIndex() {}

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         PoolingParam<AMD>& param,
         Context<AMD>& ctx) override;

    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           PoolingParam<AMD>& power_param,
           Context<AMD>& ctx) override;

    virtual SaberStatus dispatch(
            const std::vector<Tensor<AMD>*>& inputs,
            std::vector<Tensor<AMD>*>& outputs,
            PoolingParam<AMD>& param) override;

private:
    int _in_n_stride;
    int _in_c_stride;
    int _in_h_stride;
    int _in_w_stride;
    int _out_n_stride;
    int _out_c_stride;
    int _out_h_stride;
    int _out_w_stride;
    AMDKernelPtr _kernel_poo1ing_with_index;
};
template class SaberPoolingWithIndex<AMD, AK_FLOAT>;

} // namespace saber

} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_POOLING_WITH_INDEX_H
