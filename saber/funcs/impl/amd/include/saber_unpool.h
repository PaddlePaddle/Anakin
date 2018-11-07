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

#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_UNPOOL_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_UNPOOL_H

#include "saber/funcs/impl/impl_unpool.h"
#include "saber/funcs/base.h"
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"

#define AMD_NUM_THREADS 256

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberUnpool<AMD, OpDtype> : public ImplBase<AMD, OpDtype, PoolingParam<AMD>> {

public:
    typedef Tensor<AMD> DataTensor_in;
    typedef Tensor<AMD> DataTensor_out;
    typedef Tensor<AMD> OpTensor;
    typedef TargetWrapper<AMD> API;
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;

    SaberUnpool() {}

    ~SaberUnpool() {}

    virtual SaberStatus
    init(const std::vector<DataTensor_in*>& inputs,
         std::vector<DataTensor_out*>& outputs,
         PoolingParam<AMD>& param,
         Context<AMD>& ctx) override;

    virtual SaberStatus
    create(const std::vector<DataTensor_in*>& inputs,
           std::vector<DataTensor_out*>& outputs,
           PoolingParam<AMD>& param,
           Context<AMD>& ctx) override;

    virtual SaberStatus dispatch(
            const std::vector<DataTensor_in*>& inputs,
            std::vector<DataTensor_out*>& outputs,
            PoolingParam<AMD>& param) override;

private:
    int _in_n_stride;
    int _in_c_stride;
    int _out_n_stride;
    int _out_c_stride;

    AMDKernelPtr _kernel_unpoo1;
};
template class SaberUnpool<AMD, AK_FLOAT>;
} // namespace saber
} // namespace anakin
#endif // ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_UNPOOL_H
