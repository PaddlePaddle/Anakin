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

#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_AXPY_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_AXPY_H

#include "saber/funcs/impl/impl_axpy.h"
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
class SaberAxpy<AMD, OpDtype> : public ImplBase<AMD, OpDtype, AxpyParam<AMD>> {
public:
    typedef TargetWrapper<AMD> API;
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;

    SaberAxpy() = default;

    ~SaberAxpy() {
        _kernels.clear();
    }

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         AxpyParam<AMD>& param,
         Context<AMD>& ctx) override;

    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           AxpyParam<AMD>& param,
           Context<AMD>& ctx) override;
    virtual SaberStatus dispatch(
            const std::vector<Tensor<AMD>*>& inputs,
            std::vector<Tensor<AMD>*>& outputs,
            AxpyParam<AMD>& param) override;

private:
    CreateKernelList(int device_id, KernelInfo& kernelInfo);
    amd_kernel_list _kernels;
};

template class SaberAxpy<AMD, AK_FLOAT>;

} // namespace saber

} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_AXPY_H
