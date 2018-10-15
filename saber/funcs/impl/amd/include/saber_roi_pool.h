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

#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_ROI_POOL_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_ROI_POOL_H

#include "saber/funcs/impl/impl_roi_pooling.h"
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
class SaberRoiPool<AMD, OpDtype> : public ImplBase<AMD, OpDtype, RoiPoolParam<AMD>> {

public:
    typedef Tensor<AMD> OpTensor;
    typedef TargetWrapper<AMD> API;
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;

    SaberRoiPool() {}

    ~SaberRoiPool() {}

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         RoiPoolParam<AMD>& param,
         Context<AMD>& ctx) override;
    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           RoiPoolParam<AMD>& param,
           Context<AMD>& ctx) override;
    virtual SaberStatus dispatch(
            const std::vector<Tensor<AMD>*>& inputs,
            std::vector<Tensor<AMD>*>& outputs,
            RoiPoolParam<AMD>& param) override;

private:
    int _in_n_stride;
    int _in_c_stride;
    int _in_h_stride;
    int _in_w_stride;
    int _out_n_stride;
    int _out_c_stride;
    int _out_h_stride;
    int _out_w_stride;

    AMDKernelPtr _kernel_roiPoo1;
};
template class SaberRoiPool<AMD, AK_FLOAT>;
} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ROI_POOL_H
