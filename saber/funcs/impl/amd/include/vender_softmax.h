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

#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_VENDER_SOFTMAX_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_VENDER_SOFTMAX_H

#include "saber/funcs/base.h"
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/funcs/impl/impl_softmax.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"

#include "saber_softmax.h"

#define AMD_NUM_THREADS 256
namespace anakin {

namespace saber {

template <DataType OpDtype>
class VenderSoftmax<AMD, OpDtype> : public ImplBase<AMD, OpDtype, SoftmaxParam<AMD>> {
public:
    typedef TargetWrapper<AMD> API;
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;

    VenderSoftmax() = default;

    ~VenderSoftmax() {
        _kernels.clear();
        if (_saber_impl)
        {
            delete _saber_impl;
            _saber_impl = nullptr;
        }
    }

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         SoftmaxParam<AMD>& param,
         Context<AMD>& ctx) override;

    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           SoftmaxParam<AMD>& param,
           Context<AMD>& ctx) override;

    virtual SaberStatus dispatch(
            const std::vector<Tensor<AMD>*>& inputs,
            std::vector<Tensor<AMD>*>& outputs,
            SoftmaxParam<AMD>& param) override;

private:
    CreateKernelList(int device_id, KernelInfo& kernelInfo);

    //! get maximum size to select which softmax kernel to call
    //! _max_dimsize is compute from shared memory size
    bool _is_continue_buf{true};
    int _max_dimsize;
    int _inner_num;
    int _outer_num;
    int _axis_size;
    int _dims;
    Tensor<AMD> _input_stride;
    Tensor<AMD> _output_stride;
    Tensor<AMD> _valid_shape;

    Tensor<AMD> _max_data;
    Tensor<AMD> _sum_data;

    amd_kernel_list _kernels;

    SaberSoftmax<AMD, OpDtype>* _saber_impl{nullptr};
};
template class VenderSoftmax<AMD, AK_FLOAT>;
} // namespace saber

} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_SOFTMAX_H
