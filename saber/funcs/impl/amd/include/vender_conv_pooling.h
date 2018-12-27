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
#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_VENDER_CONV_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_VENDER_CONV_POOLING_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_conv_pooling.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class VenderConv2DPooling<AMD, OpDtype> : public ImplBase<AMD, OpDtype, ConvPoolingParam<AMD>> {
public:
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;

    VenderConv2DPooling() {
        _kernels_ptr.clear();
        vkernel.clear();
        _outConvRelu = nullptr;
        _outGemmWorkspace = nullptr;
    }
    ~VenderConv2DPooling() {
        _kernels_ptr.clear();
        vkernel.clear();

        if (_outConvRelu) {
            delete _outConvRelu;
        }

        if (_outGemmWorkspace) {
            delete _outGemmWorkspace;
        }
    }

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         ConvPoolingParam<AMD>& param,
         Context<AMD>& ctx);

    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           ConvPoolingParam<AMD>& param,
           Context<AMD>& ctx);

    virtual SaberStatus dispatch(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        ConvPoolingParam<AMD>& param);

    SaberStatus trans_weights(Tensor<AMD>& target_weights, Tensor<AMD>& target_bias,
                              int pad_h, int pad_w, int dilation_h, int dilation_w,
                              int stride_h, int stride_w, int group) {
        if (target_weights.valid_size() > 0) {
            // conv_trans_weights<AMD, AMDHX86>(
            //        target_weights, stride_h, stride_w, group, true, nullptr);
        }

        _extern_trans = true;
        _in_place     = true;
        return SaberSuccess;
    }

    void set_solution(std::vector<KernelInfo> vkernel1) {
        if (!vkernel1.empty()) {
            vkernel.assign(vkernel1.begin(), vkernel1.end());
        }
    }

private:
    bool _in_place {false};
    bool _extern_trans {false};
    void CreateKernelList(int device_id, KernelInfo& kernelInfo);
    std::vector<AMDKernelPtr> _kernels_ptr;
    Tensor<AMD>* _outConvRelu;
    Tensor<AMD>* _outGemmWorkspace;
    std::vector<KernelInfo> vkernel;
    bool impl_vender {false};
};
} // namespace saber
} // namespace anakin
#endif // ANAKIN_SABER_FUNCS_IMPL_AMD_VENDER_CONV_POOLING_H
