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

#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_VENDER_CONV2D_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_VENDER_CONV2D_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_conv.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class VenderConv2D<AMD, OpDtype> : public ImplBase <
        AMD, OpDtype, ConvParam<AMD> > {
public:
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;

    VenderConv2D() {
        _outGemmWorkspace = nullptr;
        _slot = nullptr;
        _kernels_ptr.clear();
        vkernel.clear();
    }

    ~VenderConv2D() {
        _kernels_ptr.clear();
        vkernel.clear();

        if (_outGemmWorkspace) {
            delete _outGemmWorkspace;
        }

        if (_slot) {
            delete _slot;
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<AMD> *>& inputs,
                             std::vector<Tensor<AMD> *>& outputs,
                             ConvParam<AMD>& param, Context<AMD>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<AMD> *>& inputs,
                               std::vector<Tensor<AMD> *>& outputs,
                               ConvParam<AMD>& param, Context<AMD>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<AMD>*>& inputs,
                                 std::vector<Tensor<AMD>*>& outputs,
                                 ConvParam<AMD>& param);

    SaberStatus trans_weights(Tensor<AMD>& target_weights, Tensor<AMD>& target_bias,
                              int pad_h, int pad_w, int dilation_h, int dilation_w,
                              int stride_h, int stride_w, int group) {
        return SaberUnImplError;
    }

    void set_solution(std::vector<KernelInfo> vkernel1) {
        if (!vkernel1.empty()) {
            vkernel.assign(vkernel1.begin(), vkernel1.end());
        }
    }

private:
    std::vector<AMDKernelPtr> _kernels_ptr;
    Tensor<AMD>* _outGemmWorkspace;
    Tensor<AMD>* _slot;
    Tensor<AMD> _tensile_bias;
    void CreateKernelList(int device_id, KernelInfo& kernelInfo);
    std::vector<KernelInfo> vkernel;
    bool impl_vender {false};
};

}
}
#endif //ANAKIN_SABER_FUNCS_AMD_VENDER_CONV2D_H
