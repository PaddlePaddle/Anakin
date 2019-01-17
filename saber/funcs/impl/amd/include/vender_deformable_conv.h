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

#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_VENDER_DEFORMABLE_CONV_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_VENDER_DEFORMABLE_CONV_H

#include "saber/funcs/impl/impl_deformable_conv.h"
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/funcs/base.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"
#include "saber/funcs/funcs_utils.h"
#include "saber/funcs/impl/amd/include/amd_utils.h"

#include <miopengemm/miogemm.hpp>
#include <miopengemm/gemm.hpp>
#include <miopengemm/geometry.hpp>

namespace anakin {

namespace saber {

template <DataType OpDtype>
class VenderDeformableConv2D<AMD, OpDtype> : public ImplBase<AMD, OpDtype, ConvParam<AMD>> {
public:
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;

    typedef ImplBase<AMD, OpDtype, ConvParam<AMD>> Impl_t;
    VenderDeformableConv2D() :
        _conv_out_spatial_dim(0),
        _kernel_dim(0),
        _bottom_dim(0),
        _offset_dim(0),
        _col_offset(0),
        _output_offset(0),
        _kernel_offset(0) {
        _kernels_ptr.clear();
    }

    ~VenderDeformableConv2D() {
        _kernels_ptr.clear();
    }

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         ConvParam<AMD>& param,
         Context<AMD>& ctx) override;

    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           ConvParam<AMD>& param,
           Context<AMD>& ctx) override;

    virtual SaberStatus dispatch(
        const std::vector<Tensor<AMD>*>& inputs,
        std::vector<Tensor<AMD>*>& outputs,
        ConvParam<AMD>& param) override;

    SaberStatus trans_weights(
        Tensor<AMD>& target_weights,
        Tensor<AMD>& target_bias,
        int in_channel,
        int out_channel,
        int stride_h,
        int stride_w,
        int pad_h,
        int pad_w,
        int dilation_h,
        int dilation_w,
        int group);

private:
    amd_kernel_list _kernels;
    std::vector<AMDKernelPtr> _kernels_ptr;

    Tensor<AMD> _deform_col_buffer;

    int _conv_out_spatial_dim;
    int _kernel_dim;
    int _bottom_dim;
    int _offset_dim;
    int _col_offset;
    int _output_offset;
    int _kernel_offset;

    std::vector<cl_program> _programs;
    size_t _x_t_size;
    bool _multikernel;
    Tensor<AMD>* _outGemmWorkspace;
    Tensor<AMD>* _outCol2ImSpace;
};

template class VenderDeformableConv2D<AMD, AK_FLOAT>;
} // namespace saber

} // namespace anakin
#endif // ANAKIN_SABER_FUNCS_IMPL_AMD_VENDER_DEFORMABLE_CONV_H

