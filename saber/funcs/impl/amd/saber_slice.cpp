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

#include "include/saber_slice.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberSlice<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    SliceParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberSlice<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    SliceParam<AMD>& param,
    Context<AMD>& ctx) {
    int output_size = outputs.size();
    _slice_num      = inputs[0]->count_valid(0, param.axis);
    _slice_size     = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());

    const int count = outputs[0]->size();
    int globalsize  = inputs[0]->size();
    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "Slice.cl";
    kernelInfo.kernel_name = "Slice_normal";
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {AMD_NUM_THREADS};
    kernelInfo.g_wk        = {(globalsize + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]
                              * kernelInfo.l_wk[0]
                             };
    AMDKernelPtr kptr      = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    _kernel_Slice_normal = kptr;
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}
template <DataType OpDtype>
SaberStatus SaberSlice<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    SliceParam<AMD>& param) {
    bool err;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    //! inputs only has one tensor
    Shape shape_in  = inputs[0]->valid_shape();
    int output_size = outputs.size();

    if (output_size == 1) {
        outputs[0]->share_from(*inputs[0]);
        return SaberSuccess;
    }

    int offset_slice_axis        = 0;
    const int in_slice_axis_size = shape_in[param.axis];
    const OpDataType* in_data    = (const OpDataType*)inputs[0]->data();
    amd_kernel_list list;

    for (int i = 0; i < output_size; ++i) {
        OpDataType* out_data          = (OpDataType*)outputs[i]->mutable_data();
        const int out_slice_axis_size = outputs[i]->valid_shape()[param.axis];
        const int out_slice_size      = out_slice_axis_size * _slice_size;
        const int nthreads            = out_slice_size * _slice_num;
        AMDKernel* kernel             = _kernel_Slice_normal.get();

        kernel->SetKernelArgs(
            (int)nthreads,
            (PtrDtype)in_data,
            (int)_slice_num,
            (int)_slice_size,
            (int)in_slice_axis_size,
            (int)out_slice_axis_size,
            (int)offset_slice_axis,
            (PtrDtype)out_data);

        list.push_back(_kernel_Slice_normal);
        offset_slice_axis += out_slice_axis_size;
        err = LaunchKernel(cm, list);

        if (!err) {
            LOG(ERROR) << "Fialed to set execution.";
            return SaberInvalidValue;
        }

        list.clear();
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}

template class SaberSlice<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSlice, SliceParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberSlice, SliceParam, AMD, AK_HALF);
} // namespace saber

} // namespace anakin
