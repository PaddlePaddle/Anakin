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

#include <limits>
#include "include/saber_embedding.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberEmbedding<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    EmbeddingParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberEmbedding<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    EmbeddingParam<AMD>& param,
    Context<AMD>& ctx) {
    const int count = outputs[0]->valid_size();
    KernelInfo kernelInfo;
    kernelInfo.l_wk   = {AMD_NUM_THREADS};
    kernelInfo.wk_dim = 1;
    kernelInfo.g_wk = {(count + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]* kernelInfo.l_wk[0]};
    kernelInfo.kernel_file  = "Embedding.cl";
    kernelInfo.kernel_name  = "Embedding";
    _kernel_Scale_Embedding = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!_kernel_Scale_Embedding.get()->isInit()) {
        LOG(ERROR) << "Failed to load _kernel_Scale_Embedding ";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberEmbedding<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    EmbeddingParam<AMD>& param) {
    bool err = false;
    amd_kernel_list list;
    CHECK_EQ(inputs[0]->get_dtype(), AK_FLOAT) << " Embedding only support float inputs.";

    const int count = outputs[0]->valid_size();
    // To get the commpute command queue
    AMD_API::stream_t cm      = this->_ctx->get_compute_stream();
    const OpDataType* op_data = (const OpDataType*)(param.weight()->data());
    OpDataType* mutable_data  = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* inputs_data   = (OpDataType*)inputs[0]->data();
    AMDKernel* kernel         = _kernel_Scale_Embedding.get();
    err                       = kernel->SetKernelArgs(
                                    (PtrDtype)mutable_data,
                                    (PtrDtype)inputs_data,
                                    (PtrDtype)op_data,
                                    (int)param.emb_dim,
                                    (int)inputs[0]->num(),
                                    (int)param.padding_idx,
                                    (int)outputs[0]->valid_size());

    if (!err) {
        LOG(ERROR) << "Fail to set _kernel_Scale_Embedding->SetKernelArgs";
        return SaberInvalidValue;
    }

    list.push_back(_kernel_Scale_Embedding);
    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}
} // namespace saber
} // namespace anakin
