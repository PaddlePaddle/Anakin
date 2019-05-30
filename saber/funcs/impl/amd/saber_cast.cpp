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

#include "include/saber_cast.h"
namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberCast<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    CastParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberCast<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    CastParam<AMD>& param,
    Context<AMD>& ctx) {

    const int count = outputs[0]->size();
    _inDtype        = param.in_type;
    _outDtype       = param.out_type;

    if (_inDtype != 1 && _inDtype != 5) { // AK_FLOAT AK_INT32
        LOG(FATAL) << "Cast not impl other type: " << _inDtype;
    }

    if (_inDtype != 1 && _inDtype != 5) { // AK_FLOAT AK_INT32
        LOG(FATAL) << "Cast not impl other type: " << _outDtype;
    }

    CHECK_EQ(_inDtype, inputs[0]->get_dtype())
            << "inputs data type should be same with param.in_type";
    CHECK_EQ(_outDtype, outputs[0]->get_dtype())
            << "outputs data type should be same with param.out_type";

    KernelInfo kernelInfo;
    kernelInfo.l_wk = {AMD_NUM_THREADS};
    kernelInfo.g_wk = {(count + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]* kernelInfo.l_wk[0]};
    kernelInfo.wk_dim      = 1;
    kernelInfo.kernel_file = "Cast.cl";
    kernelInfo.kernel_name = "Cast";
    _kernel_cast           = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!_kernel_cast.get()->isInit()) {
        LOG(ERROR) << "Failed to load _kernel_cast ";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}
template <DataType OpDtype>
SaberStatus SaberCast<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    CastParam<AMD>& param) {
    bool err;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    //! inputs only has one tensor

    amd_kernel_list list;
    const int count = outputs[0]->size();
    cl_mem memObjects[2];
    memObjects[0] = (cl_mem)outputs[0]->mutable_data(); // out_data
    memObjects[1] = (cl_mem)inputs[0]->data();          // in_data

    if (_inDtype == _outDtype) {

        outputs[0]->copy_from(*inputs[0]);
        AMD_API::async_memcpy(
            outputs[0]->mutable_data(),
            0,
            (int)outputs[0]->device_id(),
            inputs[0]->data(),
            0,
            (int)inputs[0]->device_id(),
            inputs[0]->size(),
            cm,
            __DtoD());

        return SaberSuccess;
    }

    AMDKernel* kernel = _kernel_cast.get();
    kernel->SetKernelArgs(memObjects[0], memObjects[1], count);
    list.push_back(_kernel_cast);
    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fialed to set execution.";
        return SaberInvalidValue;
    }

    list.clear();

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}
// template class SaberCast<AMD, AK_FLOAT, AK_FLOAT, AK_INT32, NCHW, NCHW, NCHW>;
// template class SaberCast<AMD, AK_FLOAT, AK_INT32, AK_FLOAT, NCHW, NCHW, NCHW>;
} // namespace saber
} // namespace anakin
