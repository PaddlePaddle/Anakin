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
#include "include/saber_unpool.h"
namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberUnpool<AMD, OpDtype>::init(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    PoolingParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberUnpool<AMD, OpDtype>::create(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    PoolingParam<AMD>& param,
    Context<AMD>& ctx) {
    Shape out_stride = outputs[0]->get_stride();
    Shape in_stride  = inputs[0]->get_stride();
    int in_n_index   = inputs[0]->num_index();
    int in_c_index   = inputs[0]->channel_index();
    int out_n_index  = outputs[0]->num_index();
    int out_c_index  = outputs[0]->channel_index();
    _in_n_stride     = in_stride[in_n_index];
    _in_c_stride     = in_stride[in_c_index];
    _out_n_stride    = out_stride[out_n_index];
    _out_c_stride    = out_stride[out_c_index];

    const int count = outputs[0]->size();
    int globalsize  = inputs[0]->size();
    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "Unpool.cl";
    kernelInfo.kernel_name = "Unpool";
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {AMD_NUM_THREADS};
    kernelInfo.g_wk        = {
        (count + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]* kernelInfo.l_wk[0], 1, 1
    };
    AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    _kernel_unpoo1 = kptr;

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberUnpool<AMD, OpDtype>::dispatch(
    const std::vector<DataTensor_in*>& inputs,
    std::vector<DataTensor_out*>& outputs,
    PoolingParam<AMD>& param) {

    bool err;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    amd_kernel_list list;

    AMDKernel* kernel = _kernel_unpoo1.get();
    // To set the argument
    cl_mem memObjects[3];
    memObjects[0] = (cl_mem)inputs[0]->data();
    memObjects[1] = (cl_mem)inputs[1]->data();
    memObjects[2] = (cl_mem)outputs[0]->mutable_data();

    int count = inputs[0]->valid_size();
    int in_n  = inputs[0]->num();
    int in_c  = inputs[0]->channel();

    kernel->SetKernelArgs(
        memObjects[2],
        memObjects[0],
        memObjects[1],
        _in_n_stride,
        _in_c_stride,
        _out_n_stride,
        _out_c_stride,
        in_n,
        in_c,
        count);
    list.push_back(_kernel_unpoo1);

    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fialed to set execution.";
        return SaberInvalidValue;
    }

    list.clear();

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}

} // namespace saber
} // namespace anakin
