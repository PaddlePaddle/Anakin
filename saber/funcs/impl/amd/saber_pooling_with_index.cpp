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

#include "include/saber_pooling_with_index.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberPoolingWithIndex<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PoolingParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberPoolingWithIndex<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PoolingParam<AMD>& power_param,
    Context<AMD>& ctx) {

    const int count = outputs[0]->size();

    Shape out_stride = outputs[0]->get_stride();
    Shape in_stride  = inputs[0]->get_stride();
    int in_n_index   = inputs[0]->num_index();
    int in_c_index   = inputs[0]->channel_index();
    int in_h_index   = inputs[0]->height_index();
    int in_w_index   = inputs[0]->width_index();
    int out_n_index  = outputs[0]->num_index();
    int out_c_index  = outputs[0]->channel_index();
    int out_h_index  = outputs[0]->height_index();
    int out_w_index  = outputs[0]->width_index();
    _in_n_stride     = in_stride[in_n_index];
    _in_c_stride     = in_stride[in_c_index];
    _in_h_stride     = in_stride[in_h_index];
    _in_w_stride     = in_stride[in_w_index];
    _out_n_stride    = out_stride[out_n_index];
    _out_c_stride    = out_stride[out_c_index];
    _out_h_stride    = out_stride[out_h_index];
    _out_w_stride    = out_stride[out_w_index];

    KernelInfo kernelInfo;

    kernelInfo.kernel_file = "Pooling_with_index.cl";
    kernelInfo.kernel_name = "Pooling_with_index";
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

    _kernel_poo1ing_with_index = kptr;

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberPoolingWithIndex<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PoolingParam<AMD>& param) {
    bool err;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    amd_kernel_list list;

    // To set the argument
    cl_mem memObjects[3];
    cl_mem in_data   = (cl_mem)inputs[0]->data();
    cl_mem out_data  = (cl_mem)outputs[0]->mutable_data();
    cl_mem out_index = (cl_mem)outputs[1]->mutable_data();

    // To set the argument
    int count = outputs[0]->valid_size();
    int out_n = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    int in_h  = inputs[0]->height();
    int in_w  = inputs[0]->width();

    AMDKernel* kernel = _kernel_poo1ing_with_index.get();
    kernel->SetKernelArgs(
        out_data,
        out_index,
        in_data,
        _in_n_stride,
        _in_c_stride,
        _in_h_stride,
        _in_w_stride,
        in_h,
        in_w,
        _out_n_stride,
        _out_c_stride,
        _out_h_stride,
        _out_w_stride,
        out_h,
        out_w,
        out_n,
        out_c,
        param.pad_h,
        param.pad_w,
        param.stride_h,
        param.stride_w,
        param.window_h,
        param.window_w,
        count);
    list.push_back(_kernel_poo1ing_with_index);

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
