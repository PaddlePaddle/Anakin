/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
*/

#include "include/saber_layer_norm.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberLayerNorm<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    LayerNormParam<AMD>& param,
    Context<AMD>& ctx) {
    // get context
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberLayerNorm<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    LayerNormParam<AMD>& param,
    Context<AMD>& ctx) {
    // Shape sh_in = inputs[0]->valid_shape();
    _inner_size = inputs[0]->count_valid(param.axis, inputs[0]->dims());
    _outer_size = inputs[0]->count_valid(0, param.axis);

    Shape sh({0, 0, 0, 0});

    for (int i = 0; i < sh.dims(); ++i) {
        sh[i] = 1;
    }

    sh[0] = _outer_size;
    _mean.reshape(sh);
    _std.reshape(sh);

    if (param.scale_weights()->valid_size() == 0) {
        _flag_scale = false;
    } else {
        _flag_scale = true;
    }

    if (param.bias_weights()->valid_size() == 0) {
        _flag_bias = false;
    } else {
        _flag_bias = true;
    }

    int total_size = inputs[0]->valid_size();

    KernelInfo kernelInfo;
    kernelInfo.kernel_file  = "Layer_norm.cl";
    kernelInfo.kernel_name  = "reduce_mean";
    kernelInfo.wk_dim       = 1;
    kernelInfo.l_wk         = {AMD_NUM_THREADS};
    kernelInfo.g_wk         = {(_outer_size * 256 + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]
                               * kernelInfo.l_wk[0]
                              };
    _kernel_ptr_reduce_mean = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!_kernel_ptr_reduce_mean.get()->isInit()) {
        LOG(ERROR) << "Failed to load _kernel_ptr_reduce_mean";
        return SaberInvalidValue;
    }

    // To create _kernel_ptr_reduce_std
    kernelInfo.kernel_name = "reduce_std";
    _kernel_ptr_reduce_std = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!_kernel_ptr_reduce_std.get()->isInit()) {
        LOG(ERROR) << "Failed to load _kernel_ptr_reduce_std";
        return SaberInvalidValue;
    }

    // To create _kernel_ptr_normalize_with_scale_bias
    kernelInfo.kernel_name = "normalize_with_scale_bias";
    kernelInfo.l_wk        = {AMD_NUM_THREADS, 1, 1};
    kernelInfo.g_wk        = {(total_size + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]
                              * kernelInfo.l_wk[0]
                             };
    _kernel_ptr_normalize_with_scale_bias = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!_kernel_ptr_normalize_with_scale_bias.get()->isInit()) {
        LOG(ERROR) << "Failed to load _kernel_ptr_normalize_with_scale_bias";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberLayerNorm<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    LayerNormParam<AMD>& param) {
    bool err;
    amd_kernel_list list;
    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    int total_size       = inputs[0]->valid_size();

    // To set the argument
    const OpDataType* src = (const OpDataType*)inputs[0]->data();
    OpDataType* dst       = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* mean_ptr  = (OpDataType*)_mean.mutable_data();
    OpDataType* std_ptr   = (OpDataType*)_std.mutable_data();

    const OpDataType* scale_ptr = (const OpDataType*)param.scale_weights()->data();
    const OpDataType* bias_ptr  = (const OpDataType*)param.bias_weights()->data();

    //_kernel_ptr_reduce_mean->SetKernelArgs
    AMDKernel* kernel = _kernel_ptr_reduce_mean.get();
    err               = kernel->SetKernelArgs(
                            (int)total_size, (int)_inner_size, (PtrDtype)src, (PtrDtype)mean_ptr);

    if (!err) {
        LOG(ERROR) << "Fail to set _kernel_ptr_reduce_mean->SetKernelArgs";
        return SaberInvalidValue;
    }

    list.push_back(_kernel_ptr_reduce_mean);
    //_kernel_ptr_reduce_std->SetKernelArgs
    kernel = _kernel_ptr_reduce_std.get();
    err    = kernel->SetKernelArgs(
                 (int)total_size,
                 (int)_inner_size,
                 (OpDataType)param.eps,
                 (PtrDtype)src,
                 (PtrDtype)mean_ptr,
                 (PtrDtype)std_ptr);

    if (!err) {
        LOG(ERROR) << "Fail to set _kernel_ptr_reduce_mean->SetKernelArgs";
        return SaberInvalidValue;
    }

    list.push_back(_kernel_ptr_reduce_std);
    // _kernel_ptr_normalize_with_scale_bias
    int flag_scale, flag_bias;

    if (_flag_scale) {
        flag_scale = 1;
    } else {
        flag_scale = 0;
    }

    if (_flag_bias) {
        flag_bias = 1;
    } else {
        flag_bias = 0;
    }

    kernel = _kernel_ptr_normalize_with_scale_bias.get();

    err = kernel->SetKernelArgs(
              (int)total_size,
              (int)_inner_size,
              (PtrDtype)mean_ptr,
              (PtrDtype)std_ptr,
              (PtrDtype)scale_ptr,
              (PtrDtype)bias_ptr,
              (PtrDtype)src,
              (PtrDtype)dst,
              (int)flag_scale,
              (int)flag_bias);

    if (!err) {
        LOG(ERROR) << "Fail to set _kernel_ptr_normalize_with_scale_bias->SetKernelArgs";
        return SaberInvalidValue;
    }

    list.push_back(_kernel_ptr_normalize_with_scale_bias);

    // EXECUTION kernes in _kernel_list
    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}

template class SaberLayerNorm<AMD, AK_FLOAT>;

} // namespace saber
} // namespace anakin
