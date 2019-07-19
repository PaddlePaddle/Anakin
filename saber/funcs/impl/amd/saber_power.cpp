/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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
#include "include/saber_power.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberPower<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PowerParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberPower<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PowerParam<AMD>& param,
    Context<AMD>& ctx) {

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "create";

    const int count = outputs[0]->valid_size();
    Shape shape({inputs[0]->dims(), 1, 1, 1});
    _in_steps.re_alloc(shape);
    _out_steps.re_alloc(shape);
    _out_valid_shape.re_alloc(shape);

    Shape in_stride       = inputs[0]->get_stride();
    Shape out_stride      = outputs[0]->get_stride();
    Shape out_valid_shape = outputs[0]->valid_shape();

    AMD_API::sync_memcpy(
        _out_steps.data(),
        0,
        inputs[0]->device_id(),
        &out_stride[0],
        0,
        0,
        sizeof(int) * 4,
        __HtoD());
    AMD_API::sync_memcpy(
        _in_steps.data(),
        0,
        inputs[0]->device_id(),
        &in_stride[0],
        0,
        0,
        sizeof(int) * 4,
        __HtoD());
    AMD_API::sync_memcpy(
        _out_valid_shape.data(),
        0,
        inputs[0]->device_id(),
        &out_valid_shape[0],
        0,
        0,
        sizeof(int) * 4,
        __HtoD());
    const float power = param.power;

    KernelInfo kernelInfo;
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {256};
    kernelInfo.g_wk        = {(count + 256 - 1) / 256 * 256};
    kernelInfo.kernel_file = "Power.cl";

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        if (power == 1) {
            if ((count > 3) && (0 == count % 4)) {
                kernelInfo.g_wk        = {(count / 4 + 256 - 1) / 256 * 256};
                kernelInfo.kernel_name = "ker_scale_fwd_f4";
            } else {
                kernelInfo.kernel_name = "ker_scale_fwd";
            }
        } else {
            if ((count > 3) && (0 == count % 4)) {
                kernelInfo.g_wk        = {(count / 4 + 256 - 1) / 256 * 256};
                kernelInfo.kernel_name = "ker_power_fwd_f4";
            } else {
                kernelInfo.kernel_name = "ker_power_fwd";
            }
        }
    } else {
        if (power == 1) {
            kernelInfo.kernel_name = "ker_scale_stride_fwd";
        } else {
            kernelInfo.kernel_name = "ker_power_stride_fwd";
        }
    }

    AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    _kernel_ptr = kptr;

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberPower<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PowerParam<AMD>& param) {
    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    Shape shape({inputs[0]->dims(), 1, 1, 1});
    _in_steps.re_alloc(shape);
    _out_steps.re_alloc(shape);
    _out_valid_shape.re_alloc(shape);
    int count         = outputs[0]->valid_size();
    const float scale = param.scale;
    const float shift = param.shift;
    const float power = param.power;

    bool err = false;

    if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
        LOG(ERROR) << "Kernel is not exist";
        return SaberInvalidValue;
    }

    AMDKernel* kernel = _kernel_ptr.get();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        if (power == 1) {
            if ((count > 3) && (0 == count % 4)) {
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)(count / 4),
                          (float)scale,
                          (float)shift,
                          (PtrDtype)inputs[0]->data());
            } else {
                // To set the argument
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)count,
                          (float)scale,
                          (float)shift,
                          (PtrDtype)inputs[0]->data());
            }
        } else {
            if ((count > 3) && (0 == count % 4)) {
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)(count / 4),
                          (float)scale,
                          (float)shift,
                          (float)power,
                          (PtrDtype)inputs[0]->data());
            } else {
                // To set the argument
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)count,
                          (float)scale,
                          (float)shift,
                          (float)power,
                          (PtrDtype)inputs[0]->data());
            }
        }
    } else {
        if (power == 1) {
            // To set the argument
            err = kernel->SetKernelArgs(
                      (PtrDtype)outputs[0]->mutable_data(),
                      (int)count,
                      (float)scale,
                      (float)shift,
                      (PtrDtype)_out_valid_shape.data(),
                      (PtrDtype)_out_steps.data(),
                      (PtrDtype)_in_steps.data(),
                      (int)outputs[0]->dims(),
                      (PtrDtype)inputs[0]->data());
        } else {
            // To set the argument
            err = kernel->SetKernelArgs(
                      (PtrDtype)outputs[0]->mutable_data(),
                      (int)count,
                      (float)scale,
                      (float)shift,
                      (float)power,
                      (PtrDtype)_out_valid_shape.data(),
                      (PtrDtype)_out_steps.data(),
                      (PtrDtype)_in_steps.data(),
                      (int)outputs[0]->dims(),
                      (PtrDtype)inputs[0]->data());
        }
    }

    if (!err) {
        LOG(ERROR) << "Failed to set kernel args";
        return SaberInvalidValue;
    }

    amd_kernel_list list;
    list.push_back(_kernel_ptr);
    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Failed to set execution";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}
template class SaberPower<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberPower, PowerParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberPower, PowerParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin
