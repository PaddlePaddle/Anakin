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
#include "include/saber_permute_power.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberPermutePower<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PermutePowerParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberPermutePower<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PermutePowerParam<AMD>& param,
    Context<AMD>& ctx) {

    int count = outputs[0]->valid_size();

    _num_axes                       = inputs[0]->shape().size();
    PermuteParam<AMD> permute_param = param.permute_param;

    for (int i = 0; i < _num_axes; i++) {
        if (std::find(_order_dims.begin(), _order_dims.end(), permute_param.order[i])
                == _order_dims.end()) {
            _order_dims.push_back(permute_param.order[i]);
        }
    }

    CHECK_EQ(_num_axes, _order_dims.size());

    // set _need_permute
    _need_permute = false;

    for (int i = 0; i < _num_axes; ++i) {
        if (permute_param.order[i] != i) {
            _need_permute = true;
            break;
        }
    }

    Shape order_shape({_num_axes, 1, 1, 1});
    _permute_order.re_alloc(order_shape);
    _old_steps.re_alloc(order_shape);
    _new_steps.re_alloc(order_shape);
    _out_valid_shape.re_alloc(order_shape);
    Shape in_stride                  = inputs[0]->get_stride();
    Shape out_stride                 = outputs[0]->get_stride();
    Shape out_valid_shape            = outputs[0]->valid_shape();
    const float power                = param.has_power_param ? param.power_param.power : 1.0f;
    std::vector<int> permute_order_t = {0, 3, 1, 2};
    PermuteParam<AMD> param_t(permute_order_t);

    AMD_API::sync_memcpy(
        _old_steps.mutable_data(),
        0,
        inputs[0]->device_id(),
        &in_stride[0],
        0,
        0,
        sizeof(int) * _num_axes,
        __HtoD());
    AMD_API::sync_memcpy(
        _new_steps.mutable_data(),
        0,
        inputs[0]->device_id(),
        &out_stride[0],
        0,
        0,
        sizeof(int) * _num_axes,
        __HtoD());
    AMD_API::sync_memcpy(
        _permute_order.mutable_data(),
        0,
        inputs[0]->device_id(),
        &(permute_param.order[0]),
        0,
        0,
        sizeof(int) * _num_axes,
        __HtoD());
    AMD_API::sync_memcpy(
        _out_valid_shape.mutable_data(),
        0,
        inputs[0]->device_id(),
        &out_valid_shape[0],
        0,
        0,
        sizeof(int) * _num_axes,
        __HtoD());
    KernelInfo kernelInfo;
    kernelInfo.wk_dim      = 1;
    kernelInfo.kernel_file = "PermutePower.cl";

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {

        if (inputs[0]->num() == 1 && inputs[0]->width() == 3 && param.permute_param == param_t
                && 1) {
            int out_w = outputs[0]->width() * outputs[0]->height();
            int out_h = outputs[0]->channel();

            if (power != 1.0f) {
                kernelInfo.g_wk        = {(out_w + 256 - 1) / 256 * 256, 1, 1};
                kernelInfo.kernel_name = "ker_permute_power_fwd_transpose";
            } else {
                kernelInfo.g_wk        = {(out_w + 256 - 1) / 256 * 256, 1, 1};
                kernelInfo.kernel_name = "ker_permute_scale_fwd_transpose";
            }
        } else {
            if (power != 1.0f) {
                kernelInfo.g_wk        = {(count + 256 - 1) / 256 * 256, 1, 1};
                kernelInfo.kernel_name = "ker_permute_power_fwd";
            } else {
                kernelInfo.g_wk        = {(count + 256 - 1) / 256 * 256, 1, 1};
                kernelInfo.kernel_name = "ker_permute_scale_fwd";
            }
        }
    } else {

        if (inputs[0]->width() == 3 && param.permute_param == param_t) {
            if (power != 1.0f) {
                kernelInfo.g_wk        = {(count + 256 - 1) / 256 * 256, 1, 1};
                kernelInfo.kernel_name = "ker_nhwc_to_nchw_power";
            } else {
                kernelInfo.g_wk        = {(count + 256 - 1) / 256 * 256, 1, 1};
                kernelInfo.kernel_name = "ker_nhwc_to_nchw_scale";
            }
        } else {
            int count = outputs[0]->valid_size();

            if (power != 1.0f) {
                kernelInfo.g_wk        = {(count + 256 - 1) / 256 * 256, 1, 1};
                kernelInfo.kernel_name = "ker_permute_power_valid_fwd";
            } else {
                kernelInfo.g_wk        = {(count + 256 - 1) / 256 * 256, 1, 1};
                kernelInfo.kernel_name = "ker_permute_scale_valid_fwd";
            }
        }
    }

    AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!kptr.get()->isInit()) {
        ALOGE("Failed to load program");
        return SaberInvalidValue;
    }

    _kernel_ptr = kptr;

    ALOGD("COMPLETE CREATE KERNEL");

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberPermutePower<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PermutePowerParam<AMD>& param) {
    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    int count                        = outputs[0]->valid_size();
    const float scale                = param.has_power_param ? param.power_param.scale : 1.0f;
    const float shift                = param.has_power_param ? param.power_param.shift : 0.0f;
    const float power                = param.has_power_param ? param.power_param.power : 1.0f;
    std::vector<int> permute_order_t = {0, 3, 1, 2};
    PermuteParam<AMD> param_t(permute_order_t);

    bool err = false;

    if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
        ALOGE("Kernel is not exist");
        return SaberInvalidValue;
    }

    AMDKernel* kernel = _kernel_ptr.get();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {

        if (inputs[0]->num() == 1 && inputs[0]->width() == 3 && param.permute_param == param_t
                && 1) {
            int out_w = outputs[0]->width() * outputs[0]->height();
            int out_h = outputs[0]->channel();

            // To set the argument
            if (power != 1.0f) {
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)out_h,
                          (int)out_w,
                          (float)scale,
                          (float)shift,
                          (float)power,
                          (PtrDtype)inputs[0]->data());
            } else {
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)out_h,
                          (int)out_w,
                          (float)scale,
                          (float)shift,
                          (PtrDtype)inputs[0]->data());
            }
        } else {
            // To set the argument
            if (power != 1.0f) {
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)_num_axes,
                          (int)count,
                          (PtrDtype)_permute_order.data(),
                          (PtrDtype)_new_steps.data(),
                          (PtrDtype)_old_steps.data(),
                          (float)scale,
                          (float)shift,
                          (float)power,
                          (PtrDtype)inputs[0]->data());
            } else {
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)_num_axes,
                          (int)count,
                          (PtrDtype)_permute_order.data(),
                          (PtrDtype)_new_steps.data(),
                          (PtrDtype)_old_steps.data(),
                          (float)scale,
                          (float)shift,
                          (PtrDtype)inputs[0]->data());
            }
        }
    } else {

        if (inputs[0]->width() == 3 && param.permute_param == param_t) {
            const int out_n  = outputs[0]->num();
            const int out_c  = outputs[0]->channel();
            const int out_h  = outputs[0]->height();
            const int out_w  = outputs[0]->width();
            const int count  = out_n * out_h * out_w;
            Shape out_stride = outputs[0]->get_stride();
            Shape in_stride  = inputs[0]->get_stride();

            // To set the argument
            if (power != 1.0f) {
                err = kernel->SetKernelArgs(
                          (int)out_n,
                          (int)out_c,
                          (int)out_h,
                          (int)out_w,
                          (int)out_stride[0],
                          (int)out_stride[1],
                          (int)out_stride[2],
                          (int)out_stride[3],
                          (int)in_stride[0],
                          (int)in_stride[3],
                          (int)in_stride[1],
                          (int)in_stride[2],
                          (float)scale,
                          (float)shift,
                          (float)power,
                          (PtrDtype)inputs[0]->data());
            } else {
                err = kernel->SetKernelArgs(
                          (int)out_n,
                          (int)out_c,
                          (int)out_h,
                          (int)out_w,
                          (int)out_stride[0],
                          (int)out_stride[1],
                          (int)out_stride[2],
                          (int)out_stride[3],
                          (int)in_stride[0],
                          (int)in_stride[3],
                          (int)in_stride[1],
                          (int)in_stride[2],
                          (float)scale,
                          (float)shift,
                          (PtrDtype)inputs[0]->data());
            }
        } else {
            int count = outputs[0]->valid_size();

            // To set the argument
            if (power != 1.0f) {
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)_num_axes,
                          (int)count,
                          (PtrDtype)_permute_order.data(),
                          (PtrDtype)_new_steps.data(),
                          (PtrDtype)_old_steps.data(),
                          (PtrDtype)_out_valid_shape.data(),
                          (float)scale,
                          (float)shift,
                          (float)power,
                          (PtrDtype)inputs[0]->data());
            } else {
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)_num_axes,
                          (int)count,
                          (PtrDtype)_permute_order.data(),
                          (PtrDtype)_new_steps.data(),
                          (PtrDtype)_old_steps.data(),
                          (PtrDtype)_out_valid_shape.data(),
                          (float)scale,
                          (float)shift,
                          (PtrDtype)inputs[0]->data());
            }
        }
    }

    if (!err) {
        ALOGE("Failed to set kernel args");
        return SaberInvalidValue;
    }

    amd_kernel_list list;
    list.push_back(_kernel_ptr);
    err = LaunchKernel(cm, list);

    if (!err) {
        ALOGE("Failed to set execution");
        return SaberInvalidValue;
    }

    ALOGD("COMPLETE EXECUTION");
    return SaberSuccess;
}
template class SaberPermutePower<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberPermutePower, PermutePowerParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberPermutePower, PermutePowerParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin
