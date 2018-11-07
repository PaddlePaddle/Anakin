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
#include "include/saber_permute.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberPermute<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PermuteParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    _num_axes  = inputs[0]->valid_shape().size();

    for (int i = 0; i < _num_axes; i++) {
        if (std::find(_order_dims.begin(), _order_dims.end(), param.order[i])
                == _order_dims.end()) {
            _order_dims.push_back(param.order[i]);
        }
    }

    CHECK_EQ(_num_axes, _order_dims.size());

    // set _need_permute
    _need_permute = false;

    for (int i = 0; i < _num_axes; ++i) {
        if (param.order[i] != i) {
            _need_permute = true;
            break;
        }
    }

    Shape order_shape({_num_axes, 1, 1, 1});
    _permute_order.reshape(order_shape);
    AMD_API::sync_memcpy(
        _permute_order.mutable_data(),
        0,
        inputs[0]->device_id(),
        &(param.order[0]),
        0,
        0,
        sizeof(int) * _permute_order.size(),
        __HtoD());
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberPermute<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PermuteParam<AMD>& param,
    Context<AMD>& ctx) {

    int count = outputs[0]->valid_size();

    Shape order_shape({_num_axes, 1, 1, 1});
    _in_steps.reshape(order_shape);
    _out_steps.reshape(order_shape);
    _out_valid_shape.reshape(order_shape);

    Shape in_stride                             = inputs[0]->get_stride();
    Shape out_stride                            = outputs[0]->get_stride();
    std::vector<int> permute_order_nhwc_to_nchw = {0, 3, 1, 2};
    PermuteParam<AMD> param_nhwc_to_nchw(permute_order_nhwc_to_nchw);
    std::vector<int> permute_order_nchw_to_nhwc = {0, 2, 3, 1};
    PermuteParam<AMD> param_nchw_to_nhwc(permute_order_nchw_to_nhwc);

    AMD_API::sync_memcpy(
        _in_steps.mutable_data(),
        0,
        inputs[0]->device_id(),
        &in_stride[0],
        0,
        0,
        sizeof(int) * _in_steps.size(),
        __HtoD());
    AMD_API::sync_memcpy(
        _out_steps.mutable_data(),
        0,
        inputs[0]->device_id(),
        &out_stride[0],
        0,
        0,
        sizeof(int) * _out_steps.size(),
        __HtoD());
    AMD_API::sync_memcpy(
        _out_valid_shape.mutable_data(),
        0,
        inputs[0]->device_id(),
        &((outputs[0]->valid_shape())[0]),
        0,
        0,
        sizeof(int) * _out_valid_shape.size(),
        __HtoD());

    KernelInfo kernelInfo;
    kernelInfo.wk_dim      = 1;
    kernelInfo.kernel_file = "Permute.cl";

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        if (_need_permute) {
            if (inputs[0]->num() == 1 && inputs[0]->width() == 3 && param == param_nhwc_to_nchw) {
                int out_w = outputs[0]->width() * outputs[0]->height();
                int out_h = outputs[0]->channel();

                kernelInfo.l_wk        = {256};
                kernelInfo.g_wk        = {(out_w + 256 - 1) / 256 * 256};
                kernelInfo.kernel_name = "ker_permute_fwd_transpose";
            } else if (inputs[0]->num() == 1 && param == param_nchw_to_nhwc) {
                int out_h              = inputs[0]->width() * inputs[0]->height();
                int out_w              = inputs[0]->channel();
                kernelInfo.wk_dim      = 2;
                kernelInfo.l_wk        = {16, 16};
                kernelInfo.g_wk        = {(out_h + 16 - 1) / 16 * 16, (out_w + 16 - 1) / 16 * 16};
                kernelInfo.kernel_name = "ker_transpose";
            } else {
                kernelInfo.l_wk        = {256};
                kernelInfo.g_wk        = {(count + 256 - 1) / 256 * 256};
                kernelInfo.kernel_name = "ker_permute_fwd";
            }

            AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                ALOGE("Failed to load program");
                return SaberInvalidValue;
            }

            _kernel_ptr = kptr;

            ALOGD("COMPLETE CREATE KERNEL");
        }
    } else {
        if (_need_permute) {
            if (inputs[0]->num() == 1 && inputs[0]->width() == 3 && param == param_nhwc_to_nchw) {
                int out_w              = outputs[0]->width() * outputs[0]->height();
                int out_h              = outputs[0]->channel();
                kernelInfo.l_wk        = {256};
                kernelInfo.g_wk        = {(out_w + 256 - 1) / 256 * 256};
                kernelInfo.kernel_name = "ker_permute_fwd_transpose";
            } else if (inputs[0]->num() == 1 && param == param_nchw_to_nhwc) {
                kernelInfo.wk_dim = 2;
                kernelInfo.l_wk   = {16, 16};
                kernelInfo.g_wk   = {(inputs[0]->num() * inputs[0]->channel() + 16 - 1) / 16 * 16,
                                     (inputs[0]->height() * inputs[0]->width() + 16 - 1) / 16 * 16
                                    };
                kernelInfo.kernel_name = "ker_nchw_to_nhwc";
            } else {
                kernelInfo.l_wk        = {256};
                kernelInfo.g_wk        = {(count + 256 - 1) / 256 * 256};
                kernelInfo.kernel_name = "ker_permute_fwd";
            }

            AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                ALOGE("Failed to load program");
                return SaberInvalidValue;
            }

            _kernel_ptr = kptr;

            ALOGD("COMPLETE CREATE KERNEL");
        }
    }

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberPermute<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PermuteParam<AMD>& param) {
    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    int count                                   = outputs[0]->valid_size();
    std::vector<int> permute_order_nhwc_to_nchw = {0, 3, 1, 2};
    PermuteParam<AMD> param_nhwc_to_nchw(permute_order_nhwc_to_nchw);
    std::vector<int> permute_order_nchw_to_nhwc = {0, 2, 3, 1};
    PermuteParam<AMD> param_nchw_to_nhwc(permute_order_nchw_to_nhwc);

    bool err = false;

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        if (_need_permute) {

            if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
                ALOGE("Kernel is not exist");
                return SaberInvalidValue;
            }

            AMDKernel* kernel = _kernel_ptr.get();

            if (inputs[0]->num() == 1 && inputs[0]->width() == 3 && param == param_nhwc_to_nchw) {
                int out_w = outputs[0]->width() * outputs[0]->height();
                int out_h = outputs[0]->channel();

                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)out_h,
                          (int)out_w,
                          (PtrDtype)inputs[0]->data());
            } else if (inputs[0]->num() == 1 && param == param_nchw_to_nhwc) {
                int out_h = inputs[0]->width() * inputs[0]->height();
                int out_w = inputs[0]->channel();

                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)out_h,
                          (int)out_w,
                          (PtrDtype)inputs[0]->data());
            } else {
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)_num_axes,
                          (int)count,
                          (PtrDtype)_permute_order.data(),
                          (PtrDtype)_out_steps.data(),
                          (PtrDtype)_in_steps.data(),
                          (PtrDtype)inputs[0]->data());
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
        } else {
            outputs[0]->copy_from(*inputs[0]);
        }
    } else {
        if (_need_permute) {

            if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
                ALOGE("Kernel is not exist");
                return SaberInvalidValue;
            }

            AMDKernel* kernel = _kernel_ptr.get();

            if (inputs[0]->num() == 1 && inputs[0]->width() == 3 && param == param_nhwc_to_nchw) {
                int out_w = outputs[0]->width() * outputs[0]->height();
                int out_h = outputs[0]->channel();

                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)outputs[0]->num(),
                          (int)outputs[0]->channel(),
                          (int)outputs[0]->height(),
                          (int)outputs[0]->width(),
                          (PtrDtype)_out_steps.data(),
                          (PtrDtype)_in_steps.data(),
                          (PtrDtype)inputs[0]->data());
            } else if (inputs[0]->num() == 1 && param == param_nchw_to_nhwc) {
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)inputs[0]->num(),
                          (int)inputs[0]->channel(),
                          (int)inputs[0]->height(),
                          (int)outputs[0]->width(),
                          (int)inputs[0]->width(),
                          (PtrDtype)_out_steps.data(),
                          (PtrDtype)_in_steps.data(),
                          (PtrDtype)inputs[0]->data());
            } else {
                err = kernel->SetKernelArgs(
                          (PtrDtype)outputs[0]->mutable_data(),
                          (int)inputs[0]->num(),
                          (int)inputs[0]->channel(),
                          (int)inputs[0]->height(),
                          (int)outputs[0]->width(),
                          (int)inputs[0]->width(),
                          (PtrDtype)_out_steps.data(),
                          (PtrDtype)_in_steps.data(),
                          (PtrDtype)inputs[0]->data());
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
        } else {
            outputs[0]->copy_from(*inputs[0]);
        }
    }

    ALOGD("COMPLETE EXECUTION");
    return SaberSuccess;
}
template class SaberPermute<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberPermute, PermuteParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberPermute, PermuteParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin
