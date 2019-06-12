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
#include "include/saber_activation.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberActivation<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ActivationParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberActivation<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ActivationParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;

    KernelInfo kernelInfo;
    int data_width = 4;
    int global_size =
        inputs[0]->num() * inputs[0]->channel() * inputs[0]->width() * inputs[0]->height();
    global_size = (global_size + data_width - 1) / data_width;
    int local_size = 512;
    kernelInfo.kernel_file = "Activation.cl";
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {local_size};
    kernelInfo.g_wk        = {(global_size + local_size - 1) / local_size * local_size};
    kernelInfo.comp_options = std::string("-DWIDTH=") + std::to_string(data_width);

    switch (param.active) {
    case Active_relu:
        kernelInfo.kernel_name = "Relu";
        break;

    case Active_sigmoid:
        kernelInfo.kernel_name = "Sigmoid";
        break;

    case Active_tanh:
        kernelInfo.kernel_name = "Tanh";
        break;

    case Active_stanh:
        kernelInfo.kernel_name = "Stanh";
        break;

    case Active_clipped_relu:
        kernelInfo.kernel_name = "Clipped_Relu";
        break;

    case Active_elu:
        kernelInfo.kernel_name = "Elu";
        break;

    case Active_prelu:
        kernelInfo.kernel_name = "Prelu";
        break;
    }

    // To create the program
    AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    _kernel_ptr = kptr;

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberActivation<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ActivationParam<AMD>& param) {
    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    // To set the argument
    int in_n = inputs[0]->num();
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();

    int in_n_stride = inputs[0]->get_stride()[0];
    int in_c_stride = inputs[0]->get_stride()[1];
    int in_h_stride = inputs[0]->get_stride()[2];
    int in_w_stride = inputs[0]->get_stride()[3];

    int out_n_stride = outputs[0]->get_stride()[0];
    int out_c_stride = outputs[0]->get_stride()[1];
    int out_h_stride = outputs[0]->get_stride()[2];
    int out_w_stride = outputs[0]->get_stride()[3];

    if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
        LOG(ERROR) << "Kernel is not exist";
        return SaberInvalidValue;
    }

    AMDKernel* kernel = _kernel_ptr.get();

    bool err = false;

    switch (param.active) {
    case Active_relu:
        /* command_queue, num_of_wait_events, wait_events, event */
        err = kernel->SetKernelArgs(
                  (PtrDtype)inputs[0]->data(),
                  (PtrDtype)outputs[0]->mutable_data(),
                  (int)in_n,
                  (int)in_c,
                  (int)in_h,
                  (int)in_w,
                  (int)in_n_stride,
                  (int)in_c_stride,
                  (int)in_h_stride,
                  (int)in_w_stride,
                  (int)out_n_stride,
                  (int)out_c_stride,
                  (int)out_h_stride,
                  (int)out_w_stride,
                  0.0f);
        break;

    case Active_sigmoid:
    case Active_tanh:
        err = kernel->SetKernelArgs(
                  (PtrDtype)inputs[0]->data(),
                  (PtrDtype)outputs[0]->mutable_data(),
                  (int)in_n,
                  (int)in_c,
                  (int)in_h,
                  (int)in_w,
                  (int)in_n_stride,
                  (int)in_c_stride,
                  (int)in_h_stride,
                  (int)in_w_stride,
                  (int)out_n_stride,
                  (int)out_c_stride,
                  (int)out_h_stride,
                  (int)out_w_stride);
        break;

    case Active_stanh:
        err = kernel->SetKernelArgs(
                  (PtrDtype)inputs[0]->data(),
                  (PtrDtype)outputs[0]->mutable_data(),
                  (int)in_n,
                  (int)in_c,
                  (int)in_h,
                  (int)in_w,
                  (int)in_n_stride,
                  (int)in_c_stride,
                  (int)in_h_stride,
                  (int)in_w_stride,
                  (int)out_n_stride,
                  (int)out_c_stride,
                  (int)out_h_stride,
                  (int)out_w_stride,
                  (float)param.negative_slope,
                  (float)param.coef);
        break;

    case Active_clipped_relu:
    case Active_elu:
        err = kernel->SetKernelArgs(
                  (PtrDtype)inputs[0]->data(),
                  (PtrDtype)outputs[0]->mutable_data(),
                  (int)in_n,
                  (int)in_c,
                  (int)in_h,
                  (int)in_w,
                  (int)in_n_stride,
                  (int)in_c_stride,
                  (int)in_h_stride,
                  (int)in_w_stride,
                  (int)out_n_stride,
                  (int)out_c_stride,
                  (int)out_h_stride,
                  (int)out_w_stride,
                  (float)param.coef);
        break;

    case Active_prelu:
        err = kernel->SetKernelArgs(
                  (PtrDtype)inputs[0]->data(),
                  (PtrDtype)outputs[0]->mutable_data(),
                  (int)in_n,
                  (int)in_c,
                  (int)in_h,
                  (int)in_w,
                  (int)in_n_stride,
                  (int)in_c_stride,
                  (int)in_h_stride,
                  (int)in_w_stride,
                  (int)out_n_stride,
                  (int)out_c_stride,
                  (int)out_h_stride,
                  (int)out_w_stride,
                  (PtrDtype)param.prelu_param.slope->data(),
                  (int)param.prelu_param.channel_shared);
        break;
    }

    amd_kernel_list list;
    list.push_back(_kernel_ptr);
    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    return SaberSuccess;
}

template class SaberActivation<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberActivation, ActivationParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberActivation, ActivationParam, AMD, AK_HALF);
} // namespace saber
} // namespace anakin
