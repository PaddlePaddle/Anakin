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
#include "include/saber_direct_deconv.h"
namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;
typedef Tensor<AMD> TensorDf4;

template <>
SaberStatus SaberDirectDeconv<AMD, AK_FLOAT>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param,
    Context<AMD>& ctx) {

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "create";

    KernelInfo kernelInfo;

    int isBias           = (param.bias()->size() > 0) ? 1 : 0;
    int relu_flag        = (param.activation_param.active == Active_relu) ? 1 : 0;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    kernelInfo.wk_dim = 3;
    kernelInfo.l_wk        = {16, 16, 1};
    kernelInfo.g_wk        = {(outputs[0]->width() + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]* kernelInfo.l_wk[0], (outputs[0]->height() + kernelInfo.l_wk[1] - 1) / kernelInfo.l_wk[1]* kernelInfo.l_wk[1], inputs[0]->num()* outputs[0]->channel()};
    kernelInfo.kernel_file = "Deconv.cl";
    kernelInfo.kernel_name = "direct_deconv";
    _kernel_ptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}

template <>
SaberStatus SaberDirectDeconv<AMD, AK_FLOAT>::init(
    const std::vector<Tensor<AMD> *>& inputs,
    std::vector<Tensor<AMD> *>& outputs,
    ConvParam<AMD>& param, Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return SaberSuccess;
}

template <>
SaberStatus SaberDirectDeconv<AMD, AK_FLOAT>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param) {
    bool err = false;
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "dispatch";

    if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
        LOG(ERROR) << "Kernel is not exist";
        return SaberInvalidValue;
    }

    AMDKernel* kernel = _kernel_ptr.get();

    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int num = inputs[0]->num();
    int ch_in = inputs[0]->channel();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int ch_out = outputs[0]->channel();

    int kernel_h = param.weight()->height();
    int kernel_w = param.weight()->width();

    int channel_in_stride = hin * win;
    int channel_out_stride = hout * wout;
    int kernel_size = kernel_h * kernel_w;

    bool bias_flag = param.bias()->valid_size() > 0;
    bool relu_flag = param.activation_param.has_active;
    LOG(ERROR) << "num:" << num << " " << ch_in << " out:" << ch_out << " " << hout << " " << wout <<
               " " << channel_out_stride
               << " in:" << hin << " " << win << " " << channel_in_stride
               << " kernel:" << kernel_h << " " << kernel_w  << " " << kernel_size
               << " stride:" << param.stride_h << " " << param.stride_w
               << " pad:" << param.pad_h << " " << param.pad_w
               << " dilation:" << param.dilation_h << " " << param.dilation_w;

    err = kernel->SetKernelArgs((PtrDtype)inputs[0]->data(), (PtrDtype)param.bias()->data(),
                                (PtrDtype)param.weight()->data(),
                                num, ch_in, ch_out, hout, wout, channel_out_stride,
                                hin, win, channel_in_stride,
                                kernel_h, kernel_w, kernel_size,
                                param.stride_h, param.stride_w,
                                param.pad_h, param.pad_w,
                                param.dilation_h, param.dilation_w,
                                (PtrDtype)outputs[0]->mutable_data(), (int)bias_flag, (int)relu_flag);

    amd_kernel_list list;
    list.push_back(_kernel_ptr);
    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}

template class SaberDirectDeconv<AMD, AK_FLOAT>;
} // namespace saber
} // namespace anakin
