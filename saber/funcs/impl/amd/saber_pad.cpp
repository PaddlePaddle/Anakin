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
#include "include/saber_pad.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberPad<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PadParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberPad<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PadParam<AMD>& param,
    Context<AMD>& ctx) {

    CHECK_EQ(2, param.pad_c.size());
    CHECK_EQ(2, param.pad_h.size());
    CHECK_EQ(2, param.pad_w.size());
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
    _img_offset      = _out_c_stride * param.pad_c[0] + _out_h_stride * param.pad_h[0]
                       + _out_w_stride * param.pad_w[0];

    int globalSize = inputs[0]->valid_size();
    KernelInfo kernelInfo;
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {256};
    kernelInfo.g_wk        = {globalSize};
    kernelInfo.kernel_file = "Pad.cl";
    kernelInfo.kernel_name = "Pad";

    AMDKernelPtr kptr = NULL;
    kptr              = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!kptr.get()->isInit()) {
        ALOGE("Failed to load program");
        return SaberInvalidValue;
    }

    _kernel = kptr;

    ALOGD("COMPLETE CREATE KERNEL");

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberPad<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PadParam<AMD>& param) {

    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    if (_kernel == NULL || _kernel.get() == NULL) {
        ALOGE("Kernel is not exist");
        return SaberInvalidValue;
    }

    AMDKernel* kernel = _kernel.get();
    bool err          = false;

    err = kernel->SetKernelArgs(
              (PtrDtype)inputs[0]->data(),
              (PtrDtype)outputs[0]->mutable_data(),
              (int)_in_n_stride,
              (int)_in_c_stride,
              (int)_in_h_stride,
              (int)_in_w_stride,
              (int)_out_n_stride,
              (int)_out_c_stride,
              (int)_out_h_stride,
              (int)_out_w_stride,
              (int)inputs[0]->num(),
              (int)inputs[0]->channel(),
              (int)inputs[0]->height(),
              (int)inputs[0]->width(),
              (int)_img_offset);

    if (!err) {
        ALOGE("Failed to set kernel args");
        return SaberInvalidValue;
    }

    amd_kernel_list list;
    list.push_back(_kernel);
    err = LaunchKernel(cm, list);

    if (!err) {
        ALOGE("Fail to set execution");
        return SaberInvalidValue;
    }

    ALOGD("COMPLETE EXECUTION");
    return SaberSuccess;
}

template class SaberPad<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberPad, PadParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin
