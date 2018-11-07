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

#include "include/saber_lrn.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberLrn<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    LrnParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberLrn<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    LrnParam<AMD>& param,
    Context<AMD>& ctx) {

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

    int globalSize = outputs[0]->valid_size() / outputs[0]->channel();
    int localSize  = 256;
    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "Lrn.cl";
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {256};
    kernelInfo.g_wk        = {(globalSize + localSize - 1) / localSize * localSize};
    kernelInfo.kernel_name = "Lrn";

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
SaberStatus SaberLrn<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    LrnParam<AMD>& param) {

    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
        ALOGE("Kernel is not exist");
        return SaberInvalidValue;
    }

    AMDKernel* kernel = _kernel_ptr.get();
    bool err          = false;

    err = kernel->SetKernelArgs(
              (PtrDtype)inputs[0]->data(),
              (PtrDtype)outputs[0]->mutable_data(),
              (int)_in_n_stride,
              (int)_in_c_stride,
              (int)_in_h_stride,
              (int)_in_w_stride,
              (int)outputs[0]->num(),
              (int)outputs[0]->channel(),
              (int)outputs[0]->height(),
              (int)outputs[0]->width(),
              (float)param.alpha,
              (float)param.beta,
              (float)param.k,
              (int)param.local_size);

    if (!err) {
        ALOGE("Failed to set kernel args.");
        return SaberInvalidValue;
    }

    amd_kernel_list list;
    list.push_back(_kernel_ptr);
    err = LaunchKernel(cm, list);

    if (!err) {
        ALOGE("Failed to set execution");
        return SaberInvalidValue;
    }

    ALOGE("COMPLETE EXECUTION");
    return SaberSuccess;
}
template class SaberLrn<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberLrn, LrnParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberLrn, LrnParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin