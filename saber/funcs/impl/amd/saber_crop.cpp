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
#include "include/saber_crop.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberCrop<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    CropParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberCrop<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    CropParam<AMD>& param,
    Context<AMD>& ctx) {

    // compute the kernel input param.
    Shape in_stride = inputs[0]->get_stride();
    int in_n_index  = inputs[0]->num_index();
    int in_c_index  = inputs[0]->channel_index();
    int in_h_index  = inputs[0]->height_index();
    int in_w_index  = inputs[0]->width_index();
    _img_offset     = 0;

    if (param.axis == 1) {
        CHECK_EQ(param.offset.size(), 3);
        _img_offset += param.offset[0] * in_stride[in_c_index];
        _img_offset += param.offset[1] * in_stride[in_h_index];
        _img_offset += param.offset[2] * in_stride[in_w_index];
    } else if (param.axis == 2) {
        CHECK_EQ(param.offset.size(), 2);
        _img_offset += param.offset[0] * in_stride[in_h_index];
        _img_offset += param.offset[1] * in_stride[in_w_index];
    } else if (param.axis == 3) {
        CHECK_EQ(param.offset.size(), 1);
        _img_offset += param.offset[0] * in_stride[in_w_index];
    } else {
        return SaberInvalidValue;
    }

    Shape out_stride = outputs[0]->get_stride();
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

    int global_size =
        inputs[0]->num() * inputs[0]->channel() * inputs[0]->width() * inputs[0]->height();
    int local_size = 256;

    KernelInfo kernelInfo;
    kernelInfo.wk_dim      = 1;
    kernelInfo.l_wk        = {local_size};
    kernelInfo.g_wk        = {(global_size + local_size - 1) / local_size * local_size};
    kernelInfo.kernel_file = "Crop.cl";
    kernelInfo.kernel_name = "Crop";

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
SaberStatus SaberCrop<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    CropParam<AMD>& param) {
    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
        LOG(ERROR) << "Kernel is not exist";
        return SaberInvalidValue;
    }

    AMDKernel* kernel = _kernel_ptr.get();

    bool err = false;
    err      = kernel->SetKernelArgs(
                   (PtrDtype)outputs[0]->mutable_data(),
                   (PtrDtype)inputs[0]->data(),
                   (int)_in_n_stride,
                   (int)_in_c_stride,
                   (int)_in_h_stride,
                   (int)_in_w_stride,
                   (int)_out_n_stride,
                   (int)_out_c_stride,
                   (int)_out_h_stride,
                   (int)_out_w_stride,
                   (int)outputs[0]->num(),
                   (int)outputs[0]->channel(),
                   (int)outputs[0]->height(),
                   (int)outputs[0]->width(),
                   (int)_img_offset);

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
template class SaberCrop<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberCrop, CropParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberCrop, CropParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin
