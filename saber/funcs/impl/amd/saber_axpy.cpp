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
#include <limits>
#include "include/saber_axpy.h"
namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberAxpy<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    AxpyParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberAxpy<AMD, OpDtype>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    _kernels.push_back(kptr);
}

template <DataType OpDtype>
SaberStatus SaberAxpy<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    AxpyParam<AMD>& param,
    Context<AMD>& ctx) {
    _kernels.clear();
    const int count     = outputs[0]->size();
    int img_size                 = outputs[0]->height() * outputs[0]->width();
    cl_context context  = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; // anakin device id to AMD device
    device          = dev.get_device();
    context         = dev.get_context();

    if (img_size > 3 && img_size % 4 == 0) {
        KernelInfo kernelInfo;
        kernelInfo.kernel_file = "Axpy.cl";
        kernelInfo.wk_dim      = 1;
        kernelInfo.kernel_type = SABER;
        kernelInfo.l_wk        = {256};
        kernelInfo.g_wk = {(count / 4   + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]* kernelInfo.l_wk[0]};

        kernelInfo.kernel_name = "Axpy_float4";
        CreateKernelList(inputs[0]->device_id(), kernelInfo);
        return SaberSuccess;

    } else  if (img_size > 1 && img_size % 2 == 0) {
        KernelInfo kernelInfo;
        kernelInfo.kernel_file = "Axpy.cl";
        kernelInfo.wk_dim      = 1;
        kernelInfo.kernel_type = SABER;
        kernelInfo.l_wk        = {256};
        kernelInfo.g_wk = {(count / 2   + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]* kernelInfo.l_wk[0]};

        kernelInfo.kernel_name = "Axpy_float2";
        CreateKernelList(inputs[0]->device_id(), kernelInfo);
        return SaberSuccess;

    } else {

        KernelInfo kernelInfo;
        kernelInfo.kernel_file = "Axpy.cl";
        kernelInfo.wk_dim      = 1;
        kernelInfo.kernel_type = SABER;
        kernelInfo.l_wk        = {256};
        kernelInfo.g_wk = {(count   + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]* kernelInfo.l_wk[0]};

        kernelInfo.kernel_name = "Axpy";
        CreateKernelList(inputs[0]->device_id(), kernelInfo);

        return SaberSuccess;
    }
}
template <DataType OpDtype>
SaberStatus SaberAxpy<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    AxpyParam<AMD>& param) {
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    bool err                     = false;
    const int count              = outputs[0]->size();
    int img_size                 = outputs[0]->height() * outputs[0]->width();
    amd_kernel_list::iterator it = _kernels.begin();

    OpDataType* data_out      = (OpDataType*)outputs[0]->mutable_data(); // out_data
    OpDataType* in_data_scale = (OpDataType*)inputs[0]->data();          // in_data_scale
    OpDataType* in_data_x     = (OpDataType*)inputs[1]->data();          // in_data_x
    OpDataType* in_data_y     = (OpDataType*)inputs[2]->data();          // in_data_y

    if (img_size > 3 && img_size % 4 == 0) {
        err = it->get()->SetKernelArgs(
                  (int)(count / 4),
                  (int)img_size,
                  (PtrDtype)in_data_scale,
                  (PtrDtype)in_data_x,
                  (PtrDtype)in_data_y,
                  (PtrDtype)data_out);

        if (!err) {
            LOG(ERROR) << "Fail to set execution";
            return SaberInvalidValue;
        }
    } else  if (img_size > 1 && img_size % 2 == 0) {
        err = it->get()->SetKernelArgs(
                  (int)(count / 2),
                  (int)img_size,
                  (PtrDtype)in_data_scale,
                  (PtrDtype)in_data_x,
                  (PtrDtype)in_data_y,
                  (PtrDtype)data_out);

        if (!err) {
            LOG(ERROR) << "Fail to set execution";
            return SaberInvalidValue;
        }
    } else {

        err = it->get()->SetKernelArgs(
                  (int)(count),
                  (int)img_size,
                  (PtrDtype)in_data_scale,
                  (PtrDtype)in_data_x,
                  (PtrDtype)in_data_y,
                  (PtrDtype)data_out);

        if (!err) {
            LOG(ERROR) << "Fail to set execution";
            return SaberInvalidValue;
        }
    }

    err = LaunchKernel(cm, _kernels);

    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    return SaberSuccess;
}

template class SaberAxpy<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberAxpy, AxpyParam, AMD, AK_HALF);
DEFINE_OP_TEMPLATE(SaberAxpy, AxpyParam, AMD, AK_INT8);
} // namespace saber
} // namespace anakin
