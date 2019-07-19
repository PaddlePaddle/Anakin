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
#include "include/saber_mvn.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberMvn<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    MvnParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberMvn<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    MvnParam<AMD>& param,
    Context<AMD>& ctx) {
    const int count = outputs[0]->valid_size();
    int num         = inputs[0]->num() * inputs[0]->channel();
    int inner_dim   = inputs[0]->height() * inputs[0]->width();

    if (param.across_channels) {
        num = inputs[0]->num();
        // inner_dim *= inputs[0]->channel();
    }

    Shape shape = inputs[0]->valid_shape();

    for (int i = 0; i < shape.size(); i++) {
        shape[i] = 1;
    }

    shape[0] = num;
    _mean.reshape(shape);

    if (param.normalize_variance) {
        _sd.reshape(shape);
    }

    AMDKernelPtr kptr;
    KernelInfo kernelInfo;
    kernelInfo.wk_dim      = 3;
    kernelInfo.kernel_file = "Mvn.cl";

    if (param.normalize_variance) {

        kernelInfo.kernel_name = "sum_square";
        kernelInfo.l_wk        = {256, 1, 1};
        kernelInfo.g_wk        = {256, (inner_dim + 256 - 1) / 256, num};

        kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to create kernel";
            return SaberInvalidValue;
        }

        _kernels_ptr.push_back(kptr);

        kernelInfo.kernel_name = "normalize_square";
        kernelInfo.l_wk        = {256, 1, 1};
        kernelInfo.g_wk        = {256, (inner_dim + 256 - 1) / 256, num};
        kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to create kernel";
            return SaberInvalidValue;
        }

        _kernels_ptr.push_back(kptr);
    } else {

        kernelInfo.kernel_name = "sum";
        kernelInfo.l_wk        = {256, 1, 1};
        kernelInfo.g_wk        = {256, (inner_dim + 256 - 1) / 256, num};

        kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to create kernel";
            return SaberInvalidValue;
        }

        _kernels_ptr.push_back(kptr);

        kernelInfo.kernel_name = "normalize";
        kernelInfo.l_wk        = {256, 1, 1};
        kernelInfo.g_wk        = {256, (inner_dim + 256 - 1) / 256, num};
        kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to create kernel";
            return SaberInvalidValue;
        }

        _kernels_ptr.push_back(kptr);
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberMvn<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    MvnParam<AMD>& param) {
    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    int num       = inputs[0]->num() * inputs[0]->channel();
    int inner_dim = inputs[0]->height() * inputs[0]->width();

    if (param.across_channels) {
        num = inputs[0]->num();
        inner_dim *= inputs[0]->channel();
    }

    AMD_API::mem_set(_mean.mutable_data(), 0, _mean.valid_size() * sizeof(OpDataType));
    int j    = 0; // kernel index
    bool err = false;

    amd_kernel_list list;

    if (param.normalize_variance) {
        AMD_API::mem_set(_sd.mutable_data(), 0, _mean.valid_size() * sizeof(OpDataType));

        if (_kernels_ptr[j] == NULL || _kernels_ptr[j].get() == NULL) {
            LOG(ERROR) << "Kernel is not exist";
            return SaberInvalidValue;
        }

        err = _kernels_ptr[j].get()->SetKernelArgs(
                  (PtrDtype)inputs[0]->data(),
                  (int)num,
                  (int)inner_dim,
                  (PtrDtype)_mean.mutable_data(),
                  (PtrDtype)_sd.mutable_data());
        list.push_back(_kernels_ptr[j]);
        j++;

        if (_kernels_ptr[j] == NULL || _kernels_ptr[j].get() == NULL) {
            LOG(ERROR) << "Kernel is not exist";
            return SaberInvalidValue;
        }

        err = _kernels_ptr[j].get()->SetKernelArgs(
                  (PtrDtype)inputs[0]->data(),
                  (int)num,
                  (int)inner_dim,
                  (float)(1.0 / inner_dim),
                  (PtrDtype)_mean.mutable_data(),
                  (PtrDtype)_sd.mutable_data(),
                  (float)param.eps,
                  (PtrDtype)outputs[0]->mutable_data());
        list.push_back(_kernels_ptr[j]);

    } else {
        if (_kernels_ptr[j] == NULL || _kernels_ptr[j].get() == NULL) {
            LOG(ERROR) << "Kernel is not exist";
            return SaberInvalidValue;
        }

        err = _kernels_ptr[j].get()->SetKernelArgs(
                  (PtrDtype)inputs[0]->data(),
                  (int)num,
                  (int)inner_dim,
                  (PtrDtype)_mean.mutable_data());
        list.push_back(_kernels_ptr[j]);
        j++;

        if (_kernels_ptr[j] == NULL || _kernels_ptr[j].get() == NULL) {
            LOG(ERROR) << "Kernel is not exist";
            return SaberInvalidValue;
        }

        err = _kernels_ptr[j].get()->SetKernelArgs(
                  (PtrDtype)inputs[0]->data(),
                  (int)num,
                  (int)inner_dim,
                  (float)(1.0 / inner_dim),
                  (PtrDtype)_mean.mutable_data(),
                  (PtrDtype)outputs[0]->mutable_data());
        list.push_back(_kernels_ptr[j]);
    }

    if (!err) {
        LOG(ERROR) << "Failed to set kernel args";
        return SaberInvalidValue;
    }

    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Failed to set execution";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}
template class SaberMvn<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberMvn, MvnParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberMvn, MvnParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin
