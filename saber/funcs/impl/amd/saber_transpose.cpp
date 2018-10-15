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
#include "include/saber_transpose.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberTranspose<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    TransposeParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberTranspose<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    TransposeParam<AMD>& param,
    Context<AMD>& ctx) {

    int global_size =
        inputs[0]->num() * inputs[0]->channel() * inputs[0]->width() * inputs[0]->height();
    int local_size = 256;

    KernelInfo kernelInfo;
    kernelInfo.wk_dim      = 2;
    kernelInfo.l_wk        = {16, 16};
    kernelInfo.g_wk        = {inputs[0]->width() + 16 - 1, inputs[0]->height() + 16 - 1};
    kernelInfo.kernel_file = "Transpose.cl";
    kernelInfo.kernel_name = "transpose_2d";

    AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    _kernel_ptr = kptr;

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberTranspose<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    TransposeParam<AMD>& param) {
    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
        LOG(ERROR) << "Kernel is not exist";
        return SaberInvalidValue;
    }

    AMDKernel* kernel = _kernel_ptr.get();

    int w_out = outputs[0]->width();
    int h_out = outputs[0]->height();
    int c_out = outputs[0]->channel();
    int n_out = outputs[0]->num();

    int w_in = inputs[0]->width();
    int h_in = inputs[0]->height();
    int c_in = inputs[0]->channel();
    int n_in = inputs[0]->num();

    int num_idx     = inputs[0]->num_index();
    int channel_idx = inputs[0]->channel_index();
    int height_idx  = inputs[0]->height_index();
    int width_idx   = inputs[0]->width_index();

    int dims = inputs[0]->dims();

    CHECK_EQ(c_in, c_out) << "input channel should = output channel";
    CHECK_EQ(n_in, n_out) << "input batch size should = output batch size";
    CHECK_EQ(h_in, w_out) << "input width size should = output height size";
    CHECK_EQ(w_in, h_out) << "input height size should = output width size";

    bool err = false;
    err      = kernel->SetKernelArgs(
                   (PtrDtype)outputs[0]->mutable_data(),
                   (PtrDtype)inputs[0]->data(),
                   (int)n_in,
                   (int)c_in,
                   (int)h_in,
                   (int)w_in);

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

    return SaberSuccess;
}
template class SaberTranspose<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberTranspose, TransposeParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberTranspose, TransposeParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin
