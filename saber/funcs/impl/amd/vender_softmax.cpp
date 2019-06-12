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
/*
   MIT License

   Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

#include <limits>
#include "include/vender_softmax.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus VenderSoftmax<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    SoftmaxParam<AMD>& param,
    Context<AMD>& ctx) {

    SaberStatus status;

    this->_ctx = &ctx;
    status = create(inputs, outputs, param, ctx);

    return status;
}

static int nextPow2(int v) {

    if (v == 1) {
        return (v << 1);
    } else {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }
}

template <DataType OpDtype>
VenderSoftmax<AMD, OpDtype>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    _kernels.push_back(kptr);
}

template <DataType OpDtype>
SaberStatus VenderSoftmax<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    SoftmaxParam<AMD>& param,
    Context<AMD>& ctx) {

    _kernels.clear();
    KernelInfo kernelInfo;

    int count = inputs[0]->valid_shape()[param.axis];
    int grid_size = inputs[0]->count_valid(0, param.axis) * inputs[0]->count_valid(param.axis + 1,
                    inputs[0]->dims());

    //The below section of code are as MIT license, the permission notice is from above (line 106 to 145)
    // To set local work size
    int local_work_size;
    kernelInfo.kernel_file = "MIOpenSoftmax.cl";

    if (count < 16128) {
        local_work_size = 256;
        kernelInfo.kernel_name = "SoftmaxForward_Fast";
    } else {
        local_work_size = 512;
        kernelInfo.kernel_name = "SoftmaxForward";
    }

    kernelInfo.wk_dim = 3;
    kernelInfo.l_wk   = {local_work_size, 1, 1};

    int num_batch = count < local_work_size ? nextPow2(local_work_size / count) : 1;

    // To set comp_options
    kernelInfo.comp_options = std::string(" -DMIOPEN_USE_FP32=1")
                              + std::string(" -DMIOPEN_USE_FP16=0")
                              + std::string(" -DNUM_BATCH=") + std::to_string(num_batch);

    if (num_batch == 1) { // CSR-Vector like approach

        // Control the max. number of workgroups launched so that we do not
        // start getting workgroup scheduling overheads
        size_t workgroups = std::min(grid_size, 64 * 40 * 8);
        kernelInfo.g_wk   = {workgroups* kernelInfo.l_wk[0], 1, 1};
    } else { // CSR-Stream like approach

        // num_threads iterating over channels for one spatial_dim
        int batch_size = local_work_size / num_batch;
        // num_channels each threads iterates over to cover all the channels
        int u_batch_size = count > batch_size ? nextPow2(count / batch_size) : 1;

        size_t workgroups =
            grid_size % num_batch == 0 ? grid_size / num_batch : grid_size / num_batch + 1;
        kernelInfo.g_wk = {workgroups* kernelInfo.l_wk[0], 1, 1};

        kernelInfo.comp_options += " -DBATCH_SIZE=" + std::to_string(batch_size)
                                   + " -DU_BATCH_SIZE=" + std::to_string(u_batch_size);
    }

    kernelInfo.kernel_type = MIOPEN;
    CreateKernelList(inputs[0]->device_id(), kernelInfo);

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus VenderSoftmax<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    SoftmaxParam<AMD>& param) {

    AMD_API::stream_t cm         = this->_ctx->get_compute_stream();
    bool err                     = false;
    amd_kernel_list::iterator it = _kernels.begin();

    int count = inputs[0]->valid_shape()[param.axis];
    int grid_size = inputs[0]->count_valid(0, param.axis) * inputs[0]->count_valid(param.axis + 1,
                    inputs[0]->dims());
    int spatial_dim = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());

    // To set the argument
    err = it->get()->SetKernelArgs(
              (PtrDtype)inputs[0]->data(),
              (PtrDtype)outputs[0]->mutable_data(),
              count,
              grid_size,
              spatial_dim);

    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    err = LaunchKernel(cm, _kernels);

    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(VenderSoftmax, SoftmaxParam, AMD, AK_HALF);
DEFINE_OP_TEMPLATE(VenderSoftmax, SoftmaxParam, AMD, AK_INT8);
} // namespace saber

} // namespace anakin
