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

    // saber impl help to handle vender un-impl case.
    if (!(inputs[0]->height() == 1 && inputs[0]->width() == 1 && param.axis == 1)) {
        if (this->_saber_impl == nullptr)
        {
            this->_saber_impl = new SaberSoftmax<AMD, OpDtype>{};
        }
        this->_saber_impl->init(inputs, outputs, param, ctx);
    }

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

    if (inputs[0]->height() == 1 && inputs[0]->width() == 1 && param.axis == 1) {
        // To set local work size
        kernelInfo.wk_dim = 3;
        kernelInfo.l_wk   = {256, 1, 1};

        int c         = inputs[0]->channel();
        int num_batch = c < 256 ? nextPow2(256 / c) : 1;
        int grid_size = inputs[0]->num() * inputs[0]->width() * inputs[0]->height();

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
            int batch_size = 256 / num_batch;
            // num_channels each threads iterates over to cover all the channels
            int u_batch_size = c > batch_size ? nextPow2(c / batch_size) : 1;

            size_t workgroups =
                grid_size % num_batch == 0 ? grid_size / num_batch : grid_size / num_batch + 1;
            kernelInfo.g_wk = {workgroups* kernelInfo.l_wk[0], 1, 1};

            kernelInfo.comp_options += " -DBATCH_SIZE=" + std::to_string(batch_size)
                                       + " -DU_BATCH_SIZE=" + std::to_string(u_batch_size);
        }

        kernelInfo.kernel_file = "MIOpenSoftmax.cl";
        kernelInfo.kernel_name = "SoftmaxForward";
        kernelInfo.kernel_type = MIOPEN;
        CreateKernelList(inputs[0]->device_id(), kernelInfo);
    }

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

    if (inputs[0]->height() == 1 && inputs[0]->width() == 1 && param.axis == 1) {
        // To set the argument
        outputs[0]->copy_from(*inputs[0]);
        err = it->get()->SetKernelArgs(
                  (PtrDtype)outputs[0]->mutable_data(),
                  (int)inputs[0]->channel(),
                  (int)inputs[0]->num() * inputs[0]->width() * inputs[0]->height(),
                  (int)inputs[0]->width() * inputs[0]->height());

        if (!err) {
            LOG(ERROR) << "Fail to set execution";
            return SaberInvalidValue;
        }
    } else {
        return this->_saber_impl->dispatch(inputs, outputs, param);
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
