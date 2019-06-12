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

#include "saber/funcs/impl/amd/include/saber_pooling.h"

namespace anakin {
namespace saber {

static bool tryPoolingGeneral(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PoolingParam<AMD>& param,
    KernelInfo& kernelInfo) {
    if (param.window_h * param.window_w < 32) {
        return false;
    }

    if ((param.window_h > param.stride_h || param.window_w > param.stride_w)
            && outputs[0]->count(2, 4) != 1) {
        return false;
    }

    int group_size   = 256;
    int group_size_0 = 256;  // adder

    while (group_size_0 * 8 > param.window_h * param.window_w && group_size_0 > 1) {
        group_size_0 = group_size_0 >> 1;
    }

    int group_size_1 = group_size / group_size_0;

    int global_size_0 = group_size_0;
    int global_size_1 = (outputs[0]->size() + group_size_1 - 1) / group_size_1 * group_size_1;

    kernelInfo.wk_dim      = 3;
    kernelInfo.l_wk        = {group_size_0, group_size_1, 1};
    kernelInfo.g_wk        = {global_size_0, global_size_1, 1};
    kernelInfo.kernel_file = "PoolingGeneral.cl";
    kernelInfo.kernel_name = "PoolingGeneral";

    kernelInfo.comp_options = std::string(" -DGROUP_SIZE=") + std::to_string(group_size)
                              + std::string(" -DGROUP_SIZE_0=") + std::to_string(group_size_0)
                              + std::string(" -DGROUP_SIZE_1=") + std::to_string(group_size_1)
                              + std::string(" -DPOOLING_TYPE=") + std::to_string(param.pooling_type)
                              + std::string(" -DADDER=") + std::to_string(group_size_0);

    return true;
}

static bool tryPoolingWithShare(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PoolingParam<AMD>& param,
    KernelInfo& kernelInfo) {
    if (outputs[0]->size() < 1024 * 32 * 32) {
        return false;
    }

    if (param.window_h <= param.stride_h || param.window_w <= param.stride_w) {
        return false;
    }

    int group_size_1 = 256;
    int group_size_0 = 256;

    while (group_size_1 * 2 > outputs[0]->height() && group_size_1 > 1) {
        group_size_1 = group_size_1 >> 1;
    }

    while (group_size_0 * 2 > outputs[0]->width() && group_size_0 > 1) {
        group_size_0 = group_size_0 >> 1;
    }

    while (group_size_0 * group_size_1 > 256) {
        if (group_size_0 > group_size_1) {
            group_size_0 = group_size_0 >> 1;
        } else {
            group_size_1 = group_size_1 >> 1;
        }
    }

    int y_tile = (outputs[0]->height() + group_size_1 - 1) / group_size_1;
    int x_tile = (outputs[0]->width() + group_size_0 - 1) / group_size_0;

    int y_cache_size = (group_size_1 - 1) * param.stride_h + param.window_h;
    int x_cache_size = (group_size_0 - 1) * param.stride_w + param.window_w;

    if (group_size_0 * group_size_1 < 64) {
        return false;
    }

    if (y_cache_size * x_cache_size > 2000) {
        return false;
    }

    kernelInfo.wk_dim = 3;
    kernelInfo.l_wk        = {group_size_0, group_size_1, 1};
    kernelInfo.g_wk        = {x_tile * group_size_0, y_tile * group_size_1,  outputs[0]->num()* outputs[0]->channel()};
    kernelInfo.kernel_file = "PoolingWithShare.cl";
    kernelInfo.kernel_name = "PoolingWithShare";

    kernelInfo.comp_options = std::string(" -DGROUP_SIZE_0=") + std::to_string(group_size_0)
                              + std::string(" -DGROUP_SIZE_1=") + std::to_string(group_size_1)
                              + std::string(" -DPOOLING_TYPE=") + std::to_string(param.pooling_type)
                              + std::string(" -DCACHE_SIZE_0=") + std::to_string(x_cache_size)
                              + std::string(" -DCACHE_SIZE_1=") + std::to_string(y_cache_size);

    return true;
}
//The below section of code are as MIT license, the permission notice is from above (line 146 to 192)
static bool tryPoolingGen(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PoolingParam<AMD>& param,
    KernelInfo& kernelInfo) {
    kernelInfo.wk_dim = 3;
    kernelInfo.l_wk        = {256, 1, 1};
    kernelInfo.g_wk        = {64 * 64 * 40, 1, 1};
    kernelInfo.kernel_file = "PoolingGen.cl";
    kernelInfo.kernel_name = "mloPooling";

    int bot_batch_stride   = inputs[0]->width() * inputs[0]->height() * inputs[0]->channel();
    int bot_channel_stride = inputs[0]->width() * inputs[0]->height();

    int top_batch_stride   = outputs[0]->width() * outputs[0]->height() * outputs[0]->channel();
    int top_channel_stride = outputs[0]->width() * outputs[0]->height();

    // set comp_options...
    kernelInfo.comp_options =
        std::string(" -DMLO_POOLING_OP_ID=") + std::to_string(param.pooling_type)
        + std::string(" -DMLO_POOLING_KERNEL_SZ0=") + std::to_string(param.window_w)
        + std::string(" -DMLO_POOLING_KERNEL_SZ1=") + std::to_string(param.window_h)
        + std::string(" -DMLO_POOLING_PAD0=") + std::to_string(param.pad_w)
        + std::string(" -DMLO_POOLING_PAD1=") + std::to_string(param.pad_h)
        + std::string(" -DMLO_POOLING_STRIDE0=") + std::to_string(param.stride_w)
        + std::string(" -DMLO_POOLING_STRIDE1=") + std::to_string(param.stride_h)
        + std::string(" -DMLO_POOLING_N_OUTPUTS=") + std::to_string(outputs[0]->channel())
        + std::string(" -DMLO_POOLING_N_CHANNELS=") + std::to_string(inputs[0]->channel())
        + std::string(" -DMLO_POOLING_GROUP_SZ0=8")
        + std::string(" -DMLO_POOLING_GROUP_SZ1=8")
        + std::string(" -DMLO_POOLING_BOT_BATCH_STRIDE=") + std::to_string(bot_batch_stride)
        + std::string(" -DMLO_POOLING_BOT_CHANNEL_STRIDE=") + std::to_string(bot_channel_stride)
        + std::string(" -DMLO_POOLING_BOT_STRIDE=") + std::to_string(inputs[0]->width())
        + std::string(" -DMLO_POOLING_TOP_BATCH_STRIDE=") + std::to_string(top_batch_stride)
        + std::string(" -DMLO_POOLING_TOP_CHANNEL_STRIDE=") + std::to_string(top_channel_stride)
        + std::string(" -DMLO_POOLING_TOP_STRIDE=") + std::to_string(outputs[0]->width())
        + std::string(" -DMLO_POOLING_BOT_WIDTH=") + std::to_string(inputs[0]->width())
        + std::string(" -DMLO_POOLING_BOT_HEIGHT=") + std::to_string(inputs[0]->height())
        + std::string(" -DMLO_POOLING_TOP_WIDTH=") + std::to_string(outputs[0]->width())
        + std::string(" -DMLO_POOLING_TOP_HEIGHT=") + std::to_string(outputs[0]->height())
        + std::string(" -DBATCH_NUM=") + std::to_string(inputs[0]->num())
        + std::string(" -DCU_NUM=64")
        + std::string(" -DMIOPEN_USE_FP32=1")
        + std::string(" -DMIOPEN_USE_FP16=0");

    return true;
}

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberPooling<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PoolingParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberPooling<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PoolingParam<AMD>& param,
    Context<AMD>& ctx) {
    KernelInfo kernelInfo;

#ifdef ENABLE_DEBUG
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "param.pooling_type=" << param.pooling_type
                                         << " param.window_h=" << param.window_h
                                         << " param.window_w=" << param.window_w
                                         << " param.pad_h=" << param.pad_h
                                         << " param.pad_w=" << param.pad_w
                                         << " param.stride_h=" << param.stride_h
                                         << " param.stride_w=" << param.stride_w
                                         << " param.global_pooling=" << param.global_pooling;
#endif

    if (tryPoolingGeneral(inputs, outputs, param, kernelInfo)) {
        ;
    } else if (tryPoolingWithShare(inputs, outputs, param, kernelInfo)) {
        ;
    } else if (tryPoolingGen(inputs, outputs, param, kernelInfo)) {
        ;
    } else {
        LOG(ERROR) << "no pooling function is selected.";
    }

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
SaberStatus SaberPooling<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PoolingParam<AMD>& param) {
#ifdef ENABLE_DEBUG
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "dispatch";
#endif

    if (_kernel_ptr == NULL || _kernel_ptr.get() == NULL) {
        LOG(ERROR) << "Kernel is not exist";
        return SaberInvalidValue;
    }

    AMDKernel* kernel = _kernel_ptr.get();

    bool err = false;

    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    if (kernel->GetName() == "PoolingGeneral" || kernel->GetName() == "PoolingWithShare") {
        err = kernel->SetKernelArgs((PtrDtype)inputs[0]->data(),
                                    (PtrDtype)outputs[0]->mutable_data(),
                                    (int)inputs[0]->num(),
                                    (int)inputs[0]->channel(),
                                    (int)inputs[0]->height(),
                                    (int)inputs[0]->width(),
                                    (int)outputs[0]->height(),
                                    (int)outputs[0]->width(),
                                    (int)param.window_h,
                                    (int)param.window_w,
                                    (int)param.stride_h,
                                    (int)param.stride_w,
                                    (int)param.pad_h,
                                    (int)param.pad_w);
    } else if (kernel->GetName() == "mloPoolingG" || kernel->GetName() == "mloPooling") {
        err = kernel->SetKernelArgs((PtrDtype)inputs[0]->data(), (PtrDtype)outputs[0]->mutable_data(),
                                    (PtrDtype)0);
    } else {
        LOG(ERROR) << " ***** ERROR ***** : kernel name is not exist " << kernel->GetName();
    }

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
template class SaberPooling<AMD, AK_FLOAT>;
} // namespace saber
} // namespace anakin
