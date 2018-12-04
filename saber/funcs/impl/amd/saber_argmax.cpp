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

#include "include/saber_argmax.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberArgmax<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ArgmaxParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberArgmax<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ArgmaxParam<AMD>& param,
    Context<AMD>& ctx) {

    int localSize  = 256;
    int globalSize = 0;

    if (!param.has_axis) {
        int inner_dim = inputs[0]->count(1, inputs[0]->dims());
        int outer_dim = inputs[0]->num();
        int group_num = (inner_dim + localSize - 1) / localSize;
        _group_max_value.re_alloc(Shape({outer_dim, group_num, 1, 1}, Layout_NCHW));
        _group_max_index.re_alloc(Shape({outer_dim, group_num, 1, 1}, Layout_NCHW));
    }

    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "Argmax.cl";
    kernelInfo.l_wk        = {localSize};
    kernelInfo.wk_dim      = 1;

    std::string strLocalSize = std::to_string(kernelInfo.l_wk[0]);
    std::string strTreeMemSize =
        std::to_string(2 * sizeof(float) * kernelInfo.l_wk[0] * param.top_k);
    kernelInfo.comp_options = std::string(" -DLOCAL_WORK_SIZE=") + strLocalSize
                              + std::string(" -DTREE_MEM_SIZE=") + strTreeMemSize;

    AMDKernelPtr kptr = NULL;

    if (param.has_axis) {
        int count         = inputs[0]->count(0, inputs[0]->dims());
        int dim           = inputs[0]->shape()[param.axis];
        int inner_dim     = inputs[0]->count(param.axis + 1, inputs[0]->dims());
        int total_threads = count / dim;
        globalSize        = (total_threads + localSize - 1) / localSize * localSize;
        kernelInfo.g_wk   = {globalSize};

        if (param.top_k == 1) {
            kernelInfo.kernel_name = "top1_channel";
            kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernel_map[TOP_1_CHANNEL]     = kptr;
            _globalWorkSize[TOP_1_CHANNEL] = kernelInfo.g_wk[0];
            _localWorkSize[TOP_1_CHANNEL]  = kernelInfo.l_wk[0];
        } else {
            kernelInfo.kernel_name = "topk_channel";
            kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernel_map[TOPK_CHANNEL]     = kptr;
            _globalWorkSize[TOPK_CHANNEL] = kernelInfo.g_wk[0];
            _localWorkSize[TOPK_CHANNEL]  = kernelInfo.l_wk[0];
        }
    } else {
        int inner_dim = inputs[0]->count(1, inputs[0]->dims());
        int outer_dim = inputs[0]->num();

        if (param.top_k == 1) {
            if (inner_dim / kernelInfo.l_wk[0] < 10) {
                globalSize             = outer_dim * kernelInfo.l_wk[0];
                kernelInfo.g_wk        = {globalSize};
                kernelInfo.kernel_name = "top1";
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (NULL == kptr) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernel_map[TOP_1]     = kptr;
                _globalWorkSize[TOP_1] = kernelInfo.g_wk[0];
                _localWorkSize[TOP_1]  = kernelInfo.l_wk[0];
            } else {
                kernelInfo.kernel_name = "block_top1";
                int groupNum           = (inner_dim + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0];
                globalSize             = groupNum * outer_dim * kernelInfo.l_wk[0];
                kernelInfo.g_wk        = {globalSize};
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (NULL == kptr) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernel_map[BLOCK_TOP_1]     = kptr;
                _globalWorkSize[BLOCK_TOP_1] = kernelInfo.g_wk[0];
                _localWorkSize[BLOCK_TOP_1]  = kernelInfo.l_wk[0];

                kernelInfo.kernel_name = "top1_big";
                globalSize             = outer_dim * kernelInfo.l_wk[0];
                kernelInfo.g_wk        = {globalSize};
                kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (NULL == kptr) {
                    LOG(ERROR) << "Failed to load program";
                    return SaberInvalidValue;
                }

                _kernel_map[TOP_1_BIG]     = kptr;
                _globalWorkSize[TOP_1_BIG] = kernelInfo.g_wk[0];
                _localWorkSize[TOP_1_BIG]  = kernelInfo.l_wk[0];
            }
        } else {
            globalSize             = outer_dim * kernelInfo.l_wk[0];
            kernelInfo.g_wk        = {globalSize};
            kernelInfo.kernel_name = "topk_heap_shared";
            kptr                   = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (NULL == kptr) {
                LOG(ERROR) << "Failed to load program";
                return SaberInvalidValue;
            }

            _kernel_map[TOPK_HEAP_SHARED]     = kptr;
            _globalWorkSize[TOPK_HEAP_SHARED] = kernelInfo.g_wk[0];
            _localWorkSize[TOPK_HEAP_SHARED]  = kernelInfo.l_wk[0];
        }
    }

    LOG(ERROR) << "COMPLETE CREATE KERNEL";

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberArgmax<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ArgmaxParam<AMD>& param) {
    // to get the compute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    // to get the argument
    int outer_dim     = inputs[0]->count(0, param.axis);
    bool err          = false;
    AMDKernel* kernel = NULL;

    amd_kernel_list list;

    if (param.has_axis) {
        int count         = inputs[0]->count(0, inputs[0]->dims());
        int dim           = inputs[0]->shape()[param.axis];
        int inner_dim     = inputs[0]->count(param.axis + 1, inputs[0]->dims());
        int total_threads = count / dim;

        if (param.top_k == 1) {
            kernel = _kernel_map[TOP_1_CHANNEL].get();
            err    = kernel->SetKernelArgs(
                         (PtrDtype)inputs[0]->data(),
                         (int)outer_dim,
                         (int)dim,
                         (int)inner_dim,
                         (int)param.out_max_val,
                         (PtrDtype)outputs[0]->mutable_data());

            if (!err) {
                LOG(ERROR) << "Failed to set kernel args";
                return SaberInvalidValue;
            }

            list.push_back(_kernel_map[TOP_1_CHANNEL]);
        } else {
            kernel = _kernel_map[TOPK_CHANNEL].get();
            err    = kernel->SetKernelArgs(
                         (PtrDtype)inputs[0]->data(),
                         (int)outer_dim,
                         (int)dim,
                         (int)inner_dim,
                         (int)param.top_k,
                         (int)param.out_max_val,
                         (PtrDtype)outputs[0]->mutable_data());

            if (!err) {
                LOG(ERROR) << "Failed to set kernel args";
                return SaberInvalidValue;
            }

            list.push_back(_kernel_map[TOPK_CHANNEL]);
        }
    } else {
        int inner_dim = inputs[0]->count(1, inputs[0]->dims());
        int outer_dim = inputs[0]->num();

        if (param.top_k == 1) {
            if (inner_dim / _localWorkSize[TOP_1] < 10) {
                kernel = _kernel_map[TOP_1].get();
                err    = kernel->SetKernelArgs(
                             (PtrDtype)inputs[0]->data(),
                             (int)outer_dim,
                             (int)inner_dim,
                             (int)param.out_max_val,
                             (PtrDtype)outputs[0]->mutable_data());

                if (!err) {
                    LOG(ERROR) << "Failed to set kernel args";
                    return SaberInvalidValue;
                }

                list.push_back(_kernel_map[TOP_1]);
            } else {
                kernel = _kernel_map[BLOCK_TOP_1].get();
                err    = kernel->SetKernelArgs(
                             (PtrDtype)inputs[0]->data(),
                             (int)outer_dim,
                             (int)inner_dim,
                             (PtrDtype)_group_max_value.mutable_data(),
                             (PtrDtype)_group_max_index.mutable_data());

                if (!err) {
                    LOG(ERROR) << "Failed to set kernel args";
                    return SaberInvalidValue;
                }

                list.push_back(_kernel_map[BLOCK_TOP_1]);

                int groupNum =
                    (inner_dim + _localWorkSize[TOP_1_BIG] - 1) / _localWorkSize[TOP_1_BIG];
                kernel = _kernel_map[TOP_1_BIG].get();
                err    = kernel->SetKernelArgs(
                             (PtrDtype)_group_max_value.data(),
                             (PtrDtype)_group_max_index.data(),
                             (int)outer_dim,
                             (int)groupNum,
                             (int)param.out_max_val,
                             (PtrDtype)outputs[0]->mutable_data());

                if (!err) {
                    LOG(ERROR) << "Failed to set kernel args";
                    return SaberInvalidValue;
                }

                list.push_back(_kernel_map[TOP_1_BIG]);
            }
        } else {
            kernel = _kernel_map[TOPK_HEAP_SHARED].get();
            err    = kernel->SetKernelArgs(
                         (PtrDtype)outputs[0]->mutable_data(),
                         (int)outer_dim,
                         (int)inner_dim,
                         (int)param.top_k,
                         (int)param.out_max_val,
                         (PtrDtype)inputs[0]->data());

            if (!err) {
                LOG(ERROR) << "Failed to set kernel args";
                return SaberInvalidValue;
            }

            list.push_back(_kernel_map[TOPK_HEAP_SHARED]);
        }
    }

    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    LOG(INFO) << "COMPLETE EXECUTION";
    return SaberSuccess;
}

template class SaberArgmax<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberArgmax, ArgmaxParam, AMD, AK_INT8);
DEFINE_OP_TEMPLATE(SaberArgmax, ArgmaxParam, AMD, AK_HALF);
} // namespace saber
} // namespace anakin
