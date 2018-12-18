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
#include "saber/funcs/impl/amd/include/saber_conv.h"
#include "saber/funcs/impl/amd/include/saber_conv_depthwise.h"

namespace anakin {
namespace saber {
template <DataType OpDtype>
SaberStatus SaberDepthWiseConv<OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberDepthWiseConv<OpDtype>::create(
    const std::vector<Tensor<AMD> *>& inputs,
    std::vector<Tensor<AMD> *>& outputs,
    ConvParam<AMD>& param, Context<AMD>& ctx) {
    this->_ctx = &ctx;
    KernelInfo kernelInfo;
    bool isBias = false;

    if (param.bias()->valid_size() > 0) {
        isBias = true;
    }

    if ((param.group == inputs[0]->channel()) && (param.group == outputs[0]->channel())) {
        LOG(INFO) << "Group conv's kernel_type " << kernelInfo.kernel_type;

        int isActiveRelu = 0;

        if (param.activation_param.has_active) {
            if (param.activation_param.active == Active_relu) {
                isActiveRelu = 1;
            }
        }

        kernelInfo.comp_options += std::string(" -DMLO_CONV_BIAS=") + std::to_string(isBias) +
                                   std::string(" -DMLO_CONV_ACTIVE_RELU=") + std::to_string(isActiveRelu);

        kernelInfo.wk_dim = 1;
        kernelInfo.l_wk   = {256};
        kernelInfo.g_wk   = {(inputs[0]->num() * inputs[0]->channel() * outputs[0]->height()
                              * outputs[0]->width()
                              + 256 - 1)
                             / 256 * 256
                            };
        kernelInfo.kernel_file = "Depthwiseconv.cl";
        kernelInfo.kernel_name = "Depthwiseconv";
        kernelInfo.kernel_type = SABER;
        AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to load program";
            return SaberInvalidValue;
        }

        _kernels_ptr.push_back(kptr);
    } else {
        LOG(ERROR) << "Not implementation !!!";
    }

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberDepthWiseConv<OpDtype>::dispatch(
    const std::vector<Tensor<AMD> *>& inputs,
    std::vector<Tensor<AMD> *>& outputs,
    ConvParam<AMD>& param) {

    CHECK_EQ(inputs[0]->get_dtype(), AK_FLOAT);
    CHECK_EQ(outputs[0]->get_dtype(), AK_FLOAT);

    int err;
    amd_kernel_list list;
    bool isBias   = param.bias()->size() > 0 ? true : false;

    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
        LOG(ERROR) << "Kernel is not exist";
        return SaberInvalidValue;
    }

    for (int i = 0; i < _kernels_ptr.size(); i++) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "kernel size:" << _kernels_ptr.size() << " name:" <<
                                             _kernels_ptr[i].get()->GetName();

        if (_kernels_ptr[i].get()->GetName() == "Depthwiseconv") {
            if (isBias) {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)inputs[0]->data(),
                          (int)inputs[0]->num(),
                          (int)inputs[0]->channel(),
                          (int)inputs[0]->height(),
                          (int)inputs[0]->width(),
                          (int)outputs[0]->height(),
                          (int)outputs[0]->width(),
                          (int)param.weight()->height(),
                          (int)param.weight()->width(),
                          (int)param.stride_h,
                          (int)param.stride_w,
                          (int)param.pad_h,
                          (int)param.pad_w,
                          (PtrDtype)outputs[0]->mutable_data(),
                          (PtrDtype)param.weight()->data(),
                          (PtrDtype)param.bias()->data());
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)inputs[0]->data(),
                          (int)inputs[0]->num(),
                          (int)inputs[0]->channel(),
                          (int)inputs[0]->height(),
                          (int)inputs[0]->width(),
                          (int)outputs[0]->height(),
                          (int)outputs[0]->width(),
                          (int)param.weight()->height(),
                          (int)param.weight()->width(),
                          (int)param.stride_h,
                          (int)param.stride_w,
                          (int)param.pad_h,
                          (int)param.pad_w,
                          (PtrDtype)outputs[0]->mutable_data(),
                          (PtrDtype)param.weight()->data());
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else {
            LOG(ERROR) << "Not implementation!!";
        }
    }

    if (list.size() > 0) {
        err = LaunchKernel(cm, list);

        if (!err) {
            LOG(ERROR) << "Fail to set execution :" << err;
            return SaberInvalidValue;
        }
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";
    return SaberSuccess;
}

template class SaberDepthWiseConv<AK_FLOAT>;
template class SaberDepthWiseConv<AK_HALF>;
template class SaberDepthWiseConv<AK_INT8>;
}
}
