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
#include "saber/funcs/conv.h"
#include "saber/funcs/impl/amd/include/saber_conv_depthwise.h"
#include "saber/funcs/impl/amd/include/vender_conv.h"
#include "saber/funcs/impl/amd/include/amd_utils.h"

namespace anakin {
namespace saber {

typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;
typedef Tensor<AMD> TensorDf4;

template <DataType OpDtype>
SaberStatus SaberConv2D<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
void SaberConv2D<AMD, OpDtype>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return;
    }

    _kernels_ptr.push_back(kptr);
}

template <DataType OpDtype>
SaberStatus SaberConv2D<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param,
    Context<AMD>& ctx) {
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "create";

    this->_ctx = &ctx;

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "num=" << inputs[0]->num() << " channel=" <<
                                         inputs[0]->channel()
                                         << " height=" << inputs[0]->height() << " width=" << inputs[0]->width();

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "stride_h=" << param.stride_h << " stride_w=" <<
                                         param.stride_w << " height="
                                         << param.weight()->height() << " width=" << param.weight()->width()
                                         << " group=" << param.group << " dilation_h=" << param.dilation_h
                                         << " dilation_w=" << param.dilation_w << " pad_h=" << param.pad_h << " pad_w"
                                         << param.pad_w << " alpha=" << param.alpha << " beta=" << param.beta;

    if (param.activation_param.has_active) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "activation.active=" << param.activation_param.active;
    }

    if (param.bias()->valid_size() > 0) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "bias size=" << param.bias()->size();
    }

    std::vector<KernelInfo> solution = FindSolution(inputs, outputs, param);

    if (!solution.empty()) {
        for (auto s : solution) {
            if (s.kernel_name == "conv1x1_act"
                    || s.kernel_name == "InnerProduct") {
                CreateKernelList(inputs[0]->device_id(), s);
            } else {
                _use_vender = true;
            }
        }

        if (_use_vender) {
            VenderConv2D<AMD, OpDtype>* vc = new VenderConv2D<AMD, OpDtype>;
            vc->set_solution(solution);
            this->_impl = vc;
            this->_impl->create(inputs, outputs, param, ctx);
        }
    } else {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "No solution found!!!";

        if ((param.group == inputs[0]->channel()) && (param.group == outputs[0]->channel())) {
            SaberDepthWiseConv<OpDtype>* sdc = new SaberDepthWiseConv<OpDtype>;
            this->_impl = sdc;
        } else {
            // not 1x1
            VenderConv2D<AMD, OpDtype>* vc = new VenderConv2D<AMD, OpDtype>;
            this->_impl = vc;
        }

        this->_impl->create(inputs, outputs, param, ctx);
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberConv2D<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param) {

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "dispatch";
    int err;
    amd_kernel_list list;
    bool isBias   = param.bias()->size() > 0 ? true : false;
    bool isActive = false;
    float negative_slope = 1.0f;
    unsigned int out_offset = 0;
    unsigned int in_offset  = 0;
    float floatObjects[2]   = {1.0f, 0.0f};

    // To get the commpute command queue
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << " num=" << inputs[0]->num() << " channel=" <<
                                         inputs[0]->channel()
                                         << " height=" << inputs[0]->height() << " width=" << inputs[0]->width()
                                         << " param.weight()->num()=" << param.weight()->num()
                                         << " param.weight()->channel()=" << param.weight()->channel()
                                         << " param.weight()->width()=" << param.weight()->width()
                                         << " param.weight()->height()=" << param.weight()->height() << " param.group="
                                         << param.group << " param.pad_h=" << param.pad_h << " param.pad_w=" << param.pad_w
                                         << " param.stride_h=" << param.stride_h << " param.stride_w=" << param.stride_w
                                         << " param.dilation_h=" << param.dilation_h
                                         << " param.dilation_w=" << param.dilation_w << " param.alpha=" << param.alpha
                                         << " param.beta=" << param.beta;

    if (isBias) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << " param.bias()->size()=" << param.bias()->size()
                                             << " param.bias()->channel()=" << param.bias()->channel()
                                             << " param.bias()->width()=" << param.bias()->width()
                                             << " param.bias()->height()=" << param.bias()->height();
    }

    if (param.activation_param.has_active) {
        isActive = true;

        if (param.activation_param.active == Active_relu) {
            negative_slope = 0.0f;
        } else {
            negative_slope = param.activation_param.negative_slope;
        }

        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << " param.activation_param.has_active="
                                             << param.activation_param.has_active
                                             << " param.activation_param.negative_slope=" << param.activation_param.negative_slope
                                             << " param.activation_param.active=" << param.activation_param.active
                                             << " param.activation_param.coef=" << param.activation_param.coef;
    }

    if (_impl != nullptr) {
        return _impl->dispatch(inputs, outputs, param);
    } else {
        if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
            LOG(ERROR) << "Kernel is not exist";
            return SaberInvalidValue;
        }
    }

    for (int i = 0; i < _kernels_ptr.size(); i++) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "kernel size:" << _kernels_ptr.size() << " name:" <<
                                             _kernels_ptr[i].get()->GetName();

        if (_kernels_ptr[i].get()->GetName() == "conv1x1_act") {
            if (isBias) {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)param.weight()->data(),
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)param.bias()->data(),
                          (PtrDtype)outputs[0]->mutable_data(),
                          negative_slope);
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)param.weight()->data(),
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)outputs[0]->mutable_data(),
                          negative_slope);
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "InnerProduct") {
            if (isBias) {
                if (isActive) {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)param.bias()->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              negative_slope);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)param.bias()->data(),
                              (PtrDtype)outputs[0]->mutable_data());
                }
            } else {
                if (isActive) {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              negative_slope);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)outputs[0]->mutable_data());
                }
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else {
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "Not implement kernel name:" <<
                                                 _kernels_ptr[i].get()->GetName();
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

template class SaberConv2D<AMD, AK_FLOAT>;
template class SaberConv2D<AMD, AK_HALF>;
template class SaberConv2D<AMD, AK_INT8>;
} // namespace saber
} // namespace anakin
