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
#include "saber/funcs/impl/amd/include/saber_conv_pooling.h"
#include "saber/funcs/conv.h"
#include "saber/funcs/impl/amd/include/amd_utils.h"
#include "saber/funcs/impl/amd/include/vender_conv_pooling.h"

namespace anakin {
namespace saber {
typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;

template <>
SaberConv2DPooling<AMD, AK_FLOAT>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    _kernels_ptr.push_back(kptr);
}

template <>
SaberStatus SaberConv2DPooling<AMD, AK_FLOAT>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvPoolingParam<AMD>& param,
    Context<AMD>& ctx) {

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "create";

    // NOTE: The width and height of output are parameters for convolution in conv_act_pooling
    std::vector<Tensor<AMD>*> conv_outputs;
    Tensor<AMD>* conv_out = new Tensor<AMD>();
    conv_outputs.push_back(conv_out);
    Conv<AMD, AK_FLOAT> conv;
    conv.compute_output_shape(inputs, conv_outputs, param.conv_param);
    conv_out->re_alloc(conv_out->shape());
    _outConvRelu = conv_out;

    bool isBias = (param.conv_param.bias()->size() > 0) ? true : false;
    std::vector<KernelInfo> solution = FindSolutionWithPooling(inputs, _outConvRelu, outputs, param);

    if (!solution.empty()) {
        bool _use_vender = false;

        for (auto s : solution) {
            if (s.kernel_name == "conv1x1_act_pool"
                    || s.kernel_name == "conv1x1_act"
                    || s.kernel_name == "mloPooling"
                    || s.kernel_name == "PoolingGeneral"
                    || s.kernel_name == "PoolingWithShare") {
                if (s.kernel_name == "conv1x1_act") {
                    if (_conv1x1_act_lock == nullptr) {
                        _conv1x1_act_lock = new Tensor<AMD>();
                    }

                    _conv1x1_act_lock->re_alloc(Shape({outputs[0]->count(1, 4)}, Layout_W), AK_FLOAT);
                }

                CreateKernelList(inputs[0]->device_id(), s);
            } else {
                _use_vender = true;
                break;
            }
        }

        if (_use_vender) {
            VenderConv2DPooling<AMD, AK_FLOAT>* vcp = new VenderConv2DPooling<AMD, AK_FLOAT>;
            vcp->set_solution(solution);
            this->_impl = vcp;
            this->_impl->create(inputs, outputs, param, ctx);

            if (_outConvRelu) {
                delete _outConvRelu;
                _outConvRelu = nullptr;
            }
        } else if (solution[0].kernel_name  == "conv1x1_act_pool") {
            if (_outConvRelu) {
                delete _outConvRelu;
                _outConvRelu = nullptr;
            }
        }
    } else {
        VenderConv2DPooling<AMD, AK_FLOAT>* vcp = new VenderConv2DPooling<AMD, AK_FLOAT>;
        this->_impl = vcp;
        this->_impl->create(inputs, outputs, param, ctx);

        if (_outConvRelu) {
            delete _outConvRelu;
            _outConvRelu = nullptr;
        }
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}

template <>
SaberStatus SaberConv2DPooling<AMD, AK_FLOAT>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvPoolingParam<AMD>& param,
    Context<AMD>& ctx) {
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "init";
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConv2DPooling<AMD, AK_FLOAT>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvPoolingParam<AMD>& param) {
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "dispatch";

    bool err;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    amd_kernel_list list;
    bool isBias = (param.conv_param.bias()->size() > 0) ? true : false;
    bool isActive = false;
    bool needBias = false;
    float negative_slope = 1.0f;
    unsigned int out_offset = 0;
    unsigned int in_offset  = 0;
    float floatObjects[2]   = {1.0f, 0.0f};

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << " num=" << inputs[0]->num() << " channel=" <<
                                         inputs[0]->channel()
                                         << " height=" << inputs[0]->height() << " width=" << inputs[0]->width()
                                         << " param.conv_param.weight()->num()=" << param.conv_param.weight()->num()
                                         << " param.conv_param.weight()->channel()="
                                         << param.conv_param.weight()->channel()
                                         << " param.conv_param.weight()->width()=" << param.conv_param.weight()->width()
                                         << " param.conv_param.weight()->height()=" << param.conv_param.weight()->height()
                                         << " param.conv_param.group=" << param.conv_param.group
                                         << " param.conv_param.pad_h=" << param.conv_param.pad_h
                                         << " param.conv_param.pad_w=" << param.conv_param.pad_w
                                         << " param.conv_param.stride_h=" << param.conv_param.stride_h
                                         << " param.conv_param.stride_w=" << param.conv_param.stride_w
                                         << " param.conv_param.dilation_h=" << param.conv_param.dilation_h
                                         << " param.conv_param.dilation_w=" << param.conv_param.dilation_w
                                         << " param.conv_param.alpha=" << param.conv_param.alpha
                                         << " param.conv_param.beta=" << param.conv_param.beta
                                         << " param.pooling_param.window_h=" << param.pooling_param.window_h
                                         << " param.pooling_param.window_w=" << param.pooling_param.window_w
                                         << " param.pooling_param.pad_h=" << param.pooling_param.pad_h
                                         << " param.pooling_param.pad_w=" << param.pooling_param.pad_w
                                         << " param.pooling_param.stride_h=" << param.pooling_param.stride_h
                                         << " param.pooling_param.stride_w=" << param.pooling_param.stride_w
                                         << " param.pooling_param.pooling_type=" << param.pooling_param.pooling_type
                                         << " param.pooling_param.global_pooling=" << param.pooling_param.global_pooling
                                         << " param.pooling_param.cmp_out_shape_floor_as_conv="
                                         << param.pooling_param.cmp_out_shape_floor_as_conv;

    if (isBias) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "param.conv_param.bias()->size()=" <<
                                             param.conv_param.bias()->size()
                                             << " param.conv_param.bias()->channel()=" << param.conv_param.bias()->channel()
                                             << " param.conv_param.bias()->width()=" << param.conv_param.bias()->width()
                                             << " param.conv_param.bias()->height()=" << param.conv_param.bias()->height();
    }

    if (param.conv_param.activation_param.has_active) {
        isActive = true;

        if (param.conv_param.activation_param.active == Active_relu) {
            negative_slope = 0.0f;
        } else {
            negative_slope = param.conv_param.activation_param.negative_slope;
        }

        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "param.has_active=" <<
                                             param.conv_param.activation_param.has_active
                                             << " param.conv_param.activation_param.negative_slope="
                                             << param.conv_param.activation_param.negative_slope
                                             << " param.conv_param.activation_param.active="
                                             << param.conv_param.activation_param.active
                                             << " param.conv_param.activation_param.coef="
                                             << param.conv_param.activation_param.coef;
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

        if (_kernels_ptr[i].get()->GetName() == "conv1x1_act_pool") {
            if (isBias) {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)param.conv_param.weight()->data(),
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)(isBias) ? (PtrDtype)param.conv_param.bias()->data() : nullptr,
                          (PtrDtype)outputs[0]->mutable_data(),
                          negative_slope);
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)param.conv_param.weight()->data(),
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)outputs[0]->mutable_data(),
                          negative_slope);
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "conv1x1_act") {
            if (isBias) {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)param.conv_param.weight()->data(),
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)param.conv_param.bias()->data(),
                          (PtrDtype)_conv1x1_act_lock->mutable_data(),
                          (PtrDtype)_outConvRelu->mutable_data(),
                          negative_slope,
                          (unsigned int)inputs[0]->channel(),
                          (unsigned int)inputs[0]->height(),
                          (unsigned int)inputs[0]->width(),
                          (unsigned int)param.conv_param.weight()->num());
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)param.conv_param.weight()->data(),
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)_conv1x1_act_lock->mutable_data(),
                          (PtrDtype)_outConvRelu->mutable_data(),
                          negative_slope,
                          (unsigned int)inputs[0]->channel(),
                          (unsigned int)inputs[0]->height(),
                          (unsigned int)inputs[0]->width(),
                          (unsigned int)param.conv_param.weight()->num());
            }


            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "mloPooling") {
            if (needBias) {
                if (isBias) {
                    if (isActive) {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)_outConvRelu->data(),
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  (PtrDtype)param.conv_param.bias()->data(),
                                  negative_slope,
                                  (PtrDtype)0);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)_outConvRelu->data(),
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  (PtrDtype)param.conv_param.bias()->data(),
                                  (PtrDtype)0);
                    }
                } else {
                    if (isActive) {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)_outConvRelu->data(),
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  negative_slope,
                                  (PtrDtype)0);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)_outConvRelu->data(),
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  (PtrDtype)0);
                    }
                }
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)_outConvRelu->data(),
                          (PtrDtype)outputs[0]->mutable_data(),
                          (PtrDtype)0);
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "PoolingGlobal") {
            if (needBias) {
                if (isBias) {
                    if (isActive) {
                        err = _kernels_ptr[i].get()->SetKernelArgs((PtrDtype)_outConvRelu->data(),
                                (PtrDtype)outputs[0]->mutable_data(),
                                (PtrDtype)param.conv_param.bias()->data(),
                                negative_slope,
                                (int)_outConvRelu->num(),
                                (int)_outConvRelu->channel(),
                                (int)_outConvRelu->height(),
                                (int)_outConvRelu->width(),
                                (int)param.pooling_param.pad_h,
                                (int)param.pooling_param.pad_w);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs((PtrDtype)_outConvRelu->data(),
                                (PtrDtype)outputs[0]->mutable_data(),
                                (PtrDtype)param.conv_param.bias()->data(),
                                (int)_outConvRelu->num(),
                                (int)_outConvRelu->channel(),
                                (int)_outConvRelu->height(),
                                (int)_outConvRelu->width(),
                                (int)param.pooling_param.pad_h,
                                (int)param.pooling_param.pad_w);
                    }
                } else {
                    if (isActive) {
                        err = _kernels_ptr[i].get()->SetKernelArgs((PtrDtype)_outConvRelu->data(),
                                (PtrDtype)outputs[0]->mutable_data(),
                                negative_slope,
                                (int)_outConvRelu->num(),
                                (int)_outConvRelu->channel(),
                                (int)_outConvRelu->height(),
                                (int)_outConvRelu->width(),
                                (int)param.pooling_param.pad_h,
                                (int)param.pooling_param.pad_w);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs((PtrDtype)_outConvRelu->data(),
                                (PtrDtype)outputs[0]->mutable_data(),
                                (int)_outConvRelu->num(),
                                (int)_outConvRelu->channel(),
                                (int)_outConvRelu->height(),
                                (int)_outConvRelu->width(),
                                (int)param.pooling_param.pad_h,
                                (int)param.pooling_param.pad_w);
                    }
                }
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs((PtrDtype)_outConvRelu->data(),
                        (PtrDtype)outputs[0]->mutable_data(),
                        (int)_outConvRelu->num(),
                        (int)_outConvRelu->channel(),
                        (int)_outConvRelu->height(),
                        (int)_outConvRelu->width(),
                        (int)param.pooling_param.pad_h,
                        (int)param.pooling_param.pad_w);
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "PoolingGeneral"
                   || _kernels_ptr[i].get()->GetName() == "PoolingWithShare") {
            if (needBias) {
                if (isBias) {
                    if (isActive) {
                        err = _kernels_ptr[i].get()->SetKernelArgs((PtrDtype)_outConvRelu->data(),
                                (PtrDtype)outputs[0]->mutable_data(),
                                (PtrDtype)param.conv_param.bias()->data(),
                                negative_slope,
                                (int)_outConvRelu->num(),
                                (int)_outConvRelu->channel(),
                                (int)_outConvRelu->height(),
                                (int)_outConvRelu->width(),
                                (int)outputs[0]->height(),
                                (int)outputs[0]->width(),
                                (int)param.pooling_param.window_h,
                                (int)param.pooling_param.window_w,
                                (int)param.pooling_param.stride_h,
                                (int)param.pooling_param.stride_w,
                                (int)param.pooling_param.pad_h,
                                (int)param.pooling_param.pad_w);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs((PtrDtype)_outConvRelu->data(),
                                (PtrDtype)outputs[0]->mutable_data(),
                                (PtrDtype)param.conv_param.bias()->data(),
                                (int)_outConvRelu->num(),
                                (int)_outConvRelu->channel(),
                                (int)_outConvRelu->height(),
                                (int)_outConvRelu->width(),
                                (int)outputs[0]->height(),
                                (int)outputs[0]->width(),
                                (int)param.pooling_param.window_h,
                                (int)param.pooling_param.window_w,
                                (int)param.pooling_param.stride_h,
                                (int)param.pooling_param.stride_w,
                                (int)param.pooling_param.pad_h,
                                (int)param.pooling_param.pad_w);
                    }
                } else {
                    if (isActive) {
                        err = _kernels_ptr[i].get()->SetKernelArgs((PtrDtype)_outConvRelu->data(),
                                (PtrDtype)outputs[0]->mutable_data(),
                                negative_slope,
                                (int)_outConvRelu->num(),
                                (int)_outConvRelu->channel(),
                                (int)_outConvRelu->height(),
                                (int)_outConvRelu->width(),
                                (int)outputs[0]->height(),
                                (int)outputs[0]->width(),
                                (int)param.pooling_param.window_h,
                                (int)param.pooling_param.window_w,
                                (int)param.pooling_param.stride_h,
                                (int)param.pooling_param.stride_w,
                                (int)param.pooling_param.pad_h,
                                (int)param.pooling_param.pad_w);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs((PtrDtype)_outConvRelu->data(),
                                (PtrDtype)outputs[0]->mutable_data(),
                                (int)_outConvRelu->num(),
                                (int)_outConvRelu->channel(),
                                (int)_outConvRelu->height(),
                                (int)_outConvRelu->width(),
                                (int)outputs[0]->height(),
                                (int)outputs[0]->width(),
                                (int)param.pooling_param.window_h,
                                (int)param.pooling_param.window_w,
                                (int)param.pooling_param.stride_h,
                                (int)param.pooling_param.stride_w,
                                (int)param.pooling_param.pad_h,
                                (int)param.pooling_param.pad_w);
                    }
                }
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs((PtrDtype)_outConvRelu->data(),
                        (PtrDtype)outputs[0]->mutable_data(),
                        (int)_outConvRelu->num(),
                        (int)_outConvRelu->channel(),
                        (int)_outConvRelu->height(),
                        (int)_outConvRelu->width(),
                        (int)outputs[0]->height(),
                        (int)outputs[0]->width(),
                        (int)param.pooling_param.window_h,
                        (int)param.pooling_param.window_w,
                        (int)param.pooling_param.stride_h,
                        (int)param.pooling_param.stride_w,
                        (int)param.pooling_param.pad_h,
                        (int)param.pooling_param.pad_w);

            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else {
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "disptach non-implementation kernel: " <<
                                                 _kernels_ptr[i].get()->GetName();
        }
    }

    if (list.size() > 0) {
        err = LaunchKernel(cm, list);

        if (!err) {
            LOG(ERROR) << "Fail to set execution";
            return SaberInvalidValue;
        }
    }

    return SaberSuccess;
}
template class SaberConv2DPooling<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConv2DPooling, ConvPoolingParam, AMD, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConv2DPooling, ConvPoolingParam, AMD, AK_INT8);

} // namespace saber
} // namespace anakin
