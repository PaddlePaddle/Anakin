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
#include "saber/funcs/impl/amd/include/vender_conv_pooling.h"
#include "saber/funcs/conv.h"
#include "saber/funcs/impl/amd/include/amd_utils.h"
#include "saber/funcs/impl/amd/include/amd_gemm.h"

namespace anakin {
namespace saber {
typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;

template <DataType OpDtype>
SaberStatus VenderConv2DPooling<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvPoolingParam<AMD>& param,
    Context<AMD>& ctx) {
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "init";
    impl_vender = true;
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
void VenderConv2DPooling<AMD, OpDtype>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }

    _kernels_ptr.push_back(kptr);
}

template <DataType OpDtype>
SaberStatus VenderConv2DPooling<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvPoolingParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender) << "create";

    LOG_IF_S(INFO, impl_vender)
            << " N " << inputs[0]->num()
            << " C " << inputs[0]->channel()
            << " H " << inputs[0]->height()
            << " W " << inputs[0]->width();

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender)
            << "op param K " << param.conv_param.weight()->num()
            << " Y " << param.conv_param.weight()->height() << " X " << param.conv_param.weight()->width()
            << " SH " << param.conv_param.stride_h << " SW " << param.conv_param.stride_w
            << " PH " << param.conv_param.pad_h << " PW " << param.conv_param.pad_w
            << " DH " << param.conv_param.dilation_h << " DW " << param.conv_param.dilation_w
            << " Alpha " << param.conv_param.alpha << " Beta " << param.conv_param.beta << " GP " <<
            param.conv_param.group
            << " PWH " << param.pooling_param.window_h << " PWW " << param.pooling_param.window_w
            << " PPH " << param.pooling_param.pad_h << " PPW " << param.pooling_param.pad_w
            << " PSH " << param.pooling_param.stride_h << " PSW " << param.pooling_param.stride_w
            << " PType " << param.pooling_param.pooling_type << " GP " << param.pooling_param.global_pooling
            << " CMP " << param.pooling_param.cmp_out_shape_floor_as_conv;

    if (param.conv_param.activation_param.has_active) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender) << "activation.ActType=" <<
                param.conv_param.activation_param.active
                << " slop " << param.conv_param.activation_param.negative_slope
                << " coef " << param.conv_param.activation_param.coef;
    }

    bool isBias = false;

    if (param.conv_param.bias()->valid_size() > 0) {
        LOG_IF_S(INFO, impl_vender) << "bias size=" << param.conv_param.bias()->size();
        isBias = true;
    }

    cl_context context  = 0;
    cl_device_id device = 0;
    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()]; // anakin device id to AMD device
    device          = dev.get_device();
    context         = dev.get_context();
    std::string dev_name = dev._info._device_name;

    // NOTE: The width and height of output are parameters for convolution in conv_act_pooling
    std::vector<Tensor<AMD>*> conv_outputs;
    Tensor<AMD>* conv_out = new Tensor<AMD>();
    conv_outputs.push_back(conv_out);
    Conv<AMD, AK_FLOAT> conv;
    conv.compute_output_shape(inputs, conv_outputs, param.conv_param);
    conv_out->re_alloc(conv_out->shape());
    _outConvRelu = conv_out;

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "VenderConv2DPooling create kernel number:" <<
                                         vkernel.size();
    KernelInfo kernelInfo;

    if (!vkernel.empty()) {
        for (int i = 0; i < vkernel.size(); i++) {
            kernelInfo = vkernel[i];

            if (kernelInfo.kernel_name == "xGemm") {
                _outGemmWorkspace = new Tensor<AMD>();
                std::vector<AMDKernelPtr> vkptr;
                bool needBiasRelu = false;
                bool bias = false;
                bool relu = false;
                _outGemmWorkspace->re_alloc(
                    Shape({(inputs[0]->num() * 2),
                           std::max({inputs[0]->channel(),
                                     param.conv_param.weight()->channel(),
                                     param.conv_param.weight()->num()
                                    }),
                           std::max((inputs[0]->height()), (outputs[0]->height())),
                           std::max((inputs[0]->width()), (outputs[0]->width()))
                          }));

                if (!findGenericGemm(true, vkptr, inputs, _outConvRelu, param.conv_param,
                                     _outGemmWorkspace, ctx, needBiasRelu)) {
                    return SaberInvalidValue;
                }

                if (needBiasRelu) {
                    bias = isBias;
                    relu = param.conv_param.activation_param.has_active;
                }

                BiasReluPool(
                    vkptr,
                    inputs[0]->device_id(),
                    inputs[0]->num(),
                    param.conv_param.weight()->num(),
                    _outConvRelu->height(),
                    _outConvRelu->width(),
                    _outConvRelu->channel(),
                    outputs[0]->height(),
                    outputs[0]->width(),
                    outputs[0]->channel(),
                    param.pooling_param.window_h,
                    param.pooling_param.window_w,
                    param.pooling_param.stride_h,
                    param.pooling_param.stride_w,
                    param.pooling_param.pad_h,
                    param.pooling_param.pad_w,
                    param.pooling_param.pooling_type,
                    bias,
                    relu);

                for (int i = 0; i < vkptr.size(); i++) {
                    _kernels_ptr.push_back(vkptr[i]);
                }

                vkptr.clear();
            } else {
                if (kernelInfo.kernel_name == "conv7x7c3h448w448k64u2v2p3q3f1b1prelupooling"
                        || kernelInfo.kernel_name == "conv7x7c3h448w448k64u2v2p3q3f1b0prelupooling"
                        || kernelInfo.kernel_name == "conv7x7c3h224w224k64u2v2p3q3f1b1prelu"
                        || kernelInfo.kernel_name == "conv7x7c3h224w224k64u2v2p3q3f1b0prelu") {
                    kernelInfo.wk_dim      = 3;
                }

                CreateKernelList(inputs[0]->device_id(), kernelInfo);
            }
        }
    } else {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "No solution found!!!";
        // not 1x1
        bool needExtrakernel = false;
        std::vector<AMDKernelPtr> vkptr;
        _outGemmWorkspace = new Tensor<AMD>();
        _outGemmWorkspace->re_alloc(
            Shape({(param.conv_param.weight()->height() * param.conv_param.weight()->width()),
                   std::max({inputs[0]->channel(),
                             param.conv_param.weight()->channel(),
                             param.conv_param.weight()->num()
                            }),
                   std::max((inputs[0]->height()), (outputs[0]->height())),
                   std::max((inputs[0]->width()), (outputs[0]->width()))
                  }));

        if (!findGenericGemm(false, vkptr,
                             inputs,
                             _outConvRelu,
                             param.conv_param,
                             _outConvRelu,
                             ctx, needExtrakernel)) {
            return SaberInvalidValue;
        }

        BiasReluPool(
            vkptr,
            inputs[0]->device_id(),
            inputs[0]->num(),
            param.conv_param.weight()->num(),
            _outConvRelu->height(),
            _outConvRelu->width(),
            _outConvRelu->channel(),
            outputs[0]->height(),
            outputs[0]->width(),
            outputs[0]->channel(),
            param.pooling_param.window_h,
            param.pooling_param.window_w,
            param.pooling_param.stride_h,
            param.pooling_param.stride_w,
            param.pooling_param.pad_h,
            param.pooling_param.pad_w,
            param.pooling_param.pooling_type,
            isBias,
            param.conv_param.activation_param.has_active);

        for (int i = 0; i < vkptr.size(); i++) {
            _kernels_ptr.push_back(vkptr[i]);
        }

        vkptr.clear();
    }

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus VenderConv2DPooling<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvPoolingParam<AMD>& param) {
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

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender) << " num=" << inputs[0]->num() << " channel=" <<
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
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender) << "param.conv_param.bias()->size()=" <<
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

        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender) << "param.has_active=" <<
                param.conv_param.activation_param.has_active
                << " param.conv_param.activation_param.negative_slope="
                << param.conv_param.activation_param.negative_slope
                << " param.conv_param.activation_param.active="
                << param.conv_param.activation_param.active
                << " param.conv_param.activation_param.coef="
                << param.conv_param.activation_param.coef;
    }

    for (int i = 0; i < _kernels_ptr.size(); i++) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "kernel size:" << _kernels_ptr.size() << " name:" <<
                                             _kernels_ptr[i].get()->GetName();

        if ((_kernels_ptr[i].get()->GetName() == "MIOpenConvUni")
                || (_kernels_ptr[i].get()->GetName() == "MIOpenConv1x1")
                || (_kernels_ptr[i].get()->GetName() == "MIOpenConv1x1pquv")
                || (_kernels_ptr[i].get()->GetName() == "MIOpenCvD3x3_WSR0")
                || (_kernels_ptr[i].get()->GetName() == "MIOpenCDFGen")
                || (_kernels_ptr[i].get()->GetName() == "MIOpenCDFGen4")) {

            if (isBias) {
                if (isActive) {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.conv_param.weight()->data(),
                              (PtrDtype)param.conv_param.bias()->data(),
                              (PtrDtype)_outConvRelu->mutable_data(),
                              negative_slope,
                              0.0f);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.conv_param.weight()->data(),
                              (PtrDtype)param.conv_param.bias()->data(),
                              (PtrDtype)_outConvRelu->mutable_data(),
                              0.0f);
                }
            } else {
                if (isActive) {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.conv_param.weight()->data(),
                              (PtrDtype)_outConvRelu->mutable_data(),
                              negative_slope,
                              0.0f);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.conv_param.weight()->data(),
                              (PtrDtype)_outConvRelu->mutable_data(),
                              0.0f);
                }
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "sp3AsmConv3x3F") {
            int d_n_groups = 64, d_flags = 0;
            PtrDtype biasMemObject = isBias ? param.conv_param.bias()->data() : 0;

            if (isBias && isActive
                    && param.pooling_param.pooling_type == Pooling_max
                    && param.pooling_param.window_h == 2
                    && param.pooling_param.window_w == 2
                    && param.pooling_param.stride_h == 2
                    && param.pooling_param.stride_w == 2
                    && param.pooling_param.pad_h == 0
                    && param.pooling_param.pad_w == 0) {
                err                    = _kernels_ptr[i].get()->SetKernelArgs(
                                             (unsigned int)inputs[0]->num(),
                                             (unsigned int)inputs[0]->channel(),
                                             (unsigned int)inputs[0]->height(),
                                             (unsigned int)inputs[0]->width(),
                                             (unsigned int)param.conv_param.weight()->num(),
                                             (unsigned int)d_n_groups,
                                             (unsigned int)d_flags,
                                             negative_slope,
                                             (PtrDtype)inputs[0]->data(),
                                             (PtrDtype)param.conv_param.weight()->data(),
                                             (PtrDtype)outputs[0]->mutable_data(),
                                             (PtrDtype)biasMemObject);
            } else {
                err                    = _kernels_ptr[i].get()->SetKernelArgs(
                                             (unsigned int)inputs[0]->num(),
                                             (unsigned int)inputs[0]->channel(),
                                             (unsigned int)inputs[0]->height(),
                                             (unsigned int)inputs[0]->width(),
                                             (unsigned int)param.conv_param.weight()->num(),
                                             (unsigned int)d_n_groups,
                                             (unsigned int)d_flags,
                                             negative_slope,
                                             (PtrDtype)inputs[0]->data(),
                                             (PtrDtype)param.conv_param.weight()->data(),
                                             (PtrDtype)_outConvRelu->mutable_data(),
                                             (PtrDtype)biasMemObject);
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "mloPooling") {
            if (needBias) {
                if (isBias) {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)_outConvRelu->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              (PtrDtype)param.conv_param.bias()->data(),
                              negative_slope);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)_outConvRelu->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              negative_slope);
                }
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)_outConvRelu->data(),
                          (PtrDtype)outputs[0]->mutable_data(),
                          1.0f);
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "mloPoolingG") {
            if (needBias) {
                if (isBias) {
                    if (isActive) {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)_outConvRelu->data(),
                                  (PtrDtype)param.conv_param.bias()->data(),
                                  negative_slope,
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  (PtrDtype) nullptr);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)_outConvRelu->data(),
                                  (PtrDtype)param.conv_param.bias()->data(),
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  (PtrDtype) nullptr);
                    }
                } else {
                    if (isActive) {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)_outConvRelu->data(),
                                  negative_slope,
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  (PtrDtype) nullptr);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)_outConvRelu->data(),
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  (PtrDtype) nullptr);
                    }
                }
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)_outConvRelu->data(),
                          (PtrDtype)outputs[0]->mutable_data(),
                          (PtrDtype) nullptr);
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "transpose_NCHW2CNHW_opt"
                   || _kernels_ptr[i].get()->GetName() == "transpose_NCHW2CNHW") {
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "GEMM 1x1, 14x14";

            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)_outGemmWorkspace->mutable_data());

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i++]);

            out_offset = (inputs[0]->num()) * (inputs[0]->channel())
                         * (_outConvRelu->height()) * (_outConvRelu->width());

            if (_kernels_ptr[i].get()->GetName() != "miog_betac_alphaab") {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)_outGemmWorkspace->mutable_data(), out_offset, floatObjects[1]);

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i++]);

                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)_outGemmWorkspace->mutable_data(),
                          in_offset,
                          (PtrDtype)param.conv_param.weight()->data(),
                          0,
                          (PtrDtype)_outGemmWorkspace->mutable_data(),
                          out_offset,
                          floatObjects[0]);
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)_outGemmWorkspace->data(),
                          in_offset,
                          (PtrDtype)param.conv_param.weight()->data(),
                          0,
                          (PtrDtype)_outGemmWorkspace->mutable_data(),
                          out_offset,
                          floatObjects[0],
                          floatObjects[1]);
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i++]);

            if (isBias) {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)_outGemmWorkspace->mutable_data(),
                          (PtrDtype)_outConvRelu->mutable_data(),
                          (PtrDtype)param.conv_param.bias()->data(),
                          negative_slope);
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)_outGemmWorkspace->mutable_data(),
                          (PtrDtype)_outConvRelu->mutable_data(),
                          negative_slope);
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "miog_betac_alphaab"
                   || _kernels_ptr[i].get()->GetName() == "miog_betac") {
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "GEMM 1x1";

            if (_kernels_ptr[i].get()->GetName() != "miog_betac_alphaab" || inputs[0]->num() > 1) {
                needBias = true;
            }

            for (int j = 0; j < (inputs[0]->num()); j++) { //so far, only responsible for batch size is 1
                in_offset = j * (inputs[0]->channel()) * (inputs[0]->height())
                            * (inputs[0]->width());
                out_offset = j * (param.conv_param.weight()->num()) * _outConvRelu->height()
                             * _outConvRelu->width();
                i = 0;

                if (_kernels_ptr[i].get()->GetName() != "miog_betac_alphaab") {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              _outConvRelu->mutable_data(), out_offset, floatObjects[1]);

                    if (!err) {
                        LOG(ERROR) << "Fail to set kernel args :" << err;
                        return SaberInvalidValue;
                    }

                    list.push_back(_kernels_ptr[i++]);
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              in_offset,
                              (PtrDtype)param.conv_param.weight()->data(),
                              0,
                              (PtrDtype)_outConvRelu->mutable_data(),
                              out_offset,
                              floatObjects[0]);
                } else {
                    if (isBias) {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)param.conv_param.bias()->data(),
                                  negative_slope,
                                  (PtrDtype)inputs[0]->data(),
                                  in_offset,
                                  (PtrDtype)param.conv_param.weight()->data(),
                                  0,
                                  (PtrDtype)_outConvRelu->mutable_data(),
                                  out_offset,
                                  floatObjects[0],
                                  floatObjects[1]);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  negative_slope,
                                  (PtrDtype)inputs[0]->data(),
                                  in_offset,
                                  (PtrDtype)param.conv_param.weight()->data(),
                                  0,
                                  (PtrDtype)_outConvRelu->mutable_data(),
                                  out_offset,
                                  floatObjects[0],
                                  floatObjects[1]);
                    }
                }

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i]);
                err = LaunchKernel(cm, list);

                if (!err) {
                    LOG(ERROR) << "Fail to set execution :" << err;
                    return SaberInvalidValue;
                }
            }
        } else if (_kernels_ptr[i].get()->GetName() == "Im2Col") {
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "GEMM Not 1x1";
            needBias = true;
            std::vector<AMDKernelPtr> v_temp;
            int data_size = (inputs[0]->num()) * (inputs[0]->channel())
                            * (inputs[0]->height()) * (inputs[0]->width());

            for (int j = 0; j < (inputs[0]->num()); j++) {
                out_offset = j * param.conv_param.weight()->num() * _outConvRelu->height()
                             * _outConvRelu->width();
                in_offset =
                    j * inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();
                i = 0;

                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (int)(data_size - in_offset),
                          (PtrDtype)inputs[0]->data(),
                          (int)in_offset,
                          (int)inputs[0]->height(),
                          (int)inputs[0]->width(),
                          (int)param.conv_param.weight()->height(),
                          (int)param.conv_param.weight()->width(),
                          (int)_outConvRelu->height(),
                          (int)_outConvRelu->width(),
                          (int)param.conv_param.pad_h,
                          (int)param.conv_param.pad_w,
                          (int)param.conv_param.stride_h,
                          (int)param.conv_param.stride_w,
                          (int)param.conv_param.dilation_h,
                          (int)param.conv_param.dilation_w,
                          (PtrDtype)_outGemmWorkspace->mutable_data());

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i++]);

                if (_kernels_ptr[i].get()->GetName() != "miog_betac_alphaab") {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)_outConvRelu->mutable_data(), out_offset, floatObjects[1]);

                    if (!err) {
                        LOG(ERROR) << "Fail to set kernel args :" << err;
                        return SaberInvalidValue;
                    }

                    list.push_back(_kernels_ptr[i++]);

                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)_outGemmWorkspace->mutable_data(),
                              0,
                              (PtrDtype)param.conv_param.weight()->data(),
                              0,
                              (PtrDtype)_outConvRelu->mutable_data(),
                              out_offset,
                              floatObjects[0]);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)_outGemmWorkspace->mutable_data(),
                              0,
                              (PtrDtype)param.conv_param.weight()->data(),
                              0,
                              (PtrDtype)_outConvRelu->mutable_data(),
                              out_offset,
                              floatObjects[0],
                              floatObjects[1]);
                }

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i]);
                err = LaunchKernel(cm, list);

                if (!err) {
                    LOG(ERROR) << "Fail to set execution :" << err;
                    return SaberInvalidValue;
                }
            }
        } else if (_kernels_ptr[i].get()->GetName() == "conv7x7c3h448w448k64u2v2p3q3f1b1prelupooling") {
            float paddingVal = 0.0f;
            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)param.conv_param.weight()->data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      paddingVal,
                      negative_slope,
                      (PtrDtype)param.conv_param.bias()->data());

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "conv7x7c3h448w448k64u2v2p3q3f1b0prelupooling") {
            float paddingVal = 0.0f;
            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)param.conv_param.weight()->data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      paddingVal,
                      negative_slope);

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "conv7x7c3h224w224k64u2v2p3q3f1b1prelu") {
            float paddingVal = 0.0f;
            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)param.conv_param.weight()->data(),
                      (PtrDtype)_outConvRelu->mutable_data(),
                      paddingVal,
                      negative_slope,
                      (PtrDtype)param.conv_param.bias()->data());

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "conv7x7c3h224w224k64u2v2p3q3f1b0prelu") {
            float paddingVal = 0.0f;
            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)param.conv_param.weight()->data(),
                      (PtrDtype)_outConvRelu->mutable_data(),
                      paddingVal,
                      negative_slope);

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "pooling_f3x3_s2x2") {
            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (PtrDtype)_outConvRelu->data(),
                      (PtrDtype)outputs[0]->mutable_data());

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
template class VenderConv2DPooling<AMD, AK_FLOAT>;
template class VenderConv2DPooling<AMD, AK_HALF>;
template class VenderConv2DPooling<AMD, AK_INT8>;

} // namespace saber
} // namespace anakin
