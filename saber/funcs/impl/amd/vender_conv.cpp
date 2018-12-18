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
#include "saber/core/impl/amd/utils/amd_file_utils.h"
#include "saber/funcs/impl/amd/include/vender_conv.h"
#include "saber/funcs/impl/amd/include/amd_utils.h"
#include "saber/funcs/impl/amd/include/amd_gemm.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
void VenderConv2D<AMD, OpDtype>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return;
    }

    _kernels_ptr.push_back(kptr);
}

template <DataType OpDtype>
SaberStatus VenderConv2D<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvParam<AMD>& param,
    Context<AMD>& ctx) {
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "init";
    impl_vender = true;
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus VenderConv2D<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD> *>& inputs,
    std::vector<Tensor<AMD> *>& outputs,
    ConvParam<AMD>& param, Context<AMD>& ctx) {
    this->_ctx = &ctx;
    bool isBias = false;

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender) << "num=" << inputs[0]->num() << " channel=" <<
            inputs[0]->channel()
            << " height=" << inputs[0]->height() << " width=" << inputs[0]->width();

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender) << "stride_h=" << param.stride_h << " stride_w="
            <<
            param.stride_w << " height="
            << param.weight()->height() << " width=" << param.weight()->width()
            << " group=" << param.group << " dilation_h=" << param.dilation_h
            << " dilation_w=" << param.dilation_w << " pad_h=" << param.pad_h << " pad_w"
            << param.pad_w << " alpha=" << param.alpha << " beta=" << param.beta;

    if (param.activation_param.has_active) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender) << "activation.active=" <<
                param.activation_param.active;
    }

    if (param.bias()->valid_size() > 0) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender) << "bias size=" << param.bias()->size();
        isBias = true;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "VenderConv2D create kernel number:" << vkernel.size();
    KernelInfo kernelInfo;

    if (!vkernel.empty()) {
        for (int i = 0; i < vkernel.size(); i++) {
            kernelInfo = vkernel[i];

            if (kernelInfo.kernel_name == "xGemm") {
                _outGemmWorkspace = new Tensor<AMD>();
                std::vector<AMDKernelPtr> vkptr;
                vkptr.clear();
                bool needExtrakernel = false;

                _outGemmWorkspace->re_alloc(
                    Shape({(inputs[0]->num() * 2),
                           std::max({inputs[0]->channel(),
                                     param.weight()->channel(),
                                     param.weight()->num()
                                    }),
                           std::max((inputs[0]->height()), (outputs[0]->height())),
                           std::max((inputs[0]->width()), (outputs[0]->width()))
                          }));


                if (!findGenericGemm(true, vkptr, inputs, outputs[0], param,
                                     _outGemmWorkspace, ctx, needExtrakernel)) {
                    return SaberInvalidValue;
                }

                if (needExtrakernel) {
                    BiasReluPool(
                        vkptr,
                        inputs[0]->device_id(),
                        inputs[0]->num(),
                        param.weight()->num(),
                        0,
                        0,
                        0,
                        outputs[0]->height(),
                        outputs[0]->width(),
                        outputs[0]->channel(),
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        isBias,
                        param.activation_param.has_active);
                }

                for (int i = 0; i < vkptr.size(); i++) {
                    _kernels_ptr.push_back(vkptr[i]);
                }

                vkptr.clear();
            } else {
                if (kernelInfo.kernel_name == "conv7x7c3h224w224k64u2v2p3q3f1b1prelu"
                        || kernelInfo.kernel_name == "conv7x7c3h224w224k64u2v2p3q3f1b0prelu") {
                    kernelInfo.wk_dim      = 3;
                } else if (kernelInfo.kernel_name == "MIOpenGroupConvUni") {
                    kernelInfo.wk_dim      = 3;
                } else if (kernelInfo.kernel_name == "ConvFwd1x1") {
                    int slot_size = 1;
                    kernelInfo.kernel_type = MIOPEN;

                    if (inputs[0]->num() == 1) {
                        if (inputs[0]->height() == 7 || (inputs[0]->height() == 14 && inputs[0]->channel() == 1024)) {
                            slot_size = 1024;
                        } else if (inputs[0]->height() == 28 || (inputs[0]->height() == 14
                                   && inputs[0]->channel() == 256)) {
                            slot_size = 2048;
                        } else {
                            slot_size = 1664;

                            if (param.weight()->num() == 256) {
                                slot_size = 1600;
                            }
                        }
                    }

                    else if (inputs[0]->height() == 7 && inputs[0]->channel() == 2048 && param.weight()->num() == 512) {
                        slot_size = 32761;
                    } else if (inputs[0]->height() == 14 && inputs[0]->channel() == 1024 && inputs[0]->num() == 2
                               && param.weight()->num() == 256) {
                        slot_size = 32584;
                    }

                    _slot = new Tensor<AMD>();
                    _slot->re_alloc(
                        Shape({slot_size}, Layout_W));

                    int out_channels = param.weight()->num();
                    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
                    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);
                    _tensile_bias.re_alloc(bias_s, AK_FLOAT);
                    fill_tensor_const(_tensile_bias, 0.f, cm);
                }

                CreateKernelList(inputs[0]->device_id(), kernelInfo);
            }
        }
    } else {
        //gemm, not 1x1
        _outGemmWorkspace = new Tensor<AMD>();
        std::vector<AMDKernelPtr> vkptr;
        bool needExtrakernel = false;
        _outGemmWorkspace->re_alloc(
            Shape({(param.weight()->height() * param.weight()->width()),
                   std::max({inputs[0]->channel(),
                             param.weight()->channel(),
                             param.weight()->num()
                            }),
                   std::max((inputs[0]->height()), (outputs[0]->height())),
                   std::max((inputs[0]->width()), (outputs[0]->width()))
                  }));

        if (!findGenericGemm(false, vkptr,
                             inputs,
                             outputs[0],
                             param,
                             _outGemmWorkspace,
                             ctx, needExtrakernel)) {
            return SaberInvalidValue;
        }

        if (needExtrakernel) {
            BiasReluPool(
                vkptr,
                inputs[0]->device_id(),
                inputs[0]->num(),
                param.weight()->num(),
                0,
                0,
                0,
                outputs[0]->height(),
                outputs[0]->width(),
                outputs[0]->channel(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                isBias,
                param.activation_param.has_active);
        }

        for (int i = 0; i < vkptr.size(); i++) {
            _kernels_ptr.push_back(vkptr[i]);
        }

        vkptr.clear();
    }

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus VenderConv2D<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD> *>& inputs,
    std::vector<Tensor<AMD> *>& outputs,
    ConvParam<AMD>& param) {
    CHECK_EQ(inputs[0]->get_dtype(), AK_FLOAT);
    CHECK_EQ(outputs[0]->get_dtype(), AK_FLOAT);

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

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender) << " num=" << inputs[0]->num() << " channel=" <<
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
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender) << " param.bias()->size()=" <<
                param.bias()->size()
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

        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG & impl_vender) << " param.activation_param.has_active="
                << param.activation_param.has_active
                << " param.activation_param.negative_slope=" << param.activation_param.negative_slope
                << " param.activation_param.active=" << param.activation_param.active
                << " param.activation_param.coef=" << param.activation_param.coef;
    }

    if (_kernels_ptr[0] == NULL || _kernels_ptr[0].get() == NULL) {
        LOG(ERROR) << "Kernel is not exist";
        return SaberInvalidValue;
    }

    for (int i = 0; i < _kernels_ptr.size(); i++) {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "kernel size:" << _kernels_ptr.size() << " name:" <<
                                             _kernels_ptr[i].get()->GetName();

        if ((_kernels_ptr[i].get()->GetName() == "MIOpenConvUni")
                || (_kernels_ptr[i].get()->GetName() == "MIOpenConv1x1")
                || (_kernels_ptr[i].get()->GetName() == "MIOpenConv1x1pquv")
                || (_kernels_ptr[i].get()->GetName() == "MIOpenCvD3x3_WSR0")
                || (_kernels_ptr[i].get()->GetName() == "MIOpenCDFGen")
                || (_kernels_ptr[i].get()->GetName() == "MIOpenCDFGen4")
                || (_kernels_ptr[i].get()->GetName() == "MIOpenGroupConvUni")) {

            if (isBias) {
                if (isActive) {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)param.bias()->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              negative_slope,
                              0.0f);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)param.bias()->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              0.0f);
                }
            } else {
                if (isActive) {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              negative_slope,
                              0.0f);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
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

            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (unsigned int)inputs[0]->num(),
                      (unsigned int)inputs[0]->channel(),
                      (unsigned int)inputs[0]->height(),
                      (unsigned int)inputs[0]->width(),
                      (unsigned int)param.weight()->num(),
                      (unsigned int)d_n_groups,
                      (unsigned int)d_flags,
                      negative_slope,
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)param.weight()->data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      (PtrDtype)param.bias()->data());

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "ConvFwd1x1") {
            if (isBias) {
                if (isActive) {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)param.bias()->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              (PtrDtype)_slot->mutable_data(),
                              param.activation_param.negative_slope);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)param.bias()->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              (PtrDtype)_slot->mutable_data(),
                              1.0f);
                }
            } else {
                if (isActive) {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)_tensile_bias.data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              (PtrDtype)_slot->mutable_data(),
                              param.activation_param.negative_slope);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)_tensile_bias.data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              (PtrDtype)_slot->mutable_data(),
                              1.0f);
                }
            }

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "conv7x7c3h224w224k64u2v2p3q3f1b1prelu") {
            float paddingVal = 0.0f;
            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)param.weight()->data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      paddingVal,
                      negative_slope,
                      (PtrDtype)param.bias()->data());

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "conv7x7c3h224w224k64u2v2p3q3f1b0prelu") {
            float paddingVal = 0.0f;
            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)param.weight()->data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      paddingVal,
                      negative_slope);

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
                         * (outputs[0]->height()) * (outputs[0]->width());

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
                          (PtrDtype)param.weight()->data(),
                          0,
                          (PtrDtype)_outGemmWorkspace->mutable_data(),
                          out_offset,
                          floatObjects[0]);
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)_outGemmWorkspace->data(),
                          in_offset,
                          (PtrDtype)param.weight()->data(),
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
                          (PtrDtype)outputs[0]->mutable_data(),
                          (PtrDtype)param.bias()->data(),
                          negative_slope);
            } else {
                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (PtrDtype)_outGemmWorkspace->mutable_data(),
                          (PtrDtype)outputs[0]->mutable_data(),
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

            for (int j = 0; j < (inputs[0]->num()); j++) { //so far, only responsible for batch size is 1
                in_offset = j * (inputs[0]->channel()) * (inputs[0]->height())
                            * (inputs[0]->width());
                out_offset = j * (param.weight()->num()) * outputs[0]->height()
                             * outputs[0]->width();
                i = 0;

                if (_kernels_ptr[i].get()->GetName() != "miog_betac_alphaab") {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              outputs[0]->mutable_data(), out_offset, floatObjects[0]);

                    if (!err) {
                        LOG(ERROR) << "Fail to set kernel args :" << err;
                        return SaberInvalidValue;
                    }

                    list.push_back(_kernels_ptr[i++]);
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              in_offset,
                              (PtrDtype)param.weight()->data(),
                              0,
                              (PtrDtype)outputs[0]->mutable_data(),
                              out_offset,
                              floatObjects[0]);
                } else {
                    if (isBias) {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)param.bias()->data(),
                                  negative_slope,
                                  (PtrDtype)inputs[0]->data(),
                                  in_offset,
                                  (PtrDtype)param.weight()->data(),
                                  0,
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  out_offset,
                                  floatObjects[0],
                                  floatObjects[1]);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  negative_slope,
                                  (PtrDtype)inputs[0]->data(),
                                  in_offset,
                                  (PtrDtype)param.weight()->data(),
                                  0,
                                  (PtrDtype)outputs[0]->mutable_data(),
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
            int data_size = (inputs[0]->num()) * (inputs[0]->channel())
                            * (inputs[0]->height()) * (inputs[0]->width());

            for (int j = 0; j < (inputs[0]->num()); j++) {
                out_offset = j * param.weight()->num() * outputs[0]->height()
                             * outputs[0]->width();
                in_offset =
                    j * inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();
                i = 0;

                err = _kernels_ptr[i].get()->SetKernelArgs(
                          (int)(data_size - in_offset),
                          (PtrDtype)inputs[0]->data(),
                          (int)in_offset,
                          (int)inputs[0]->height(),
                          (int)inputs[0]->width(),
                          (int)param.weight()->height(),
                          (int)param.weight()->width(),
                          (int)outputs[0]->height(),
                          (int)outputs[0]->width(),
                          (int)param.pad_h,
                          (int)param.pad_w,
                          (int)param.stride_h,
                          (int)param.stride_w,
                          (int)param.dilation_h,
                          (int)param.dilation_w,
                          (PtrDtype)_outGemmWorkspace->mutable_data());

                if (!err) {
                    LOG(ERROR) << "Fail to set kernel args :" << err;
                    return SaberInvalidValue;
                }

                list.push_back(_kernels_ptr[i++]);

                if (_kernels_ptr[i].get()->GetName() != "miog_betac_alphaab") {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)outputs[0]->mutable_data(), out_offset, floatObjects[0]);

                    if (!err) {
                        LOG(ERROR) << "Fail to set kernel args :" << err;
                        return SaberInvalidValue;
                    }

                    list.push_back(_kernels_ptr[i++]);
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)_outGemmWorkspace->mutable_data(),
                              0,
                              (PtrDtype)param.weight()->data(),
                              0,
                              (PtrDtype)outputs[0]->mutable_data(),
                              out_offset,
                              floatObjects[0]);
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)_outGemmWorkspace->mutable_data(),
                              0,
                              (PtrDtype)param.weight()->data(),
                              0,
                              (PtrDtype)outputs[0]->mutable_data(),
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
        } else if (_kernels_ptr[i].get()->GetName() == "BiasReluBoth") {
            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (PtrDtype)outputs[0]->mutable_data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      (PtrDtype)param.bias()->data(),
                      negative_slope,
                      (inputs[0]->num()),
                      (outputs[0]->channel()),
                      (outputs[0]->height()),
                      (outputs[0]->width()),
                      1,
                      1);

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "BiasOnly") {
            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (PtrDtype)outputs[0]->mutable_data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      (PtrDtype)param.bias()->data(),
                      negative_slope,
                      (inputs[0]->num()),
                      (outputs[0]->channel()),
                      (outputs[0]->height()),
                      (outputs[0]->width()));

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "ReluUni") {
            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (PtrDtype)outputs[0]->mutable_data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      negative_slope);

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

template class VenderConv2D<AMD, AK_FLOAT>;
template class VenderConv2D<AMD, AK_HALF>;
template class VenderConv2D<AMD, AK_INT8>;
}
}
