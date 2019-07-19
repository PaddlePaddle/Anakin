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
                    int slot_size = kernelInfo.tensile_slot_size == 0 ? 1 : kernelInfo.tensile_slot_size;
                    int l2_size = kernelInfo.tensile_l2_size == 0 ? 1 : kernelInfo.tensile_l2_size;
                    int dbg_size = kernelInfo.tensile_dbg_size == 0 ? 1 : kernelInfo.tensile_dbg_size;

                    _slot = new Tensor<AMD>();
                    _slot->re_alloc(
                        Shape({slot_size}, Layout_W), AK_FLOAT);

                    _l2 = new Tensor<AMD>();
                    _l2->re_alloc(
                        Shape({l2_size}, Layout_W), AK_FLOAT);

                    _dbg = new Tensor<AMD>();
                    _dbg->re_alloc(
                        Shape({dbg_size}, Layout_W), AK_FLOAT);

                    int out_channels = param.weight()->num();
                    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
                    Shape bias_s({1, out_channels, 1, 1}, Layout_NCHW);
                    _tensile_bias.re_alloc(bias_s, AK_FLOAT);
                    fill_tensor_const(_tensile_bias, 0.f, cm);
                } else if (param.group != 1 && kernelInfo.kernel_name == "sp3AsmConv3x3F") {
                    _subgroup_input = new Tensor<AMD>();
                    _subgroup_weight = new Tensor<AMD>();
                    _subgroup_bias = new Tensor<AMD>();
                    _subgroup_output = new Tensor<AMD>();

                    _subgroup_input->re_alloc(Shape({inputs[0]->num(),
                                                     inputs[0]->channel() / param.group,
                                                     inputs[0]->height(),
                                                     inputs[0]->width()
                                                    }, Layout_NCHW));

                    _subgroup_weight->re_alloc(Shape({param.weight()->num() / param.group,
                                                      param.weight()->channel() / param.group,
                                                      param.weight()->height(),
                                                      param.weight()->width()
                                                     }, Layout_NCHW));

                    _subgroup_bias->re_alloc(Shape({1, param.weight()->num() / param.group, 1, 1}, Layout_NCHW));

                    _subgroup_output->re_alloc(Shape({outputs[0]->num(),
                                                      outputs[0]->channel() / param.group,
                                                      outputs[0]->height(),
                                                      outputs[0]->width()
                                                     }, Layout_NCHW));
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

            if (param.group == 1) {
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
            } else {
                for (int g = 0; g < param.group; g++) {
                    // copy input buffer
                    int input_src_offset = g * _subgroup_input->count(1, 4) * sizeof(OpDataType);

                    for (int n = 0; n < inputs[0]->num(); n++) {
                        int input_dst_offset = n * _subgroup_input->count(1, 4) * sizeof(OpDataType);
                        AMD_API::async_memcpy(_subgroup_input->mutable_data(), input_dst_offset,
                                              _subgroup_input->device_id(),
                                              inputs[0]->data(), input_src_offset, inputs[0]->device_id(),
                                              _subgroup_input->count(1, 4) * sizeof(OpDataType), cm, __DtoD());
                        input_src_offset += inputs[0]->count(1, 4) * sizeof(OpDataType);
                    }

                    // copy weight buffer
                    int weight_offset = g * _subgroup_weight->size() * sizeof(OpDataType);
                    AMD_API::async_memcpy(_subgroup_weight->mutable_data(), 0, _subgroup_weight->device_id(),
                                          param.weight()->data(), weight_offset, param.weight()->device_id(),
                                          _subgroup_weight->size() * sizeof(OpDataType), cm, __DtoD());

                    // copy bias buffer
                    if (isBias) {
                        int bias_offset = g * _subgroup_bias->size() * sizeof(OpDataType);
                        AMD_API::async_memcpy(_subgroup_bias->mutable_data(), 0, _subgroup_bias->device_id(),
                                              param.bias()->data(), bias_offset, param.bias()->device_id(),
                                              _subgroup_bias->size() * sizeof(OpDataType), cm, __DtoD());
                    }

                    // setup winogard kernel
                    amd_kernel_list local_list;
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (unsigned int)_subgroup_input->num(),
                              (unsigned int)_subgroup_input->channel(),
                              (unsigned int)_subgroup_input->height(),
                              (unsigned int)_subgroup_input->width(),
                              (unsigned int)_subgroup_weight->num(),
                              (unsigned int)d_n_groups,
                              (unsigned int)d_flags,
                              negative_slope,
                              (PtrDtype)_subgroup_input->data(),
                              (PtrDtype)_subgroup_weight->data(),
                              (PtrDtype)_subgroup_output->mutable_data(),
                              (PtrDtype)_subgroup_bias->data());

                    if (!err) {
                        LOG(ERROR) << "Fail to set kernel args :" << err;
                        return SaberInvalidValue;
                    }

                    // launch winogard kernel
                    local_list.push_back(_kernels_ptr[i]);

                    if (local_list.size() > 0) {
                        err = LaunchKernel(cm, local_list);

                        if (!err) {
                            LOG(ERROR) << "Fail to set execution :" << err;
                            return SaberInvalidValue;
                        }
                    }

                    // copy output buffer
                    int output_dst_offset = g * _subgroup_output->count(1, 4) * sizeof(OpDataType);

                    for (int n = 0; n < inputs[0]->num(); n++) {
                        int output_src_offset = n * _subgroup_output->count(1, 4) * sizeof(OpDataType);
                        AMD_API::async_memcpy(outputs[0]->mutable_data(), output_dst_offset, outputs[0]->device_id(),
                                              _subgroup_output->data(), output_src_offset, _subgroup_output->device_id(),
                                              _subgroup_output->count(1, 4) * sizeof(OpDataType), cm, __DtoD());
                        output_dst_offset += outputs[0]->count(1, 4) * sizeof(OpDataType);
                    }
                }
            }
        }  else if (_kernels_ptr[i].get()->GetName() == "sp3AsmConvRxSU") {
            int d_n_groups = 64, d_flags = 0;
            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (unsigned int)inputs[0]->num(),
                      (unsigned int)inputs[0]->channel(),
                      (unsigned int)inputs[0]->height(),
                      (unsigned int)inputs[0]->width(),
                      (unsigned int)param.weight()->num(),
                      (unsigned int)d_n_groups,
                      (unsigned int)d_flags,
                      (unsigned int)0,
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)param.weight()->data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      (PtrDtype)nullptr,
                      (unsigned int)param.weight()->height(),
                      (unsigned int)param.weight()->width(),
                      (unsigned int)param.pad_h,
                      (unsigned int)param.pad_w,
                      (unsigned int)outputs[0]->height(),
                      (unsigned int)outputs[0]->width()
                  );

            if (!err) {
                LOG(ERROR) << "Fail to set kernel args :" << err;
                return SaberInvalidValue;
            }

            list.push_back(_kernels_ptr[i]);
        } else if (_kernels_ptr[i].get()->GetName() == "sp3AsmConvRxSU_CBA") {
            int d_n_groups = 64, d_flags = 0;

            if (isBias) {
                d_flags |= 128;
            }

            if (isActive) {
                d_flags |= 256;
            }

            err = _kernels_ptr[i].get()->SetKernelArgs(
                      (unsigned int)inputs[0]->num(),
                      (unsigned int)inputs[0]->channel(),
                      (unsigned int)inputs[0]->height(),
                      (unsigned int)inputs[0]->width(),
                      (unsigned int)param.weight()->num(),
                      (unsigned int)d_n_groups,
                      (unsigned int)d_flags,
                      (unsigned int)0,
                      (PtrDtype)inputs[0]->data(),
                      (PtrDtype)param.weight()->data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      (PtrDtype)nullptr,
                      (unsigned int)param.weight()->height(),
                      (unsigned int)param.weight()->width(),
                      (unsigned int)param.pad_h,
                      (unsigned int)param.pad_w,
                      (unsigned int)outputs[0]->height(),
                      (unsigned int)outputs[0]->width(),
                      (PtrDtype)param.bias()->data(),
                      negative_slope
                  );

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
                              (PtrDtype)outputs[0]->mutable_data(),
                              (PtrDtype)param.bias()->data(),
                              (PtrDtype)_slot->mutable_data(),
                              (PtrDtype)_l2->mutable_data(),
                              negative_slope,
                              (PtrDtype)_dbg->mutable_data());
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              (PtrDtype)param.bias()->data(),
                              (PtrDtype)_slot->mutable_data(),
                              (PtrDtype)_l2->mutable_data(),
                              1.0f,
                              (PtrDtype)_dbg->mutable_data());
                }
            } else {
                if (isActive) {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              (PtrDtype)_tensile_bias.data(),
                              (PtrDtype)_slot->mutable_data(),
                              (PtrDtype)_l2->mutable_data(),
                              negative_slope,
                              (PtrDtype)_dbg->mutable_data());
                } else {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              (PtrDtype)param.weight()->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              (PtrDtype)_tensile_bias.data(),
                              (PtrDtype)_slot->mutable_data(),
                              (PtrDtype)_l2->mutable_data(),
                              1.0f,
                              (PtrDtype)_dbg->mutable_data());
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
            if (i >= _kernels_ptr.size()) {
                return SaberInvalidValue;
            }
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
                if (i >= _kernels_ptr.size()) {
                    return SaberInvalidValue;
                }

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
            if (i >= _kernels_ptr.size()) {
                return SaberInvalidValue;
            }

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
                              outputs[0]->mutable_data(), out_offset, floatObjects[1]);

                    if (!err) {
                        LOG(ERROR) << "Fail to set kernel args :" << err;
                        return SaberInvalidValue;
                    }

                    list.push_back(_kernels_ptr[i++]);
                    if (i >= _kernels_ptr.size()) {
                        return SaberInvalidValue;
                    }
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              in_offset,
                              (PtrDtype)param.weight()->data(),
                              0,
                              (PtrDtype)outputs[0]->mutable_data(),
                              out_offset,
                              floatObjects[0]);
                } else {
                    if (inputs[0]->num() == 1) {
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
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
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

                list.clear();
            }
        } else if (_kernels_ptr[i].get()->GetName() == "Im2Col") {
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "GEMM Not 1x1";
            int data_size = (inputs[0]->num()) * (inputs[0]->channel())
                            * (inputs[0]->height()) * (inputs[0]->width());

            for (int j = 0; j < (inputs[0]->num()); j++) {
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

                for (int k = 0; k < param.group; k++) {
                    i = 1;
                    out_offset = j * param.weight()->num() * outputs[0]->height()
                                 * outputs[0]->width() + (k * ((param.weight()->num() / param.group) * outputs[0]->height()
                                                          * outputs[0]->width()));
                    unsigned int wei_offset = k * (inputs[0]->channel() / param.group) * param.weight()->channel() *
                                              param.weight()->height() * param.weight()->width();
                    unsigned int wksp_offset = k * (inputs[0]->channel() / param.group) * param.weight()->height() *
                                               param.weight()->width() * outputs[0]->height() * outputs[0]->width();

                    if (_kernels_ptr[i].get()->GetName() != "miog_betac_alphaab") {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)outputs[0]->mutable_data(), out_offset, floatObjects[1]);

                        if (!err) {
                            LOG(ERROR) << "Fail to set kernel args :" << err;
                            return SaberInvalidValue;
                        }

                        list.push_back(_kernels_ptr[i++]);
                        if (i >= _kernels_ptr.size()) {
                            return SaberInvalidValue;
                        }
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)_outGemmWorkspace->mutable_data(),
                                  wksp_offset,
                                  (PtrDtype)param.weight()->data(),
                                  wei_offset,
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  out_offset,
                                  floatObjects[0]);
                    } else {
                        err = _kernels_ptr[i].get()->SetKernelArgs(
                                  (PtrDtype)_outGemmWorkspace->mutable_data(),
                                  wksp_offset,
                                  (PtrDtype)param.weight()->data(),
                                  wei_offset,
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

                    list.clear();
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
