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
#include "saber/funcs/impl/amd/include/saber_conv_pooling.h"
#include "saber/funcs/conv.h"
#include "saber/funcs/impl/amd/include/amd_utils.h"
#include "saber/funcs/impl/amd/include/amd_gemm.h"

namespace anakin {
namespace saber {
typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;

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

    KernelInfo kernelInfo;

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

    bool isBias = (param.conv_param.bias()->size() > 0) ? true : false;

    int data_len;
    miopen::ConvolutionContext convContext;
    convContext.direction.Set(1);
#ifdef ENABLE_AMD_DO_SEARCH
    convContext.do_search        = true;
#else
    convContext.do_search        = false;
#endif
    convContext.general_compile_options += "";
    // context.SetStream(&profile_h);
    convContext.n_inputs         = inputs[0]->channel();
    convContext.in_height        = inputs[0]->height();
    convContext.in_width         = inputs[0]->width();
    convContext.kernel_size1     = param.conv_param.weight()->width();
    convContext.kernel_size0     = param.conv_param.weight()->height();
    convContext.n_outputs        = param.conv_param.weight()->num();
    convContext.out_height       = _outConvRelu->height();
    convContext.out_width        = _outConvRelu->width();
    convContext.batch_sz         = inputs[0]->num();
    convContext.pad0             = param.conv_param.pad_w;
    convContext.pad1             = param.conv_param.pad_h;
    convContext.kernel_stride0   = param.conv_param.stride_h;
    convContext.kernel_stride1   = param.conv_param.stride_w;
    convContext.kernel_dilation0 = param.conv_param.dilation_w;
    convContext.kernel_dilation1 = param.conv_param.dilation_h;
    convContext.bias             = isBias;
    convContext.float_size       = 32;
    convContext.in_layout        = "NCHW";
    convContext.in_data_type     = "FP32";
    convContext.save_srch_req    = true;
    convContext.use_asm_kernels  = true;
    convContext.use_binaries     = true;
    convContext.weights_layout   = "";
    convContext.out_data_type    = "FP32";
    convContext.out_layout       = "NCHW";
    data_len                     = convContext.in_data_type == "FP32" ? 4 : 2;
    convContext.bot_sz = convContext.batch_sz * convContext.n_inputs * convContext.in_height
                         * convContext.in_width * data_len;
    convContext.top_sz = convContext.batch_sz * convContext.n_outputs * convContext.out_height
                         * convContext.out_width * data_len;
    convContext.weights_sz = convContext.n_outputs * convContext.n_inputs * convContext.kernel_size0
                             * convContext.kernel_size1 * data_len;
    convContext.bias_sz                 = outputs[0]->channel();
    convContext.deconvolution           = 0;
    convContext.in_stride               = inputs[0]->get_stride()[2];
    convContext.out_stride              = _outConvRelu->get_stride()[2];
    convContext.in_channel_stride       = convContext.in_stride * convContext.in_height;
    convContext.in_batch_stride         = convContext.in_channel_stride * convContext.n_inputs;
    convContext.out_channel_stride      = convContext.out_stride * convContext.out_height;
    convContext.out_batch_stride        = convContext.out_channel_stride * convContext.n_outputs;
    convContext.rmv                     = rocm_meta_version::AMDHSA_1_0;
    convContext.general_compile_options = " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";

    convContext.has_active = param.conv_param.activation_param.has_active;

    convContext.has_pooling               = true;
    convContext.poolingContext.batch_sz   = _outConvRelu->num();
    convContext.poolingContext.n_inputs   = _outConvRelu->channel();
    convContext.poolingContext.in_height  = _outConvRelu->height();
    convContext.poolingContext.in_width   = _outConvRelu->width();
    convContext.poolingContext.n_outputs  = outputs[0]->channel();
    convContext.poolingContext.out_height = outputs[0]->height();
    convContext.poolingContext.out_width  = outputs[0]->width();

    switch (param.pooling_param.pooling_type) {
    case Pooling_max:
        convContext.poolingContext.pooling_type = (PoolingType)MLO_POOLING_OP_MAX;
        break;

    case Pooling_average_exclude_padding:
    case Pooling_average_include_padding:
        convContext.poolingContext.pooling_type = (PoolingType)MLO_POOLING_OP_AVE;
        break;

    case Pooling_unknow:
    case Pooling_max_deterministic:
    default:
        LOG(ERROR) << "Unknown polling type";
        return SaberInvalidValue;
    }

    convContext.poolingContext.pad1           = param.pooling_param.pad_h;
    convContext.poolingContext.pad0           = param.pooling_param.pad_w;
    convContext.poolingContext.kernel_size1   = param.pooling_param.window_h;
    convContext.poolingContext.kernel_size0   = param.pooling_param.window_w;
    convContext.poolingContext.kernel_stride1 = param.pooling_param.stride_h;
    convContext.poolingContext.kernel_stride0 = param.pooling_param.stride_w;

    miopen::Db db = anakin::saber::GetDb(dev._info._device_name, dev._info._compute_core_num);
    miopen::Handle::setClEnv(context, device);
    miopen::Handle handle /*(context, device)*/;
    convContext.SetStream(&handle);

    miopen::solver::ConvSolution solution = miopen::solver::SearchForSolution <
                                            miopen::solver::ConvBinWinograd3x3U,
                                            miopen::solver::ConvOclDirectFwd1x1AMD,
                                            // miopen::solver::ConvAsm3x3U,
                                            // miopen::solver::ConvAsm1x1U,
                                            miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
                                            miopen::solver::ConvOclDirectFwdGen,
                                            miopen::solver::ConvOclDirectFwd3x3,
                                            miopen::solver::ConvOclDirectFwd1x1,
                                            miopen::solver::ConvOclDirectFwd > (convContext, db);
    miopen::Handle::clearClEnv();

    if (solution.construction_params.size() > 0) {
        for (auto s : solution.construction_params) {
            kernelInfo = s; // assign MIOpen kernelInfo to Saber kernelInfo

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
        } else if (_kernels_ptr[i].get()->GetName() == "conv1x1_act_pool") {
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
                //i = 0;

                if (_kernels_ptr[i].get()->GetName() != "miog_betac_alphaab") {
                    err = _kernels_ptr[i].get()->SetKernelArgs(
                              _outConvRelu->mutable_data(), out_offset, 0.0f);

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
                          (size_t)in_offset,
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
                              (PtrDtype)_outConvRelu->mutable_data(), out_offset, 0.0f);

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
template class SaberConv2DPooling<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConv2DPooling, ConvPoolingParam, AMD, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConv2DPooling, ConvPoolingParam, AMD, AK_INT8);

} // namespace saber
} // namespace anakin
