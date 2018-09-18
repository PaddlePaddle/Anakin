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

namespace anakin {
namespace saber {
typedef TargetWrapper<AMD> AMD_API;
typedef Env<AMD> AMD_ENV;

typedef struct KernelArguType {
    size_t size;
    void* ptr;
    bool isVal;
} T_KernelArgu;

template <>
SaberStatus SaberConv2DPooling<AMD, AK_FLOAT>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvPoolingParam<AMD>& param,
    Context<AMD>& ctx) {
    ALOGD("init");
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberConv2DPooling<AMD, AK_FLOAT>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);

    if (!kptr.get()->isInit()) {
        ALOGE("Failed to load program");
        return SaberInvalidValue;
    }

    _kernels.push_back(kptr);
}

template <>
SaberStatus SaberConv2DPooling<AMD, AK_FLOAT>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvPoolingParam<AMD>& param,
    Context<AMD>& ctx) {
    ALOGD("create");

    KernelInfo kernelInfo;

    _kernels.clear();

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
    convContext.do_search = true;
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
    convContext.bias_sz                 = 0;
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
        ALOGE("Unknown polling type");
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
                                            // miopen::solver::ConvAsm3x3U,
                                            // miopen::solver::ConvAsm1x1U,
                                            miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
                                            miopen::solver::ConvOclDirectFwdGen,
                                            miopen::solver::ConvOclDirectFwd3x3,
                                            miopen::solver::ConvOclDirectFwd1x1,
                                            miopen::solver::ConvOclDirectFwd > (convContext, db);
    miopen::Handle::clearClEnv();

    for (auto s : solution.construction_params) {
        kernelInfo = s; // assign MIOpen kernelInfo to Saber kernelInfo
        CreateKernelList(inputs[0]->device_id(), kernelInfo);
    }

    return SaberSuccess;
}

template <>
SaberStatus SaberConv2DPooling<AMD, AK_FLOAT>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvPoolingParam<AMD>& param) {
    ALOGD("dispatch");

    bool err;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    bool isBias = (param.conv_param.bias()->size() > 0) ? true : false;

    if (param.conv_param.activation_param.has_active == 0) {
        param.conv_param.activation_param.negative_slope = 1;
    }

    ALOGD(" num=" << inputs[0]->num() << " channel=" << inputs[0]->channel()
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
          << " param.has_active=" << param.conv_param.activation_param.has_active
          << " param.conv_param.activation_param.negative_slope="
          << param.conv_param.activation_param.negative_slope
          << " param.conv_param.activation_param.active="
          << param.conv_param.activation_param.active
          << " param.conv_param.activation_param.coef="
          << param.conv_param.activation_param.coef
          << " param.pooling_param.window_h=" << param.pooling_param.window_h
          << " param.pooling_param.window_w=" << param.pooling_param.window_w
          << " param.pooling_param.pad_h=" << param.pooling_param.pad_h
          << " param.pooling_param.pad_w=" << param.pooling_param.pad_w
          << " param.pooling_param.stride_h=" << param.pooling_param.stride_h
          << " param.pooling_param.stride_w=" << param.pooling_param.stride_w
          << " param.pooling_param.pooling_type=" << param.pooling_param.pooling_type
          << " param.pooling_param.global_pooling=" << param.pooling_param.global_pooling
          << " param.pooling_param.cmp_out_shape_floor_as_conv="
          << param.pooling_param.cmp_out_shape_floor_as_conv);

    if (isBias) {
        ALOGD("param.conv_param.bias()->size()=" << param.conv_param.bias()->size()
              << " param.conv_param.bias()->channel()=" << param.conv_param.bias()->channel()
              << " param.conv_param.bias()->width()=" << param.conv_param.bias()->width()
              << " param.conv_param.bias()->height()=" << param.conv_param.bias()->height());
    }

    for (amd_kernel_list::iterator it = _kernels.begin(); it != _kernels.end(); it++) {
        ALOGD("it->get()->GetName()=" << it->get()->GetName());

        if (it->get()->GetName() == "sp3AsmConv3x3F") {
            int d_n_groups = 64, d_flags = 0;
            PtrDtype biasMemObject = isBias ? param.conv_param.bias()->data() : 0;
            err                    = it->get()->SetKernelArgs(
                                         (unsigned int)inputs[0]->num(),
                                         (unsigned int)inputs[0]->channel(),
                                         (unsigned int)inputs[0]->height(),
                                         (unsigned int)inputs[0]->width(),
                                         (unsigned int)param.conv_param.weight()->num(),
                                         (unsigned int)d_n_groups,
                                         (unsigned int)d_flags,
                                         (float)param.conv_param.activation_param.negative_slope,
                                         (PtrDtype)inputs[0]->data(),
                                         (PtrDtype)param.conv_param.weight()->data(),
                                         (PtrDtype)_outConvRelu->mutable_data(),
                                         (PtrDtype)biasMemObject);

            if (!err) {
                ALOGE("Fail to set execution");
                return SaberInvalidValue;
            }
        } else if (it->get()->GetName() == "mloPooling") {
            err = it->get()->SetKernelArgs(
                      (PtrDtype)_outConvRelu->data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      (float)1.0f);

            if (!err) {
                ALOGE("Fail to set execution");
                return SaberInvalidValue;
            }

        } else {
        }
    }

    err = LaunchKernel(cm, _kernels);

    if (!err) {
        ALOGE("Fail to set execution");
        return SaberInvalidValue;
    }

    return SaberSuccess;
}
template class SaberConv2DPooling<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConv2DPooling, ConvPoolingParam, AMD, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConv2DPooling, ConvPoolingParam, AMD, AK_INT8);

} // namespace saber
} // namespace anakin
