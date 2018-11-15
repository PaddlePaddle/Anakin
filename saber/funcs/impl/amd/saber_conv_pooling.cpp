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
                                            miopen::solver::ConvOclDirectFwd1x1AMD,
                                            // miopen::solver::ConvAsm3x3U,
                                            // miopen::solver::ConvAsm1x1U,
                                            // miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
                                            miopen::solver::ConvOclDirectFwdGen,
                                            miopen::solver::ConvOclDirectFwd3x3,
                                            miopen::solver::ConvOclDirectFwd1x1,
                                            miopen::solver::ConvOclDirectFwd > (convContext, db);
    miopen::Handle::clearClEnv();

    if (solution.construction_params.size() > 0) {
        for (auto s : solution.construction_params) {
            kernelInfo = s; // assign MIOpen kernelInfo to Saber kernelInfo
            CreateKernelList(inputs[0]->device_id(), kernelInfo);
        }
    } else {
        _is_gemm = true;
        AMDKernelPtr kptr;

        if (param.conv_param.weight()->width() == 1 && param.conv_param.weight()->height() == 1
                && param.conv_param.pad_w == 0 && param.conv_param.pad_h == 0
                && param.conv_param.dilation_w == 1 && param.conv_param.dilation_h == 1
                && param.conv_param.stride_w == 1 && param.conv_param.stride_h == 1) {
            ALOGD("GEMM 1x1");

            int K = (inputs[0]->channel()) * (param.conv_param.weight()->height())
                    * (param.conv_param.weight()->width());
            int M       = (param.conv_param.weight()->num());
            int N       = (_outConvRelu->height()) * (_outConvRelu->width());
            float alpha = 1.0;
            float beta  = 0.0;
            bool tA     = false;
            bool tB     = false;
            bool tC     = false;
            int lda     = K;
            int ldb     = N;
            int ldc     = N;

            MIOpenGEMM::Geometry tgg {};
            tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');
            AMD_API::stream_t cm = this->_ctx->get_compute_stream();

            /////////////////////////////////////////////////////////////
            // gemm kernel
            // jn : print search results to terminal
            bool miopengemm_verbose = false;

            // jn : print warning messages when the returned kernel(s) might be sub-optimal
            bool miopengemm_warnings = false;

            // jn : find with no workspace
            MIOpenGEMM::Solution soln = MIOpenGEMM::find(
                                            0.003f,
                                            cm,
                                            (PtrDtype)inputs[0]->data(),
                                            (PtrDtype)param.conv_param.weight()->data(),
                                            (PtrDtype)_outConvRelu->mutable_data(),
                                            false,
                                            tgg,
                                            miopengemm_verbose,
                                            miopengemm_warnings);

            std::string kernel_clstring;
            size_t local_work_size;
            size_t global_work_size;
            int errCode;

            int i = 0;

            if (soln.v_tgks.size() == 2) {
                _multikernel = true;

                // jn : the main kernel is at the back of the solution vector
                kernel_clstring = soln.v_tgks[i].kernstr;
                tempfix::set_offsets_to_uint(kernel_clstring, 1);

                kernelInfo.kernel_name = soln.v_tgks[i].fname;
                local_work_size        = soln.v_tgks[i].local_work_size;
                global_work_size       = soln.v_tgks[i].global_work_size;

                kernelInfo.kernel_file = kernel_clstring;
                kernelInfo.l_wk        = {local_work_size, 1, 1};
                kernelInfo.g_wk        = {global_work_size, 1, 1};
                kernelInfo.kernel_type = SOURCE;

                kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (!kptr.get()->isInit()) {
                    ALOGE("Failed to create kernel");
                    return SaberInvalidValue;
                }

                _kernels.push_back(kptr);

                i++;
            }

            // jn : the main kernel is at the back of the solution vector
            kernel_clstring = soln.v_tgks[i].kernstr;
            tempfix::set_offsets_to_uint(kernel_clstring, 3);

            kernelInfo.kernel_name = soln.v_tgks[i].fname;
            local_work_size        = soln.v_tgks[i].local_work_size;
            global_work_size       = soln.v_tgks[i].global_work_size;

            kernelInfo.kernel_file = kernel_clstring;
            kernelInfo.l_wk        = {local_work_size, 1, 1};
            kernelInfo.g_wk        = {global_work_size, 1, 1};
            kernelInfo.kernel_type = SOURCE;

            // To create the program
            kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                ALOGE("Failed to create kernel");
                return SaberInvalidValue;
            }

            _kernels.push_back(kptr);
        } else { // not 1x1
            ALOGD("Not 1x1");
            _outGemmWorkspace = new Tensor<AMD>();
            _outGemmWorkspace->re_alloc(Shape({
                (param.conv_param.weight()->height() * param.conv_param.weight()->width()),
                std::max({
                    inputs[0]->channel(),
                    param.conv_param.weight()->channel(),
                    param.conv_param.weight()->num()
                }),
                std::max((inputs[0]->height()), (_outConvRelu->height())),
                std::max((inputs[0]->width()), (_outConvRelu->width()))
            }));

            int K = (inputs[0]->channel()) * (param.conv_param.weight()->height())
                    * (param.conv_param.weight()->width());
            int M       = (param.conv_param.weight()->num());
            int N       = (_outConvRelu->height()) * (_outConvRelu->width());
            float alpha = 1.0;
            float beta  = 0.0;
            bool tA     = false;
            bool tB     = false;
            bool tC     = false;
            int lda     = K;
            int ldb     = N;
            int ldc     = N;

            MIOpenGEMM::Geometry tgg {};
            tgg = MIOpenGEMM::Geometry(true, tB, tA, tC, ldb, lda, ldc, N, M, K, 0, 'f');

            AMD_API::stream_t cm = this->_ctx->get_compute_stream();

            /////////////////////////////////////////////////////////////
            // gemm kernel
            // jn : print search results to terminal
            bool miopengemm_verbose = false;

            // jn : print warning messages when the returned kernel(s) might be sub-optimal
            bool miopengemm_warnings = false;

            Im2ColGPU(
                kernelInfo,
                kptr,
                inputs[0]->device_id(),
                inputs[0]->channel(),
                inputs[0]->height(),
                inputs[0]->width(),
                param.conv_param.weight()->height(),
                param.conv_param.weight()->width(),
                _outConvRelu->height(),
                _outConvRelu->width(),
                param.conv_param.pad_h,
                param.conv_param.pad_w,
                param.conv_param.stride_h,
                param.conv_param.stride_w,
                param.conv_param.dilation_h,
                param.conv_param.dilation_w);

            if (!kptr.get()->isInit()) {
                ALOGE("Failed to create kernel");
                return SaberInvalidValue;
            }

            _kernels.push_back(kptr);

            // jn : find with no workspace
            MIOpenGEMM::Solution soln = MIOpenGEMM::find(
                                            0.003f,
                                            cm,
                                            (PtrDtype)inputs[0]->data(),
                                            (PtrDtype)param.conv_param.weight()->data(),
                                            (PtrDtype)_outConvRelu->mutable_data(),
                                            false,
                                            tgg,
                                            miopengemm_verbose,
                                            miopengemm_warnings);

            std::string kernel_clstring;
            size_t local_work_size;
            size_t global_work_size;
            int errCode;

            int i                   = 0;
            kernelInfo.comp_options = "";

            if (soln.v_tgks.size() == 2) {
                _multikernel = true;

                // jn : the main kernel is at the back of the solution vector
                kernel_clstring = soln.v_tgks[i].kernstr;
                tempfix::set_offsets_to_uint(kernel_clstring, 1);

                kernelInfo.kernel_name = soln.v_tgks[i].fname;
                local_work_size        = soln.v_tgks[i].local_work_size;
                global_work_size       = soln.v_tgks[i].global_work_size;

                kernelInfo.kernel_file = kernel_clstring;
                kernelInfo.l_wk        = {local_work_size, 1, 1};
                kernelInfo.g_wk        = {global_work_size, 1, 1};
                kernelInfo.kernel_type = SOURCE;

                kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

                if (!kptr.get()->isInit()) {
                    ALOGE("Failed to create kernel");
                    return SaberInvalidValue;
                }

                _kernels.push_back(kptr);

                i++;
            }

            // jn : the main kernel is at the back of the solution vector
            kernel_clstring = soln.v_tgks[i].kernstr;
            tempfix::set_offsets_to_uint(kernel_clstring, 3);

            kernelInfo.kernel_name = soln.v_tgks[i].fname;
            local_work_size        = soln.v_tgks[i].local_work_size;
            global_work_size       = soln.v_tgks[i].global_work_size;

            kernelInfo.kernel_file = kernel_clstring;
            kernelInfo.l_wk        = {local_work_size, 1, 1};
            kernelInfo.g_wk        = {global_work_size, 1, 1};
            kernelInfo.kernel_type = SOURCE;

            // To create the program
            kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);

            if (!kptr.get()->isInit()) {
                ALOGE("Failed to create kernel");
                return SaberInvalidValue;
            }

            _kernels.push_back(kptr);
        }

        // bias+ relu + pooling kernel

        std::vector<AMDKernelPtr> vkptr;
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
            if (!vkptr[i].get()->isInit()) {
                ALOGE("Failed to create kernel");
                return SaberInvalidValue;
            }

            _kernels.push_back(vkptr[i]);
        }
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

    bool isBias   = (param.conv_param.bias()->size() > 0) ? true : false;
    bool isActive = param.conv_param.activation_param.has_active;

    if (param.conv_param.activation_param.has_active == 0) {
        param.conv_param.activation_param.negative_slope = 1;
    }

    ALOGD(" num=" << inputs[0]->num());
    ALOGD(" channel=" << inputs[0]->channel());
    ALOGD(" height=" << inputs[0]->height());
    ALOGD(" width=" << inputs[0]->width());
    ALOGD(" param.conv_param.weight()->num()=" << param.conv_param.weight()->num());
    ALOGD(" param.conv_param.weight()->channel()=" << param.conv_param.weight()->channel());
    ALOGD(" param.conv_param.weight()->width()=" << param.conv_param.weight()->width());
    ALOGD(" param.conv_param.weight()->height()=" << param.conv_param.weight()->height());

    if (isBias) {
        ALOGD(" param.conv_param.bias()->size()=" << param.conv_param.bias()->size());
        ALOGD(" param.conv_param.bias()->channel()=" << param.conv_param.bias()->channel());
        ALOGD(" param.conv_param.bias()->width()=" << param.conv_param.bias()->width());
        ALOGD(" param.conv_param.bias()->height()=" << param.conv_param.bias()->height());
    } else {
        ALOGD(" Bias is disable");
    }

    ALOGD(" param.conv_param.group=" << param.conv_param.group);
    ALOGD(" param.conv_param.pad_h=" << param.conv_param.pad_h);
    ALOGD(" param.conv_param.pad_w=" << param.conv_param.pad_w);
    ALOGD(" param.conv_param.stride_h=" << param.conv_param.stride_h);
    ALOGD(" param.conv_param.stride_w=" << param.conv_param.stride_w);
    ALOGD(" param.conv_param.dilation_h=" << param.conv_param.dilation_h);
    ALOGD(" param.conv_param.dilation_w=" << param.conv_param.dilation_w);
    ALOGD(" param.conv_param.alpha=" << param.conv_param.alpha);
    ALOGD(" param.conv_param.beta=" << param.conv_param.beta);
    ALOGD(" param.has_active=" << param.conv_param.activation_param.has_active);
    ALOGD(" param.conv_param.activation_param.negative_slope="
          << param.conv_param.activation_param.negative_slope);
    ALOGD(" param.conv_param.activation_param.active=" << param.conv_param.activation_param.active);
    ALOGD(" param.conv_param.activation_param.coef=" << param.conv_param.activation_param.coef);
    ALOGD(" param.pooling_param.window_h=" << param.pooling_param.window_h);
    ALOGD(" param.pooling_param.window_w=" << param.pooling_param.window_w);
    ALOGD(" param.pooling_param.pad_h=" << param.pooling_param.pad_h);
    ALOGD(" param.pooling_param.pad_w=" << param.pooling_param.pad_w);
    ALOGD(" param.pooling_param.stride_h=" << param.pooling_param.stride_h);
    ALOGD(" param.pooling_param.stride_w=" << param.pooling_param.stride_w);
    ALOGD(" param.pooling_param.pooling_type=" << param.pooling_param.pooling_type);
    ALOGD(" param.pooling_param.global_pooling=" << param.pooling_param.global_pooling);
    ALOGD(" param.pooling_param.cmp_out_shape_floor_as_conv="
          << param.pooling_param.cmp_out_shape_floor_as_conv);

    if (_is_gemm) {
        unsigned int out_offset = 0;
        unsigned int in_offset  = 0;
        float floatObjects[2]   = {1.0f, 0.0f};
        amd_kernel_list list;
        std::vector<AMDKernelPtr> _kernels_ptr;
        amd_kernel_list::iterator it = _kernels.begin();

        if (param.conv_param.weight()->width() == 1 && param.conv_param.weight()->height() == 1
                && param.conv_param.pad_w == 0 && param.conv_param.pad_h == 0
                && param.conv_param.dilation_w == 1 && param.conv_param.dilation_h == 1
                && param.conv_param.stride_w == 1 && param.conv_param.stride_h == 1) {
            ALOGD("GEMM 1x1");

            for (int j = 0; j < (inputs[0]->num()); j++) {
                in_offset =
                    j * (inputs[0]->channel()) * (inputs[0]->height()) * (inputs[0]->width());
                out_offset = j * (param.conv_param.weight()->num()) * _outConvRelu->height()
                             * _outConvRelu->width();
                it = _kernels.begin();

                if (_multikernel) {
                    err = it->get()->SetKernelArgs(_outConvRelu->mutable_data(), out_offset, 0.0f);

                    if (!err) {
                        ALOGE("Fail to set kernel args :" << err);
                        return SaberInvalidValue;
                    }

                    list.push_back(*(it++));

                    err = it->get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              in_offset,
                              (PtrDtype)param.conv_param.weight()->data(),
                              0,
                              (PtrDtype)_outConvRelu->mutable_data(),
                              out_offset,
                              floatObjects[0]);

                    if (!err) {
                        ALOGE("Fail to set kernel args :" << err);
                        return SaberInvalidValue;
                    }

                    list.push_back(*(it++));
                } else {
                    err = it->get()->SetKernelArgs(
                              (PtrDtype)inputs[0]->data(),
                              in_offset,
                              (PtrDtype)param.conv_param.weight()->data(),
                              0,
                              (PtrDtype)_outConvRelu->mutable_data(),
                              out_offset,
                              floatObjects[0],
                              floatObjects[1]);

                    if (!err) {
                        ALOGE("Fail to set kernel args :" << err);
                        return SaberInvalidValue;
                    }

                    list.push_back(*(it++));
                }

                err = LaunchKernel(cm, list, true);

                if (!err) {
                    ALOGE("Fail to set execution :" << err);
                    return SaberInvalidValue;
                }
            }
        } else {
            ALOGD("GEMM Not 1x1");
            int data_size = (inputs[0]->num()) * (inputs[0]->channel()) * (inputs[0]->height())
                            * (inputs[0]->width());

            for (int j = 0; j < (inputs[0]->num()); j++) {
                out_offset = j * param.conv_param.weight()->num() * _outConvRelu->height()
                             * _outConvRelu->width();
                in_offset = j * inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();
                it        = _kernels.begin();
                err       = it->get()->SetKernelArgs(
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
                    ALOGE("Fail to set kernel args :" << err);
                    return SaberInvalidValue;
                }

                list.push_back(*(it++));

                if (_multikernel) {
                    err = it->get()->SetKernelArgs(
                              (PtrDtype)_outConvRelu->mutable_data(), out_offset, 0.0f);

                    if (!err) {
                        ALOGE("Fail to set kernel args :" << err);
                        return SaberInvalidValue;
                    }

                    list.push_back(*(it++));
                    err = it->get()->SetKernelArgs(
                              (PtrDtype)_outGemmWorkspace->mutable_data(),
                              0,
                              (PtrDtype)param.conv_param.weight()->data(),
                              0,
                              (PtrDtype)_outConvRelu->mutable_data(),
                              out_offset,
                              floatObjects[0]);
                } else {
                    err = it->get()->SetKernelArgs(
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
                    ALOGE("Fail to set kernel args :" << err);
                    return SaberInvalidValue;
                }

                list.push_back(*(it++));
                err = LaunchKernel(cm, list);

                if (!err) {
                    ALOGE("Fail to set execution :" << err);
                    return SaberInvalidValue;
                }
            }
        }

        float slope = 1.0f;

        if (isActive) {
            slope = param.conv_param.activation_param.negative_slope;
        }

        for (; it != _kernels.end(); it++) {
            if (it->get()->GetName() == "mloPooling") {
                if (isBias) {
                    err = it->get()->SetKernelArgs(
                              (PtrDtype)_outConvRelu->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              (PtrDtype)param.conv_param.bias()->data(),
                              slope);
                } else {
                    err = it->get()->SetKernelArgs(
                              (PtrDtype)_outConvRelu->data(),
                              (PtrDtype)outputs[0]->mutable_data(),
                              slope);
                }
            } else if (it->get()->GetName() == "MIOpenBiasReluBoth") {
                err = it->get()->SetKernelArgs(
                          (PtrDtype)_outConvRelu->data(),
                          (PtrDtype)_outConvRelu->mutable_data(),
                          (PtrDtype)param.conv_param.bias()->data(),
                          slope,
                          (inputs[0]->num()),
                          (_outConvRelu->channel()),
                          (_outConvRelu->height()),
                          (_outConvRelu->width()),
                          1,
                          1);
            } else if (it->get()->GetName() == "MIOpenBias") {
                err = it->get()->SetKernelArgs(
                          (PtrDtype)_outConvRelu->data(),
                          (PtrDtype)_outConvRelu->mutable_data(),
                          (PtrDtype)param.conv_param.bias()->data(),
                          slope,
                          (inputs[0]->num()),
                          (_outConvRelu->channel()),
                          (_outConvRelu->height()),
                          (_outConvRelu->width()));
            } else if (it->get()->GetName() == "ReluUni") {
                err = it->get()->SetKernelArgs(
                          (PtrDtype)_outConvRelu->data(),
                          (PtrDtype)_outConvRelu->mutable_data(),
                          slope);
            } else if (it->get()->GetName() == "mloPoolingG") {
                if (isBias) {
                    if (isActive) {
                        err = it->get()->SetKernelArgs(
                                  (PtrDtype)_outConvRelu->data(),
                                  (PtrDtype)param.conv_param.bias()->data(),
                                  slope,
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  (PtrDtype) nullptr);
                    } else {
                        err = it->get()->SetKernelArgs(
                                  (PtrDtype)_outConvRelu->data(),
                                  (PtrDtype)param.conv_param.bias()->data(),
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  (PtrDtype) nullptr);
                    }
                }
                else
                {
                    if (isActive) {
                        err = it->get()->SetKernelArgs(
                                  (PtrDtype)_outConvRelu->data(),
                                  slope,
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  (PtrDtype) nullptr);
                    } else {
                        err = it->get()->SetKernelArgs(
                                  (PtrDtype)_outConvRelu->data(),
                                  (PtrDtype)outputs[0]->mutable_data(),
                                  (PtrDtype) nullptr);
                    }
                }
            } else {
                ALOGE("not handle kernel:" << it->get()->GetName());
                return SaberInvalidValue;
            }

            if (!err) {
                ALOGE("Fail to set kernel args :" << err);
                return SaberInvalidValue;
            }

            list.push_back(*it);
        }

        if (list.size() > 0) {
            err = LaunchKernel(cm, list);

            if (!err) {
                ALOGE("Fail to set execution :" << err);
                return SaberInvalidValue;
            }
        }

        return SaberSuccess;
    }

    for (amd_kernel_list::iterator it = _kernels.begin(); it != _kernels.end(); it++) {
        ALOGD("it->get()->GetName()=" << it->get()->GetName());

        if ((it->get()->GetName() == "MIOpenConvUni") || (it->get()->GetName() == "MIOpenConv1x1")
                || (it->get()->GetName() == "MIOpenConv1x1pquv")
                || (it->get()->GetName() == "MIOpenCvD3x3_WSR0")
                || (it->get()->GetName() == "MIOpenCDFGen")
                || (it->get()->GetName() == "MIOpenCDFGen4")) {
            PtrDtype memObjects[4] = {0, 0, 0, 0};
            memObjects[0]          = (PtrDtype)inputs[0]->data();
            memObjects[1]          = (PtrDtype)param.conv_param.weight()->data();
            memObjects[2]          = (isBias) ? (PtrDtype)param.conv_param.bias()->data() : nullptr;
            memObjects[3]          = (PtrDtype)_outConvRelu->mutable_data();

            if (isBias) {
                if (isActive) {
                    err = it->get()->SetKernelArgs(
                              (PtrDtype)memObjects[0],
                              (PtrDtype)memObjects[1],
                              (PtrDtype)memObjects[2],
                              (PtrDtype)memObjects[3],
                              param.conv_param.activation_param.negative_slope,
                              0.0f);
                } else {
                    err = it->get()->SetKernelArgs(
                              (PtrDtype)memObjects[0],
                              (PtrDtype)memObjects[1],
                              (PtrDtype)memObjects[2],
                              (PtrDtype)memObjects[3],
                              0.0f);
                }
            } else {
                if (isActive) {
                    err = it->get()->SetKernelArgs(
                              (PtrDtype)memObjects[0],
                              (PtrDtype)memObjects[1],
                              (PtrDtype)memObjects[3],
                              param.conv_param.activation_param.negative_slope,
                              0.0f);
                } else {
                    err = it->get()->SetKernelArgs(
                              (PtrDtype)memObjects[0],
                              (PtrDtype)memObjects[1],
                              (PtrDtype)memObjects[3],
                              0.0f);
                }
            }

            if (!err) {
                ALOGE("Fail to set kernel args :" << err);
                return SaberInvalidValue;
            }
        } else if (it->get()->GetName() == "sp3AsmConv3x3F") {
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
        } else if (it->get()->GetName() == "conv1x1_act_pool") {
            float negative_slope = 0.0f;

            if (isActive) {
                negative_slope = 0.0f;
            } else {
                negative_slope = 1.0f;
            }

            if (isBias) {
                err = it->get()->SetKernelArgs(
                          (PtrDtype)param.conv_param.weight()->data(),
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)(isBias) ? (PtrDtype)param.conv_param.bias()->data() : nullptr,
                          (PtrDtype)outputs[0]->mutable_data(),
                          negative_slope);
            } else {
                err = it->get()->SetKernelArgs(
                          (PtrDtype)param.conv_param.weight()->data(),
                          (PtrDtype)inputs[0]->data(),
                          (PtrDtype)outputs[0]->mutable_data(),
                          negative_slope);
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

        } else if (it->get()->GetName() == "mloPoolingG") {
            err = it->get()->SetKernelArgs(
                      (PtrDtype)_outConvRelu->data(),
                      (PtrDtype)outputs[0]->mutable_data(),
                      (PtrDtype) nullptr);

            if (!err) {
                ALOGE("Fail to set execution");
                return SaberInvalidValue;
            }

        } else {
            ALOGD("disptach non-implementation kernel: " << it->get()->GetName());
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
