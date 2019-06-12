/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
*/

#include "include/saber_scale.h"

namespace anakin {

namespace saber {

typedef TargetWrapper<AMD> AMD_API;

template <DataType OpDtype>
SaberStatus SaberScale<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ScaleParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    _axis      = (param.num_axes == 0) ? 0 : param.axis;
    _num_axes  = param.num_axes >= 0 ? param.num_axes : inputs[0]->shape().dims() - _axis;
    _bias_term = param.bias_term;

    if (param.scale_w.size() > 0) {
        _weight.re_alloc(Shape({param.scale_w.size(), 1, 1, 1}), OpDtype);
        AMD_API::sync_memcpy(
            _weight.mutable_data(),
            0,
            inputs[0]->device_id(),
            &param.scale_w[0],
            0,
            0,
            sizeof(OpDataType) * param.scale_w.size(),
            __HtoD());
    }

    if (param.bias_term) {
        _bias.re_alloc(Shape({param.scale_b.size(), 1, 1, 1}), OpDtype);
        AMD_API::sync_memcpy(
            _bias.mutable_data(),
            0,
            inputs[0]->device_id(),
            &param.scale_b[0],
            0,
            0,
            sizeof(OpDataType) * param.scale_w.size(),
            __HtoD());
    }

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberScale<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ScaleParam<AMD>& param,
    Context<AMD>& ctx) {
    const int count = outputs[0]->size();

    _inner_dim = inputs[0]->count(_axis + _num_axes, inputs[0]->shape().dims());
    _scale_dim = inputs[0]->count(_axis, _axis + _num_axes);

    if (inputs.size() == 1) {
        CHECK_EQ(_scale_dim, param.scale_w.size()) << "scale dim not valid";
    }

    if (count > 3 && count % 4 == 0) {
        KernelInfo kernelInfo;
        kernelInfo.kernel_file = "Scale.cl";
        kernelInfo.wk_dim      = 1;
        kernelInfo.kernel_type = SABER;
        kernelInfo.l_wk        = {AMD_NUM_THREADS};
        kernelInfo.g_wk        = {(count / 4  + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]* kernelInfo.l_wk[0]};

        kernelInfo.kernel_name   = "Scale_singleBias_float4";
        _kernel_Scale_singleBias = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!_kernel_Scale_singleBias.get()->isInit()) {
            LOG(ERROR) << "Failed to load _kernel_Scale_singleBias ";
            return SaberInvalidValue;
        }
    } else {
        KernelInfo kernelInfo;
        kernelInfo.kernel_file   = "Scale.cl";
        kernelInfo.wk_dim        = 1;
        kernelInfo.kernel_type   = SABER;
        kernelInfo.l_wk          = {AMD_NUM_THREADS};
        kernelInfo.g_wk          = {(count  + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0] * kernelInfo.l_wk[0]};
        kernelInfo.kernel_name   = "Scale_singleBias";
        _kernel_Scale_singleBias = CreateKernel(inputs[0]->device_id(), &kernelInfo);

        if (!_kernel_Scale_singleBias.get()->isInit()) {
            LOG(ERROR) << "Failed to load _kernel_Scale_singleBias ";
            return SaberInvalidValue;
        }
    }

    KernelInfo kernelInfo;
    kernelInfo.kernel_file  = "Scale.cl";
    kernelInfo.wk_dim       = 1;
    kernelInfo.kernel_type  = SABER;
    kernelInfo.l_wk         = {AMD_NUM_THREADS};
    kernelInfo.g_wk         = {(count + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0] * kernelInfo.l_wk[0]};
    kernelInfo.kernel_name  = "Scale_multiBias";
    kernelInfo.kernel_type  = SABER;
    _kernel_Scale_multiBias = CreateKernel(inputs[0]->device_id(), &kernelInfo);

    if (!_kernel_Scale_multiBias.get()->isInit()) {
        LOG(ERROR) << "Failed to load program _kernel_Scale_multiBias";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE CREATE KERNEL";
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberScale<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ScaleParam<AMD>& param) {
    bool err                  = false;
    AMD_API::stream_t cm      = this->_ctx->get_compute_stream();
    const OpDataType* in_data = (OpDataType*)inputs[0]->data();
    OpDataType* out_data      = (OpDataType*)outputs[0]->mutable_data();
    const int count           = inputs[0]->valid_size();
    int bias_flag;
    amd_kernel_list list;

    if (inputs.size() > 1) {
        _scale_dim = inputs[1]->valid_size();
        _inner_dim = count / _scale_dim;
    }

    if (_scale_dim > 1 || inputs.size() > 1) { //_kernel_Scale_multiBias
        OpDataType* scale_data = inputs.size() > 1 ? (OpDataType*)inputs[1]->data() : (OpDataType*)_weight.data();
        OpDataType* bias_data = param.bias_term ? (OpDataType*)_bias.data() : (OpDataType*)NULL;

        if (bias_data == NULL) {
            bias_flag = 0;
        } else {
            bias_flag = 1;
        }

        AMDKernel* kernel = _kernel_Scale_multiBias.get();
        err               = kernel->SetKernelArgs(
                                (PtrDtype)out_data,
                                (PtrDtype)in_data,
                                (PtrDtype)scale_data,
                                (PtrDtype)bias_data,
                                (int)count,
                                (int)_scale_dim,
                                (int)_inner_dim,
                                (int)bias_flag);

        if (!err) {
            LOG(ERROR) << "Fail to set _kernel_Scale_multiBias->SetKernelArgs";
            return SaberInvalidValue;
        }

        list.push_back(_kernel_Scale_multiBias);
    } else { //_kernel_Scale_siguleBias
        OpDataType scale = param.scale_w[0];
        OpDataType bias  = 0;

        if (_bias_term) {
            bias = param.scale_b[0];
        }

        AMDKernel* kernel = _kernel_Scale_singleBias.get();

        if (count > 3 && count % 4 == 0) {
            err = kernel->SetKernelArgs(
                      (PtrDtype)out_data,
                      (PtrDtype)in_data,
                      (OpDataType)scale,
                      (OpDataType)bias,
                      (int)count / 4);
        } else {
            err = kernel->SetKernelArgs(
                      (PtrDtype)out_data,
                      (PtrDtype)in_data,
                      (OpDataType)scale,
                      (OpDataType)bias,
                      (int)count);
        }

        if (!err) {
            LOG(ERROR) << "Fail to s _kernel_Scale_singleBias->SetKernelArgs";
            return SaberInvalidValue;
        }

        list.push_back(_kernel_Scale_singleBias);
    }

    err = LaunchKernel(cm, list);

    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "COMPLETE EXECUTION";

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberScale, ScaleParam, AMD, AK_HALF);

} // namespace saber
} // namespace anakin
