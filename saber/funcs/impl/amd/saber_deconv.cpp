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

#include "saber/funcs/impl/amd/include/saber_deconv.h"
#include "saber/funcs/impl/amd/include/saber_depthwise_deconv.h"
#include "saber/funcs/impl/amd/include/saber_direct_deconv.h"
#include "saber/funcs/impl/amd/include/vender_deconv.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberDeconv2D<AMD, AK_FLOAT>::create(
    const std::vector<Tensor<AMD> *>& inputs,
    std::vector<Tensor<AMD> *>& outputs,
    ConvParam<AMD>& param, Context<AMD>& ctx) {

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "create";

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG)
            << "AMD Summary: input size N " << inputs[0]->num()
            << " C " << inputs[0]->channel()
            << " H " << inputs[0]->height()
            << " W " << inputs[0]->width();

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG)
            << "AMD Summary: op param K " << param.weight()->num()
            << " Y " << param.weight()->height() << " X " << param.weight()->width()
            << " SH " << param.stride_h << " SW " << param.stride_w
            << " PH " << param.pad_h << " PW " << param.pad_w
            << " DH " << param.dilation_h << " DW " << param.dilation_w
            << " Alpha " << param.alpha << " Beta " << param.beta << " GP " << param.group
            << " hasAct " << param.activation_param.has_active
            << " ActType " << param.activation_param.active
            << " slop " << param.activation_param.negative_slope
            << " coef " << param.activation_param.coef;

    _impl->create(inputs, outputs, param, ctx);
    return SaberSuccess;
}

template <>
SaberStatus SaberDeconv2D<AMD, AK_FLOAT>::init(
    const std::vector<Tensor<AMD> *>& inputs,
    std::vector<Tensor<AMD> *>& outputs,
    ConvParam<AMD>& param, Context<AMD>& ctx) {

    this->_ctx = &ctx;

    int in_channel = inputs[0]->channel();
    int out_channel = outputs[0]->channel();

    if (param.group == in_channel && param.group == out_channel) {
        _impl = new SaberDepthwiseDeconv<AMD, AK_FLOAT>;
    } else {
        //_impl = new SaberDirectDeconv<AMD, AK_FLOAT>;
        _impl = new VenderDeconv2D<AMD, AK_FLOAT>;
    }

    _impl->init(inputs, outputs, param, ctx);
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberDeconv2D<AMD, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<AMD> *>& inputs,
        std::vector<Tensor<AMD> *>& outputs,
        ConvParam<AMD>& param) {
    _impl->dispatch(inputs, outputs, param);
}

template <>
SaberStatus SaberDeconv2D<AMD, AK_FLOAT>::trans_weights(Tensor<AMD>& target_weights,
        Tensor<AMD>& target_bias,
        int in_channel, int out_channel,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dilation_h, int dilation_w,
        int group) {

    return SaberSuccess;
}

template <>
SaberStatus SaberDeconv2D<AMD, AK_HALF>::trans_weights(Tensor<AMD>& target_weights,
        Tensor<AMD>& target_bias,
        int in_channel, int out_channel,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dilation_h, int dilation_w,
        int group) {

    return SaberUnImplError;
}

template <>
SaberStatus SaberDeconv2D<AMD, AK_INT8>::trans_weights(Tensor<AMD>& target_weights,
        Tensor<AMD>& target_bias,
        int in_channel, int out_channel,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dilation_h, int dilation_w,
        int group) {
    return SaberUnImplError;
}

template class SaberDeconv2D<AMD, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberDeconv2D, ConvParam, AMD, AK_HALF);
DEFINE_OP_TEMPLATE(SaberDeconv2D, ConvParam, AMD, AK_INT8);

}
}
