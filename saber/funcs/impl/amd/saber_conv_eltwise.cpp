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
#include "saber/funcs/impl/amd/include/saber_conv_eltwise.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberConvEltwise<AMD, AK_FLOAT>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvEltwiseParam<AMD>& param,
    Context<AMD>& ctx) {

    _ctx = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);
    _inner_tensor.reshape(_inner_shape);
    _inner_tensor_v.resize(2);
    _inner_tensor_v[0] = &_inner_tensor;
    _conv.create(inputs, _inner_tensor_v, param.conv_param, ctx);
    _eltwise.create(_inner_tensor_v, outputs, param.eltwise_param, ctx);
    return SaberSuccess;
}

template <>
SaberStatus SaberConvEltwise<AMD, AK_FLOAT>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvEltwiseParam<AMD>& param,
    Context<AMD>& ctx) {
    _ctx         = &ctx;
    _inner_shape = conv_compute_shape(inputs[0]->valid_shape(), param.conv_param);

    _inner_tensor.re_alloc(_inner_shape, AK_FLOAT);
    _inner_tensor_v.resize(2);
    _inner_tensor_v[0] = &_inner_tensor;
    _conv.init(inputs, _inner_tensor_v, param.conv_param, ctx);
    _eltwise.init(_inner_tensor_v, outputs, param.eltwise_param, ctx);
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberConvEltwise<AMD, AK_FLOAT>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    ConvEltwiseParam<AMD>& param) {
    _conv.dispatch(inputs, _inner_tensor_v, param.conv_param);
    _inner_tensor_v[1] = outputs[0];
    _eltwise.dispatch(_inner_tensor_v, outputs, param.eltwise_param);
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberConvEltwise, ConvEltwiseParam, AMD, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConvEltwise, ConvEltwiseParam, AMD, AK_INT8);
} // namespace saber
} // namespace anakin
