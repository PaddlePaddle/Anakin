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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_H

#include "saber/funcs/impl/impl_deconv.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberDeconv2D<NV, OpDtype> : \
    public ImplBase<NV, OpDtype, ConvParam<NV>>
{
public:
typedef Tensor<NV> OpTensor;

SaberDeconv2D() :_use_k4_s2_p1(false) {}

~SaberDeconv2D() {}

virtual SaberStatus init(const std::vector<OpTensor *>& inputs,
                         std::vector<OpTensor *>& outputs,
                         ConvParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

virtual SaberStatus create(const std::vector<OpTensor *>& inputs,
                           std::vector<OpTensor *>& outputs,
                           ConvParam<NV>& param, Context<NV> &ctx) {
    _use_k4_s2_p1 = true;
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.weight()->width()==4);
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.weight()->height()==4);
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.stride_h==2);
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.stride_w==2);
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.pad_h==1);
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.pad_w==1);
    _use_k4_s2_p1 = _use_k4_s2_p1 && (param.group==1);
    if (_use_k4_s2_p1) {
        int in_channel = inputs[0]->channel();
        int out_channel = outputs[0]->channel();
        scale_to_new_tensor_k4_s2_p1_deconv<4>(param.mutable_weight(),
                                               in_channel, out_channel);
//            LOG(INFO)<<"scale weights finished!!";
    }
    return SaberSuccess;
}

virtual SaberStatus dispatch(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             ConvParam<NV>& param);
private:
bool _use_k4_s2_p1;
};


} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_H
