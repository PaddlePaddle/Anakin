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

#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_DECONV_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_DECONV_H

#include "saber/funcs/impl/impl_deconv.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberDeconv2D<AMD, OpDtype> :
    public ImplBase<AMD, OpDtype, ConvParam<AMD> > {
public:
    typedef ImplBase<AMD, OpDtype, ConvParam<AMD> > Impl_t;
    SaberDeconv2D() = default;

    ~SaberDeconv2D() {
        if (_impl) {
            delete _impl;
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<AMD> *>& inputs,
                             std::vector<Tensor<AMD> *>& outputs,
                             ConvParam<AMD>& param, Context<AMD>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<AMD> *>& inputs,
                               std::vector<Tensor<AMD> *>& outputs,
                               ConvParam<AMD>& param, Context<AMD>& ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<AMD>*>& inputs,
                                 std::vector<Tensor<AMD>*>& outputs,
                                 ConvParam<AMD>& param);

    SaberStatus trans_weights(Tensor<AMD>& target_weights,
                              Tensor<AMD>& target_bias,
                              int in_channel, int out_channel,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w,
                              int dilation_h, int dilation_w,
                              int group);
private:
    Impl_t* _impl {nullptr};
};

} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_DECONV_H
