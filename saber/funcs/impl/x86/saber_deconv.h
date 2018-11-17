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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_DECONV_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_DECONV_H

#include "saber/funcs/impl/impl_deconv.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberDeconv2D<X86, OpDtype> :
        public ImplBase<X86, OpDtype, ConvParam<X86> > {
public:
    typedef ImplBase<X86, OpDtype, ConvParam<X86> > Impl_t;
    SaberDeconv2D() = default;

    ~SaberDeconv2D() {
        if (_impl) {
            delete _impl;
        }
    }

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvParam<X86>& param, Context<X86>& ctx);

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvParam<X86>& param, Context<X86> &ctx);

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvParam<X86>& param);

    SaberStatus trans_weights(Tensor<X86> &target_weights,
            Tensor<X86> &target_bias, int in_channel, int out_channel,
            int stride_h, int stride_w, int pad_h, int pad_w,
            int dilation_h, int dilation_w, int group);
private:

    Impl_t* _impl{nullptr};
};

} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_H
