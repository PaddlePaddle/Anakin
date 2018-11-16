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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_AFFINE_CHANNEL_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_AFFINE_CHANNEL_H

#include "saber/funcs/impl/impl_affine_channel.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberAffineChannel<X86, OpDtype>: public ImplBase<X86, OpDtype, AffineChannelParam<X86> > {

public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberAffineChannel() {}
    ~SaberAffineChannel() {}

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86> *>& outputs,
                             AffineChannelParam<X86> &param,
                             Context<X86> &ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86> *>& outputs,
                               AffineChannelParam<X86> &crop_param,
                               Context<X86> &ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<X86> *>& inputs,
                                 std::vector<Tensor<X86> *>& outputs,
                                 AffineChannelParam<X86> &param);

private:
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_AFFINE_CHANNEL_H
