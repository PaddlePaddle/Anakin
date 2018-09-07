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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_IM2SEQUENCE_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_IM2SEQUENCE_H

#include "saber/funcs/impl/impl_im2sequence.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberIm2Sequence<X86, OpDtype>:\
    public ImplBase<X86, OpDtype, Im2SequenceParam<X86> > {

public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberIm2Sequence() {}

    ~SaberIm2Sequence() {}

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                             std::vector<Tensor<X86> *>& outputs,
                             Im2SequenceParam<X86> &param,
                             Context<X86> &ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }
    
    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                               std::vector<Tensor<X86> *>& outputs,
                               Im2SequenceParam<X86> &param,
                               Context<X86> &ctx) {
        N = inputs[0]->num();
        C = inputs[0]->channel();
        H = inputs[0]->height();
        W = inputs[0]->width();
        //extern kernel height
        kernel_extern_h = param.dilation_h * (param.window_h - 1) + 1;
        output_height = (H + param.pad_up + param.pad_down - kernel_extern_h)
                            / param.stride_h + 1;

        //extern kernel width.
        kernel_extern_w = param.dilation_w * (param.window_w - 1) + 1;
        output_width = (W + param.pad_left + param.pad_right - kernel_extern_w)
                        / param.stride_w + 1;
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<X86> *>& inputs,
                                 std::vector<Tensor<X86> *>& outputs,
                                 Im2SequenceParam<X86> &param);

private:
    int N, C, H, W;
    int output_height;
    int output_width;
    int kernel_extern_h;
    int kernel_extern_w;
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_IM2SEQUENCE_H
