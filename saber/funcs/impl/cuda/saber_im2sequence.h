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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_IM2SEQUENCE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_IM2SEQUENCE_H

#include "saber/funcs/impl/impl_im2sequence.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberIm2Sequence<NV, OpDtype>:\
    public ImplBase<NV, OpDtype, Im2SequenceParam<NV> > {

public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberIm2Sequence() {}

    ~SaberIm2Sequence() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                             std::vector<Tensor<NV> *>& outputs,
                             Im2SequenceParam<NV> &param,
                             Context<NV> &ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }
    
    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                               std::vector<Tensor<NV> *>& outputs,
                               Im2SequenceParam<NV> &param,
                               Context<NV> &ctx) {
        int input_height = inputs[0]->height(); // P
        _kernel_exten_h = param.dilation_h * (param.window_h - 1) + 1;
        _output_height = (input_height + param.pad_up + param.pad_down - _kernel_exten_h)
                         / param.stride_h + 1;

        int input_width = inputs[0]->width(); // Q
        _kernel_exten_w = param.dilation_w * (param.window_w - 1) + 1;
        _output_width = (input_width + param.pad_left + param.pad_right - _kernel_exten_w)
                        / param.stride_w + 1;

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                                 std::vector<Tensor<NV> *>& outputs,
                                 Im2SequenceParam<NV> &param);

private:
    int _output_height;
    int _output_width;
    int _kernel_exten_h;
    int _kernel_exten_w;
};

template class SaberIm2Sequence<NV, AK_FLOAT>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_IM2SEQUENCE_H
