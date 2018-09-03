/* Copyright (c) 2018 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CROP_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_CROP_H

#include "saber/funcs/impl/impl_power.h"
#include "saber/funcs/power.h"
namespace anakin {
namespace saber {

template <DataType OpDtype>
class SaberPower<X86, OpDtype> :
    public ImplBase<
        X86, OpDtype,
        PowerParam<X86> >
{
public:
    
    SaberPower()
    {}

    ~SaberPower() {
    }

    virtual SaberStatus init(const std::vector<Tensor<X86>*>& inputs,
                             std::vector<Tensor<X86>*>& outputs,
                             PowerParam<X86> &param,
                             Context<X86> &ctx) {
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    };

    virtual SaberStatus create(const std::vector<Tensor<X86>*>& inputs,
                               std::vector<Tensor<X86>*>& outputs,
                               PowerParam<X86> &param,
                               Context<X86> &ctx) {
        Shape shape({inputs[0]->dims(), 1, 1, 1});
        _in_steps.re_alloc(shape, OpDtype);
        _out_steps.re_alloc(shape, OpDtype);
        _out_valid_shape.re_alloc(shape, OpDtype);
        Shape in_stride = inputs[0]->get_stride();
        Shape out_stride = outputs[0]->get_stride();
        Shape out_valid_shape = outputs[0]->valid_shape();
        memcpy(_out_steps.data(), &out_stride[0], sizeof(int)*4);
        memcpy(_in_steps.data(), &in_stride[0], sizeof(int)*4);
        memcpy(_out_valid_shape.data(), &out_valid_shape[0], sizeof(int)*4);
        return SaberSuccess;
    };

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                                 std::vector<Tensor<X86>*>& outputs,
                                 PowerParam<X86> &param) override;
private:
    Tensor<X86> _in_steps;
    Tensor<X86> _out_steps;
    Tensor<X86> _out_valid_shape;

    
};

}
}
#endif
