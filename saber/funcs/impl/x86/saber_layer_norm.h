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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LAYER_NORM_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LAYER_NORM_H

#include "saber/funcs/impl/impl_layer_norm.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberLayerNorm<X86, OpDtype>:public ImplBase<X86, OpDtype, LayerNormParam<X86> > {

public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;

    SaberLayerNorm() = default;
    ~SaberLayerNorm() {}

    virtual SaberStatus init(const std::vector<Tensor<X86>* >& inputs,
                             std::vector<Tensor<X86>* >& outputs,
                             LayerNormParam<X86> &param,
                             Context<X86> &ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<X86>* >& inputs,
                               std::vector<Tensor<X86>* >& outputs,
                               LayerNormParam<X86> &param,
                               Context<X86> &ctx) {

        inner_size = inputs[0]->count_valid(param.axis, inputs[0]->dims());
        outer_size = inputs[0]->count_valid(0, param.axis);

        if (param.scale_weights()->valid_size() == 0) {
            flag_scale = false;
        } else {
            flag_scale = true;
        }
        if (param.bias_weights()->valid_size() == 0) {
            flag_bias = false;
        } else {
            flag_bias = true;
        }
        
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<X86>* >& inputs,
                                 std::vector<Tensor<X86>* >& outputs,
                                 LayerNormParam<X86> &param);


private:
    int inner_size;
    int outer_size;
    bool flag_scale{true};
    bool flag_bias{true};
};

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_X86_SABER_LAYER_NORM_H
