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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_LAYER_NORM_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_LAYER_NORM_H

#include "saber/funcs/impl/impl_layer_norm.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberLayerNorm<NV, OpDtype>:public ImplBase<NV, OpDtype, LayerNormParam<NV> > {

public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberLayerNorm() = default;
    ~SaberLayerNorm() {}

    virtual SaberStatus init(const std::vector<Tensor<NV>* >& inputs,
                             std::vector<Tensor<NV>* >& outputs,
                             LayerNormParam<NV> &param,
                             Context<NV> &ctx) {
        // get context
        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV>* >& inputs,
                               std::vector<Tensor<NV>* >& outputs,
                               LayerNormParam<NV> &param,
                               Context<NV> &ctx) {
        //Shape sh_in = inputs[0]->valid_shape();
        _inner_size = inputs[0]->count_valid(param.axis, inputs[0]->dims());
        _outer_size = inputs[0]->count_valid(0, param.axis);

        Shape sh({0, 0, 0, 0});
        for (int i = 0; i < sh.dims(); ++i) {
            sh[i] = 1;
        }
        sh[0] = _outer_size;
        _mean.reshape(sh);
        _std.reshape(sh);

        if (param.scale_weights()->valid_size() == 0) {
            _flag_scale = false;
        } else {
            _flag_scale = true;
        }
        if (param.bias_weights()->valid_size() == 0) {
            _flag_bias = false;
        } else {
            _flag_bias = true;
        }

        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV>* >& inputs,
                                 std::vector<Tensor<NV>* >& outputs,
                                 LayerNormParam<NV> &param);


private:
    Tensor<NV> _mean;
    Tensor<NV> _std;
    int _inner_size;
    int _outer_size;
    bool _flag_scale{true};
    bool _flag_bias{true};
};
template class SaberLayerNorm<NV, AK_FLOAT>;
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_LAYER_NORM_H
