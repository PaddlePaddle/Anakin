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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SCALE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SCALE_H

#include "saber/funcs/impl/impl_scale.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberScale<NV, OpDtype>:
    public ImplBase<NV, OpDtype, ScaleParam<NV>> 
{

public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberScale()
    {}

    ~SaberScale() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            ScaleParam<NV>& param, Context<NV>& ctx) {
        this->_ctx = &ctx;
        _axis = (param.num_axes == 0) ? 0 : param.axis;
        _num_axes = param.num_axes >= 0 ? param.num_axes : inputs[0]->dims() - _axis;
        _bias_term = param.bias_term;
        if (param.scale_w.size() > 0) {   
            _weight.re_alloc(Shape({param.scale_w.size(), 1, 1, 1}), OpDtype);
            cudaMemcpy(_weight.mutable_data(), &param.scale_w[0], 
                    sizeof(OpDataType) * param.scale_w.size(), cudaMemcpyHostToDevice);
        }
        if (param.bias_term) {
            _bias.re_alloc(Shape({param.scale_b.size(), 1, 1, 1}), OpDtype);
            cudaMemcpy(_bias.mutable_data(), &param.scale_b[0], 
                    sizeof(OpDataType) * param.scale_w.size(), cudaMemcpyHostToDevice);
        }
        
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            ScaleParam<NV>& param, Context<NV> &ctx) {
        this->_ctx = &ctx;
        _inner_dim = inputs[0]->count_valid(_axis + _num_axes, inputs[0]->dims());
        _scale_dim = inputs[0]->count_valid(_axis, _axis + _num_axes);
        if (inputs.size() == 1) {
            CHECK_EQ(_scale_dim, param.scale_w.size()) << "scale dim not valid";
        }
        return SaberSuccess;
    }
    
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                          std::vector<Tensor<NV>*>& outputs,
                          ScaleParam<NV>& param);
private:
    int _axis;
    int _num_axes;
    bool _bias_term;
    int _inner_dim;
    int _scale_dim;
    Tensor<NV> _weight;
    Tensor<NV> _bias;
};

//template class SaberScale<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_SCALE_H
