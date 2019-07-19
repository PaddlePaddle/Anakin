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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_REDUCE_MIN_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_REDUCE_MIN_H

#include "saber/funcs/impl/impl_reduce_min.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberReduceMin<X86, OpDtype> :
    public ImplBase<
        X86, OpDtype,
        ReduceMinParam<X86> > {
public:
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
    SaberReduceMin() {}
    ~SaberReduceMin() {}

    virtual SaberStatus init(const std::vector<Tensor<X86> *>& inputs,
                            std::vector<Tensor<X86> *>& outputs,
                            ReduceMinParam<X86>& param, Context<X86>& ctx) {
        
        this->_ctx = &ctx;
        create(inputs, outputs, param, ctx);

        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<Tensor<X86> *>& inputs,
                            std::vector<Tensor<X86> *>& outputs,
                            ReduceMinParam<X86>& param, Context<X86> &ctx) {

        _n = inputs[0]->num();
        _c = inputs[0]->channel();
        _h = inputs[0]->height();
        _w = inputs[0]->width();
        // int count = input[0]->valid_size();
        _rank = inputs[0]->valid_shape().size();

        _reduce_dim = param.reduce_dim;
        if (!_reduce_dim.empty()) {
            //not empty
            for (int i = 0; i < _reduce_dim.size(); ++i) {
                if (_reduce_dim[i] < 0) {
                    _reduce_dim[i] += _rank;
                }
            }
        }
        return SaberSuccess;
    }
    
    virtual SaberStatus dispatch(const std::vector<Tensor<X86>*>& inputs,
                          std::vector<Tensor<X86>*>& outputs,
                          ReduceMinParam<X86>& param);

private:
    int _n;
    int _c;
    int _h;
    int _w;
    int _rank; //The dimentions of a tensor.
    std::vector<int> _reduce_dim;
};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_MATCH_MATRIX_H
