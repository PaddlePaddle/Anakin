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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_REDUCE_MIN_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_REDUCE_MIN_H

#include "saber/funcs/impl/impl_reduce_min.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberReduceMin<NV, OpDtype> :
    public ImplBase<
        NV, OpDtype,
        ReduceMinParam<NV> > {
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;
    SaberReduceMin() {}
    ~SaberReduceMin() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            ReduceMinParam<NV>& param, Context<NV>& ctx) {
        
        this->_ctx = &ctx;
        create(inputs, outputs, param, ctx);

        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                            std::vector<Tensor<NV> *>& outputs,
                            ReduceMinParam<NV>& param, Context<NV> &ctx) {
        
        _num = inputs[0]->num();
        _channel = inputs[0]->channel();
        _height = inputs[0]->height();
        _width = inputs[0]->width();
        _rank = inputs[0]->valid_shape().size();
        if (!param.reduce_dim.empty()) {
            //reduce dim isn't empty
           
            for (int i = 0; i < param.reduce_dim.size(); ++i) {
                if (param.reduce_dim[i] < 0) {
                    _reduce_dim.push_back(param.reduce_dim[i] + _rank);
                }else {
                    _reduce_dim.push_back(param.reduce_dim[i]);
                }
            }
        }

        return SaberSuccess;
    }
    
    virtual SaberStatus dispatch(const std::vector<Tensor<NV>*>& inputs,
                          std::vector<Tensor<NV>*>& outputs,
                          ReduceMinParam<NV>& param);

private:
    int _rank; // dimetions
    int _num;
    int _channel;
    int _height;
    int _width;
    std::vector<int> _reduce_dim;
    Tensor<NV> _tensor_tmp;

};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_MATCH_MATRIX_H
