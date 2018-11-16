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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ARGMAX_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_ARGMAX_H

#include "saber/funcs/impl/impl_argmax.h"

namespace anakin{

namespace saber{

template <DataType OpDtype>
class SaberArgmax<NV, OpDtype> : 
    public ImplBase<
        NV, OpDtype,
        ArgmaxParam<NV> > 
{
public:
    typedef typename DataTrait<NV, OpDtype>::Dtype OpDataType;

    SaberArgmax(){}

    ~SaberArgmax() {}

    virtual SaberStatus init(const std::vector<Tensor<NV> *>& inputs,
                        std::vector<Tensor<NV> *>& outputs,
                        ArgmaxParam<NV>& param, 
                        Context<NV> &ctx) {
        return create(inputs, outputs, param, ctx);//SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<Tensor<NV> *>& inputs,
                        std::vector<Tensor<NV> *>& outputs,
                        ArgmaxParam<NV>& param, 
                        Context<NV>& ctx) {
        this->_ctx = &ctx;
        if (!param.has_axis) {
            int inner_dim = inputs[0]->count_valid(1, inputs[0]->dims());
            int outer_dim = inputs[0]->num();
            int block_num = CUDA_GET_BLOCKS(inner_dim);
            _block_max_value.re_alloc(Shape({outer_dim, block_num, 1, 1}, Layout_NCHW), OpDtype);
            _block_max_index.re_alloc(Shape({outer_dim, block_num, 1, 1}, Layout_NCHW), OpDtype);
        }
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<Tensor<NV> *>& inputs,
                        std::vector<Tensor<NV> *>& outputs,
                        ArgmaxParam<NV>& param);

private:
    Tensor<NV> _block_max_value;
    Tensor<NV> _block_max_index;
};

}

}

#endif //ANAKIN_SABER_FUNCS_IMPL_SABER_ARGMAX_H
