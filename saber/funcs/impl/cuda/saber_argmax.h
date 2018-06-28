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

#ifndef ANAKIN_SABER_FUNCS_IMPL_SABER_ARGMAX_H
#define ANAKIN_SABER_FUNCS_IMPL_SABER_ARGMAX_H

#include "saber/funcs/impl/impl_argmax.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class SaberArgmax<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, 
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        ArgmaxParam<Tensor<NV, OpDtype, LayOutType_op> > > 
{
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberArgmax()
    {}

    ~SaberArgmax() {

    }

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                        std::vector<DataTensor_out *>& outputs,
                        ArgmaxParam<OpTensor>& param, 
                        Context<NV> &ctx) {
        this->_ctx = &ctx;
        if (!param.has_axis) {
            int inner_dim = inputs[0]->count(1, inputs[0]->dims());
            int outer_dim = inputs[0]->num();
            int block_num = CUDA_GET_BLOCKS(inner_dim);
            _block_max_value.re_alloc(Shape(outer_dim, block_num, 1, 1));
            _block_max_index.re_alloc(Shape(outer_dim, block_num, 1, 1));
        }
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                        std::vector<DataTensor_out *>& outputs,
                        ArgmaxParam<OpTensor>& param, 
                        Context<NV>& ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in *>& inputs,
                        std::vector<DataTensor_out *>& outputs,
                        ArgmaxParam<OpTensor>& param);

private:
    Tensor<NV, inDtype, LayOutType_in> _block_max_value;
    Tensor<NV, inDtype, LayOutType_in> _block_max_index;
};

template class SaberArgmax<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

}

}

#endif //ANAKIN_SABER_FUNCS_IMPL_SABER_ARGMAX_H