/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

template <DataType OpDtype,
            DataType inDtype,
            DataType outDtype,
            typename LayOutType_op,
            typename LayOutType_in,
            typename LayOutType_out>
    class SaberLayerNorm<NV, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>:\
    public ImplBase<
            Tensor<NV, inDtype, LayOutType_in>,
            Tensor<NV, outDtype, LayOutType_out>,
            Tensor<NV, OpDtype, LayOutType_op>,
            LayerNormParam<Tensor<NV, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberLayerNorm() = default;
    ~SaberLayerNorm() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             LayerNormParam<OpTensor> &param,
                             Context<NV> &ctx) {
        // get context
        this->_ctx = ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               LayerNormParam<OpTensor> &param,
                               Context<NV> &ctx) {
        //Shape sh_in = inputs[0]->valid_shape();
        _inner_size = inputs[0]->count_valid(param.axis, inputs[0]->dims());
        _outer_size = inputs[0]->count_valid(0, param.axis);

        Shape sh = Shape::zero(inputs[0]->dims());
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

    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 LayerNormParam<OpTensor> &param);


private:
    OpTensor _mean;
    OpTensor _std;
    int _inner_size;
    int _outer_size;
    bool _flag_scale{true};
    bool _flag_bias{true};
};
template class SaberLayerNorm<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_LAYER_NORM_H
