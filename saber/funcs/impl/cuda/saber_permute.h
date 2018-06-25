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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PERMUTE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PERMUTE_H

#include "saber/funcs/impl/impl_permute.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
            DataType inDtype,
            DataType outDtype,
            typename LayOutType_op,
            typename LayOutType_in,
            typename LayOutType_out>
class SaberPermute<NV, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>:\
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>,
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        PermuteParam<Tensor<NV, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberPermute() {}

    ~SaberPermute() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             PermuteParam<OpTensor> &param,
                             Context<NV> &ctx) {
        this->_ctx = &ctx;
        _num_axes = inputs[0]->valid_shape().size();
        for (int i = 0; i < _num_axes; i++) {
            if (std::find(_order_dims.begin(), _order_dims.end(),
                          param.order[i]) == _order_dims.end()) {
                _order_dims.push_back(param.order[i]);
            }
        }

        CHECK_EQ(_num_axes, _order_dims.size());

        // set _need_permute
        _need_permute = false;
        for (int i = 0; i < _num_axes; ++i) {
            if (param.order[i] != i) {
                _need_permute = true;
                break;
            }
        }
        Shape order_shape = {_num_axes, 1, 1, 1};
        _permute_order.reshape(order_shape);
        cudaMemcpy(_permute_order.mutable_data(), &(param.order[0]),
                   sizeof(int) * _permute_order.size(), cudaMemcpyHostToDevice);
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               PermuteParam<OpTensor> &param,
                               Context<NV> &ctx) {

        Shape order_shape = {_num_axes, 1, 1, 1};
        _in_steps.reshape(order_shape);
        _out_steps.reshape(order_shape);
        _out_valid_shape.reshape(order_shape);

        Shape in_stride = inputs[0]->get_stride();
        Shape out_stride = outputs[0]->get_stride();

        cudaMemcpy(_in_steps.mutable_data(), &in_stride[0],
                   sizeof(int) * _in_steps.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(_out_steps.mutable_data(), &out_stride[0],
                   sizeof(int) * _out_steps.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(_out_valid_shape.mutable_data(), &((outputs[0]->valid_shape())[0]),
                   sizeof(int) * _out_valid_shape.size(), cudaMemcpyHostToDevice);
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 PermuteParam<OpTensor> &param);

private:
    int _num_axes;
    bool _need_permute;
    std::vector<int> _order_dims;
    Tensor<NV, AK_INT32, LayOutType_in> _permute_order;
    Tensor<NV, AK_INT32, LayOutType_in> _in_steps;
    Tensor<NV, AK_INT32, LayOutType_out> _out_steps;
    Tensor<NV, AK_INT32, LayOutType_out> _out_valid_shape;
};

template class SaberPermute<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_PERMUTE_H
