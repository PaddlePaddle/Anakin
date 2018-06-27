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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_POWER_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_POWER_H

#include "saber/funcs/impl/impl_power.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
            DataType inDtype,
            DataType outDtype,
            typename LayOutType_op,
            typename LayOutType_in,
            typename LayOutType_out>
class SaberPower<NV, OpDtype, inDtype, outDtype, \
    LayOutType_op, LayOutType_in, LayOutType_out>:\
    public ImplBase<
            Tensor<NV, inDtype, LayOutType_in>,
            Tensor<NV, outDtype, LayOutType_out>,
            Tensor<NV, OpDtype, LayOutType_op>,
            PowerParam<Tensor<NV, OpDtype, LayOutType_op>>> {

public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberPower() {}
    ~SaberPower() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             PowerParam<OpTensor> &power_param,
                             Context<NV> &ctx) {
        this->_ctx = &ctx;
        return SaberSuccess;
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               PowerParam<OpTensor> &power_param,
                               Context<NV> &ctx) {
        Shape shape = {inputs[0]->dims(), 1, 1, 1};
        _in_steps.re_alloc(shape);
        _out_steps.re_alloc(shape);
        _out_valid_shape.re_alloc(shape);
        Shape in_stride = inputs[0]->get_stride();
        Shape out_stride = outputs[0]->get_stride();
        Shape out_valid_shape = outputs[0]->valid_shape();
        cudaMemcpy((void*)(_out_steps.data()), &out_stride[0], sizeof(int)*4, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)(_in_steps.data()), &in_stride[0], sizeof(int)*4, cudaMemcpyHostToDevice);
        cudaMemcpy((void*)(_out_valid_shape.data()), &out_valid_shape[0], sizeof(int)*4, cudaMemcpyHostToDevice);
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 PowerParam<OpTensor> &power_param);
private:
    Tensor<NV, AK_INT32, LayOutType_in> _in_steps;
    Tensor<NV, AK_INT32, LayOutType_out> _out_steps;
    Tensor<NV, AK_INT32, LayOutType_out> _out_valid_shape;

};

template class SaberPower<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_POWER_H