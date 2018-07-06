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

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_ACTIVATION_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_ACTIVATION_H

#include "saber/funcs/impl/impl_activation.h"
#include "saber/funcs/impl/arm/impl/neon_mathfun.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/core/context.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class SaberActivation<ARM, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<ARM, inDtype, LayOutType_in>,
        Tensor<ARM, outDtype, LayOutType_out>,
        Tensor<ARM, OpDtype, LayOutType_op>,
        ActivationParam<Tensor<ARM, OpDtype, LayOutType_op> > >
{
public:
    typedef Tensor<ARM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<ARM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<ARM, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberActivation()
    {}

    ~SaberActivation() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             ActivationParam<OpTensor> &param,
                             Context<ARM> &ctx) override{
      return create(inputs, outputs, param, ctx);
    }
    
    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               ActivationParam<OpTensor> &param,
                               Context<ARM> &ctx) override;
    
    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 ActivationParam<OpTensor> &param) override;
public:
    int _threads;
    int _nums_per_thread;
    int _dim16;
    int _dim16_remain;
    int _dim4;
    int _dim4_remain;
    int _remain;
    int _size;
    int _channel;
    int _num;

};

}

}
#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_ACTIVATION_H
