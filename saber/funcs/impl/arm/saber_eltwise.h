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
#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_ELTWISE_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_ELTWISE_H

#include "saber/funcs/impl/impl_eltwise.h"

#ifdef USE_ARM_PLACE
namespace anakin{

namespace saber{

typedef void (*eltwise_func)(const float* din_a, \
    const float* din_b, float* dout, const int size);


template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberEltwise<ARM, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>:\
public ImplBase<
        Tensor<ARM, inDtype, LayOutType_in>,
        Tensor<ARM, outDtype, LayOutType_out>,
        Tensor<ARM, OpDtype, LayOutType_op>,
        EltwiseParam<Tensor<ARM, OpDtype, LayOutType_op>>> {
public:
    typedef Tensor<ARM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<ARM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<ARM, OpDtype, LayOutType_op> OpTensor;

    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberEltwise() {}
    ~SaberEltwise() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             EltwiseParam<OpTensor> &param, Context<ARM> &ctx) override {
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,\
                               std::vector<DataTensor_out*>& outputs,\
                               EltwiseParam<OpTensor> &param, \
                               Context<ARM> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs, \
                                 std::vector<DataTensor_out*>& outputs, \
                                 EltwiseParam<OpTensor> &param) override;

private:
    eltwise_func _impl{nullptr};
};

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_POOLING_H
