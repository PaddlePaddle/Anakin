/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SOFTMAX_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SOFTMAX_H
#include "saber/funcs/impl/impl_softmax.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{


template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberSoftmax<ARM, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<ARM, inDtype, LayOutType_in>,
        Tensor<ARM, outDtype, LayOutType_out>,
        Tensor<ARM, OpDtype, LayOutType_op>,
        SoftmaxParam<Tensor<ARM, OpDtype, LayOutType_op>>> {
public:
    typedef TargetWrapper<ARM> API;
    typedef Tensor<ARM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<ARM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<ARM, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberSoftmax() = default;
    ~SaberSoftmax() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             SoftmaxParam<OpTensor> &param, Context<ARM> &ctx) override {
        // get context
        this->_ctx = ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               SoftmaxParam<OpTensor> &param, Context<ARM> &ctx) override {

        Shape shape_in = inputs[0]->valid_shape();
        Shape shape_out = outputs[0]->valid_shape();
        _outer_num = inputs[0]->count_valid(0, param.axis);
        _inner_num = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
        _axis_size = shape_in[param.axis];

        int buffer_size = this->_inner_num * this->_outer_num;
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 SoftmaxParam<OpTensor> &param);

private:
    int _axis_size{0};
    int _inner_num{0};
    int _outer_num{0};

};

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_FUNCS_IMPL_ARM_SABER_SOFTMAX_H
