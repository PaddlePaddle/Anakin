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

#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_SEQUENCE_CONV_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_SEQUENCE_CONV_H

#include "saber/funcs/impl/impl_sequence_conv.h"
#include "saber/saber_funcs_param.h"


namespace anakin {
namespace saber {

template <DataType OpDtype,
          DataType inDtype,
          DataType outDtype,
          typename LayOutType_op,
          typename LayOutType_in,
          typename LayOutType_out>
class SaberSequenceConv<X86, OpDtype, inDtype, outDtype,
          LayOutType_op, LayOutType_in, LayOutType_out> : public ImplBase <
          Tensor<X86, inDtype, LayOutType_in>,
          Tensor<X86, outDtype, LayOutType_out>,
          Tensor<X86, OpDtype, LayOutType_op>,
          SequenceConvParam<Tensor<X86, OpDtype, LayOutType_op> > > {
public:
    typedef Tensor<X86, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<X86, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<X86, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype DataType_in;
    typedef typename DataTensor_out::Dtype DataType_out;
    typedef typename OpTensor::Dtype DataType_op;

    SaberSequenceConv() = default;

    ~SaberSequenceConv() {}

    virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                             std::vector<DataTensor_out*>& outputs,
                             SequenceConvParam<OpTensor>& param,
                             Context<X86>& ctx) {
        this->_ctx = ctx;
        CHECK_EQ(param.padding_trainable, false) << "not support padding_trainable==true";
        CHECK_EQ(param.context_stride, 1) << "not support context_stride!=1";

        if (param.padding_tensor != nullptr) {
            CHECK_EQ(1, 0) << "not support padding_tensor";
        }

        CHECK_NOTNULL(param.filter_tensor);
        _hidden_size = param.filter_tensor->height() / param.context_length;
        _feature_size = param.filter_tensor->width();
        _up_pad = std::max(0, -param.context_start);
        _down_pad = std::max(0, param.context_start + param.context_length - 1);
        _hidden_kernel_size = _hidden_size * param.context_length;
        return create(inputs, outputs, param, ctx);
    };

    virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                               std::vector<DataTensor_out*>& outputs,
                               SequenceConvParam<OpTensor>& param,
                               Context<X86>& ctx) {
        return SaberSuccess;
    };

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 SequenceConvParam<OpTensor>& param);
private:
    OpTensor _temp_im2col_tensor;
    int _hidden_size;
    int _feature_size;
    int _hidden_kernel_size;
    int _up_pad;
    int _down_pad;
};
}
}

#endif