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
#ifndef ANAKIN_SABER_FUNCS_ARM_IMPL_SABER_CONV_ACT_POOLING_H
#define ANAKIN_SABER_FUNCS_ARM_IMPL_SABER_CONV_ACT_POOLING_H

#include "saber/funcs/impl/arm/saber_conv_act.h"
#include "saber/funcs/impl/arm/saber_pooling.h"
#include "saber/funcs/impl/impl_conv_act_pooling.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
class SaberConv2DActPooling<ARM, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<ARM, inDtype, LayOutType_in>,
        Tensor<ARM, outDtype, LayOutType_out>,
        Tensor<ARM, OpDtype, LayOutType_op>,
        ConvActivePoolingParam<Tensor<ARM, OpDtype, LayOutType_op> > > {
public:
    typedef Tensor<ARM, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<ARM, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<ARM, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;


    SaberConv2DActPooling();

    ~SaberConv2DActPooling();

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                             std::vector<DataTensor_out *>& outputs,
                             ConvActivePoolingParam<OpTensor> &param, Context<ARM> &ctx) override;

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                               std::vector<DataTensor_out *>& outputs,
                               ConvActivePoolingParam<OpTensor> &param, Context<ARM> &ctx) override;

    virtual SaberStatus dispatch(const std::vector<DataTensor_in *>& inputs,
                                 std::vector<DataTensor_out *>& outputs,
                                 ConvActivePoolingParam<OpTensor> &param) override;

    void get_conv_out_tensor(const std::vector<DataTensor_in *>& inputs,
                              ConvActivePoolingParam<OpTensor> &param) {

        ConvParam<OpTensor> conv_param = param.conv_param;

        Shape conv_out_shape = inputs[0]->valid_shape();
        // append the $n and $c/$k, output: N * K * P * Q
        int num_idx = inputs[0]->num_index();
        int channel_idx = inputs[0]->channel_index();
        int height_idx = inputs[0]->height_index();
        int width_idx = inputs[0]->width_index();

        conv_out_shape[num_idx] = inputs[0]->num(); // N
        conv_out_shape[channel_idx] = conv_param.weight()->num(); // K

        int input_dim = inputs[0]->height(); // P
        int kernel_exten = conv_param.dilation_h * (conv_param.weight()->height() - 1) + 1;
        int output_dim = (input_dim + 2 * conv_param.pad_h - kernel_exten)
                         / conv_param.stride_h + 1;

        conv_out_shape[height_idx] = output_dim;

        input_dim = inputs[0]->width(); // Q
        kernel_exten = conv_param.dilation_w * (conv_param.weight()->width() - 1) + 1;
        output_dim = (input_dim + 2 * conv_param.pad_w - kernel_exten)
                     / conv_param.stride_w + 1;

        conv_out_shape[width_idx] = output_dim;

        _tensor_tmp.reshape(conv_out_shape);
        _vtensor_tmp[0] = &_tensor_tmp;
    }

private:
    SaberConv2DAct<ARM, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>* _conv_act_op;
    SaberPooling<ARM, OpDtype, inDtype, outDtype, LayOutType_op, LayOutType_in, LayOutType_out>* _pool_op;

    ConvActiveParam<OpTensor>* _conv_act_param;

    DataTensor_in _tensor_tmp;
    std::vector<DataTensor_in *> _vtensor_tmp;
};

} //namespace saber

} //namespace anakin
#endif // USE_ARM_PLACE

#endif //ANAKIN_SABER_FUNCS_ARM_IMPL_SABER_CONV_ACT_POOLING_H
