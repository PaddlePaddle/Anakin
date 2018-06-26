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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_H

#include "saber/funcs/impl/impl_deconv.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class SaberDeconv2D<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, 
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        ConvParam<Tensor<NV, OpDtype, LayOutType_op> > > 
{
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberDeconv2D() :_use_k4_s2_p1(false) {}

    ~SaberDeconv2D() {}

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvParam<OpTensor>& param, Context<NV>& ctx) {

        this->_ctx = &ctx;

        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvParam<OpTensor>& param, Context<NV> &ctx) {
        _use_k4_s2_p1 = true;
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.weight()->width()==4);
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.weight()->height()==4);
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.stride_h==2);
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.stride_w==2);
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.pad_h==1);
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.pad_w==1);
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.group==1);
        if (_use_k4_s2_p1) {
            int in_channel = inputs[0]->channel();
            int out_channel = outputs[0]->channel();
            scale_to_new_tensor_k4_s2_p1_deconv<4>(param.mutable_weight(),
                                                   in_channel, out_channel);
//            LOG(INFO)<<"scale weights finished!!";
        } 
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          ConvParam<OpTensor>& param);
private:
    bool _use_k4_s2_p1;
};
template class SaberDeconv2D<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;

} //namespace saber

} //namespace anakin
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_H
