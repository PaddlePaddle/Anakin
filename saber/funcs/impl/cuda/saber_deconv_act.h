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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_ACT_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_ACT_H

#include "saber/funcs/impl/impl_deconv_act.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"

namespace anakin {

namespace saber {

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class SaberDeconv2DAct<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out> : \
    public ImplBase<
        Tensor<NV, inDtype, LayOutType_in>, 
        Tensor<NV, outDtype, LayOutType_out>,
        Tensor<NV, OpDtype, LayOutType_op>,
        ConvActiveParam<Tensor<NV, OpDtype, LayOutType_op> > > 
{
public:
    typedef Tensor<NV, inDtype, LayOutType_in> DataTensor_in;
    typedef Tensor<NV, outDtype, LayOutType_out> DataTensor_out;
    typedef Tensor<NV, OpDtype, LayOutType_op> OpTensor;
    typedef typename DataTensor_in::Dtype InDataType;
    typedef typename DataTensor_out::Dtype OutDataType;
    typedef typename OpTensor::Dtype OpDataType;

    SaberDeconv2DAct() : _use_k4_s2_p1(false) {}

    ~SaberDeconv2DAct() {}

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvActiveParam<OpTensor>& param, Context<NV>& ctx) {

        this->_ctx = &ctx;
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvActiveParam<OpTensor>& param, Context<NV> &ctx) {
        _use_k4_s2_p1 = true;
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.conv_param.weight()->width()==4);
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.conv_param.weight()->height()==4);
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.conv_param.stride_h==2);
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.conv_param.stride_w==2);
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.conv_param.pad_h==1);
        _use_k4_s2_p1 = _use_k4_s2_p1 && (param.conv_param.pad_w==1);
        if (_use_k4_s2_p1) {
            int in_channel = inputs[0]->channel();
            int out_channel = outputs[0]->channel();
            scale_to_new_tensor_k4_s2_p1_decov<4>(new_weights_dev,
                                               param.conv_param.weight(),
                                               in_channel, out_channel);
//            LOG(INFO)<<"scale weights finished!!";
        }
        //update_weights(param);

        return SaberSuccess;

    }
    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          ConvActiveParam<OpTensor>& param);
private:
    bool _use_k4_s2_p1;
    OpTensor new_weights_dev;
};
template class SaberDeconv2DAct<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
} // namespace saber

} // namespace anakin
#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_DECONV_ACT_H
