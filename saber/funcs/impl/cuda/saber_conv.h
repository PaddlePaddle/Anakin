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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV2D_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV2D_H

#include <vector>
#include "saber/funcs/impl/impl_conv.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin{

namespace saber{

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class SaberConv2D<NV, OpDtype, inDtype, outDtype,\
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

    SaberConv2D() {}

    ~SaberConv2D() {}

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvParam<OpTensor>& param, Context<NV> &ctx) {
        this->_ctx = &ctx;
        //This is an ugly impl for now
        if (param.stride_h == 1 &&
                param.stride_w == 1 &&
                param.weight()->height() == 3 &&
                param.weight()->width() == 3 && param.group == 1) {

            dispatch_func = winograd_conv<OutDataType, OpDataType>;
        } else if (param.group == 1) {
            const int K = param.weight()->num();
            if (K % 4 == 0) {
                if (param.bias()->size() > 0) {
                    dispatch_func = direct_conv_bias_Kdivis4<OutDataType, OpDataType>;
                } else {
                    dispatch_func = direct_conv_Kdivis4<OutDataType, OpDataType>;
                }
            } else {
                if (param.bias()->size() > 0) {
                    dispatch_func = direct_conv_bias_Kindiv4<OutDataType, OpDataType>;
                } else {
                    dispatch_func = direct_conv_Kindiv4<OutDataType, OpDataType>;
                }
            }
        } else {
          return SaberUnImplError;
        }
        _kernel_height = param.weight()->height();
        _kernel_width = param.weight()->width();
        trans_weights(inputs, outputs, param, ctx);
        cudaDeviceSynchronize();
        return create(inputs, outputs, param, ctx);
    }
    
    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvParam<OpTensor>& param, Context<NV>& ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          ConvParam<OpTensor>& param){
        //err code?
        Shape shape_in = inputs[0]->valid_shape();
        Shape shape_out = outputs[0]->valid_shape();
        //LOG(WARNING) << "shape in: " << shape_in[0] << ", " << shape_in[1] << ", " << shape_in[2] << ", " << shape_in[3];
        const OutDataType* bias_data = nullptr;
        if (param.bias()->size() > 0) {
            bias_data = param.bias()->data();
        }
        //LOG(WARNING) << "saber conv dispatch";
        //CUDA_CHECK(cudaGetLastError());
        //LOG(WARNING) << "saber conv check previous error";
        //LOG(WARNING) << "width = " << inputs[0]->width() << ", height = " << inputs[0]->height() << ", channel = " << inputs[0]->channel();
        //LOG(WARNING) << "kw = " << param.weight()->width() << ", kh = " << param.weight()->height();
        //LOG(WARNING) << "group = " << param.group << ", filter = " << outputs[0]->channel();

        dispatch_func(inputs[0]->data(), outputs[0]->mutable_data(),
                      param.weight()->data(),
                      bias_data,
                      inputs[0]->num(),
                      inputs[0]->channel(),
                      inputs[0]->height(),
                      inputs[0]->width(),
                      outputs[0]->channel(),
                      outputs[0]->height(),
                      outputs[0]->width(),
                      shape_in[1],
                      shape_in[2],
                      shape_in[3],
                      shape_out[1],
                      shape_out[2],
                      shape_out[3],
                      _kernel_height,
                      _kernel_width,
                      param.pad_h,
                      param.pad_w,
                      param.stride_h,
                      param.stride_w,
                      param.dilation_h,
                      param.dilation_w,
                      param.group,
                      param.alpha,
                      param.beta,
                      this->_ctx->get_compute_stream());
        
        CUDA_CHECK(cudaGetLastError()); 
        return SaberSuccess;
    }

    void trans_weights(const std::vector<DataTensor_in *>& inputs,
                       std::vector<DataTensor_out *>& outputs,
                       ConvParam<OpTensor>& param, Context<NV> &ctx) {

        Tensor<X86, OpDtype, LayOutType_op> trans_weights_host;
        if (param.stride_h == 1 &&
            param.stride_w == 1 &&
            param.weight()->height() == 3 &&
            param.weight()->width() == 3 && param.group == 1)
        {
            //Update weights if need
            Shape weight_shape = param.weight()->shape();
            Tensor<X86, OpDtype, LayOutType_out> new_weight;
            new_weight.re_alloc(weight_shape);
            new_weight.copy_from(*(param.weight()));
            OpDataType *weight_data = new_weight.mutable_data();

            int round_in_channel = i_align_up(inputs[0]->channel(), 8);
            int round_out_channel = i_align_up(param.weight()->num(), 32);
            int weight4x4_size = round_in_channel * round_out_channel * 4 * 4;
            Shape old_shape = param.weight()->shape();
            trans_weights_host.re_alloc({weight4x4_size, 1, 1 ,1});
            OpDataType* _host_work_space = trans_weights_host.mutable_data();
            transform_3x3_weight_2_4x4(weight_data, _host_work_space, param.weight()->num(),
                                       round_out_channel, inputs[0]->channel(), round_in_channel);

            param.mutable_weight()->re_alloc({weight4x4_size, 1, 1, 1});
            param.mutable_weight()->copy_from(trans_weights_host);
            param.mutable_weight()->set_shape(old_shape);

        } else if (param.group == 1) {

            int weight_size = (param.weight()->shape()).count();
            Tensor<X86, OpDtype, LayOutType_out> weight_host;
            weight_host.re_alloc(param.weight()->shape());
            weight_host.copy_from(*(param.weight()));
            const OpDataType *weight_data = weight_host.data();
            trans_weights_host.re_alloc(param.weight()->shape());
            OpDataType* _host_work_space = trans_weights_host.mutable_data();

            transpose_filter_KCRS_2_CRSK(weight_data, _host_work_space, \
                                         param.weight()->num(), \
                                         param.weight()->channel(), \
                                         param.weight()->height(), \
                                         param.weight()->width());

            param.mutable_weight()->re_alloc(param.weight()->shape());
            param.mutable_weight()->copy_from(trans_weights_host);
        }
        cudaDeviceSynchronize();
    }

private:

    int _kernel_height;
    int _kernel_width;
    std::function<void(const InDataType*,
      OutDataType*,
      const OpDataType*,
      const OutDataType*,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      float,
      float, 
      cudaStream_t)> dispatch_func;
};
template class SaberConv2D<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}

}


#endif //ANAKIN_SABER_FUNCS_SABER_CONV2D_H
