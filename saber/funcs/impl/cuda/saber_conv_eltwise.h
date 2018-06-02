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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_ELTWISE_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_ELTWISE_H

#include <vector>
#include "saber/funcs/impl/impl_conv_eltwise.h"
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
class SaberConv2DEltWise<NV, OpDtype, inDtype, outDtype,\
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

    SaberConv2DEltWise():_host_work_space(nullptr), _gpu_work_space(nullptr)
    {}

    ~SaberConv2DEltWise() {
        if (_host_work_space)
        {
            free(_host_work_space);
        }
        if (_gpu_work_space)
        {
            cudaFree(_gpu_work_space);
        }
    }

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvActiveParam<OpTensor>& param, Context<NV> &ctx)  {
        this->_ctx = ctx;
        //This is an ugly impl for now
        if (param.conv_param.stride_h == 1 && 
            param.conv_param.stride_w == 1 && 
            param.conv_param.weight()->height() == 3 && 
            param.conv_param.weight()->width() == 3)
        {
            //Update weights if need
            Shape weight_shape = param.conv_param.weight()->shape();
            Tensor<X86, OpDtype, LayOutType_op> new_weight;
            new_weight.re_alloc(weight_shape);
            new_weight.copy_from(*(param.conv_param.weight()));
            OpDataType *weight_data = new_weight.mutable_data();

            int round_in_channel = i_align_up(inputs[0]->channel(),8);
            int round_out_channel = i_align_up(param.conv_param.weight()->num(),32);

            int weight4x4_size = round_in_channel * round_out_channel * 4 * 4;
            _host_work_space = (OpDataType*)malloc(weight4x4_size * sizeof(OpDataType));
            CUDA_CHECK(cudaMalloc((void**)&_gpu_work_space, weight4x4_size*sizeof(OpDataType)));
            transform_3x3_weight_2_4x4(weight_data, _host_work_space, param.conv_param.weight()->num(), round_out_channel, inputs[0]->channel(), round_in_channel);    
            CUDA_CHECK(cudaMemcpy((void*)_gpu_work_space,
                          (void*)_host_work_space,
                          weight4x4_size*sizeof(OpDataType),
                          cudaMemcpyHostToDevice));
                
            dispatch_func = winograd_conv_eltwise<InDataType, OpDataType>;

        }else{
          return SaberUnImplError;
        }
        cudaDeviceSynchronize();
        return create(inputs, outputs, param, ctx);
    }

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvActiveParam<OpTensor>& param, Context<NV>& ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          ConvActiveParam<OpTensor>& param){
        //err code?
            Shape shape_in = inputs[0]->valid_shape();
            Shape shape_out = outputs[0]->valid_shape();
            const InDataType* bias_data = NULL;
            if (param.conv_param.bias()->size() > 0) {
                bias_data = param.conv_param.bias()->data();
            }
            dispatch_func(inputs[0]->data(), outputs[0]->mutable_data(),
                    _gpu_work_space,
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
                    param.conv_param.weight()->height(),
                    param.conv_param.weight()->width(),
                    param.conv_param.pad_h,              
                    param.conv_param.pad_w,              
                    param.conv_param.stride_h,              
                    param.conv_param.stride_w,              
                    param.conv_param.dilation_h,              
                    param.conv_param.dilation_w, 
                    param.conv_param.group, 
                    param.conv_param.alpha, 
                    param.conv_param.beta,
                    param.eltwise_param.operation, 
                    this->_ctx.get_compute_stream()); 

        CUDA_CHECK(cudaGetLastError()); 
        return SaberSuccess;
    }

private:
    OpDataType* _host_work_space;
    OpDataType* _gpu_work_space;
    std::function<void(const InDataType*, 
      OutDataType*,
      const OpDataType*,
      const InDataType*,
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
      EltwiseType,
      cudaStream_t)> dispatch_func;
};
template class SaberConv2DEltWise<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}

}


#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV_ELTWISE_H
