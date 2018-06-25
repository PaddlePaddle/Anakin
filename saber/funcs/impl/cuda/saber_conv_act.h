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

#ifndef ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV2DACT_H
#define ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV2DACT_H

#include <vector>
#include "saber/funcs/impl/impl_conv_act.h"
#include "saber/funcs/impl/cuda/base/sass_funcs.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin{

namespace saber{

template <typename dtype, bool bias_flag, bool relu_flag>
SaberStatus saber_depthwise_conv_act(const dtype* input, dtype* output, \
    int num, int cin, int hin, int win, int hout, int wout, \
    int kw, int kh, int stride_w, int stride_h, \
    int pad_h, int pad_w, const dtype* weights, const dtype* bias, \
    cudaStream_t stream);

template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
class SaberConv2DAct<NV, OpDtype, inDtype, outDtype,\
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

    SaberConv2DAct()
            : _use_k1s1p0(false)
    {}

    ~SaberConv2DAct() {}

    virtual SaberStatus create(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvActiveParam<OpTensor>& param, Context<NV>& ctx) {
        return SaberSuccess;
    }

    virtual SaberStatus init(const std::vector<DataTensor_in *>& inputs,
                            std::vector<DataTensor_out *>& outputs,
                            ConvActiveParam<OpTensor>& param, Context<NV> &ctx) {
        this->_ctx = ctx;

        _kernel_height = param.conv_param.weight()->height();
        _kernel_width = param.conv_param.weight()->width();
        _use_k1s1p0 = true;
        _use_k1s1p0 = _use_k1s1p0 && (_kernel_height == 1);
        _use_k1s1p0 = _use_k1s1p0 && (_kernel_width == 1);
        _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.pad_h == 0);
        _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.pad_w == 0);
        _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.stride_h == 1);
        _use_k1s1p0 = _use_k1s1p0 && (param.conv_param.stride_w == 1);
        _use_k1s1p0 = _use_k1s1p0 && (inputs[0]->num() == 1);
        if (_use_k1s1p0) {
            return SaberSuccess;
        }
        if (param.conv_param.group == inputs[0]->channel() && \
            param.conv_param.group == outputs[0]->channel()){
            return SaberSuccess;

        } else if (param.conv_param.stride_h == 1 &&
            param.conv_param.stride_w == 1 && 
            _kernel_height == 3 &&
            _kernel_width == 3
            &&param.conv_param.group == 1) {

            if (param.has_eltwise) {
                dispatch_func_elt = winograd_conv_eltwise<InDataType, OpDataType>;
            } else {
                dispatch_func = winograd_conv_relu<InDataType, OpDataType>;
            }
        } else if(param.conv_param.group == 1) {
            const int K = param.conv_param.weight()->num();
            if(K % 4 == 0) {
                if (param.conv_param.bias()->size() > 0)
                    dispatch_func = direct_conv_bias_relu_Kdivis4<InDataType, OpDataType>;
                else
                    return SaberUnImplError;
            } else {   // TODO: would merge the bias(with/without) version
                if (param.conv_param.bias()->size() > 0)
                    dispatch_func = direct_conv_bias_relu_Kindiv4<InDataType, OpDataType>;
                else
                    return SaberUnImplError;
            }      
        } else {
            return SaberUnImplError;
        }
        trans_weights(inputs, outputs, param, ctx);
        cudaDeviceSynchronize();

        return create(inputs, outputs, param, ctx);
    }
    virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                          std::vector<DataTensor_out*>& outputs,
                          ConvActiveParam<OpTensor>& param) {

        //err code?
        Shape shape_in = inputs[0]->valid_shape();
        Shape shape_out = outputs[0]->valid_shape();
        const InDataType* bias_data;
        if (param.conv_param.bias()->size() > 0) {
            bias_data = param.conv_param.bias()->data();
        }else{
            bias_data = nullptr;
        }

        int num = inputs[0]->num();
        int chin = inputs[0]->channel();
        int win = inputs[0]->width();
        int hin = inputs[0]->height();
        int chout = outputs[0]->channel();
        int wout = outputs[0]->width();
        int hout = outputs[0]->height();

        //LOG(INFO) << "saber conv act";
        if (_use_k1s1p0) {
//            LOG(INFO)<<"using k1s1p0";
            conv_gemm_k1s1p0(outputs[0]->mutable_data(),
                             inputs[0]->data(),
                             param.conv_param.weight()->data(),
                             chout, chin, hin, win, bias_data,
                             this->_ctx.get_compute_stream());
            return SaberSuccess;
        }
        if (param.conv_param.group == chin && param.conv_param.group == chout) {

            if (param.conv_param.bias()->size() > 0) {
                if (param.has_active) {
                    saber_depthwise_conv_act<InDataType, true, true>(inputs[0]->data(), \
                        outputs[0]->mutable_data(), num, chin, hin, win, hout, \
                        wout, _kernel_width, _kernel_height, param.conv_param.stride_w, \
                        param.conv_param.stride_h, param.conv_param.pad_w, param.conv_param.pad_h,\
                        (const OpDataType*)param.conv_param.weight()->data(), bias_data, \
                        this->_ctx.get_compute_stream());
                } else {
                    saber_depthwise_conv_act<InDataType, true, false>(inputs[0]->data(), \
                        outputs[0]->mutable_data(), num, chin, hin, win, hout, \
                        wout, _kernel_width, _kernel_height, param.conv_param.stride_w, \
                        param.conv_param.stride_h, param.conv_param.pad_w, param.conv_param.pad_h,\
                        (const OpDataType*)param.conv_param.weight()->data(), bias_data, \
                        this->_ctx.get_compute_stream());
                }

            } else {
                if (param.has_active) {
                    saber_depthwise_conv_act<InDataType, false, true>(inputs[0]->data(), \
                        outputs[0]->mutable_data(), inputs[0]->num(), inputs[0]->channel(), \
                        inputs[0]->height(), inputs[0]->width(), outputs[0]->height(), \
                        outputs[0]->width(), _kernel_width, \
                        _kernel_height, param.conv_param.stride_w, \
                        param.conv_param.stride_h, param.conv_param.pad_w, param.conv_param.pad_h,\
                        (const OpDataType*)param.conv_param.weight()->data(), bias_data, \
                        this->_ctx.get_compute_stream());
                } else {
                    saber_depthwise_conv_act<InDataType, false, false>(inputs[0]->data(), \
                        outputs[0]->mutable_data(), inputs[0]->num(), inputs[0]->channel(), \
                        inputs[0]->height(), inputs[0]->width(), outputs[0]->height(), \
                        outputs[0]->width(), _kernel_width, \
                        _kernel_height, param.conv_param.stride_w, \
                        param.conv_param.stride_h, param.conv_param.pad_w, param.conv_param.pad_h,\
                        (const OpDataType*)param.conv_param.weight()->data(), bias_data, \
                        this->_ctx.get_compute_stream());
                }
            }
        } else if (param.has_eltwise) {
            //std::cout << "In dispatch_func_elt" << std::endl;
            dispatch_func_elt(inputs[0]->data(), outputs[0]->mutable_data(), \
                param.conv_param.weight()->data(), bias_data, num, chin, hin, win, \
                chout, hout, wout,
                    shape_in[1],
                    shape_in[2],
                    shape_in[3],
                    shape_out[1],
                    shape_out[2],
                    shape_out[3], 
                    _kernel_height, _kernel_width,
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
            } else {
                dispatch_func(inputs[0]->data(), outputs[0]->mutable_data(), \
                    param.conv_param.weight()->data(), bias_data, num, chin, hin, win, \
                    chout, hout, wout, \
                    shape_in[1],
                    shape_in[2],
                    shape_in[3],
                    shape_out[1],
                    shape_out[2],
                    shape_out[3], 
                    _kernel_height,
                    _kernel_width,
                    param.conv_param.pad_h,              
                    param.conv_param.pad_w,              
                    param.conv_param.stride_h,              
                    param.conv_param.stride_w,              
                    param.conv_param.dilation_h,              
                    param.conv_param.dilation_w, 
                    param.conv_param.group, 
                    param.conv_param.alpha, 
                    param.conv_param.beta, 
                    this->_ctx.get_compute_stream());                 
            }

        return SaberSuccess;
    }
    void trans_weights(const std::vector<DataTensor_in *>& inputs,
                       std::vector<DataTensor_out *>& outputs,
                       ConvActiveParam<OpTensor>& param, Context<NV> &ctx) {
        Tensor<X86, OpDtype, LayOutType_op> trans_weights_host;
        if (_use_k1s1p0) {
            return;
        }
        if (param.conv_param.group == inputs[0]->channel() && \
            param.conv_param.group == outputs[0]->channel()){
            return;

        } else if (param.conv_param.stride_h == 1 &&
                   param.conv_param.stride_w == 1 &&
                   _kernel_height == 3 &&
                   _kernel_width == 3
                   &&param.conv_param.group == 1) {
            //Update weights if need
            Shape weight_shape = param.conv_param.weight()->shape();
            Tensor<X86, OpDtype, LayOutType_op> new_weight;
            new_weight.re_alloc(weight_shape);
            new_weight.copy_from(*(param.conv_param.weight()));
            OpDataType *weight_data = new_weight.mutable_data();

            int round_in_channel = i_align_up(inputs[0]->channel(),8);
            int round_out_channel = i_align_up(param.conv_param.weight()->num(),32);
            int weight4x4_size = round_in_channel * round_out_channel * 4 * 4;
            trans_weights_host.re_alloc({weight4x4_size, 1, 1, 1});
            OpDataType* _host_work_space;
            _host_work_space = trans_weights_host.mutable_data();

            transform_3x3_weight_2_4x4(weight_data, _host_work_space, param.conv_param.weight()->num(), round_out_channel, inputs[0]->channel(), round_in_channel);

            param.conv_param.mutable_weight()->re_alloc({weight4x4_size, 1, 1, 1});
            param.conv_param.mutable_weight()->copy_from(trans_weights_host);
        } else if(param.conv_param.group == 1) {
            Shape weight_shape = param.conv_param.weight()->shape();
            Tensor<X86, OpDtype, LayOutType_op> new_weight;
            new_weight.re_alloc(weight_shape);
            new_weight.copy_from(*(param.conv_param.weight()));
            OpDataType *weight_data = new_weight.mutable_data();

            int weight_size = param.conv_param.weight()->shape().count();
            trans_weights_host.re_alloc({weight_size, 1, 1, 1});
            OpDataType* _host_work_space;
            _host_work_space = trans_weights_host.mutable_data();

            transpose_filter_KCRS_2_CRSK(weight_data, _host_work_space, \
                                         param.conv_param.weight()->num(), \
                                         param.conv_param.weight()->channel(), \
                                         _kernel_height, \
                                         _kernel_width);

            param.conv_param.mutable_weight()->re_alloc({weight_size, 1, 1, 1});
            param.conv_param.mutable_weight()->copy_from(trans_weights_host);

        }
    }

private:
    int _kernel_height;
    int _kernel_width;

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
      cudaStream_t)> dispatch_func;

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
      EltwiseType elt_type,
      cudaStream_t)> dispatch_func_elt;

    bool _use_k1s1p0;
    void conv_gemm_k1s1p0(float* out, const float* img,
                          const float* weights, int out_channel,
                          int in_channel, int img_h, int img_w,
                          const float* bias, cudaStream_t cuda_stream) {
        float alpha = 1.0f;
        float beta = 0.0f;
        int m = out_channel;
        int k = in_channel;
        int n = img_h * img_w;
        if (ifVec(m, n, k, k, n, n)) {
            ker_gemm_32x32x32_NN_vec_bias_relu(m, n, k,
                                           alpha, weights,
                                           beta, img,
                                           out, bias,
                                           cuda_stream);
        } else {
            ker_gemm_32x32x32_NN_bias_relu(m, n, k,
                                           alpha, weights,
                                           beta, img,
                                           out, bias,
                                           cuda_stream);
        }
    }

};
template class SaberConv2DAct<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}

}


#endif //ANAKIN_SABER_FUNCS_IMPL_CUDA_SABER_CONV2DACT_H
