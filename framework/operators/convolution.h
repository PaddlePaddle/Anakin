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

#ifndef ANAKIN_OPERATOR_CONV_H
#define ANAKIN_OPERATOR_CONV_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/conv.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class ConvolutionHelper;

/// pooling op
/**
 * \brief convlution operation class
 * public inheritance Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Convolution : public Operator<Ttype, Dtype, Ptype> {
public:
    Convolution() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator convolution<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
        //auto* impl = static_cast<ConvolutionHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
        auto& param = static_cast<ConvolutionHelper<Ttype, Dtype, Ptype>*> \
                  (this->_helper)->_param_conv; \
        impl->_funcs_conv(ins, outs, param, ctx);
    }

    friend class ConvolutionHelper<Ttype, Dtype, Ptype>;
};

#define INSTANCE_CONVOLUTION(Ttype, Dtype, Ptype) \
template<> \
void Convolution<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<ConvolutionHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<ConvolutionHelper<Ttype, Dtype, Ptype>*> \
                  (this->_helper)->_param_conv; \
    impl->_funcs_conv(ins, outs, param, ctx); \
}

/**
 * \brief convlution helper class 
 * public inherit OperatorHelper
 * including init resource and shape size in convolution context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class ConvolutionHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    ConvolutionHelper()=default;

    ~ConvolutionHelper(){}

    Status InitParam() override {
                DLOG(WARNING) << "Parsing Convolution op parameter.";
        auto group = GET_PARAMETER(int, group);
        auto bias_term = GET_PARAMETER(bool, bias_term);
        auto padding = GET_PARAMETER(PTuple<int>, padding);
        auto strides = GET_PARAMETER(PTuple<int>, strides);
        auto dilation_rate = GET_PARAMETER(PTuple<int>, dilation_rate);
        auto filter_num = GET_PARAMETER(int, filter_num);
        auto kernel_size = GET_PARAMETER(PTuple<int>, kernel_size);
        auto axis = GET_PARAMETER(int, axis);
                DLOG(INFO) << "conv group : " << group;
                DLOG(INFO) << "conv bias_term: " << bias_term;
                DLOG(INFO) << "conv padding : [" << padding[0] << " " << padding[1] << "]";
                DLOG(INFO) << "conv strides : [" << strides[0] << " " << strides[1] << "]";
                DLOG(INFO) << "conv dilation_rate : [" << dilation_rate[0] << " " << dilation_rate[1] << "]";
                DLOG(INFO) << "conv filter_num : " << filter_num;
                DLOG(INFO) << "conv kernel_size : " << kernel_size[0] << " " << kernel_size[1] << "]";
                DLOG(INFO) << "conv axis : " << axis;


        auto weights = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_1);

        if (bias_term) {
            auto bias = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_2);
            saber::ConvParam<Tensor4d<Ttype, Dtype>> conv_param(group, padding[0], padding[1],
                                                                strides[0], strides[1],
                                                                dilation_rate[0], dilation_rate[1],
                                                                &(weights.d_tensor()), &(bias.d_tensor()));
            _param_conv = conv_param;
        } else {
            Tensor4d<Ttype, Dtype>* bias = new Tensor4d<Ttype, Dtype>();;
            saber::ConvParam<Tensor4d<Ttype, Dtype>> conv_param(group, padding[0], padding[1],
                                                                strides[0], strides[1],
                                                                dilation_rate[0], dilation_rate[1],
                                                                &(weights.d_tensor()), bias);
            _param_conv = conv_param;
        }

        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for convolution operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_conv.init(ins, outs, _param_conv, SPECIFY, VENDER_IMPL, ctx));
        return Status::OK();
    }

    /**
    * \brief infer the shape of output and input.
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_conv.compute_output_shape(ins, outs, _param_conv));
        return Status::OK();
    }

public:
    ///< _param_conv stand for convolution parameter               
    saber::ConvParam<Tensor4d<Ttype, Dtype>>  _param_conv;
    ///< _funcs_conv stand for convolution function
    saber::Conv<Ttype, Dtype> _funcs_conv;

private:
    ///< _dims stand for Convolution size
    PTuple<int> _dims; 
};

#ifdef USE_CUDA
INSTANCE_CONVOLUTION(NV, AK_FLOAT, Precision::FP32);
template class ConvolutionHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Convolution, ConvolutionHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_CONVOLUTION(X86, AK_FLOAT, Precision::FP32);
template class ConvolutionHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Convolution, ConvolutionHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_CONVOLUTION(ARM, AK_FLOAT, Precision::FP32);
template <>
Status ConvolutionHelper<ARM, AK_FLOAT, Precision ::FP32>::Init(OpContext<ARM> &ctx, \
    const std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& ins,
                    std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_conv.init(ins, outs, _param_conv, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

ANAKIN_REGISTER_OP_HELPER(Convolution, ConvolutionHelper, ARM, AK_FLOAT, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Convolution)
.Doc("Convolution operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("convolution")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("convolution")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("convolution")
#endif
.num_in(1)
.num_out(1)
.Args<int>("group", " group of conv ")
.Args<bool>("bias_term", " whether conv weights have bias")
.Args<PTuple<int>>("padding", "padding of conv (x, y)")
.Args<PTuple<int>>("strides", "strides of conv (x)")
.Args<PTuple<int>>("dilation_rate", "dilation rate of conv (x)")
.Args<int>("filter_num", "filter(kernel) number of weights")
.Args<PTuple<int>>("kernel_size", "kernel size of kernel (x, y)")
.Args<int>("axis", "axis of conv");

} /* namespace ops */

} /* namespace anakin */

#endif
