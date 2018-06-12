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

#ifndef ANAKIN_OPERATOR_CONV_RELU_H
#define ANAKIN_OPERATOR_CONV_RELU_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/conv_act.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class ConvReluHelper;

/// pooling op
/**
 * \brief ConvRelu implementation class
 * public inherit Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class ConvRelu : public Operator<Ttype, Dtype, Ptype> {
public:
    ConvRelu() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator power<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class ConvReluHelper<Ttype, Dtype, Ptype>;
};
#define INSTANCE_CONVRELU(Ttype, Dtype, Ptype) \
template<> \
void ConvRelu<Ttype, Dtype, Ptype>::operator()(\
    OpContext<Ttype>& ctx,\
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,\
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {\
    auto* impl =\
        static_cast<ConvReluHelper<Ttype, Dtype, Ptype>*>(this->_helper);\
    auto& param = impl->_param_conv_relu;\
    impl->_funcs_conv_relu(ins, outs, param, ctx);\
}
/**
 * \brief ConvRelu helper class to implement it
 * public inherit OperatorHelper
 * including init resource and shape size in ConvRelu context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class ConvReluHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    ConvReluHelper()=default;

    ~ConvReluHelper() {}

    Status InitParam() override {
        DLOG(WARNING) << "Parsing ConvRelu op parameter.";
        saber::ConvParam<Tensor4d<Ttype, Dtype>> _conv_param;

        // get conv param
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
        DLOG(INFO) << "conv kernel_size : [" << kernel_size[0] << " " << kernel_size[1] << "]";
        DLOG(INFO) << "conv axis : " << axis;

        auto weights = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_1);

        if (bias_term) {
            auto bias = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_2);
            saber::ConvParam<Tensor4d<Ttype, Dtype>> conv_param(group, padding[0], padding[1],
                                                                strides[0], strides[1],
                                                                dilation_rate[0], dilation_rate[1],
                                                                &(weights.d_tensor()), &(bias.d_tensor()));
            _conv_param = conv_param;
        } else {
            Tensor4d<Ttype, Dtype>* bias = new Tensor4d<Ttype, Dtype>();;
            saber::ConvParam<Tensor4d<Ttype, Dtype>> conv_param(group, padding[0], padding[1],
                                                                strides[0], strides[1],
                                                                dilation_rate[0], dilation_rate[1],
                                                                &(weights.d_tensor()), bias);
            _conv_param = conv_param;
        }

        // get relu param
        auto alpha = GET_PARAMETER(float, relu_0_alpha);
        ActivationParam<Tensor4d<Ttype, Dtype>> active_param(Active_relu);//, alpha); // TEMP


        ConvActiveParam<Tensor4d<Ttype, Dtype>> conv_act_param(_conv_param, active_param);
        _param_conv_relu = conv_act_param;

        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for ConvRelu operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_conv_relu.init(ins, outs, _param_conv_relu, SPECIFY, SABER_IMPL, ctx));
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
        SABER_CHECK(_funcs_conv_relu.compute_output_shape(ins, outs, _param_conv_relu));
        return Status::OK();
    }

public:
    ///< _param_conv_relu stand for ConvRelu parameter
    saber::ConvActiveParam<Tensor4d<Ttype, Dtype>> _param_conv_relu;
    ///< _funcs_conv_relu stand for ConvRelu function 
    saber::ConvAct<Ttype, Dtype> _funcs_conv_relu;
};


#ifdef USE_CUDA
INSTANCE_CONVRELU(NV, AK_FLOAT, Precision::FP32);
template <>
Status ConvReluHelper<NV, AK_FLOAT, Precision::FP32>::Init(OpContext<NV>& ctx, \
        const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins, \
        std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    if (_param_conv_relu.conv_param.group == 1) {
        SABER_CHECK(_funcs_conv_relu.init(ins, outs, _param_conv_relu, SPECIFY, SABER_IMPL, ctx));
    } else {
        SABER_CHECK(_funcs_conv_relu.init(ins, outs, _param_conv_relu, SPECIFY, VENDER_IMPL, ctx));
    }
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(ConvRelu, ConvReluHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_CONVRELU(X86, AK_FLOAT, Precision::FP32);
template class ConvReluHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvRelu, ConvReluHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_CONVRELU(ARM, AK_FLOAT, Precision::FP32);
template class ConvReluHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvRelu, ConvReluHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(ConvRelu)
.Doc("ConvRelu operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("ConvRelu")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("ConvRelu")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("ConvRelu")
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
.Args<int>("axis", "axis of conv")
.Args<float>("relu_0_alpha", " alpha for relu");

} /* namespace ops */

} /* namespace anakin */

#endif
