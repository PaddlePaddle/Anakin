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

#ifndef ANAKIN_OPERATOR_DECONV_H
#define ANAKIN_OPERATOR_DECONV_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/deconv.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
class DeconvolutionHelper;

/// pooling op
/**
 * \brief Deconvolution operation class
 * public inheritance Operator
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class Deconvolution : public Operator<Ttype, Dtype, Ptype> {
public:
    Deconvolution() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
        LOG(ERROR) << "Not Impl Yet Operator convolution<TargetType:"<<"unknown"<<","
                   <<type_id<typename DataTypeWarpper<Dtype>::type>().type_info()<<">";
    }

    friend class DeconvolutionHelper<Ttype, Dtype, Ptype>;
};

#define INSTANCE_DECONV(Ttype, Dtype, Ptype) \
template<> \
void Deconvolution<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<DeconvolutionHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<DeconvolutionHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_deconv; \
    impl->_funcs_deconv(ins, outs, param, ctx); \
}
/**
 * \brief Deconvlution helper class 
 * public inherit OperatorHelper
 * including init resource and shape size in deconvolution context
 */
template<typename Ttype, DataType Dtype, Precision Ptype>
class DeconvolutionHelper : public OperatorHelper<Ttype, Dtype, Ptype> {
public:
    DeconvolutionHelper()=default;

    ~DeconvolutionHelper(){}

    Status InitParam() override {
        DLOG(WARNING) << "Parsing Deconvolution op parameter.";
        auto group = GET_PARAMETER(int, group);
        auto bias_term = GET_PARAMETER(bool, bias_term);
        auto padding = GET_PARAMETER(PTuple<int>, padding);
        auto strides = GET_PARAMETER(PTuple<int>, strides);
        auto dilation_rate = GET_PARAMETER(PTuple<int>, dilation_rate);
        auto filter_num = GET_PARAMETER(int, filter_num);
        auto kernel_size = GET_PARAMETER(PTuple<int>, kernel_size);
        auto axis = GET_PARAMETER(int, axis);

        auto weights = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_1);

        if (bias_term) {
            auto bias = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_2);
            saber::ConvParam<Tensor4d<Ttype, Dtype>> conv_param(group, padding[0], padding[1],
                                                                strides[0], strides[1],
                                                                dilation_rate[0], dilation_rate[1],
                                                                &(weights.d_tensor()), &(bias.d_tensor()));
            _param_deconv = conv_param;
        } else {
            Tensor4d<Ttype, Dtype>* bias = new Tensor4d<Ttype, Dtype>();;
            saber::ConvParam<Tensor4d<Ttype, Dtype>> conv_param(group, padding[0], padding[1],
                                                                strides[0], strides[1],
                                                                dilation_rate[0], dilation_rate[1],
                                                                &(weights.d_tensor()), bias);
            _param_deconv = conv_param;
        }

        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for deconvolution operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, 
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {

        SABER_CHECK(_funcs_deconv.init(ins, outs, _param_deconv, SPECIFY, SABER_IMPL, ctx));

        return Status::OK();
    }

    /**
    * \brief initial all the resource needed by pooling
    * \param ctx stand for Deconvolution operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                      std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) override {
        SABER_CHECK(_funcs_deconv.compute_output_shape(ins, outs, _param_deconv));
        return Status::OK();
    }

public:
    ///< _param_deconv stand for deconvolution parameter
    saber::ConvParam<Tensor4d<Ttype, Dtype>>  _param_deconv;
    ///< _funcs_deconv stand for deconvolution function
    saber::Deconv<Ttype, Dtype> _funcs_deconv;

private:
    ///< _dims stand for batchNorm size
    PTuple<int> _dims; 
};

#ifdef USE_CUDA
INSTANCE_DECONV(NV, AK_FLOAT, Precision::FP32);
template<>
Status DeconvolutionHelper<NV, AK_FLOAT, Precision::FP32>::Init(OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
        std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    bool p = true;
    p = p && (_param_deconv.weight()->width() == 4);
    p = p && (_param_deconv.weight()->height() == 4);
    p = p && (_param_deconv.pad_h == 1);
    p = p && (_param_deconv.pad_w == 1);
    p = p && (_param_deconv.stride_h == 2);
    p = p && (_param_deconv.stride_w == 2);
    p = p && (ins[0]->channel() <= 64);
    p = p && (ins[0]->width() % 64 == 0);
    p = p || ((ins[0]->channel() == _param_deconv.group)
              && (ins[0]->channel() == outs[0]->channel()));

    //p = p && (_param_deconv.group == 1);
    //    LOG(ERROR)<<"DECONV INIT";
    if (p) {
        //        LOG(ERROR)<<"using saber deconv";
        SABER_CHECK(_funcs_deconv.init(ins, outs, _param_deconv, SPECIFY, SABER_IMPL, ctx));
    } else {
        SABER_CHECK(_funcs_deconv.init(ins, outs, _param_deconv, SPECIFY, VENDER_IMPL/*SABER_IMPL*/, ctx));
    }

    return Status::OK();
}
template class DeconvolutionHelper<NV, AK_FLOAT, Precision::FP32>;
template class DeconvolutionHelper<NV, AK_FLOAT, Precision::FP16>;
template class DeconvolutionHelper<NV, AK_FLOAT, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Deconvolution, DeconvolutionHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_DECONV(ARM, AK_FLOAT, Precision::FP32);
template class DeconvolutionHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Deconvolution, DeconvolutionHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Deconvolution)
.Doc("Deconvolution operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("deconvolution")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("deconvolution")
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
