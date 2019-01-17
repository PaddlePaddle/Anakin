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
#include "framework/operators/deconvolution.h"

namespace anakin {

namespace ops {

#define INSTANCE_DECONV(Ttype, Ptype) \
template<> \
void Deconvolution<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
        std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<DeconvolutionHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<DeconvolutionHelper<Ttype, Ptype>*>(this->_helper)->_param_deconv; \
    impl->_funcs_deconv(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status DeconvolutionHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Deconvolution op parameter.";
    auto group = GET_PARAMETER(int, group);
    auto bias_term = GET_PARAMETER(bool, bias_term);
    auto padding = GET_PARAMETER(PTuple<int>, padding);
    auto strides = GET_PARAMETER(PTuple<int>, strides);
    auto dilation_rate = GET_PARAMETER(PTuple<int>, dilation_rate);
    auto filter_num = GET_PARAMETER(int, filter_num);
    auto kernel_size = GET_PARAMETER(PTuple<int>, kernel_size);
    auto axis = GET_PARAMETER(int, axis);

	using pblock_type = PBlock<Ttype>;
    auto weights = GET_PARAMETER(pblock_type, weight_1);
    // fixme, resize deconv weights scale

    if (bias_term) {
        auto bias = GET_PARAMETER(pblock_type, weight_2);
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                              strides[0], strides[1],
                                              dilation_rate[0], dilation_rate[1],
                                              &(weights.d_tensor()), &(bias.d_tensor()));
        _param_deconv = conv_param;
    } else {
        Tensor4d<Ttype>* bias = new Tensor4d<Ttype>();;
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                              strides[0], strides[1],
                                              dilation_rate[0], dilation_rate[1],
                                              &(weights.d_tensor()), bias);
        _param_deconv = conv_param;
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status DeconvolutionHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    if (std::is_same<Ttype,X86>::value){
        SABER_CHECK(_funcs_deconv.init(ins, outs, _param_deconv, SPECIFY, SABER_IMPL, ctx));
        return Status::OK();
    }
    SABER_CHECK(_funcs_deconv.init(ins, outs, _param_deconv, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status DeconvolutionHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_deconv.compute_output_shape(ins, outs, _param_deconv));
    return Status::OK();
}
#ifdef AMD_GPU
INSTANCE_DECONV(AMD, Precision::FP32);
template<>
Status DeconvolutionHelper<AMD, Precision::FP32>::Init(OpContext<AMD>& ctx,
        const std::vector<Tensor4dPtr<AMD> >& ins,
        std::vector<Tensor4dPtr<AMD >>& outs) {
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
template class DeconvolutionHelper<AMD, Precision::FP32>;
template class DeconvolutionHelper<AMD, Precision::FP16>;
template class DeconvolutionHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Deconvolution, DeconvolutionHelper, AMD, Precision::FP32);
#endif


#ifdef USE_CUDA
INSTANCE_DECONV(NV, Precision::FP32);
template<>
Status DeconvolutionHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV> >& ins,
        std::vector<Tensor4dPtr<NV >>& outs) {
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
template class DeconvolutionHelper<NV, Precision::FP32>;
template class DeconvolutionHelper<NV, Precision::FP16>;
template class DeconvolutionHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Deconvolution, DeconvolutionHelper, NV, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_DECONV(X86, Precision::FP32);
template class DeconvolutionHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Deconvolution, DeconvolutionHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_DECONV(ARM, Precision::FP32);
template class DeconvolutionHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Deconvolution, DeconvolutionHelper, ARM, Precision::FP32);
#endif

#if defined BUILD_LITE
INSTANCE_DECONV(X86, Precision::FP32);
template class DeconvolutionHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Deconvolution, DeconvolutionHelper, X86, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Deconvolution)
.Doc("Deconvolution operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("deconvolution")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("deconvolution")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("deconvolution")
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


