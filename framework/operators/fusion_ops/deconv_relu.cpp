#include "framework/operators/fusion_ops/deconv_relu.h"

namespace anakin {

namespace ops {

#define INSTANCE_DECONVRELU(Ttype, Ptype) \
template<> \
void DeconvRelu<Ttype, Ptype>::operator()(\
    OpContext<Ttype>& ctx,\
    const std::vector<Tensor4dPtr<Ttype> >& ins,\
    std::vector<Tensor4dPtr<Ttype> >& outs) {\
    auto* impl =\
        static_cast<DeconvReluHelper<Ttype, Ptype>*>(this->_helper);\
    auto& param = impl->_param_deconv_relu;\
    impl->_funcs_deconv_relu(ins, outs, param, ctx);\
}


/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
DeconvReluHelper<Ttype, Ptype>::~DeconvReluHelper() {
}

template<typename Ttype, Precision Ptype>
Status DeconvReluHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing DeconvRelu op parameter.";

    // get conv param
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
    
    // get relu param
    auto alpha = GET_PARAMETER(float, relu_0_alpha);
    ActivationParam<Ttype> active_param(Active_relu);//, alpha); // TEMP

    if (bias_term) {
        auto bias = GET_PARAMETER(pblock_type, weight_2);
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                              strides[0], strides[1],
                                              dilation_rate[0], dilation_rate[1],
                                              &(weights.d_tensor()), &(bias.d_tensor()),
                                              active_param);
        _param_deconv_relu = conv_param;
    } else {
        Tensor4d<Ttype>* bias = new Tensor4d<Ttype>();;
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                              strides[0], strides[1],
                                              dilation_rate[0], dilation_rate[1],
                                              &(weights.d_tensor()), bias,
                                              active_param);
        _param_deconv_relu = conv_param;
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status DeconvReluHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    bool p = true;
    p = p && (_param_deconv_relu.weight()->width() == 4);
    p = p && (_param_deconv_relu.weight()->height() == 4);
    p = p && (_param_deconv_relu.pad_h == 1);
    p = p && (_param_deconv_relu.pad_w == 1);
    p = p && (_param_deconv_relu.stride_h == 2);
    p = p && (_param_deconv_relu.stride_w == 2);
    p = p && (ins[0]->channel() <= 64);
    p = p && (ins[0]->width() % 32 == 0);
    p = p || ((ins[0]->channel() == _param_deconv_relu.group)
              && (ins[0]->channel() == outs[0]->channel()));

    if (std::is_same<Ttype, X86>::value) {
        p = true;
    }

    //    LOG(ERROR)<<"DECONV RELU INIT";
    if (p) {
        //                LOG(ERROR)<<"DECONV RELU SELECTED";
        _funcs_deconv_relu.init(ins, outs, _param_deconv_relu, SPECIFY, SABER_IMPL, ctx);
    } else {
        _funcs_deconv_relu.init(ins, outs, _param_deconv_relu, SPECIFY, VENDER_IMPL, ctx);
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status DeconvReluHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_deconv_relu.compute_output_shape(ins, outs, _param_deconv_relu);
    return Status::OK();
}

#ifdef USE_X86_PLACE
INSTANCE_DECONVRELU(X86, Precision::FP32)
ANAKIN_REGISTER_OP_HELPER(DeconvRelu, DeconvReluHelper, X86, Precision::FP32);
#endif

#ifdef USE_CUDA
INSTANCE_DECONVRELU(NV, Precision::FP32)
template class DeconvReluHelper<NV, Precision::FP32>;
template class DeconvReluHelper<NV, Precision::FP16>;
template class DeconvReluHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class DeconvReluHelper<ARM, Precision::FP32>;
template class DeconvReluHelper<ARM, Precision::FP16>;
template class DeconvReluHelper<ARM, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(DeconvRelu, DeconvReluHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(DeconvRelu, DeconvReluHelper, ARM, Precision::FP32);
#endif
#ifdef BUILD_LITE
INSTANCE_DECONVRELU(X86, Precision::FP32)
template class DeconvReluHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(DeconvRelu, DeconvReluHelper, X86, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(DeconvRelu)
.Doc("DeconvRelu operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("deconv_relu")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("deconv_relu")
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


