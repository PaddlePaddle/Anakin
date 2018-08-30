#include "framework/operators/fusion_ops/conv_3x3_relu.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void SassConvRelu<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<SassConvReluHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<SassConvReluHelper<NV, Precision::FP32>*>
                  (this->_helper)->_param_conv_relu;
    impl->_funcs_conv_relu(ins, outs, param, ctx);
}
#endif

#ifdef AMD_GPU
template<>
void SassConvRelu<AMD, Precision::FP32>::operator()(
    OpContext<AMD>& ctx,
    const std::vector<Tensor4dPtr<AMD> >& ins,
    std::vector<Tensor4dPtr<AMD> >& outs) {
    auto* impl = static_cast<SassConvReluHelper<AMD, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<SassConvReluHelper<AMD, Precision::FP32>*>
                  (this->_helper)->_param_conv_relu;
    impl->_funcs_conv_relu(ins, outs, param, ctx);
}
#endif
/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
SassConvReluHelper<Ttype, Ptype>::~SassConvReluHelper() {
}

template<typename Ttype, Precision Ptype>
Status SassConvReluHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing SassConvRelu op parameter.";

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
        _param_conv_relu = conv_param;
    } else {
        Tensor4d<Ttype>* bias = new Tensor4d<Ttype>();;
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                              strides[0], strides[1],
                                              dilation_rate[0], dilation_rate[1],
                                              &(weights.d_tensor()), bias,
                                              active_param);
        _param_conv_relu = conv_param;
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SassConvReluHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_conv_relu.init(ins, outs, _param_conv_relu, SPECIFY, SABER_IMPL, ctx);
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SassConvReluHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_conv_relu.compute_output_shape(ins, outs, _param_conv_relu);
    return Status::OK();
}

#ifdef USE_CUDA
template class SassConvReluHelper<NV, Precision::FP32>;
template class SassConvReluHelper<NV, Precision::FP16>;
template class SassConvReluHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class SassConvReluHelper<ARM, Precision::FP32>;
template class SassConvReluHelper<ARM, Precision::FP16>;
template class SassConvReluHelper<ARM, Precision::INT8>;
#endif

#ifdef AMD_GPU
template class SassConvReluHelper<AMD, Precision::FP32>;
template class SassConvReluHelper<AMD, Precision::FP16>;
template class SassConvReluHelper<AMD, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(SassConvRelu, SassConvReluHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(SassConvRelu, SassConvReluHelper, ARM, Precision::FP32);
#endif

#ifdef AMD_GPU
ANAKIN_REGISTER_OP_HELPER(SassConvRelu, SassConvReluHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(SassConvRelu)
.Doc("SassConvRelu fusion operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("convolution_batchnorm_scale_relu")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("convolution_batchnorm_scale_relu")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("convolution_batchnorm_scale_relu")
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


