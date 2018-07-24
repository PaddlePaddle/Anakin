#include "framework/operators/conv_3x3.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void SassConvolution<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<SassConvolutionHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<SassConvolutionHelper<NV, Precision::FP32>*>
                  (this->_helper)->_param_conv;
    impl->_funcs_conv(ins, outs, param, ctx);
}
#endif

#ifdef USE_AMD
template<>
void SassConvolution<AMD, Precision::FP32>::operator()(
    OpContext<AMD>& ctx,
    const std::vector<Tensor4dPtr<AMD> >& ins,
    std::vector<Tensor4dPtr<AMD> >& outs) {
    auto* impl = static_cast<SassConvolutionHelper<AMD, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<SassConvolutionHelper<AMD, Precision::FP32>*>
                  (this->_helper)->_param_conv;
    impl->_funcs_conv(ins, outs, param, ctx);
}
#endif
/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
SassConvolutionHelper<Ttype, Ptype>::~SassConvolutionHelper() {
}

template<typename Ttype, Precision Ptype>
Status SassConvolutionHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing SassConvolution op parameter.";
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

    if (bias_term) {
        auto bias = GET_PARAMETER(pblock_type, weight_2);
        saber::ConvParam<Tensor4d<Ttype>> conv_param(group, padding[0], padding[1],
                                              strides[0], strides[1],
                                              dilation_rate[0], dilation_rate[1],
                                              &(weights.d_tensor()), &(bias.d_tensor()));
        _param_conv = conv_param;
    } else {
        Tensor4d<Ttype>* bias = new Tensor4d<Ttype>();;
        saber::ConvParam<Tensor4d<Ttype>> conv_param(group, padding[0], padding[1],
                                              strides[0], strides[1],
                                              dilation_rate[0], dilation_rate[1],
                                              &(weights.d_tensor()), bias);
        _param_conv = conv_param;
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SassConvolutionHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_conv.init(ins, outs, _param_conv, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SassConvolutionHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_conv.compute_output_shape(ins, outs, _param_conv));
    return Status::OK();
}

#ifdef USE_CUDA
template class SassConvolutionHelper<NV, Precision::FP32>;
template class SassConvolutionHelper<NV, Precision::FP16>;
template class SassConvolutionHelper<NV, Precision::INT8>;
#endif

//#ifdef USE_ARM_PLACE
//template class SassConvolutionHelper<ARM, Precision::FP32>;
//template class SassConvolutionHelper<ARM, Precision::FP16>;
//template class SassConvolutionHelper<ARM, Precision::INT8>;
//#endif

#ifdef USE_AMD
template class SassConvolutionHelper<AMD, Precision::FP32>;
template class SassConvolutionHelper<AMD, Precision::FP16>;
template class SassConvolutionHelper<AMD, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(SassConvolution, SassConvolutionHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
//ANAKIN_REGISTER_OP_HELPER(SassConvolution, SassConvolutionHelper, ARM, Precision::FP32);
#endif

#ifdef USE_AMD
ANAKIN_REGISTER_OP_HELPER(SassConvolution, SassConvolutionHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(SassConvolution)
.Doc("SassConvolution operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("convolution")
#endif
#ifdef USE_AMD
.__alias__<AMD, Precision::FP32>("convolution")
#endif
//#ifdef USE_ARM_PLACE
//.__alias__<ARM, Precision::FP32>("convolution")
//#endif
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


