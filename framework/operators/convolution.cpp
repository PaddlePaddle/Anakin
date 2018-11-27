#include "framework/operators/convolution.h"

namespace anakin {

namespace ops {

#define INSTANCE_CONVOLUTION(Ttype, Ptype) \
template<> \
void Convolution<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<ConvolutionHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<ConvolutionHelper<Ttype, Ptype>*> \
                  (this->_helper)->_param_conv; \
    impl->_funcs_conv(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status ConvolutionHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Convolution op parameter.";
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
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                              strides[0], strides[1],
                                              dilation_rate[0], dilation_rate[1],
                                              &(weights.d_tensor()), &(bias.d_tensor()));
        _param_conv = conv_param;
    } else {
        Tensor4d<Ttype>* bias = new Tensor4d<Ttype>();;
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                              strides[0], strides[1],
                                              dilation_rate[0], dilation_rate[1],
                                              &(weights.d_tensor()), bias);
        _param_conv = conv_param;
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConvolutionHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    auto group = GET_PARAMETER(int, group);
    auto strides = GET_PARAMETER(PTuple<int>, strides);
    auto weights = GET_PARAMETER(PBlock<Ttype>, weight_1);
    SABER_CHECK(_funcs_conv.init(ins, outs, _param_conv, SPECIFY, SABER_IMPL, ctx));

    // check if weights have been transposed
    auto is_weights_transed = CHECK_PARAMETER(is_weights_transed);
    if(!is_weights_transed) {
        SET_PARAMETER(is_weights_transed, true, bool);
        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                                    std::bind(&Conv<Ttype, PrecisionWrapper<Ptype>::saber_type>::trans_weights, 
                                    &_funcs_conv, _1, _2, _3, _4, _5),
                                    weights.d_tensor(), 
                                    strides[0], strides[1], 
                                    group, 
                                    SABER_IMPL);
        weights.map_to_host();
    } else {
        PBlock<Ttype> weight_empty;
        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                                    std::bind(&Conv<Ttype, PrecisionWrapper<Ptype>::saber_type>::trans_weights, 
                                    &_funcs_conv, _1, _2, _3, _4, _5),
                                    weight_empty.d_tensor(), 
                                    strides[0], strides[1], 
                                    group, 
                                    SABER_IMPL);
    }
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConvolutionHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_conv.compute_output_shape(ins, outs, _param_conv));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_CONVOLUTION(NV, Precision::FP32);
template <>
Status ConvolutionHelper<NV, Precision ::FP32>::Init(OpContext<NV> &ctx, \
    const std::vector<Tensor4dPtr<NV> >& ins,
                    std::vector<Tensor4dPtr<NV> >& outs) {
    auto group = GET_PARAMETER(int, group);
    auto strides = GET_PARAMETER(PTuple<int>, strides);
    auto weights = GET_PARAMETER(PBlock<NV>, weight_1);
    SABER_CHECK(_funcs_conv.init(ins, outs, _param_conv, SPECIFY, VENDER_IMPL, ctx));

    auto is_weights_transed = CHECK_PARAMETER(is_weights_transed);
    if(!is_weights_transed) {
        SET_PARAMETER(is_weights_transed, true, bool);
        graph::GraphGlobalMem<NV>::Global().template apply<Level_0>(std::bind(
                                   &Conv<NV, PrecisionWrapper<Precision ::FP32>::saber_type>::trans_weights, 
                                   &_funcs_conv, _1, _2, _3, _4, _5),
                                   weights.d_tensor(), 
                                   strides[0], strides[1], 
                                   group, 
                                   VENDER_IMPL);
        weights.map_to_host();
    }
    return Status::OK();
}

template class ConvolutionHelper<NV, Precision::FP32>;
template class ConvolutionHelper<NV, Precision::FP16>;
template class ConvolutionHelper<NV, Precision::INT8>;

ANAKIN_REGISTER_OP_HELPER(Convolution, ConvolutionHelper, NV, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_CONVOLUTION(X86, Precision::FP32);
template class ConvolutionHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Convolution, ConvolutionHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_CONVOLUTION(ARM, Precision::FP32);
template class ConvolutionHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Convolution, ConvolutionHelper, ARM, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_CONVOLUTION(AMD, Precision::FP32);
template class ConvolutionHelper<AMD, Precision::FP32>;
template class ConvolutionHelper<AMD, Precision::FP16>;
template class ConvolutionHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Convolution, ConvolutionHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Convolution)
.Doc("Convolution operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("convolution")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("convolution")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("convolution")
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


