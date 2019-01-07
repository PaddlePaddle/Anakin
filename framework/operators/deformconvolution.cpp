#include "framework/operators/deformconvolution.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void DeformConvolution<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV > >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<DeformConvolutionHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<DeformConvolutionHelper<NV, Precision::FP32>*>
                  (this->_helper)->_param_deform_conv;
    impl->_funcs_deform_conv(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
DeformConvolutionHelper<Ttype, Ptype>::~DeformConvolutionHelper() {
}

template<typename Ttype, Precision Ptype>
Status DeformConvolutionHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing DeformConvolution op parameter.";
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
        saber::DeformableConvParam<Ttype> deform_conv_param(group, padding[0], padding[1],
                strides[0], strides[1],
                dilation_rate[0], dilation_rate[1],
                &(weights.d_tensor()), &(bias.d_tensor()));
        _param_deform_conv = deform_conv_param;
    } else {
        Tensor4d<Ttype>* bias = new Tensor4d<Ttype>();;
        saber::DeformableConvParam<Ttype> deform_conv_param(group, padding[0], padding[1],
                strides[0], strides[1],
                dilation_rate[0], dilation_rate[1],
                &(weights.d_tensor()), bias);
        _param_deform_conv = deform_conv_param;
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status DeformConvolutionHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_deform_conv.init(ins, outs, _param_deform_conv, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status DeformConvolutionHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_deform_conv.compute_output_shape(ins, outs, _param_deform_conv));
    return Status::OK();
}

#ifdef USE_CUDA
template class DeformConvolutionHelper<NV, Precision::FP32>;
template class DeformConvolutionHelper<NV, Precision::FP16>;
template class DeformConvolutionHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class DeformConvolutionHelper<ARM, Precision::FP32>;
template class DeformConvolutionHelper<ARM, Precision::FP16>;
template class DeformConvolutionHelper<ARM, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(DeformConvolution, DeformConvolutionHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(DeformConvolution, DeformConvolutionHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(DeformConvolution)
.Doc("DeformConvolution operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("deformable_convolution")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("defromable_convolution")
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


