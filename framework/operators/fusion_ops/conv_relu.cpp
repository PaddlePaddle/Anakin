#include "framework/operators/fusion_ops/conv_relu.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void ConvRelu<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl =
        static_cast<ConvReluHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = impl->_param_conv_relu;
    impl->_funcs_conv_relu(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
ConvReluHelper<Ttype, Dtype, Ptype>::~ConvReluHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ConvReluHelper<Ttype, Dtype, Ptype>::InitParam() {
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

	using pblock_type = PBlock<typename DataTypeWarpper<Dtype>::type, Ttype>;
    auto weights = GET_PARAMETER(pblock_type, weight_1);

    if (bias_term) {
        auto bias = GET_PARAMETER(pblock_type, weight_2);
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

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ConvReluHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
#if 0

    if (_param_conv_relu.conv_param.group == ins[0]->channel() && \
            _param_conv_relu.conv_param.group == outs[0]->channel()) {
        _funcs_conv_relu.init(ins, outs, _param_conv_relu, SPECIFY, SABER_IMPL, ctx);
    } else {
        _funcs_conv_relu.init(ins, outs, _param_conv_relu, SPECIFY, VENDER_IMPL, ctx);
    }

#else

    if (_param_conv_relu.conv_param.group == 1) {
        _funcs_conv_relu.init(ins, outs, _param_conv_relu, SPECIFY, SABER_IMPL, ctx);
    } else {
        _funcs_conv_relu.init(ins, outs, _param_conv_relu, SPECIFY, VENDER_IMPL, ctx);
    }

#endif
    //_funcs_conv_relu.init(ins, outs, _param_conv_relu, SPECIFY, VENDER_IMPL, ctx);
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ConvReluHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    _funcs_conv_relu.compute_output_shape(ins, outs, _param_conv_relu);
    return Status::OK();
}

#ifdef USE_CUDA
template class ConvReluHelper<NV, AK_FLOAT, Precision::FP32>;
template class ConvReluHelper<NV, AK_FLOAT, Precision::FP16>;
template class ConvReluHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class ConvReluHelper<ARM, AK_FLOAT, Precision::FP32>;
template class ConvReluHelper<ARM, AK_FLOAT, Precision::FP16>;
template class ConvReluHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(ConvRelu, ConvReluHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(ConvRelu, ConvReluHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(ConvRelu)
.Doc("ConvRelu operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("power")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("power")
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


