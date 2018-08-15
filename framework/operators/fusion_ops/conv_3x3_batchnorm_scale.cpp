#include "framework/operators/fusion_ops/conv_3x3_batchnorm_scale.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void SassConvBatchnormScale<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl = static_cast<SassConvBatchnormScaleHelper<NV, AK_FLOAT, Precision::FP32>*>
                 (this->_helper);
    auto& param = static_cast<SassConvBatchnormScaleHelper<NV, AK_FLOAT, Precision::FP32>*>
                  (this->_helper)->_param_conv_batchnorm_scale_relu;
    impl->_funcs_conv_batchnorm_scale_relu(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
SassConvBatchnormScaleHelper<Ttype, Dtype, Ptype>::~SassConvBatchnormScaleHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SassConvBatchnormScaleHelper<Ttype, Dtype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing SassConvBatchnormScale op parameter.";
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


    // get batchnorm param
    auto epsilon = GET_PARAMETER(float, batchnorm_0_epsilon);
    auto momentum = GET_PARAMETER(float, batchnorm_0_momentum);
    auto batch_norm_weight_1 = GET_PARAMETER(pblock_type, batchnorm_0_weight_1);
    auto batch_norm_weight_1_vector = batch_norm_weight_1.vector();
    auto batch_norm_weight_2 = GET_PARAMETER(pblock_type, batchnorm_0_weight_2);
    auto batch_norm_weight_2_vector = batch_norm_weight_2.vector();
    auto batch_norm_weight_3 = GET_PARAMETER(pblock_type, batchnorm_0_weight_3);
    auto batch_norm_weight_3_vector = batch_norm_weight_3.vector();
    BatchnormParam<Tensor4d<Ttype, Dtype>> batchnorm_param(batch_norm_weight_1_vector,
                                        batch_norm_weight_2_vector,
                                        batch_norm_weight_3_vector[0],
                                        momentum, epsilon);
    // get scale param
    auto scale_num_axes = GET_PARAMETER(int, scale_0_num_axes);
    auto scale_bias_term = GET_PARAMETER(bool, scale_0_bias_term);
    auto scale_axis = GET_PARAMETER(int, scale_0_axis);
    auto scale_weight_1 = GET_PARAMETER(pblock_type, scale_0_weight_1);
    auto scale_weight_1_vector = scale_weight_1.vector();
    auto scale_weight_2 = GET_PARAMETER(pblock_type, scale_0_weight_2);
    auto  scale_weight_2_vector = scale_weight_2.vector();
    saber::ScaleParam<Tensor4d<Ttype, Dtype>> scale_param(scale_weight_1_vector,  scale_weight_2_vector,
                                           scale_bias_term, scale_axis, scale_num_axes);

    // get relu param
    //auto alpha = GET_PARAMETER(float, relu_0_alpha);
    //ActivationParam<Tensor4d<Ttype, Dtype>> active_param(Active_relu);//, alpha); // TEMP

	// check if conv has eltwise_relu op attr
	if(check_attr("merge_type")) {
		LOG(ERROR) << "detect eltwise relu!!!!!!!! ";
		auto type = GET_PARAMETER(std::string, merge_type);
    	auto alpha = GET_PARAMETER(float, merge_relu_0_alpha);
    	auto coeff = GET_PARAMETER(PTuple<float>, merge_coeff);
    	ActivationParam<Tensor4d<Ttype, Dtype>> activation_param(Active_relu);
    	EltwiseType elt_type;
    	if (type == "Add") {
        	elt_type = Eltwise_sum;
    	} else if (type == "Max") {
        	elt_type = Eltwise_max;
    	} else {
        	elt_type = Eltwise_prod;
    	}
    	saber::EltwiseParam<Tensor4d<Ttype, Dtype>>  eltwise_param(elt_type, coeff.vector());
    	EltwiseActiveParam<Tensor4d<Ttype, Dtype>> eltwise_relu_param(eltwise_param, activation_param);

		ConvActiveParam<Tensor4d<Ttype, Dtype>> conv_act_param(_conv_param, batchnorm_param, scale_param, eltwise_relu_param);
		_param_conv_batchnorm_scale_relu = conv_act_param;
	} else { 
		ConvActiveParam<Tensor4d<Ttype, Dtype>> conv_act_param(_conv_param, batchnorm_param, scale_param); 
		_param_conv_batchnorm_scale_relu = conv_act_param;
	}

    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SassConvBatchnormScaleHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    _funcs_conv_batchnorm_scale_relu.init(ins, outs, _param_conv_batchnorm_scale_relu, SPECIFY, SABER_IMPL, ctx);
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SassConvBatchnormScaleHelper<Ttype, Dtype, Ptype>::InferShape(
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    _funcs_conv_batchnorm_scale_relu.compute_output_shape(ins, outs, _param_conv_batchnorm_scale_relu);
    return Status::OK();
}

#ifdef USE_CUDA
template class SassConvBatchnormScaleHelper<NV, AK_FLOAT, Precision::FP32>;
template class SassConvBatchnormScaleHelper<NV, AK_FLOAT, Precision::FP16>;
template class SassConvBatchnormScaleHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class SassConvBatchnormScaleHelper<ARM, AK_FLOAT, Precision::FP32>;
template class SassConvBatchnormScaleHelper<ARM, AK_FLOAT, Precision::FP16>;
template class SassConvBatchnormScaleHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(SassConvBatchnormScale, SassConvBatchnormScaleHelper, NV,
                          AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(SassConvBatchnormScale, SassConvBatchnormScaleHelper, ARM,
                          AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(SassConvBatchnormScale)
.Doc("SassConvBatchnormScale fusion operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("convolution_batchnorm_scale_relu")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("convolution_batchnorm_scale_relu")
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
                .Args<float>("relu_0_alpha", " alpha for relu")
                .Args<int>("scale_0_num_axes", " num axes for scale")
                .Args<bool>("scale_0_bias_term", "whether scale has bias")
                .Args<int>("scale_0_axis", "axis for scale")
                .Args<float>("batchnorm_0_epsilon", "epsilon for batchnorm")
                .Args<float>("batchnorm_0_momentum", "momentum for batchnorm");

} /* namespace ops */

} /* namespace anakin */


