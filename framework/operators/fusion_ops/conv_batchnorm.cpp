#include "framework/operators/fusion_ops/conv_batchnorm.h"

namespace anakin {

namespace ops {

#define INSTANCE_CONVBATCHNORM(Ttype, Dtype, Ptype) \
template<> \
void ConvBatchnorm<Ttype, Dtype, Ptype>::operator()(\
    OpContext<Ttype>& ctx,\
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,\
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {\
    auto* impl = static_cast<ConvBatchnormHelper<Ttype, Dtype, Ptype>*>(this->_helper);\
    auto& param = static_cast<ConvBatchnormHelper<Ttype, Dtype, Ptype>*>\
                  (this->_helper)->_param_conv_batchnorm;\
    SABER_CHECK(impl->_funcs_conv_batchnorm(ins, outs, param, ctx));\
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ConvBatchnormHelper<Ttype, Dtype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing ConvBatchnorm op parameter.";
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

	ConvActiveParam<Tensor4d<Ttype, Dtype>> conv_act_param(_conv_param, batchnorm_param);
	_param_conv_batchnorm = conv_act_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ConvBatchnormHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_conv_batchnorm.init(ins, outs, \
        _param_conv_batchnorm, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ConvBatchnormHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_conv_batchnorm.compute_output_shape(ins, outs, \
        _param_conv_batchnorm));
    return Status::OK();
}

#ifdef USE_ARM_PLACE
INSTANCE_CONVBATCHNORM(ARM, AK_FLOAT, Precision::FP32);
template class ConvBatchnormHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvBatchnorm, ConvBatchnormHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_CUDA
INSTANCE_CONVBATCHNORM(NV, AK_FLOAT, Precision::FP32);
template<>
Status ConvBatchnormHelper<NV, AK_FLOAT, Precision::FP32>::Init(OpContext<NV>& ctx, \
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins, \
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    _funcs_conv_batchnorm.init(ins, outs, _param_conv_batchnorm, SPECIFY, VENDER_IMPL, ctx);
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(ConvBatchnorm, ConvBatchnormHelper, NV, AK_FLOAT,
                          Precision::FP32);
#endif
#if defined(BUILD_LITE)
INSTANCE_CONVBATCHNORM(X86, AK_FLOAT, Precision::FP32);
template class ConvBatchnormHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvBatchnorm, ConvBatchnormHelper, X86, AK_FLOAT,
                          Precision::FP32);
#endif


//! register op
ANAKIN_REGISTER_OP(ConvBatchnorm)
.Doc("ConvBatchnorm fusion operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("convolution_batchnorm_scale")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("convolution_batchnorm_scale")
#endif

#if defined(BUILD_LITE)
.__alias__<X86, AK_FLOAT, Precision::FP32>("convolution_batchnorm_scale")
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
                .Args<float>("batchnorm_0_epsilon", "epsilon for batchnorm")
                .Args<float>("batchnorm_0_momentum", "momentum for batchnorm");

} /* namespace ops */

} /* namespace anakin */


