#include "framework/operators/fusion_ops/conv_act.h"

namespace anakin {

namespace ops {

#define INSTANCE_CONVACT(Ttype, Ptype) \
template<> \
void ConvAct<Ttype, Ptype>::operator()(\
    OpContext<Ttype>& ctx,\
    const std::vector<Tensor4dPtr<Ttype> >& ins,\
    std::vector<Tensor4dPtr<Ttype> >& outs) {\
    auto* impl =\
        static_cast<ConvActHelper<Ttype, Ptype>*>(this->_helper);\
    auto& param = impl->_param_conv_act;\
    impl->_funcs_conv_act(ins, outs, param, ctx);\
}

template<typename Ttype, Precision Ptype>
Status ConvActHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing ConvAct op parameter.";
    saber::ConvParam<Ttype> _conv_param;
    saber::ActivationParam<Ttype> _act_param;

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
    // resize weights scale
    auto& w = weights.h_tensor();
    if (w.get_scale().size() == 1){
        float scale_tmp = w.get_scale()[0];
        std::vector<float> w_scale(filter_num, scale_tmp);
        w.set_scale(w_scale);
    }

    // get act param
    ActivationParam<Ttype> param_act;
    auto type = GET_PARAMETER(std::string, act_0_type);
     if (type == "TanH") {
        ActivationParam<Ttype> param_activation(Active_tanh);
        param_act = param_activation;
    } else if (type == "Sigmoid") {
        ActivationParam<Ttype> param_activation(Active_sigmoid);
        param_act = param_activation;
    } else if (type == "PReLU") {
        auto channel_shared = GET_PARAMETER(bool, channel_shared);
        using pblock_type = PBlock<Ttype>;
        auto weights = GET_PARAMETER(pblock_type, weight_1);

        PreluParam<Ttype> prelu_param(channel_shared, &(weights.d_tensor()));
        
        ActivationParam<Ttype> param_activation(Active_prelu, 0, 0, prelu_param);
        param_act = param_activation;
    } else if (type == "Stanh") {
        ActivationParam<Ttype> param_activation(Active_stanh);
        param_act = param_activation;
    } else if (type == "Relu") {
         auto alpha = GET_PARAMETER(float, relu_0_alpha);
         ActivationParam<Ttype> param_activation(Active_relu, alpha);
         param_act = param_activation;
    } else if (type == "ClippedRelu") {
         ActivationParam<Ttype> param_activation(Active_clipped_relu);
         param_act = param_activation;
    } else if (type == "Elu") {
         ActivationParam<Ttype> param_activation(Active_elu);
         param_act = param_activation;
    } else {
        LOG(FATAL) << "Other Activation type" << type << " should be replace by other ops.";
    }

    if (bias_term) {
        auto bias = GET_PARAMETER(pblock_type, weight_2);
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                strides[0], strides[1], dilation_rate[0], dilation_rate[1],
                &(weights.d_tensor()), &(bias.d_tensor()), param_act);
        _param_conv_act = conv_param;
    } else {
        Tensor4d<Ttype>* bias = new Tensor4d<Ttype>();;
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                strides[0], strides[1], dilation_rate[0], dilation_rate[1],
                &(weights.d_tensor()), bias, param_act);
        _param_conv_act = conv_param;
    }

    return Status::OK();

}

template<typename Ttype, Precision Ptype>
Status ConvActHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    SABER_CHECK(_funcs_conv_act.init(ins, outs, _param_conv_act, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConvActHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_conv_act.compute_output_shape(ins, outs, _param_conv_act);
    return Status::OK();
}

#if defined(BUILD_LITE)
INSTANCE_CONVACT(X86, Precision::FP32);
template class ConvActHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ConvAct, ConvActHelper, X86, Precision::FP32);
#endif


//! register op
ANAKIN_REGISTER_OP(ConvAct)
.Doc("ConvAct operator")
#if defined(BUILD_LITE)
.__alias__<X86, Precision::FP32>("conv_act")
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
.Args<bool>("channel_shared", "prelu channel is shared or not ");

} /* namespace ops */

} /* namespace anakin */


