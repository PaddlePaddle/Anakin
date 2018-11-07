#include "framework/operators/activation.h"

namespace anakin {

namespace ops {

#define INSTANCE_ACTIVATION(Ttype, Ptype) \
template<> \
void Activation<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<ActivationHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ActivationHelper<Ttype, Ptype>*>(this->_helper)->_param_activation; \
    impl->_funcs_activation(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
ActivationHelper<Ttype, Ptype>::~ActivationHelper() {
}

template<typename Ttype, Precision Ptype>
Status ActivationHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Activation op parameter.";
    auto type = GET_PARAMETER(std::string, type);

    if (type == "TanH") {
        ActivationParam<Ttype> param_activation(Active_tanh);
        _param_activation = param_activation;
    } else if (type == "Sigmoid") {
        ActivationParam<Ttype> param_activation(Active_sigmoid);
        _param_activation = param_activation;
    } else if (type == "PReLU") {
        auto channel_shared = GET_PARAMETER(bool, channel_shared);
        using pblock_type = PBlock<Ttype>;
        auto weights = GET_PARAMETER(pblock_type, weight_1);

        PreluParam<Ttype> prelu_param(channel_shared, &(weights.d_tensor()));

        ActivationParam<Ttype> param_activation(Active_prelu, 0, 0, prelu_param);
        _param_activation = param_activation;
    } else if (type == "Stanh") {
        ActivationParam<Ttype> param_activation(Active_stanh);
        _param_activation = param_activation;
    } else if (type == "Relu") {
        ActivationParam<Ttype> param_activation(Active_relu);
        _param_activation = param_activation;
    } else if (type == "ClippedRelu") {
        ActivationParam<Ttype> param_activation(Active_clipped_relu);
        _param_activation = param_activation;
    } else if (type == "Elu") {
        ActivationParam<Ttype> param_activation(Active_elu);
        _param_activation = param_activation;
    } else {
        LOG(FATAL) << "Other Activation type" << type << " should be replace by other ops.";
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ActivationHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_activation.init(ins, outs, _param_activation, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ActivationHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_activation.compute_output_shape(ins, outs, _param_activation));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ACTIVATION(NV, Precision::FP32);

template<>
Status ActivationHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx,
        const std::vector< Tensor4dPtr<NV> >& ins,
        std::vector< Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_activation.init(ins, outs, _param_activation, STATIC, VENDER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Activation, ActivationHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_ACTIVATION(X86, Precision::FP32);
INSTANCE_ACTIVATION(X86, Precision::FP16);
INSTANCE_ACTIVATION(X86, Precision::INT8);
template class ActivationHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Activation, ActivationHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ACTIVATION(ARM, Precision::FP32);
template class ActivationHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Activation, ActivationHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_ACTIVATION(AMD, Precision::FP32);
template class ActivationHelper<AMD, Precision::FP32>;
template class ActivationHelper<AMD, Precision::FP16>;
template class ActivationHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Activation, ActivationHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Activation)
.Doc("Activation operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("activation")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("activation")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("activation")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("activation")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " type of Activation ")
.Args<bool>("channel_shared", "prelu channel is shared or not ");

} /* namespace ops */

} /* namespace anakin */

