#include "framework/operators/activation.h"

namespace anakin {

namespace ops {

#define INSTANCE_ACTIVATION(Ttype, Dtype, Ptype) \
template<> \
void Activation<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<ActivationHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ActivationHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_activation; \
    impl->_funcs_activation(ins, outs, param, ctx); \
}

/// TODO ... specialization other type of operator

/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
ActivationHelper<Ttype, Dtype, Ptype>::~ActivationHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ActivationHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Activation op parameter.";
    auto type = GET_PARAMETER(std::string, type);
    if (type == "TanH") {
        ActivationParam<Tensor4d<Ttype, Dtype>> param_activation(Active_tanh);
        _param_activation = param_activation;
    } else if (type == "Sigmoid") {
        ActivationParam<Tensor4d<Ttype, Dtype>> param_activation(Active_sigmoid);
        _param_activation = param_activation;
    } else if (type == "Stanh") {
        ActivationParam<Tensor4d<Ttype, Dtype>> param_activation(Active_stanh);
        _param_activation = param_activation;
    }else if (type == "Relu") {
        ActivationParam<Tensor4d<Ttype, Dtype>> param_activation(Active_relu);
        _param_activation = param_activation;
    }else if (type == "ClippedRelu") {
        ActivationParam<Tensor4d<Ttype, Dtype>> param_activation(Active_clipped_relu);
        _param_activation = param_activation;
    }else if (type == "Elu") {
        ActivationParam<Tensor4d<Ttype, Dtype>> param_activation(Active_elu);
        _param_activation = param_activation;
    }else {
        LOG(FATAL) << "Other Activation type" << type << " should be replace by other ops.";
    }

    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ActivationHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
                                                   const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                                                   std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_activation.init(ins, outs, _param_activation, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ActivationHelper<Ttype, Dtype, Ptype>::InferShape(const
                                                         std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                                                         std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_activation.compute_output_shape(ins, outs, _param_activation));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ACTIVATION(NV, AK_FLOAT, Precision::FP32);
INSTANCE_ACTIVATION(NV, AK_FLOAT, Precision::FP16);
INSTANCE_ACTIVATION(NV, AK_FLOAT, Precision::INT8);
template class ActivationHelper<NV, AK_FLOAT, Precision::FP32>;
template class ActivationHelper<NV, AK_FLOAT, Precision::FP16>;
template class ActivationHelper<NV, AK_FLOAT, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Activation, ActivationHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_ACTIVATION(X86, AK_FLOAT, Precision::FP32);
INSTANCE_ACTIVATION(X86, AK_FLOAT, Precision::FP16);
INSTANCE_ACTIVATION(X86, AK_FLOAT, Precision::INT8);
template class ActivationHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Activation, ActivationHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ACTIVATION(ARM, AK_FLOAT, Precision::FP32);
template class ActivationHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Activation, ActivationHelper, ARM, AK_FLOAT, Precision::FP32);
#endif//arm

//! register op
ANAKIN_REGISTER_OP(Activation)
.Doc("Activation operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("activation")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("activation")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("activation")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " type of Activation ");

} /* namespace ops */

} /* namespace anakin */

