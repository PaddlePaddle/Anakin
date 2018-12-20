#include "framework/operators/fusion_ops/eltwise_prelu.h"

namespace anakin {

namespace ops {

#define INSTANCE_ELTWISE_PRELU(Ttype, Ptype) \
template<> \
void EltwiseActivation<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<EltwiseActivationHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<EltwiseActivationHelper<Ttype, Ptype>*> \
                  (this->_helper)->_param_eltwise_prelu; \
    impl->_funcs_eltwise_prelu(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
EltwiseActivationHelper<Ttype, Ptype>::~EltwiseActivationHelper() {
}

template<typename Ttype, Precision Ptype>
Status EltwiseActivationHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing EltwiseActivation op parameter.";
    //FIND_PARAMETER(type);
    auto type = GET_PARAMETER(std::string, type);
   // auto alpha = GET_PARAMETER(float, relu_0_alpha);
    auto coeff = GET_PARAMETER(PTuple<float>, coeff);

    auto channel_shared = GET_PARAMETER(bool, prelu_0_channel_shared);
    //printf("channel_shared: %d \n", channel_shared);
    using pblock_type = PBlock<Ttype>;
    auto weights = GET_PARAMETER(pblock_type, prelu_0_weight_1);

    PreluParam<Ttype> prelu_param(channel_shared, &(weights.d_tensor()));
        
    ActivationParam<Ttype> activation_param(Active_prelu, 0, 0, prelu_param);

    EltwiseType elt_type;

    if (type == "Add") {
        elt_type = Eltwise_sum;
    } else if (type == "Max") {
        elt_type = Eltwise_max;
    } else {
        elt_type = Eltwise_prod;
    }

    //    Shape shape_coeff(1, 1, 1, coeff.size());
    //    Tensor<X86, Dtype> thcoeff(shape_coeff);
    //    for (int i = 0; i < thcoeff.size(); ++i) {
    //        thcoeff.mutable_data()[i] = coeff[i];
    //    }
    //    Tensor4d<Ttype, Dtype> * tdcoeff_p = new Tensor4d<Ttype, Dtype>();
    //    tdcoeff_p->re_alloc(shape_coeff);
    //    tdcoeff_p->copy_from(thcoeff);
    //
    //    saber::EltwiseParam<Tensor4d<Ttype, Dtype>>    eltwise_param(elt_type, tdcoeff_p);
    saber::EltwiseParam<Ttype>  eltwise_param(elt_type, coeff.vector());
    EltwiseActiveParam<Ttype> eltwise_prelu_param(eltwise_param, activation_param);
    _param_eltwise_prelu = eltwise_prelu_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status EltwiseActivationHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_eltwise_prelu.init(ins, outs, _param_eltwise_prelu, SPECIFY, SABER_IMPL, ctx);
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status EltwiseActivationHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_eltwise_prelu.compute_output_shape(ins, outs, _param_eltwise_prelu);
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ELTWISE_PRELU(NV, Precision::FP32);
template class EltwiseActivationHelper<NV, Precision::FP32>;
template class EltwiseActivationHelper<NV, Precision::FP16>;
template class EltwiseActivationHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ELTWISE_PRELU(ARM, Precision::FP32);
template class EltwiseActivationHelper<ARM, Precision::FP32>;
template class EltwiseActivationHelper<ARM, Precision::FP16>;
template class EltwiseActivationHelper<ARM, Precision::INT8>;
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
INSTANCE_ELTWISE_PRELU(X86, Precision::FP32);
template class EltwiseActivationHelper<X86, Precision::FP32>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(EltwiseActivation, EltwiseActivationHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(EltwiseActivation, EltwiseActivationHelper, ARM, Precision::FP32);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
ANAKIN_REGISTER_OP_HELPER(EltwiseActivation, EltwiseActivationHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(EltwiseActivation)
.Doc("EltwiseActivation operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("eltwise_prelu")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("eltwise_prelu")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, Precision::FP32>("eltwise_prelu")
#endif
#ifdef AMD_GPU
//.__alias__<AMD, Precision::FP32>("eltwise_prelu")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " eltwise type( string )")
.Args<PTuple<float>>("coeff", "coeff of eltwise")
.Args<bool>("channel_shared", "prelu channel is shared or not ");

} /* namespace ops */

} /* namespace anakin */


