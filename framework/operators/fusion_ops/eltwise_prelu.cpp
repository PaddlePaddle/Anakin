#include "framework/operators/fusion_ops/eltwise_prelu.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void EltwiseActivation<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl = static_cast<EltwiseActivationHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<EltwiseActivationHelper<NV, AK_FLOAT, Precision::FP32>*>
                  (this->_helper)->_param_eltwise_prelu;
    impl->_funcs_eltwise_prelu(ins, outs, param, ctx);
}
#endif
#ifdef USE_ARM_PLACE
template<>
void EltwiseActivation<ARM, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<ARM>& ctx,
    const std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& outs) {
    auto* impl = static_cast<EltwiseActivationHelper<ARM, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<EltwiseActivationHelper<ARM, AK_FLOAT, Precision::FP32>*>
                  (this->_helper)->_param_eltwise_prelu;
    impl->_funcs_eltwise_prelu(ins, outs, param, ctx);
}
#endif
#ifdef USE_X86_PLACE
template<>
void EltwiseActivation<X86, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<X86>& ctx,
    const std::vector<Tensor4dPtr<X86, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<X86, AK_FLOAT> >& outs) {
    auto* impl = static_cast<EltwiseActivationHelper<X86, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<EltwiseActivationHelper<X86, AK_FLOAT, Precision::FP32>*>
                  (this->_helper)->_param_eltwise_prelu;
    impl->_funcs_eltwise_prelu(ins, outs, param, ctx);
}
#endif
/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
EltwiseActivationHelper<Ttype, Dtype, Ptype>::~EltwiseActivationHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status EltwiseActivationHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing EltwiseActivation op parameter.";
    //FIND_PARAMETER(type);
    auto type = GET_PARAMETER(std::string, type);
   // auto alpha = GET_PARAMETER(float, relu_0_alpha);
    auto coeff = GET_PARAMETER(PTuple<float>, coeff);

    auto channel_shared = GET_PARAMETER(bool, prelu_0_channel_shared);
    //printf("channel_shared: %d \n", channel_shared);
    using pblock_type = PBlock<typename DataTypeWarpper<Dtype>::type, Ttype>;
    auto weights = GET_PARAMETER(pblock_type, prelu_0_weight_1);

    PreluParam<Tensor4d<Ttype, Dtype>> prelu_param(channel_shared, &(weights.d_tensor()));
        
    ActivationParam<Tensor4d<Ttype, Dtype>> activation_param(Active_prelu, 0, 0, prelu_param);

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
    saber::EltwiseParam<Tensor4d<Ttype, Dtype>>  eltwise_param(elt_type, coeff.vector());
    EltwiseActiveParam<Tensor4d<Ttype, Dtype>> eltwise_prelu_param(eltwise_param, activation_param);
    _param_eltwise_prelu = eltwise_prelu_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status EltwiseActivationHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    _funcs_eltwise_prelu.init(ins, outs, _param_eltwise_prelu, SPECIFY, SABER_IMPL, ctx);
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status EltwiseActivationHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    _funcs_eltwise_prelu.compute_output_shape(ins, outs, _param_eltwise_prelu);
    return Status::OK();
}

#ifdef USE_CUDA
template class EltwiseActivationHelper<NV, AK_FLOAT, Precision::FP32>;
template class EltwiseActivationHelper<NV, AK_FLOAT, Precision::FP16>;
template class EltwiseActivationHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class EltwiseActivationHelper<ARM, AK_FLOAT, Precision::FP32>;
template class EltwiseActivationHelper<ARM, AK_FLOAT, Precision::FP16>;
template class EltwiseActivationHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
template class EltwiseActivationHelper<X86, AK_FLOAT, Precision::FP32>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(EltwiseActivation, EltwiseActivationHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(EltwiseActivation, EltwiseActivationHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
ANAKIN_REGISTER_OP_HELPER(EltwiseActivation, EltwiseActivationHelper, X86, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(EltwiseActivation)
.Doc("EltwiseActivation operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("eltwise_prelu")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("eltwise_prelu")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, AK_FLOAT, Precision::FP32>("eltwise_prelu")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " eltwise type( string )")
.Args<PTuple<float>>("coeff", "coeff of eltwise")
.Args<bool>("channel_shared", "prelu channel is shared or not ");

} /* namespace ops */

} /* namespace anakin */


