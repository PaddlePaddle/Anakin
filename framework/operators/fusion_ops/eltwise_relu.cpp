#include "framework/operators/fusion_ops/eltwise_relu.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void EltwiseRelu<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl = static_cast<EltwiseReluHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<EltwiseReluHelper<NV, AK_FLOAT, Precision::FP32>*>
                  (this->_helper)->_param_eltwise_relu;
    impl->_funcs_eltwise_relu(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
EltwiseReluHelper<Ttype, Dtype, Ptype>::~EltwiseReluHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status EltwiseReluHelper<Ttype, Dtype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing EltwiseRelu op parameter.";
    auto type = GET_PARAMETER(std::string, type);
    auto alpha = GET_PARAMETER(float, relu_0_alpha);
    auto coeff = GET_PARAMETER(PTuple<float>, coeff);

    ActivationParam<Tensor4d<Ttype, Dtype>> activation_param(Active_relu);

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
    EltwiseActiveParam<Tensor4d<Ttype, Dtype>> eltwise_relu_param(eltwise_param, activation_param);
    _param_eltwise_relu = eltwise_relu_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status EltwiseReluHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    _funcs_eltwise_relu.init(ins, outs, _param_eltwise_relu, SPECIFY, SABER_IMPL, ctx);
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status EltwiseReluHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    _funcs_eltwise_relu.compute_output_shape(ins, outs, _param_eltwise_relu);
    return Status::OK();
}

#ifdef USE_CUDA
template class EltwiseReluHelper<NV, AK_FLOAT, Precision::FP32>;
template class EltwiseReluHelper<NV, AK_FLOAT, Precision::FP16>;
template class EltwiseReluHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class EltwiseReluHelper<ARM, AK_FLOAT, Precision::FP32>;
template class EltwiseReluHelper<ARM, AK_FLOAT, Precision::FP16>;
template class EltwiseReluHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(EltwiseRelu, EltwiseReluHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(EltwiseRelu, EltwiseReluHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(EltwiseRelu)
.Doc("EltwiseRelu operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("eltwise")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("eltwise")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " eltwise type( string )")
.Args<float>("relu_0_alpha", " alpha for relu")
.Args<PTuple<float>>("coeff", "coeff of eltwise");

} /* namespace ops */

} /* namespace anakin */


