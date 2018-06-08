#include "framework/operators/eltwise_op.h"

namespace anakin {

namespace ops {

//#ifdef USE_CUDA
//template<>
//void Eltwise<NV, AK_FLOAT, Precision::FP32>::operator()(
//    OpContext<NV>& ctx,
//    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
//    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
//    auto* impl = static_cast<EltwiseHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
//    auto& param = static_cast<EltwiseHelper<NV, AK_FLOAT, Precision::FP32>*>
//                  (this->_helper)->_param_eltwise;
//    impl->_funcs_eltwise(ins, outs, param, ctx);
//}
//#endif

/// TODO ... specialization other type of operator
#define INSTANCE_ELTWISE(Ttype, Dtype, Ptype) \
template<> \
void Eltwise<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<EltwiseHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = static_cast<EltwiseHelper<Ttype, Dtype, Ptype>*> \
                  (this->_helper)->_param_eltwise; \
    impl->_funcs_eltwise(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
EltwiseHelper<Ttype, Dtype, Ptype>::~EltwiseHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status EltwiseHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Eltwise op parameter.";
    auto type = GET_PARAMETER(std::string, type);
    auto coeff = GET_PARAMETER(PTuple<float>, coeff);
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

    //    saber::EltwiseParam<Tensor4d<Ttype, Dtype>>    eltwise_param(elt_type, tdcoeff_p);
    saber::EltwiseParam<Tensor4d<Ttype, Dtype> > eltwise_param(elt_type, coeff.vector());
    _param_eltwise = eltwise_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status EltwiseHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_eltwise.init(ins, outs, _param_eltwise, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status EltwiseHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_eltwise.compute_output_shape(ins, outs, _param_eltwise));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ELTWISE(NV, AK_FLOAT, Precision::FP32);
template class EltwiseHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Eltwise, EltwiseHelper, NV, AK_FLOAT, Precision::FP32);
template class EltwiseHelper<NV, AK_FLOAT, Precision::FP16>;
template class EltwiseHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
INSTANCE_ELTWISE(ARM, AK_FLOAT, Precision::FP32);
template class EltwiseHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Eltwise, EltwiseHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef ANAKIN_TYPE_FP16
template class EltwiseHelper<ARM, AK_FLOAT, Precision::FP16>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class EltwiseHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

#endif//arm

//! register op
ANAKIN_REGISTER_OP(Eltwise)
.Doc("Eltwise operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("eltwise")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("eltwise")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " eltwise type( string )")
.Args<PTuple<float>>("coeff", "coeff of eltwise");

} /* namespace ops */

} /* namespace anakin */


