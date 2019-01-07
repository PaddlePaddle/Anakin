#include "framework/operators/eltwise_op.h"

namespace anakin {

namespace ops {

#define INSTANCE_ELTWISE(Ttype, Ptype) \
template<> \
void Eltwise<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<EltwiseHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<EltwiseHelper<Ttype, Ptype>*> \
                  (this->_helper)->_param_eltwise; \
    impl->_funcs_eltwise(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status EltwiseHelper<Ttype, Ptype>::InitParam() {
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
    saber::EltwiseParam<Ttype> eltwise_param(elt_type, coeff.vector());
    _param_eltwise = eltwise_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status EltwiseHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx, const std::vector<Tensor4dPtr<Ttype>> &ins,
                           std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_eltwise.init(ins, outs, _param_eltwise, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status EltwiseHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>> &ins,
                                 std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_eltwise.compute_output_shape(ins, outs, _param_eltwise));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ELTWISE(NV, Precision::FP32);
template class EltwiseHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Eltwise, EltwiseHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_ELTWISE(X86, Precision::FP32);
template class EltwiseHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Eltwise, EltwiseHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ELTWISE(ARM, Precision::FP32);
template class EltwiseHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Eltwise, EltwiseHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Eltwise)
.Doc("Eltwise operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("eltwise")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("eltwise")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("eltwise")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " eltwise type( string )")
.Args<PTuple<float>>("coeff", "coeff of eltwise");


} /* namespace ops */

} /* namespace anakin */


