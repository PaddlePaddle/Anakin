
#include "framework/operators/cast.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Cast<NV, Precision::FP32>::operator()(
        OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV> >& ins,
        std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<CastHelper<NV, Precision::FP32>*>(
            this->_helper);
    auto& param = static_cast<CastHelper<NV, Precision::FP32>*>(
            this->_helper)->_param_cast;
    impl->_funcs_cast(ins, outs, param, ctx);
}
#endif

#ifdef USE_X86_PLACE
template<>
void Cast<X86, Precision::FP32>::operator()(
        OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl = static_cast<CastHelper<X86, Precision::FP32>*>(
            this->_helper);
    auto& param = static_cast<CastHelper<X86, Precision::FP32>*>(
            this->_helper)->_param_cast;
    impl->_funcs_cast(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator

/// set helper
template<typename Ttype, Precision Ptype>
CastHelper<Ttype, Ptype>::~CastHelper() {
}

template<typename Ttype, Precision Ptype>
Status CastHelper<Ttype, Ptype>::InitParam() {
            DLOG(WARNING) << "Parsing Cast op parameter.";
    auto in_type = GET_PARAMETER(int, in_type);
    auto out_type = GET_PARAMETER(int, out_type);
    CastParam<Ttype> param_cast(in_type, out_type);
    _param_cast = param_cast;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status CastHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    SABER_CHECK(_funcs_cast.init(ins, outs, _param_cast,
            SPECIFY, SABER_IMPL, ctx));

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status CastHelper<Ttype, Ptype>::InferShape(
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    SABER_CHECK(_funcs_cast.compute_output_shape(ins, outs, _param_cast));
    return Status::OK();
}

#ifdef USE_CUDA
template class CastHelper<NV, Precision::FP32>;
template class CastHelper<NV, Precision::FP16>;
template class CastHelper<NV, Precision::INT8>;
#endif
#ifdef USE_X86_PLACE
template class CastHelper<X86, Precision::FP32>;
template class CastHelper<X86, Precision::FP16>;
template class CastHelper<X86, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Cast, CastHelper, NV, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(Cast, CastHelper, X86, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Cast)
.Doc("Cast operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("cast")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("cast")
#endif
.num_in(1)
.num_out(1)
.Args<int>("in_type", "in_type of cast param")
.Args<int>("out_type", "out_type of cast param");

} /* namespace ops */

} /* namespace anakin */


