#include "framework/operators/concat.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
Status ConcatHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Concat op parameter.";
    auto axis = GET_PARAMETER(int, axis);
    ConcatParam<Ttype> param_concat(axis);
    _param_concat = param_concat;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConcatHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx,
                                  const std::vector<Tensor4dPtr<Ttype> >& ins,
                                    std::vector<Tensor4dPtr<Ttype> >& outs){
    SABER_CHECK(_funcs_concat.init(ins, outs, _param_concat, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ConcatHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>> &ins,
                                std::vector<Tensor4dPtr<Ttype>> &outs) {
    SABER_CHECK(_funcs_concat.compute_output_shape(ins, outs, _param_concat));
    return Status::OK();
}


#define INSTANCE_CONCAT(Ttype, Ptype) \
template<> \
void Concat<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
                std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<ConcatHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ConcatHelper<Ttype, Ptype>*>(this->_helper)->_param_concat; \
    impl->_funcs_concat(ins, outs, param, ctx); \
}

#ifdef USE_CUDA
INSTANCE_CONCAT(NV, Precision::FP32);
template class ConcatHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_CONCAT(ARM, Precision::FP32);
template class ConcatHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, ARM, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_CONCAT(X86, Precision::FP32);
template class ConcatHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Concat)
.Doc("Concat operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("concat")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("concat")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("concat")
#endif
.num_in(2)
.num_out(1)
.Args<int>("axis", " axis for concat the input ");

} /* namespace ops */

} /* namespace anakin */


