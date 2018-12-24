#include "framework/operators/soft_sign.h"

namespace anakin {

namespace ops {

#define INSTANCE_SOFT_SIGN(Ttype, Ptype) \
template<> \
void SoftSign<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<SoftSignHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SoftSignHelper<Ttype, Ptype>*>(this->_helper)->_param_soft_sign; \
    impl->_funcs_soft_sign(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
SoftSignHelper<Ttype, Ptype>::~SoftSignHelper() {
}

template<typename Ttype, Precision Ptype>
Status SoftSignHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing SoftSign op parameter.";
    SoftSignParam<Ttype> param_soft_sign;
    _param_soft_sign = param_soft_sign;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SoftSignHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_soft_sign.init(ins, outs, _param_soft_sign, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SoftSignHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_soft_sign.compute_output_shape(ins, outs, _param_soft_sign));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SOFT_SIGN(NV, Precision::FP32);
template class SoftSignHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SoftSign, SoftSignHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_SOFT_SIGN(X86, Precision::FP32);
template class SoftSignHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SoftSign, SoftSignHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SOFT_SIGN(ARM, Precision::FP32);
template class SoftSignHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SoftSign, SoftSignHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_SOFT_SIGN(AMD, Precision::FP32);
template class SoftSignHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SoftSign, SoftSignHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(SoftSign)
.Doc("SoftSign operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("soft_sign")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("soft_sign")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("soft_sign")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("soft_sign")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */

