#include "framework/operators/cos_sim.h"

namespace anakin {

namespace ops {

#define INSTANCE_COS_SIM(Ttype, Ptype) \
template<> \
void CosSim<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<CosSimHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<CosSimHelper<Ttype, Ptype>*>(this->_helper)->_param_cos_sim; \
    impl->_funcs_cos_sim(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
CosSimHelper<Ttype, Ptype>::~CosSimHelper() {
}

template<typename Ttype, Precision Ptype>
Status CosSimHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing CosSim op parameter.";
    CosSimParam<Ttype> param_cos_sim;
    _param_cos_sim = param_cos_sim;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status CosSimHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_cos_sim.init(ins, outs, _param_cos_sim, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status CosSimHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_cos_sim.compute_output_shape(ins, outs, _param_cos_sim));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_COS_SIM(NV, Precision::FP32);
template class CosSimHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(CosSim, CosSimHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_COS_SIM(X86, Precision::FP32);
template class CosSimHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(CosSim, CosSimHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_COS_SIM(ARM, Precision::FP32);
template class CosSimHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(CosSim, CosSimHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_COS_SIM(AMD, Precision::FP32);
template class CosSimHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(CosSim, CosSimHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(CosSim)
.Doc("CosSim operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("cos_sim")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("cos_sim")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("cos_sim")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("cos_sim")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */

