#include "framework/operators/fusion.h"

namespace anakin {

namespace ops {

/// TODO ... specialization other type of operator
#define INSTANCE_FUSION(Ttype, Ptype) \
template<> \
void Fusion<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<FusionHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = impl->_param_fusion; \
    impl->_funcs_fusion(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
FusionHelper<Ttype, Ptype>::~FusionHelper() {
}

template<typename Ttype, Precision Ptype>
Status FusionHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Fusion op parameter.";
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status FusionHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
#ifdef USE_BM_PLACE
    _param_fusion.model_path = ctx.get_bmodel_path();
#endif
    SABER_CHECK(_funcs_fusion.init(ins, outs, _param_fusion, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status FusionHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
                                              std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_fusion.compute_output_shape(ins, outs, _param_fusion));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_FUSION(NV, Precision::FP32);
template class FusionHelper<NV, Precision::FP32>;
template class FusionHelper<NV, Precision::FP16>;
template class FusionHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Fusion, FusionHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_FUSION(AMD, Precision::FP32);
template class FusionHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Fusion, FusionHelper, AMD, Precision::FP32);
#endif

#ifdef USE_BM_PLACE
INSTANCE_FUSION(BM, Precision::FP32);
INSTANCE_FUSION(BM, Precision::FP16);
template class FusionHelper<BM, Precision::FP32>;
template class FusionHelper<BM, Precision::FP16>;
template class FusionHelper<BM, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Fusion, FusionHelper, BM, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Fusion, FusionHelper, BM, Precision::FP16);
#endif 

#ifdef USE_MLU
INSTANCE_FUSION(MLU, Precision::FP32);
INSTANCE_FUSION(MLU, Precision::FP16);
template class FusionHelper<MLU, Precision::FP32>;
template class FusionHelper<MLU, Precision::FP16>;
template class FusionHelper<MLU, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Fusion, FusionHelper, MLU, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(Fusion, FusionHelper, MLU, Precision::FP16);
#endif  // USE_MLU

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_FUSION(X86, Precision::FP32);
template class FusionHelper<X86, Precision::FP32>;
template class FusionHelper<X86, Precision::FP16>;
template class FusionHelper<X86, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Fusion, FusionHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_FUSION(ARM, Precision::FP32);
template class FusionHelper<ARM, Precision::FP32>;
template class FusionHelper<ARM, Precision::FP16>;
template class FusionHelper<ARM, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Fusion, FusionHelper, ARM, Precision::FP32);
#endif //arm

//! register op
ANAKIN_REGISTER_OP(Fusion)
.Doc("Fusion operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("Fusion")
#endif

#ifdef USE_BM_PLACE
.__alias__<BM, Precision::FP32>("Fusion")
#endif  

#ifdef USE_MLU
.__alias__<MLU, Precision::FP32>("Fusion")
#endif  // USE_MLU

#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("Fusion")
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("Fusion")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("Fusion")
#endif
.num_in(1)
.num_out(1)
.Args<bool>("out_max_val", " out_max_val for fusion ")
.Args<unsigned int>("top_k", " top_k for fusion")
.Args<int>("axis", " axis for fusion");
} /* namespace ops */

} /* namespace anakin */


