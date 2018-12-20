#include "framework/operators/mean.h"

namespace anakin {

namespace ops {

#define INSTANCE_MEAN(Ttype, Ptype) \
template<> \
void Mean<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<MeanHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<MeanHelper<Ttype, Ptype>*> \
                  (this->_helper)->_param_mean; \
    impl->_funcs_mean(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
MeanHelper<Ttype, Ptype>::~MeanHelper() {
}

template<typename Ttype, Precision Ptype>
Status MeanHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Mean op parameter.";
    MeanParam<Ttype> param_mean;
    _param_mean = param_mean;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status MeanHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_mean.init(ins, outs, _param_mean, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status MeanHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_mean.compute_output_shape(ins, outs, _param_mean));
    return Status::OK();
}

#ifdef AMD_GPU
INSTANCE_MEAN(AMD, Precision::FP32);
template class MeanHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Mean, MeanHelper, AMD, Precision::FP32);
#endif
#ifdef USE_CUDA
INSTANCE_MEAN(NV, Precision::FP32);
template class MeanHelper<NV, Precision::FP32>;
template class MeanHelper<NV, Precision::FP16>;
template class MeanHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Mean, MeanHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
INSTANCE_MEAN(ARM, Precision::FP32);
template class MeanHelper<ARM, Precision::FP32>;
template class MeanHelper<ARM, Precision::FP16>;
template class MeanHelper<ARM, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Mean, MeanHelper, ARM, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
INSTANCE_MEAN(X86, Precision::FP32);
template class MeanHelper<X86, Precision::FP32>;
template class MeanHelper<X86, Precision::FP16>;
template class MeanHelper<X86, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Mean, MeanHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Mean)
.Doc("Mean operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("mean")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("mean")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("mean")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("mean")
#endif
.num_in(1)
.num_out(1)
.Args<int>("groups", " split tensor's channel by size groups. ");

} /* namespace ops */

} /* namespace anakin */


