#include "framework/operators/mean.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Mean<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl =
        static_cast<MeanHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param =
        static_cast<MeanHelper<NV, Precision::FP32>*>(this->_helper)->_param_mean;
    impl->_funcs_mean(ins, outs, param, ctx);
}
#endif

#ifdef USE_X86_PLACE
template<>
void Mean<X86, Precision::FP32>::operator()(
        OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl =
            static_cast<MeanHelper<X86, Precision::FP32>*>(this->_helper);
    auto& param =
            static_cast<MeanHelper<X86, Precision::FP32>*>(this->_helper)->_param_mean;
    impl->_funcs_mean(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


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

#ifdef USE_CUDA
template class MeanHelper<NV, Precision::FP32>;
template class MeanHelper<NV, Precision::FP16>;
template class MeanHelper<NV, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class MeanHelper<ARM, Precision::FP32>;
template class MeanHelper<ARM, Precision::FP16>;
template class MeanHelper<ARM, Precision::INT8>;
#endif
#ifdef USE_X86_PLACE
template class MeanHelper<X86, Precision::FP32>;
template class MeanHelper<X86, Precision::FP16>;
template class MeanHelper<X86, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Mean, MeanHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Mean, MeanHelper, ARM, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
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
.num_in(1)
.num_out(1)
.Args<int>("groups", " split tensor's channel by size groups. ");

} /* namespace ops */

} /* namespace anakin */


