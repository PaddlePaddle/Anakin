#include "framework/operators/ctc_align.h"

namespace anakin {

namespace ops {

#define INSTANCE_CTC_ALIGN(Ttype, Ptype) \
template<> \
void CtcAlign<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
                std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<CtcAlignHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<CtcAlignHelper<Ttype, Ptype>*>(this->_helper)->_param_ctc_align; \
    impl->_funcs_ctc_align(ins, outs, param, ctx); \
}

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
CtcAlignHelper<Ttype, Ptype>::~CtcAlignHelper() {
}

template<typename Ttype, Precision Ptype>
Status CtcAlignHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing CtcAlign op parameter.";
    auto merge_repeated = GET_PARAMETER(bool, merge_repeated);
    auto blank = GET_PARAMETER(int, blank);

    CtcAlignParam<Ttype> ctc_align_param(blank, merge_repeated);
    _param_ctc_align = ctc_align_param;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status CtcAlignHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx, 
                                                const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_ctc_align.init(ins, outs, _param_ctc_align, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status CtcAlignHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
                                                      std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_ctc_align.compute_output_shape(ins, outs, _param_ctc_align));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_CTC_ALIGN(NV, Precision::FP32);
template class CtcAlignHelper<NV, Precision::FP32>;
template class CtcAlignHelper<NV, Precision::FP16>;
template class CtcAlignHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(CtcAlign, CtcAlignHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_CTC_ALIGN(AMD, Precision::FP32);
template class CtcAlignHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(CtcAlign, CtcAlignHelper, AMD, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_CTC_ALIGN(ARM, Precision::FP32);
template class CtcAlignHelper<ARM, Precision::FP32>;
template class CtcAlignHelper<ARM, Precision::FP16>;
template class CtcAlignHelper<ARM, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(CtcAlign, CtcAlignHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(CtcAlign)
    .Doc("CtcAlign operator")
#ifdef USE_CUDA
    .__alias__<NV, Precision::FP32>("ctc_align")
#endif
#ifdef USE_ARM_PLACE
    .__alias__<ARM, Precision::FP32>("ctc_align")
#endif
#ifdef AMD_GPU
    .__alias__<AMD, Precision::FP32>("ctc_align")
#endif
    .num_in(1)
    .num_out(1)
    .Args<bool>("merge_repeated", " merge_repeated for ctc_align.")
    .Args<int>("blank",  "blank for ctc_align.");

} /* namespace ops */

} /* namespace anakin */


