#include "framework/operators/sproposal.h"

namespace anakin {

namespace ops {

#define INSTANCE_SPROPOSAL(Ttype, Ptype) \
template<> \
void SProposal<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<SProposalHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SProposalHelper<Ttype, Ptype>*>(this->_helper)->_param_sproposal; \
    impl->_funcs_sproposal(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
SProposalHelper<Ttype, Ptype>::~SProposalHelper() {}

template<typename Ttype, Precision Ptype>
Status SProposalHelper<Ttype, Ptype>::InitParam() {

    DLOG(WARNING) << "Parsing SProposal op parameter.";

    auto scale = GET_PARAMETER(PTuple<int>, scale);
    auto ratio = GET_PARAMETER(PTuple<float>, ratio);

    auto feat_stride = GET_PARAMETER(int, feat_stride);
    auto basesize = GET_PARAMETER(int, basesize);
    auto boxminsize = GET_PARAMETER(int, boxminsize);
    auto pre_nms_topn = GET_PARAMETER(int, pre_nms_topn);
    auto post_nms_topn = GET_PARAMETER(int, post_nms_topn);
    auto nms_thresh = GET_PARAMETER(float, nms_thresh);
    SProposalParam<Ttype> param_sproposal(scale.vector(), ratio.vector(),
            feat_stride, basesize, boxminsize, pre_nms_topn, post_nms_topn, nms_thresh);
    _param_sproposal = param_sproposal;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SProposalHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    saber::ImplEnum impl_e = SABER_IMPL;
    if (std::is_same<Ttype, X86>::value) {
        impl_e = SABER_IMPL;
    }
    SABER_CHECK(_funcs_sproposal.init(ins, outs, _param_sproposal, SPECIFY, impl_e, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SProposalHelper<Ttype, Ptype>::InferShape(
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_sproposal.compute_output_shape(ins, outs, _param_sproposal));
    return Status::OK();
}

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_SPROPOSAL(X86, Precision::FP32);
INSTANCE_SPROPOSAL(X86, Precision::FP16);
INSTANCE_SPROPOSAL(X86, Precision::INT8);
template class SProposalHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SProposal, SProposalHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SPROPOSAL(ARM, Precision::FP32);
template class SProposalHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SProposal, SProposalHelper, ARM, Precision::FP32);
#endif//arm

//! register op
ANAKIN_REGISTER_OP(SProposal)
.Doc("SProposal operator")
#if defined USE_X86_PLACE || defined(BUILD_LITE)
.__alias__<X86, Precision::FP32>("sproposal")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("sproposal")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("scale", "scale of sproposal")
.Args<PTuple<float>>("ratio", "ratio of sproposal")
.Args<int>("feat_stride", "feat_stride of sproposal")
.Args<int>("basesize", "basesize of sproposal")
.Args<int>("boxminsize", "boxminsize of sproposal")
.Args<int>("pre_nms_topn", "pre_nms_topn of sproposal")
.Args<int>("post_nms_topn", "post_nms_topn of sproposal")
.Args<float>("nms_thresh", "nms_thresh of sproposal");

} /* namespace ops */

} /* namespace anakin */

