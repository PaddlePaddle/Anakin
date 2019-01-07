#include "framework/operators/generate_proposals.h"

namespace anakin {

namespace ops {

#define INSTANCE_ACTIVATION(Ttype, Ptype) \
template<> \
void GenerateProposals<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<GenerateProposalsHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<GenerateProposalsHelper<Ttype, Ptype>*>(this->_helper)->_param_generate_proposals; \
    impl->_funcs_generate_proposals(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
GenerateProposalsHelper<Ttype, Ptype>::~GenerateProposalsHelper() {
}

template<typename Ttype, Precision Ptype>
Status GenerateProposalsHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing GenerateProposals op parameter.";
    auto pre_nms_top_n = GET_PARAMETER(int, pre_nms_top_n);
    auto post_nms_top_n = GET_PARAMETER(int, post_nms_top_n);
    auto nms_thresh = GET_PARAMETER(float, nms_thresh);
    auto min_size = GET_PARAMETER(float, min_size);
    auto eta = GET_PARAMETER(float, eta);
    GenerateProposalsParam<Ttype> param_generate_proposals(pre_nms_top_n, post_nms_top_n, nms_thresh, min_size, eta);
    _param_generate_proposals = param_generate_proposals;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status GenerateProposalsHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_generate_proposals.init(ins, outs, _param_generate_proposals, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status GenerateProposalsHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_generate_proposals.compute_output_shape(ins, outs, _param_generate_proposals));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ACTIVATION(NV, Precision::FP32);

template<>
Status GenerateProposalsHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx, 
                                                   const std::vector< Tensor4dPtr<NV> > & ins, 
                                                   std::vector< Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_generate_proposals.init(ins, outs, _param_generate_proposals, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(GenerateProposals, GenerateProposalsHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_ACTIVATION(X86, Precision::FP32);
INSTANCE_ACTIVATION(X86, Precision::FP16);
INSTANCE_ACTIVATION(X86, Precision::INT8);
template class GenerateProposalsHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(GenerateProposals, GenerateProposalsHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ACTIVATION(ARM, Precision::FP32);
template class GenerateProposalsHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(GenerateProposals, GenerateProposalsHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_ACTIVATION(AMD, Precision::FP32);
template class GenerateProposalsHelper<AMD, Precision::FP32>;
template class GenerateProposalsHelper<AMD, Precision::FP16>;
template class GenerateProposalsHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(GenerateProposals, GenerateProposalsHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(GenerateProposals)
.Doc("GenerateProposals operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("generate_proposals")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("generate_proposals")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("generate_proposals")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("generate_proposals")
#endif
.num_in(1)
.num_out(1)
.Args<int>("pre_nms_top_n", "prelu channel is shared or not ")
.Args<int>("post_nms_top_n", "post_nms_top_n")
.Args<float>("nms_thresh", "nms_thresh")
.Args<float>("min_size", "min_size ")
.Args<float>("eta", "eta");

} /* namespace ops */

} /* namespace anakin */

