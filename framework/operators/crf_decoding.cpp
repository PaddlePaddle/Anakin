#include "framework/operators/crf_decoding.h"

namespace anakin {

namespace ops {

#ifdef USE_X86_PLACE
template<>
void CrfDecoding<X86, Precision::FP32>::operator()(
    OpContext<X86>& ctx,
    const std::vector<Tensor4dPtr<X86> >& ins,
    std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl = static_cast<CrfDecodingHelper<X86, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<CrfDecodingHelper<X86, Precision::FP32>*>(this->_helper)->_param_crf_decoding;
    impl->_funcs_crf_decoding(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
CrfDecodingHelper<Ttype, Ptype>::~CrfDecodingHelper() {
}

template<typename Ttype, Precision Ptype>
Status CrfDecodingHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing CrfDecoding op parameter.";

	using pblock_type = PBlock<Ttype>;
    auto weights = GET_PARAMETER(pblock_type, weight_1);
    saber::CrfDecodingParam<Ttype> crf_decoding_param(&(weights.d_tensor()));
    _param_crf_decoding = crf_decoding_param;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status CrfDecodingHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_crf_decoding.init(ins, outs, _param_crf_decoding, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status CrfDecodingHelper<Ttype, Ptype>::InferShape(
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_crf_decoding.compute_output_shape(ins, outs, _param_crf_decoding));
    return Status::OK();
}

#ifdef USE_CUDA
template class CrfDecodingHelper<NV, Precision::FP32>;
template class CrfDecodingHelper<NV, Precision::FP16>;
template class CrfDecodingHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class CrfDecodingHelper<ARM, Precision::FP32>;
template class CrfDecodingHelper<ARM, Precision::FP16>;
template class CrfDecodingHelper<ARM, Precision::INT8>;
#endif

#ifdef USE_X86_PLACE
template class CrfDecodingHelper<X86, Precision::FP32>;
template class CrfDecodingHelper<X86, Precision::FP16>;
template class CrfDecodingHelper<X86, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(CrfDecoding, CrfDecodingHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(CrfDecoding, CrfDecodingHelper, ARM, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(CrfDecoding, CrfDecodingHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(CrfDecoding)
.Doc("CrfDecoding operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("CrfDecoding")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("CrfDecoding")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("CrfDecoding")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


