#include "framework/operators/crf_decoding.h"

namespace anakin {

namespace ops {

#ifdef USE_X86_PLACE
template<>
void CrfDecoding<X86, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<X86>& ctx,
    const std::vector<Tensor4dPtr<X86, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<X86, AK_FLOAT> >& outs) {
    auto* impl = static_cast<CrfDecodingHelper<X86, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<CrfDecodingHelper<X86, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_crf_decoding;
    impl->_funcs_crf_decoding(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
CrfDecodingHelper<Ttype, Dtype, Ptype>::~CrfDecodingHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status CrfDecodingHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing CrfDecoding op parameter.";

	using pblock_type = PBlock<typename DataTypeWarpper<Dtype>::type, Ttype>;
    auto weights = GET_PARAMETER(pblock_type, weight_1);
    saber::CrfDecodingParam<Tensor4d<Ttype, Dtype>> crf_decoding_param(&(weights.d_tensor()));
    _param_crf_decoding = crf_decoding_param;

    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status CrfDecodingHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_crf_decoding.init(ins, outs, _param_crf_decoding, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status CrfDecodingHelper<Ttype, Dtype, Ptype>::InferShape(
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_crf_decoding.compute_output_shape(ins, outs, _param_crf_decoding));
    return Status::OK();
}

#ifdef USE_CUDA
template class CrfDecodingHelper<NV, AK_FLOAT, Precision::FP32>;
template class CrfDecodingHelper<NV, AK_FLOAT, Precision::FP16>;
template class CrfDecodingHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class CrfDecodingHelper<ARM, AK_FLOAT, Precision::FP32>;
template class CrfDecodingHelper<ARM, AK_FLOAT, Precision::FP16>;
template class CrfDecodingHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_X86_PLACE
template class CrfDecodingHelper<X86, AK_FLOAT, Precision::FP32>;
template class CrfDecodingHelper<X86, AK_FLOAT, Precision::FP16>;
template class CrfDecodingHelper<X86, AK_FLOAT, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(CrfDecoding, CrfDecodingHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(CrfDecoding, CrfDecodingHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(CrfDecoding, CrfDecodingHelper, X86, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(CrfDecoding)
.Doc("CrfDecoding operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("CrfDecoding")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("CrfDecoding")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("CrfDecoding")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


