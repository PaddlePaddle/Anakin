#include "framework/operators/embedding.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Embedding<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl =
        static_cast<EmbeddingHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param =
        static_cast<EmbeddingHelper<NV, Precision::FP32>*>(this->_helper)->_param_embedding;
    impl->_funcs_embedding(ins, outs, param, ctx);
}
#endif

#ifdef USE_X86_PLACE
template<>
void Embedding<X86, Precision::FP32>::operator()(
        OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl =
            static_cast<EmbeddingHelper<X86, Precision::FP32>*>(this->_helper);
    auto& param =
            static_cast<EmbeddingHelper<X86, Precision::FP32>*>(this->_helper)->_param_embedding;
    impl->_funcs_embedding(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
EmbeddingHelper<Ttype, Ptype>::~EmbeddingHelper() {
}

template<typename Ttype, Precision Ptype>
Status EmbeddingHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Embedding op parameter.";
    auto word_num = GET_PARAMETER(int, word_num);
    auto emb_dim = GET_PARAMETER(int, emb_dim);
    auto padding_idx = GET_PARAMETER(int, padding_idx);
	using pblock_type = PBlock<Ttype>;
    auto weights = GET_PARAMETER(pblock_type, weight_1);

    EmbeddingParam<Ttype> param_embedding(word_num, emb_dim, padding_idx, &(weights.d_tensor()));
    _param_embedding = param_embedding;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status EmbeddingHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_embedding.init(ins, outs, _param_embedding, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status EmbeddingHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_embedding.compute_output_shape(ins, outs, _param_embedding));
    return Status::OK();
}

#ifdef USE_CUDA
template class EmbeddingHelper<NV, Precision::FP32>;
template class EmbeddingHelper<NV, Precision::FP16>;
template class EmbeddingHelper<NV, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class EmbeddingHelper<ARM, Precision::FP32>;
template class EmbeddingHelper<ARM, Precision::FP16>;
template class EmbeddingHelper<ARM, Precision::INT8>;
#endif
#ifdef USE_X86_PLACE
template class EmbeddingHelper<X86, Precision::FP32>;
template class EmbeddingHelper<X86, Precision::FP16>;
template class EmbeddingHelper<X86, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Embedding, EmbeddingHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Embedding, EmbeddingHelper, ARM, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(Embedding, EmbeddingHelper, X86, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Embedding)
.Doc("Embedding operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("embedding")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("embedding")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("embedding")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " type of Embedding ");

} /* namespace ops */

} /* namespace anakin */


