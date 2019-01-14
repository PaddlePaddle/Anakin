#include "framework/operators/product_quant_embedding_with_vsum.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void ProductQuantEmbeddingWithVsum<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl =
        static_cast<ProductQuantEmbeddingWithVsumHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param =
        static_cast<ProductQuantEmbeddingWithVsumHelper<NV, Precision::FP32>*>(this->_helper)->_param_product_quant_embedding_with_vsum;
    impl->_funcs_product_quant_embedding_with_vsum(ins, outs, param, ctx);
}
#endif

#ifdef USE_X86_PLACE
template<>
void ProductQuantEmbeddingWithVsum<X86, Precision::FP32>::operator()(
        OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl =
            static_cast<ProductQuantEmbeddingWithVsumHelper<X86, Precision::FP32>*>(this->_helper);
    auto& param =
            static_cast<ProductQuantEmbeddingWithVsumHelper<X86, Precision::FP32>*>(this->_helper)->_param_product_quant_embedding_with_vsum;
    impl->_funcs_product_quant_embedding_with_vsum(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
ProductQuantEmbeddingWithVsumHelper<Ttype, Ptype>::~ProductQuantEmbeddingWithVsumHelper() {
}

template<typename Ttype, Precision Ptype>
Status ProductQuantEmbeddingWithVsumHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing ProductQuantEmbeddingWithVsum op parameter.";
    auto word_voc = GET_PARAMETER(int, word_voc);
    auto word_emb = GET_PARAMETER(int, word_emb);
    auto max_seq_len = GET_PARAMETER(int, max_seq_len);
    auto top_unigram = GET_PARAMETER(int, top_unigram);
    auto sec_unigram = GET_PARAMETER(int, sec_unigram);
    auto thd_unigram = GET_PARAMETER(int, thd_unigram);
    auto top_bigram = GET_PARAMETER(int, top_bigram);
    auto sec_bigram = GET_PARAMETER(int, sec_bigram);
    auto thd_bigram = GET_PARAMETER(int, thd_bigram);
    auto top_collocation = GET_PARAMETER(int, top_collocation);
    auto sec_collocation = GET_PARAMETER(int, sec_collocation);
    auto thd_collocation = GET_PARAMETER(int, thd_collocation);
    

    using pblock_type = PBlock<Ttype>;
    auto embedding_0 = GET_PARAMETER(pblock_type, weight_3);
    auto embedding_1 = GET_PARAMETER(pblock_type, weight_6);
    auto embedding_2 = GET_PARAMETER(pblock_type, weight_9);
    auto quant_dict_0 = GET_PARAMETER(pblock_type, weight_2);
    auto quant_dict_1 = GET_PARAMETER(pblock_type, weight_5);
    auto quant_dict_2 = GET_PARAMETER(pblock_type, weight_8);

    ProductQuantEmbeddingWithVsumParam<Ttype> param_product_quant_embedding_with_vsum(word_emb,
            word_voc,
            top_unigram,
            top_bigram,
            top_collocation,
            sec_unigram,
            sec_bigram,
            sec_collocation,
            thd_unigram,
            thd_bigram,
            thd_collocation,
            max_seq_len,
            &(embedding_0.d_tensor()),
            &(embedding_1.d_tensor()),
            &(embedding_2.d_tensor()),
            &(quant_dict_0.d_tensor()),
            &(quant_dict_1.d_tensor()),
            &(quant_dict_2.d_tensor()));

    _param_product_quant_embedding_with_vsum = param_product_quant_embedding_with_vsum;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ProductQuantEmbeddingWithVsumHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_product_quant_embedding_with_vsum.init(ins, outs, _param_product_quant_embedding_with_vsum, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ProductQuantEmbeddingWithVsumHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_product_quant_embedding_with_vsum.compute_output_shape(ins, outs, _param_product_quant_embedding_with_vsum));
    return Status::OK();
}

#ifdef USE_CUDA
template class ProductQuantEmbeddingWithVsumHelper<NV, Precision::FP32>;
template class ProductQuantEmbeddingWithVsumHelper<NV, Precision::FP16>;
template class ProductQuantEmbeddingWithVsumHelper<NV, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class ProductQuantEmbeddingWithVsumHelper<ARM, Precision::FP32>;
template class ProductQuantEmbeddingWithVsumHelper<ARM, Precision::FP16>;
template class ProductQuantEmbeddingWithVsumHelper<ARM, Precision::INT8>;
#endif
#ifdef USE_X86_PLACE
template class ProductQuantEmbeddingWithVsumHelper<X86, Precision::FP32>;
template class ProductQuantEmbeddingWithVsumHelper<X86, Precision::FP16>;
template class ProductQuantEmbeddingWithVsumHelper<X86, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(ProductQuantEmbeddingWithVsum, ProductQuantEmbeddingWithVsumHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(ProductQuantEmbeddingWithVsum, ProductQuantEmbeddingWithVsumHelper, ARM, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(ProductQuantEmbeddingWithVsum, ProductQuantEmbeddingWithVsumHelper, X86, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(ProductQuantEmbeddingWithVsum)
.Doc("ProductQuantEmbeddingWithVsum operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("product_quant_embedding_with_vsum")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("product_quant_embedding_with_vsum")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("product_quant_embedding_with_vsum")
#endif
.num_in(1)
.num_out(1)
.Args<int>("word_num", "word_num")
.Args<int>("emb_dim", " emb_dim ")
.Args<int>("padding_idx", " padding idx ")
.Args<int>("num_direct", " num direct 1 or 2");

} /* namespace ops */

} /* namespace anakin */


