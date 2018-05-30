#include "framework/operators/embedding.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Embedding<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl =
        static_cast<EmbeddingHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param =
        static_cast<EmbeddingHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_embedding;
    impl->_funcs_embedding(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
EmbeddingHelper<Ttype, Dtype, Ptype>::~EmbeddingHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status EmbeddingHelper<Ttype, Dtype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing Embedding op parameter.";
    auto word_num = GET_PARAMETER(int, word_num);
    auto emb_dim = GET_PARAMETER(int, emb_dim);
    auto padding_idx = GET_PARAMETER(int, padding_idx);
    auto weights = GET_PARAMETER(PBlock<typename DataTypeWarpper<Dtype>::type>, weight_1);

    EmbeddingParam<Tensor4d<Ttype, Dtype>> param_embedding(word_num, emb_dim, padding_idx, &(weights.d_tensor()));
    _param_embedding = param_embedding;

    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status EmbeddingHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_embedding.init(ins, outs, _param_embedding, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status EmbeddingHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_embedding.compute_output_shape(ins, outs, _param_embedding));
    return Status::OK();
}

#ifdef USE_CUDA
template class EmbeddingHelper<NV, AK_FLOAT, Precision::FP32>;
template class EmbeddingHelper<NV, AK_FLOAT, Precision::FP16>;
template class EmbeddingHelper<NV, AK_FLOAT, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class EmbeddingHelper<ARM, AK_FLOAT, Precision::FP32>;
template class EmbeddingHelper<ARM, AK_FLOAT, Precision::FP16>;
template class EmbeddingHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Embedding, EmbeddingHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Embedding, EmbeddingHelper, ARM, AK_FLOAT, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Embedding)
.Doc("Embedding operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("embedding")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("embedding")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " type of Embedding ");

} /* namespace ops */

} /* namespace anakin */


