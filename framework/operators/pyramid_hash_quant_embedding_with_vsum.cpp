#include "framework/operators/pyramid_hash_quant_embedding_with_vsum.h"

namespace anakin {

namespace ops {

#define INSTANCE_PYRAMID_HASH_QUANT_EMBEDDING_WITH_VSUM(Ttype, Ptype) \
template<> \
void PyramidHashQuantEmbeddingWithVsum<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<PyramidHashQuantEmbeddingWithVsumHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<PyramidHashQuantEmbeddingWithVsumHelper<Ttype, Ptype>*>(this->_helper)->_param_pyramid_hash_quant_embedding_with_vsum; \
    impl->_funcs_pyramid_hash_quant_embedding_with_vsum(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
PyramidHashQuantEmbeddingWithVsumHelper<Ttype, Ptype>::~PyramidHashQuantEmbeddingWithVsumHelper() {
}

template<typename Ttype, Precision Ptype>
Status PyramidHashQuantEmbeddingWithVsumHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing PyramidHashQuantEmbeddingWithVsum op parameter.";
    auto space_size = GET_PARAMETER(int, space_size);
    auto emb_size = GET_PARAMETER(int, emb_size);
    auto pyramid_layer = GET_PARAMETER(int, pyramid_layer);
    auto rand_len = GET_PARAMETER(int, rand_len);
    auto white_list_len = GET_PARAMETER(int, white_list_len);
    auto black_list_len = GET_PARAMETER(int, black_list_len);
    auto dropout_percent = GET_PARAMETER(float, dropout_percent);
    using pblock_type = PBlock<Ttype>;
    auto quant_dict = GET_PARAMETER(pblock_type, weight_2);
    auto hash_space = GET_PARAMETER(pblock_type, weight_3);
    auto white_filter = GET_PARAMETER(pblock_type, weight_4);
    auto black_filter = GET_PARAMETER(pblock_type, weight_5);
    Tensor<Ttype>* white_filter_tensor = NULL;
    Tensor<Ttype>* black_filter_tensor = NULL;
    if (white_list_len) {
        white_filter_tensor = &(white_filter.d_tensor());
    } 
    if (black_list_len) {
        black_filter_tensor = &(black_filter.d_tensor());
    } 

    PyramidHashQuantEmbeddingParam<Ttype> param_pyramid_hash_quant_embedding_with_vsum(space_size,
       emb_size,
       pyramid_layer,
       rand_len,
       white_list_len,
       black_list_len,
       dropout_percent,
       &(quant_dict.d_tensor()),
       &(hash_space.d_tensor()),
       white_filter_tensor,
       black_filter_tensor);

    _param_pyramid_hash_quant_embedding_with_vsum = param_pyramid_hash_quant_embedding_with_vsum;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PyramidHashQuantEmbeddingWithVsumHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_pyramid_hash_quant_embedding_with_vsum.init(ins, outs, _param_pyramid_hash_quant_embedding_with_vsum, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PyramidHashQuantEmbeddingWithVsumHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_pyramid_hash_quant_embedding_with_vsum.compute_output_shape(ins, outs, _param_pyramid_hash_quant_embedding_with_vsum));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_PYRAMID_HASH_QUANT_EMBEDDING_WITH_VSUM(NV, Precision::FP32);
template class PyramidHashQuantEmbeddingWithVsumHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PyramidHashQuantEmbeddingWithVsum, PyramidHashQuantEmbeddingWithVsumHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_PYRAMID_HASH_QUANT_EMBEDDING_WITH_VSUM(X86, Precision::FP32);
template class PyramidHashQuantEmbeddingWithVsumHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PyramidHashQuantEmbeddingWithVsum, PyramidHashQuantEmbeddingWithVsumHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_PYRAMID_HASH_QUANT_EMBEDDING_WITH_VSUM(ARM, Precision::FP32);
template class PyramidHashQuantEmbeddingWithVsumHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PyramidHashQuantEmbeddingWithVsum, PyramidHashQuantEmbeddingWithVsumHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_PYRAMID_HASH_QUANT_EMBEDDING_WITH_VSUM(AMD, Precision::FP32);
template class PyramidHashQuantEmbeddingWithVsumHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PyramidHashQuantEmbeddingWithVsum, PyramidHashQuantEmbeddingWithVsumHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(PyramidHashQuantEmbeddingWithVsum)
.Doc("PyramidHashQuantEmbeddingWithVsum operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("pyramid_hash_quant_embedding_with_vsum")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("pyramid_hash_quant_embedding_with_vsum")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("pyramid_hash_quant_embedding_with_vsum")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("pyramid_hash_quant_embedding_with_vsum")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */

