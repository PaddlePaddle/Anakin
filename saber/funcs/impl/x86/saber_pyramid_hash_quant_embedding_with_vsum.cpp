
#include "saber/funcs/impl/x86/saber_pyramid_hash_quant_embedding_with_vsum.h"
#include "mkl.h"
#if defined(__AVX2__) and defined(__FMA__)
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
#endif
#include <cmath>
extern "C"{
    #include "xxHash/xxhash.h"
    #include "bloomfilter/bloomfilter.h"
}

namespace anakin{
namespace saber {

bool should_use_term(
            const float* term, 
            bloomfilter* white_filter_ptr, 
            bloomfilter* black_filter_ptr, 
            size_t len){
        return
            (!white_filter_ptr || 1 == bloomfilter_get(white_filter_ptr, 
                                                 term, 
                                                 len * sizeof(float))) &&
            (!black_filter_ptr || 0 == bloomfilter_get(black_filter_ptr, 
                                                       term, 
                                                       len * sizeof(float)));
}

template <DataType OpDtype>
SaberStatus SaberPyramidHashQuantEmbeddingWithVsum<X86, OpDtype>::hash_embedding_forward(const OpDataType* buffer, 
          int len,
          const OpDataType* quant_dict,
          const unsigned char* weights,
          OpDataType* out) {
    for (unsigned int j = 0; j < _emb_size; j += _rand_len) {
        unsigned int pos = XXH32(buffer, len * sizeof(OpDataType), j) % _space_size;
        //LOG(INFO)<< "pos:" <<pos << " _emb_size "<< _emb_size << " _rand_len:"<< _rand_len << "_dropout_percent" << _dropout_percent;
        for (unsigned int k = 0; k < _rand_len; ++k) {
            out[j + k] += _dropout_percent * quant_dict[weights[pos + k]];
            //out[j + k] += quant_dict[weights[pos + k]];
        }
    }
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberPyramidHashQuantEmbeddingWithVsum<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        PyramidHashQuantEmbeddingParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    _space_size = param.space_size;
    _emb_size = param.emb_size;
    _pyramid_layer = param.pyramid_layer;
    _rand_len = param.rand_len;
    _white_filter_size = param.white_list_len;
    _black_filter_size = param.black_list_len;
    _dropout_percent = param.dropout_percent;
    _quant_bit = 8;
    _dict_size = 1 << _quant_bit;
    CHECK_EQ(param.quant_dict->valid_size(), _dict_size);
    CHECK_EQ(param.hash_space->valid_size(), _space_size + _rand_len);
    if (param.white_filter != NULL) {
        CHECK_EQ(param.white_filter->valid_size(), _white_filter_size);
    }
    if (param.black_filter != NULL) {
        CHECK_EQ(param.black_filter->valid_size(), _black_filter_size);
    }
    
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberPyramidHashQuantEmbeddingWithVsum<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        PyramidHashQuantEmbeddingParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberPyramidHashQuantEmbeddingWithVsum<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        PyramidHashQuantEmbeddingParam<X86> &param) {
    CHECK_EQ(inputs.size(), 1) << "PyramidHashQuantEmbedding input num need be  1, but is" << inputs.size();
    CHECK_EQ(outputs.size(), 1) << "PyramidHashQuantEmbedding input num need be  1, but is" << outputs.size();
    size_t count = inputs[0]->valid_size();

    const OpDataType *input_data = (const OpDataType*)inputs[0]->data();
    OpDataType *output_data = (OpDataType*)outputs[0]->mutable_data();
    const unsigned char* weights = (const unsigned char*) param.hash_space->data();
    const float* quant_dict = (const float*)param.quant_dict->data();
    CHECK(weights !=  NULL) << "embedding matrix weights is NULL";

    bloomfilter* white_filter_ptr = NULL;
    bloomfilter* black_filter_ptr = NULL;
    if (_white_filter_size) {
        white_filter_ptr = (bloomfilter*)param.white_filter->mutable_data();
    }
    if (_black_filter_size) {
        black_filter_ptr = (bloomfilter*)param.black_filter->mutable_data();
    }

    auto in_seq_offset = inputs[0]->get_seq_offset()[0];
    memset(output_data, 0, sizeof(OpDataType)*outputs[0]->valid_size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < in_seq_offset.size() - 1; i++) {
        int cur_len = in_seq_offset[i+1] - in_seq_offset[i];
        auto tmp_out_data = output_data + i * _emb_size;
        auto in_tmp = input_data + in_seq_offset[i];

        if (cur_len < 2) {
            memset(tmp_out_data, 0, sizeof(OpDataType) * _emb_size);
        } else {
            for (int j = 1; j < param.pyramid_layer && j < cur_len; j++) {
                for (int k = 0; k < cur_len - j; k++) {
                    if (should_use_term(&in_tmp[k], white_filter_ptr, black_filter_ptr, j + 1)) {
                        hash_embedding_forward(&in_tmp[k], j + 1, quant_dict, weights,
                                tmp_out_data);
                    }
                }
            }
        }
    }
    return SaberSuccess;
} 
template class SaberPyramidHashQuantEmbeddingWithVsum<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberPyramidHashQuantEmbeddingWithVsum, PyramidHashQuantEmbeddingParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberPyramidHashQuantEmbeddingWithVsum, PyramidHashQuantEmbeddingParam, X86, AK_INT8);
}
}
   
