#include "anakin_thread.h"
#include "saber/funcs/impl/x86/saber_product_quant_embedding_with_vsum.h"
#include "mkl.h"
#if defined(__AVX2__) and defined(__FMA__)
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
#endif
#include <cmath>

namespace anakin{
namespace saber {
bool decode_4d12b( const unsigned char *in,
                   unsigned int ilen,
                   unsigned int *out,
                   unsigned int olen) {
    if (ilen % 3 != 0) {
        LOG(INFO) << "error, ilen mod 3 != 0";
        return false;
    }
    if (ilen * 2 != olen * 3) {
        LOG(INFO) << "error, ilen * 2 != olen * 3";
        return false;
    }
    memset(out, 0, olen * sizeof(unsigned int));
    for (unsigned int i = 0; i < ilen / 3; i++) {
        unsigned char *raw_ptr = (unsigned char *)(out + i * 2);
        auto tmp_in = in + 3 * i;
        raw_ptr[0] = tmp_in[0];
        raw_ptr[1] = tmp_in[1] & 0x0f;
        raw_ptr[4] = tmp_in[2];
        raw_ptr[5] = tmp_in[1] >> 4;
    }
    return true;
}

void get_cur_idx(int word_idx, const int* word_offset, const int* real_offset, int offset_len, int* real_idx, int* case_idx) {
    CHECK_EQ(offset_len, 9);
    int index = 0;
    if (word_idx < word_offset[4]) {
        if (word_idx < word_offset[2]) {
            if (word_idx < word_offset[1]) {
                if (word_idx < word_offset[0]) {
                    index = 0;
                } else {
                    index = 1;
                }
            } else {
                index = 2;
            }
        } else {
            if (word_idx < word_offset[3]) {
                index = 3;
            } else {
                index = 4;
            }
        }
    } else { 
        if (word_idx < word_offset[6]) {
            if (word_idx < word_offset[5]) {
                index = 5;
            } else {
                index = 6;
            }
        } else {
            if (word_idx < word_offset[7]) {
                index = 7;
            } else {
                index = 8;
            }
        }
    }
    *case_idx = index % 3;
    if (index > 0) {
        *real_idx = word_idx - word_offset[index - 1] + real_offset[index]; 
    } else {
        *real_idx = word_idx;
    }
}

template <DataType OpDtype>
SaberStatus SaberProductQuantEmbeddingWithVsum<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ProductQuantEmbeddingWithVsumParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    _voc_size = param.word_voc;
    _emb_size = param.word_emb;
    _max_seq_len = param.max_seq_len;
    
    _unigram_num[0] = param.top_unigram;
    _unigram_num[1] = param.sec_unigram;
    _unigram_num[2] = param.thd_unigram;
    
    _bigram_num[0] = param.top_bigram;
    _bigram_num[1] = param.sec_bigram;
    _bigram_num[2] = param.thd_bigram;

    _collocation_num[0] = param.top_collocation;
    _collocation_num[1] = param.sec_collocation;
    _collocation_num[2] = param.thd_collocation;
    int _level_num = 3;
    for (unsigned int i = 0; i < _level_num; i++) {
        _word_num[i] = _unigram_num[i] + _bigram_num[i] + _collocation_num[i];
        _quant_dict[i] = NULL;
    }

    _chnl_num[0] = 1;                 // log quant
    _chnl_num[1] = _emb_size / 2;     // 2d8b product quant
    _chnl_num[2] = _emb_size / 4;     // 4d12b product quant
    
    _word_len[0] = _emb_size;
    _word_len[1] = _chnl_num[1];
    _word_len[2] = _chnl_num[2] / 2 * 3;
    
    _dict_size[0] = 256;
    _dict_size[1] = 2 * 256;
    _dict_size[2] = 4 * 4096;
    _word_offset[0] = _unigram_num[0];
    _word_offset[1] = _word_offset[0] + _unigram_num[1];
    _word_offset[2] = _word_offset[1] + _unigram_num[2];
    
    _word_offset[3] = _word_offset[2] + _bigram_num[0];
    _word_offset[4] = _word_offset[3] + _bigram_num[1];
    _word_offset[5] = _word_offset[4] + _bigram_num[2];
    
    _word_offset[6] = _word_offset[5] + _collocation_num[0];
    _word_offset[7] = _word_offset[6] + _collocation_num[1];
    _word_offset[8] = _word_offset[7] + _collocation_num[2];

    _real_offset[0] = 0;
    _real_offset[1] = 0;
    _real_offset[2] = 0;

    _real_offset[3] = _unigram_num[0];
    _real_offset[4] = _unigram_num[1];
    _real_offset[5] = _unigram_num[2];

    _real_offset[6] = _unigram_num[0] + _bigram_num[0];
    _real_offset[7] = _unigram_num[1] + _bigram_num[1];
    _real_offset[8] = _unigram_num[2] + _bigram_num[2];

    _buf = new unsigned int[anakin_get_num_procs() * _chnl_num[2]];

    _weights[0] = (const unsigned char*)param.embedding_0->data();
    _weights[1] = (const unsigned char*)param.embedding_1->data();
    _weights[2] = (const unsigned char*)param.embedding_2->data();

    _quant_dict[0] = (const float*)param.quant_dict_0->data();
    _quant_dict[1] = (const float*)param.quant_dict_1->data();
    _quant_dict[2] = (const float*)param.quant_dict_2->data();

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberProductQuantEmbeddingWithVsum<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ProductQuantEmbeddingWithVsumParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberProductQuantEmbeddingWithVsum<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ProductQuantEmbeddingWithVsumParam<X86> &param) {

    auto offset = inputs[0]->get_seq_offset()[0];
    int seq_num =  offset.size() - 1;

    outputs[0]->reshape(Shape({seq_num, _emb_size, 1, 1}, Layout_NCHW));
    
    const OpDataType *input_data = (const OpDataType*)inputs[0]->data();
    OpDataType *output_data = (OpDataType*)outputs[0]->mutable_data();
    memset(output_data, 0, sizeof(OpDataType) * outputs[0]->valid_size());
    std::vector<std::vector<std::vector<int>>> real_index;
    real_index.resize(seq_num);
    #pragma omp parallel for schedule(static)
    for (int seq_id = 0; seq_id  < seq_num; seq_id++) {
        real_index[seq_id].resize(3);
        int cur_len = offset[seq_id+1] - offset[seq_id];
        int len = _max_seq_len == -1 ? cur_len : std::min(cur_len, _max_seq_len);
        for (int i = 0; i < len; i++) {
            int word_idx = static_cast<int>(input_data[offset[seq_id] + i]);
            int real_idx = 0;
            int case_idx = 0;
            get_cur_idx(word_idx, _word_offset, _real_offset, 9, &real_idx, &case_idx);
            real_index[seq_id][case_idx].push_back(real_idx);
        }
    }
    #pragma omp parallel for schedule(static)
    for (int seq_id = 0; seq_id  < seq_num; seq_id++) {
        auto tmp_buf = _buf + anakin_get_thread_num() * _chnl_num[2];
        auto tmp_out_data = output_data + seq_id * _emb_size;
        
        memset(tmp_out_data, 0, sizeof(OpDataType)*_emb_size);
        //case 0:
        for (int i = 0; i < real_index[seq_id][0].size(); i++) {
            const unsigned char* word_pos = _weights[0] + real_index[seq_id][0][i] * _word_len[0];
            for (int j = 0; j < _word_len[0]; j++) {
                tmp_out_data[j] += _quant_dict[0][word_pos[j]];
            }
        }
        //case 1:
        for (int i = 0; i < real_index[seq_id][1].size(); i++) {
            const unsigned char* word_pos = _weights[1] + real_index[seq_id][1][i] * _word_len[1];
            for (int j = 0; j < _chnl_num[1]; j++) {
                const float * curr_dict = _quant_dict[1] + j * _dict_size[1] + word_pos[j] * 2;
                auto tmp_out = tmp_out_data  + j * 2;
                tmp_out[0] += curr_dict[0];
                tmp_out[1] += curr_dict[1];
            }
        }
        //case 2:
        for (int i = 0; i < real_index[seq_id][2].size(); i++) {
            const unsigned char* word_pos = _weights[2] + real_index[seq_id][2][i] * _word_len[2];
            decode_4d12b(word_pos, _word_len[2], tmp_buf, _chnl_num[2]);
            for (int j = 0; j < _chnl_num[2]; j++) {
               const float * curr_dict = _quant_dict[2] + j * _dict_size[2] + tmp_buf[j] * 4;
                auto tmp_out = tmp_out_data  + j * 4;
                tmp_out[0] += curr_dict[0];
                tmp_out[1] += curr_dict[1];
                tmp_out[2] += curr_dict[2];
                tmp_out[3] += curr_dict[3];
            }
        }
    }
            
    std::vector<int> out_offset;
    for (int i = 0; i < seq_num; i++) {
        out_offset.push_back(i);
    }
    out_offset.push_back(seq_num);
    outputs[0]->set_seq_offset(std::vector<std::vector<int>>{out_offset});
    return SaberSuccess;
}

template class SaberProductQuantEmbeddingWithVsum<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberProductQuantEmbeddingWithVsum, ProductQuantEmbeddingWithVsumParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberProductQuantEmbeddingWithVsum, ProductQuantEmbeddingWithVsumParam, X86, AK_INT8);
}
} // namespace anakin
