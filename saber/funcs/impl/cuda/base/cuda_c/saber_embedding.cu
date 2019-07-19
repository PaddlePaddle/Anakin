#include "saber/funcs/impl/cuda/saber_embedding.h"
#include "cuda_fp16.h"

namespace anakin{
namespace saber{

template<typename InDataType, typename OpDataType, typename OutDataType>
__global__ void ker_embedding_fwd(OutDataType * out_data,
                                const InDataType* in_data,
                                const OpDataType* tabel,
                                const int total_word_num,
                                const int emb_dim,
                                const int word_num,
                                const int padding_idx,
                                const int out_count) {

    CUDA_KERNEL_LOOP(tid, out_count){
        int emb_id =  tid % emb_dim;
        int word_id = tid / emb_dim;
        int word_idx_in_tabel = (int)(in_data[word_id]);
        if (word_idx_in_tabel != padding_idx) {
            out_data[tid] = OutDataType(tabel[word_idx_in_tabel * emb_dim + emb_id]);
        } else {
            out_data[tid] = OutDataType(0.f);
        }
    }
}

template<typename InDataType, typename OpDataType, typename OutDataType>
__global__ void ker_embedding_fwd(OutDataType * pos_out_data,
                                OutDataType* rvs_out_data,
                                const InDataType* in_data,
                                const OpDataType* tabel,
                                const int* word_seq_map,
                                const int* seq_offset,
                                const int total_word_num,
                                const int emb_dim,
                                const int word_num,
                                const int padding_idx,
                                const int out_count) {

    CUDA_KERNEL_LOOP(tid, out_count){
        int emb_id =  tid % emb_dim;
        int word_id = tid / emb_dim;
        int seq_id = word_seq_map[word_id];
        int word_id_in_cur_seq = word_id - seq_offset[seq_id];
        int rvs_word_id = seq_offset[seq_id + 1] - 1 - word_id_in_cur_seq;
        int rvs_out_index = rvs_word_id * emb_dim + emb_id;

        int word_idx_in_tabel = (int)(in_data[word_id]);
        if (word_idx_in_tabel != padding_idx) {
            auto data = tabel[word_idx_in_tabel * emb_dim + emb_id];
            pos_out_data[tid] = OutDataType(data);
            rvs_out_data[rvs_out_index] = OutDataType(data);
        } else {
            pos_out_data[tid] = OutDataType(0.f);
            rvs_out_data[tid] = OutDataType(0.f);
        }
    }
}

template <DataType OpDtype>
SaberStatus SaberEmbedding<NV, OpDtype>::dispatch( \
    const std::vector<Tensor<NV>*>& inputs,
    std::vector<Tensor<NV>*>& outputs,
	EmbeddingParam<NV>& param) {

    CHECK_EQ(inputs[0]->get_dtype(), AK_FLOAT) <<" Embedding only support float inputs.";
    const OpDataType *op_data = (const OpDataType*)(param.weight()->data());

    const int count = outputs[0]->valid_size();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    
    //outputs: chose corresponding informations of words.
    //inputs: word_id [Its type maybe float or int]
    //outputs = weights[inputs[j]].
    if (param.num_direct == 2) {
        auto seq_offset = inputs[0]->get_seq_offset()[0];
        int batch_size = seq_offset.size() - 1;
        
        std::vector<int> word_seq_map;
        for (int i = 0; i < batch_size; i++) {
            for (int j = seq_offset[i]; j < seq_offset[i+1]; j++) {
                word_seq_map.push_back(i);
            }
        }
        _word_seq_map.reshape(Shape({seq_offset[batch_size], 1, 1, 1}));
        _seq_offset.reshape(Shape({batch_size +1, 1,1, 1}));
        cudaMemcpyAsync(_seq_offset.mutable_data(), &seq_offset[0], 
             sizeof(int) * seq_offset.size(),
             cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(_word_seq_map.mutable_data(), &word_seq_map[0],
             sizeof(int) * inputs[0]->valid_size(),
             cudaMemcpyHostToDevice, cuda_stream);
        
        ker_embedding_fwd<float, OpDataType, OpDataType>
        <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    (OpDataType*)outputs[0]->mutable_data(), 
                    (OpDataType*)outputs[1]->mutable_data(), 
                    (const float*)inputs[0]->data(), op_data,
                    (const int*)_word_seq_map.data(),
                    (const int*)_seq_offset.data(),
                    param.word_num, param.emb_dim, inputs[0]->num(),
                    param.padding_idx, outputs[0]->valid_size());
    } else {
        ker_embedding_fwd<float, OpDataType, OpDataType>
        <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    (OpDataType*)outputs[0]->mutable_data(), 
                    (const float*)inputs[0]->data(), op_data, 
                    param.word_num, param.emb_dim, inputs[0]->num(),
                    param.padding_idx, outputs[0]->valid_size());
    }
    for (auto output : outputs) {
        output->set_seq_offset(inputs[0]->get_seq_offset());
    }
    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberEmbedding<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberEmbedding, EmbeddingParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberEmbedding, EmbeddingParam, NV, AK_INT8);
}
}
