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
    ker_embedding_fwd<float, OpDataType, OpDataType>
    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                (OpDataType*)outputs[0]->mutable_data(), 
                (const float*)inputs[0]->data(), op_data, 
                param.word_num, param.emb_dim, inputs[0]->num(),
                param.padding_idx, outputs[0]->valid_size());
    
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberEmbedding<NV, AK_FLOAT>;
template class SaberEmbedding<NV, AK_INT8>;

}
}