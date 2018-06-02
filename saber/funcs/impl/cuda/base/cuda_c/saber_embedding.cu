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

template <>
SaberStatus SaberEmbedding<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, \
        NCHW, NCHW, NCHW>::dispatch( \
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        EmbeddingParam<OpTensor>& param) {

    const InDataType *in_data = (const InDataType*)inputs[0]->data();
    const OpDataType *op_data = (const InDataType*)(param.weight()->data());
    OutDataType *out_data = (OutDataType*)outputs[0]->mutable_data();

    const int count = outputs[0]->valid_size();
    cudaStream_t cuda_stream = this->_ctx.get_compute_stream();

    ker_embedding_fwd<InDataType, OpDataType, OutDataType>
            <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
            out_data, in_data, op_data, param.word_num, param.emb_dim, inputs[0]->num(),
            param.padding_idx, outputs[0]->valid_size());

    CUDA_POST_KERNEL_CHECK;
        return SaberSuccess;
}

}
}
