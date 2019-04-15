#include "saber/funcs/impl/cuda/saber_sequence_depadding.h"
#include "saber/core/tensor_op.h"
#define BUILD_DEV __device__

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void ker_sequence_depadding_fwd(Dtype * out_data,
                             const Dtype* in_data,
                             const int* seq_id_map,
                             const int seq_num,
                             const int max_len,
                             const int emb_size,
                             const int count) {
    CUDA_KERNEL_LOOP(tid, count) {
        int emb_id =  tid % emb_size;
        int word_id = tid / emb_size;
        int seq_id = seq_id_map[word_id];
        out_data[tid] = in_data[seq_id * emb_size + emb_id];
    }
}

template <DataType OpDtype>
SaberStatus SaberSequenceDePadding<NV, OpDtype>::create( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        SequenceDePaddingParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberSequenceDePadding<NV, OpDtype>::init( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        SequenceDePaddingParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberSequenceDePadding<NV, OpDtype>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        SequenceDePaddingParam<NV>& param) {

    const OpDataType *in_data = (const OpDataType*)inputs[0]->data();
    OpDataType *out_data = (OpDataType*)outputs[0]->mutable_data();

    const int count = outputs[0]->valid_size();

    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();

    int max_len = inputs[0]->get_seq_offset()[0][1];
    int seq_num = inputs[0]->get_seq_offset()[0].size() - 1;
    int emb_size = inputs[0]->count_valid(1, inputs[0]->dims());

    auto src_seq_offset = inputs[1]->get_seq_offset()[0];
    auto pad_seq_offset = inputs[0]->get_seq_offset()[0];
    std::vector<int> seq_id_map;
    for (int i = 0;i < seq_num; i++) {
        int cur_len = src_seq_offset[i+1] - src_seq_offset[i];
        for (int j = 0; j < cur_len; j++) {
            seq_id_map.push_back(i * max_len + j);
        }
    }
    int map_size = seq_id_map.size();
    _seq_id_map.reshape(Shape({map_size, 1, 1, 1}, Layout_NCHW));
    int* seq_id_map_data = (int*)_seq_id_map.mutable_data();
    cudaMemcpyAsync(seq_id_map_data, &seq_id_map[0], sizeof(int) * seq_id_map.size(), cudaMemcpyHostToDevice, cuda_stream);

    ker_sequence_depadding_fwd<OpDataType><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data,
                        in_data,
                        seq_id_map_data,
                        seq_num,
                        max_len,
                        emb_size,
                        count);

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}


template class SaberSequenceDePadding<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSequenceDePadding, SequenceDePaddingParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSequenceDePadding, SequenceDePaddingParam, NV, AK_INT8);
}
}
