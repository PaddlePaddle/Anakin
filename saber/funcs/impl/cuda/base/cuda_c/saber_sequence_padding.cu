#include "saber/funcs/impl/cuda/saber_sequence_padding.h"
#include "saber/core/tensor_op.h"
#define BUILD_DEV __device__

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void ker_sequence_padding_fwd(Dtype * out_data,
                             const Dtype* in_data,
                             const int* offset,
                             const int seq_num,
                             const int max_len,
                             const int emb_size,
                             const int count) {
    CUDA_KERNEL_LOOP(tid, count) {
        int emb_id =  tid % emb_size;
        int word_id = tid / emb_size;
        int seq_id = word_id / max_len;
        int word_id_in_seq = word_id % max_len;
        int cur_len = offset[seq_id + 1] - offset[seq_id];
        if (word_id_in_seq < cur_len) {
            out_data[tid] = in_data[(offset[seq_id] + word_id_in_seq) * emb_size + emb_id];
        } else {
            out_data[tid] = 0.f;
        }
    }
}

template <DataType OpDtype>
SaberStatus SaberSequencePadding<NV, OpDtype>::create( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        SequencePaddingParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberSequencePadding<NV, OpDtype>::init( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        SequencePaddingParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    int seq_num = inputs[0]->get_seq_offset()[0].size() - 1;
    Shape offset_shape({seq_num + 1, 1, 1, 1}, Layout_NCHW);
    _in_seq_offset.re_alloc(offset_shape, AK_INT32);
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberSequencePadding<NV, OpDtype>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        SequencePaddingParam<NV>& param) {

    const OpDataType *in_data = (const OpDataType*)inputs[0]->data();
    OpDataType *out_data = (OpDataType*)outputs[0]->mutable_data();

    const int count = outputs[0]->valid_size();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int max_len = outputs[0]->get_seq_offset()[0][1];
    int seq_num = outputs[0]->get_seq_offset()[0].size() - 1;
    int emb_size = inputs[0]->count_valid(1, inputs[0]->dims());
    _in_seq_offset.reshape(Shape({seq_num+1, 1, 1, 1}, Layout_NCHW));
    int* offset_data = (int*)_in_seq_offset.mutable_data();
    auto in_seq_offset = inputs[0]->get_seq_offset()[0];
    cudaMemcpyAsync(offset_data, &in_seq_offset[0], sizeof(int) * in_seq_offset.size(), cudaMemcpyHostToDevice, cuda_stream);

    ker_sequence_padding_fwd<OpDataType><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data,
                        in_data,
                        offset_data,
                        seq_num,
                        max_len,
                        emb_size,
                        count);

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}


template class SaberSequencePadding<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSequencePadding, SequencePaddingParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSequencePadding, SequencePaddingParam, NV, AK_INT8);
}
}
