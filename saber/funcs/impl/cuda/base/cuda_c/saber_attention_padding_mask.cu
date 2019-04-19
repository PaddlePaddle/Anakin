#include "saber/funcs/impl/cuda/saber_attention_padding_mask.h"
#include "saber/core/tensor_op.h"
#define BUILD_DEV __device__

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void ker_attention_padding_mask_fwd(Dtype * out_data,
                             const Dtype* attn_data,
                             const int* src_offset,
                             const int attn_seq_num,
                             const int attn_seq_len,
                             const int src_seq_num,
                             const int src_seq_len,
                             const Dtype mask,
                             const int count) {
    CUDA_KERNEL_LOOP(tid, count) {
        int src_word_id =  tid % src_seq_len;
        int tmp_tid = tid / src_seq_len;
        int attn_seq_id = tmp_tid / attn_seq_len;
        int attn_word_id = tmp_tid % attn_seq_len; 
        int src_seq_id = attn_seq_id % src_seq_num;
        int cur_len = src_offset[src_seq_id+1] - src_offset[src_seq_id];
        if (src_word_id >= cur_len) {
            out_data[tid] = mask;
        } else {
            out_data[tid] = attn_data[tid];
        }
    }
}

template <DataType OpDtype>
SaberStatus SaberAttentionPaddingMask<NV, OpDtype>::create( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        AttentionPaddingMaskParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberAttentionPaddingMask<NV, OpDtype>::init( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        AttentionPaddingMaskParam<NV>& param, Context<NV>& ctx) {
    _src_offset.set_dtype(AK_INT32);
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberAttentionPaddingMask<NV, OpDtype>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        AttentionPaddingMaskParam<NV>& param) {

    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();

    const OpDataType *attn_data = (const OpDataType*)inputs[0]->data();
    const OpDataType *src_data = (const OpDataType*)inputs[1]->data();
    OpDataType *out_data = (OpDataType*)outputs[0]->mutable_data();

    const int count = outputs[0]->valid_size();
    int attn_seq_num = inputs[0]->get_seq_offset()[0].size() - 1;
    int attn_seq_len = inputs[0]->get_seq_offset()[0][1];
    int src_seq_len = inputs[0]->count_valid(1, inputs[0]->dims());
    auto src_offset = inputs[1]->get_seq_offset()[0];
    int src_seq_num = src_offset.size() - 1;

    _src_offset.reshape(Shape({src_seq_num+1, 1, 1, 1}, Layout_NCHW));
    int* src_offset_data = (int*)_src_offset.mutable_data();
    cudaMemcpyAsync(src_offset_data, &src_offset[0], sizeof(int) * (src_seq_num+1), cudaMemcpyHostToDevice, cuda_stream);

    ker_attention_padding_mask_fwd<OpDataType><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data,
                attn_data,
                src_offset_data,
                attn_seq_num,
                attn_seq_len,
                src_seq_num,
                src_seq_len,
                param.mask,
                count);

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}


template class SaberAttentionPaddingMask<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberAttentionPaddingMask, AttentionPaddingMaskParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberAttentionPaddingMask, AttentionPaddingMaskParam, NV, AK_INT8);
}
}
