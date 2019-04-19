
#include "saber/funcs/impl/x86/saber_attention_padding_mask.h"
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberAttentionPaddingMask<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        AttentionPaddingMaskParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberAttentionPaddingMask<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        AttentionPaddingMaskParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberAttentionPaddingMask<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        AttentionPaddingMaskParam<X86> &param) {
    auto src_offset = inputs[1]->get_seq_offset()[0];
    auto attn_offset = inputs[0]->get_seq_offset()[0];
    int src_len = inputs[1]->count_valid(1, inputs[1]->dims());
    int attn_seq_num = attn_offset.size() - 1;
    int src_seq_num = src_offset.size() - 1;
    int attn_seq_len = attn_offset[1];
    int src_seq_len = src_offset[1];
    CHECK_EQ(attn_seq_num % src_seq_num, 0) << "Missmatch batch size";

    size_t count = inputs[0]->valid_size();
    OpDataType *attn_data = (OpDataType*)inputs[0]->mutable_data();
    OpDataType *src_data = (OpDataType*)inputs[1]->mutable_data();
    OpDataType *output_data = (OpDataType*)outputs[0]->mutable_data();
    memcpy(output_data, attn_data, count * sizeof(OpDataType));
    for (int i = 0; i < attn_seq_num; ++i) {
        for (int j = 0; j < attn_seq_len; ++j) {
            auto tmp_output_data = output_data + src_seq_len * (attn_seq_len * i + j);
            int src_seq_idx = i % src_seq_num;
            int cur_len = src_offset[src_seq_idx+1]-src_offset[src_seq_idx];
            auto tmp_src_data = src_data + src_seq_idx * src_seq_len;
            for (int k = cur_len; k < src_seq_len; k++) {
                tmp_output_data[k] = param.mask;
            }
        }
    }

    return SaberSuccess;
}

template class SaberAttentionPaddingMask<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberAttentionPaddingMask, AttentionPaddingMaskParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberAttentionPaddingMask, AttentionPaddingMaskParam, X86, AK_INT8);
}
} // namespace anakin
