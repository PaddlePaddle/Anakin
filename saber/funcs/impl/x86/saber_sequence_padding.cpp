
#include "saber/funcs/impl/x86/saber_sequence_padding.h"
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberSequencePadding<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SequencePaddingParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberSequencePadding<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SequencePaddingParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberSequencePadding<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SequencePaddingParam<X86> &param) {

    size_t len = inputs[0]->valid_size();
    OpDataType *input_data = (OpDataType*)inputs[0]->mutable_data();
    OpDataType *output_data = (OpDataType*)outputs[0]->mutable_data();
    int max_len = 0;
    auto seq_offset = inputs[0]->get_seq_offset()[0];
    int seq_num = seq_offset.size() - 1;
    int emb_size = inputs[0]->count_valid(1, inputs[0]->dims());
    for (int i = 0; i < seq_num; i++) {
        int cur_len = seq_offset[i+1] - seq_offset[i];
        max_len = cur_len > max_len ? cur_len : max_len;
    }
    Shape out_shape = inputs[0]->valid_shape();
    out_shape[0] = seq_num * max_len;
    outputs[0]->reshape(out_shape);
    for (size_t i = 0; i < seq_num; i++) {
        int start = i * max_len * emb_size;
        int cur_len = seq_offset[i+1] - seq_offset[i];
        int pad_start =  start + cur_len * emb_size;
        int pad_num = max_len - cur_len;
        memcpy(output_data + start, input_data + seq_offset[i] * emb_size, cur_len * emb_size * sizeof(OpDataType));
        if (pad_num > 0) {
            memset(output_data + pad_start, 0, pad_num * emb_size * sizeof(OpDataType));
        }
    }
    
    std::vector<int> out_offset;
    for (int i = 0; i < seq_num + 1; i++) {
        out_offset.push_back(i * max_len);
    }
    outputs[0]->set_seq_offset({out_offset});

    return SaberSuccess;
}

template class SaberSequencePadding<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSequencePadding, SequencePaddingParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSequencePadding, SequencePaddingParam, X86, AK_INT8);
}
} // namespace anakin
