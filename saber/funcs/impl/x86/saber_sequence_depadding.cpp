
#include "saber/funcs/impl/x86/saber_sequence_depadding.h"
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberSequenceDePadding<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SequenceDePaddingParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberSequenceDePadding<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SequenceDePaddingParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberSequenceDePadding<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SequenceDePaddingParam<X86> &param) {
    typedef typename DataTrait<X86, OpDtype>::Dtype OpDataType;
    OpDataType *input_data = (OpDataType*)inputs[0]->mutable_data();
    OpDataType *output_data = (OpDataType*)outputs[0]->mutable_data();
    auto pad_offset = inputs[0]->get_seq_offset()[0];
    auto src_offset = inputs[1]->get_seq_offset()[0];
    int seq_num = src_offset.size() - 1;
    int emb_size = inputs[0]->count_valid(1, inputs[0]->dims());

    for (size_t i = 0; i < seq_num; i++) {
        int src_len_i = src_offset[i+1] - src_offset[i];
        int pad_len_i = pad_offset[i+1] - pad_offset[i];
        CHECK_LE(src_len_i, pad_len_i) << "pad sequence length is bigger than source sequence length";
        memcpy(output_data + src_offset[i] * emb_size, input_data + i * pad_len_i * emb_size, src_len_i * emb_size * sizeof(OpDataType));
    }

    return SaberSuccess;
}

template class SaberSequenceDePadding<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSequenceDePadding, SequenceDePaddingParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSequenceDePadding, SequenceDePaddingParam, X86, AK_INT8);
}
} // namespace anakin
