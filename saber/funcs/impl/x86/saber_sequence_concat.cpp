
#include "saber/funcs/impl/x86/saber_sequence_concat.h"
#include "saber/funcs/impl/x86/saber_sequence_concat.h"
#include "mkl.h"
#if defined(__AVX2__) and defined(__FMA__)
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
#endif
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberSequenceConcat<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SequenceConcatParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberSequenceConcat<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SequenceConcatParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberSequenceConcat<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SequenceConcatParam<X86> &param) {
    OpDataType *output_data = (OpDataType*)outputs[0]->mutable_data();
    int emb_size = inputs[0]->valid_size() / inputs[0]->num();
    int seq_num = inputs[0]->get_seq_offset()[0].size() - 1;
    for (int i = 1; i < inputs.size(); i++) {
        int cur_emb_size = inputs[i]->valid_size() / inputs[i]->num();
        int cur_seq_num  = inputs[i]->get_seq_offset()[0].size() - 1;
        CHECK_EQ(emb_size, cur_emb_size) << "sequence concat emb size must be the same";
        CHECK_EQ(seq_num, cur_seq_num) << "sequence concat seq num must be the same";
    }

    for (int i = 0; i < seq_num; i++) {
        for (int j = 0; j < inputs.size(); j++) {
            size_t cur_len = inputs[j]->get_seq_offset()[0][i+1] - inputs[j]->get_seq_offset()[0][i];

            const OpDataType *input_data = (const OpDataType*)inputs[j]->data() + inputs[j]->get_seq_offset()[0][i] * emb_size;
            memcpy(output_data, input_data, sizeof(OpDataType) * cur_len * emb_size);
            output_data += cur_len * emb_size;
        }
    }

    return SaberSuccess;
}

template class SaberSequenceConcat<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSequenceConcat, SequenceConcatParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSequenceConcat, SequenceConcatParam, X86, AK_INT8);
}
} // namespace anakin
