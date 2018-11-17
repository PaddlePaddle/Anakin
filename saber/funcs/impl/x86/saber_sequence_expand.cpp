#include "saber/funcs/impl/x86/saber_sequence_expand.h"
#include <cmath>

namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberSequenceExpand<X86, OpDtype>::init(
                const std::vector<OpTensor*>& inputs,
                std::vector<OpTensor*>& outputs,
                SequenceExpandParam<X86>& param,
Context<X86>& ctx) {

    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberSequenceExpand<X86, OpDtype>::create(
                const std::vector<OpTensor*>& inputs,
                std::vector<OpTensor*>& outputs,
                SequenceExpandParam<X86>& param,
Context<X86>& ctx) {
    this->_ctx = &ctx;

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberSequenceExpand<X86, OpDtype>::dispatch(
                const std::vector<OpTensor*>& inputs,
                std::vector<OpTensor*>& outputs,
SequenceExpandParam<X86>& param) {

    // TODO !! need add other types of sequence_expand

    auto ref_offset = inputs[1]->get_seq_offset()[0];
    size_t len = inputs[0]->valid_size();
    OpDataType* input_data = static_cast<const OpDataType* >(inputs[0]->data());
    OpDataType* output_data =  static_cast<OpDataType* >(outputs[0]->mutable_data());
    int dim = inputs[0]->valid_size() / inputs[0]->num();

    if (inputs[0]->get_seq_offset().size() == 0) {
        for (int i = 0; i < ref_offset.size() - 1; i++) {
            for (int j = ref_offset[i]; j < ref_offset[i + 1]; j++) {
                memcpy(output_data + j * dim, input_data + i * dim, sizeof(OpDataType) * dim);
            }
        }

        outputs[0]->set_seq_offset({ref_offset});
    } else {
        std::vector<int> out_offset;
        int cum = 0;
        auto cur_offset = inputs[0]->get_seq_offset()[0];
        for (int i = 0; i < ref_offset.size() - 1; i++) {
            int cur_len = cur_offset[i + 1] - cur_offset[i];

            for (int j = ref_offset[i]; j < ref_offset[i + 1]; j++) {

                memcpy(output_data + cum * dim, input_data + i * dim, sizeof(OpDataType) * dim * cur_len);
                cum += cur_len;
                out_offset.push_back(cum);
            }
        }

        outputs[0]->set_seq_offset({out_offset});
    }

    return SaberSuccess;
}

template class SaberSequenceExpand<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSequenceExpand, SequenceExpandParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSequenceExpand, SequenceExpandParam, X86, AK_INT8);
}
} // namespace anakin
