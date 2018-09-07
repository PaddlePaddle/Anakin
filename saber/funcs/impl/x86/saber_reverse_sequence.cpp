
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/x86/saber_reverse_sequence.h"

namespace anakin {
namespace saber {

template<DataType OpDtype>
SaberStatus SaberReverseSequence<X86, OpDtype>::init(const std::vector<OpTensor*>& inputs,
                                                  std::vector<OpTensor*>& outputs,
                                                  EmptyParam<X86> &param,
                                                  Context<X86> &ctx) {
    return create(inputs,outputs,param,ctx);
};
template<DataType OpDtype>
SaberStatus SaberReverseSequence<X86, OpDtype>::create(const std::vector<OpTensor*>& inputs,
                                                    std::vector<OpTensor*>& outputs,
                                                    EmptyParam<X86> &param,
                                                    Context<X86> &ctx) {
    int input_size=inputs.size();
    CHECK_EQ(input_size,1)<<"only support one input now";
    return SaberSuccess;
};
template<DataType OpDtype>
SaberStatus SaberReverseSequence<X86, OpDtype>::dispatch(const std::vector<OpTensor*>& inputs,
                                                      std::vector<OpTensor*>& outputs,
                                                      EmptyParam<X86> &param) {
    int input_size=inputs.size();
    CHECK_EQ(input_size,1)<<"only support one input now";

    std::vector<std::vector<int>> offset_vec=inputs[0]->get_seq_offset();
    std::vector<int> offset=offset_vec[offset_vec.size()-1];
    const OpDataType* in= static_cast<const OpDataType*>(inputs[0]->data());
    OpDataType* out=static_cast<OpDataType*>(outputs[0]->mutable_data());
    int batch_size=offset.size()-1;
    int word_size=inputs[0]->valid_shape()[1];
    for (int i = 0; i < batch_size; i++) {
        int seq_len = offset[i + 1] - offset[i];
        int start_word_id=offset[i];
        for (int j = 0; j < seq_len; j++) {
            int output_offset = word_size * (start_word_id + seq_len - j - 1);
            int input_offset = word_size * (start_word_id + j);
            memcpy(out + output_offset, in + input_offset, word_size * sizeof(OpDataType));
        }
    }
    return SaberSuccess;

};

template class SaberReverseSequence<X86, AK_INT32>;
template class SaberReverseSequence<X86, AK_FLOAT>;
template class SaberReverseSequence<X86, AK_HALF>;
template class SaberReverseSequence<X86, AK_INT8>;

}
}