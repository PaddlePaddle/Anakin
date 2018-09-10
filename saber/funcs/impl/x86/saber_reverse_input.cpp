
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/x86/saber_reverse_input.h"

namespace anakin {
namespace saber {

template<DataType OpDtype>
SaberStatus SaberReverseInput<X86, OpDtype>::init(const std::vector<OpTensor*>& inputs,
                         std::vector<OpTensor*>& outputs,
                         EmptyParam<X86> &param,
                         Context<X86> &ctx) {
    return create(inputs,outputs,param,ctx);
};
template<DataType OpDtype>
SaberStatus SaberReverseInput<X86, OpDtype>::create(const std::vector<OpTensor*>& inputs,
                           std::vector<OpTensor*>& outputs,
                           EmptyParam<X86> &param,
                           Context<X86> &ctx) {
    return SaberSuccess;
};
template<DataType OpDtype>
SaberStatus SaberReverseInput<X86, OpDtype>::dispatch(const std::vector<OpTensor*>& inputs,
                             std::vector<OpTensor*>& outputs,
                             EmptyParam<X86> &param) {
    int input_size=inputs.size();
    for(int input_id=0;input_id<input_size;++input_id){
        std::vector<std::vector<int>> offset_vec=inputs[input_id]->get_seq_offset();
        std::vector<int> offset=offset_vec[offset_vec.size()-1];
        const OpDataType* in= static_cast<const OpDataType*>(inputs[input_id]->data());
        OpDataType* out=static_cast<OpDataType*>(outputs[input_id]->mutable_data());
        for(int sequence_id=0;sequence_id<offset.size()-1;sequence_id++){
            int start=offset[sequence_id];
            int end=offset[sequence_id+1]-1;
            for(int index=0;index<=end-start;index++){
                out[end-index]=in[start+index];
            }
        }
    }
    return SaberSuccess;

};

template class SaberReverseInput<X86, AK_INT32>;
template class SaberReverseInput<X86, AK_FLOAT>;
template class SaberReverseInput<X86, AK_HALF>;
template class SaberReverseInput<X86, AK_INT8>;

}
}