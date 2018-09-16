#include "saber/funcs/impl/x86/saber_unpool.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberUnpool<X86, OpDtype>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs,\
    PoolingParam<X86>& param) {

    const InDataType* in_data = (const InDataType*)inputs[0]->data();
    const OutDataType* in_max_index = (const OutDataType*)inputs[1]->data();
    OutDataType* out_data = (OutDataType*)outputs[0]->mutable_data();
    int count = inputs[0]->valid_size();
    int in_c = inputs[0]->channel();
    memset(out_data, 0, outputs[0]->valid_size() * sizeof(OpDataType));
    for(int i = 0; i < count; ++i){
        int num_out = i / _in_n_stride;
        int c_out = (i / _in_c_stride) % in_c;
        int out_index = num_out * _out_n_stride + c_out * _out_c_stride;
        int max_index = in_max_index[i];
        out_data[out_index + max_index] = in_data[i];
    }

    return SaberSuccess;
}

} //namespace saber

} //namespace anakin
