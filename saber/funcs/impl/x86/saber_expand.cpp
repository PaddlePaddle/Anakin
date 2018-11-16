#include "saber/funcs/impl/x86/saber_expand.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include <cmath>
namespace anakin {
namespace saber {
/*
 *conv shift compute formula
 *input: inputs[0] {N. M};
 *input: inputs[1] {N, K}
 */

template <DataType OpDtype>
SaberStatus SaberExpand<X86, OpDtype>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs, \
    ExpandParam<X86>& param) {
    
    const OpDataType* src = (const OpDataType*)inputs[0]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();

    auto out_shape = outputs[0]->valid_shape();
    auto in_shape = inputs[0]->valid_shape();
    auto expand_times = param.expand_times;
    int dims = expand_times.size();

    int inner_num = 1;
    int i = dims - 1;
    int outer_num = in_shape.count(0, i);
    inner_num *= in_shape[i];
    for (int j = 0; j < outer_num; j++) {
        for (int k = 0; k < expand_times[i]; k++) {
            memcpy(dst + (j * expand_times[i] + k) * inner_num, src + j * inner_num, sizeof(OpDataType) * inner_num);
        }
    }
    inner_num *= expand_times[i];
    for (int i = dims - 2; i >= 0; i--) {
        int outer_num = in_shape.count(0, i);
        inner_num *= in_shape[i];
        for (int j = 0; j < outer_num; j++) {
            for (int k = 0; k < expand_times[i]; k++) {
                memcpy(dst + (j * expand_times[i] + k) * inner_num, dst + j* inner_num, sizeof(OpDataType) * inner_num);
            }
        }
        inner_num *= expand_times[i];
    }
    return SaberSuccess;
}

template class SaberExpand<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberExpand, ExpandParam, X86, AK_INT16);
DEFINE_OP_TEMPLATE(SaberExpand, ExpandParam, X86, AK_INT8);
}
}
