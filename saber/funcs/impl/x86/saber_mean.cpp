#include "saber/funcs/impl/x86/saber_mean.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberMean<X86, OpDtype>::dispatch(const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    MeanParam<X86>& param) {

    const OpDataType* input_ptr = (const OpDataType*)inputs[0]->data();
    OpDataType* output_ptr = (OpDataType*)outputs[0]->mutable_data();
    int n = inputs[0]->valid_size();
    OpDataType s = (OpDataType)0.0;

# pragma omp parallel for reduction(+:s)
    for (int i = 0; i < n; i++) {
        s += input_ptr[i];
    }
    s /= n;
    output_ptr[0] = s;

    return SaberSuccess;
}

template class SaberMean<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberMean, MeanParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberMean, MeanParam, X86, AK_INT8);

} // namespace saber.
} // namespace anakin.