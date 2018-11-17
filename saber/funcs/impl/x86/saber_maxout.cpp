#include "saber/funcs/impl/x86/saber_maxout.h"

namespace anakin {
namespace saber {

template <DataType OpDtype>
SaberStatus SaberMaxOut<X86, OpDtype>::dispatch(const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    MaxOutParam<X86>& param) {

    const OpDataType* input_ptr = (const OpDataType*)inputs[0]->data();
    OpDataType* output_ptr = (OpDataType*)outputs[0]->mutable_data();

    int batch_size = outputs[0]->num();
    int channel = outputs[0]->channel();
    int height = outputs[0]->height();
    int width = outputs[0]->width();
    int group = param.groups;

    int feature_size = height * width;
    int feature_map_size = feature_size * channel;

    for (int i = 0; i < batch_size; i++) {
        int n_id = i * feature_map_size;
        for (int c = 0; c < channel; c++) {
            int c_id = c * feature_size;
            for (int f = 0; f < feature_size; f++) {
                int src_id = (n_id + c_id) * group + f;
                OpDataType max = input_ptr[src_id];
                for (int g = 0; g < group; g++) {
                    OpDataType tmp = input_ptr[src_id + g * feature_size];
                    max = max < tmp ? tmp : max; 
                }
                output_ptr[n_id + c_id + f] = max;
            }
        }
    }

    return SaberSuccess;
}

template class SaberMaxOut<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberMaxOut, MaxOutParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberMaxOut, MaxOutParam, X86, AK_INT8);

} // namespace saber.
} // namespace anakin.
