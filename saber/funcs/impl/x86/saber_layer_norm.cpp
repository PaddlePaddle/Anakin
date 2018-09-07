#include "saber/funcs/impl/x86/saber_layer_norm.h"
#include <math.h>

namespace anakin{

namespace saber{

template <DataType OpDtype>
SaberStatus SaberLayerNorm<X86, OpDtype>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs, \
    LayerNormParam<X86> &param) {

    const OpDataType* src = (const OpDataType*)inputs[0]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();
    const OpDataType* bias = (const OpDataType*)(param.bias_weights()->data());
    const OpDataType* scale = (const OpDataType*)(param.scale_weights()->data());

    for (int i = 0; i < outer_size; ++i) {
        OpDataType mean = 0;
        OpDataType std = 0;
        const OpDataType* src_ptr = src + i * inner_size;
        OpDataType* dst_ptr = dst + i * inner_size;
        for (int j = 0; j < inner_size; ++j) {
            mean += src_ptr[j];
        }
        mean /= inner_size;
        for (int j = 0; j < inner_size; ++j) {
            std += (src_ptr[j] - mean) * (src_ptr[j] - mean);
        }
        std = std / inner_size;
        //printf("std pre: %.6f\n", std);
        std = 1.f / (sqrtf(std) + param.eps);
        //printf("mean: %.6f, std: %.6f\n", mean, std);
        for (int j = 0; j < inner_size; ++j) {
           dst_ptr[j] = (flag_scale? scale[j] : 1) * (src_ptr[j] - mean) * std + (flag_bias? bias[j] : 0);
        }
    }

    return SaberSuccess;
}

template class SaberLayerNorm<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberLayerNorm, LayerNormParam, X86, AK_INT16);
DEFINE_OP_TEMPLATE(SaberLayerNorm, LayerNormParam, X86, AK_INT8);
} //namespace anakin

} //namespace anakin
