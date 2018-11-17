#include "saber/funcs/impl/x86/saber_affine_channel.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include <cmath>
namespace anakin {
namespace saber {

/**
 *  @brief  formula: Input 0 X (NCHW or NHWC).
 *             where,Input 1 Scale (C)
 *                   Input 2 Bias  (C)
 *                   Output = Scale * X + Bias.
 * 
 */
template <DataType OpDtype>
SaberStatus SaberAffineChannel<X86, OpDtype>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs, \
    AffineChannelParam<X86>& param) {
    
    const OpDataType* src = (const OpDataType*)inputs[0]->data();
    const OpDataType* scale = (const OpDataType*)inputs[1]->data();
    const OpDataType* bias = (const OpDataType*)inputs[2]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();
    int channel_idx = inputs[0]->channel_index();
    int channel = inputs[0]->channel();
    CHECK_EQ(inputs[1]->valid_size(), channel) << "affine channel input scale dims are not valid";
    CHECK_EQ(inputs[2]->valid_size(), channel) << "affine channel input bias dims are not valid";
    int outer_num = inputs[0]->count_valid(0, channel_idx);
    int inner_num = inputs[0]->count_valid(channel_idx+1, inputs[0]->dims());
    int id = 0;
    for (int i = 0; i < outer_num; i++) {
        for (int j = 0; j < channel; j++) {
            for (int k = 0; k < inner_num; k++) {
                dst[id] = src[id] * scale[j] + bias[j];
                id++;
            }
        }
    }

    return SaberSuccess;
}

template class SaberAffineChannel<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberAffineChannel, AffineChannelParam, X86, AK_INT16);
DEFINE_OP_TEMPLATE(SaberAffineChannel, AffineChannelParam, X86, AK_INT8);
}
}
