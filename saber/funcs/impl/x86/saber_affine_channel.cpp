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
    outputs[0]->reshape(outputs[0]->valid_shape());
    
    const OpDataType* src = (const OpDataType*)inputs[0]->data();
    const OpDataType* scale = (const OpDataType*)param.weight()->data();
    const OpDataType* bias = (const OpDataType*)param.bias()->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();
    int channel_idx = inputs[0]->channel_index();
    int channel = inputs[0]->channel();
    CHECK_EQ(param.weight()->valid_size(), channel) << "affine channel input scale dims are not valid";
    CHECK_EQ(param.bias()->valid_size(), channel) << "affine channel input bias dims are not valid";
    int outer_num = inputs[0]->count_valid(0, channel_idx);
    int inner_num = inputs[0]->count_valid(channel_idx+1, inputs[0]->dims());
    int id = 0;
    //for (int i = 0; i < outputs[0]->valid_size(); i++) {
    //    dst[i] = 0.1f;
    //}
    for (int i = 0; i < outer_num; i++) {
        for (int j = 0; j < channel; j++) {
            for (int k = 0; k < inner_num; k++) {
                dst[id] = src[id] * scale[j] + bias[j];
                id++;
                //LOG(INFO) << "id" << id << " channel:" << channel << "inner_num: " << inner_num << " j: " << j;
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
