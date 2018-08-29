#include "saber/funcs/impl/x86/saber_lrn.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {

/**
 * @brief get sum of x^2 between channels [size elements]
 * 
 * @tparam dtype 
 * @tparam TargetType_H 
 * @param input 
 * @param num_id:  the i-th graph.
 * @param channel_id: the j-th channel within i-th graph.
 * @param offset_within_channel: the pixel's offset within a channel.
 * @param offset_num: the first address of i-th graph.
 * @param c 
 * @param h 
 * @param w 
 * @param size 
 * @return dtype 
 */
template <typename dtype, typename TargetType_H>
dtype lrn_square(const Tensor<TargetType_H>& input, 
                 int channel_id, 
                 int offset_within_channel, 
                 int offset_num, 
                 int c, 
                 int h, 
                 int w, 
                 int size) {
    int pre_pad = (size - 1) / 2;
    dtype res = 0;
    const dtype* src = (const dtype*)input.data() + offset_num;

    //handle left channels with padding situation.
    if (channel_id - pre_pad < 0) {
        for (int i = 0; i <= channel_id; i++) {
            res += src[i * h * w + offset_within_channel] * src[i * h * w + offset_within_channel];
        }
    }
    //handle left channels.
    if (channel_id - pre_pad >= 0) {
        for (int i = channel_id - pre_pad; i <= channel_id; i++) {
            res += src[i * h * w + offset_within_channel] * src[i * h * w + offset_within_channel];
        }
    }

    //handle right channels.
    if (channel_id + pre_pad < c) {
        for (int i = channel_id + 1; i <= channel_id + pre_pad; i++) {
            res += src[i * h * w + offset_within_channel] * src[i * h * w + offset_within_channel];
        }
    }

    //handle right channels with padding situation.
    if (channel_id + pre_pad >= c && channel_id + 1 < c) {
        for (int i = channel_id + 1; i < c; i++) {
            res += src[i * h * w + offset_within_channel] * src[i * h * w + offset_within_channel];
        }
    }
    return res;
}

/**
 *  @brief  formula: (k + alpha * sigma((x(i))^2)) ^ beta.
 *             where,
 *                   local_size = 5(default), means 5 channels in succession.
 *                   sigma((x(i))^2): sum of x^2 of k channels in succession.
 * 
 * 
 */
template <DataType OpDtype>
SaberStatus SaberLrn<X86, OpDtype>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs, \
    LrnParam<X86>& param) {
    
    const OpDataType* src = (const OpDataType*)inputs[0]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();
    OpDataType square;
    int offset_within_channel = 0;
    int offset_num = 0;
    int dst_id;
    int size = param.local_size;
    int pre_pad = (size - 1) / 2;

    for (int i = 0; i < N; i++) {
        offset_num = i * C * H * W;
        for (int j = 0; j < C; j++) {
            for (int l = 0; l < H; l++) {
                for (int m = 0; m < W; m++) {
                    offset_within_channel = l * W + m;
                    dst_id = offset_num + j * H * W + offset_within_channel;
                    square = lrn_square<OpDataType, X86>(*inputs[0], j, offset_within_channel, offset_num, C, H, W, size);
                    dst[dst_id] = src[dst_id] * pow(param.k + param.alpha * square, -param.beta);
                }
            }
        }
    }

    return SaberSuccess;
}

template class SaberLrn<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberLrn, LrnParam, X86, AK_INT16);
DEFINE_OP_TEMPLATE(SaberLrn, LrnParam, X86, AK_INT8);
}
}
