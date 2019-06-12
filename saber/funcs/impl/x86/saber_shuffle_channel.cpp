#include "saber/funcs/impl/x86/saber_shuffle_channel.h"

namespace anakin{

namespace saber{

template <typename Dtype>
void shuffle_kernel(Dtype* output, const Dtype* input, int group_row, int group_col, int len) {
    for (int i = 0; i < group_row; ++i) {
        for (int j = 0; j < group_col; ++j) {
            const Dtype* p_i = input + (i * group_col + j) * len;
            Dtype* p_o = output + (j * group_row + i) * len;
            memcpy(p_o, p_i, len * sizeof(Dtype));
        }
    }
}

template <>
SaberStatus SaberShuffleChannel<X86, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ShuffleChannelParam<X86> &param) {


    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int fea_size = channel * height * width;
    int spatial_size = height * width;

    int group_row = param.group;
    int group_col = channel / param.group;
    const float* din = static_cast<const float*>(inputs[0]->data());
    float* dout = static_cast<float*>(outputs[0]->data());
    for (int i = 0; i < num; ++i) {
        shuffle_kernel(dout + i * fea_size, din + i * fea_size, group_row, group_col, spatial_size);
    }

    return SaberSuccess;
}
template <>
SaberStatus SaberShuffleChannel<X86, AK_INT8>::dispatch(\
        const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ShuffleChannelParam<X86> &param) {


    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int fea_size = channel * height * width;
    int spatial_size = height * width;

    int group_row = param.group;
    int group_col = channel / param.group;
    const char* din = static_cast<const char*>(inputs[0]->data());
    char* dout = static_cast<char*>(outputs[0]->data());
    for (int i = 0; i < num; ++i) {
        shuffle_kernel(dout + i * fea_size, din + i * fea_size, group_row, group_col, spatial_size);
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberShuffleChannel, ShuffleChannelParam, X86, AK_HALF);

} //namespace anakin

} //namespace anakin
