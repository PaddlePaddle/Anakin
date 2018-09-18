
#include "saber/funcs/impl/x86/saber_topk_pooling.h"
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberTopKPooling<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        TopKPoolingParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberTopKPooling<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        TopKPoolingParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberTopKPooling<X86, OpDtype>::get_topk(std::vector<OpDataType>& src,
        int top_k, int real_k, OpDataType* dst) {
    for (int k = 0; k < real_k; k++) {
        float max_data = -1e10;
        int max_index = -1;
        for (int i = 0; i < src.size(); i++) {
            if (max_data < src[i]) {
                max_index = i;
                max_data = src[i];
            }   
        }
        src[max_index] = -1e10;
        dst[k] = max_data;
    }
    for (int k = real_k; k < top_k; k++) {
       dst[k] = (OpDataType) 0.f;
    }
}


template <DataType OpDtype>
SaberStatus SaberTopKPooling<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        TopKPoolingParam<X86> &param) {
    CHECK_EQ(inputs.size(), 2) <<"topk pooling need two inputs";
    auto height_offset = inputs[1]->get_seq_offset()[0];
    auto width_offset = inputs[0]->get_seq_offset()[0];

    const OpDataType* input_data = (const OpDataType*)inputs[0]->data();
    OpDataType* output_data = (OpDataType*) outputs[0]->data();

    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height_stride = inputs[0]->height();
    int width_stride = inputs[0]->width();

    int top_k = param.top_k;
    int feat_map_num = param.feat_map_num;
    CHECK_EQ(feat_map_num, channel) <<"feat map num is not valid";

    Shape output_shape(std::vector<int>{num, channel, top_k, 1});
    outputs[0]->reshape(output_shape);
    
    for (int i = 0; i < num; i++) {
        int height = height_offset[i + 1] - height_offset[i];
        int width = width_offset[i + 1] - width_offset[i];
        int real_k = top_k < height * width  ? top_k : height * width;
        int feat_map_size = height_stride * width_stride;
        for (int c = 0; c < channel; c++) {
            OpDataType* tmp_out_data = output_data + (i * channel + c) * top_k;
            OpDataType* tmp_in_data = input_data + (i * channel + c) * feat_map_size;
            std::vector<OpDataType> vec;

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    auto value =  tmp_in_data[h * width_stride + w];
                    vec.push_back(value);
                }
            }
            get_topk(vec, top_k, real_k, tmp_out_data);
        }
    }

    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;
}

template class SaberTopKPooling<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberTopKPooling, TopKPoolingParam, X86, AK_INT16);
DEFINE_OP_TEMPLATE(SaberTopKPooling, TopKPoolingParam, X86, AK_INT8);
}
} // namespace anakin
