
#include "saber/funcs/impl/x86/saber_topk_avg_pooling.h"
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberTopKAvgPooling<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        TopKAvgPoolingParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberTopKAvgPooling<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        TopKAvgPoolingParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberTopKAvgPooling<X86, OpDtype>::get_topk(std::vector<OpDataType>& src,
        int top_k, int real_k, std::vector<OpDataType>& dst) {
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
    return SaberSuccess;
}


template <DataType OpDtype>
SaberStatus SaberTopKAvgPooling<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        TopKAvgPoolingParam<X86> &param) {
    CHECK_EQ(inputs.size(), 3) <<"topk avg pooling need three inputs";
    auto height_offset = inputs[1]->get_seq_offset()[0];
    auto width_offset = inputs[2]->get_seq_offset()[0];

    const OpDataType* input_data = (const OpDataType*)inputs[0]->data();
    OpDataType* output_data = (OpDataType*) outputs[0]->data();

    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height_stride = inputs[0]->height();
    int width_stride = inputs[0]->width();

    int feat_map_num = param.feat_map_num;
    CHECK_EQ(feat_map_num, channel) <<"feat map num is not valid";
    int dim0 = 0;
    if (param.is_pooling_by_row) {
        dim0 = inputs[1]->num();
        outputs[0]->set_seq_offset(inputs[1]->get_seq_offset());
    } else {
        dim0 = inputs[2]->num();
        outputs[0]->set_seq_offset(inputs[2]->get_seq_offset());
    }  
    int num_k = param.top_ks.size();
    int max_k = param.top_ks[num_k - 1];
    auto offset = outputs[0]->get_seq_offset()[0];
    Shape output_shape({offset[offset.size() - 1], channel * num_k, 1, 1});
    outputs[0]->reshape(output_shape);
    
    
    std::vector<OpDataType> vec;
    std::vector<OpDataType> topk_value(max_k);
    std::vector<OpDataType> cumsum(max_k);
    //topk_value.clear();
    //cumsum.clear();
    //topk_value.resize(max_k);
    //cumsum.resize(max_k);
    for (int i = 0; i < num; i++) {
        int height = height_offset[i + 1] - height_offset[i];
        int width = width_offset[i + 1] - width_offset[i];
        int feat_map_size = height_stride * width_stride;
        if (param.is_pooling_by_row) {
            int real_k = max_k < width  ? max_k : width;
            for (int h = 0; h < height; h++) {
                for (int c = 0; c < channel; c++) {
                    auto tmp_in_data = input_data + ((i *channel + c) * height_stride + h) * width_stride;
                    auto tmp_out_data = output_data + ((height_offset[i] + h) * channel  + c) * num_k;
                    vec.clear();
                    for (int w = 0; w < width; w++) {
                        vec.push_back(tmp_in_data[w]);
                    }
                    get_topk(vec, max_k, real_k, topk_value);
                    cumsum[0] = topk_value[0];
                    for (int m = 1; m < max_k; m++) {
                        cumsum[m] = cumsum[m-1] + topk_value[m];
                    }
                    for (int m = 0; m < param.top_ks.size(); m++) {
         //               LOG(INFO) << "cumsum.size" << cumsum.size() << " top_ks" << param.top_ks.size() << " m " << m << " id " << param.top_ks[m] - 1;
                        if (param.top_ks[m] < 1 || param.top_ks[m] > cumsum.size()) {
                        LOG(INFO) << "cumsum.size" << cumsum.size() << " top_ks" << param.top_ks.size() << " m " << m << " id " << param.top_ks[m] - 1;
        LOG(FATAL) <<" invalid memory ";
 }
                        tmp_out_data[m] = cumsum[param.top_ks[m] - 1] / param.top_ks[m];
                        //tmp_out_data[0] = cumsum[param.top_ks[m] - 1] / param.top_ks[m];
                    }
                }
            }
        } else {
            int real_k = max_k < height  ? max_k : height;
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channel; c++) {
                    auto tmp_in_data = input_data + ((i *channel + c) * height_stride) * width_stride + w;
                    auto tmp_out_data = output_data + ((width_offset[i] + w ) * channel +  c) * num_k;
                    vec.clear();
                    for (int h = 0; h < height; h++) {
                        vec.push_back(tmp_in_data[h * width_stride]);
                    }
                    get_topk(vec, max_k, real_k, topk_value);
                    cumsum[0] = vec[topk_value[0]];
                    for (int m = 1; m < max_k; m++) {
                        cumsum[m] = cumsum[m-1] + topk_value[m];
                    }
                    for (int m = 0; m < param.top_ks.size(); m++) {
                        tmp_out_data[m] = cumsum[param.top_ks[m] - 1] / param.top_ks[m];
                    }
                }
            }
        }
    }
    return SaberSuccess;
}

template class SaberTopKAvgPooling<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberTopKAvgPooling, TopKAvgPoolingParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberTopKAvgPooling, TopKAvgPoolingParam, X86, AK_INT8);
}
} // namespace anakin
