#include "saber/funcs/impl/arm/saber_shuffle_channel.h"
#include "saber/funcs/type_trans.h"

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
SaberStatus SaberShuffleChannel<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        ShuffleChannelParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int fea_size = channel * height * width;
    int spatial_size = height * width;

    int group_row = param.group;
    int group_col = channel / param.group;
    const float* din = nullptr;
    float* dout = nullptr;
    if (inputs[0]->get_dtype() == AK_INT8) {
        _tmp_in.set_dtype(AK_FLOAT);
        _tmp_in.reshape(inputs[0]->valid_shape());
        trans_tensor_dtype<ARM, AK_INT8, AK_FLOAT>(*inputs[0], _tmp_in, inputs[0]->get_scale()[0], 1.f, {1.f});
        din = static_cast<const float*>(_tmp_in.data());
    } else {
        din = static_cast<const float*>(inputs[0]->data());
    }
    if (outputs[0]->get_dtype() == AK_INT8){
        _tmp_out.set_dtype(AK_FLOAT);
        _tmp_out.reshape(outputs[0]->valid_shape());
        dout = static_cast<float*>(_tmp_out.mutable_data());
        for (int i = 0; i < num; ++i) {
            shuffle_kernel(dout + i * fea_size, din + i * fea_size, group_row, group_col, spatial_size);
        }
        trans_tensor_dtype<ARM, AK_FLOAT, AK_INT8>(_tmp_out, *outputs[0], outputs[0]->get_scale()[0], 1.f, {1.f});
    } else{
        dout = static_cast<float*>(outputs[0]->data());
        for (int i = 0; i < num; ++i) {
            shuffle_kernel(dout + i * fea_size, din + i * fea_size, group_row, group_col, spatial_size);
        }
    }
    for (int i = 0; i < num; ++i) {
        shuffle_kernel(dout + i * fea_size, din + i * fea_size, group_row, group_col, spatial_size);
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "ShuffleChannel fp32: " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("ShuffleChannel", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
template <>
SaberStatus SaberShuffleChannel<ARM, AK_INT8>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        ShuffleChannelParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int fea_size = channel * height * width;
    int spatial_size = height * width;

    int group_row = param.group;
    int group_col = channel / param.group;
    const char* din = nullptr;
    char* dout = nullptr;
    if (inputs[0]->get_dtype() == AK_FLOAT) {
        _tmp_in.set_dtype(AK_INT8);
        _tmp_in.reshape(inputs[0]->valid_shape());
        trans_tensor_dtype<ARM, AK_FLOAT, AK_INT8>(*inputs[0], _tmp_in, inputs[0]->get_scale()[0], 1.f, {1.f});
        din = static_cast<const char*>(_tmp_in.data());
    } else {
        din = static_cast<const char*>(inputs[0]->data());
    }
    if (outputs[0]->get_dtype() == AK_FLOAT){
        _tmp_out.set_dtype(AK_INT8);
        _tmp_out.reshape(outputs[0]->valid_shape());
        dout = static_cast<char*>(_tmp_out.mutable_data());
        for (int i = 0; i < num; ++i) {
            shuffle_kernel(dout + i * fea_size, din + i * fea_size, group_row, group_col, spatial_size);
        }
        trans_tensor_dtype<ARM, AK_INT8, AK_FLOAT>(_tmp_out, *outputs[0], outputs[0]->get_scale()[0], 1.f, {1.f});
    } else{
        dout = static_cast<char*>(outputs[0]->data());
        for (int i = 0; i < num; ++i) {
            shuffle_kernel(dout + i * fea_size, din + i * fea_size, group_row, group_col, spatial_size);
        }
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "ShuffleChannel int8: " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("ShuffleChannel", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberShuffleChannel, ShuffleChannelParam, ARM, AK_HALF);

} //namespace anakin

} //namespace anakin
