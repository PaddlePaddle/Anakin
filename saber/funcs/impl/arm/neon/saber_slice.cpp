#include "saber/funcs/impl/arm/saber_slice.h"
#include "saber/funcs/type_trans.h"

namespace anakin{

namespace saber{
template <>
SaberStatus SaberSlice<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        SliceParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    DataType tensor_in_type = inputs[0]->get_dtype();
    const float* din = nullptr;
    float* dout = nullptr;
    int offset_slice_axis = 0;
    const int in_slice_axis = inputs[0]->valid_shape()[param.axis];
    for (int i = 0; i < outputs.size(); ++i) {
        if (tensor_in_type == AK_INT8) {
            _tmp_in.set_dtype(AK_FLOAT);
            _tmp_in.reshape(inputs[0]->valid_shape());
            trans_tensor_dtype<ARM, AK_INT8, AK_FLOAT>(*inputs[0], _tmp_in, outputs[i]->get_scale()[0], 1.f, {1.f});
            din = static_cast<const float*>(_tmp_in.data());
        } else {
            din = static_cast<const float*>(inputs[0]->data());
        }
        if (outputs[i]->get_dtype() == AK_INT8) {
            _tmp_out.set_dtype(AK_FLOAT);
            _tmp_out.reshape(outputs[i]->valid_shape());
            dout = static_cast<float*>(_tmp_out.mutable_data());
        } else {
            dout = static_cast<float*>(outputs[i]->mutable_data());
        }

        const int out_slice_axis = outputs[i]->valid_shape()[param.axis];
        for (int n = 0; n < _slice_num; ++n) {
            const int out_offset = n * out_slice_axis * _slice_size;
            const int in_offset = (n * in_slice_axis + offset_slice_axis) * _slice_size;
            memcpy((void*)(dout + out_offset), (void*)(din + in_offset), \
                sizeof(float) * out_slice_axis * _slice_size);
        }
        offset_slice_axis += out_slice_axis;
        if (outputs[i]->get_dtype() == AK_INT8) {
            trans_tensor_dtype<ARM, AK_FLOAT, AK_INT8>(_tmp_out, *outputs[i], outputs[i]->get_scale()[0], 1.f, {1.f});
        }
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Slice fp32: " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("Slice", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
template <>
SaberStatus SaberSlice<ARM, AK_INT8>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        SliceParam<ARM> &param) {

#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    DataType tensor_in_type = inputs[0]->get_dtype();
    const char* din = nullptr;
    char* dout = nullptr;
    int offset_slice_axis = 0;
    const int in_slice_axis = inputs[0]->valid_shape()[param.axis];
    for (int i = 0; i < outputs.size(); ++i) {
        if (tensor_in_type == AK_FLOAT) {
            _tmp_in.set_dtype(AK_INT8);
            _tmp_in.reshape(inputs[0]->valid_shape());
            trans_tensor_dtype<ARM, AK_FLOAT, AK_INT8>(*inputs[0], _tmp_in, outputs[i]->get_scale()[0], 1.f, {1.f});
            din = static_cast<const char*>(_tmp_in.data());
        } else {
            din = static_cast<const char*>(inputs[0]->data());
        }
        DataType tensor_out_type = outputs[i]->get_dtype();
        if (tensor_out_type == AK_FLOAT) {
            _tmp_out.set_dtype(AK_INT8);
            _tmp_out.reshape(outputs[i]->valid_shape());
            dout = static_cast<char*>(_tmp_out.mutable_data());
        } else {
            dout = static_cast<char*>(outputs[i]->mutable_data());
        }
        const int out_slice_axis = outputs[i]->valid_shape()[param.axis];
        for (int n = 0; n < _slice_num; ++n) {
            const int out_offset = n * out_slice_axis * _slice_size;
            const int in_offset = (n * in_slice_axis + offset_slice_axis) * _slice_size;
            memcpy((void*)(dout + out_offset), (void*)(din + in_offset), \
                sizeof(char) * out_slice_axis * _slice_size);
        }
        offset_slice_axis += out_slice_axis;
        if (tensor_out_type == AK_FLOAT) {
            trans_tensor_dtype<ARM, AK_INT8, AK_FLOAT>(_tmp_out, *outputs[i], outputs[i]->get_scale()[0], 1.f, {1.f});
        }
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Slice int8: " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    ops.ops = 0;
    ops.ts = ts;
    OpTimer::add_timer("Slice", ops);
    OpTimer::add_timer("total", ops);
#endif

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberSlice, SliceParam, ARM, AK_HALF);

} //namespace anakin

} //namespace anakin
