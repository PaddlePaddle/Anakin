#include "saber/funcs/impl/arm/saber_concat.h"

namespace anakin{

namespace saber{

template <typename dtype>
void concat_kernel_arm(const int len, const dtype* src, dtype* dst) {
    if (dst != src) {
        memcpy(dst, src, sizeof(dtype) * len);
    }
}

template <>
SaberStatus SaberConcat<ARM, AK_FLOAT>::dispatch(\
        const std::vector<Tensor<ARM> *>& inputs,
        std::vector<Tensor<ARM> *>& outputs,
        ConcatParam<ARM> &param) {
#ifdef ENABLE_OP_TIMER
    this->_timer.clear();
    this->_timer.start(*this->_ctx);
#endif
    int input_size = inputs.size();

    //! get output data, valid shape and stride shape
    int offset_concat_axis = 0;
    Shape out_shape = outputs[0]->valid_shape();
    const int out_concat_axis = out_shape[param.axis];

    if (inputs.size() == 1) {
        outputs[0]->copy_from(*inputs[0]);
        return SaberSuccess;
    }

    OpDataType* dout = (OpDataType*)outputs[0]->mutable_data();

    for (int i = 0; i < input_size; ++i) {
        Shape sh_in = inputs[i]->valid_shape();
        const OpDataType* din = (const OpDataType*)inputs[i]->data();
        const int in_concat_axis = sh_in[param.axis];
        for (int n = 0; n < _num_concats; ++n) {
            concat_kernel_arm<OpDataType>(in_concat_axis * _concat_input_size,
                            din + n * in_concat_axis * _concat_input_size,
                            dout + (n * out_concat_axis + offset_concat_axis)
                                       * _concat_input_size);
        }
        offset_concat_axis += in_concat_axis;
    }
#ifdef ENABLE_OP_TIMER
    this->_timer.end(*this->_ctx);
    float ts = this->_timer.get_average_ms();
    LOG(INFO) << "Cast : " << this->_op_name.c_str() << " : time: " << ts;
    GOPS ops;
    //fixme
    float sum  = 0;
    for (int i = 0; i < inputs.size(); i++){
        sum += inputs[i]->valid_size();
    }
    ops.ops = 2.f * sum;
    ops.ts = ts;
    OpTimer::add_timer("Cast", ops);
    OpTimer::add_timer("total", ops);
#endif
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberConcat, ConcatParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConcat, ConcatParam, ARM, AK_INT8);

} //namespace anakin

} //namespace anakin
