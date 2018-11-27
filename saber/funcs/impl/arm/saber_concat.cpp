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
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberConcat, ConcatParam, ARM, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConcat, ConcatParam, ARM, AK_INT8);
//template class SaberConcat<ARM, AK::FLOAT>;

} //namespace anakin

} //namespace anakin
