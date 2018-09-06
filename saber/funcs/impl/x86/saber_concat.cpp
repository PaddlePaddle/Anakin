#include "saber/funcs/impl/x86/saber_concat.h"

namespace anakin{

namespace saber{

template <typename dtype>
void concat_kernel(const int len, const dtype* src, dtype* dst) {
    if (dst != src) {
        memcpy(dst, src, sizeof(dtype) * len);
    }
}

template <DataType OpDtype>
SaberStatus SaberConcat<X86, OpDtype>::dispatch(const std::vector<Tensor<X86>*>& inputs,
            std::vector<Tensor<X86>*>& outputs, ConcatParam<X86> &param) {

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
            concat_kernel<OpDataType>(in_concat_axis * _concat_input_size,
                            din + n * in_concat_axis * _concat_input_size,
                            dout + (n * out_concat_axis + offset_concat_axis)
                                       * _concat_input_size);
        }
        offset_concat_axis += in_concat_axis;
    }
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;
}

template class SaberConcat<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConcat, ConcatParam, X86, AK_INT16);
DEFINE_OP_TEMPLATE(SaberConcat, ConcatParam, X86, AK_INT8);
} //namespace anakin

} //namespace anakin
