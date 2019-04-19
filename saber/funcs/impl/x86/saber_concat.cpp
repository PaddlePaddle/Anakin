#include "saber/funcs/impl/x86/saber_concat.h"

namespace anakin {

namespace saber {

template <typename dtype>
void concat_kernel(const int len, const dtype* src, dtype* dst) {
    if (dst != src) {
        memcpy(dst, src, sizeof(dtype) * len);
    }
}
template <>
SaberStatus SaberConcat<X86, AK_FLOAT>::create(const std::vector<Tensor<X86>*>& inputs,
                   std::vector<Tensor<X86>*>& outputs,
                   ConcatParam<X86> &param, Context<X86> &ctx){

    _num_concats = inputs[0]->count_valid(0, param.axis);
    _concat_input_size = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
    return SaberSuccess;
}

template <>
SaberStatus SaberConcat<X86, AK_FLOAT>::dispatch(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs, ConcatParam<X86>& param) {

    int input_size = inputs.size();
    //! get output data, valid shape and stride shape
    int offset_concat_axis = 0;
    Shape out_shape = outputs[0]->valid_shape();
    const int out_concat_axis = out_shape[param.axis];

    if (inputs[0]->get_layout() == Layout_NCHW_C8R) {
        for (int i = 1; i < input_size; i++) {
            CHECK_EQ(inputs[i]->get_layout(), Layout_NCHW_C8R) << "concat layout should euqal";
        }

        CHECK_EQ(outputs[0]->get_layout(), Layout_NCHW_C8R) << "concat output layout should euqal";

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

template <>
SaberStatus SaberConcat<X86, AK_INT8>::create(const std::vector<Tensor<X86>*>& inputs,
                                              std::vector<Tensor<X86>*>& outputs,
                                              ConcatParam<X86> &param,
                                              Context<X86> &ctx) {

    return SaberSuccess;
}

template <>
SaberStatus SaberConcat<X86, AK_INT8>::dispatch(const std::vector<Tensor<X86>*>& inputs,
                                                std::vector<Tensor<X86>*>& outputs,
                                                ConcatParam<X86> &param) {

    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberConcat<X86, OpDtype>::init_conf(jit::jit_concat_conf_t &jpp,
                                                 const std::vector<Tensor<X86>*> &inputs,
                                                 std::vector<Tensor<X86>*> &outputs,
                                                 ConcatParam<X86> &param){
    return SaberSuccess;
};

template <DataType OpDtype>
SaberStatus SaberConcat<X86, OpDtype>::check_conf(const jit::jit_concat_conf_t &jpp,
                                                  const std::vector<Tensor<X86>*> &inputs,
                                                  std::vector<Tensor<X86>*> &outputs,
                                                  ConcatParam<X86> &param){
    return SaberSuccess;
};
template <>
SaberStatus SaberConcat<X86, AK_INT8>::init_conf(jit::jit_concat_conf_t &jpp,
                                                 const std::vector<Tensor<X86>*> &inputs,
                                                 std::vector<Tensor<X86>*> &outputs,
                                                 ConcatParam<X86> &param) {
    return SaberSuccess;
}

template <>
SaberStatus SaberConcat<X86, AK_INT8>::check_conf(const jit::jit_concat_conf_t &jpp,
                                                  const std::vector<Tensor<X86>*> &inputs,
                                                  std::vector<Tensor<X86>*> &outputs,
                                                  ConcatParam<X86> &param) {
    return SaberSuccess;
}


template class SaberConcat<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConcat, ConcatParam, X86, AK_HALF);
} //namespace anakin

} //namespace anakin
