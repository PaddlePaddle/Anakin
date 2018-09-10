#include "saber/funcs/impl/x86/saber_axpy.h"

namespace anakin{

namespace saber{


template <typename dtype>
void axpy_kernel(const int len, const dtype* src, dtype* dst) {
    if (dst != src) {
        memcpy(dst, src, sizeof(dtype) * len);
    }
}
template <DataType OpDtype>
SaberStatus SaberAxpy<X86, OpDtype>::dispatch(const std::vector<Tensor<X86>*>& inputs,
            std::vector<Tensor<X86>*>& outputs, AxpyParam<X86> &param) {
    //compare
    if (!(inputs[1]->valid_shape() == outputs[0]->valid_shape()) 
        || !(inputs[2]->valid_shape() == outputs[0]->valid_shape())) {
         return SaberUnKownError;
    }

    const OpDataType* scale = (OpDataType*)inputs[0]->data();
    const OpDataType* x = (OpDataType*)inputs[1]->data();
    const OpDataType* y = (OpDataType*)inputs[2]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();

    int num = inputs[2]->num();
    int channel = inputs[2]->channel();
    int size = inputs[2]->height() * inputs[2]->width();
    int in_channel = channel * size;
    // scale*x + y
    for (int i = 0; i < num; i++){
        const OpDataType* din_ptr = x + i * in_channel;
        const OpDataType* bias_ptr = y + i * in_channel;
        const OpDataType* scale_ptr = scale + i * channel;
        OpDataType* dout_ptr = dst + i * in_channel;
        for(int j = 0; j < channel; j++){
            const OpDataType* din_ch_ptr = din_ptr + j * size;
            OpDataType* dout_ch_ptr = dout_ptr + j * size;
            const OpDataType* scale_ch_ptr = scale_ptr + j;
            const OpDataType* bias_ch_ptr = bias_ptr + j * size;
            for (int k = 0; k < size; k++){
                dout_ch_ptr[k] = din_ch_ptr[k] * scale_ch_ptr[0] + bias_ch_ptr[k];
            }
        }
    }
    return SaberSuccess;
}

template class SaberAxpy<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberAxpy, AxpyParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberAxpy, AxpyParam, X86, AK_INT8);
} //namespace anakin

} //namespace anakin
