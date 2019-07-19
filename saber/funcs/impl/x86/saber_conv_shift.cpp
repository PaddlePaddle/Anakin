#include "saber/funcs/impl/x86/saber_conv_shift.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include <cmath>
namespace anakin {
namespace saber {
/*
 *conv shift compute formula
 *input: inputs[0] {N. M};
 *input: inputs[1] {N, K}
 */

template <DataType OpDtype>
SaberStatus SaberConvShift<X86, OpDtype>::dispatch(\
    const std::vector<Tensor<X86> *>& inputs, \
    std::vector<Tensor<X86> *>& outputs, \
    ConvShiftParam<X86>& param) {
    
    const OpDataType* src = (const OpDataType*)inputs[0]->data();
    const OpDataType* ker = (const OpDataType*)inputs[1]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();

    int num_0 = inputs[0]->num();
    int num_1 = inputs[1]->num();
    int num_out = outputs[0]->num();
    int out_width = outputs[0]->count_valid(1, outputs[0]->dims());
    int input_width = inputs[0]->count_valid(1, inputs[0]->dims());
    int kernel_width = inputs[1]->count_valid(1, inputs[1]->dims());
    int half_kernel_width = (kernel_width - 1) / 2;
    CHECK_EQ(out_width, input_width);
    CHECK_EQ(num_0, num_1) << "conv shift two inputs num are not equal";
    CHECK_EQ(num_0, num_out) << "conv shift input batchsize and output batchsize are not equal";

    for (int i = 0; i < num_0; i++) {
        auto src_tmp = src + i * input_width;
        auto ker_tmp = ker + i * kernel_width; 
        for (int j = 0; j < input_width; j++) {
            OpDataType res = 0;
            for (int k = 0; k < kernel_width; k++) {
                int index = (j + k - half_kernel_width + input_width) % input_width;
                res += src_tmp[index] * ker_tmp[k];
            }
            dst[i * out_width + j] = res;
        }
    }

    return SaberSuccess;
}

template class SaberConvShift<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConvShift, ConvShiftParam, X86, AK_INT16);
DEFINE_OP_TEMPLATE(SaberConvShift, ConvShiftParam, X86, AK_INT8);
}
}
