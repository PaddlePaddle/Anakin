
#include "saber/funcs/impl/x86/saber_arithmetic.h"
#include "mkl.h"
#if defined(__AVX2__) and defined(__FMA__)
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
#endif
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberArithmetic<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ArithmeticParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberArithmetic<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ArithmeticParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberArithmetic<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ArithmeticParam<X86> &param) {
    const OpDataType *input_data_0 = (const OpDataType*)inputs[0]->data();
    const OpDataType *input_data_1 = (const OpDataType*)inputs[1]->data();
    OpDataType *output_data = (OpDataType*)outputs[0]->mutable_data();
    auto seq_offset_0 = inputs[0]->get_seq_offset()[0];
    auto seq_offset_1 = inputs[1]->get_seq_offset()[0];
    int seq_num = inputs[0]->get_seq_offset()[0].size() - 1;
    int inner_size = inputs[0]->count_valid(1, inputs[0]->dims());
    

    // out[j] = input_0[j] + input_1[j] if j < count_0 && j < count_1;
    // out[j] = input_0[j] if j < count_0 && j >= count_1;
    if (param.op_type == SUM) {
        size_t len = inputs[0]->valid_size();
        for (int i = 0; i < seq_num; i++) {
            int len_0 = (seq_offset_0[i+1] - seq_offset_0[i]) * inner_size;
            int len_1 = (seq_offset_1[i+1] - seq_offset_1[i]) * inner_size; 
            auto input_0 = input_data_0 + seq_offset_0[i] * inner_size;
            auto input_1 = input_data_1 + seq_offset_1[i] * inner_size;
            auto out = output_data + seq_offset_0[i] * inner_size;
            int len = std::min(len_0, len_1);
#if defined(__AVX2__) and defined(__FMA__)
            avx2_vector_sum(input_0, input_1, len, out);
#else
#pragma omp parallel for schedule(static)
            for (int j = 0; j < len; j++) {
                out[j] = input_0[j] + input_1[j];
            }
#endif
            if (len_0 > len) {
                memcpy(out + len, input_0 + len, sizeof(OpDataType) * (len_0 -len));
            }
            
        }
    }

    // out[j] = input_0[j] - input_1[j] if j < count_0 && j < count_1;
    // out[j] = input_0[j] if j < count_0 && j >= count_1;
    if (param.op_type == SUB) {
        size_t len = inputs[0]->valid_size();
        for (int i = 0; i < seq_num; i++) {
            int len_0 = (seq_offset_0[i+1] - seq_offset_0[i]) * inner_size;
            int len_1 = (seq_offset_1[i+1] - seq_offset_1[i]) * inner_size;
            auto input_0 = input_data_0 + seq_offset_0[i] * inner_size;
            auto input_1 = input_data_1 + seq_offset_1[i] * inner_size;
            auto out = output_data + seq_offset_0[i] * inner_size;
            int len = std::min(len_0, len_1);
#if defined(__AVX2__) and defined(__FMA__)
            avx2_vector_sub(input_0, input_1, len, out);
#else
#pragma omp parallel for schedule(static)
            for (int j = 0; j < len; j++) {
                out[j] = input_0[j] - input_1[j];
            }
#endif
            if (len_0 > len) {
                memcpy(out + len, input_0 + len, sizeof(OpDataType) * (len_0 -len));
            }
        }
    }
    // out[j] = input_0[j] * input_1[j] if j < count_0 && j < count_1;
    // out[j] = input_0[j] if j < count_0 && j >= count_1;
    if (param.op_type == MUL) {
        size_t len = inputs[0]->valid_size();
        for (int i = 0; i < seq_num; i++) {
            int len_0 = (seq_offset_0[i+1] - seq_offset_0[i]) * inner_size;
            int len_1 = (seq_offset_1[i+1] - seq_offset_1[i]) * inner_size;
            auto input_0 = input_data_0 + seq_offset_0[i] * inner_size;
            auto input_1 = input_data_1 + seq_offset_1[i] * inner_size;
            auto out = output_data + seq_offset_0[i] * inner_size;
            int len = std::min(len_0, len_1);
#if defined(__AVX2__) and defined(__FMA__)
            avx2_vector_mul(input_0, input_1, len, out);
#else
#pragma omp parallel for schedule(static)
            for (int j = 0; j < len; j++) {
                out[j] = input_0[j] * input_1[j];
            }
#endif
            if (len_0 > len) {
                memcpy(out + len, input_0 + len, sizeof(OpDataType) * (len_0 -len));
            }
        }
    }

    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;
}

template class SaberArithmetic<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberArithmetic, ArithmeticParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberArithmetic, ArithmeticParam, X86, AK_INT8);
}
} // namespace anakin
