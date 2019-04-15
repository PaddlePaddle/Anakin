
#include "saber/funcs/impl/x86/saber_cos_sim.h"
#include "mkl.h"
#if defined(__AVX2__) and defined(__FMA__)
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
#endif
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberCosSim<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        CosSimParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberCosSim<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        CosSimParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberCosSim<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        CosSimParam<X86> &param) {
    CHECK_EQ(inputs.size(), 2) << "CosSim input num need be  2, but is" << inputs.size();
    CHECK_EQ(outputs.size(), 1) << "CosSim input num need be  1, but is" << outputs.size();
    size_t count_0 = inputs[0]->valid_size();
    size_t count_1 = inputs[1]->valid_size();
    CHECK_EQ(count_0, count_1) << "input0 and input1 valid size is not equal";

    size_t num = inputs[0]->num();
    size_t inner_size = count_0 / inputs[0]->num();
    const OpDataType *input0_data = (const OpDataType*)inputs[0]->data();
    const OpDataType *input1_data = (const OpDataType*)inputs[1]->data();
    OpDataType *output_data = (OpDataType*)outputs[0]->mutable_data();
#if defined(__AVX2__) and defined(__FMA__)
   avx2_cos_sim(input0_data, input1_data, num, inner_size, param.epsilon, output_data);
#else
    for (size_t n = 0; n < num; n++) {
        auto input0_square_sum = (OpDataType)0;
        auto input1_square_sum = (OpDataType)0;
        auto input01_prod_sum = (OpDataType)0;
#pragma omp parallel for schedule(static) reduction(+:input0_square_sum, input1_square_sum, input01_prod_sum)
        for (size_t i = 0; i < inner_size; i++) {
            input0_square_sum += input0_data[i] * input0_data[i];
            input1_square_sum += input1_data[i] * input1_data[i];
            input01_prod_sum += input0_data[i] * input1_data[i];
        }
        float bc = input0_square_sum * input1_square_sum;
        if (bc < param.epsilon) {
            output_data[n] = 0;
        } else {
            output_data[n] = input01_prod_sum / sqrt(bc);
        }
        input0_data += inner_size; 
        input1_data += inner_size; 
    }
#endif

    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i]->set_seq_offset(inputs[i]->get_seq_offset());
    }
    return SaberSuccess;
}

template class SaberCosSim<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberCosSim, CosSimParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberCosSim, CosSimParam, X86, AK_INT8);
}
} // namespace anakin
