#include "saber/funcs/impl/x86/saber_soft_sign.h"
#include "mkl.h"
#if defined(__AVX2__) and defined(__FMA__)
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
#endif
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberSoftSign<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SoftSignParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberSoftSign<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SoftSignParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberSoftSign<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        SoftSignParam<X86> &param) {
    // y=  x / (1.0 + fabs(x))
    for (size_t vc = 0; vc < inputs.size(); vc++) {
        size_t len = inputs[vc]->valid_size();
        OpDataType *input_data = (OpDataType*)inputs[vc]->mutable_data();
        OpDataType *output_data = (OpDataType*)outputs[vc]->mutable_data();
//#if defined(__AVX2__) and defined(__FMA__)
//        avx2_vector_soft_sign(input_data, len, output_data);
//#else
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < len; i++) {
            OpDataType tmp = input_data[i] > 0 ? input_data[i] : -input_data[i];
            output_data[i] = input_data[i] / (1 + tmp);
        }
//#endif
    }

    return SaberSuccess;
}

template class SaberSoftSign<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSoftSign, SoftSignParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSoftSign, SoftSignParam, X86, AK_INT8);
}
} // namespace anakin
