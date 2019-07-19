#include "saber/funcs/impl/x86/saber_aligned_mat_mul.h"
#include "mkl.h"
#if defined(__AVX2__) and defined(__FMA__)
#include "saber/funcs/impl/x86/saber_avx2_funcs.h"
#endif
#include <cmath>

namespace anakin{
namespace saber {

template <DataType OpDtype>
SaberStatus SaberAlignedMatMul<X86, OpDtype>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        AlignedMatMulParam<X86> &param,
        Context<X86> &ctx) {
    _alpha = param.scale;
    _beta = 0.f;
    _trans_a = param.is_transpose_X ? CblasTrans : CblasNoTrans;
    _trans_b = param.is_transpose_Y ? CblasTrans : CblasNoTrans;
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberAlignedMatMul<X86, OpDtype>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        AlignedMatMulParam<X86> &param,
        Context<X86> &ctx) {
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberAlignedMatMul<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        AlignedMatMulParam<X86> &param) {
    const OpDataType* src0 = (OpDataType*)inputs[0]->data();
    const OpDataType* src1 = (OpDataType*)inputs[1]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();
    auto seq_offset_0 = inputs[0]->get_seq_offset()[0];
    auto seq_offset_1 = inputs[1]->get_seq_offset()[0];
    int inner_A = inputs[0]->count_valid(1, inputs[0]->dims());
    int inner_B = inputs[1]->count_valid(1, inputs[1]->dims());
    int batch_A = seq_offset_0[1];
    int batch_B = seq_offset_1[1];
    int M = param.is_transpose_X ? inner_A : batch_A;
    int N = param.is_transpose_Y ? batch_B: inner_B;
    int K_A = param.is_transpose_X ? batch_A : inner_A;
    int K_B = param.is_transpose_Y ? inner_B : batch_B;
    CHECK_EQ(K_A, K_B) << "mat mul two inputs K is not equal";
    int K = K_A;
    int lda = param.is_transpose_X ? M : K;
    int ldb = param.is_transpose_Y ? K : N;
    int ldc = N;
    int seq_num = seq_offset_0.size() - 1;
    for (int i = 0; i < seq_num; i++) {
        cblas_sgemm(CblasRowMajor, _trans_a, _trans_b, M, N, K_A, _alpha, src0 + i * batch_A * inner_A, lda, src1 + i * batch_B * inner_B, ldb, _beta, dst + i * M * N, ldc);
    }

    return SaberSuccess;
}

template class SaberAlignedMatMul<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberAlignedMatMul, AlignedMatMulParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberAlignedMatMul, AlignedMatMulParam, X86, AK_INT8);
}
} // namespace anakin
