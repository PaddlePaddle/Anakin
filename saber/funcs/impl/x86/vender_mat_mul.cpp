#include "saber/funcs/impl/x86/vender_mat_mul.h"


namespace anakin{

namespace saber{

template <DataType OpDtype>
SaberStatus SaberMatMul<X86, OpDtype>::dispatch(
        const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        MatMulParam<X86>  &param) {
    
    CHECK_EQ(OpDtype, AK_FLOAT) << "vender mat mul only support float now!";
    const OpDataType* src0 = (OpDataType*)inputs[0]->data();
    const OpDataType* src1 = (OpDataType*)inputs[1]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();
 
    for (int i = 0; i < batch; i++) {
        
        cblas_sgemm(layout, transa, transb, M, N, K, alpha, src0 + i * M * K, lda, src1 + i * K * N, ldb, beta, dst + i * M * N, ldc);
    }
}

template class SaberMatMul<X86, AK_FLOAT>;

} // namespace saber;

} // namespace anakin;