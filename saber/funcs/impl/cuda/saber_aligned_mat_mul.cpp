#include "saber/funcs/impl/cuda/saber_aligned_mat_mul.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberAlignedMatMul<NV, OpDtype>::dispatch(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        AlignedMatMulParam<NV>  &param) {
    

    cudaStream_t stream = this->_ctx->get_compute_stream();
    const OpDataType* X = (const OpDataType*)inputs[0]->data();
    const OpDataType* Y = (const OpDataType*)inputs[1]->data();
    OpDataType* out = (OpDataType*)outputs[0]->mutable_data();
    auto seq_offset_x = inputs[0]->get_seq_offset()[0];
    auto seq_offset_y = inputs[1]->get_seq_offset()[0];
    CHECK_EQ(seq_offset_x.size(), seq_offset_y.size()) << "AlignedMatMul inputs have different seq num";
    int seq_num = seq_offset_x.size() - 1;
    int inner_A = inputs[0]->count_valid(1, inputs[0]->dims());
    int inner_B = inputs[1]->count_valid(1, inputs[1]->dims());
    int batch_A = seq_offset_x[1];
    int batch_B = seq_offset_y[1];
    int M = param.is_transpose_X ? inner_A : batch_A;
    int N = param.is_transpose_Y ? batch_B: inner_B;
    int K_A = param.is_transpose_X ? batch_A : inner_A;
    int K_B = param.is_transpose_Y ? inner_B : batch_B;
    CHECK_EQ(K_A, K_B) << "mat mul two inputs K is not equal";
    int K = K_A;
    _kernel = saber_find_fast_sass_gemm(param.is_transpose_X, param.is_transpose_Y, M, N, K);

    //should add batch gemm here
    for (int b = 0; b < seq_num; b++) {
        _kernel(M, N, K, param.scale,
            X + b * M * K,
            0.f, 
            Y + b * K * N,
            out + b * M * N, stream);
    }
    // print_tensor(*outputs[0]);
    return SaberSuccess;
}

template class SaberAlignedMatMul<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberAlignedMatMul, AlignedMatMulParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberAlignedMatMul, AlignedMatMulParam, NV, AK_INT8);

} // namespace saber;

} // namespace anakin;
