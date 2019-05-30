#include "saber/funcs/impl/cuda/saber_aligned_mat_mul.h"
#include "saber/funcs/impl/cuda/vender_batch_gemm.h"

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
    int M = param.M;
    int N = param.N;
    int K = param.K;
    int A_stride = M * K;
    int B_stride = K * N;
    int C_stride = M * N;
    if (0) {
    _kernel = saber_find_fast_sass_gemm(param.is_transpose_X, param.is_transpose_Y, M, N, K);

    //should add batch gemm here
    for (int b = 0; b < seq_num; b++) {
        _kernel(M, N, K, param.scale,
            X + b * A_stride,
            0.f, 
            Y + b * B_stride,
            out + b * C_stride, stream);
    }
    } else if (1){
        OpDataType* A[seq_num];
        OpDataType* B[seq_num];
        OpDataType* C[seq_num];
        for (int b = 0; b < seq_num; b++) {
            A[b] = X + b * A_stride;
            B[b] = Y + b * B_stride;
            C[b] = out + b * C_stride;
        }
        _vender_batch_gemm.dispatch(param.scale, 0.f, A, B, M, N, K, C, seq_num);
    } else {
        OpDataType* A[3 * seq_num];
        for (int b = 0; b < seq_num; b++) {
            A[b] = X + b * A_stride;
            A[b + seq_num] = Y + b * B_stride;
            A[b + seq_num *2] = out + b * C_stride;
        }
        //_vender_batch_gemm.dispatch(1.0f, 0.f, A, M, N, K, seq_num);
    }
    // print_tensor(*outputs[0]);
    return SaberSuccess;
}

template class SaberAlignedMatMul<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberAlignedMatMul, AlignedMatMulParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberAlignedMatMul, AlignedMatMulParam, NV, AK_INT8);

} // namespace saber;

} // namespace anakin;
