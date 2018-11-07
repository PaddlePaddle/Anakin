#include "saber/funcs/impl/cuda/saber_mat_mul.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberMatMul<NV, OpDtype>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    MatMulParam<NV>&  param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();
    const OpDataType* X = (const OpDataType*)inputs[0]->data();
    const OpDataType* Y = (const OpDataType*)inputs[1]->data();
    OpDataType* out = (OpDataType*)outputs[0]->mutable_data();

    //should add batch gemm here
    for (int b = 0; b < param._b; b++) {
        _kernel(param._m, param._n, param._k, 1.f,
                X + b * param._m * param._k,
                0.f,
                Y + b * param._k * param._n,
                out + b * param._m * param._n, stream);
    }

    // LOG(INFO) << "I'm in saber_mat_mul dipatch";
    // print_tensor(*outputs[0]);
    return SaberSuccess;
}

template class SaberMatMul<NV, AK_FLOAT>;

} // namespace saber;

} // namespace anakin;
