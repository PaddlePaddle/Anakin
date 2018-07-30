#include "saber/funcs/impl/cuda/saber_lstm_unit.h"
#include "saber/core/tensor_op.h"
#include "cuda_inline_activation.h"
namespace anakin {

namespace saber {

template <typename Dtype>
__global__ void lstm_unit_kernel(const int nthreads, const int dim,
                               const Dtype* C_prev, const Dtype* X, Dtype* C, Dtype* H,
                               const Dtype forget_bias) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; \
       index += blockDim.x * gridDim.x){
        const int n = index / dim;
        const int d = index % dim;

        const Dtype* X_offset = X + 4 * dim * n;
        const Dtype i = sigmoid(X_offset[d]);
        const Dtype f = sigmoid(X_offset[1 * dim + d] + forget_bias);
        const Dtype o = sigmoid(X_offset[2 * dim + d]);
        const Dtype g = tanh(X_offset[3 * dim + d]);
        const Dtype c_prev = C_prev[index];
        const Dtype c = f * c_prev + i * g;
        C[index] = c;
        const Dtype tanh_c = tanh(c);
        H[index] = o * tanh_c;
    }
}

template<>
SaberStatus
SaberLstmUnit<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>::dispatch(
    const std::vector < DataTensor_in* >& inputs,
    std::vector < DataTensor_out* >& outputs,
    LstmUnitParam < OpTensor >& param) {

    DataTensor_in* h_in=inputs[0];
    DataTensor_in* c_in=inputs[1];
    int seq_length=h_in->num();
    int hidden_size=h_in->channel();

    DataTensor_out* h_out=outputs[0];
    DataTensor_out* c_out=outputs[1];

    int block = 512;
    int n = seq_length * hidden_size;
    int grid = (n + block - 1) / block;

    lstm_unit_kernel<<<grid, block>>>(n, hidden_size, c_in->data(), h_in->data()
            , c_out->mutable_data(), h_out->mutable_data(), param.forget_bias);

    return SaberSuccess;
}

template class SaberLstmUnit<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW>;
}
}

