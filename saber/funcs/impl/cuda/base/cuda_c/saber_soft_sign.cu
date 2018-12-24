#include "saber/funcs/impl/cuda/saber_soft_sign.h"
#include "cuda_fp16.h"

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void ker_soft_sign_fwd(Dtype * out_data,
                             const Dtype* in_data,
                             const int count) {
    CUDA_KERNEL_LOOP(tid, count) {
        Dtype in_var = in_data[tid];
        Dtype in_abs = in_var > 0 ? in_var : -in_var;
        out_data[tid] = in_var / (in_abs  + (Dtype)1.f);
    }
}

template <DataType OpDtype>
SaberStatus SaberSoftSign<NV, OpDtype>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        SoftSignParam<NV>& param) {

    const OpDataType *in_data = (const OpDataType*)inputs[0]->data();
    OpDataType *out_data = (OpDataType*)outputs[0]->mutable_data();

    const int count = inputs[0]->valid_size();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();

    //y = x / (x + 1)
    ker_soft_sign_fwd<OpDataType>
            <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
            out_data, in_data, count);

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberSoftSign<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSoftSign, SoftSignParam, NV, AK_INT8);
DEFINE_OP_TEMPLATE(SaberSoftSign, SoftSignParam, NV, AK_HALF);
}
}
