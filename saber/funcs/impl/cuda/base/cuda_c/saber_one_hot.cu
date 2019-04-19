
#include "saber/funcs/impl/cuda/saber_one_hot.h"

namespace anakin {

namespace saber {

template <>
SaberStatus SaberOneHot<NV, AK_FLOAT>::create(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        OneHotParam<NV>& param, Context<NV>& ctx) {
    return SaberSuccess;
}

template <>
SaberStatus SaberOneHot<NV, AK_FLOAT>::init(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        OneHotParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}
__global__ void fill_one_hot_kernel(const float* in_ptr,
        float* out_ptr, const int dim, const int depth) {

    CUDA_KERNEL_LOOP(tid, dim) {
        out_ptr[tid * depth + (int)in_ptr[tid]] = 1.0;
    }
}
template <>
SaberStatus SaberOneHot<NV, AK_FLOAT>::dispatch(
        const std::vector<Tensor<NV> *>& inputs,
        std::vector<Tensor<NV> *>& outputs,
        OneHotParam<NV>& param) {

    auto stream = _ctx->get_compute_stream();
    const float* input_ptr = (const float*)inputs[0]->data();
    float* output_ptr = (float*)outputs[0]->mutable_data();
    int _depth = param.depth;
    int dims = inputs[0]->valid_size();
    cudaMemsetAsync(output_ptr,
            0,
            outputs[0]->valid_size() * outputs[0]->get_dtype_size(),
            stream);
    fill_one_hot_kernel<<<CUDA_GET_BLOCKS(dims), CUDA_NUM_THREADS, 0, stream>>>(
            input_ptr, output_ptr, dims, _depth);
    return SaberSuccess;
}

template class SaberOneHot<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberOneHot, OneHotParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberOneHot, OneHotParam, NV, AK_INT8);

}
}