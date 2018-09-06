#include "saber/funcs/impl/cuda/saber_scale.h"
#include "cuda_fp16.h"

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void ker_scale_fwd(Dtype * out_data,
                   const Dtype* in_data,
                   const Dtype* scale_data,
                   const Dtype* bias_data,
                   const int count,
                   const int scale_dim,
                   const int inner_dim) {
    CUDA_KERNEL_LOOP(tid, count){
        int scale_id = (tid / inner_dim) % scale_dim;
        Dtype scale = scale_data[scale_id];
        if (bias_data == nullptr) {
             out_data[tid] = scale * in_data[tid];
        } else {
             out_data[tid] = scale * in_data[tid] + bias_data[scale_id];
        }
    }
}

template<typename Dtype>
__global__ void ker_scale_fwd(Dtype * out_data,
                   const Dtype* in_data,
                   const Dtype scale,
                   const Dtype bias,
                   const int count) {
    CUDA_KERNEL_LOOP(tid, count){
        out_data[tid] = scale * in_data[tid] + bias;
    }
}


template <>
SaberStatus SaberScale<NV, AK_FLOAT>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ScaleParam<NV>& param) {

    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();

    auto in_data = inputs[0]->data();
    auto out_data = outputs[0]->mutable_data();
    const int count = inputs[0]->valid_size();
    if (inputs.size() > 1) {
        _scale_dim = inputs[1]->valid_size();
        _inner_dim = count / _scale_dim;
    }
    if (_scale_dim > 1 || inputs.size() > 1) {
        auto scale_data = inputs.size() > 1 ? inputs[1]->data() : _weight.data();
        auto bias_data = param.bias_term ? _bias.data() : NULL;
        ker_scale_fwd<OpDataType>
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                (OpDataType*)out_data, (const OpDataType*)in_data, (const OpDataType*)scale_data, \
                (const OpDataType*)bias_data, count, _scale_dim, _inner_dim);
    } else {
        auto scale = param.scale_w[0];
        OpDataType bias = 0;
        if (_bias_term) {
            bias = param.scale_b[0];
        }
        ker_scale_fwd<OpDataType>
                <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                (OpDataType*)out_data, (const OpDataType*)in_data, scale, bias, count);
    }

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

}
}
