#include "saber/funcs/impl/cuda/saber_expand.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_expand_fwd(Dtype * dst, \
                    const Dtype* src,
                    const int* in_shape,
                    const int* expand_times,
                    const int dims,
                    const int count)
{
     int idx = threadIdx.x + blockIdx.x * blockDim.x;
     if (idx >= count) {
         return;
     }
     int in_idx = 0;
     int out_num = 1;
     int in_num = 1;
     for (int i = dims - 1; i >= 0; i--) {
         int cur_id = (idx / out_num) % in_shape[i];
         in_idx += cur_id * in_num; 
         out_num *= expand_times[i] * in_shape[i];
         in_num *= in_shape[i];
     }
     dst[idx] = src[in_idx];
}

template <DataType OpDtype>
SaberStatus SaberExpand<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    ExpandParam<NV>& param) {
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    auto in_shape = inputs[0]->valid_shape();
    auto out_shape = outputs[0]->valid_shape();
    auto expand_times = param.expand_times;
    int dims = expand_times.size();
    
    cudaMemcpyAsync(_expand_times.mutable_data(), &expand_times[0], sizeof(int) * dims, cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(_in_shape.mutable_data(), &in_shape[0], sizeof(int) * dims, cudaMemcpyHostToDevice, cuda_stream);

    const OpDataType* src = (const OpDataType*)inputs[0]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();

    int count = outputs[0]->valid_size();
    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        ker_expand_fwd<OpDataType>\
                 <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                 dst, src, (const int*)_in_shape.data(), (const int*)_expand_times.data(), dims, count);
    }

    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberExpand, ExpandParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberExpand, ExpandParam, NV, AK_INT8);
}
}
