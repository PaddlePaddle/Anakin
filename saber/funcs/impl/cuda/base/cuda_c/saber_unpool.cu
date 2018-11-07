#include "saber/funcs/impl/cuda/saber_unpool.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_unpool_max_fwd(Dtype * out_data, \
                    const Dtype* in_data,
                    const Dtype* in_max_index,
                    const int in_n_stride,
                    const int in_c_stride,
                    const int out_n_stride,
                    const int out_c_stride,
                    const int in_n, 
                    const int in_c,
                    const int num_threads)
{
    CUDA_KERNEL_LOOP(tid, num_threads){
        int n = tid / in_n_stride;
        int c = (tid / in_c_stride) % in_c;
        int out_offset = n * out_n_stride + c * out_c_stride; 
        int index = in_max_index[tid];
        out_data[out_offset + index] = in_data[tid];
    }
}

template <DataType OpDtype>
SaberStatus SaberUnpool<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs,\
    PoolingParam<NV>& param) {

    const InDataType* in_data = (const InDataType*)inputs[0]->data();
    const OutDataType* in_max_index = (const OutDataType*)inputs[1]->data();
    OutDataType* out_data = (OutDataType*)outputs[0]->mutable_data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = inputs[0]->valid_size();
    int in_n = inputs[0]->num();
    int in_c = inputs[0]->channel();
    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        cudaMemsetAsync(out_data, 0, sizeof(InDataType) * outputs[0]->valid_size(), cuda_stream);
        ker_unpool_max_fwd<InDataType>\
                 <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                 out_data,
                 in_data, in_max_index,\
                 _in_n_stride, _in_c_stride,\
                 _out_n_stride, _out_c_stride,\
                 in_n, in_c, count);
    }

    return SaberSuccess;
}

} //namespace saber

} //namespace anakin
