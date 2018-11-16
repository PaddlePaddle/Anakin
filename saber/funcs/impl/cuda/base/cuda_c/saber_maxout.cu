#include "saber/funcs/impl/cuda/saber_maxout.h"

namespace anakin {
namespace saber {

template <typename dtype>
__global__ void max_out(const dtype* input_ptr, dtype* output_ptr, const int count, 
                        const int num_out, const int c_out, const int h_out, const int w_out, const int groups) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_num = blockDim.x * gridDim.x;
    int feature_size = h_out * w_out;
    int feature_map_size = feature_size * c_out;
    for (int i = tid; i < count; i += thread_num) {
        int batch_index = i / feature_map_size;
        int channel_index = (i / feature_size) % c_out;
        int feature_inner_index = i % feature_size;
        int src_index = (batch_index * feature_map_size + channel_index * feature_size) * groups + feature_inner_index;
        dtype max = input_ptr[src_index]; //get first element.
        for (int j = 1; j < groups; j++) {
            dtype tmp = input_ptr[src_index + j * feature_size];
            max = max < tmp ? tmp: max;
        }
        output_ptr[i] = max;
    }
}

template <DataType OpDtype>
SaberStatus SaberMaxOut<NV, OpDtype>::dispatch(const std::vector<Tensor<NV>*>& inputs,
    std::vector<Tensor<NV>*>& outputs,
    MaxOutParam<NV>& param) {
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    const OpDataType* input_ptr = (const OpDataType*)inputs[0]->data();
    OpDataType* output_ptr = (OpDataType*)outputs[0]->mutable_data();
    int count = outputs[0]->valid_size();
    max_out<OpDataType><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
        input_ptr,
        output_ptr,
        count,
        _num_out,
        _c_out,
        _h_out,
        _w_out,
        param.groups    
    );

    CUDA_POST_KERNEL_CHECK;

    return SaberSuccess;
}

template class SaberMaxOut<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberMaxOut, MaxOutParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberMaxOut, MaxOutParam, NV, AK_INT8);

} // namespace saber.
} // namespace anakin.