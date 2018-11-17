#include "saber/funcs/impl/cuda/saber_shuffle_channel.h"

namespace anakin{

namespace saber{

template <typename Dtype>
__global__ void ShuffleChannelKernel(const int nthreads, const int feature_map_size, \
        Dtype *output, const Dtype *input, int group_row, int group_column, int len) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int n = index / group_row / group_column / len;
        const int i = (index / group_column / len) % group_row;
        const int j = index / len % group_column;
        const int k = index - (n * feature_map_size + (i * group_column + j) * len);
        Dtype* p_o = output + n * feature_map_size + (j * group_row + i) * len;
        p_o[k] = input[index];
    }
}


template <DataType OpDtype>
SaberStatus SaberShuffleChannel<NV, OpDtype>::dispatch(const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    ShuffleChannelParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();

    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    int feature_map_size = channel * height * width;
    int sp_sz = height * width;

    int group_row = param.group;
    int group_column = channel / group_row;
    int count = num * group_column * group_row * sp_sz;

    const OpDataType* bottom_data = static_cast<const OpDataType*>(inputs[0]->data());
    OpDataType* top_data = static_cast<OpDataType*>(outputs[0]->mutable_data());

    ShuffleChannelKernel<OpDataType> <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
            count, feature_map_size, top_data, bottom_data, group_row, group_column, sp_sz);

    return SaberSuccess;
}
} //namespace anakin

} //namespace anakin
