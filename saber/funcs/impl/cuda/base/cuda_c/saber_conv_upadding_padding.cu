#include "saber/funcs/impl/cuda/saber_conv_upadding_padding.h"
#include "saber/funcs/saber_util.h"

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void var_conv_unpadding_padding(Dtype* output,
        const Dtype* input,
        const int* offset_w,
        const int batch_size,
        const int channel_num,
        const int src_height,
        const int src_width,
        const int dst_height,
        const int dst_width) {
    // each thread process one channel of a matching matrix
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_num = gridDim.x * blockDim.x;

    if (thread_idx >= batch_size * channel_num * dst_height * dst_width) {
        return;
    }
    int batch_idx = thread_idx / (channel_num * dst_height * dst_width);
    int channel_idx = (thread_idx / (dst_height * dst_width)) % channel_num;
    int height_idx = (thread_idx / dst_width) % dst_height;
    int width_idx = thread_idx % dst_width;
    int width = offset_w[batch_idx + 1] - offset_w[batch_idx];
    if (width_idx < width) {
        output[thread_idx] = input[((batch_idx * channel_num  + channel_idx)* src_height  + height_idx) * src_width + width_idx];
    } else {
        output[thread_idx] = 0;
    }
}

template <typename Dtype>
void anakin_gpu_var_conv_unpadding_padding(Dtype* output_data,
        const Dtype* input,
        const int* offset_w,
        const int batch_size,
        const int channel_num,
        const int src_height,
        const int src_width,
        const int dst_height,
        const int dst_width,
        cudaStream_t stream) {

    int blocks = CUDA_GET_BLOCKS(batch_size * channel_num * dst_height * dst_width);
    int threads = CUDA_NUM_THREADS;
    var_conv_unpadding_padding<Dtype> <<< blocks, threads, 0, stream>>>(output_data,
            input,
            offset_w,
            batch_size,
            channel_num,
            src_height,
            src_width,
            dst_height,
            dst_width);
}


template <>
SaberStatus SaberConvUnpaddingPadding<NV, AK_FLOAT>::dispatch(
    const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    ConvUnpaddingPaddingParam<NV>& param) {
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

    const OpDataType* in_ptr = static_cast<const OpDataType*>(inputs[0]->data());
    OpDataType* out_ptr = static_cast<OpDataType*>(outputs[0]->mutable_data());
    //    const int* gpu_height_offset_ptr=static_cast<const int*>(_height_offset_tensor.data());
    std::vector<int> width_vector = inputs[0]->get_seq_offset()[0];
    utils::try_expand_tensor(_width_offset_tensor, width_vector.size());
    CUDA_CHECK(cudaMemcpyAsync(_width_offset_tensor.mutable_data(), width_vector.data(),
                              sizeof(int)*width_vector.size(), cudaMemcpyHostToDevice, this->_ctx->get_compute_stream()));
	const int* gpu_width_offset_ptr = static_cast<const int*>(_width_offset_tensor.data());

    Shape in_shape = inputs[0]->valid_shape();
    int in_num = in_shape[0];
    int in_channel = in_shape[1];
    int in_height = in_shape[2];
    int in_width = in_shape[3];
    Shape out_shape = outputs[0]->valid_shape();
    int out_height = out_shape[2];
    int out_width = out_shape[3];

    anakin_gpu_var_conv_unpadding_padding(out_ptr,
                                          in_ptr,
                                          gpu_width_offset_ptr,
                                          in_num,
                                          in_channel,
                                          in_height,
                                          in_width,
                                          out_height,
                                          out_width,
                                          this->_ctx->get_compute_stream());
    return SaberSuccess;
}

template class SaberConvUnpaddingPadding<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberConvUnpaddingPadding, ConvUnpaddingPaddingParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConvUnpaddingPadding, ConvUnpaddingPaddingParam, NV, AK_INT8);
}
}
