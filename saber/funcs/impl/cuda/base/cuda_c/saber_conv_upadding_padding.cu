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

    for (int mm_idx = thread_idx; mm_idx < channel_num * batch_size; mm_idx += thread_num) {
        int batch_idx   = mm_idx / channel_num;
        int channel_idx = mm_idx % channel_num;
        int width  = offset_w[batch_idx + 1] - offset_w[batch_idx];
        Dtype* p_dst = output + mm_idx * dst_height * dst_width;
        const Dtype* p_src = input + mm_idx * src_height * src_width;

        for (int i = 0; i < dst_height; ++i) {
            Dtype* p_dst_tmp = p_dst + i * dst_width;
            const Dtype* p_src_tmp = p_src + i * src_width;

            for (int j = 0; j < dst_width; ++j) {
                if (i < dst_height && j < width) {
                    *(p_dst_tmp + j) = *(p_src_tmp + j);
                } else {
                    *(p_dst_tmp + j) = 0;
                }
            }
        }
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

    int blocks=CUDA_GET_BLOCKS(batch_size * channel_num);
    int threads=CUDA_NUM_THREADS;
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

    const OpDataType* in_ptr= static_cast<const OpDataType*>(inputs[0]->data());
    OpDataType* out_ptr= static_cast<OpDataType*>(outputs[0]->mutable_data());
//    const int* gpu_height_offset_ptr=static_cast<const int*>(_height_offset_tensor.data());
    const int* gpu_width_offset_ptr=static_cast<const int*>(_width_offset_tensor.data());

    Shape in_shape=inputs[0]->valid_shape();
    int in_num=in_shape[0];
    int in_channel=in_shape[1];
    int in_height=in_shape[2];
    int in_width=in_shape[3];
    Shape out_shape=outputs[0]->valid_shape();
    int out_height=out_shape[2];
    int out_width=out_shape[3];

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
