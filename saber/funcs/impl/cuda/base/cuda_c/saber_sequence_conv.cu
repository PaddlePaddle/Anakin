#include "saber/funcs/impl/cuda/saber_sequence_conv.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/tensor_op.h"
#include "saber/funcs/funcs_utils.h"

namespace anakin {
namespace saber {


template <typename Dtype>
static __global__ void im2col_2d_ocf_kernel(const Dtype* in, int start, int stride, int pad_up, int pad_down,
        int kernel_size,
        Dtype* out, int seq_length, int hidden_size, int amount) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < amount) {
        int hidden_index = gid % hidden_size;
        int col = gid / hidden_size % kernel_size;
        int row = gid / hidden_size / kernel_size;

        int index = (row + col - pad_up+start);
        int out_index = (row * kernel_size + col) * hidden_size;

        if (index < 0 || index >= seq_length) {
            out[out_index + hidden_index] = 0;
        } else {
            //                    printf("%d -> %d [%f]\n",index+hidden_index,out_index+hidden_index,in[index+hidden_index]);
            out[out_index + hidden_index] = in[index * hidden_size + hidden_index];
        }
    }
}

template <>
SaberStatus SaberSequenceConv<NV, AK_FLOAT>::dispatch(
    const std::vector<OpTensor*>& inputs,
    std::vector<OpTensor*>& outputs,
    SequenceConvParam<NV>& param) {
    CHECK_GE(param.padding_trainable,false)<<"not support padding_trainable";
    OpTensor* in_data = inputs[0];
    OpTensor* out_data = outputs[0];
    std::vector<std::vector<int>> offset_vec_vec = in_data->get_seq_offset();
    std::vector<int> offset = offset_vec_vec[offset_vec_vec.size() - 1];
    out_data->set_seq_offset(offset_vec_vec);

    int word_num = offset[offset.size() - 1];
    utils::try_expand_tensor(_temp_im2col_tensor, word_num * param.filter_tensor->height());

    for (int i = 0; i < offset.size() - 1; ++i) {
        int start = offset[i];
        int seq_length = offset[i + 1] - offset[i];
        const int thread_num_in_block=256;
        int amount=seq_length*param.context_length*_hidden_size;

        im2col_2d_ocf_kernel<<<utils::div_up(amount,thread_num_in_block),thread_num_in_block,0,_ctx->get_compute_stream()>>>(
                static_cast<const OpDataType*>(in_data->data()) + _hidden_size * start,_word_start,
                      param.context_stride, _up_pad, _down_pad,
                      param.context_length, static_cast<OpDataType*>(_temp_im2col_tensor.mutable_data()) +
                      _hidden_kernel_size * start, seq_length,
                      _hidden_size,amount);
    }

    _gemm_im2col(word_num, _feature_size, _hidden_kernel_size, 1.f,
         static_cast<const OpDataType*>(_temp_im2col_tensor.data()),0.f,
         static_cast<const OpDataType*>(param.filter_tensor->data()),
         static_cast<OpDataType*>(out_data->mutable_data()),_ctx->get_compute_stream());

    return SaberSuccess;
}
template class SaberSequenceConv<NV, AK_FLOAT>;

}
}
