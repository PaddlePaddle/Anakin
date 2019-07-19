#include "saber/funcs/impl/cuda/saber_affine_channel.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_affine_channel_fwd(Dtype * out_data, \
                    const Dtype* in_data,
                    const Dtype* scale_data,
                    const Dtype* bias_data,
                    const int outer_num,
                    const int channel,
                    const int inner_num,
                    const int count)
{
    CUDA_KERNEL_LOOP(tid, count){
        const int channel_id =  (tid / inner_num) % channel;
        out_data[tid] = in_data[tid] * scale_data[channel_id] + bias_data[channel_id];
    }
}

template <DataType OpDtype>
SaberStatus SaberAffineChannel<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    AffineChannelParam<NV>& param) {

    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    const OpDataType* scale_data = (const OpDataType*)param.weight()->data();
    const OpDataType* bias_data = (const OpDataType*)param.bias()->data();
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = outputs[0]->valid_size();
    int channel_idx = inputs[0]->channel_index();
    int outer_num = inputs[0]->count_valid(0, channel_idx);
    int channel = inputs[0]->channel();
    int inner_num = inputs[0]->count_valid(channel_idx+1, inputs[0]->dims());
    CHECK_EQ(param.weight()->valid_size(), channel) << "affine channel input scale dims are not valid";
    CHECK_EQ(param.bias()->valid_size(), channel) << "affine channel input bias dims are not valid";

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        ker_affine_channel_fwd<OpDataType>\
                 <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                 out_data, in_data, scale_data, bias_data, outer_num, channel, inner_num,
                 count);
    }

    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberAffineChannel, AffineChannelParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberAffineChannel, AffineChannelParam, NV, AK_INT8);
}
}
