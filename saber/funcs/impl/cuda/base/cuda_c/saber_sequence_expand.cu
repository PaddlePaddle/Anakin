#include "saber/funcs/impl/cuda/saber_sequence_expand.h"
#include "cuda_fp16.h"

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void ker_relu_fwd(Dtype * out_data,
                   const Dtype* in_data, const int count, Dtype neg_slop,
                   int in_n, int in_c, int in_h, int in_w,
                   int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                   int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride) {
    CUDA_KERNEL_LOOP(tid, count){
        int w =  tid % in_w;
        int h = (tid / (in_w)) % in_h;
        int c = (tid / (in_h * in_w)) % in_c;
        int n = (tid / (in_c * in_h * in_w)) % in_n;

        int in_idx = n * in_n_stride
                   + c * in_c_stride
                   + h * in_h_stride
                   + w * in_w_stride;

        int out_idx =  n * out_n_stride
                     + c * out_c_stride
                     + h * out_h_stride
                     + w * out_w_stride;

        Dtype in_var = in_data[in_idx];
        out_data[out_idx] = in_var > Dtype(0) ? in_var : in_var * neg_slop;
    }
}


template <>
SaberStatus SaberSequenceExpand<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, \
        NCHW, NCHW, NCHW>::dispatch( \
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SequenceExpandParam<OpTensor>& param) {

    Shape in_shape = inputs[0]->valid_shape();
    Shape out_shape = outputs[0]->valid_shape();

    const InDataType *in_data = (const InDataType*)inputs[0]->data();
    OutDataType *out_data = (OutDataType*)outputs[0]->mutable_data();

    const int count = inputs[0]->valid_size();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();

    InDataType negative_slope = param.negative_slope;
    InDataType coef = param.coef;

    ker_relu_fwd<InDataType>
            <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
            out_data, in_data, count, negative_slope,
            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
            stride_in[0], stride_in[1], stride_in[2], stride_in[3],
            stride_out[0], stride_out[1], stride_out[2], stride_out[3]);

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

}
}
