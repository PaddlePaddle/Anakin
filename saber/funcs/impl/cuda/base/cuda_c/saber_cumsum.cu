#include "saber/funcs/impl/cuda/saber_cumsum.h"
#include "cuda_fp16.h"

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void ker_cumsum_fwd(Dtype * out_data,
                             const Dtype* in_data, const int count,
                             int pre, int post, int cum_num
                             ) {
    CUDA_KERNEL_LOOP(tid, count) {
        int post_id = tid % post;
        int pre_id = tid / post;
        int idx = pre_id * cum_num * post + post_id;
        Dtype* out = out_data + idx;
        out[0] = in_data[idx];
        for (int i = 1; i < cum_num; i++) {
            out[i * post] = out[(i-1) * post] + in_data[idx + i * post];
        }
    }
}

template<typename Dtype>
__global__ void ker_cumsum_exclusive_fwd(Dtype * out_data,
                             const Dtype* in_data, const int count,
                             int pre, int post, int cum_num
                             ) {
    CUDA_KERNEL_LOOP(tid, count) {
        int post_id = tid % post;
        int pre_id = tid / post;
        int idx = pre_id * cum_num * post + post_id;
        Dtype* out = out_data + idx;
        out[0] = 0;
        for (int i = 1; i < cum_num; i++) {
            out[i * post] = out[(i-1) * post] + in_data[idx + (i-1) * post];
        }
    }
}


template<typename Dtype>
__global__ void ker_reverse_cumsum_fwd(Dtype * out_data,
                             const Dtype* in_data, const int count,
                             int pre, int post, int cum_num
                             ) {
    CUDA_KERNEL_LOOP(tid, count) {
        int post_id = tid % post;
        int pre_id = tid / post;
        int idx = pre_id * cum_num * post + post_id;
        Dtype* out = out_data + idx;
        out[0] = in_data[idx + (cum_num - 1) * post];
        for (int i = 1; i < cum_num; i++) {
            out[i * post] = out[(i-1) * post] + in_data[idx + (cum_num - i - 1) * post];
        }
    }
}

template<typename Dtype>
__global__ void ker_reverse_cumsum_exclusive_fwd(Dtype * out_data,
                             const Dtype* in_data, const int count,
                             int pre, int post, int cum_num
                             ) {
    CUDA_KERNEL_LOOP(tid, count) {
        int post_id = tid % post;
        int pre_id = tid / post;
        int idx = pre_id * cum_num * post + post_id;
        Dtype* out = out_data + idx;
        out[0] = 0;
        for (int i = 1; i < cum_num; i++) {
            out[i * post] = out[(i-1) * post] + in_data[idx + (cum_num - i) * post];
        }
    }
}


template <DataType OpDtype>
SaberStatus SaberCumsum<NV, OpDtype>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        CumsumParam<NV>& param) {

    Shape in_shape = inputs[0]->valid_shape();
    Shape out_shape = outputs[0]->valid_shape();

    const OpDataType *in_data = (const OpDataType*)inputs[0]->data();
    OpDataType *out_data = (OpDataType*)outputs[0]->mutable_data();

    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    bool reverse = param.reverse;
    bool exclusive = param.exclusive;
    int axis = param.axis < 0 ? param.axis + inputs[0]->dims() : param.axis ;
    int pre = inputs[0]->count_valid(0, axis);
    int post = inputs[0]->count_valid(axis + 1, inputs[0]->dims());
    int cum_num = inputs[0]->valid_shape()[axis];
    int count = pre * post;
    if (reverse) {
        if (exclusive) {
            ker_reverse_cumsum_exclusive_fwd<OpDataType>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data, count, pre, post, cum_num);
        } else {
            ker_reverse_cumsum_fwd<OpDataType>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data, count, pre, post, cum_num);
        }
    } else {
        if (exclusive) {
            ker_cumsum_exclusive_fwd<OpDataType>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data, count, pre, post, cum_num);

        } else {
            ker_cumsum_fwd<OpDataType>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data, count, pre, post, cum_num);
        }
    }
    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberCumsum<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberCumsum, CumsumParam, NV, AK_INT8);
DEFINE_OP_TEMPLATE(SaberCumsum, CumsumParam, NV, AK_HALF);
}
}
