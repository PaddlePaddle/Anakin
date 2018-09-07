#include "saber/funcs/impl/cuda/saber_lrn.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_cross_map_region_norm_fwd(Dtype * out_data, \
                    const Dtype* in_data,
                    const int in_n_stride,
                    const int in_c_stride,
                    const int in_h_stride, 
                    const int in_w_stride,
                    const int in_n,
                    const int in_c,
                    const int in_h,
                    const int in_w,
                    Dtype alpha,
                    Dtype beta,
                    Dtype k,
                    const int size,
                    const int num_threads)
{
    CUDA_KERNEL_LOOP(tid, num_threads){
        const int n = tid / (in_h * in_w);
        const int h = (tid / in_w) % in_h;
        const int w = tid % in_w;
        const int offset = n * in_n_stride + h * in_h_stride + w * in_w_stride;
        const Dtype* in = in_data + offset;
        Dtype* out = out_data + offset;
        const int pre_pad = (size - 1) / 2;
        const int post_pad = size - pre_pad - 1;

        Dtype accum = 0;
        int index = 0;
        while (index < in_c + post_pad) {
            if (index < in_c) {
                 Dtype val = in[index * in_c_stride];
                 accum += val * val;
            }
            if (index >= size) {
                 Dtype val = in[(index - size) * in_c_stride];
                 accum -= val * val;
            }
            if (index >= post_pad) {
                 Dtype mid = k + accum * alpha;
                 int off = (index - post_pad) * in_c_stride;
                 out[off] = in[off] * pow(mid, -beta);
            }
            ++index;
        }
    }
}

template <DataType OpDtype>
SaberStatus SaberLrn<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    LrnParam<NV>& param) {

    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int out_n = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    int count = outputs[0]->valid_size() / out_c;

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        ker_cross_map_region_norm_fwd<OpDataType>\
                 <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                 out_data, in_data, \
                 _in_n_stride, _in_c_stride, _in_h_stride, _in_w_stride,\
                 out_n, out_c, out_h, out_w,
                 param.alpha, param.beta, param.k, param.local_size, count);
    }

    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberLrn, LrnParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberLrn, LrnParam, NV, AK_INT8);
}
}
