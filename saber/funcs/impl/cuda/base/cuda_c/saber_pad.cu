#include "saber/funcs/impl/cuda/saber_pad.h"
#include "cuda_fp16.h"

namespace anakin {

namespace saber {

template <typename Dtype>
__global__ void ker_pad_fwd(Dtype * out_data, \
                    const Dtype* in_data,
                    const int in_n_stride,
                    const int in_c_stride,
                    const int in_h_stride, 
                    const int in_w_stride,
                    const int out_n_stride,
                    const int out_c_stride,
                    const int out_h_stride,
                    const int out_w_stride,
                    const int in_n, 
                    const int in_c,
                    const int in_h,
                    const int in_w,
                    const int num_threads)
{
    CUDA_KERNEL_LOOP(tid, num_threads){
        int n = (tid / in_n_stride) % in_n;
        int c = (tid / in_c_stride) % in_c;
        int h = (tid / in_h_stride) % in_h;
        int w = (tid / in_w_stride) % in_w;
        int out_offset = n * out_n_stride + c * out_c_stride + h * out_h_stride + w * out_w_stride; 
        out_data[out_offset] = in_data[tid];
    }
}

template <DataType OpDtype>
SaberStatus SaberPad<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    PadParam<NV>& param) {

    const dtype* in_data = static_cast<const dtype*>(inputs[0]->data());
    dtype* out_data = static_cast<dtype*>(outputs[0]->mutable_data());
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = inputs[0]->valid_size();
    int in_n = inputs[0]->num();
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    int out_size = outputs[0]->valid_size();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        cudaMemsetAsync(out_data, 0, out_size * sizeof(dtype), cuda_stream);
        ker_pad_fwd<dtype>\
                 <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                 out_data + _img_offset, in_data, _in_n_stride, \
                 _in_c_stride, _in_h_stride, _in_w_stride,\
                 _out_n_stride, _out_c_stride, _out_h_stride, _out_w_stride,\
                 in_n, in_c, in_h, in_w, count);
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberPad, PadParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberPad, PadParam, NV, AK_INT8);
}
}
