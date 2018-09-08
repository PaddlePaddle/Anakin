#include "saber/funcs/impl/cuda/saber_pooling_with_index.h"
#include "cuda_fp16.h"
#include <cfloat>

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_pool_with_index_fwd(Dtype * out_data, 
                    Dtype* out_index,
                    const Dtype* in_data,
                    const int in_n_stride,
                    const int in_c_stride,
                    const int in_h_stride, 
                    const int in_w_stride,
                    const int in_h,
                    const int in_w,
                    const int out_n_stride,
                    const int out_c_stride,
                    const int out_h_stride,
                    const int out_w_stride,
                    const int out_h,
                    const int out_w,
                    const int in_n, 
                    const int in_c,
                    const int pad_h,
                    const int pad_w,
                    const int stride_h,
                    const int stride_w,
                    const int window_h,
                    const int window_w,
                    const int num_threads)
{
    CUDA_KERNEL_LOOP(tid, num_threads){
        int n = (tid / out_n_stride) % in_n;
        int c = (tid / out_c_stride) % in_c;
        int h = (tid / out_h_stride) % out_h;
        int w = (tid / out_w_stride) % out_w;
        Dtype max_data = -FLT_MAX;
        Dtype max_index = 0;
        int start_h = h * stride_h - pad_h;
        int end_h = start_h + window_h;
        start_h = start_h < 0 ? 0 : start_h;
        end_h = end_h > in_h ? in_h : end_h;

        int start_w = w * stride_w - pad_w;
        int end_w = start_w + window_w;
        start_w = start_w < 0 ? 0 : start_w;
        end_w = end_w > in_w ? in_w : end_w;
        
        int in_offset = n * in_n_stride + c * in_c_stride; 
        for (int i = start_h; i < end_h; i++) {
            for (int j = start_w; j < end_w; j++) {
                Dtype data = in_data[in_offset + i * in_h_stride + j * in_w_stride];
                if (data > max_data) {
                    max_data = data;
                    max_index = i * in_w + j;
                }
            }
        }
        out_data[tid] = max_data;
        out_index[tid] = max_index;
    }
}

template <DataType OpDtype>
SaberStatus SaberPoolingWithIndex<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    PoolingParam<NV>& param) {

    const dtype* in_data = static_cast<const dtype*>(inputs[0]->data());
    dtype* out_data = static_cast<dtype*>(outputs[0]->mutable_data());
    dtype* out_index = static_cast<dtype*>(outputs[1]->mutable_data());
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = outputs[0]->valid_size();
    int out_n = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        ker_pool_with_index_fwd<dtype>\
                 <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                 out_data, out_index, in_data, \
                 _in_n_stride, _in_c_stride, \
                 _in_h_stride, _in_w_stride,\
                 in_h, in_w, \
                 _out_n_stride, _out_c_stride, \
                 _out_h_stride, _out_w_stride,\
                 out_h, out_w, out_n, out_c, \
                 param.pad_h, param.pad_w, \
                 param.stride_h, param.stride_w, \
                 param.window_h, param.window_w, count);
        return SaberSuccess;
    } else {
        LOG(ERROR) <<"pooling_with_index only support continue memory";
        return SaberUnImplError;
    }
}
DEFINE_OP_TEMPLATE(SaberPoolingWithIndex, PoolingParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberPoolingWithIndex, PoolingParam, NV, AK_INT8);
}
}
