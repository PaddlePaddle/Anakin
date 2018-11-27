#include "saber/funcs/impl/cuda/saber_crop.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_crop_fwd(Dtype * out_data, \
                    const Dtype* in_data,
                    const int in_n_stride,
                    const int in_c_stride,
                    const int in_h_stride, 
                    const int in_w_stride,
                    const int out_n_stride,
                    const int out_c_stride,
                    const int out_h_stride,
                    const int out_w_stride,
                    const int out_n, 
                    const int out_c,
                    const int out_h,
                    const int out_w,
                    const int num_threads)
{
    CUDA_KERNEL_LOOP(tid, num_threads){
        int n = (tid / out_n_stride) % out_n;
        int c = (tid / out_c_stride) % out_c;
        int h = (tid / out_h_stride) % out_h;
        int w = (tid / out_w_stride) % out_w;
        int in_offset = n * in_n_stride + c * in_c_stride + h * in_h_stride + w * in_w_stride; 
        out_data[tid] = in_data[in_offset];
    }
}


template <>
SaberStatus SaberCrop<NV,AK_FLOAT>::dispatch(const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    CropParam<NV>& param) {

    const OpDataType* in_data = (const OpDataType*)inputs[0]->data();
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = outputs[0]->valid_size();
    int out_n = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
           ker_crop_fwd<OpDataType>\
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                    out_data, in_data + _img_offset, \
                    _in_n_stride, _in_c_stride, _in_h_stride, _in_w_stride,\
                    _out_n_stride, _out_c_stride, _out_h_stride, _out_w_stride,\
                    out_n, out_c, out_h, out_w, count);
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberCrop, CropParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberCrop, CropParam, NV, AK_INT8);
}
}
