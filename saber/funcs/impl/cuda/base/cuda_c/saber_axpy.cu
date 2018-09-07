#include "saber/funcs/impl/cuda/saber_axpy.h"
#include "cuda_fp16.h"


namespace anakin{

namespace saber{

template<typename Dtype>
__global__ void ker_axpy_fwd(int n, int img_size,  
        const Dtype* scale, const Dtype* x, const Dtype* y, Dtype* dst) {
    CUDA_KERNEL_LOOP(idx, n) {
        int scale_id = idx / img_size;
        dst[idx] = scale[scale_id] * x[idx] + y[idx];
    }
}

template <>
SaberStatus SaberAxpy<NV, AK_FLOAT>::dispatch( \
                        const std::vector<Tensor<NV> *>& inputs,
                        std::vector<Tensor<NV> *>& outputs,
                        AxpyParam<NV>& param){
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    if (!(inputs[1]->valid_shape() == outputs[0]->valid_shape()) 
        || !(inputs[2]->valid_shape() == outputs[0]->valid_shape())) {
         return SaberUnKownError;
    }

    const OpDataType* scale = (OpDataType*)inputs[0]->data();
    const OpDataType* x = (OpDataType*)inputs[1]->data();
    const OpDataType* y = (OpDataType*)inputs[2]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();
    int img_size = outputs[0]->height() * outputs[0]->width();
    int count = outputs[0]->valid_size();
    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()
        && inputs[1]->is_continue_mem() && inputs[2]->is_continue_mem()) {
        ker_axpy_fwd<OpDataType><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>\
                (count, img_size, scale, x, y, dst);
    }
    //LOG(INFO) << "passed";
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberAxpy, AxpyParam, NV, AK_INT8);
DEFINE_OP_TEMPLATE(SaberAxpy, AxpyParam, NV, AK_HALF);
} //namespace anakin

} //namespace anakin
