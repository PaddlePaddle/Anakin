#include "saber/funcs/impl/cuda/saber_axpy.h"
#include "cuda_fp16.h"


namespace anakin{

namespace saber{

template <typename DataDtype>
__global__ void ker_axpy_fwd(int n, int img_size,  
        const DataDtype* scale, const DataDtype* x, const DataDtype* y, DataDtype* dst) {
    CUDA_KERNEL_LOOP(idx, n) {
        int scale_id = idx / img_size;
        dst[idx] = scale[scale_id] * x[idx] + y[idx];
    }
}

template <DataType OpDtype ,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberAxpy<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(const std::vector<DataTensor_in *>& inputs,
    std::vector<DataTensor_out *>& outputs,
    AxpyParam<OpTensor>& param) {

    cudaStream_t cuda_stream = this->_ctx.get_compute_stream();
    if (!(inputs[1]->valid_shape() == outputs[0]->valid_shape()) 
        || !(inputs[2]->valid_shape() == outputs[0]->valid_shape())) {
         return SaberUnKownError;
    }

    const InDataType* scale = inputs[0]->data();
    const InDataType* x = inputs[1]->data();
    const InDataType* y = inputs[2]->data();
    OutDataType* dst = outputs[0]->mutable_data();
    int img_size = outputs[0]->height() * outputs[0]->width();
    int count = outputs[0]->valid_size();
    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()
        && inputs[1]->is_continue_mem() && inputs[2]->is_continue_mem()) {
        ker_axpy_fwd<InDataType><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>\
                (count, img_size, scale, x, y, dst);
    }
    return SaberSuccess;
}
} //namespace anakin

} //namespace anakin
