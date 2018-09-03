#include "saber/funcs/impl/cuda/saber_spp.h"
#include "cuda_fp16.h"

namespace anakin {

namespace saber {

#if 0    
template <typename Dtype>
__global__ void ker_concat_fwd(Dtype* out_data, const Dtype* in_data,
                               const int n,
                               const int w,
                               const int n_stride, const int nthreads) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int n_id = index / w;
        const int w_id = index % w;
        const int out_index = n_id * n_stride + w_id;
        out_data[out_index] = in_data[index];
    }
}

template <DataType OpDtype,
            DataType inDtype,
            DataType outDtype,
            typename LayOutType_op,
            typename LayOutType_in,
            typename LayOutType_out>
SaberStatus SaberSpp<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(\
    const std::vector<DataTensor_in *>& inputs, \
    std::vector<DataTensor_out *>& outputs, \
    SPPParam<OpTensor>& param) {

    const InDataType* in_data = inputs[0]->data();
    OutDataType* out_data = outputs[0]->mutable_data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = outputs[0]->valid_size();
    int out_n = outputs[0]->num();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        std::vector<OpTensor*> pool_outputs;
        pool_outputs.resize(1);
        for (int i = 0; i < param.pyramid_height; i++) {
            pool_outputs[0] = _pooling_output[i];
            (*_pooling[i])(inputs, pool_outputs, _pooling_param[i], this->_ctx);
            int valid_size  = pool_outputs[0]->valid_size();
            int offset = (pow(4, i) - 1) / 3;
            ker_concat_fwd<InDataType><<<CUDA_GET_BLOCKS(valid_size),CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data + offset, 
                    pool_outputs[0]->data(), 
                    pool_outputs[0]->num() * pool_outputs[0]->channel(), 
                    pool_outputs[0]->height() * pool_outputs[0]->width(), 
                    outputs[0]->width(), 
                    valid_size);
        }
    }

    return SaberSuccess;
}
#endif
} //namespace saber

} //namespace anakin
