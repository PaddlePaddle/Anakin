#include "saber/funcs/impl/cuda/saber_cast.h"
#include "cuda_fp16.h"


namespace anakin {
namespace saber {

template <typename Dtype, typename Ttype>
__global__ void ker_cast_fwd(Ttype * out_data, \
                    const Dtype* in_data,
                    const int num_threads)
{
    CUDA_KERNEL_LOOP(tid, num_threads){
        out_data[tid] = static_cast<Ttype>(in_data[tid]);
    }
}



template <DataType OpDtype>
SaberStatus SaberCast<NV, OpDtype>::dispatch(const std::vector<Tensor<NV> *>& inputs,
    std::vector<Tensor<NV> *>& outputs,
    CastParam<NV>& param) {

    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int count = outputs[0]->valid_size();
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

    if(_inDtype == _outDtype){
        outputs[0]->copy_from(*inputs[0]);
        return SaberSuccess;
    }

    if(inputs[0]->get_dtype() == 1){//AK_FLOAT
        const float* in_data = (const float*)inputs[0]->data();
        int* out_data = (int*)outputs[0]->mutable_data();
        if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
            ker_cast_fwd<float, int>\
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                        out_data, in_data, count);
        }
        
    }
    
    if(inputs[0]->get_dtype() == 5){//AK_INT32
        const int* in_data = (const int*)inputs[0]->data();
        float* out_data = (float*)outputs[0]->mutable_data();
        if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
            ker_cast_fwd<int, float>\
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                        out_data, in_data, count);
        }
    }

    return SaberSuccess;
}
template class SaberCast<NV, AK_FLOAT>;
template class SaberCast<NV, AK_INT32>;
DEFINE_OP_TEMPLATE(SaberCast, CastParam, NV, AK_INT8);
DEFINE_OP_TEMPLATE(SaberCast, CastParam, NV, AK_HALF);
}
}
