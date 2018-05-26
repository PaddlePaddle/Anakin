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



template <DataType OpDtype,
    DataType inDtype,
    DataType outDtype,
    typename LayOutType_op,
    typename LayOutType_in,
    typename LayOutType_out>
SaberStatus SaberCast<NV, OpDtype, inDtype, outDtype,\
    LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(const std::vector<DataTensor_in *>& inputs,
    std::vector<DataTensor_out *>& outputs,
    CastParam<OpTensor>& param) {

    const InDataType* in_data = inputs[0]->data();
    OutDataType* out_data = outputs[0]->mutable_data();
    cudaStream_t cuda_stream = this->_ctx.get_compute_stream();

    int count = outputs[0]->valid_size();
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
           ker_cast_fwd<InDataType, OutDataType>\
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(\
                    out_data, in_data, \
                    count);
    }

    return SaberSuccess;
}

}
}
