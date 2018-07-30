#include "saber/funcs/impl/cuda/saber_sequence_expand.h"
#include "cuda_fp16.h"

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void ker_sequence_expand(Dtype * out_data,
                   const Dtype* in_data, const int * seq_id_map,  const int count, const int dim) {
    CUDA_KERNEL_LOOP(tid, count){
        int dim_id = tid % dim;
        int word_id = tid / dim;
        int seq_id = seq_id_map[word_id];
        Dtype in_var = in_data[seq_id * dim + dim_id];
        out_data[tid] = in_var; 
        
    }
}


template <>
SaberStatus SaberSequenceExpand<NV, AK_FLOAT, AK_FLOAT, AK_FLOAT, \
        NCHW, NCHW, NCHW>::dispatch( \
        const std::vector<DataTensor_in*>& inputs,
        std::vector<DataTensor_out*>& outputs,
        SequenceExpandParam<OpTensor>& param) {

    Shape in_shape = inputs[0]->valid_shape();
    Shape out_shape = outputs[0]->valid_shape();

    const InDataType *in_data = (const InDataType*)inputs[0]->data();
    OutDataType *out_data = (OutDataType*)outputs[0]->mutable_data();

    const int count = inputs[0]->valid_size();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    auto ref_seq_offset = inputs[1]->get_seq_offset();
    auto cur_seq_offset = inputs[0]->get_seq_offset();
    std::vector<int> id_map;
    if (cur_seq_offset.size() == 0) {
        for (int i = 0; i < ref_seq_offset.size(); i++) {
            for (int j = ref_seq_offset[i]; j < ref_seq_offset[i+1]; j++) {
                id_map.push_back(i);
            }
        }
    } else {
        for (int i = 0; i < ref_seq_offset.size(); i++) {
            for (int j = ref_seq_offset[i]; j < ref_seq_offset[i+1]; j++) {
                for (int k = cur_seq_offset[i]; k < cur_seq_offset[i+1]; k++) {
                     id_map.push_back(k);
                }
            }
        }
     }

     _seq_id_map.reshape(Shape(id_map.size(), 1, 1, 1));
     cudaMemcpyAsync(_seq_id_map.mutable_data(), &id_map[0], sizeof(int) * id_map.size(), cudaMemcpyHostToDevice, cuda_stream);

     ker_sequence_expand<InDataType>
             <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
             out_data, in_data, _seq_id_map.data(), count, inputs[0]->valid_size() / in_shape[0]);

     CUDA_POST_KERNEL_CHECK;
     return SaberSuccess;
}

}
}
