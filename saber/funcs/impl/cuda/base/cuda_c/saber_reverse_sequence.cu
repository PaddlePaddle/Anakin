
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/cuda/saber_reverse_sequence.h"
#include "saber/funcs/saber_util.h"

namespace anakin {
namespace saber {

template<DataType OpDtype>
SaberStatus SaberReverseSequence<NV, OpDtype>::init(const std::vector<OpTensor*>& inputs,
                                                 std::vector<OpTensor*>& outputs,
                                                 EmptyParam<NV> &param,
                                                 Context<NV> &ctx) {
    this->_ctx=&ctx;

    return create(inputs,outputs,param,ctx);
};
template<DataType OpDtype>
SaberStatus SaberReverseSequence<NV, OpDtype>::create(const std::vector<OpTensor*>& inputs,
                                                   std::vector<OpTensor*>& outputs,
                                                   EmptyParam<NV> &param,
                                                   Context<NV> &ctx) {
    if(this->_ctx=&ctx){
        this->_ctx=&ctx;
    }
    int input_size=inputs.size();
    CHECK_EQ(input_size,1)<<"only support one input now";
    return SaberSuccess;
};

static inline int round_up(int k, int c) {
    return ((k + c - 1) / c);
}

template <typename Dtype>
__global__ static void ker_reverse_sequence(const Dtype* in,Dtype* out,int length,int word_size,int* offset){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<length){
        int word_id=tid/word_size;
        int word_inner_id=tid%word_size;
        out[offset[word_id]*word_size+word_inner_id]=in[tid];
    }
}

template<DataType OpDtype>
SaberStatus SaberReverseSequence<NV, OpDtype>::dispatch(const std::vector<OpTensor*>& inputs,
                                                     std::vector<OpTensor*>& outputs,
                                                     EmptyParam<NV> &param) {
    int input_size=inputs.size();
    CHECK_EQ(input_size,1)<<"only support one input now";

    cudaStream_t stream=this->_ctx->get_compute_stream();
    std::vector<std::vector<int>> offset_vec=inputs[0]->get_seq_offset();
    std::vector<int> offset=offset_vec[offset_vec.size()-1];


    int batch_size=offset.size()-1;
    int word_size=inputs[0]->valid_shape()[1];
    int word_sum=offset[batch_size];

    utils::try_expand_tensor(_offset_map,word_sum);
    utils::try_expand_tensor(_offset_map_cu,word_sum);
    int* offset_map_ptr= static_cast<int*>(_offset_map.mutable_data());
    int* offset_map_cu_ptr= static_cast<int*>(_offset_map_cu.mutable_data());

    for (int i = 0; i < batch_size; i++) {
        int seq_len = offset[i + 1] - offset[i];
        int start_word_id=offset[i];
        for (int j = 0; j < seq_len; j++) {
            offset_map_ptr[start_word_id+seq_len-1-j]=start_word_id+j;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(offset_map_cu_ptr,offset_map_ptr, sizeof(int)*word_sum,cudaMemcpyHostToDevice,stream));
    int tid_sum=word_sum*word_size;
    int block_dim=256;
    if(tid_sum<block_dim){
        block_dim=tid_sum;
    }
    int grid_dim=round_up(tid_sum,block_dim);
    const OpDataType* in= static_cast<const OpDataType*>(inputs[0]->data());
    OpDataType* out=static_cast<OpDataType*>(outputs[0]->mutable_data());
    ker_reverse_sequence<<<grid_dim,block_dim,0,stream>>>(in,out,tid_sum,word_size,offset_map_cu_ptr);

    return SaberSuccess;

};

template class SaberReverseSequence<NV, AK_INT32>;
template class SaberReverseSequence<NV, AK_FLOAT>;
template class SaberReverseSequence<NV, AK_HALF>;
template class SaberReverseSequence<NV, AK_INT8>;

}
}
