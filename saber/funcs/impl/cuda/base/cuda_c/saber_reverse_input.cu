
#include "saber/saber_funcs_param.h"
#include "saber/funcs/impl/cuda/saber_reverse_input.h"
#include "saber/funcs/saber_util.h"

namespace anakin {
namespace saber {

template<DataType OpDtype>
SaberStatus SaberReverseInput<NV, OpDtype>::init(const std::vector<OpTensor*>& inputs,
                                                  std::vector<OpTensor*>& outputs,
                                                  EmptyParam<NV> &param,
                                                  Context<NV> &ctx) {
    this->_ctx=&ctx;
    for(int i=0;i<inputs.size();++i){
        _offset_map_vec.push_back(*new Tensor<NVHX86>());
        _offset_map_vec[i].set_dtype(AK_INT32);
        _offset_map_cu_vec.push_back(*new OpTensor());
        _offset_map_cu_vec[i].set_dtype(AK_INT32);
    }

    return create(inputs,outputs,param,ctx);
};
template<DataType OpDtype>
SaberStatus SaberReverseInput<NV, OpDtype>::create(const std::vector<OpTensor*>& inputs,
                                                    std::vector<OpTensor*>& outputs,
                                                    EmptyParam<NV> &param,
                                                    Context<NV> &ctx) {
    if(this->_ctx=&ctx){
        this->_ctx=&ctx;
    }
    return SaberSuccess;
};

static inline int round_up(int k, int c) {
    return ((k + c - 1) / c) * c;
}

template <typename Dtype>
__global__ static void ker_reverse_input(const Dtype* in,Dtype* out,int length,int* offset){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<length){
        out[offset[tid]]=in[tid];
    }
}

template<DataType OpDtype>
SaberStatus SaberReverseInput<NV, OpDtype>::dispatch(const std::vector<OpTensor*>& inputs,
                                                      std::vector<OpTensor*>& outputs,
                                                      EmptyParam<NV> &param) {
    int input_size=inputs.size();

    cudaStream_t stream=this->_ctx->get_compute_stream();
    for(int input_id=0;input_id<input_size;++input_id){
        std::vector<std::vector<int>> offset_vec=inputs[input_id]->get_seq_offset();
        std::vector<int> offset=offset_vec[offset_vec.size()-1];
        int word_sum=offset[offset.size()-1];
        utils::try_expand_tensor(_offset_map_vec[input_id],word_sum);
        utils::try_expand_tensor(_offset_map_cu_vec[input_id],word_sum);
        int* offset_map_ptr= static_cast<int*>(_offset_map_vec[input_id].mutable_data());
        int* offset_map_cu_ptr= static_cast<int*>(_offset_map_cu_vec[input_id].mutable_data());
        for(int sequence_id=0;sequence_id<offset.size()-1;sequence_id++){
            int start=offset[sequence_id];
            int end=offset[sequence_id+1]-1;
            for(int index=0;index<=end-start;index++){
                offset_map_ptr[end-index]=start+index;
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(offset_map_cu_ptr,offset_map_ptr, sizeof(int)*word_sum,cudaMemcpyHostToDevice,stream));
        int block_dim=256;
        if(word_sum<block_dim){
            block_dim=word_sum;
        }
        int grid_dim=round_up(word_sum,block_dim);
        const OpDataType* in= static_cast<const OpDataType*>(inputs[input_id]->data());
        OpDataType* out=static_cast<OpDataType*>(outputs[input_id]->mutable_data());
        ker_reverse_input<<<grid_dim,block_dim,0,stream>>>(in,out,word_sum,offset_map_cu_ptr);
    }

    return SaberSuccess;

};

template class SaberReverseInput<NV, AK_INT32>;
template class SaberReverseInput<NV, AK_FLOAT>;
template class SaberReverseInput<NV, AK_HALF>;
template class SaberReverseInput<NV, AK_INT8>;

}
}