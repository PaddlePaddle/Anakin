#include "saber/funcs/impl/cuda/saber_sequence_concat.h"
#include "saber/core/tensor_op.h"
#define BUILD_DEV __device__

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void ker_sequence_concat_fwd(Dtype * out_data,
                             const uint64_t* in_locate_data, 
                             const int* o2i_map,
                             const int* o2i_w_map,
                             const int seq_num,
                             const int emb_size,
                             const int count) {
    CUDA_KERNEL_LOOP(tid, count) {
        int emb_id = tid % emb_size;
        int word_id = tid / emb_size;
        int input_id = o2i_map[word_id];
        int cur_word_id = o2i_w_map[word_id];
        const Dtype* in_data = (const Dtype*)(in_locate_data[input_id]);
        out_data[tid] = in_data[cur_word_id * emb_size + emb_id];
    }
}


template <>
SaberStatus SaberSequenceConcat<NV, AK_FLOAT>::create( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        SequenceConcatParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return SaberSuccess;
}

template <>
SaberStatus SaberSequenceConcat<NV, AK_FLOAT>::init( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        SequenceConcatParam<NV>& param, Context<NV>& ctx) {
    int out_num = 0;
    for (int i = 0; i < inputs.size(); i++) {
        out_num += inputs[i]->num();
    }
    Shape shape({out_num, 1, 1, 1}, Layout_NCHW);
	_out2in_map_tensor.re_alloc(shape, AK_INT32);
    _out2in_word_map_tensor.re_alloc(shape, AK_INT32);
    
    int in_num = inputs.size();
    Shape in_locate_shape({in_num, 1, 1, 1}, Layout_NCHW);
    _in_locate_tensor.re_alloc(in_locate_shape, AK_UINT64);

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberSequenceConcat<NV, AK_FLOAT>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        SequenceConcatParam<NV>& param) {
/*
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int seq_num = inputs[0]->get_seq_offset()[0].size() - 1;
    const int emb_size = inputs[0]->valid_size() / inputs[0]->num();
    float *output_data = (float*)outputs[0]->mutable_data();
    for (int i = 0; i < seq_num; i++) {
        for (int j = 0; j < inputs.size(); j++) {
            size_t cur_len = inputs[j]->get_seq_offset()[0][i+1] - inputs[j]->get_seq_offset()[0][i];

            const OpDataType *input_data = (const OpDataType*)inputs[j]->data() + inputs[j]->get_seq_offset()[0][i] * emb_size;
            cudaMemcpyAsync(output_data, input_data, sizeof(OpDataType) * cur_len * emb_size, cudaMemcpyDeviceToDevice, cuda_stream);
            output_data += cur_len * emb_size;
        }
    }
*/

    int seq_num = inputs[0]->get_seq_offset()[0].size() - 1;
    const int emb_size = inputs[0]->valid_size() / inputs[0]->num();
    for (int i = 1; i < inputs.size(); i++) {
        int cur_emb_size = inputs[i]->valid_size() / inputs[i]->num();
        int cur_seq_num  = inputs[i]->get_seq_offset()[0].size() - 1;
        CHECK_EQ(emb_size, cur_emb_size) << "sequence concat emb size must be the same";
        CHECK_EQ(seq_num, cur_seq_num) << "sequence concat seq num must be the same";
    }

    float *out_data = (float*)outputs[0]->mutable_data();
    std::vector<uint64_t> in_locate_vec;
    for (int i = 0; i < inputs.size(); i++) {
        //in_locate_vec.push_back(static_cast<uint64_t>(inputs[i]->data()));
        in_locate_vec.push_back((uint64_t)(inputs[i]->data()));
    }
    std::vector<int> out2in_map;
    std::vector<int> out2in_word_map;
    for (int i = 0; i < seq_num; i++) {
        for (int j = 0; j < inputs.size(); j++) {
             auto offset = inputs[j]->get_seq_offset()[0];
             int cur_len = offset[i+1] - offset[i];
             for (int k = 0; k < cur_len; k++) {
                  out2in_map.push_back(j);
                  out2in_word_map.push_back(offset[i] + k);
             }
        } 
    }
    int word_num = out2in_map.size();
    Shape o2i_map_shape({word_num, 1, 1, 1}, Layout_NCHW);
    _out2in_map_tensor.reshape(o2i_map_shape);
    _out2in_word_map_tensor.reshape(o2i_map_shape);

    
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int* gpu_o2i_map_data = (int *)_out2in_map_tensor.mutable_data();
    int* gpu_o2i_w_map_data = (int *)_out2in_word_map_tensor.mutable_data();
    uint64_t* gpu_in_locate_data = (uint64_t*)_in_locate_tensor.mutable_data();

    cudaMemcpyAsync(gpu_o2i_map_data, &out2in_map[0], sizeof(int) * out2in_map.size(), cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(gpu_o2i_w_map_data, &out2in_word_map[0], sizeof(int) * out2in_word_map.size(), cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(gpu_in_locate_data, &in_locate_vec[0], sizeof(uint64_t) * in_locate_vec.size(), cudaMemcpyHostToDevice, cuda_stream);


    int count = inputs[0]->valid_size();
    for (int i = 1; i < inputs.size(); i++) {
        count += inputs[i]->valid_size();
    }
    ker_sequence_concat_fwd<float>
            <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
            out_data, gpu_in_locate_data, gpu_o2i_map_data, gpu_o2i_w_map_data, 
            seq_num, emb_size, count);

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberSequenceConcat<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberSequenceConcat, SequenceConcatParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberSequenceConcat, SequenceConcatParam, NV, AK_INT8);
}
}
