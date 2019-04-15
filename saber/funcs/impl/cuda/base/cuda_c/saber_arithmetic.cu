#include "saber/funcs/impl/cuda/saber_arithmetic.h"
#include "saber/core/tensor_op.h"
#include "saber/core/target_wrapper.h"

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void ker_arithmetic_sum_fwd(Dtype * out_data,
                             const Dtype* in_data_0,
                             const Dtype* in_data_1,
                             const int* offset_0,
                             const int* offset_1,
                             const int* word_id_to_seq_id,
                             const int seq_num,
                             const int inner_size,
                             const int count) {
    CUDA_KERNEL_LOOP(tid, count) {
        int emb_id =  tid % inner_size;
        int word_id = tid / inner_size;
        int seq_id = word_id_to_seq_id[word_id];
        int word_id_in_cur_seq = word_id - offset_0[seq_id];
        int seq_len_1 = offset_1[seq_id+1] - offset_1[seq_id];
        if (word_id_in_cur_seq < seq_len_1) {
             out_data[tid] = in_data_0[tid] + in_data_1[(offset_1[seq_id] + word_id_in_cur_seq) * inner_size + emb_id];
        } else {
             out_data[tid] = in_data_0[tid];
        }
    }
}

template<typename Dtype>
__global__ void ker_arithmetic_sub_fwd(Dtype * out_data,
                             const Dtype* in_data_0,
                             const Dtype* in_data_1,
                             const int* offset_0,
                             const int* offset_1,
                             const int* word_id_to_seq_id,
                             const int seq_num,
                             const int inner_size,
                             const int count) {
    CUDA_KERNEL_LOOP(tid, count) {
        int emb_id =  tid % inner_size;
        int word_id = tid / inner_size;
        int seq_id = word_id_to_seq_id[word_id];
        int word_id_in_cur_seq = word_id - offset_0[seq_id];
        int seq_len_1 = offset_1[seq_id+1] - offset_1[seq_id];
        if (word_id_in_cur_seq < seq_len_1) {
             out_data[tid] = in_data_0[tid] - in_data_1[(offset_1[seq_id] + word_id_in_cur_seq) * inner_size + emb_id];
        } else {
             out_data[tid] = in_data_0[tid];
        }
    }
}

template<typename Dtype>
__global__ void ker_arithmetic_mul_fwd(Dtype * out_data,
                             const Dtype* in_data_0,
                             const Dtype* in_data_1,
                             const int* offset_0,
                             const int* offset_1,
                             const int* word_id_to_seq_id,
                             const int seq_num,
                             const int inner_size,
                             const int count) {
    CUDA_KERNEL_LOOP(tid, count) {
        int emb_id =  tid % inner_size;
        int word_id = tid / inner_size;
        int seq_id = word_id_to_seq_id[word_id];
        int word_id_in_cur_seq = word_id - offset_0[seq_id];
        int seq_len_1 = offset_1[seq_id+1] - offset_1[seq_id];
        if (word_id_in_cur_seq < seq_len_1) {
             out_data[tid] = in_data_0[tid] * in_data_1[(offset_1[seq_id] + word_id_in_cur_seq) * inner_size + emb_id];
        } else {
             out_data[tid] = in_data_0[tid];
        }
    }
}



template <>
SaberStatus SaberArithmetic<NV, AK_FLOAT>::create( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ArithmeticParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    return SaberSuccess;
}

template <>
SaberStatus SaberArithmetic<NV, AK_FLOAT>::init( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ArithmeticParam<NV>& param, Context<NV>& ctx) {

    this->_ctx = &ctx;
    Shape shape({inputs[0]->num(), 1, 1, 1}, Layout_NCHW);
    word_id_to_seq_id.re_alloc(shape, AK_INT32);

    int offset_size = inputs[0]->get_seq_offset()[0].size();
    Shape offset_shape(std::vector<int>{offset_size, 1, 1, 1}, Layout_NCHW);
    offset_tensor_0.re_alloc(offset_shape, AK_INT32);
    offset_tensor_1.re_alloc(offset_shape, AK_INT32);
    
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberArithmetic<NV, AK_FLOAT>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ArithmeticParam<NV>& param) {

    const float *in_data_0 = (const float*)inputs[0]->data();
    const float *in_data_1 = (const float*)inputs[1]->data();
    float *out_data = (float*)outputs[0]->mutable_data();

    const int inner_size = inputs[0]->valid_size() / inputs[0]->num();
    const int count = inputs[0]->valid_size();

    Shape shape({inputs[0]->num(), 1, 1, 1}, Layout_NCHW);
    word_id_to_seq_id.reshape(shape);

    auto offset_0 = inputs[0]->get_seq_offset()[0];
    auto offset_1 = inputs[1]->get_seq_offset()[0];
    std::vector<int> word_seq_map;
    for (int i = 0; i < offset_0.size() - 1; i++) {
        for (int j = offset_0[i]; j < offset_0[i+1]; j++) {
            word_seq_map.push_back(i);
        }
    }
    
    int seq_num = offset_0.size() - 1;
    Shape offset_shape({seq_num + 1, 1, 1, 1}, Layout_NCHW);
    offset_tensor_0.reshape(offset_shape);
    offset_tensor_1.reshape(offset_shape);
    auto offset_data_0 = (int*)offset_tensor_0.mutable_data();
    auto offset_data_1 = (int*)offset_tensor_1.mutable_data();
    
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    int* gpu_map_data = (int *)word_id_to_seq_id.mutable_data();

    cudaMemcpyAsync(gpu_map_data, &word_seq_map[0], sizeof(int) * word_seq_map.size(), cudaMemcpyHostToDevice,cuda_stream);

    cudaMemcpyAsync(offset_data_0, &offset_0[0],  sizeof(int) * offset_0.size(), cudaMemcpyHostToDevice, cuda_stream);

    cudaMemcpyAsync(offset_data_1, &offset_1[0],  sizeof(int) * offset_1.size(), cudaMemcpyHostToDevice, cuda_stream);

    switch (param.op_type) {
        //out[0] = input_0[0] + input_1[0]
        case SUM:

            ker_arithmetic_sum_fwd<float>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data_0, in_data_1, offset_data_0, offset_data_1,
                    gpu_map_data, seq_num, inner_size, count);
            break;

        //out[0] = input_0[0] - input_1[0]
        case SUB:
            ker_arithmetic_sub_fwd<float>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data_0, in_data_1, offset_data_0, offset_data_1,
                    gpu_map_data, seq_num, inner_size, count);
            break;

        //out[0] = input_0[0] * input_1[0]
        case MUL:
            ker_arithmetic_mul_fwd<float>
                    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
                    out_data, in_data_0, in_data_1, offset_data_0, offset_data_1,
                    gpu_map_data, seq_num, inner_size, count);
            break;

    }
    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberArithmetic<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberArithmetic, ArithmeticParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberArithmetic, ArithmeticParam, NV, AK_INT8);
}
}
