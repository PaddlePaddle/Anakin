#include "saber/funcs/impl/cuda/saber_lstmp.h"
#include "saber/core/tensor_op.h"
#include "cuda_inline_activation.h"
#include "cuda_utils.h"
namespace anakin {

namespace saber {

static void cudnn_gemm(cublasHandle_t handle, const bool TransA,
                           const bool TransB, const int M, const int N, const int K,
                           const float alpha, const float* A, const float* B, const float beta,
                           float* C) {
    // Note that cublas follows fortran order.
    int lda = (!TransA/* == CblasNoTrans*/) ? K : M;
    int ldb = (!TransB/* == CblasNoTrans*/) ? N : K;
    cublasOperation_t cuTransA =
            (!TransA/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB =
            (!TransB/* == CblasNoTrans*/) ? CUBLAS_OP_N : CUBLAS_OP_T;
    CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA,
                             N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <typename Dtype,bool first_iter>
__global__ void kernel_lstm_with_peephole(
    const Dtype* w_x,  const Dtype* b_i, const Dtype* b_f, const Dtype* b_c, const Dtype* b_o,
    const Dtype* w_ci, const Dtype* w_cf, const Dtype* w_co, Dtype* cell, const int hidden_size,
    const int batch_size,
    Dtype* output) {


    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_id = thread_id / hidden_size;
    const int tid = thread_id % hidden_size;

    if (tid < hidden_size && batch_id < batch_size) {
        const int emit_wx_offset = batch_id * hidden_size * 4;
        const Dtype* w_x_i = w_x + emit_wx_offset;
        const Dtype* w_x_f = w_x_i + hidden_size ;
        const Dtype* w_x_c = w_x_f + hidden_size;
        const Dtype* w_x_o = w_x_c + hidden_size;
        Dtype* gate_h_p = output + batch_id * hidden_size;
        Dtype* gate_c_p = cell + batch_id * hidden_size;
        if(first_iter){
            const Dtype gate_i = Sigmoid(w_x_i[tid] + b_i[tid]);
            const Dtype gate_f = Sigmoid(w_x_f[tid] + b_f[tid]);

            const Dtype gate_c_s = Tanh(w_x_c[tid]  + b_c[tid]);
            const Dtype gate_c = gate_i * gate_c_s;
            const Dtype gate_o = Sigmoid(w_x_o[tid] + b_o[tid] + gate_c * w_co[tid]);
            gate_c_p[tid] = gate_c;
            gate_h_p[tid] = gate_o * Tanh(gate_c);
        }else{
            const Dtype c_1 = gate_c_p[tid];
            const Dtype gate_i = Sigmoid(w_x_i[tid] + b_i[tid] + w_ci[tid] * c_1);
            const Dtype gate_f = Sigmoid(w_x_f[tid] + b_f[tid] + w_cf[tid] * c_1);

            const Dtype gate_c_s = Tanh(w_x_c[tid]  + b_c[tid]);
            const Dtype gate_c = gate_f * c_1 + gate_i * gate_c_s;
            const Dtype gate_o = Sigmoid(w_x_o[tid] + b_o[tid] + gate_c * w_co[tid]);
            gate_c_p[tid] = gate_c;
            gate_h_p[tid] = gate_o * Tanh(gate_c);
        }
    }
}

template <typename Dtype,bool first_iter>
void cal_lstm_batch(int emit_word_id_size, Dtype* temp_wx,
                 const Dtype* weight_peephole,
                    Dtype* hout, Dtype* inner_cell, const Dtype* b_i_in, const Dtype* b_f_in, const Dtype* b_c_in,
                 const Dtype* b_o_in, int hidden_size,cudaStream_t cuda_stream){
    const int block_dim=256;
    const int grid_dim=utils::div_up(emit_word_id_size*hidden_size,block_dim);
    const Dtype* wc_i=weight_peephole;
    const Dtype* wc_f=weight_peephole+hidden_size;
    const Dtype* wc_o=weight_peephole+2*hidden_size;
    kernel_lstm_with_peephole<Dtype,first_iter><<<grid_dim,block_dim,0, cuda_stream>>>(temp_wx,b_i_in,b_f_in,b_c_in,b_o_in,wc_i,wc_f,wc_o,inner_cell,hidden_size,emit_word_id_size,hout);

};

template <typename Dtype>
__global__ void kernel_vTanh(Dtype* data,int count){
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id<count){
        data[thread_id]=Tanh(data[thread_id]);
    }
};

template <typename Dtype>
static inline void vTanh(Dtype* data,int count,cudaStream_t cuda_stream){
    kernel_vTanh<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(data,count);
}


template<>
SaberStatus
SaberLstmp<NV, AK_FLOAT>::dispatch(
    const std::vector < Tensor<NV>* >& inputs,
    std::vector < Tensor<NV>* >& outputs,
    LstmParam < NV >& param) {
    auto offset_vec = inputs[0]->get_seq_offset();
    CHECK_EQ(offset_vec.size(), 1);
    auto offset = offset_vec[0];
    CHECK_EQ(offset.size(), 2);
    const int skip_num = param.skip_num;
    CHECK_GT(skip_num, 1);
    int word_num = inputs[0]->num();
    int word_dim = inputs[0]->channel();
    int iter_num = utils::div_up(word_num, skip_num);

    utils::try_expand_tensor(_wx_tensor,word_num*4*_inner_hidden_dim);
    utils::try_expand_tensor(_temp_hidden_tensor,skip_num*_inner_hidden_dim);
    utils::try_expand_tensor(_temp_cell_tensor,skip_num*_inner_hidden_dim);

    float* wx_ptr = static_cast<float*>(_wx_tensor.mutable_data());
    const float* x_ptr = static_cast<const float*>(inputs[0]->data());
    const float* weights_x_ptr = static_cast<const float*>(param.weight()->data());
    const float* weights_h_ptr = weights_x_ptr + word_dim * _inner_hidden_dim * 4;
    const float* weights_project_ptr = weights_h_ptr + _output_hidden_dim * _inner_hidden_dim * 4;
    const float* weights_bias_ptr = static_cast<const float*>(param.bias()->data());
    const float* weights_bias_i_ptr = weights_bias_ptr;
    const float* weights_bias_f_ptr = weights_bias_i_ptr + _inner_hidden_dim;
    const float* weights_bias_c_ptr = weights_bias_f_ptr + _inner_hidden_dim;
    const float* weights_bias_o_ptr = weights_bias_c_ptr + _inner_hidden_dim;
    const float* weights_peephole_ptr = weights_bias_ptr + _inner_hidden_dim * 4;
    float* output_ptr = static_cast<float*>(outputs[0]->mutable_data());
    float* temp_hidden_out = static_cast<float*>(_temp_hidden_tensor.mutable_data());
    float* temp_cell_out = static_cast<float*>(_temp_cell_tensor.mutable_data());

    cudaStream_t stream=_ctx->get_compute_stream();
    cudnn_gemm(_handle,false, false, word_num, 4*_inner_hidden_dim, word_dim, 1.f, x_ptr, weights_x_ptr, 0.f, wx_ptr);

    for (int i = 0; i < iter_num; i++) {
        const int run_batch_dim=(i==(iter_num-1))?(word_num-skip_num*i):skip_num;
        float* wx_iter = wx_ptr + i * skip_num * 4 * _inner_hidden_dim;
        if(i>=1){
            float* hidden_in = output_ptr + (i - 1) * skip_num * _output_hidden_dim;
            cudnn_gemm(_handle,false, false, run_batch_dim, 4*_inner_hidden_dim, _output_hidden_dim, 1.f, hidden_in, weights_h_ptr,
                 1.f, wx_iter);

            cal_lstm_batch<float,false>(run_batch_dim, wx_iter, weights_peephole_ptr, temp_hidden_out, temp_cell_out,weights_bias_i_ptr,weights_bias_f_ptr,weights_bias_c_ptr,weights_bias_o_ptr,_inner_hidden_dim,stream);

        }else{
            cal_lstm_batch<float,true>(run_batch_dim, wx_iter, weights_peephole_ptr, temp_hidden_out, temp_cell_out,weights_bias_i_ptr,weights_bias_f_ptr,weights_bias_c_ptr,weights_bias_o_ptr,_inner_hidden_dim,stream);
        }

        float* hidden_out = output_ptr + i * skip_num * _output_hidden_dim;
        cudnn_gemm(_handle,false,false,run_batch_dim,_output_hidden_dim,_inner_hidden_dim,1.f,temp_hidden_out,weights_project_ptr,0.f,hidden_out);
        vTanh(hidden_out,run_batch_dim*_output_hidden_dim,stream);
    }
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    return SaberSuccess;

};


DEFINE_OP_TEMPLATE(SaberLstmp, LstmParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberLstmp, LstmParam, NV, AK_INT8);
}
}

