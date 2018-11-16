#include "saber/funcs/impl/cuda/saber_match_matrix.h"
#include "cuda_fp16.h"

namespace anakin{
namespace saber{
template <typename dtype>
void gpu_transpose(cublasHandle_t handle, const dtype *src, int M, int N, dtype *dst);

template<>
void gpu_transpose<float>(cublasHandle_t handle, const float *src, int M, int N, float *dst){
    float alpha = 1.0;
    float beta = 0.0;
    CUBLAS_CHECK(
        cublasSgeam(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    M, N,
                    &alpha,
                    src, N,
                    &beta,
                    dst, M,
                    dst, M)
    );
}
template<typename dtype>
__global__ void padding_out(const dtype* src,
                            const int* offset,
                            const int seq_num_r,
                            const int max_len_r,
                            const int tl,
                            const int count,
                            dtype* dst) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_num = blockDim.x * gridDim.x;
    for (tid = threadIdx.x + blockIdx.x * blockDim.x; tid < count; tid += thread_num) {
        int seq_id = tid / (tl * max_len_r);
        int tl_id = (tid / (max_len_r)) % tl;
        int r_id = tid % max_len_r;
        int cur_len = offset[seq_id + 1] - offset[seq_id];
        if (r_id < cur_len) {
           dst[tid] = src[(offset[seq_id] + r_id) * tl + tl_id];
        } else {
           dst[tid] = 0.f;
        }
    }
}

template <DataType OpDtype>
SaberStatus SaberMatchMatrix<NV, OpDtype>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        MatchMatrixParam<NV>& param) {
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    Shape in_shape = inputs[0]->valid_shape();
    Shape out_shape = outputs[0]->valid_shape();
    CHECK(inputs[0]->get_seq_offset().size() > 0 && inputs[0]->get_seq_offset()[0].size() > 0) 
            << "input_0 sequence offset is not valid";
    CHECK(inputs[1]->get_seq_offset().size() > 0 && inputs[1]->get_seq_offset()[0].size() > 0)
            << "input_1 sequence offset is not valid";
    int dim_t = param.dim_t;
    int dim_in = param.dim_in;
    auto offset_l = inputs[0]->get_seq_offset()[0];
    auto offset_r = inputs[1]->get_seq_offset()[0];
    _offset_r.reshape(std::vector<int>{offset_r[offset_r.size() - 1], 1, 1, 1});
    cudaMemcpyAsync(_offset_r.mutable_data(), &offset_r[0],
            sizeof(int) * offset_r.size(),
            cudaMemcpyHostToDevice, cuda_stream);

    int len_l = offset_l[1] - offset_l[0];
    int len_r = offset_r[offset_r.size() - 1];
    
    
    const OpDataType *input_l = (const OpDataType*)inputs[0]->data();
    const OpDataType *input_r = (const OpDataType*)inputs[1]->data();
    const OpDataType *weight_data = (const OpDataType*)param.weight()->data();
    OpDataType* input_l_transform = (OpDataType*)_input_l_transform.mutable_data();
    OpDataType* input_l_transform_reorganize = (OpDataType*)_input_l_transform_reorganize.mutable_data();
    OpDataType* output_tmp = (OpDataType*)_output_tmp.mutable_data();
    OpDataType *out_data = (OpDataType*)outputs[0]->mutable_data();
    _gemm_l_transform.init(true, true, dim_t * dim_in, len_l, dim_in, *(this->_ctx));
    _gemm_l_transform.dispatch(1.0f, 0.f, weight_data, input_l, input_l_transform);
    for (int i = 0; i < dim_t; i++) {
        int offset = i * dim_in * len_l;
        gpu_transpose(_handle, 
              input_l_transform + offset,
              dim_in,
              len_l,
              input_l_transform_reorganize + offset);
    }
    _gemm_r_transform.init(false, true, len_r, dim_t * len_l, dim_in, *(this->_ctx));
    _gemm_r_transform.dispatch(1.0f, 0.f, input_r, input_l_transform_reorganize, output_tmp);
    int max_len_r = 0;
    for (int i = 0; i < offset_r.size() - 1; i++) {
        int cur_len = offset_r[i+1] - offset_r[i];
        max_len_r = cur_len > max_len_r ? cur_len : max_len_r;
    }
    int seq_num = offset_r.size() - 1;
    int count = seq_num * max_len_r * dim_t * len_l;
    padding_out<OpDataType><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 
            0, cuda_stream>>>((const OpDataType*)(_output_tmp.data()),
                              (const int*)(_offset_r.data()), 
                              seq_num,
                              max_len_r, 
                              dim_t * len_l,
                              count,
                              out_data);
     
    CUDA_POST_KERNEL_CHECK;
    
    return SaberSuccess;
}

template class SaberMatchMatrix<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberMatchMatrix, MatchMatrixParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberMatchMatrix, MatchMatrixParam, NV, AK_INT8);
}
}
