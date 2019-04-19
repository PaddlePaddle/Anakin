#include "saber/funcs/impl/cuda/saber_mean.h"

namespace anakin {
namespace saber {
 
template <typename dtype, unsigned int blockSize>
__global__ void mean_kernel(const dtype* input, dtype* output, const int count) {
    
    int tid = threadIdx.x;
    int n_id = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_num = blockDim.x * gridDim.x;
    extern __shared__ dtype sdata[];
    if (n_id==0) output[0] = (dtype)0.0;
    dtype sum = (dtype)0.0;
    for (int thread = n_id; thread < count; thread += thread_num) {
        sum += input[thread];
    }
    sdata[tid] = sum;
    __syncthreads();

    int powOf2 = blockDim.x;
    if (powOf2 & (powOf2-1)) {
        // thread block is not pow of 2.
        while (powOf2 & (powOf2-1)) {
            powOf2 &= (powOf2-1);
        }
        // find a num which is pow of 2.
        if (tid >= powOf2) {
            sdata[tid - powOf2] += sdata[tid];
        }
        __syncthreads();
    }
    for (unsigned int i = powOf2 >> 1; i > 0; i>>=1) {
        if ( tid < i) {
            sdata[tid] += sdata[tid + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sdata[0] /= count;
        atomicAdd(&output[0], sdata[0]);
    }
}

//compute a mean of input tensor's all elements.
template <DataType OpDtype>
SaberStatus SaberMean<NV, OpDtype>::dispatch(const std::vector<Tensor<NV>*>& inputs,
    std::vector<Tensor<NV>*>& outputs,
    MeanParam<NV>& param) {
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    const OpDataType* input_ptr = (const OpDataType*)inputs[0]->data();
    OpDataType* output_ptr = (OpDataType*)outputs[0]->mutable_data();
    int count = inputs[0]->valid_size();
    int thread_num;
    int grid;
    unsigned int blockSize;
    if (count < CUDA_NUM_THREADS) {
        thread_num = count;
        grid = 1;
        blockSize = count;
    } else {
        thread_num = CUDA_NUM_THREADS;
        if (CUDA_GET_BLOCKS(count) >= 128)
            grid = 64;
        else
            grid = CUDA_GET_BLOCKS(count);
        blockSize = CUDA_NUM_THREADS;
    }

    mean_kernel<OpDataType, CUDA_NUM_THREADS><<<grid, thread_num, thread_num*4, cuda_stream>>>(
        input_ptr,
        output_ptr,
        count
    );

    CUDA_POST_KERNEL_CHECK;

    return SaberSuccess;
}

template class SaberMean<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberMean, MeanParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberMean, MeanParam, NV, AK_INT8);

} // namespace saber.
} // namespace anakin.