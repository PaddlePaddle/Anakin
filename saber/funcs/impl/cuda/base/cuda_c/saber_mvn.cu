#include "saber/funcs/impl/cuda/saber_mvn.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {
template <typename Dtype, unsigned int blockSize>
__global__ void sum_square(const Dtype* in_data,
                           const int height,
                           const int width,
                           Dtype* sum,
                           Dtype* square_sum) {
    __shared__ Dtype share_data[blockSize];
    __shared__ Dtype share_square[blockSize];
    int offset  = blockIdx.y * width;

    const Dtype* tmp_in_data = in_data + offset;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    if (index < width) {
        Dtype result = tmp_in_data[index];
        Dtype square_result = tmp_in_data[index] * tmp_in_data[index];
        for (int idx = index + stride; idx < width; idx += stride) {
            Dtype data = tmp_in_data[idx];
            result += data;
            square_result += data * data;
        }
        share_data[threadIdx.x] = result;
        share_square[threadIdx.x] = square_result;
    } else {
        share_data[threadIdx.x] = 0;
        share_square[threadIdx.x] = 0;
    }
    __syncthreads();
    int tid = threadIdx.x;
    if (blockSize >= 512) {
        if (tid < 256) {
            share_data[tid] += share_data[tid + 256];
            share_square[tid] += share_square[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            share_data[tid] += share_data[tid + 128];
            share_square[tid] += share_square[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            share_data[tid] += share_data[tid + 64];
            share_square[tid] += share_square[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile Dtype *vsum = share_data;
        volatile Dtype *vsquare = share_square;
        if (blockSize >= 64) {
            vsum[tid] += vsum[tid + 32];
            vsquare[tid] += vsquare[tid + 32];
        }
        if (blockSize >= 32) {
            vsum[tid] += vsum[tid + 16];
            vsquare[tid] += vsquare[tid + 16];
        }
        if (blockSize >= 16) {
            vsum[tid] += vsum[tid + 8];
            vsquare[tid] += vsquare[tid + 8];
        }
        if (blockSize >= 8) {
            vsum[tid] += vsum[tid + 4];
            vsquare[tid] += vsquare[tid + 4];
        }
        if (blockSize >= 4) {
            vsum[tid] += vsum[tid + 2];
            vsquare[tid] += vsquare[tid + 2];
        }
        if (blockSize >= 2) {
            vsum[tid] += vsum[tid + 1];
            vsquare[tid] += vsquare[tid + 1];
        }
    }
    if (threadIdx.x == 0) {
        atomicAdd(&sum[blockIdx.y], share_data[0]);
        atomicAdd(&square_sum[blockIdx.y], share_square[0]);
    }
}

template <typename Dtype>
__global__ void normalize(const Dtype* in_data,
                           const int height,
                           const int width,
                           const Dtype scale,
                           const Dtype* sum,
                           const Dtype* square_sum,
                           const Dtype eps,
                           Dtype* out_data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockIdx.y;
    __shared__ Dtype share_data[2];
    if (threadIdx.x == 0) {
         Dtype mean = sum[idy] * scale;
         share_data[0] = mean;
         share_data[1] = 1.0f / (sqrt(square_sum[idy] * scale - mean * mean) + eps);
    }
    __syncthreads();
    if (idx < width && idy < height) {
        int index  = idx + idy * width;
        out_data[index] = (in_data[index] - share_data[0]) * share_data[1];
    }
}

template <typename Dtype>
__global__ void normalize(const Dtype* in_data,
                           const int height,
                           const int width,
                           const Dtype scale,
                           const Dtype* sum,
                           Dtype* out_data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = blockDim.y;
    __shared__ Dtype share_data[1];
    if (threadIdx.x == 0) {
         share_data[0] = sum[idy] * scale;
    }
    __syncthreads();
    if (idx < width && idy < height) {
        int index  = idx + idy * width;
        out_data[index] = in_data[index] - share_data[0];
    }
}

template <typename Dtype, unsigned int blockSize>
__global__ void sum(const Dtype* in_data,
                         const int height,
                         const int width,
                         Dtype* sum) {
    __shared__ Dtype share_data[CUDA_NUM_THREADS];
    int offset  = blockIdx.y * width;

    const Dtype* tmp_in_data = in_data + offset;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    if (index < width) {
        Dtype result = tmp_in_data[index];
        for (int tid = index + stride; tid < width; tid += stride) {
            Dtype data = tmp_in_data[tid];
            result += data;
        }
        share_data[threadIdx.x]= result;
    } else {
        share_data[threadIdx.x] = 0;
    }
    __syncthreads();
    int tid = threadIdx.x;
    if (blockSize >= 512) {
        if (tid < 256) {
            share_data[tid] += share_data[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            share_data[tid] += share_data[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            share_data[tid] += share_data[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32) {
        if (blockSize >= 64) {
            share_data[tid] += share_data[tid + 32];
        }
        if (blockSize >= 32) {
            share_data[tid] += share_data[tid + 16];
        }
        if (blockSize >= 16) {
            share_data[tid] += share_data[tid + 8];
        }
        if (blockSize >= 8) {
            share_data[tid] += share_data[tid + 4];
        }
        if (blockSize >= 4) {
            share_data[tid] += share_data[tid + 2];
        }
        if (blockSize >= 2) {
            share_data[tid] += share_data[tid + 1];
        }
    }
    if (threadIdx.x == 0) {
        atomicAdd(&sum[blockIdx.y], share_data[0]);
    }
}

template <DataType OpDtype>
SaberStatus SaberMvn<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    MvnParam<NV>& param) {

    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    const OpDataType * in_data = (const OpDataType*)inputs[0]->data();
    OpDataType * out_data = (OpDataType*)outputs[0]->mutable_data();
    int num = inputs[0]->num() * inputs[0]->channel();
    int inner_dim = inputs[0]->height() * inputs[0]->width();
    if (param.across_channels) {
        num = inputs[0]->num();
        inner_dim *= inputs[0]->channel();
    }

    int block_num = (inner_dim + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    dim3 grid(block_num, num);
    OpDataType* mean = (OpDataType*)_mean.mutable_data();
    cudaMemsetAsync(mean, 0,  _mean.valid_size() * sizeof(OpDataType), cuda_stream);
    if (param.normalize_variance) {
        OpDataType* sd = (OpDataType*)_sd.mutable_data();
        cudaMemsetAsync(sd, 0,  _sd.valid_size() * sizeof(OpDataType), cuda_stream);
        sum_square<OpDataType, CUDA_NUM_THREADS><<<grid, CUDA_NUM_THREADS, 0, cuda_stream>>>(\
            in_data, num, inner_dim, mean, sd);
        normalize<OpDataType><<<grid, CUDA_NUM_THREADS, 0, cuda_stream>>>(\
            in_data, num, inner_dim, 1.0 / inner_dim, mean, sd, param.eps, out_data);
    } else {
        sum<OpDataType, CUDA_NUM_THREADS><<<grid, CUDA_NUM_THREADS, 0, cuda_stream>>>(\
            in_data, num, inner_dim, mean);
        normalize<OpDataType><<<grid, CUDA_NUM_THREADS, 0, cuda_stream>>>(\
            in_data, num, inner_dim, 1.0f / inner_dim, mean, out_data);
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberMvn, MvnParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberMvn, MvnParam, NV, AK_INT8);
}
}
