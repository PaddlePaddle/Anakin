#include "saber/funcs/impl/cuda/saber_layer_norm.h"
#include <math.h>

namespace anakin{

namespace saber{

template <unsigned int blockSize, typename dtype >
__global__ void reduce_mean(int total_size, int inner_size, \
        const dtype* src, dtype* mean) {

    extern __shared__ dtype sdata[];
    int tid = threadIdx.x;
    //int gridSize = blockSize * 2 * gridDim.y;
    sdata[tid] = (dtype)0;
    for (int j = tid; j < inner_size; j += blockSize) {
        sdata[tid] += src[blockIdx.x * inner_size + j];
    }
    __syncthreads();
    if (blockSize >= 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if ( tid < 32 ) {
        volatile dtype *vsum = sdata;
        if (blockSize >= 64) {
            vsum[tid] += vsum[tid + 32];
        }
        if (blockSize >= 32) {
            vsum[tid] += vsum[tid + 16];
        }
        if (blockSize >= 16) {
            vsum[tid] += vsum[tid + 8];
        }
        if (blockSize >= 8) {
            vsum[tid] += vsum[tid + 4];
        }
        if (blockSize >= 4) {
            vsum[tid] += vsum[tid + 2];
        }
        if (blockSize >= 2) {
            vsum[tid] += vsum[tid + 1];
        }
        //! write result for this block to global mem
        if (tid == 0) {
            vsum[0] = vsum[0] / inner_size;
            mean[blockIdx.x] = vsum[0];
            //printf("mean: %d, %.6f\n", blockIdx.x, vsum[0]);
        }
    }

}

template <unsigned int blockSize, typename dtype >
__global__ void reduce_std(int total_size, int inner_size, const dtype eps, \
        const dtype* src, const dtype* mean, dtype* std) {

    extern __shared__ dtype sdata[];
    int tid = threadIdx.x;
    //int gridSize = blockSize * 2 * gridDim.y;
    sdata[tid] = (dtype)0;
    for (int j = tid; j < inner_size; j += blockSize) {
        sdata[tid] += (src[blockIdx.x * inner_size + j] - mean[blockIdx.x]) *
                (src[blockIdx.x * inner_size + j] - mean[blockIdx.x]);
    }
    __syncthreads();
    if (blockSize >= 1024) {
        if (tid < 512) {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if ( tid < 32 ) {
        volatile dtype *vsum = sdata;
        if (blockSize >= 64) {
            vsum[tid] += vsum[tid + 32];
        }
        if (blockSize >= 32) {
            vsum[tid] += vsum[tid + 16];
        }
        if (blockSize >= 16) {
            vsum[tid] += vsum[tid + 8];
        }
        if (blockSize >= 8) {
            vsum[tid] += vsum[tid + 4];
        }
        if (blockSize >= 4) {
            vsum[tid] += vsum[tid + 2];
        }
        if (blockSize >= 2) {
            vsum[tid] += vsum[tid + 1];
        }
        //! write result for this block to global mem
        if (tid == 0) {
            vsum[0] = vsum[0] / inner_size;
            //printf("std pre: %d, %.6f\n", blockIdx.x, vsum[0]);
            std[blockIdx.x] = 1.f / (sqrtf(vsum[0]) + eps);
            //printf("std: %d, %.6f\n", blockIdx.x, std[blockIdx.x]);
        }
    }

}

//! normalize with scale and bias
template <typename dtype, bool flag_scale, bool flag_bias>
__global__ void normalize_with_scale_bias_kernel(
        int total_size, int inner_size, \
        const dtype* mean, const dtype* std, \
        const dtype* scale, const dtype* bias, \
        const dtype* src, dtype* dst) {

    CUDA_KERNEL_LOOP(idx, total_size) {
        int outer_idx = (idx / inner_size);
        int inner_idx = (idx % inner_size);
        if (flag_scale) {
            if (flag_bias) {
                dst[idx] = (src[idx] - mean[outer_idx]) * std[outer_idx] * scale[inner_idx] + bias[inner_idx];
            } else {
                dst[idx] = (src[idx] - mean[outer_idx]) * std[outer_idx] * scale[inner_idx];
            }
        } else {
            if (flag_bias) {
                dst[idx] = (src[idx] - mean[outer_idx]) * std[outer_idx] + bias[inner_idx];
            } else {
                dst[idx] = (src[idx] - mean[outer_idx]) * std[outer_idx];
            }
        }
    }
}


template <DataType OpDtype>
SaberStatus SaberLayerNorm<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    LayerNormParam<NV> &param) {


    cudaStream_t stream = this->_ctx->get_compute_stream();

    int total_size = inputs[0]->valid_size();

    const OpDataType* src = (const OpDataType*)inputs[0]->data();
    OpDataType* dst = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* mean_ptr = (OpDataType*)_mean.mutable_data();
    OpDataType* std_ptr = (OpDataType*)_std.mutable_data();

    const OpDataType* scale_ptr = (const OpDataType*)param.scale_weights()->data();
    const OpDataType* bias_ptr = (const OpDataType*)param.bias_weights()->data();

    const size_t share_mem_size = CUDA_NUM_THREADS * sizeof(OpDataType);

    //! get mean
    reduce_mean<CUDA_NUM_THREADS, OpDataType>\
        <<<_outer_size, CUDA_NUM_THREADS, share_mem_size, stream>>>\
        (total_size, _inner_size, src, mean_ptr);
    //! get std
    reduce_std<CUDA_NUM_THREADS, OpDataType>\
        <<<_outer_size, CUDA_NUM_THREADS, share_mem_size, stream>>>\
        (total_size, _inner_size, param.eps, src, mean_ptr, std_ptr);

    if (_flag_scale) {
        if (_flag_bias) {
            normalize_with_scale_bias_kernel<OpDataType, true, true>\
            <<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>\
            (total_size, _inner_size, mean_ptr, std_ptr, scale_ptr, bias_ptr, src, dst);
        } else {
            normalize_with_scale_bias_kernel<OpDataType, true, false>\
            <<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>\
            (total_size, _inner_size, mean_ptr, std_ptr, scale_ptr, bias_ptr, src, dst);
        }
    } else {
        if (_flag_bias) {
            normalize_with_scale_bias_kernel<OpDataType, false, true>\
            <<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>\
            (total_size, _inner_size, mean_ptr, std_ptr, scale_ptr, bias_ptr, src, dst);
        } else {
            normalize_with_scale_bias_kernel<OpDataType, false, false>\
            <<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>\
            (total_size, _inner_size, mean_ptr, std_ptr, scale_ptr, bias_ptr, src, dst);
        }
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberLayerNorm, LayerNormParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberLayerNorm, LayerNormParam, NV, AK_INT8);
} //namespace anakin

} //namespace anakin
