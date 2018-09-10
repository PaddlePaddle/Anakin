#include "saber/funcs/impl/cuda/saber_normalize.h"
#include <math.h>

namespace anakin{

namespace saber{

template <typename Dtype, bool has_scale, bool shared>
__global__ void normalize_kernel_no_across_spatial(const int size_in_channel, const int n,\
const int channels,const Dtype* scale, const Dtype* bottom_data, Dtype* top_data, const float eps, const int p){
    CUDA_KERNEL_LOOP(index, size_in_channel*n){
        float sqr_sum = 0.f;
        int num_index=index/size_in_channel;
        int index_in_channel=index%size_in_channel;
        int data_index=num_index*channels*size_in_channel+index_in_channel;
        for (int i = 0; i < channels; ++i) {
            if (p == 1) {
                sqr_sum += fabsf(bottom_data[data_index + i * size_in_channel]);
            } else {
                sqr_sum += bottom_data[data_index + i * size_in_channel] * \
                    bottom_data[data_index + i * size_in_channel];
            }
        }
        float norm;
        if (p == 1) {
            norm = 1.f / (sqr_sum + eps);
        } else {
            norm = 1.f / sqrtf(sqr_sum+ eps);
        }

        for (int i = 0; i < channels; ++i) {
            if (has_scale) {
                if (shared) {
                    top_data[data_index + i * size_in_channel] = \
                        bottom_data[data_index + i * size_in_channel] * scale[0]*norm;
                } else {
                    top_data[data_index + i * size_in_channel] = \
                        bottom_data[data_index + i * size_in_channel] * scale[i]*norm;
                }
            } else {
                top_data[data_index + i * size_in_channel] = \
                        bottom_data[data_index + i * size_in_channel] * norm;
            }
        }

    }
}

template <typename dtype>
__global__ void gpu_pow_reverse(int n, \
    const dtype* src, dtype* dst, dtype alpha, dtype eps) {
    CUDA_KERNEL_LOOP(idx, n) {
        dst[idx] = 1 / (pow(src[idx], alpha) + eps);
    }
}

template <unsigned int blockSize, typename dtype >
__global__ void reduce_add_atomic(int total_size, int p, int inner_size, \
        const dtype* src, dtype* dst) {
    extern __shared__ dtype sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.y * inner_size + blockIdx.x * blockSize + tid;
    int idx_limit = (blockIdx.y + 1) * inner_size;
    //int gridSize = blockSize * 2 * gridDim.y;
#if 0
    dtype sum = 0;
    while (i < total_size) {
        sum += src[i] * src[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (i + blockSize < total_size) {
            sum += src[i + blockSize] * src[i + blockSize];
        }
        i += gridSize;
    }
#endif
    //! L1 norm
    if (p == 1) {
        sdata[tid] = i < idx_limit ? fabsf(src[i]) : 0;
    } else {
        //! L2 norm
        sdata[tid] = i < idx_limit ? src[i] * src[i] : 0;
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
#if 0
    dtype sum = sdata[tid];
    __syncthreads();
    if ( tid < 32 ) {
        //! Fetch final intermediate sum from 2nd warp
        if (blockSize >= 64) {
            sum += sdata[tid + 32];
        }
        //! Reduce final warp using shuffle
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down(sum, offset);
        }
        //! write result for this block to global mem
        if (tid == 0) {
            atomicAdd(dst + blockIdx.y, sum);
        }
    }
#endif
#if 1
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
            atomicAdd(dst + blockIdx.y, *vsum);
        }
    }
#endif
};

//! normalize with scale
template <typename dtype, bool channel_shared>
__global__ void normalize_with_scale_kernel(
        int n, int inner_size, int channel_stride, int channel_size, \
        const dtype* norm, const dtype* scale, \
        const dtype* src, dtype* dst) {
    CUDA_KERNEL_LOOP(idx, n) {
        int outer_idx = idx / inner_size;
        if (channel_shared) {
            dst[idx] = src[idx] * norm[outer_idx] * scale[0];
        } else {
            int channel = (idx / channel_stride) % channel_size;
            dst[idx] = src[idx] * norm[outer_idx] * scale[channel];
            //printf("channel: %d, scale: %.2f\n", channel, scale[channel]);
        }
    }
}

//! normalize without scale
template <typename dtype>
__global__ void normalize_kernel(int n, int inner_size, \
    const dtype* norm, const dtype* src, dtype* dst) {

    CUDA_KERNEL_LOOP(idx, n) {
        int outer_idx = idx / inner_size;
        dst[idx] = src[idx] * norm[outer_idx];
    }
}

//! normalize with scale
template <typename dtype, bool channel_shared>
__global__ void normalize_with_scale_compute_norm_kernel(
        int n, int inner_size, int channel_stride, int channel_size, \
        const dtype* norm, const dtype* scale, \
        const dtype* src, dtype* dst) {
    __shared__ dtype sdata[1];
    int tid = threadIdx.x;
    int i = blockIdx.y * inner_size + blockIdx.x * blockDim.x + tid;
    int idx_limit = (blockIdx.y + 1) * inner_size;
    if(tid == 0) {
        sdata[0] = 1 / (sqrtf(norm[blockIdx.y] / inner_size) + 1e-6f);
    }
    __syncthreads();
    if (channel_shared) {
        if (i < idx_limit) {
            dst[i] = src[i] * sdata[0] * scale[0];
        }
    } else {
        if (i < idx_limit) {
            int channel = (i / channel_stride) % channel_size;
            dst[i] = src[i] * sdata[0] * scale[channel];
        }
    }
}

//! normalize without scale
template <typename dtype>
__global__ void normalize_compute_norm_kernel(int n, int inner_size, \
    const dtype* norm, const dtype* src, dtype* dst) {

    __shared__ dtype sdata[1];
    int tid = threadIdx.x;
    int i = blockIdx.y * inner_size + blockIdx.x * blockDim.x + tid;
    int idx_limit = (blockIdx.y + 1) * inner_size;
    if(tid == 0) {
        sdata[0] = 1 / (sqrtf(norm[blockIdx.y] / inner_size) + 1e-6f);
    }
    __syncthreads();
    if (i < idx_limit) {
        dst[i] = src[i] * sdata[0];
    }
}

template <>
SaberStatus SaberNormalize<NV, AK_FLOAT>::dispatch(\
    const std::vector<DataTensor_in*>& inputs, \
    std::vector<DataTensor_out*>& outputs, \
    NormalizeParam<NV> &param) {
    cudaStream_t stream = this->_ctx->get_compute_stream();
    const float* src = static_cast<float*>(inputs[0]->data());
    float* dst = static_cast<float*>(outputs[0]->mutable_data());
    if (!param.across_spatial) {
        int num=inputs[0]->num();
        int size_in_channel = inputs[0]->width() * inputs[0]->height();
        int thread_num=size_in_channel*num;
        int channel = inputs[0]->channel();
        if (param.has_scale) {
            if (param.channel_shared) {
                normalize_kernel_no_across_spatial<float, true, true> \
                    <<<CUDA_GET_BLOCKS(thread_num), CUDA_NUM_THREADS, 0, stream>>>\
                    (size_in_channel,num, channel, static_cast<float*>(param.scale->data()), src, dst, param.eps, param.p);
            } else {
                normalize_kernel_no_across_spatial<float, true, false> \
                    <<<CUDA_GET_BLOCKS(thread_num), CUDA_NUM_THREADS, 0, stream>>>\
                    (size_in_channel,num, channel, static_cast<float*>(param.scale->data()), src, dst, param.eps, param.p);
            }
        } else {
            normalize_kernel_no_across_spatial<float, false, false> \
                <<<CUDA_GET_BLOCKS(thread_num), CUDA_NUM_THREADS, 0, stream>>>\
                (size_in_channel, num,channel, nullptr, src, dst, param.eps, param.p);
        }
    } else {

        float* norm_reduce_ptr = static_cast<float*>(_norm_reduce.mutable_data());
        const size_t share_mem_size = CUDA_NUM_THREADS * sizeof(float);
        //! compute sum across C * H * W or H * W
        int blockx = CUDA_NUM_THREADS;
        int gridy = _norm_size;
        //! each thread compute one value
        int gridx = (_compute_size + blockx - 1) / blockx;
        dim3 grid(gridx, gridy);
        //cudaMemsetAsync(norm_reduce_ptr, 0, sizeof(float) * _norm_size, stream);
        cudaMemset(norm_reduce_ptr, 0, sizeof(float) * _norm_size);
        reduce_add_atomic<CUDA_NUM_THREADS, float>\
        <<<grid, CUDA_NUM_THREADS, share_mem_size, stream>>>\
        (_size, param.p, _compute_size, src, norm_reduce_ptr);
#if 0 //compute norm in one kernel
        if (param.has_scale) {
        //! scale is shared across channel
        if (param.channel_shared) {
            normalize_with_scale_compute_norm_kernel<float, true>\
                <<<grid, CUDA_NUM_THREADS, 0, stream>>>\
                (_size, _compute_size, _channel_stride, _channels, _norm_reduce.data(), \
                param.scale->data(), inputs[0]->data(), outputs[0]->mutable_data());
        } else {//! scale is diffs across channel
            normalize_with_scale_compute_norm_kernel<float, false>\
                <<<grid, CUDA_NUM_THREADS, 0, stream>>>\
                (_size, _compute_size, _channel_stride, _channels, _norm_reduce.data(), \
                param.scale->data(), inputs[0]->data(), outputs[0]->mutable_data());
        }
    } else { //! without scale
        normalize_compute_norm_kernel<float>\
            <<<grid, CUDA_NUM_THREADS, 0, stream>>>\
                (_size, _compute_size, _norm_reduce.data(), \
                inputs[0]->data(), outputs[0]->mutable_data());
    }
    cudaDeviceSynchronize();
#else
        //compute norm and result individually
        //! compute square root
        const float eps = param.eps;
        float pw = 0.5f;
        if (param.p == 1) {
            pw = 1.f;
        }
        gpu_pow_reverse<float><<<CUDA_GET_BLOCKS(_norm_size), CUDA_NUM_THREADS, 0, stream>>>\
            (_norm_size, static_cast<float*>(_norm_reduce.data()), static_cast<float*>(_norm_reduce.mutable_data()), pw, eps);

        //! compute output with scale
        if (param.has_scale) {
            //! scale is shared across channel
            if (param.channel_shared) {
                normalize_with_scale_kernel<float, true>\
                <<<CUDA_GET_BLOCKS(_size), CUDA_NUM_THREADS, 0, stream>>>\
                (_size, _compute_size, _channel_stride, _channels, static_cast<float*>(_norm_reduce.data()), \
                static_cast<float*>(param.scale->data()), static_cast<float*>(inputs[0]->data()), static_cast<float*>(outputs[0]->mutable_data()));
            } else {//! scale is diffs across channel
                normalize_with_scale_kernel<float, false>\
                <<<CUDA_GET_BLOCKS(_size), CUDA_NUM_THREADS, 0, stream>>>\
                (_size, _compute_size, _channel_stride, _channels, static_cast<float*>(_norm_reduce.data()), \
                static_cast<float*>(param.scale->data()), static_cast<float*>(inputs[0]->data()), static_cast<float*>(outputs[0]->mutable_data()));
            }
        } else { //! without scale
            normalize_kernel<float><<<CUDA_GET_BLOCKS(_size), CUDA_NUM_THREADS, 0, stream>>>\
                (_size, _compute_size, static_cast<float*>(_norm_reduce.data()), \
                static_cast<float*>(inputs[0]->data()), static_cast<float*>(outputs[0]->mutable_data()));
        }
#endif
    }
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberNormalize, NormalizeParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberNormalize, NormalizeParam, NV, AK_INT8);
} //namespace anakin

} //namespace anakin
