#include "saber/funcs/impl/cuda/saber_argmax.h"
#include "cuda_fp16.h"
#include <cfloat>

namespace anakin {
namespace saber {
template <typename Dtype, unsigned int blockSize>
__global__ void top1(const Dtype* in_data,
                     const int height,
                     const int width,
                     bool out_max_val,
                     Dtype* out_data) {
    if (blockIdx.x > height) {
       return;
    }
    __shared__ Dtype share_data[CUDA_NUM_THREADS];
    __shared__ Dtype share_index[CUDA_NUM_THREADS];
    int offset  = blockIdx.x * width;
    const Dtype* tmp_in_data = in_data + offset;
    Dtype minest = -1e32;
    int index = threadIdx.x;
    if (index < width) {
        Dtype result = tmp_in_data[index];
        Dtype idx = index;
        for (int tid = index + blockDim.x; tid < width; tid += blockDim.x) {
            if (result < tmp_in_data[tid]) {
               result = tmp_in_data[tid];
               idx = tid;
            }
        }
        share_data[index] = result;
        share_index[index] = idx;
    } else {
        share_data[index] = minest;
        share_index[index] = -1;
    }
    __syncthreads();

    if (blockSize >= 512) {
        if (index < 256) {
            int index2 = index + 256;
            if (share_data[index2] > share_data[index]) {
                share_data[index] = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (index < 128) {
            int index2 = index + 128;
            if (share_data[index2] > share_data[index]) {
                share_data[index] = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (index < 64) {
            int index2 = index + 64;
            if (share_data[index2] > share_data[index]) {
                share_data[index] = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }
        __syncthreads();
    }
    if (index < 32) {
        volatile Dtype *vmax = share_data;
        volatile Dtype *vindex = share_index;
        if (blockSize >= 64) {
            int index2 = index + 32;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 32) {
            int index2 = index + 16;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 16) {
            int index2 = index + 8;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 8) {
            int index2 = index + 4;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 4) {
            int index2 = index + 2;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 2) {
            int index2 = index + 1;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
    }

    __syncthreads();
    if (index == 0) {
       if (!out_max_val) {
           out_data[blockIdx.x] = share_index[0];
       } else {
           out_data[2 * blockIdx.x] = share_index[0];
           out_data[2 * blockIdx.x + 1] = share_data[0];
       }
    }
}

template <typename Dtype, unsigned int blockSize>
__global__ void block_top1(const Dtype* in_data,
                     const int height,
                     const int width,
                     Dtype* out_data,
                     Dtype* out_index) {
    __shared__ Dtype share_data[CUDA_NUM_THREADS];
    __shared__ Dtype share_index[CUDA_NUM_THREADS];
    int offset  = blockIdx.y * width + blockIdx.x * CUDA_NUM_THREADS;
    const Dtype* tmp_in_data = in_data + offset;
    Dtype minest = -1e32;
    int index = threadIdx.x;
    if (index + blockIdx.x * CUDA_NUM_THREADS  < width) {
        share_data[index] = tmp_in_data[index];
        share_index[index] = threadIdx.x;
    } else {
        share_data[index] = minest;
        share_index[index] = -1;
    }
    __syncthreads();

   
    if (blockSize >= 512) {
        if (index < 256) {
            int index2 = index + 256;
            if (share_data[index2] > share_data[index]) {
                share_data[index] = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (index < 128) {
            int index2 = index + 128;
            if (share_data[index2] > share_data[index]) {
                share_data[index] = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (index < 64) {
            int index2 = index + 64;
            if (share_data[index2] > share_data[index]) {
                share_data[index] = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }
        __syncthreads();
    }
    if (index < 32) {
        volatile Dtype *vmax = share_data;
        volatile Dtype *vindex = share_index;
        if (blockSize >= 64) {
            int index2 = index + 64;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 32) {
            int index2 = index + 16;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 16) {
            int index2 = index + 8;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 8) {
            int index2 = index + 4;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 4) {
            int index2 = index + 2;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 2) {
            int index2 = index + 1;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
    }

    __syncthreads();
    if (index == 0) {
       int offset = blockIdx.y * gridDim.x + blockIdx.x;
       out_data[offset] = share_data[0];
       out_index[offset] = share_index[0];
    }
}
template <typename Dtype, unsigned int blockSize>
__global__ void top1(const Dtype* in_data,
                     const Dtype* in_index,
                     const int height,
                     const int width,
                     bool out_max_val,
                     Dtype* out_data) {
    __shared__ Dtype share_data[blockSize];
    __shared__ Dtype share_index[blockSize];
    int offset  = blockIdx.x * width;
    const Dtype* tmp_in_data = in_data + offset;
    const Dtype* tmp_in_index = in_index + offset;
    Dtype minest = -1e10;
    int index = threadIdx.x;
    if (index < width) {
        Dtype result = tmp_in_data[index];
        Dtype idx = index;
        for (int tid = index + blockDim.x; tid < width; tid += blockDim.x) {
            if (result < tmp_in_data[tid]) {
               result = tmp_in_data[tid];
               idx = tid;
            }
        }
        share_data[index] = result;
        share_index[index] = idx;
    } else {
        share_data[index] = minest;
        share_index[index] = -1;
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (index < 256) {
            int index2 = index + 256;
            if (share_data[index2] > share_data[index]) {
                share_data[index] = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (index < 128) {
            int index2 = index + 128;
            if (share_data[index2] > share_data[index]) {
                share_data[index] = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (index < 64) {
            int index2 = index + 64;
            if (share_data[index2] > share_data[index]) {
                share_data[index] = share_data[index2];
                share_index[index] = share_index[index2];
            }
        }
        __syncthreads();
    }
    if (index < 32) {
        volatile Dtype *vmax = share_data;
        volatile Dtype *vindex = share_index;
        if (blockSize >= 64) {
            int index2 = index + 64;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 32) {
            int index2 = index + 16;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 16) {
            int index2 = index + 8;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 8) {
            int index2 = index + 4;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 4) {
            int index2 = index + 2;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
        if (blockSize >= 2) {
            int index2 = index + 1;
            if (vmax[index2] > vmax[index]) {
                vmax[index] = vmax[index2];
                vindex[index] = vindex[index2];
            }
        }
    }

    __syncthreads();
    if (index == 0) {
       int block_id = share_index[0];
       if (!out_max_val) {
           out_data[blockIdx.x] = block_id * CUDA_NUM_THREADS + tmp_in_index[block_id];
       } else {
           out_data[2 * blockIdx.x] = block_id * CUDA_NUM_THREADS + tmp_in_index[block_id];
           out_data[2 * blockIdx.x + 1] = share_data[0];
       }
    }
}

template <typename Dtype>
__global__ void top1_channel(const Dtype* in_data,
                     const int num,
                     const int channel,
                     const int inner_dim,
                     bool out_max_val,
                     Dtype* out_data) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id > num * inner_dim) {
        return;
    }
    int num_id = thread_id / inner_dim;
    int inner_id = thread_id % inner_dim;
    //
    const Dtype* tmp_in_data = in_data + num_id * channel * inner_dim + inner_id;
    Dtype max_data = tmp_in_data[0];
    Dtype max_id = 0;
    for (int i = 1; i < channel; i++) {
        Dtype data = tmp_in_data[i*inner_dim];
        if (max_data < data) {
            max_data = data;
            max_id = i;
        }
    }
    
    out_data[thread_id] = out_max_val ? max_data : max_id;
}
template <typename Dtype> 
__device__ void adjust_small_heap_with_index_device(Dtype* tree, Dtype *index_tree,int index,int length){
    while (2 * index + 1 < length) {
        int child_index = 2 * index + 1;
        
        if (child_index + 1 < length && tree[child_index + 1] < tree[child_index]) {
            child_index++;
        }
        if (tree[index] > tree[child_index]) {
            Dtype t = tree[index];
            tree[index] = tree[child_index];
            tree[child_index] = t;
            int t_index = index_tree[index];
            index_tree[index] = index_tree[child_index];
            index_tree[child_index] = t_index;
            index = child_index;
        } else {
            break;
        }
    }
}

template <typename Dtype> 
__device__ void adjust_small_heap_with_index_device_stride(Dtype* tree, Dtype *index_tree,int index,int length, int stride){
    while (2 * index + 1 < length) {
        int child_index = 2 * index + 1;
        int off_0 = child_index * stride;
        int off_1 = (child_index + 1) * stride;
        if (child_index + 1 < length && tree[off_1] < tree[off_0]) {
            child_index++;
        }
        int child_off = child_index * stride;
        int cur_off = index * stride;
        if (tree[cur_off] > tree[child_off]) {
            Dtype t = tree[cur_off];
            tree[cur_off] = tree[child_off];
            tree[child_off] = t;
            int t_index = index_tree[cur_off];
            index_tree[cur_off] = index_tree[child_off];
            index_tree[child_off] = t_index;
            index = child_index;
        } else {
            break;
        }
    }
}


template <typename Dtype>
__global__ void topk_channel(const Dtype* in_data,
                     const int num,
                     const int channel,
                     const int inner_dim,
                     const int top_k,
                     bool out_max_val,
                     Dtype* out_data) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id > num * inner_dim) {
        return;
    }
    int num_id = thread_id / inner_dim;
    int inner_id = thread_id % inner_dim;
    //
    const Dtype* tmp_in_data = in_data + num_id * channel * inner_dim + inner_id;
    extern  __shared__ Dtype trees[];
    Dtype* small_heap_tree = trees + threadIdx.x * top_k;
    Dtype* tree_index = trees + threadIdx.x * top_k + blockDim.x * top_k;
    for (int i = 0; i < top_k; i++) {
        small_heap_tree[i] = -FLT_MAX;
        tree_index[i] = -1;
    }
    for (int i = 0; i < channel; i++) {
        Dtype data = tmp_in_data[i*inner_dim];
        if (data > small_heap_tree[0]) {
             small_heap_tree[0] = data;
             tree_index[0] = i;
             adjust_small_heap_with_index_device(small_heap_tree, tree_index, 0, top_k);
        }  
    }
    Dtype* out =  out_data + num_id * top_k * inner_dim + inner_id;
    for (int i = top_k - 1; i >= 0; i--) {
        out[i * inner_dim] = out_max_val ? small_heap_tree[0] : tree_index[0];
        small_heap_tree[0] = FLT_MAX;
        tree_index[0] = -1;
        adjust_small_heap_with_index_device(small_heap_tree, tree_index, 0, top_k);
    }
}

/*trees size is k * blockDim.x*/
template <typename Dtype, int blockSize>
__global__ void topk_heap_shared(Dtype *out_data, int n, int inner_dim, const int top_k, const bool out_max_val, const Dtype *in_data){
    extern  __shared__ Dtype trees[];
    const int block_id = blockIdx.x;
    const int tid = threadIdx.x;
    Dtype *cur_tree = trees + tid * top_k;
    Dtype *cur_tree_index = cur_tree + top_k * blockDim.x;
    for (int i = 0; i < top_k; i++){
        cur_tree[i] = -FLT_MAX;
        cur_tree_index[i] = -1;
    }
    
/*build small heap for every thread in one picture*/
    const Dtype* in = in_data + block_id * inner_dim;
    for (int i = tid; i < inner_dim; i += blockDim.x){
        if (in[i] > cur_tree[0]) {
            cur_tree[0] = in[i];
            cur_tree_index[0] = i;
            adjust_small_heap_with_index_device(cur_tree, cur_tree_index, 0, top_k);
        }
    }
    __syncthreads();
    if (blockSize >= 512) {
        if (tid < 256) {
             Dtype* next_tree = cur_tree + 256 * top_k;
             Dtype* next_tree_index = cur_tree_index + 256 * top_k;
            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0] = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, 0, top_k);
                }
            }
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
             Dtype* next_tree = cur_tree + 128 * top_k;
             Dtype* next_tree_index = cur_tree_index + 128 * top_k;
            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0] = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, 0, top_k);
                }
            }
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
             Dtype* next_tree = cur_tree + 64 * top_k;
             Dtype* next_tree_index = cur_tree_index + 64 * top_k;
            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0] = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, 0, top_k);
                }
            }
        }
        __syncthreads();
    }
    if (blockSize >= 64) {
        if (tid < 32) {
             Dtype* next_tree = cur_tree + 32 * top_k;
             Dtype* next_tree_index = cur_tree_index + 32 * top_k;
            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0] = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, 0, top_k);
                }
            }
        }
        __syncthreads();
    }
    if (blockSize >= 32) {
        if (tid < 16) {
             Dtype* next_tree = cur_tree + 16 * top_k;
             Dtype* next_tree_index = cur_tree_index + 16 * top_k;
            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0] = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, 0, top_k);
                }
            }
        }
        __syncthreads();
    }
    if (blockSize >= 16) {
        if (tid < 8) {
             Dtype* next_tree = cur_tree + 8 * top_k;
             Dtype* next_tree_index = cur_tree_index + 8 * top_k;
            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0] = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, 0, top_k);
                }
            }
        }
        __syncthreads();
    }
    if (blockSize >= 8) {
        if (tid < 4) {
             Dtype* next_tree = cur_tree + 4 * top_k;
             Dtype* next_tree_index = cur_tree_index + 4 * top_k;
            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0] = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, 0, top_k);
                }
            }
        }
        __syncthreads();
    }
    if (blockSize >= 4) {
        if (tid < 2) {
             Dtype* next_tree = cur_tree + 2 * top_k;
             Dtype* next_tree_index = cur_tree_index + 2 * top_k;
            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0] = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, 0, top_k);
                }
            }
        }
        __syncthreads();
    }
    if (blockSize >= 2) {
        if (tid < 1) {
             Dtype* next_tree = cur_tree + 1 * top_k;
             Dtype* next_tree_index = cur_tree_index + 1 * top_k;
            for (int i = 0; i < top_k; i++) {
                if (next_tree[i] > cur_tree[0]) {
                    cur_tree[0] = next_tree[i];
                    cur_tree_index[0] = next_tree_index[i];
                    adjust_small_heap_with_index_device(cur_tree, cur_tree_index, 0, top_k);
                }
            }
        }
        __syncthreads();
    }
   
    if (tid == 0) {
        int stride = out_max_val ? block_id * top_k * 2 : block_id * top_k;
        Dtype* out =  out_data + stride;
        for (int i = top_k - 1; i >= 0; i--) {
            if (!out_max_val) {
                out[i] = cur_tree_index[0];
            } else {
                out[i + top_k] = cur_tree[0];
                out[i] = cur_tree_index[0];
            }
            cur_tree[0] = FLT_MAX;
            cur_tree_index[0] = -1;
            adjust_small_heap_with_index_device(cur_tree, cur_tree_index, 0, top_k);
        }
    }
}
    
template <>
SaberStatus SaberArgmax<NV, AK_FLOAT>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        ArgmaxParam<NV>& param) {    
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    outputs[0]->set_seq_offset(inputs[0]->get_seq_offset());
    const OpDataType * in_data = (const OpDataType*)inputs[0]->data();
    OpDataType * out_data = (OpDataType*)outputs[0]->mutable_data();
    int outer_dim = inputs[0]->count_valid(0, param.axis);
    if (param.has_axis) {
        int count = inputs[0]->count_valid(0, inputs[0]->dims());
        int dim = inputs[0]->shape()[param.axis];
        int inner_dim = inputs[0]->count_valid(param.axis + 1, inputs[0]->dims());
        int total_threads = count / dim;
        if (param.top_k == 1) {
            top1_channel<OpDataType><<<CUDA_GET_BLOCKS(total_threads), CUDA_NUM_THREADS, 0, cuda_stream>>>(in_data, outer_dim, dim, inner_dim, param.out_max_val, out_data);
        } else {
            topk_channel<OpDataType><<<CUDA_GET_BLOCKS(total_threads), CUDA_NUM_THREADS, 2 * sizeof(OpDataType) * CUDA_NUM_THREADS * param.top_k, cuda_stream>>>(in_data, outer_dim, dim, inner_dim, param.top_k, param.out_max_val, out_data);
        }
    } else {
        int inner_dim = inputs[0]->count_valid(1, inputs[0]->dims());
        int outer_dim = inputs[0]->num();
        if (param.top_k == 1) {
            if (inner_dim / CUDA_NUM_THREADS < 10) {
                int block_size = pow(2, ceil(log(inner_dim) / log(2)));
                block_size  = block_size > CUDA_NUM_THREADS ? CUDA_NUM_THREADS : block_size;
                top1<OpDataType, CUDA_NUM_THREADS><<<outer_dim, CUDA_NUM_THREADS, 0, cuda_stream>>>(in_data, outer_dim, inner_dim, param.out_max_val, out_data);
            } else {
                int block_num = CUDA_GET_BLOCKS(inner_dim);
                dim3 grid(block_num, outer_dim);
                block_top1<OpDataType, CUDA_NUM_THREADS><<<grid, CUDA_NUM_THREADS, 0, cuda_stream>>>(in_data, outer_dim, inner_dim, (OpDataType*)_block_max_value.mutable_data(), (OpDataType*)_block_max_index.mutable_data());
                top1<OpDataType, CUDA_NUM_THREADS><<<outer_dim, CUDA_NUM_THREADS, 0, cuda_stream>>>((OpDataType*)_block_max_value.data(), (OpDataType*)_block_max_index.data(), outer_dim, block_num, param.out_max_val, out_data);
            }
        } else {
             topk_heap_shared<OpDataType, CUDA_NUM_THREADS><<<outer_dim,  CUDA_NUM_THREADS, sizeof(OpDataType) * CUDA_NUM_THREADS * param.top_k * 2, cuda_stream>>>(out_data, outer_dim, inner_dim, param.top_k, param.out_max_val,  in_data);
        }
    }

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberArgmax, ArgmaxParam, NV, AK_INT8);
DEFINE_OP_TEMPLATE(SaberArgmax, ArgmaxParam, NV, AK_HALF);
}
}
