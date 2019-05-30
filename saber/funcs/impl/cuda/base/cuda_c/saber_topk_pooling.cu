#include "saber/funcs/impl/cuda/saber_topk_pooling.h"
#include "cuda_fp16.h"

namespace anakin{
namespace saber{

template <typename Dtype>
__global__ void top_k_pooling_batch_kernel_reduction(Dtype *output_data,       // out
                                                     const Dtype *input,             // in 0
                                                     const int *height_offset, // in 1 offset
                                                     const int *width_offset,  // in 0 offset
                                                     const int batch_size,     // batch size in0.shape[0]
                                                     const int channel_num,    // feat_map_num
                                                     const int height_stride,  // in0.shape[2]
                                                     const int width_stride,   // in0.shape[3]
                                                     const int k)              // topk
{
    const Dtype * input_start = input + (blockIdx.x * channel_num + blockIdx.y)* height_stride * width_stride; 
    Dtype * output_start = output_data + (blockIdx.x * channel_num + blockIdx.y)* k;

    int width = width_offset[blockIdx.x + 1] - width_offset[blockIdx.x];
    int height = height_offset[blockIdx.x + 1] - height_offset[blockIdx.x];
    int real_k = k < height*width ? k : height*width;

    extern __shared__ Dtype smem[];

    Dtype min_val = -100000.0f;
    for(int j = threadIdx.x; j < height*width; j += blockDim.x) {
        int index_tmp = (j / width) * width_stride + j%width;
        smem[j] = input_start[index_tmp];
    }
    __syncthreads();
    // get max val
    int t = 0;
    for(; t < real_k; t++) {
        // reduction
        for(int gap = height*width; gap > 1;) { 
            if(threadIdx.x == 0) { // edge cond
                if((gap % 2) != 0) {
                    Dtype value_first = smem[0];
                    Dtype value_gap =  smem[gap-1];
                    if(value_first < value_gap) {
                        smem[0] = value_gap;
                        smem[gap-1] = value_first;
                    }
                }
            }
            gap>>=1;
            for(int j=threadIdx.x; j < gap; j+=blockDim.x) {
                Dtype value_first =  smem[j];
                Dtype value_gap =  smem[j+gap];
                if(value_first < value_gap) {
                    smem[j] = value_gap;
                    smem[j+gap] = value_first;
                }
            }
            __syncthreads();
        }
        if(threadIdx.x == 0) {
            output_start[t] = smem[0];
            smem[0] = min_val;
        }
    }
    for(int i=threadIdx.x; i < (k-t); i += blockDim.x) {
        output_start[ t + i ] = 0.0f;
    }
}


template <DataType OpDtype>
SaberStatus SaberTopKPooling<NV, OpDtype>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        TopKPoolingParam<NV>& param) {
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();
    Shape in_shape = inputs[0]->valid_shape();
    Shape out_shape = outputs[0]->valid_shape();
    CHECK(inputs[0]->get_seq_offset().size() > 0 && inputs[0]->get_seq_offset()[0].size() > 0) 
            << "input_0 sequence offset is not valid";
    CHECK(inputs[1]->get_seq_offset().size() > 0 && inputs[1]->get_seq_offset()[0].size() > 0)
            << "input_1 sequence offset is not valid";
    int width_offset_len = inputs[0]->get_seq_offset()[0].size();
    Shape width_offset_shape(std::vector<int>{width_offset_len, 1, 1, 1});
    _width_offset.reshape(width_offset_shape);
    cudaMemcpyAsync(_width_offset.mutable_data(), &(inputs[0]->get_seq_offset()[0][0]),
            sizeof(int) * width_offset_len,
            cudaMemcpyHostToDevice, cuda_stream);
    int height_offset_len = inputs[1]->get_seq_offset()[0].size();
    Shape height_offset_shape(std::vector<int>{height_offset_len, 1, 1, 1});
    _height_offset.reshape(height_offset_shape);
    cudaMemcpyAsync(_height_offset.mutable_data(), &(inputs[1]->get_seq_offset()[0][0]),
            sizeof(int) * height_offset_len, 
            cudaMemcpyHostToDevice, cuda_stream);

    const OpDataType *in_data = (const OpDataType*)inputs[0]->data();
    OpDataType *out_data = (OpDataType*)outputs[0]->mutable_data();

    const int count = inputs[0]->valid_size();
    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();
    dim3 blocks(num, channel);
    dim3 threads(32, 1);
    const int* height_offset = (const int*)(_height_offset.data());
    const int* width_offset = (const int*)(_width_offset.data());
    
    int feat_map_size = height *  width;
    top_k_pooling_batch_kernel_reduction<OpDataType><<<blocks, threads, feat_map_size*sizeof(OpDataType), cuda_stream>>>(out_data,
                                                                                                          in_data,
                                                                                                          height_offset,
                                                                                                          width_offset,
                                                                                                          num,
                                                                                                          channel,
                                                                                                          height,
                                                                                                          width,
                                                                                                          param.top_k);

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberTopKPooling<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberTopKPooling, TopKPoolingParam, NV, AK_INT8);
DEFINE_OP_TEMPLATE(SaberTopKPooling, TopKPoolingParam, NV, AK_HALF);
}
}
