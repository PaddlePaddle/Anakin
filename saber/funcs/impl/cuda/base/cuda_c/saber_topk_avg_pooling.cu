#include "saber/funcs/impl/cuda/saber_topk_avg_pooling.h"
#include "cuda_fp16.h"

namespace anakin{
namespace saber{

template<typename Dtype>
__global__ void topk_avg_pooling_kernel_by_row_improve(Dtype *output_data,
                                                       const Dtype *input,
                                                       const int* gpu_input_offset_l,
                                                       const int* gpu_input_offset_r,
                                                       const int row_max,
                                                       const int col_max,
                                                       const int topk_size,
                                                       const int *topks,
                                                       const int feat_map_num) {
    int row = gpu_input_offset_l[blockIdx.x+1] - gpu_input_offset_l[blockIdx.x]; // 8
    int col = gpu_input_offset_r[blockIdx.x+1] - gpu_input_offset_r[blockIdx.x]; // 30
    
    int max_k = topks[topk_size-1];
    max_k = max_k < col ? max_k : col;


    extern __shared__ Dtype smem[]; // H*W

    const Dtype *fm_row_in_data = input + \
                                    blockIdx.x * row_max * feat_map_num * col_max + \
                                    blockIdx.y * row_max * col_max;

    for(int i = threadIdx.x; i < row*col_max; i+=blockDim.x) {
        smem[i] = fm_row_in_data[i];
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < row; idx+=blockDim.x) {
        Dtype *fm_row_out_data = output_data + \
                                    (gpu_input_offset_l[blockIdx.x] + idx) * feat_map_num * topk_size + \
                                    blockIdx.y * topk_size;

        Dtype* smem_start_col = smem + idx * col_max;

        int counter = max_k;//topk_size;
        Dtype last_max_val = -20000.0;
        while(counter) {
            Dtype max_val = -10000.0;
            int max_pos = 0;//-1;
            int m = 0;
            for(; m< col; m++) {
                Dtype cur_data = smem_start_col[m];
                if (cur_data > max_val) {
                    max_val = cur_data;
                    max_pos = m;
                    last_max_val = max_val;
                }
            }
            if(max_val < -9999.0) { // == -10000.0
                 max_val = last_max_val;
            }
            smem_start_col[max_pos] = -10000000.0; 

            int i =  max_k - counter;
            for (int c = 0; c < topk_size; c++) {
                if(i <= topks[c]-1) {
                    fm_row_out_data[c] += max_val;
                }
            }
            counter--;
        }
        __syncthreads();
        // compute avg
        for (int i=0; i < topk_size; i++) {
            fm_row_out_data[i] = fm_row_out_data[i] / topks[i];
        }
    }
}


template <DataType OpDtype>
SaberStatus SaberTopKAvgPooling<NV, OpDtype>::dispatch( \
        const std::vector<Tensor<NV>*>& inputs,
        std::vector<Tensor<NV>*>& outputs,
        TopKAvgPoolingParam<NV>& param) {
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
    cudaMemsetAsync(out_data, 0, sizeof(OpDataType) * outputs[0]->valid_size(), cuda_stream);

    int num = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height = inputs[0]->height();
    int width = inputs[0]->width();

    const int* height_offset = (const int*)(_height_offset.data());
    const int* width_offset = (const int*)(_width_offset.data());
    
    int feat_map_size = height *  width;

    dim3 blocks(num, channel);
    dim3 threads(32, 1);
    if (param.is_pooling_by_row) {
        topk_avg_pooling_kernel_by_row_improve<OpDataType><<<blocks, threads, feat_map_size*sizeof(OpDataType), cuda_stream>>>(out_data,
                                                                                                                 in_data,
                                                                                                                 height_offset,
                                                                                                                 width_offset,
                                                                                                                 height,
                                                                                                                 width,
                                                                                                                 param.top_ks.size(),
                                                                                                                 (const int*)(_top_ks.data()),
                                                                                                                 param.feat_map_num);
    } else {
        // NOT IMPL
        LOG(FATAL) << " TOP_K AVG POOLING BY COL NOT IMPL YET! ";
    }

    CUDA_POST_KERNEL_CHECK;
    return SaberSuccess;
}

template class SaberTopKAvgPooling<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberTopKAvgPooling, TopKAvgPoolingParam, NV, AK_INT8);
DEFINE_OP_TEMPLATE(SaberTopKAvgPooling, TopKAvgPoolingParam, NV, AK_HALF);
}
}
