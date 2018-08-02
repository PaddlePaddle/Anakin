#include "cuda_utils.h"
namespace anakin {

namespace saber {


template<typename Dtype>
__global__ void trans_map2in(Dtype* output, const Dtype* input, const int* map, int count,
                             int lastdim) {
    CUDA_KERNEL_LE(tid, count) {
        int seq = tid / lastdim;
        output[tid] = input[map[seq] * lastdim + tid % lastdim];
        //        printf("in %d = %f\n",tid,output[tid]);
    }
}

template<typename Dtype>
__global__ void trans_map2out(Dtype* output, const Dtype* input, const int* map, int count,
                              int lastdim) {
    CUDA_KERNEL_LE(tid, count) {
        int seq = tid / lastdim;
        output[map[seq] * lastdim + tid % lastdim] = input[tid];
        //        printf("out %d = %f\n",map[seq]*lastdim + tid % lastdim,output[map[seq]*lastdim + tid % lastdim]);
    }
}

template<typename Dtype>
void trans_map2out_cfunc(const Dtype* input, Dtype* output, int word_size, int seq_sum,
                         cudaStream_t stream,
                         int* dev_map_vec) {
    int count = seq_sum * word_size;
    int block_dim = count;
    int grid_dim = 1;

    if (count > 1024) {
        block_dim = 256;
        grid_dim = (count + block_dim - 1) / block_dim;
    }

    trans_map2out << < grid_dim, block_dim, 0, stream >> > (output, input, dev_map_vec,
                  count, word_size);

    //    cudaDeviceSynchronize();
}

template<typename Dtype>
void trans_map2in_cfunc(const Dtype* input, Dtype* output, int hidden_size, int seq_sum,
                        cudaStream_t stream,
                        int* dev_map_vec) {
    int count = seq_sum * hidden_size;
    int block_dim = count;
    int grid_dim = 1;
    if (count > 1024) {
        block_dim = 256;
        grid_dim = (count + block_dim - 1) / block_dim;
    }

    trans_map2in << < grid_dim, block_dim, 0, stream >> > (output, input, dev_map_vec,
                 count, hidden_size);

}
template void trans_map2in_cfunc<float>(const float* input, float* output, int hidden_size, int seq_sum,
                                 cudaStream_t stream,
                                 int* dev_map_vec);
template void trans_map2out_cfunc<float>(const float* input, float* output, int word_size, int seq_sum,
                         cudaStream_t stream,
                         int* dev_map_vec);
}
}