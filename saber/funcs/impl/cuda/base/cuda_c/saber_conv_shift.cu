#include "saber/funcs/impl/cuda/saber_conv_shift.h"
#include "cuda_fp16.h"

namespace anakin {
namespace saber {

template <typename Dtype>
__global__ void ker_conv_shift_fwd(Dtype * out_data, \
                    const Dtype* x_data,
                    const Dtype* y_data,
                    const int num,
                    const int x_w,
                    const int y_w)
{
    int width_id  = blockIdx.x * blockDim.x + threadIdx.x;
    int end_id = (blockIdx.x + 1) * blockDim.x >= x_w ? x_w : (blockIdx.x + 1) * blockDim.x;
    int num_id = blockIdx.y;
    auto x_tmp = x_data + num_id * x_w;
    auto y_tmp = y_data + num_id * y_w;
    int half_y_w = y_w /2;
    /*share mem*/
    extern __shared__ Dtype mem[];
    Dtype *sy = &mem[blockDim.x + y_w];
    for (int i = width_id; i < end_id + y_w; i += blockDim.x) {
        int id = (i - half_y_w + x_w) % x_w;
        mem[i] = x_tmp[id];
    }
    for (int i = width_id; i < y_w; i += blockDim.x) {
        sy[i] = y_tmp[i];
    }
    __syncthreads();
    //if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x== 0 && blockIdx.y == 1) {
    //    printf("blockIdx.x:%d, blockIdx.y:%d", gridDim.x, gridDim.y);
    //    for (int i = 0; i< x_w + y_w; i++) {
    //        printf("b_idy: %d, b_idx:%d, i: %d, %f\n", blockIdx.y, blockIdx.x, i, mem[i]);
    //    }
    //    for (int i = 0; i< y_w; i++) {
    //        printf("b_idy: %d, b_idx:%d, i: %d, %f\n", blockIdx.y, blockIdx.x, i, sy[i]);
    //    }
    //}
    if (blockIdx.y >= num || width_id >= x_w) {
       return;
    }
    Dtype res = 0.f;
    for (int i = 0; i < y_w; i++) {
       res += mem[i + width_id] * sy[i];
    }
    out_data[num_id * x_w + width_id] = res;
}

template <DataType OpDtype>
SaberStatus SaberConvShift<NV, OpDtype>::dispatch(\
    const std::vector<Tensor<NV> *>& inputs, \
    std::vector<Tensor<NV> *>& outputs, \
    ConvShiftParam<NV>& param) {
    int out_n = outputs[0]->num();
    int x_n = inputs[0]->num();
    int y_n = inputs[1]->num();
    int out_width = outputs[0]->count_valid(1, outputs[0]->dims());
    int x_width = inputs[0]->count_valid(1, inputs[0]->dims());
    int y_width = inputs[1]->count_valid(1, inputs[1]->dims());
    CHECK_EQ(x_n, y_n) << "conv shift the two inputs batch size are not equal";
    CHECK_EQ(x_n, out_n) << "conv shift input batch size and out batch size are not equal";
    CHECK_EQ(x_width , out_width) << "input and output width are not equal";

    const OpDataType* x_data = (const OpDataType*)inputs[0]->data();
    const OpDataType* y_data = (const OpDataType*)inputs[1]->data();
    OpDataType* out_data = (OpDataType*)outputs[0]->mutable_data();
    cudaStream_t cuda_stream = this->_ctx->get_compute_stream();

    int count = outputs[0]->valid_size();
    int num_x_blocks = (x_width  + (CUDA_NUM_THREADS - 1)) / CUDA_NUM_THREADS;
    dim3 grid_dim(num_x_blocks, out_n);
    int mem_per_block = (CUDA_NUM_THREADS + 2 * y_width) * sizeof(OpDataType);

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        ker_conv_shift_fwd<OpDataType>\
                 <<<grid_dim, CUDA_NUM_THREADS, mem_per_block, cuda_stream>>>(\
                 out_data, x_data, y_data, out_n, x_width, y_width);
    }

    return SaberSuccess;
}

DEFINE_OP_TEMPLATE(SaberConvShift, ConvShiftParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberConvShift, ConvShiftParam, NV, AK_INT8);
}
}
