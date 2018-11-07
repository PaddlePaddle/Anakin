#include "saber/funcs/impl/cuda/saber_transpose.h"
#include <math.h>

namespace anakin {

namespace saber {

#define SABER_TRANSPOSE_TILE_DIM 16
template <typename dtype>
__global__ void transpose_tile_2d_if(dtype *odata, const dtype *idata, int num, int channel, int height, int width) {
    // Handle to thread block group
    __shared__ float tile[SABER_TRANSPOSE_TILE_DIM][SABER_TRANSPOSE_TILE_DIM + 1];
    for (int i = 0; i < num * channel; ++i) {
        unsigned int offset = i * height * width;
        unsigned int yIndex;
        unsigned int xIndex;


        xIndex = blockIdx.x * SABER_TRANSPOSE_TILE_DIM + threadIdx.x;
        yIndex = blockIdx.y * SABER_TRANSPOSE_TILE_DIM + threadIdx.y;

        if (xIndex < width && yIndex < height) {


            unsigned int index_in = xIndex + (yIndex)* width;
            tile[threadIdx.y][threadIdx.x] = idata[offset + index_in];
        }
        __syncthreads();

        xIndex = blockIdx.y * SABER_TRANSPOSE_TILE_DIM + threadIdx.x;
        yIndex = blockIdx.x * SABER_TRANSPOSE_TILE_DIM + threadIdx.y;
        if (xIndex < height && yIndex < width) {
            unsigned int index_out = xIndex + (yIndex)* height;
            odata[offset + index_out] = tile[threadIdx.x][threadIdx.y];
        }
        __syncthreads();

    }
}

template <DataType OpDtype>
SaberStatus SaberTranspose<NV, OpDtype>::dispatch(\
    const std::vector<DataTensor_in *>& inputs,\
    std::vector<DataTensor_out *>& outputs, \
    TransposeParam<NV>& param) {

    cudaStream_t stream = this->_ctx->get_compute_stream();

    int w_out = outputs[0]->width();
    int h_out = outputs[0]->height();
    int c_out = outputs[0]->channel();
    int n_out = outputs[0]->num();

    int w_in = inputs[0]->width();
    int h_in = inputs[0]->height();
    int c_in = inputs[0]->channel();
    int n_in = inputs[0]->num();

    int num_idx = inputs[0]->num_index();
    int channel_idx = inputs[0]->channel_index();
    int height_idx = inputs[0]->height_index();
    int width_idx = inputs[0]->width_index();

    int dims = inputs[0]->dims();

    CHECK_EQ(c_in, c_out) << "input channel should = output channel";
    CHECK_EQ(n_in, n_out) << "input batch size should = output batch size";
    CHECK_EQ(h_in, w_out) << "input width size should = output height size";
    CHECK_EQ(w_in, h_out) << "input height size should = output width size";

    int block_x = SABER_TRANSPOSE_TILE_DIM;
    int block_y = SABER_TRANSPOSE_TILE_DIM;
    int grid_x = (w_in+SABER_TRANSPOSE_TILE_DIM - 1) / SABER_TRANSPOSE_TILE_DIM;
    int grid_y = (h_in+SABER_TRANSPOSE_TILE_DIM - 1) / SABER_TRANSPOSE_TILE_DIM;
    dim3 block(block_x, block_y);
    dim3 grid(grid_x, grid_y);

    const InDataType* in_data = (const InDataType*)inputs[0]->data();
    OutDataType* out_data = (OutDataType*)outputs[0]->mutable_data();

    transpose_tile_2d_if<<<grid, block, 0, stream>>>(out_data, in_data, n_in, c_in, h_in, w_in);

    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberTranspose, TransposeParam, NV, AK_HALF);
DEFINE_OP_TEMPLATE(SaberTranspose, TransposeParam, NV, AK_INT8);
}//namespace saber

}//namespace anakin