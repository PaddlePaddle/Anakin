
#include "saber/funcs/impl/cuda/reorder.h"

namespace anakin {
namespace saber {

template <typename vtype, typename dtype>
__global__
void transform_nchw_2_c4(dtype* out_data, const dtype* in_data,
                         int valid_num, int valid_channel_4, int valid_height, int valid_width,
                         int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                         int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
                         int count) {

    dtype load0, load1, load2, load3;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    int write_w = (gid) % valid_width;
    int write_h = (gid / (out_h_stride)) % valid_height;
    int write_c = (gid / (out_c_stride)) % valid_channel_4;
    int write_n = (gid / (out_n_stride)) % valid_num;

    int in_offset = write_n * in_n_stride
                    + write_c * in_c_stride * 4
                    + write_h * in_h_stride
                    + write_w * in_w_stride;

    int out_offset = write_n * out_n_stride
                     + write_c * out_c_stride
                     + write_h * out_h_stride
                     + write_w;

    if (gid < count) {
        vtype write;
        load0 = in_data[in_offset];
        write.x = load0;

        in_offset += in_c_stride;
        load1 = in_data[in_offset];
        write.y = load1;

        in_offset += in_c_stride;
        load2 = in_data[in_offset];
        write.z = load2;

        in_offset += in_c_stride;
        load3 = in_data[in_offset];
        write.w = load3;

        ((vtype*)out_data)[out_offset] = write;
    }
}

template<>
SaberStatus convert_nchw_to_nchwc4<NV>(Tensor<NV> &out_tensor,
        const Tensor<NV> &in_tensor, Context<NV> ctx) {

    CHECK_EQ(out_tensor.get_dtype(), in_tensor.get_dtype());
    const void * in_data = in_tensor.data();
    void * out_data = out_tensor.mutable_data();

    Shape in_stride = in_tensor.get_stride();
    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();
    int count = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];
    cudaStream_t cuda_stream = ctx.get_compute_stream();
    if (out_tensor.get_dtype() == AK_INT8) {
        transform_nchw_2_c4<char4, char>
                << < CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS,
                0, cuda_stream >> > ((char*)out_data, (const char*)in_data,
                out_shape[0], out_shape[1], out_shape[2], out_shape[3],
                in_stride[0], in_stride[1], in_stride[2], in_stride[3],
                out_shape[1] * out_shape[2] * out_shape[3],
                out_shape[2] * out_shape[3], out_shape[3], 1,
                count);
    } else if (out_tensor.get_dtype() == AK_FLOAT) {
        transform_nchw_2_c4<float4, float>
                << < CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS,
                0, cuda_stream >> > ((float*)out_data, (const float*)in_data,
                out_shape[0], out_shape[1], out_shape[2], out_shape[3],
                in_stride[0], in_stride[1], in_stride[2], in_stride[3],
                out_shape[1] * out_shape[2] * out_shape[3],
                out_shape[2] * out_shape[3], out_shape[3], 1,
                count);
    } else {
        LOG(FATAL) << "NOT SUPPORT THIS DATATYPE in reorder!!!";
    }
    return SaberSuccess;
}

template <typename vtype, typename dtype>
__global__
void transform_nchwc4_2_nchw(dtype* out_data, const dtype* in_data,
                         int valid_num, int valid_channel_4, int valid_height, int valid_width,
                         int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                         int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
                         int count) {

    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    int read_w = (gid) % valid_width;
    int read_h = (gid / (in_h_stride)) % valid_height;
    int read_c = (gid / (in_c_stride)) % valid_channel_4;
    int read_n = (gid / (in_n_stride)) % valid_num;

    int in_offset = read_n * in_n_stride
                    + read_c * in_c_stride
                    + read_h * in_h_stride
                    + read_w;

    int out_offset = read_n * out_n_stride
                     + read_c * (out_c_stride << 2)
                     + read_h * out_h_stride
                     + read_w * out_w_stride;

    if (gid < count) {
        vtype readin = ((const vtype*)in_data)[in_offset];
        out_data[out_offset] = readin.x; out_offset += out_c_stride;
        out_data[out_offset] = readin.y; out_offset += out_c_stride;
        out_data[out_offset] = readin.z; out_offset += out_c_stride;
        out_data[out_offset] = readin.w;
    }
}

template<>
SaberStatus convert_nchwc4_to_nchw<NV>(
        Tensor<NV> &out_tensor,
        const Tensor<NV> &in_tensor,
        Context<NV> ctx) {

    CHECK_EQ(out_tensor.get_dtype(), in_tensor.get_dtype());

    Shape out_stride = out_tensor.get_stride();
    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();
    int count = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3];

    const void * in_data = in_tensor.data();
    void * out_data = out_tensor.mutable_data();

    cudaStream_t cuda_stream = ctx.get_compute_stream();
    if (out_tensor.get_dtype() == AK_INT8) {
        transform_nchwc4_2_nchw<char4, char>
                << < CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream >> > (
                        (char*)out_data, (const char*)in_data,
                        in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                        in_shape[1] * in_shape[2] * in_shape[3],
                        in_shape[2] * in_shape[3], in_shape[3], 1,
                        out_stride[0], out_stride[1], out_stride[2], out_stride[3], count);
    } else if (out_tensor.get_dtype() == AK_FLOAT) {
        transform_nchwc4_2_nchw<float4, float>
                << < CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream >> > (
                        (float*)out_data, (const float*)in_data,
                in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                in_shape[1] * in_shape[2] * in_shape[3],
                in_shape[2] * in_shape[3], in_shape[3], 1,
                out_stride[0], out_stride[1], out_stride[2], out_stride[3], count);
    } else {
        LOG(FATAL) << "NOT SUPPORT THIS DATATYPE in reorder!!!";
    }

    return SaberSuccess;
}


} // namespace saber
} // namespace anakin