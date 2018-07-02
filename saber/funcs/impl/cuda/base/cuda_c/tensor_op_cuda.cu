#include "saber/core/tensor_op.h"
#include <limits>

namespace anakin{

namespace saber{

template <typename Dtype>
__global__ void set_device_data(Dtype* data_ptr, Dtype value, int size){
    CUDA_KERNEL_LOOP(index, size){
        data_ptr[index] = value;
    }
}

template <typename Dtype>
__global__ void print_device_data(const Dtype* data_ptr, int size, int width){
    for (int i = 0; i < size; i++){
        printf("%.2f ", static_cast<float>(data_ptr[i]));
        if ((i + 1) % width == 0){
            printf("\n");
        }
    }
    printf("\n");
}

template <typename Dtype>
__global__ void cuda_cvt_data(const float* src, Dtype* dst, Dtype scale, int size){
    CUDA_KERNEL_LOOP(index, size){
        dst[index] = static_cast<Dtype>(src[index] * scale);
    }
}

template <class Tensor_t>
void fill_tensor_device_const(Tensor_t& tensor, \
    typename Tensor_t::dtype value, \
    typename Tensor_t::API::stream_t stream){

    typedef typename Tensor_t::dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());
    int size = tensor.size();
    set_device_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(data_ptr, value, size);
    CUDA_POST_KERNEL_CHECK;
};


template <class Tensor_t>
void fill_tensor_device_rand(Tensor_t& tensor, typename Tensor_t::API::stream_t stream) {

    typedef typename Tensor_t::dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());
    int size = tensor.size();

    float* data_f;
    cudaMalloc(&data_f, size * sizeof(float));

    curandGenerator_t gen;
    CHECK_EQ(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandSetPseudoRandomGeneratorSeed(gen, rand()), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandGenerateUniform(gen, data_f, size), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandDestroyGenerator(gen), CURAND_STATUS_SUCCESS);

    Dtype scale = std::numeric_limits<Dtype>::max();

    cuda_cvt_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(data_f, data_ptr, scale, size);
    cudaDeviceSynchronize();
    cudaFree(data_f);

    CUDA_POST_KERNEL_CHECK;
};

template <class Tensor_t>
void fill_tensor_device_rand(Tensor_t& tensor, typename Tensor_t::dtype vstart, \
    typename Tensor_t::dtype vend, typename Tensor_t::API::stream_t stream) {

    typedef typename Tensor_t::dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());
    int size = tensor.size();

    float* data_f;
    cudaMalloc(&data_f, size * sizeof(float));

    curandGenerator_t gen;
    CHECK_EQ(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandSetPseudoRandomGeneratorSeed(gen, rand()), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandGenerateUniform(gen, data_f, size), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandDestroyGenerator(gen), CURAND_STATUS_SUCCESS);

    Dtype scale = vend - vstart;

    cuda_cvt_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(data_f, data_ptr, scale, size);
    cudaDeviceSynchronize();
    cudaFree(data_f);

    CUDA_POST_KERNEL_CHECK;
};

template <class Tensor_t>
void print_tensor_device(Tensor_t& tensor, typename Tensor_t::API::stream_t stream){

    typedef typename Tensor_t::Dtype Dtype;
    LOG(INFO) << "device tensor size: " << tensor.size();
    const Dtype* data_ptr = static_cast<const Dtype*>(tensor.get_buf()->get_data());
    int size = tensor.size();
    print_device_data<<<1, 1, 0, stream>>>(data_ptr, size, tensor.width());
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
};

#define FILL_TENSOR_NV(type, layout) \
    template void fill_tensor_device_const<Tensor<NV, type, layout>>\
        (Tensor<NV, type, layout>& tensor, DataTrait<NV, type>::dtype value, \
        typename TargetWrapper<NV>::stream_t stream); \
    template void fill_tensor_device_rand<Tensor<NV, type, layout>>\
        (Tensor<NV, type, layout>& tensor, typename TargetWrapper<NV>::stream_t stream); \
    template void fill_tensor_device_rand<Tensor<NV, type, layout>>\
        (Tensor<NV, type, layout>& tensor, DataTrait<NV, type>::dtype vstart, \
        DataTrait<NV, type>::dtype vend, typename TargetWrapper<NV>::stream_t stream); \
    template void print_tensor_device<Tensor<NV, type, layout>>\
        (Tensor<NV, type, layout>& tensor, typename TargetWrapper<NV>::stream_t stream);

FILL_TENSOR_NV(AK_FLOAT, NCHW);
FILL_TENSOR_NV(AK_FLOAT, NHWC);
FILL_TENSOR_NV(AK_FLOAT, NHW);
FILL_TENSOR_NV(AK_FLOAT, NW);
FILL_TENSOR_NV(AK_FLOAT, HW);
FILL_TENSOR_NV(AK_FLOAT, W);

FILL_TENSOR_NV(AK_INT8, NCHW);
FILL_TENSOR_NV(AK_INT8, NHWC);
FILL_TENSOR_NV(AK_INT8, NHW);
FILL_TENSOR_NV(AK_INT8, NW);
FILL_TENSOR_NV(AK_INT8, HW);
FILL_TENSOR_NV(AK_INT8, W);

// INT8 NCHW_C4
template void fill_tensor_device_const<Tensor<NV, AK_INT8, NCHW_C4>>(Tensor<NV, AK_INT8, NCHW_C4>& tensor, \
    char value, typename TargetWrapper<NV>::stream_t stream);
template void fill_tensor_device_rand<Tensor<NV, AK_INT8, NCHW_C4>>(Tensor<NV, AK_INT8, NCHW_C4>& tensor, \
    typename TargetWrapper<NV>::stream_t stream);

template <>
void print_tensor_device<Tensor<NV, AK_INT8, NCHW_C4>>(Tensor<NV, AK_INT8, NCHW_C4>& tensor, \
    typename TargetWrapper<NV>::stream_t stream) {

    typedef typename Tensor<NV, AK_INT8, NCHW_C4>::Dtype Dtype;
            LOG(INFO) << "device tensor size: " << tensor.size();
    const Dtype* data_ptr = (const Dtype*)tensor.get_buf()->get_data();
    int size = tensor.size();
    print_device_data<<<1, 1, 0, stream>>>(data_ptr, size, tensor.width() * 4);
    CUDA_POST_KERNEL_CHECK;
};

// use BLOCKCOUNT and THREADNUM
__global__
void int8nchwc4_fp32nchw(float* out_data, const char* in_data,
                         int valid_num, int valid_channel_4, int valid_height, int valid_width,
                         int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                         int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
                         float* scale, int count) {

    float load0, load1, load2, load3;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    int read_w = (gid) % valid_width;
    int read_h = (gid / (in_h_stride)) % valid_height;
    int read_c = (gid / (in_c_stride)) % valid_channel_4;
    int read_n = (gid / (in_n_stride)) % valid_num;
    int scale_index = read_c << 2;

    int in_offset = read_n * in_n_stride
                  + read_c * in_c_stride
                  + read_h * in_h_stride
                  + read_w;

    int out_offset = read_n * out_n_stride
                   + read_c * (out_c_stride << 2)
                   + read_h * out_h_stride
                   + read_w * out_w_stride;

    if (gid < count) {

        char4 readin = __ldg(&((const char4*)in_data)[in_offset]);

        load0 = static_cast<float>(readin.x);
        load1 = static_cast<float>(readin.y);
        load2 = static_cast<float>(readin.z);
        load3 = static_cast<float>(readin.w);

        out_data[out_offset] = load0 * scale[scale_index]; out_offset += out_c_stride;
        out_data[out_offset] = load1 * scale[scale_index + 1]; out_offset += out_c_stride;
        out_data[out_offset] = load2 * scale[scale_index + 2]; out_offset += out_c_stride;
        out_data[out_offset] = load3 * scale[scale_index + 3];
    }
}

template<>
SaberStatus DataTensorTransformHelper::transform<Tensor<NV, AK_FLOAT, NCHW>, Tensor<NV, AK_INT8, NCHW_C4> >(
        Tensor<NV, AK_FLOAT, NCHW> &out_tensor,
        const Tensor<NV, AK_INT8, NCHW_C4> &in_tensor, Context<NV> ctx){

    Shape out_stride = out_tensor.get_stride();

    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();

    int count = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3];

    const char * in_data = in_tensor.data();
    float * out_data = out_tensor.mutable_data();

    cudaStream_t cuda_stream = ctx.get_compute_stream();
    int8nchwc4_fp32nchw<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, in_data,
            in_shape[0], in_shape[1], in_shape[2], in_shape[3],
            in_shape[1] * in_shape[2] * in_shape[3],
            in_shape[2] * in_shape[3],
            in_shape[3], 1,
            out_stride[0], out_stride[1], out_stride[2], out_stride[3],
            _weight_scale, count);

    return SaberSuccess;
}

__global__
void transform_nchw_2_c4(char* out_data, const float* in_data,
                         int valid_num, int valid_channel_4, int valid_height, int valid_width,
                         int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                         int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
                         float scale,
                         int count) {

    int load0, load1, load2, load3;
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    int write_w = (gid) % valid_width;
    int write_h = (gid / (out_h_stride)) % valid_height;
    int write_c = (gid / (out_c_stride)) % valid_channel_4;
    int write_n = (gid / (out_n_stride)) % valid_num;

    int in_offset = write_n * in_n_stride
                  + write_c * (in_c_stride << 2)
                  + write_h * in_h_stride
                  + write_w * in_w_stride;

    int out_offset = write_n * out_n_stride
                   + write_c * out_c_stride
                   + write_h * out_h_stride
                   + write_w;

    if (gid < count) {

        char4 write;

        load0 = __float2int_rn(__ldg(&in_data[in_offset]) * scale);
        write.x = static_cast<char>(load0);

        in_offset += in_c_stride;
        load1 = __float2int_rn(__ldg(&in_data[in_offset]) * scale);
        write.y = static_cast<char>(load1);

        in_offset += in_c_stride;
        load2 = __float2int_rn(__ldg(&in_data[in_offset]) * scale);
        write.z = static_cast<char>(load2);

        in_offset += in_c_stride;
        load3 = __float2int_rn(__ldg(&in_data[in_offset]) * scale);
        write.w = static_cast<char>(load3);

        ((char4*)out_data)[out_offset] = write;

    }
}

template<>
SaberStatus DataTensorTransformHelper::transform<Tensor<NV, AK_INT8, NCHW_C4>, Tensor<NV, AK_FLOAT, NCHW> >(
        Tensor<NV, AK_INT8, NCHW_C4> &out_tensor,
        const Tensor<NV, AK_FLOAT, NCHW> &in_tensor, Context<NV> ctx){

    const float * in_data = in_tensor.data();
    char * out_data = out_tensor.mutable_data();

    Shape in_stride = in_tensor.get_stride();

    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();
    int count = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3];

    cudaStream_t cuda_stream = ctx.get_compute_stream();
    transform_nchw_2_c4<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(out_data, in_data,
            out_shape[0], out_shape[1], out_shape[2], out_shape[3],
            in_stride[0], in_stride[1], in_stride[2], in_stride[3],
            out_shape[1] * out_shape[2] * out_shape[3],
            out_shape[2] * out_shape[3], out_shape[3], 1,
            (1.f / _in_scale), count);

    return SaberSuccess;
}

__global__ void transform_nchw_2_nchw(float * out_data,
                             const float* in_data, const int count,
                             int in_n, int in_c, int in_h, int in_w,
                             int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
                             int out_n, int out_c, int out_h, int out_w,
                             int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
                             float *scale) {
    CUDA_KERNEL_LOOP(tid, count){
        int read_w =  tid % in_w;
        int read_h = (tid / (in_w)) % in_h;
        int read_c = (tid / (in_h * in_w)) % in_c;
        int read_n = (tid / (in_c * in_h * in_w)) % in_n;

        int write_w =  tid % out_w;
        int write_h = (tid / (out_w)) % out_h;
        int write_c = (tid / (out_h * out_w)) % out_c;
        int write_n = (tid / (out_c * out_h * out_w)) % out_n;

        int in_idx = read_n * in_n_stride
                     + read_c * in_c_stride
                     + read_h * in_h_stride
                     + read_w * in_w_stride;

        int out_idx = write_n * out_n_stride
                      + write_c * out_c_stride
                      + write_h * out_h_stride
                      + write_w * out_w_stride;

        float in_var = in_data[in_idx];
        float in_scale = scale[read_c];
        out_data[out_idx] = in_var * in_scale;
    }
}
template<>
SaberStatus DataTensorTransformHelper::transform<Tensor<NV, AK_FLOAT, NCHW>, Tensor<NV, AK_FLOAT, NCHW> >(
        Tensor<NV, AK_FLOAT, NCHW> &out_tensor,
        const Tensor<NV, AK_FLOAT, NCHW> &in_tensor, Context<NV> ctx){

    Shape in_shape = in_tensor.valid_shape();
    Shape out_shape = out_tensor.valid_shape();

    Shape stride_in = in_tensor.get_stride();
    Shape stride_out = out_tensor.get_stride();

    const float *in_data = (const float*)in_tensor.data();
    float *out_data = (float*)out_tensor.mutable_data();

    const int count = in_tensor.valid_size();
    cudaStream_t cuda_stream = ctx.get_compute_stream();

    transform_nchw_2_nchw
    <<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, cuda_stream>>>(
            out_data, in_data, count,
                    in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                    stride_in[0], stride_in[1], stride_in[2], stride_in[3],
                    out_shape[0], out_shape[1], out_shape[2], out_shape[3],
                    stride_out[0], stride_out[1], stride_out[2], stride_out[3],
                    _weight_scale);

    return SaberSuccess;
}

} //namespace saber

} //namespace anakin
