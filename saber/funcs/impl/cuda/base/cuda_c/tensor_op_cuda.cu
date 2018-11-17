#include "saber/core/tensor_op.h"
#include "anakin_config.h"
#include <limits>

namespace anakin{

namespace saber{

#ifdef USE_CUDA

template <typename Dtype>
__global__ void set_device_data(Dtype* data_ptr, Dtype value, long long size) {
    CUDA_KERNEL_LOOP(index, size) {
        data_ptr[index] = value;
    }
}

template <typename Dtype>
__global__ void print_device_data(const Dtype* data_ptr, long long size, int width) {
    for (int i = 0; i < size; i++) {
        printf("%.6f ", static_cast<float>(data_ptr[i]));
        if ((i + 1) % width == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

template <typename Dtype>
__global__ void cuda_cvt_data(const float* src, Dtype* dst, Dtype vstart, Dtype scale, int size) {
    CUDA_KERNEL_LOOP(index, size) {
        dst[index] = static_cast<Dtype>(vstart + src[index] * scale);
    }
}

template <typename Dtype>
void fill_tensor_device_rand_impl(Dtype* data_ptr, long long size,
        typename Tensor<NV>::API::stream_t stream) {

    float* data_f;
    cudaMalloc(&data_f, size * sizeof(float));

    curandGenerator_t gen;
    CHECK_EQ(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandSetPseudoRandomGeneratorSeed(gen, rand()), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandGenerateUniform(gen, data_f, size), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandDestroyGenerator(gen), CURAND_STATUS_SUCCESS);

    Dtype scale = std::numeric_limits<Dtype>::max();
    Dtype z = 0;
    cuda_cvt_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(data_f, data_ptr, z, scale, size);
    cudaDeviceSynchronize();
    cudaFree(data_f);

    CUDA_POST_KERNEL_CHECK;
};

template <typename Dtype>
void fill_tensor_device_rand_impl2(Dtype* data_ptr, Dtype vstart, \
    Dtype vend, long long size , typename Tensor<NV>::API::stream_t stream) {

    float* data_f;
    cudaMalloc(&data_f, size * sizeof(float));

    curandGenerator_t gen;
    CHECK_EQ(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandSetPseudoRandomGeneratorSeed(gen, rand()), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandGenerateUniform(gen, data_f, size), CURAND_STATUS_SUCCESS);
    CHECK_EQ(curandDestroyGenerator(gen), CURAND_STATUS_SUCCESS);

    Dtype scale = vend - vstart;
    cuda_cvt_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(data_f, data_ptr, vstart, scale, size);
    cudaDeviceSynchronize();
    cudaFree(data_f);

    CUDA_POST_KERNEL_CHECK;
};

template<>
void fill_tensor_const<NV>(Tensor<NV>& tensor, float value, typename Tensor<NV>::API::stream_t stream) {
    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type) {
        case AK_UINT8: set_device_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS,
                    0, stream>>>((unsigned char*)dio, static_cast<unsigned char>(value), size); break;
        case AK_INT8: set_device_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS,
                    0, stream>>>((char*)dio, static_cast<char>(value), size); break;
        case AK_INT16: set_device_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS,
                    0, stream>>>((short*)dio, static_cast<short>(value), size); break;
        case AK_UINT16: set_device_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS,
                    0, stream>>>((unsigned short*)dio, static_cast<unsigned short>(value), size); break;
        case AK_HALF: set_device_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS,
                    0, stream>>>((short*)dio, static_cast<short>(value), size); break;
        case AK_UINT32: set_device_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS,
                    0, stream>>>((unsigned int*)dio, static_cast<unsigned int>(value), size); break;
        case AK_INT32: set_device_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS,
                    0, stream>>>((int*)dio, static_cast<int>(value), size); break;
        case AK_FLOAT: set_device_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS,
                    0, stream>>>((float*)dio, static_cast<float>(value), size); break;
        case AK_DOUBLE: set_device_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS,
                    0, stream>>>((double*)dio, static_cast<double>(value), size); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template<>
void fill_tensor_rand<NV>(Tensor<NV>& tensor, typename Tensor<NV>::API::stream_t stream) {
    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type) {
        case AK_UINT8: fill_tensor_device_rand_impl((unsigned char*)dio, size, stream); break;
        case AK_INT8: fill_tensor_device_rand_impl((char*)dio, size, stream); break;
        case AK_INT16: fill_tensor_device_rand_impl((short*)dio, size, stream); break;
        case AK_UINT16: fill_tensor_device_rand_impl((unsigned short*)dio, size, stream); break;
        case AK_UINT32: fill_tensor_device_rand_impl((unsigned int*)dio, size, stream); break;
        case AK_INT32: fill_tensor_device_rand_impl((int*)dio, size, stream); break;
        case AK_HALF: fill_tensor_device_rand_impl((short*)dio, size, stream); break;
        case AK_FLOAT: fill_tensor_device_rand_impl((float*)dio, size, stream); break;
        case AK_DOUBLE: fill_tensor_device_rand_impl((double*)dio, size, stream); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template<>
void fill_tensor_rand<NV>(Tensor<NV>& tensor, float vstart, float vend, typename Tensor<NV>::API::stream_t stream) {
    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type) {
        case AK_UINT8: fill_tensor_device_rand_impl2((unsigned char*)dio, static_cast<unsigned char>(vstart),
                                                   static_cast<unsigned char>(vend), size, stream); break;
        case AK_INT8: fill_tensor_device_rand_impl2((char*)dio, static_cast<char>(vstart), static_cast<char>(vend), size, stream); break;
        case AK_INT16: fill_tensor_device_rand_impl2((short*)dio, static_cast<short>(vstart), static_cast<short>(vend), size, stream); break;
        case AK_UINT16: fill_tensor_device_rand_impl2((unsigned short*)dio, static_cast<unsigned short>(vstart),
                                                    static_cast<unsigned short>(vend), size, stream); break;
        case AK_UINT32: fill_tensor_device_rand_impl2((unsigned int*)dio, static_cast<unsigned int>(vstart),
                                                    static_cast<unsigned int>(vend), size, stream); break;
        case AK_INT32: fill_tensor_device_rand_impl2((int*)dio, static_cast<int>(vstart), static_cast<int>(vend), size, stream); break;
        case AK_HALF: fill_tensor_device_rand_impl2((short*)dio, static_cast<short>(vstart), static_cast<short>(vend), size, stream); break;
        case AK_FLOAT: fill_tensor_device_rand_impl2((float*)dio, static_cast<float>(vstart), static_cast<float>(vend), size, stream); break;
        case AK_DOUBLE: fill_tensor_device_rand_impl2((double*)dio, static_cast<double>(vstart), static_cast<double>(vend), size, stream); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template<>
void print_tensor<NV>(Tensor<NV>& tensor, typename Tensor<NV>::API::stream_t stream) {
    LOG(INFO) << "device tensor data:" << tensor.size();
    const void* data_ptr = tensor.data();
    long long size = tensor.size();
    int width = tensor.width();
    DataType type = tensor.get_dtype();
    switch(type) {
        case AK_UINT8: print_device_data<<<1, 1,
                    0, stream>>>((const unsigned char*)data_ptr, size, width); break;
        case AK_INT8: print_device_data<<<1, 1,
                    0, stream>>>((const char*)data_ptr, size, width); break;
        case AK_UINT16: print_device_data<<<1, 1,
                    0, stream>>>((const unsigned short*)data_ptr, size, width); break;
        case AK_HALF: print_device_data<<<1, 1,
                    0, stream>>>((const short*)data_ptr, size, width); break;
        case AK_UINT32: print_device_data<<<1, 1,
                    0, stream>>>((const unsigned int*)data_ptr, size, width); break;
        case AK_INT32: print_device_data<<<1, 1,
                    0, stream>>>((const int*)data_ptr, size, width); break;
        case AK_FLOAT: print_device_data<<<1, 1,
                    0, stream>>>((const float*)data_ptr, size, width); break;
        case AK_DOUBLE: print_device_data<<<1, 1,
                    0, stream>>>((const double*)data_ptr, size, width); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
    printf("\n");
}

template<>
void print_tensor_valid<NV>(Tensor<NV>& tensor, typename Tensor<NV>::API::stream_t stream) {
    LOG(INFO) << "device tensor data:" << tensor.valid_size();
    const void* data_ptr = (const void*)((const char*)tensor.data() + tensor.data_offset() * type_length(tensor.get_dtype()));
    long long size = tensor.valid_size();
    int width = tensor.width();
    DataType type = tensor.get_dtype();
    if (tensor.is_continue_mem()) {
        switch(type) {
            case AK_UINT8: print_device_data<<<1, 1,
                        0, stream>>>((const unsigned char*)data_ptr, size, width); break;
            case AK_INT8: print_device_data<<<1, 1,
                        0, stream>>>((const char*)data_ptr, size, width); break;
            case AK_UINT16: print_device_data<<<1, 1,
                        0, stream>>>((const unsigned short*)data_ptr, size, width); break;
            case AK_HALF: print_device_data<<<1, 1,
                        0, stream>>>((const short*)data_ptr, size, width); break;
            case AK_UINT32: print_device_data<<<1, 1,
                        0, stream>>>((const unsigned int*)data_ptr, size, width); break;
            case AK_INT32: print_device_data<<<1, 1,
                        0, stream>>>((const int*)data_ptr, size, width); break;
            case AK_FLOAT: print_device_data<<<1, 1,
                        0, stream>>>((const float*)data_ptr, size, width); break;
            case AK_DOUBLE: print_device_data<<<1, 1,
                        0, stream>>>((const double*)data_ptr, size, width); break;
            default: LOG(FATAL) << "data type: " << type << " is unsupported now";
        }
        cudaDeviceSynchronize();
        CUDA_POST_KERNEL_CHECK;
        printf("\n");
    } else {
        Tensor<NV> tvalid(tensor.valid_shape());
        tvalid.copy_from(tensor);
        print_tensor<NV>(tvalid, stream);
    }
}


template<>
double tensor_mean_value<NV>(Tensor<NV>& tensor, typename Tensor<NV>::API::stream_t stream) {
    Tensor<NVHX86> tvalid(tensor.shape());
    Shape valid_shape = tensor.valid_shape();
    tensor.set_shape(tensor.shape());
    tvalid.copy_from(tensor);
    tensor.set_shape(valid_shape);
    return tensor_mean_value<NVHX86>(tvalid, stream);
}

template<>
double tensor_mean_value_valid<NV>(Tensor<NV>& tensor, typename Tensor<NV>::API::stream_t stream) {
    Tensor<NVHX86> tvalid(tensor.valid_shape());
    tvalid.copy_from(tensor);
    return tensor_mean_value<NVHX86>(tvalid, stream);
}
#endif

} //namespace saber

} //namespace anakin
