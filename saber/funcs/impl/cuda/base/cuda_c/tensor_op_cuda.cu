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
    typename Tensor_t::FDtype value, \
    typename Tensor_t::API::stream_t stream){

    typedef typename Tensor_t::FDtype Dtype;
    Dtype* data_ptr = tensor.mutable_data();
    int size = tensor.size();
    set_device_data<<<CUDA_GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(data_ptr, value, size);
    CUDA_POST_KERNEL_CHECK;
};


template <class Tensor_t>
void fill_tensor_device_rand(Tensor_t& tensor, typename Tensor_t::API::stream_t stream) {

    typedef typename Tensor_t::FDtype Dtype;
    Dtype* data_ptr = tensor.mutable_data();
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
void fill_tensor_device_rand(Tensor_t& tensor, typename Tensor_t::FDtype vstart, \
    typename Tensor_t::FDtype vend, typename Tensor_t::API::stream_t stream) {

    typedef typename Tensor_t::FDtype Dtype;
    Dtype* data_ptr = tensor.mutable_data();
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

    typedef typename Tensor_t::FDtype Dtype;
    LOG(INFO) << "device tensor size: " << tensor.size();
    const Dtype* data_ptr = tensor.data();
    int size = tensor.size();
    print_device_data<<<1, 1, 0, stream>>>(data_ptr, size, tensor.width());
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
};

template void fill_tensor_device_const<Tensor<NV>>(Tensor<NV>& tensor, Tensor<NV>::FDtype value, \
        typename TargetWrapper<NV>::stream_t stream);
template void fill_tensor_device_rand<Tensor<NV>>(Tensor<NV>& tensor, typename TargetWrapper<NV>::stream_t stream);
template void fill_tensor_device_rand<Tensor<NV>>(Tensor<NV>& tensor, Tensor<NV>::FDtype vstart, \
        Tensor<NV>::FDtype vend, typename TargetWrapper<NV>::stream_t stream);
template void print_tensor_device<Tensor<NV>>(Tensor<NV>& tensor, typename TargetWrapper<NV>::stream_t stream);

} //namespace saber

} //namespace anakin
