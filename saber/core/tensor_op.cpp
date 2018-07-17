#include "tensor_op.h"
#include "anakin_config.h"
#include <cstdlib>
#include <random>

namespace anakin {

namespace saber {

template <typename Dtype>
void fill_tensor_host_const_impl(Dtype* dio, Dtype value, int size) {
    for (int i = 0; i < size; ++i) {
        dio[i] = value;
    }
}


template <class Tensor_t>
void fill_tensor_host_const(Tensor_t& tensor, float value) {

    long long size = tensor.size();
    DataType type = tensor.get_dtype();
    switch (type){
        case AK_UINT8:
    }
    Dtype* data_ptr = static_cast<Dtype*>(tensor.mutable_data());
    int size = tensor.size();

    for (int i = 0; i < size; ++i) {
        data_ptr[i] = value;
    }
}

template <class Tensor_t>
void fill_tensor_host_rand(Tensor_t& tensor) {
    typedef typename Tensor_t::FDtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.mutable_data());

    for (int i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<Dtype>(rand());
    }
}

template <class Tensor_t>
void fill_tensor_host_seq(Tensor_t& tensor) {
    typedef typename Tensor_t::FDtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.mutable_data());

    for (int i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<Dtype>(i);
    }
}

template <class Tensor_t>
void fill_tensor_host_rand(Tensor_t& tensor, typename Tensor_t::FDtype vstart, \
                           typename Tensor_t::FDtype vend) {
    typedef typename Tensor_t::FDtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.mutable_data());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1.f);
    int size = tensor.size();
    for (int i = 0; i < size; ++i) {
        Dtype random_num = vstart + (vend - vstart) * dis(gen);
        data_ptr[i] = random_num;
    }
}

template <class Tensor_t>
void print_tensor_host(Tensor_t& tensor) {

    typedef typename Tensor_t::FDtype Dtype;
    LOG(INFO) << "host tensor data:" << tensor.size();
    const Dtype* data_ptr = static_cast<const Dtype*>(tensor.data());
    int size = tensor.size();

    for (int i = 0; i < size; ++i) {
        printf("%.2f ", static_cast<float>(data_ptr[i]));

        if ((i + 1) % tensor.width() == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

template <typename Dtype>
void tensor_cmp_host(const Dtype* src1, const Dtype* src2, \
                     int size, double& max_ratio, double& max_diff) {

    const double eps = 1e-6f;
    max_diff = fabs(src1[0] - src2[0]);
    max_ratio = 2.0 * max_diff / (src1[0] + src2[0] + eps);

    for (int i = 1; i < size; ++i) {
        double diff = fabs(src1[i] - src2[i]);

        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = 2.0 * max_diff / (src1[i] + src2[i] + eps);
        }
    }
}

#define FILL_TENSOR_HOST(target) \
    template void fill_tensor_host_const<Tensor<target>>(Tensor<target>& tensor, \
        Tensor<target>::FDtype value); \
    template void fill_tensor_host_rand<Tensor<target>>(Tensor<target>& tensor); \
    template void fill_tensor_host_rand<Tensor<target>>(Tensor<target>& tensor, Tensor<target>::FDtype vstart, \
        Tensor<target>::FDtype vend); \
    template void print_tensor_host<Tensor<target>>(Tensor<target>& tensor);\
    template void fill_tensor_host_seq<Tensor<target>>(Tensor<target>& tensor);

#if defined(BUILD_LITE) || defined(USE_X86_PLACE) || defined(USE_AMD) || defined(USE_CUDA)
FILL_TENSOR_HOST(X86)
#endif

#ifdef USE_CUDA
FILL_TENSOR_HOST(NVHX86)
#endif

#ifdef USE_ARM_PLACE
FILL_TENSOR_HOST(ARM)
#endif

template void tensor_cmp_host<float>(const float* src1, const float* src2, \
                                     int size, double& max_ratio, double& max_diff);
template void tensor_cmp_host<char>(const char* src1, const char* src2, int size, \
                                    double& max_ratio, double& max_diff);

} //namespace saber

} //namespace anakin
