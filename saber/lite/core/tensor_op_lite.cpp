#include "tensor_op_lite.h"
#include <cstdlib>
#include <cmath>
#include <random>

namespace anakin {

namespace saber {

namespace lite{

template <typename Dtype>
void fill_tensor_host_const(Tensor<Dtype>& tensor, Dtype value) {
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());
    int size = tensor.size();

    for (int i = 0; i < size; ++i) {
        data_ptr[i] = value;
    }
}

template <typename Dtype>
void fill_tensor_host_rand(Tensor<Dtype>& tensor) {
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());
    for (int i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<Dtype>(rand());
    }
}

template <typename Dtype>
void fill_tensor_host_rand(Tensor<Dtype>& tensor, Dtype vstart, Dtype vend) {
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1.f);
    int size = tensor.size();
    for (int i = 0; i < size; ++i) {
        Dtype random_num = vstart + (vend - vstart) * dis(gen);
        data_ptr[i] = random_num;
    }
}

template <typename Dtype>
void print_tensor_host(Tensor<Dtype>& tensor) {
    LOG(INFO) << "host tensor data:" << tensor.size();
    const Dtype* data_ptr = static_cast<const Dtype*>(tensor.get_buf()->get_data());
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
void print_tensor_host_valid(Tensor<Dtype>& tensor) {
    LOG(INFO) << "host tensor data:" << tensor.valid_size();
    const Dtype* data_ptr = tensor.data();
    int size = tensor.valid_size();
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

#define FILL_TENSOR_HOST(type) \
    template void fill_tensor_host_const<type>\
        (Tensor<type>& tensor, type value); \
    template void fill_tensor_host_rand<type>\
        (Tensor<type>& tensor); \
    template void fill_tensor_host_rand<type>\
        (Tensor<type>& tensor, type vstart, type vend); \
    template void print_tensor_host<type>\
        (Tensor<type>& tensor); \
    template void print_tensor_host_valid<type>\
        (Tensor<type>& tensor);

template void tensor_cmp_host<float>(const float* src1, const float* src2, \
                                     int size, double& max_ratio, double& max_diff);
template void tensor_cmp_host<char>(const char* src1, const char* src2, int size, \
                                    double& max_ratio, double& max_diff);

FILL_TENSOR_HOST(float)

} //namespace lite

} //namespace saber

} //namespace anakin
