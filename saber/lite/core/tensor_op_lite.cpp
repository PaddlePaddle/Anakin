#include "tensor_op_lite.h"
#include <cstdlib>
#include <cmath>
#include <random>

namespace anakin {

namespace saber {

namespace lite{

template <>
void fill_tensor_const(Tensor<CPU, AK_FLOAT>& tensor, float value) {
    float* data_ptr = tensor.mutable_data();
    int size = tensor.valid_size();
    for (int i = 0; i < size; ++i) {
        data_ptr[i] = value;
    }
}

template <>
void fill_tensor_rand(Tensor<CPU, AK_FLOAT>& tensor) {
    float* data_ptr = tensor.mutable_data();
    for (int i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<float>(rand());
    }
}

template <>
void fill_tensor_rand(Tensor<CPU, AK_FLOAT>& tensor, float vstart, float vend) {
    float* data_ptr = tensor.mutable_data();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1.f);
    int size = tensor.size();
    for (int i = 0; i < size; ++i) {
        float random_num = vstart + (vend - vstart) * dis(gen);
        data_ptr[i] = random_num;
    }
}

template <>
void print_tensor(Tensor<CPU, AK_FLOAT>& tensor) {
    printf("host tensor data size: %d\n", tensor.size());
    const float* data_ptr = tensor.mutable_data();
    int size = tensor.size();
    for (int i = 0; i < size; ++i) {
        printf("%.2f ", data_ptr[i]);
        if ((i + 1) % tensor.width() == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

template <>
void print_tensor_valid(Tensor<CPU, AK_FLOAT>& tensor) {
    printf("host tensor data valid size: %d\n", tensor.valid_size());
    const float* data_ptr = tensor.data();
    int size = tensor.valid_size();
    for (int i = 0; i < size; ++i) {
        printf("%.2f ", data_ptr[i]);
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

} //namespace lite

} //namespace saber

} //namespace anakin
