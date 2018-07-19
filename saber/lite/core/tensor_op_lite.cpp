#include "tensor_op_lite.h"
#include <cstdlib>
#include <cmath>
#include <random>

namespace anakin {

namespace saber {

namespace lite{

template <>
void fill_tensor_const(Tensor<CPU, AK_FLOAT>& tensor, float value) {
    float* data_ptr = (float*)tensor.get_buf()->get_data_mutable();
    int size = tensor.valid_size();
    for (int i = 0; i < size; ++i) {
        data_ptr[i] = value;
    }
}

template <>
void fill_tensor_rand(Tensor<CPU, AK_FLOAT>& tensor) {
    float* data_ptr = (float*)tensor.get_buf()->get_data_mutable();
    for (int i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<float>(rand());
    }
}

template <>
void fill_tensor_rand(Tensor<CPU, AK_FLOAT>& tensor, float vstart, float vend) {
    float* data_ptr = (float*)tensor.get_buf()->get_data_mutable();
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
void print_tensor(const Tensor<CPU, AK_FLOAT>& tensor) {
    printf("host tensor data size: %d\n", tensor.size());
    const float* data_ptr = (const float*)tensor.get_buf()->get_data();
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
void print_tensor_valid(const Tensor<CPU, AK_FLOAT>& tensor) {
    printf("host tensor data valid size: %d\n", tensor.valid_size());

    const float* data_ptr = tensor.data();


    Shape sh_real = tensor.shape();
    Shape sh_act = tensor.valid_shape();
    Shape offset_act = tensor.offset();
//    int start_w = offset_act.width();
//    int start_h = offset_act.height();
//    int start_c = offset_act.channel();
//    int start_n = offset_act.num();

    Shape stride = tensor.get_stride();

    int stride_w = stride.width();
    int stride_h = stride.height();
    int stride_c = stride.channel();
    int stride_n = stride.num();
    //int stride_h = sh_real.count(3);
    //int stride_c = sh_real.count(2);
    //int stride_n = sh_real.count(1);
    //int stride_n = sh_real.count(0);
    int w = tensor.width();
    int h = tensor.height();
    int c = tensor.channel();
    int n = tensor.num();

    const float* ptr_host = tensor.data();
    for (int in = 0; in < n; ++in) {
        const float* ptr_batch = ptr_host + in * stride_n;
        for (int ic = 0; ic < c; ++ic) {
            const float* ptr_channel = ptr_batch + ic * stride_c;
            for (int ih = 0; ih < h; ++ih) {
                const float* ptr_row = ptr_channel + ih * stride_h;
                for (int iw = 0; iw < w; ++iw) {
                    printf("%.2f ", ptr_row[iw]);
                }
                printf("\n");
            }
        }
    }
    printf("\n");
}

template <>
double tensor_mean(const Tensor<CPU, AK_FLOAT>& tensor) {

    double val = 0.0;

    Shape sh_act = tensor.valid_shape();

    Shape stride = tensor.get_stride();

    int stride_w = stride.width();
    int stride_h = stride.height();
    int stride_c = stride.channel();
    int stride_n = stride.num();

    int w = tensor.width();
    int h = tensor.height();
    int c = tensor.channel();
    int n = tensor.num();

    const float* ptr_host = tensor.data();

    for (int in = 0; in < n; ++in) {
        const float* ptr_batch = ptr_host + in * stride_n;
        for (int ic = 0; ic < c; ++ic) {
            const float* ptr_channel = ptr_batch + ic * stride_c;
            for (int ih = 0; ih < h; ++ih) {
                const float* ptr_row = ptr_channel + ih * stride_h;
                for (int iw = 0; iw < w; ++iw) {
                    val += ptr_row[iw];
                }
            }
        }
    }
    return val / tensor.valid_size();
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

template void tensor_cmp_host(const float* src1, const float* src2, int size, double& max_ratio, double& max_diff);

} //namespace lite

} //namespace saber

} //namespace anakin
