#include "tensor_op_lite.h"
#include <cstdlib>
#include <cmath>
#include <random>

namespace anakin {

namespace saber {

namespace lite{
template <typename Dtype>
void fill_tensor_host_const_impl(Dtype* dio, Dtype value, long long size) {
    for (long long i = 0; i < size; ++i) {
        dio[i] = value;
    }
}
template <>
void fill_tensor_const(Tensor<CPU>& tensor, float value) {
    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        //case AK_UINT8: fill_tensor_host_const_impl((unsigned char*)dio, static_cast<unsigned char>(value), size); break;
        case AK_INT8: fill_tensor_host_const_impl((char*)dio, static_cast<char>(value), size); break;
        //case AK_INT16: fill_tensor_host_const_impl((short*)dio, static_cast<short>(value), size); break;
        //case AK_UINT16: fill_tensor_host_const_impl((unsigned short*)dio, static_cast<unsigned short>(value), size); break;
        //case AK_HALF: fill_tensor_host_const_impl((short*)dio, static_cast<short>(value), size); break;
        //case AK_UINT32: fill_tensor_host_const_impl((unsigned int*)dio, static_cast<unsigned int>(value), size); break;
        case AK_INT32: fill_tensor_host_const_impl((int*)dio, static_cast<int>(value), size); break;
        case AK_FLOAT: fill_tensor_host_const_impl((float*)dio, static_cast<float>(value), size); break;
        //case AK_DOUBLE: fill_tensor_host_const_impl((double*)dio, static_cast<double>(value), size); break;
        default: LOGF("data type: %d is unsupported now", (int)type);
    }
}

template <typename Dtype>
void fill_tensor_host_rand_impl(Dtype* dio, long long size) {
    for (long long i = 0; i < size; ++i) {
        Dtype rand_x=static_cast<Dtype>(rand() % 256);
        dio[i] = (rand_x - 128) / 128;
    }

}
template <>
void fill_tensor_host_rand_impl<char>(char* dio, long long size) {
    for (long long i = 0; i < size; ++i) {
        dio[i] = rand() % 256 - 128;
    }
}
template <>
void fill_tensor_host_rand_impl<unsigned char>(unsigned char* dio, long long size) {
    for (long long i = 0; i < size; ++i) {
        dio[i] = rand() % 256;
    }
}

template <>
void fill_tensor_rand(Tensor<CPU>& tensor) {
    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        //case AK_UINT8: fill_tensor_host_rand_impl((unsigned char*)dio, size); break;
        case AK_INT8: fill_tensor_host_rand_impl((char*)dio, size); break;
        //case AK_INT16: fill_tensor_host_rand_impl((short*)dio, size); break;
        //case AK_UINT16: fill_tensor_host_rand_impl((unsigned short*)dio, size); break;
        //case AK_UINT32: fill_tensor_host_rand_impl((unsigned int*)dio, size); break;
        case AK_INT32: fill_tensor_host_rand_impl((int*)dio, size); break;
        //case AK_HALF: fill_tensor_host_rand_impl((short*)dio, size); break;
        case AK_FLOAT: fill_tensor_host_rand_impl((float*)dio, size); break;
        //case AK_DOUBLE: fill_tensor_host_rand_impl((double*)dio, size); break;
        default: LOGF("data type: %d is unsupported now", (int)type);
    }
}
template <typename Dtype>
void fill_tensor_host_rand_impl2(Dtype* dio, Dtype vstart, Dtype vend, long long size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1.f);
    for (long long i = 0; i < size; ++i) {
        Dtype random_num = static_cast<Dtype>(vstart + (vend - vstart) * dis(gen));
        dio[i] = random_num;
    }
}
template <>
void fill_tensor_rand(Tensor<CPU>& tensor, float vstart, float vend) {
    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        //case AK_UINT8: fill_tensor_host_rand_impl2((unsigned char*)dio, static_cast<unsigned char>(vstart), \
                                                   static_cast<unsigned char>(vend), size); break;
        case AK_INT8: fill_tensor_host_rand_impl2((char*)dio, static_cast<char>(vstart), static_cast<char>(vend), size); break;
        //case AK_INT16: fill_tensor_host_rand_impl2((short*)dio, static_cast<short>(vstart), static_cast<short>(vend), size); break;
        //case AK_UINT16: fill_tensor_host_rand_impl2((unsigned short*)dio, static_cast<unsigned short>(vstart), \
                                                    static_cast<unsigned short>(vend), size); break;
        //case AK_UINT32: fill_tensor_host_rand_impl2((unsigned int*)dio, static_cast<unsigned int>(vstart), \
                                                    static_cast<unsigned int>(vend), size); break;
        case AK_INT32: fill_tensor_host_rand_impl2((int*)dio, static_cast<int>(vstart), static_cast<int>(vend), size); break;
        //case AK_HALF: fill_tensor_host_rand_impl2((short*)dio, static_cast<short>(vstart), static_cast<short>(vend), size); break;
        case AK_FLOAT: fill_tensor_host_rand_impl2((float*)dio, static_cast<float>(vstart), static_cast<float>(vend), size); break;
        //case AK_DOUBLE: fill_tensor_host_rand_impl2((double*)dio, static_cast<double>(vstart), static_cast<double>(vend), size); break;
        default: LOGF("data type: %d is unsupported now", (int)type);
    }
}

template <typename Dtype>
void print_tensor_host_impl(const Dtype* din, long long size, int width);

template <>
void print_tensor_host_impl(const float* din, long long size, int width) {
    for (int i = 0; i < size; ++i) {
        printf("%.6f ", din[i]);
        if ((i + 1) % width == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

template <>
void print_tensor_host_impl(const int* din, long long size, int width) {
    for (int i = 0; i < size; ++i) {
        printf("%d ", din[i]);
        if ((i + 1) % width == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

template <>
void print_tensor_host_impl(const char* din, long long size, int width) {
    for (int i = 0; i < size; ++i) {
        printf("%d ", din[i]);
        if ((i + 1) % width == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

template <>
void print_tensor(const Tensor<CPU>& tensor) {
    printf("host tensor data size: %d\n", tensor.size());
    const void* data_ptr = tensor.data();
    long long size = tensor.valid_size();
    int width = tensor.width();
    DataType type = tensor.get_dtype();
    switch(type) {
        //case AK_UINT8: print_tensor_host_impl((const unsigned char*)data_ptr, size, width); break;
        case AK_INT8: print_tensor_host_impl((const char*)data_ptr, size, width); break;
        //case AK_UINT16: print_tensor_host_impl((const unsigned short*)data_ptr, size, width); break;
        //case AK_INT16: print_tensor_host_impl((const short*)data_ptr, size, width); break;
        //case AK_UINT32: print_tensor_host_impl((const unsigned int*)data_ptr, size, width); break;
        case AK_INT32: print_tensor_host_impl((const int*)data_ptr, size, width); break;
        case AK_FLOAT: print_tensor_host_impl((const float*)data_ptr, size, width); break;
        //case AK_DOUBLE: print_tensor_host_impl((const double*)data_ptr, size, width); break;
        default: LOGF("data type: %d is unsupported now", type);
    }
}

template <typename Dtype>
double tensor_mean_value_host_impl(const Dtype* din, long long size) {
    double sum = 0.0;
    for (long long i = 0; i < size; ++i) {
        sum += din[i];
    }
    return sum / size;
}

template <>
double tensor_mean(const Tensor<CPU>& tensor) {

    const void* data_ptr = tensor.data();
    long long size = tensor.valid_size();
    DataType type = tensor.get_dtype();
    switch (type) {
        //case AK_UINT8: return tensor_mean_value_host_impl((const unsigned char*)data_ptr, size);
        case AK_INT8: return tensor_mean_value_host_impl((const char*)data_ptr, size);
        //case AK_UINT16: return tensor_mean_value_host_impl((const unsigned short*)data_ptr, size);
        //case AK_INT16: return tensor_mean_value_host_impl((const short*)data_ptr, size);
        //case AK_UINT32: return tensor_mean_value_host_impl((const unsigned int*)data_ptr, size);
        case AK_INT32: return tensor_mean_value_host_impl((const int*)data_ptr, size);
        case AK_FLOAT: return tensor_mean_value_host_impl((const float*)data_ptr, size);
        //case AK_DOUBLE: return tensor_mean_value_host_impl((const double*)data_ptr, size);
        default: LOGF("data type: %d is unsupported now", (int)type);
    }
    return 0.0;
}

template <typename dtype>
void data_diff_kernel(const dtype* src1_truth, const dtype* src2, int size, double& max_ratio, double& max_diff) {

    const double eps = 1e-6f;
    max_diff = fabs(src1_truth[0] - src2[0]);
    max_ratio = fabs(max_diff) / (fabs(src1_truth[0]) + eps);
    for (int i = 1; i < size; ++i) {
        double diff = fabs(src1_truth[i] - src2[i]);
        double ratio = fabs(diff) / (fabsf(src1_truth[i]) + eps);
        if (max_ratio < ratio) {
            max_diff = diff;
            max_ratio = ratio;
        }
    }
}

template <>
void tensor_cmp_host(const Tensor<CPU>& src1_basic, const Tensor<CPU>& src2, double& max_ratio, double& max_diff) {

    const void* ptr1 = src1_basic.data();
    const void* ptr2 = src2.data();
    int size = src1_basic.valid_size();
    LCHECK_EQ(size, src2.valid_size(), "ERROR: wrong shape\n");
    LCHECK_EQ(src1_basic.get_dtype(), src2.get_dtype(), "ERROR: wrong data type\n");
    switch (src1_basic.get_dtype()) {
        case AK_FLOAT:
            data_diff_kernel(static_cast<const float*>(ptr1), \
                static_cast<const float*>(ptr2), size, max_ratio, max_diff);
            return;
        case AK_INT32:
            data_diff_kernel(static_cast<const int*>(ptr1), \
                static_cast<const int*>(ptr2), size, max_ratio, max_diff);
        case AK_INT8:
            data_diff_kernel(static_cast<const char*>(ptr1), \
                static_cast<const char*>(ptr2), size, max_ratio, max_diff);
            return;
        default: LOGF("data type: %d is unsupported now", (int)src1_basic.get_dtype());
    }
}

template <typename dtype>
void tensor_diff_kernel(const dtype* src1, const dtype* src2, dtype* dst, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = src1[i] - src2[i];
    }
}

template <>
void tensor_diff(const Tensor<CPU>& t1, const Tensor<CPU>& t2, Tensor<CPU>& tdiff) {
    const void* ptr1 = t1.data();
    const void* ptr2 = t2.data();
    void* ptr3 = tdiff.mutable_data();
    int size1 = t1.valid_size();
    int size2 = t2.valid_size();
    int size_out = tdiff.valid_size();
    LCHECK_EQ(size1, size2, "ERROR: wrong shape\n");
    LCHECK_EQ(size1, size_out, "ERROR: wrong shape\n");
    LCHECK_EQ(t1.get_dtype(), t2.get_dtype(), "ERROR: wrong data type\n");
    LCHECK_EQ(t1.get_dtype(), tdiff.get_dtype(), "ERROR: wrong data type\n");
    switch (t1.get_dtype()) {
        case AK_FLOAT:
            tensor_diff_kernel(static_cast<const float*>(ptr1), \
                static_cast<const float*>(ptr2), static_cast<float*>(ptr3), size1);
            return;
        case AK_INT32:
            tensor_diff_kernel(static_cast<const int*>(ptr1), \
                static_cast<const int*>(ptr2), static_cast<int*>(ptr3), size1);
        case AK_INT8:
            tensor_diff_kernel(static_cast<const char*>(ptr1), \
                static_cast<const char*>(ptr2), static_cast<char*>(ptr3), size1);
            return;
        default: LOGF("data type: %d is unsupported now", (int)t1.get_dtype());
    }
}

} //namespace lite

} //namespace saber

} //namespace anakin
