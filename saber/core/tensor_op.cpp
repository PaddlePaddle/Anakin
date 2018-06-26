#include "tensor_op.h"
#include "anakin_config.h"
#include <cstdlib>

namespace anakin {

namespace saber {

template <class Tensor_t>
void fill_tensor_host_const(Tensor_t& tensor, typename Tensor_t::Dtype value) {

    typedef typename Tensor_t::Dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());
    int size = tensor.size();

    for (int i = 0; i < size; ++i) {
        data_ptr[i] = value;
    }
}

template <class Tensor_t>
void fill_tensor_host_rand(Tensor_t& tensor) {
    typedef typename Tensor_t::Dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());

    for (int i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<Dtype>(rand());
    }
}

template <class Tensor_t>
void fill_tensor_host_seq(Tensor_t& tensor) {
    typedef typename Tensor_t::Dtype Dtype;
    Dtype* data_ptr = static_cast<Dtype*>(tensor.get_buf()->get_data_mutable());

    for (int i = 0; i < tensor.size(); ++i) {
        data_ptr[i] = static_cast<Dtype>(i);
    }
}

template <class Tensor_t>
void fill_tensor_host_rand(Tensor_t& tensor, typename Tensor_t::Dtype vstart, \
                           typename Tensor_t::Dtype vend) {
    typedef typename Tensor_t::Dtype Dtype;
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

template <class Tensor_t>
void print_tensor_host(Tensor_t& tensor) {

    typedef typename Tensor_t::Dtype Dtype;
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

#define FILL_TENSOR_HOST(target, type, layout) \
    template void fill_tensor_host_const<Tensor<target, type, layout>>\
        (Tensor<target, type, layout>& tensor, DataTrait<type>::dtype value); \
    template void fill_tensor_host_rand<Tensor<target, type, layout>>\
        (Tensor<target, type, layout>& tensor); \
    template void fill_tensor_host_rand<Tensor<target, type, layout>>\
        (Tensor<target, type, layout>& tensor, DataTrait<type>::dtype vstart, \
        DataTrait<type>::dtype vend); \
    template void print_tensor_host<Tensor<target, type, layout>>\
        (Tensor<target, type, layout>& tensor);\
    template void fill_tensor_host_seq<Tensor<target, type, layout>>\
        (Tensor<target, type, layout>& tensor);


FILL_TENSOR_HOST(X86, AK_FLOAT, NCHW);
FILL_TENSOR_HOST(X86, AK_FLOAT, NCHW_C16);
FILL_TENSOR_HOST(X86, AK_FLOAT, NCHW_C8);
FILL_TENSOR_HOST(X86, AK_FLOAT, NHWC);
FILL_TENSOR_HOST(X86, AK_FLOAT, NHW);
FILL_TENSOR_HOST(X86, AK_FLOAT, NW);
FILL_TENSOR_HOST(X86, AK_FLOAT, HW);
FILL_TENSOR_HOST(X86, AK_FLOAT, W);

FILL_TENSOR_HOST(X86, AK_INT8, NCHW);
FILL_TENSOR_HOST(X86, AK_INT8, NHWC);
FILL_TENSOR_HOST(X86, AK_INT8, NHW);
FILL_TENSOR_HOST(X86, AK_INT8, NW);
FILL_TENSOR_HOST(X86, AK_INT8, HW);
FILL_TENSOR_HOST(X86, AK_INT8, W);


template void tensor_cmp_host<float>(const float* src1, const float* src2, \
                                     int size, double& max_ratio, double& max_diff);
template void tensor_cmp_host<char>(const char* src1, const char* src2, int size, \
                                    double& max_ratio, double& max_diff);

template void fill_tensor_host_const<Tensor<X86, AK_INT8, NCHW_C4>>(Tensor<X86, AK_INT8, NCHW_C4>&
        tensor, char value);
template void fill_tensor_host_rand<Tensor<X86, AK_INT8, NCHW_C4>>(Tensor<X86, AK_INT8, NCHW_C4>&
        tensor);

template <>
void print_tensor_host<Tensor<X86, AK_INT8, NCHW_C4>>(Tensor<X86, AK_INT8, NCHW_C4>& tensor) {
    typedef typename Tensor<X86, AK_INT8, NCHW_C4>::Dtype Dtype;
    LOG(INFO) << "host tensor data:" << tensor.size();
    const Dtype* data_ptr = tensor.get_buf()->get_data();
    int size = tensor.size();

    for (int i = 0; i < size; ++i) {
        printf("%.2f ", static_cast<float>(data_ptr[i]));

        if ((i + 1) % (4 * tensor.width()) == 0) {
            printf("\n");
        }
    }

    printf("\n");
}
#ifdef USE_X86_PLACE
template <>
void reorder<Tensor<X86, AK_FLOAT, NCHW>, Tensor<X86, AK_FLOAT, NCHW_C16>>(Tensor<X86, AK_FLOAT, NCHW>& src, Tensor<X86, AK_FLOAT, NCHW_C16>& dst) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW_C16>::Dtype Dtype;
    int blksize = 16;
    const Dtype *src_data = src.data();
    Dtype *dst_data = dst.mutable_data();
    int width = src.width();
    int height = src.height();
    const int spatial_size = height * width;
    auto ker = [&](const Dtype *i, Dtype *o) {
        for (int w = 0; w < src.width(); ++w) {
            for (int c = 0; c < blksize; ++c) {
                const size_t nchw_off = c * spatial_size + w;
                o[w * blksize + c] = i[nchw_off];
            }
        }
    };
    int num = dst.num();
    int channel = src.channel();
    int channel_blk = channel / blksize;
#pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < num; ++n) {
        for (int C = 0; C < channel_blk; ++C) {
            for (int h = 0; h < height; ++h) {
                int input_offset = ((n * channel + blksize * C) * height + h) * width;
                int output_offset = ((n * channel_blk + C) * height + h) * blksize * width;
                auto i = &src_data[input_offset];
                auto o = &dst_data[output_offset];
                ker(i, o);
            }
        }
    }
    return;
}
template <>
void reorder<Tensor<X86, AK_FLOAT, NCHW_C16>, Tensor<X86, AK_FLOAT, NCHW>>(Tensor<X86, AK_FLOAT, NCHW_C16>& src, Tensor<X86, AK_FLOAT, NCHW>& dst) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW_C16>::Dtype Dtype;
    int blksize = 16;
    const Dtype *src_data = src.data();
    Dtype *dst_data = dst.mutable_data();
    int width = dst.width();
    int height = dst.height();
    const int spatial_size = height * width;
    auto ker = [&](const Dtype *i, Dtype *o) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < blksize; ++c) {
                const size_t nchw_off = c * spatial_size + w;
                o[nchw_off] = i[w * blksize + c];
            }
        }
    };
    int num = dst.num();
    int channel = dst.channel();
    int channel_blk = channel / blksize;
#pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < num; ++n) {
        for (int C = 0; C < channel_blk; ++C) {
            for (int h = 0; h < height; ++h) {
                int input_offset = ((n * channel_blk + C) * height + h) * blksize * width;
                int output_offset = ((n * channel + blksize * C) * height + h) * width;
                auto i = &src_data[input_offset];
                auto o = &dst_data[output_offset];
                ker(i, o);
            }
        }
    }
    return;
}
template <>
void reorder<Tensor<X86, AK_FLOAT, NCHW_C8>, Tensor<X86, AK_FLOAT, NCHW>>(Tensor<X86, AK_FLOAT, NCHW_C8>& src, Tensor<X86, AK_FLOAT, NCHW>& dst) {
    typedef typename Tensor<X86, AK_FLOAT, NCHW_C8>::Dtype Dtype;
    int blksize = 8;
    const Dtype *src_data = src.data();
    Dtype *dst_data = dst.mutable_data();
    int width = dst.width();
    int height = dst.height();
    const int spatial_size = height * width;
    auto ker = [&](const Dtype *i, Dtype *o) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < blksize; ++c) {
                const size_t nchw_off = c * spatial_size + w;
                o[nchw_off] = i[w * blksize + c];
            }
        }
    };
    int num = dst.num();
    int channel = dst.channel();
    int channel_blk = channel / blksize;
#pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < num; ++n) {
        for (int C = 0; C < channel_blk; ++C) {
            for (int h = 0; h < height; ++h) {
                int input_offset = ((n * channel_blk + C) * height + h) * blksize * width;
                int output_offset = ((n * channel + blksize * C) * height + h) * width;
                auto i = &src_data[input_offset];
                auto o = &dst_data[output_offset];
                ker(i, o);
            }
        }
    }
    return;
}
#endif

#ifdef USE_ARM_PLACE
FILL_TENSOR_HOST(ARM, AK_FLOAT, NCHW);
FILL_TENSOR_HOST(ARM, AK_FLOAT, NHWC);
FILL_TENSOR_HOST(ARM, AK_FLOAT, NHW);
FILL_TENSOR_HOST(ARM, AK_FLOAT, NW);
FILL_TENSOR_HOST(ARM, AK_FLOAT, HW);
FILL_TENSOR_HOST(ARM, AK_FLOAT, W);

FILL_TENSOR_HOST(ARM, AK_INT8, NCHW);
FILL_TENSOR_HOST(ARM, AK_INT8, NHWC);
FILL_TENSOR_HOST(ARM, AK_INT8, NHW);
FILL_TENSOR_HOST(ARM, AK_INT8, NW);
FILL_TENSOR_HOST(ARM, AK_INT8, HW);
FILL_TENSOR_HOST(ARM, AK_INT8, W);
#endif

#ifdef USE_CUDA

template<>
SaberStatus
DataTensorTransformHelper::convert_weights<Tensor<X86, AK_INT8, NCHW_C4>,
                          Tensor<X86, AK_FLOAT, NCHW> >(Tensor<X86, AK_INT8, NCHW_C4>& out_tensor,
                                  const Tensor<X86, AK_FLOAT, NCHW>& in_tensor,
Context<NV> ctx) {
    int input_channel = in_tensor.channel();
    int output_channel = out_tensor.shape()[1];
    //            LOG(INFO)<<"input_channel = "<<input_channel<<" output_channel = "<<output_channel;
    _vector_weight_scale.resize(input_channel);

    int weight_inner_dim = in_tensor.channel()
                           * in_tensor.height()
                           * in_tensor.width();
    const float* in_weight_data = in_tensor.data();

    for (int c = 0; c < input_channel; ++c) {
        float max_val = -1.f;

        for (int i = 0; i < weight_inner_dim; ++i) {
            float read_data = fabs(in_weight_data[i]);
            max_val = (read_data > max_val) ? read_data : max_val;
        }

        _vector_weight_scale[c] = max_val / 127.f;
        in_weight_data += weight_inner_dim;
        //                LOG(INFO)<<"max_val = "<<max_val<<" vector: "<<max_val / 127.f;
    }

    int o_num = out_tensor.num();
    int o_channel = output_channel;
    int o_height = out_tensor.height();
    int o_width = out_tensor.width();

    int out_n_stride = o_channel * o_height * o_width;
    int out_c_stride = o_height * o_width;
    int out_h_stride = o_width;

    Shape in_stride = in_tensor.get_stride();

    in_weight_data = in_tensor.data();
    char* out_weight_data = out_tensor.mutable_data();

    for (int idx = 0; idx < o_num * o_channel * o_height * o_width; ++idx) {

        int n = (idx / (out_n_stride)) % o_num;
        int in_offset = ((idx / (out_n_stride)) % o_num) * in_stride[0]
                        + ((idx / (out_c_stride)) % o_channel) * (in_stride[1] * 4)
                        + ((idx / (out_h_stride)) % o_height) * in_stride[2]
                        + (idx % o_width) * in_stride[3];

        int out_offset = ((idx / (out_n_stride)) % o_num) * out_n_stride
                         + ((idx / (out_c_stride)) % o_channel) * out_c_stride
                         + ((idx / (out_h_stride)) % o_height) * out_h_stride
                         + (idx % o_width);
        out_weight_data[out_offset * 4 + 0] = (char)(round(
                in_weight_data[in_offset + 0 * in_stride[1]] / _vector_weight_scale[n]));
        out_weight_data[out_offset * 4 + 1] = (char)(round(
                in_weight_data[in_offset + 1 * in_stride[1]] / _vector_weight_scale[n]));
        out_weight_data[out_offset * 4 + 2] = (char)(round(
                in_weight_data[in_offset + 2 * in_stride[1]] / _vector_weight_scale[n]));
        out_weight_data[out_offset * 4 + 3] = (char)(round(
                in_weight_data[in_offset + 3 * in_stride[1]] / _vector_weight_scale[n]));

    }

    return SaberSuccess;
}
template<>
SaberStatus
DataTensorTransformHelper::convert_bias<Tensor<X86, AK_FLOAT, NCHW>,
                          Tensor<X86, AK_FLOAT, NCHW> >(Tensor<X86, AK_FLOAT, NCHW>& out_tensor,
                                  const Tensor<X86, AK_FLOAT, NCHW>& in_tensor,
Context<NV> ctx) {
    unsigned long weight_size = _vector_weight_scale.size();
    unsigned long bias_size = in_tensor.size();
    CHECK_GT(_in_scale, 0);
    CHECK_GT(weight_size, 0);
    CHECK_EQ(bias_size, weight_size);

    const float* in_data = in_tensor.data();
    float* out_data = out_tensor.mutable_data();

    for (int i = 0; i < bias_size; ++i) {
        out_data[i] = in_data[i] / _in_scale / _vector_weight_scale[i];
    }

    return SaberSuccess;
}
#endif


#ifdef USE_BM

template<>
void fill_tensor_device_rand<Tensor<BM, AK_BM, NCHW>>(Tensor<BM, AK_BM, NCHW>& tensor, \
    typename Tensor<BM, AK_BM, NCHW>::API::stream_t stream) {

    float *host_mem_input = new float[tensor.size()];
    for (int i = 0; i < tensor.size(); ++i) {
        host_mem_input[i] = static_cast<float>(rand());
    }

    bm_device_mem_t* device_data_ptr = tensor.mutable_data();
    BMDNN_CHECK(bm_memcpy_s2d(get_bm_handle(), *device_data_ptr, bm_mem_from_system(host_mem_input)));

    delete [] host_mem_input;
}

void fill_tensor_device_rand(Tensor<BM, AK_BM, NCHW>& tensor, float vstart, \
    float vend, typename Tensor<BM, AK_BM, NCHW>::API::stream_t stream = NULL){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1.f);

    float *host_mem_input = new float[tensor.size()];
    for (int i = 0; i < tensor.size(); ++i) {
        float random_num = vstart + (vend - vstart) * dis(gen);
        host_mem_input[i] = random_num;
    }

    bm_device_mem_t* device_data_ptr = tensor.mutable_data();
    BMDNN_CHECK(bm_memcpy_s2d(get_bm_handle(), *device_data_ptr, bm_mem_from_system(host_mem_input)));

    delete [] host_mem_input;
}

void fill_tensor_device_const(Tensor<BM, AK_BM, NCHW>& tensor, float value, \
    typename Tensor<BM, AK_BM, NCHW>::API::stream_t stream = NULL){

    float *host_mem_input = new float[tensor.size()];
    for (int i = 0; i < tensor.size(); ++i) {
        host_mem_input[i] = value;
    }

    bm_device_mem_t* device_data_ptr = tensor.mutable_data();
    BMDNN_CHECK(bm_memcpy_s2d(get_bm_handle(), *device_data_ptr, bm_mem_from_system(host_mem_input)));

    delete [] host_mem_input;
}

#endif

} //namespace saber

} //namespace anakin
