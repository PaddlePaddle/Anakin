/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */


#ifndef SABER_FUNCS_IMPL_X86_X86_UTILS_H
#define SABER_FUNCS_IMPL_X86_X86_UTILS_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <type_traits>
#include "saber/funcs/impl/x86/anakin_thread.h"
#include "saber/core/common.h"
#include "saber/core/tensor.h"
#include "saber/funcs/saber_util.h"
#include "calibrate.h"
namespace anakin {
namespace saber {

#define UNUSED(x) ((void)x)
#define MAYBE_UNUSED(x) UNUSED(x)

#ifdef _WIN32
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

namespace utils {

template<typename opTensor>
static inline void try_expand_clean_tensor(opTensor& tensor, anakin::saber::Shape shape) {
    if (try_expand_tensor(tensor, shape)) {
        memset(tensor.mutable_data(), 0, tensor.valid_size()* type_length(tensor.get_dtype()));
    };
}

class ScaleUtils {
public:
    static void cvt_int32_fp32(int* data, float* scale, int m, int n) {
        float* out_data = (float*)(data);

        for (int row = 0; row < m; row++) {
            int offset_row = row * n;

            for (int col = 0; col < n; col++) {
                out_data[offset_row + col] = data[offset_row + col] * scale[col];
            }
        }
    }
    static void cvt_int32_fp32(int* data, std::vector<float>& scale, int m, int n) {
        CHECK_EQ(scale.size(), n);
        float* out_data = (float*)(data);

        for (int row = 0; row < m; row++) {
            int offset_row = row * n;

            for (int col = 0; col < n; col++) {
                out_data[offset_row + col] = data[offset_row + col] * scale[col];
            }
        }
    }
    static void scale_fp32_fp32(Tensor<X86>& data_tensor,float scale){
        CHECK_EQ(data_tensor.get_dtype(), AK_FLOAT) << "input must be fp32";
        size_t length=data_tensor.valid_size();
        float* in_data = static_cast<float*>(data_tensor.data());
        for (size_t i = 0; i < length; i++){
            in_data[i] = in_data[i]*scale;
        }
    }

    static void scale_uint8_fp32(Tensor<X86>& out_tensor, const Tensor<X86>& in_tensor){
        CHECK_EQ(in_tensor.get_dtype(), AK_UINT8) << "input must be fp32";
        CHECK_EQ(in_tensor.get_scale().size(),1);
        CHECK_EQ(out_tensor.get_dtype(), AK_FLOAT) << "input must be fp32";
        size_t length = in_tensor.valid_size();
        uint8_t* in_data = static_cast<uint8_t *>(in_tensor.data());
        float* out_data = static_cast<float *>(out_tensor.data());
        float scale=in_tensor.get_scale()[0]*(127.f/255.f);
        for (size_t i = 0; i < length; i++){
            out_data[i] = (float)in_data[i] * scale;
        }
    }

    static void scale_fp32_int8_without_scale(Tensor<X86>& out_tensor, const Tensor<X86>& in_tensor) {
        CHECK_EQ(out_tensor.get_dtype(), AK_INT8) << "output must be int8";
        CHECK_EQ(in_tensor.get_dtype(), AK_FLOAT) << "input must be fp32";
        CHECK_EQ(in_tensor.get_scale().size(), 0) << "input no scale is perfer";
        size_t length = in_tensor.valid_shape().count();
        const float* in_data = static_cast<float*>(in_tensor.data());
        float* out_data = static_cast<float*>(out_tensor.data());
        float max = -1e10;

        for (size_t i = 0; i < length; i++) {
            const float temp = fabsf(in_data[i]);
            max = max > temp ? max : temp;
        }

        float scale_value = 127.f / max;

        for (size_t i = 0; i < length; i++) {
            out_data[i] = static_cast<char>(roundf((float)in_data[i] * scale_value));
        }

        out_tensor.set_scale({1.f / scale_value});
    }

    static void get_tensor_scale(const Tensor<X86>& tensor) {
        CHECK_EQ(tensor.get_dtype(), AK_FLOAT);
        size_t length = tensor.valid_shape().count();
        float* data = static_cast<float*>(tensor.data());
        float max = -1e10;

        for (size_t i = 0; i < length; i++) {
            const float temp = fabsf(data[i]);
            max = max > temp ? max : temp;
        }
        LOG(FATAL) << "not impl";
    }
    static float get_fp32_max(const float* input, size_t size) {
        float max = -1e10;
        for (size_t i = 0; i < size; i++) {
            const float temp = fabsf(input[i]);
            max = max > temp ? max : temp;
        }
        return max;
    }

    static SaberStatus get_tensor_scale(std::vector<float>& vector_scale,
                                        const Tensor<X86>& tensor, const int axis, bool reverse = false) {

        int out_dims = tensor.valid_shape()[axis];

        long long inner_dim = tensor.count_valid(axis + 1, tensor.dims());
        const float* in_data = (const float*)(tensor.data());
        const float eps = 1e-5;

        if (reverse == false) {
            vector_scale.resize(out_dims);

            for (int c = 0; c < out_dims; ++c) {
                float max_val = -1e20;

                for (int i = 0; i < inner_dim; ++i) {
                    float read_data = fabs(in_data[i]);
                    max_val = (read_data > max_val) ? read_data : max_val;
                }

                vector_scale[c] = (max_val) / 127.f;

                in_data += inner_dim;
            }
        } else {
            vector_scale.resize(inner_dim);

            for (int i = 0; i < inner_dim; ++i) {
                float max_val = -1e20;

                for (int c = 0; c < out_dims; ++c) {
                    float read_data = fabs(in_data[c * inner_dim + i]);
                    max_val = (read_data > max_val) ? read_data : max_val;
                }

                vector_scale[i] = max_val / 127.f;
            }
        }
        return SaberSuccess;
    }

    static SaberStatus get_tensor_scale_u8(std::vector<float>& vector_scale,
                                           const Tensor<X86>& tensor, const int axis, bool reverse = false) {
        LOG(FATAL) << "not impl";
        return SaberSuccess;
    }

    static SaberStatus scale_fc_weights_to_nchw_host(Tensor<X86>& out_tensor,
            const Tensor<X86>& in_tensor) {
        CHECK_EQ(in_tensor.get_dtype(), AK_FLOAT) << "input must be ak_float";
        CHECK_EQ(out_tensor.get_dtype(), AK_INT8) << "output must be int 8";
        std::vector<float> vector_weight_scale;
        get_tensor_scale(vector_weight_scale, in_tensor, 2);
        int oc = out_tensor.height();
        int other = out_tensor.width();
        const float* in_weight_data = (const float*)in_tensor.data();
        char* out_weight_data = (char*)out_tensor.mutable_data();

        for (int idx = 0; idx < oc * other; ++idx) {

            int n = idx / other;

            out_weight_data[idx] = static_cast<char>(in_weight_data[idx] / vector_weight_scale[n]);

        }

        out_tensor.set_scale(vector_weight_scale);

        return SaberSuccess;
    }

    static SaberStatus scale_fc_weights_to_nchw_host_u8(Tensor<X86>& out_tensor,
            const Tensor<X86>& in_tensor) {
        CHECK_EQ(in_tensor.get_dtype(), AK_FLOAT) << "input must be ak_float";
        CHECK_EQ(out_tensor.get_dtype(), AK_UINT8) << "output must be int 8";
        std::vector<float> vector_weight_scale;
        get_tensor_scale(vector_weight_scale, in_tensor, 2);
        int oc = out_tensor.height();
        int other = out_tensor.width();
        const float* in_weight_data = (const float*)in_tensor.data();
        char* out_weight_data = (char*)out_tensor.mutable_data();

        for (int idx = 0; idx < oc * other; ++idx) {

            int n = idx / other;

            out_weight_data[idx] = static_cast<char>(in_weight_data[idx] / vector_weight_scale[n]);

        }

        out_tensor.set_scale(vector_weight_scale);

        return SaberSuccess;
    }

    static SaberStatus scale_gemm_xw_weights_to_nchw_host(Tensor<X86>& out_tensor,
            const Tensor<X86>& in_tensor, bool is_ic_oc = true) {
        CHECK_EQ(in_tensor.get_dtype(), AK_FLOAT) << "input must be ak_float";
        CHECK_EQ(out_tensor.get_dtype(), AK_INT8) << "output must be int 8";
        std::vector<float> vector_weight_scale;
        get_tensor_scale(vector_weight_scale, in_tensor, 2, is_ic_oc);
        int other = in_tensor.width();
        int k = in_tensor.height();
        if (!is_ic_oc){
            k = in_tensor.width();
            other = in_tensor.height();
        }
        CHECK_EQ(vector_weight_scale.size(),other);

        const float* in_weight_data = (const float*)in_tensor.data();
        char* out_weight_data = (char*)out_tensor.mutable_data();

        if (is_ic_oc) {
            for (int idx = 0; idx < k * other; ++idx) {

                int n = idx % other;

                out_weight_data[idx] = static_cast<char>(in_weight_data[idx] / vector_weight_scale[n]);

            }
        }else{
            for (int idx = 0; idx < k * other; ++idx) {

                int n = idx / k;

                out_weight_data[idx] = static_cast<char>(in_weight_data[idx] / vector_weight_scale[n]);

            }
        }

        out_tensor.set_scale(vector_weight_scale);

        return SaberSuccess;
    }

    static SaberStatus scale_bias_fp32_int32(Tensor<X86>& out_tensor,
            const Tensor<X86>& in_tensor) {
        CHECK_EQ(in_tensor.get_dtype(), AK_FLOAT) << "input must be ak_float";
        CHECK_EQ(out_tensor.get_dtype(), AK_INT32) << "output must be int 8";
        CHECK_EQ(out_tensor.get_scale().size(),
                 out_tensor.valid_size()) << "bias scale size must equal bias size";
        std::vector<float> vector_bias_scale = out_tensor.get_scale();
        const float* in_data = static_cast<const float*>(in_tensor.data());
        int* out_data = static_cast<int*>(out_tensor.mutable_data());

        for (int idx = 0; idx < in_tensor.valid_size(); ++idx) {
            out_data[idx] = static_cast<int>(in_data[idx] / vector_bias_scale[idx]);
        }

        return SaberSuccess;
    }

    static SaberStatus scale_conv_weights_to_nchw_host(Tensor<X86>& out_tensor,
            const Tensor<X86>& in_tensor) {
        CHECK_EQ(in_tensor.get_dtype(), AK_FLOAT) << "input must be ak_float";
        CHECK_EQ(out_tensor.get_dtype(), AK_INT8) << "output must be int 8";
        std::vector<float> vector_weight_scale;
        get_tensor_scale(vector_weight_scale, in_tensor, 0);
        int o_num = out_tensor.num();
        int o_channel = out_tensor.channel();
        int o_height = out_tensor.height();
        int o_width = out_tensor.width();

        int out_n_stride = o_channel * o_height * o_width;
        int out_c_stride = o_height * o_width;
        int out_h_stride = o_width;

        Shape in_stride = in_tensor.get_stride();
        const float* in_weight_data = (const float*)in_tensor.data();
        char* out_weight_data = (char*)out_tensor.mutable_data();

        for (int idx = 0; idx < o_num * o_channel * o_height * o_width; ++idx) {

            int n = (idx / (out_n_stride)) % o_num;

            out_weight_data[idx] = static_cast<char>(in_weight_data[idx] / vector_weight_scale[n]);

        }

        out_tensor.set_scale(vector_weight_scale);

        return SaberSuccess;
    }

    static inline char secur_cast2char(float value) {
        float temp = roundf(value);
        int temp_int = (int)temp;
        temp_int = temp_int > 127 ? 127 : temp_int;
        temp_int = temp_int < -128 ? -128 : temp_int;
        return (char)temp_int;
    }
    static void scale_fp32_int8(Tensor<X86>& out_tensor, const Tensor<X86>& in_tensor) {
        CHECK_EQ(out_tensor.get_dtype(), AK_INT8) << "output must be int8";
        CHECK_EQ(in_tensor.get_dtype(), AK_FLOAT) << "input must be fp32";
        auto scale_vec = in_tensor.get_scale();
        CHECK_EQ(scale_vec.size(), 1) << "scale must = 1";
        float scale_value = 1.f / in_tensor.get_scale_data()[0];
        int size = in_tensor.valid_size();
        char* out_ptr = static_cast<char*>(out_tensor.mutable_data());
        const float* in_ptr = static_cast<const float*>(in_tensor.data());

        for (int i = 0; i < size; i++) {

            out_ptr[i] = secur_cast2char(in_ptr[i] * scale_value);
        }

    }

    static void scale_fp32_int8(Tensor<X86>& out_tensor , const float* input, size_t size){
        CHECK_EQ(out_tensor.get_dtype(), AK_INT8) << "output must be int8";
        float t_max=get_fp32_max(input,size);
        float scale_value=127.f/t_max;
        char* out_ptr = static_cast<char*>(out_tensor.mutable_data());
        for (int i = 0; i < size; i++) {
            out_ptr[i] = secur_cast2char(input[i] * scale_value);
        }
        out_tensor.set_scale({1.f/scale_value});
    }

    static void scale_fp32_uint8(Tensor<X86>& out_tensor, Tensor<X86>& in_tensor) {
        CHECK_EQ(out_tensor.get_dtype(), AK_UINT8) << "output must be int8";
        CHECK_EQ(in_tensor.get_dtype(), AK_FLOAT) << "input must be fp32";
        auto scale_vec = in_tensor.get_scale();
        CHECK_EQ(scale_vec.size(), 1) << "scale must = 1";
        float scale_value = 1.f / (in_tensor.get_scale_data()[0]*(127.f/255.f));
        int size = in_tensor.valid_size();
        uint8_t * out_ptr = static_cast<uint8_t*>(out_tensor.mutable_data());
        const float* in_ptr = static_cast<const float*>(in_tensor.data());

        for (int i = 0; i < size; i++) {
            out_ptr[i] = static_cast<unsigned char>(in_ptr[i] * scale_value);
        }
    }

//    static void scale_int8_fp32(Tensor<X86>& out_tensor, Tensor<X86>& in_tensor) {
//        CHECK_EQ(out_tensor.get_dtype(), AK_FLOAT) << "output must be fp32";
//        CHECK_EQ(in_tensor.get_dtype(), AK_INT8) << "input must be int8";
//        float scale_value = 1.f / in_tensor.get_scale()[0];
//        int size = in_tensor.valid_size();
//        char* out_ptr = static_cast<char*>(out_tensor.mutable_data());
//        const float* in_ptr = static_cast<const float*>(in_tensor.data());
//
//        for (int i = 0; i < size; i++) {
//            out_ptr[i] = static_cast<char>(roundf(in_ptr[i] * scale_value));
//        }
//    }
};

template <typename HostType>
static void reorder_nchwc_nchw(Tensor<HostType>& input,
                               Tensor<HostType>& output) {

    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";

    Shape shape = output.valid_shape();
    int n_value = shape[0];
    int c_value = shape[1];
    int h_value = shape[2];
    int w_value = shape[3];
    Shape shape_input = input.valid_shape();
    int aligned_length = shape_input.get_layout_aligned_length();
    CHECK_GT(aligned_length, 0) << "input aligned should > 0";
    int c_round_divk = shape_input[1];

    c_round_divk = (shape_input.channel() + aligned_length - 1) / aligned_length;

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
    #pragma omp parallel for collapse(4) schedule(static)

    for (int n = 0; n < n_value; ++n) {
        for (int c = 0; c < c_value; ++c) {
            for (int h = 0; h < h_value; ++h) {
                //#pragma ivdep
                for (int w = 0; w < w_value; ++w) {
                    int round_c = c / aligned_length;
                    int remainder_c = c % aligned_length;
                    int input_idx = n * c_round_divk * h_value * w_value * aligned_length + round_c * h_value *
                                    w_value * aligned_length +
                                    h * w_value * aligned_length + w * aligned_length + remainder_c;
                    int output_idx = n * c_value * h_value * w_value + c * h_value * w_value  +
                                     h * w_value  + w ;

                    *(output_ptr + output_idx) = input_ptr[input_idx];
                }
            }
        }
    }

}


} // namespace utils


inline void* zmalloc(size_t size, int alignment) {
    void* ptr = nullptr;

#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
    int rc = ptr ? 0 : -1;
#else
#ifdef USE_SGX
    size_t d = alignment / sizeof(void*);
    size_t r = alignment % sizeof(void*);
    // posix_memalign requires alignment to be a power of two multiple
    // of sizeof(void *). Add this check before calling memalign.
    if (r != 0 || d == 0 || (d & (d - size_t(1))) != 0) {
        return nullptr;
    }
    ptr = memalign(alignment, size);
    int rc = ptr == nullptr;
#else
    int rc = ::posix_memalign(&ptr, alignment, size);
#endif
#endif

    return (rc == 0) ? ptr : nullptr;
}

inline void zfree(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    ::free(p);
#endif
}

//struct c_compatible {
//    enum { default_alignment = 4096 };
//
//    static void* operator new (size_t sz) {
//        return zmalloc(sz, default_alignment);
//    }
//
//    static void* operator new (size_t sz, void* p) {
//        UNUSED(sz);
//        return p;
//    }
//
//    static void* operator new[](size_t sz) {
//        return zmalloc(sz, default_alignment);
//    }
//
//    static void operator delete (void* p) {
//        zfree(p);
//    }
//
//    static void operator delete[](void* p) {
//        zfree(p);
//    }
//};

inline void yield_thread() { }

// reorder weight layout from NCHW(oc, ic, kh, kw) to OIhw16o16i
inline void weight_reorder_OIhw16o16i(Tensor<X86>& input,
                                      Tensor<X86>& output) {
    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    CHECK_EQ(output.get_dtype(), AK_FLOAT) << "only support float type";
    Shape shape = input.valid_shape();
    int oc_value = shape[0], ic_value = shape[1], kh_value = shape[2], kw_value = shape[3];

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
    #pragma omp parallel for collapse(6) schedule(static)

    for (int oc_idx = 0; oc_idx < oc_value / 16; ++oc_idx) {
        for (int ic_idx = 0; ic_idx < ic_value / 16; ++ic_idx) {
            for (int kh = 0; kh < kh_value; ++kh) {
                for (int kw = 0; kw < kw_value; ++kw) {
                    for (int oc = 0; oc < 16; ++oc) {
                        for (int ic = 0; ic < 16; ++ic) {
                            int input_idx = (oc_idx * 16 + oc) * ic_value * kh_value * kw_value +
                                            (ic_idx * 16 + ic) * kh_value * kw_value +
                                            kh * kw_value + kw;
                            int output_idx = oc_idx * ic_value / 16 * kh_value * kw_value * 16 * 16 +
                                             ic_idx * kh_value * kw_value * 16 * 16 +
                                             kh * kw_value * 16 * 16 +
                                             kw * 16 * 16 + oc * 16 + ic;

                            *(output_ptr + output_idx) = *(input_ptr + input_idx);
                        }
                    }
                }
            }
        }
    }
}

// reorder weight layout from NCHW(oc, ic, kh, kw) to OIhw16i16o
inline void weight_reorder_OIhw16i16o(Tensor<X86>& input,
                                      Tensor<X86>& output) {
    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    CHECK_EQ(output.get_dtype(), AK_FLOAT) << "only support float type";
    Shape shape = input.valid_shape();
    int oc_value = shape[0], ic_value = shape[1], kh_value = shape[2], kw_value = shape[3];

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
    #pragma omp parallel for collapse(6) schedule(static)

    for (int oc_idx = 0; oc_idx < oc_value / 16; ++oc_idx) {
        for (int ic_idx = 0; ic_idx < ic_value / 16; ++ic_idx) {
            for (int kh = 0; kh < kh_value; ++kh) {
                for (int kw = 0; kw < kw_value; ++kw) {
                    for (int ic = 0; ic < 16; ++ic) {
                        for (int oc = 0; oc < 16; ++oc) {
                            int input_idx = (oc_idx * 16 + oc) * ic_value * kh_value * kw_value +
                                            (ic_idx * 16 + ic) * kh_value * kw_value +
                                            kh * kw_value + kw;
                            int output_idx = oc_idx * ic_value / 16 * kh_value * kw_value * 16 * 16 +
                                             ic_idx * kh_value * kw_value * 16 * 16 +
                                             kh * kw_value * 16 * 16 +
                                             kw * 16 * 16 + ic * 16 + oc;

                            *(output_ptr + output_idx) = *(input_ptr + input_idx);
                        }
                    }
                }
            }
        }
    }
}

inline void weight_reorder_OIhw8o8i(Tensor<X86>& input,
                                    Tensor<X86>& output) {
    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    CHECK_EQ(output.get_dtype(), AK_FLOAT) << "only support float type";
    Shape shape = input.valid_shape();
    int oc_value = shape[0], ic_value = shape[1], kh_value = shape[2], kw_value = shape[3];

    Shape new_shape({utils::round_up(oc_value, 8), utils::round_up(ic_value, 8), kh_value, kw_value},
                    Layout_NCHW);

    if ((oc_value % 8 != 0) || (ic_value % 8 != 0)) {
        output.re_alloc(new_shape, AK_FLOAT);
    }

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
    #pragma omp parallel for collapse(6) schedule(static)

    for (int oc_idx = 0; oc_idx < new_shape[0] / 8; ++oc_idx) {
        for (int ic_idx = 0; ic_idx < new_shape[1] / 8; ++ic_idx) {
            for (int kh = 0; kh < kh_value; ++kh) {
                for (int kw = 0; kw < kw_value; ++kw) {
                    for (int oc = 0; oc < 8; ++oc) {
                        for (int ic = 0; ic < 8; ++ic) {
                            int input_idx = (oc_idx * 8 + oc) * ic_value * kh_value * kw_value +
                                            (ic_idx * 8 + ic) * kh_value * kw_value +
                                            kh * kw_value + kw;
                            int output_idx = oc_idx * new_shape[1] / 8 * kh_value * kw_value * 8 * 8 +
                                             ic_idx * kh_value * kw_value * 8 * 8 +
                                             kh * kw_value * 8 * 8 +
                                             kw * 8 * 8 + oc * 8 + ic;

                            *(output_ptr + output_idx) = ((oc_idx * 8 + oc) < oc_value && (ic_idx * 8 + ic) < ic_value)
                                                         ?  *(input_ptr + input_idx) : 0;
                        }
                    }
                }
            }
        }
    }
}

inline void weight_reorder_OIhw8o8i_ak(Tensor<X86>& input,
                                       Tensor<X86>& output) {
    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    CHECK_EQ(output.get_dtype(), AK_FLOAT) << "only support float type";
    Shape shape = input.valid_shape();
    int oc_value = shape[1], ic_value = shape[0], kh_value = shape[2], kw_value = shape[3];

    Shape new_shape({utils::round_up(oc_value, 8), utils::round_up(ic_value, 8), kh_value, kw_value},
                    Layout_NCHW);

    if ((oc_value % 8 != 0) || (ic_value % 8 != 0)) {
        output.re_alloc(new_shape, AK_FLOAT);
    }

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
    #pragma omp parallel for collapse(6) schedule(static)

    for (int oc_idx = 0; oc_idx < new_shape[0] / 8; ++oc_idx) {
        for (int ic_idx = 0; ic_idx < new_shape[1] / 8; ++ic_idx) {
            for (int kh = 0; kh < kh_value; ++kh) {
                for (int kw = 0; kw < kw_value; ++kw) {
                    for (int oc = 0; oc < 8; ++oc) {
                        for (int ic = 0; ic < 8; ++ic) {
                            int input_idx = (ic_idx * 8 + ic) * ic_value * kh_value * kw_value +
                                            (oc_idx * 8 + oc) * kh_value * kw_value +
                                            kh * kw_value + kw;
                            int output_idx = oc_idx * new_shape[1] / 8 * kh_value * kw_value * 8 * 8 +
                                             ic_idx * kh_value * kw_value * 8 * 8 +
                                             kh * kw_value * 8 * 8 +
                                             kw * 8 * 8 + oc * 8 + ic;

                            *(output_ptr + output_idx) = ((oc_idx * 8 + oc) < oc_value && (ic_idx * 8 + ic) < ic_value)
                                                         ?  *(input_ptr + input_idx) : 0;
                        }
                    }
                }
            }
        }
    }
}

// reorder weight layout from NCHW(oc, ic, kh, kw) to OIhw8i8o
inline void weight_reorder_OIhw8i8o(Tensor<X86>& input,
                                    Tensor<X86>& output) {
    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    CHECK_EQ(output.get_dtype(), AK_FLOAT) << "only support float type";
    Shape shape = input.valid_shape();
    int oc_value = shape[0], ic_value = shape[1], kh_value = shape[2], kw_value = shape[3];

    Shape new_shape({utils::round_up(oc_value, 8), utils::round_up(ic_value, 8), kh_value, kw_value},
                    Layout_NCHW);

    if ((oc_value % 8 != 0) || (ic_value % 8 != 0)) {
        output.re_alloc(new_shape, AK_FLOAT);
    }

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
    #pragma omp parallel for collapse(6) schedule(static)

    for (int oc_idx = 0; oc_idx < new_shape[0] / 8; ++oc_idx) {
        for (int ic_idx = 0; ic_idx < new_shape[1] / 8; ++ic_idx) {
            for (int kh = 0; kh < kh_value; ++kh) {
                for (int kw = 0; kw < kw_value; ++kw) {
                    for (int ic = 0; ic < 8; ++ic) {
                        for (int oc = 0; oc < 8; ++oc) {
                            int input_idx = (oc_idx * 8 + oc) * ic_value * kh_value * kw_value +
                                            (ic_idx * 8 + ic) * kh_value * kw_value +
                                            kh * kw_value + kw;
                            int output_idx = oc_idx * new_shape[1] / 8 * kh_value * kw_value * 8 * 8 +
                                             ic_idx * kh_value * kw_value * 8 * 8 +
                                             kh * kw_value * 8 * 8 +
                                             kw * 8 * 8 + ic * 8 + oc;

                            *(output_ptr + output_idx) = ((oc_idx * 8 + oc) < oc_value && (ic_idx * 8 + ic) < ic_value)
                                                         ?  *(input_ptr + input_idx) : 0;
                        }
                    }
                }
            }
        }
    }
}

// reorder weight layout from NCHW(oc, ic, kh, kw) to OIhw8i8o
inline void weight_reorder_nchw2nchw8o8i(Tensor<X86>& input,
        Tensor<X86>& output) {
    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    CHECK_EQ(output.get_dtype(), AK_FLOAT) << "only support float type";
    Shape shape = input.valid_shape();
    int oc_value = shape[0], ic_value = shape[1], kh_value = shape[2], kw_value = shape[3];

    Shape new_shape({utils::round_up(oc_value, 8), utils::round_up(ic_value, 8), kh_value, kw_value},
                    Layout_NCHW);

    if ((oc_value % 8 != 0) || (ic_value % 8 != 0)) {
        output.re_alloc(new_shape, AK_FLOAT);
    }

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
    #pragma omp parallel for collapse(6) schedule(static)

    for (int oc_idx = 0; oc_idx < new_shape[0] / 8; ++oc_idx) {
        for (int ic_idx = 0; ic_idx < new_shape[1] / 8; ++ic_idx) {
            for (int kh = 0; kh < kh_value; ++kh) {
                for (int kw = 0; kw < kw_value; ++kw) {
                    for (int oc = 0; oc < 8; ++oc) {
                        for (int ic = 0; ic < 8; ++ic) {
                            int input_idx = (oc_idx * 8 + oc) * ic_value * kh_value * kw_value +
                                            (ic_idx * 8 + ic) * kh_value * kw_value +
                                            kh * kw_value + kw;
                            int output_idx = oc_idx * new_shape[1] / 8 * kh_value * kw_value * 8 * 8 +
                                             ic_idx * kh_value * kw_value * 8 * 8 +
                                             kh * kw_value * 8 * 8 +
                                             kw * 8 * 8 + oc * 8 + ic;

                            *(output_ptr + output_idx) = ((oc_idx * 8 + oc) < oc_value && (ic_idx * 8 + ic) < ic_value)
                                                         ?  *(input_ptr + input_idx) : 0;
                        }
                    }
                }
            }
        }
    }
}


// reorder weight layout from NCHW(oc, ic, kh, kw) to OIhw4i16o4i
inline void weight_reorder_OIhw4i16o4i(Tensor<X86>& input, Tensor<X86>& output,
                                       const std::vector<float>& scale) {
    CHECK_EQ(input.get_dtype(), AK_INT8) << "only support int8 type";
    CHECK_EQ(output.get_dtype(), AK_INT8) << "only support int8 type";
    Shape shape = input.shape();
    int oc_value = shape[0], ic_value = shape[1], kh_value = shape[2], kw_value = shape[3];

    if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_INT8) {
        char* output_ptr = static_cast<char*>(output.mutable_data());
        const float* input_ptr = static_cast<const float*>(input.data());

        #pragma omp parallel for collapse(7) schedule(static)

        for (int oc_idx = 0; oc_idx < oc_value / 16; ++oc_idx) {
            for (int ic_idx = 0; ic_idx < ic_value / 16; ++ic_idx) {
                for (int kh = 0; kh < kh_value; ++kh) {
                    for (int kw = 0; kw < kw_value; ++kw) {
                        for (int ic = 0; ic < 4; ++ic) {
                            for (int oc = 0; oc < 16; ++oc) {
                                for (int icc = 0; icc < 4; ++icc) {
                                    int input_idx = (oc_idx * 16 + oc) * ic_value * kh_value * kw_value +
                                                    (ic_idx * 16 + ic * 4 + icc) * kh_value * kw_value +
                                                    kh * kw_value + kw;
                                    int output_idx = oc_idx * ic_value / 16 * kh_value * kw_value * 16 * 16 +
                                                     ic_idx * kh_value * kw_value * 16 * 16 +
                                                     kh * kw_value * 16 * 16 +
                                                     kw * 16 * 16 +
                                                     ic * 16 * 4 +
                                                     oc * 4 +
                                                     icc;
                                    float scale_v = scale[oc_idx * 16 + oc];
                                    *(output_ptr + output_idx) = (*(input_ptr + input_idx)) * scale_v;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if (input.get_dtype() == AK_INT8 && output.get_dtype() == AK_INT8) {
        char* output_ptr = static_cast<char*>(output.mutable_data());
        const char* input_ptr = static_cast<const char*>(input.data());

        #pragma omp parallel for collapse(7) schedule(static)

        for (int oc_idx = 0; oc_idx < oc_value / 16; ++oc_idx) {
            for (int ic_idx = 0; ic_idx < ic_value / 16; ++ic_idx) {
                for (int kh = 0; kh < kh_value; ++kh) {
                    for (int kw = 0; kw < kw_value; ++kw) {
                        for (int ic = 0; ic < 4; ++ic) {
                            for (int oc = 0; oc < 16; ++oc) {
                                for (int icc = 0; icc < 4; ++icc) {
                                    int input_idx = (oc_idx * 16 + oc) * ic_value * kh_value * kw_value +
                                                    (ic_idx * 16 + ic * 4 + icc) * kh_value * kw_value +
                                                    kh * kw_value + kw;
                                    int output_idx = oc_idx * ic_value / 16 * kh_value * kw_value * 16 * 16 +
                                                     ic_idx * kh_value * kw_value * 16 * 16 +
                                                     kh * kw_value * 16 * 16 +
                                                     kw * 16 * 16 +
                                                     ic * 16 * 4 +
                                                     oc * 4 +
                                                     icc;
                                    *(output_ptr + output_idx) = (*(input_ptr + input_idx));
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        ABORT_S() << "error: not support weight reorder!";
    }
}

// reorder weight layout from NCHW(oc, ic, kh, kw) to OIhwi16o
inline void weight_reorder_OIhwi16o(Tensor<X86>& input,
                                    Tensor<X86>& output) {

    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    CHECK_EQ(output.get_dtype(), AK_FLOAT) << "only support float type";
    Shape shape = input.shape();

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
    #pragma omp parallel for collapse(5) schedule(static)

    for (int oc_idx = 0; oc_idx < shape[0] / 16; ++oc_idx) {
        for (int kh = 0; kh < shape[2]; ++kh) {
            for (int kw = 0; kw < shape[3]; ++kw) {
                for (int ic = 0; ic < shape[1]; ++ic) {
                    for (int oc = 0; oc < 16; ++oc) {
                        int input_idx = (oc_idx * 16 + oc) * shape[1] * shape[2] * shape[3] +
                                        ic * shape[2] * shape[3] +
                                        kh * shape[3] + kw;
                        int output_idx = oc_idx * shape[2] * shape[3] * shape[1] * 16 +
                                         kh * shape[3] * shape[1] * 16 +
                                         kw * shape[1] * 16 +
                                         ic * 16 + oc;

                        *(output_ptr + output_idx) = *(input_ptr + input_idx);
                    }
                }
            }
        }
    }
}

inline void weight_reorder_OIhwi8o(Tensor<X86>& input,
                                   Tensor<X86>& output) {
    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    CHECK_EQ(output.get_dtype(), AK_FLOAT) << "only support float type";

    Shape shape = input.shape();
    int oc_value = shape[0], ic_value = shape[1], kh_value = shape[2], kw_value = shape[3];

    Shape new_shape({utils::round_up(oc_value, 8) / 8, ic_value, kh_value, kw_value, 8},
                    Layout_NCHW_C8);

    if (oc_value % 8 != 0) {
        output.re_alloc(new_shape, AK_FLOAT);
    }

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());

#pragma omp parallel for collapse(5) schedule(static)

    for (int oc_idx = 0; oc_idx < new_shape[0]; ++oc_idx) {
        for (int kh = 0; kh < kh_value; ++kh) {
            for (int kw = 0; kw < kw_value; ++kw) {
                for (int ic = 0; ic < ic_value; ++ic) {
                    for (int oc = 0; oc < 8; ++oc) {
                        int input_idx = (oc_idx * 8 + oc) * ic_value * kh_value * kw_value +
                                        ic * kh_value * kw_value +
                                        kh * kw_value + kw;
                        int output_idx = oc_idx * kh_value * kw_value * ic_value * 8 +
                                         kh * kw_value * ic_value * 8 +
                                         kw * ic_value * 8 +
                                         ic * 8 + oc;
                        *(output_ptr + output_idx) = ((oc_idx * 8 + oc) < oc_value) ? *(input_ptr + input_idx) : 0;
                    }
                }
            }
        }
    }
}

// reorder weight layout from NCHW to Goihw16g
static void weight_reorder_Goihw16g(Tensor<X86>& input,
                                    Tensor<X86>& output) {
    Shape shape = input.shape();
    int g_value = shape[0], oc_value = shape[1], ic_value = shape[1], kh_value = shape[2],
        kw_value = shape[3];

    if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_FLOAT) {
        float* output_ptr = static_cast<float*>(output.mutable_data());
        const float* input_ptr = static_cast<const float*>(input.data());

        #pragma omp parallel for collapse(6) schedule(static)

        for (int g_idx = 0; g_idx < g_value / 16; ++g_idx) {
            for (int oc_idx = 0; oc_idx < oc_value; ++oc_idx) {
                for (int ic_idx = 0; ic_idx < ic_value; ++ic_idx) {
                    for (int kh = 0; kh < kh_value; ++kh) {
                        for (int kw = 0; kw < kw_value; ++kw) {
                            for (int g = 0; g < 16; ++g) {
                                int input_idx = (g_idx * 16 + g) * oc_value * ic_value * kh_value * kw_value +
                                                oc_idx * ic_value * kh_value * kw_value +
                                                ic_idx * kh_value * kw_value +
                                                kh * kw_value + kw;
                                int output_idx = g_idx * oc_value * ic_value * kh_value * kw_value * 16 +
                                                 oc_idx * ic_value * kh_value * kw_value * 16 +
                                                 ic_idx * kh_value * kw_value * 16 +
                                                 kh * kw_value * 16 + kw * 16 + g;

                                *(output_ptr + output_idx) = *(input_ptr + input_idx);
                            }
                        }
                    }
                }
            }
        }
    } else if (input.get_dtype() == AK_INT8 && output.get_dtype() == AK_INT8) {
        char* output_ptr = static_cast<char*>(output.mutable_data());
        const char* input_ptr = static_cast<const char*>(input.data());

#pragma omp parallel for collapse(6) schedule(static)

        for (int g_idx = 0; g_idx < g_value / 16; ++g_idx) {
            for (int oc_idx = 0; oc_idx < oc_value; ++oc_idx) {
                for (int ic_idx = 0; ic_idx < ic_value; ++ic_idx) {
                    for (int kh = 0; kh < kh_value; ++kh) {
                        for (int kw = 0; kw < kw_value; ++kw) {
                            for (int g = 0; g < 16; ++g) {
                                int input_idx = (g_idx * 16 + g) * oc_value * ic_value * kh_value * kw_value +
                                                oc_idx * ic_value * kh_value * kw_value +
                                                ic_idx * kh_value * kw_value +
                                                kh * kw_value + kw;
                                int output_idx = g_idx * oc_value * ic_value * kh_value * kw_value * 16 +
                                                 oc_idx * ic_value * kh_value * kw_value * 16 +
                                                 ic_idx * kh_value * kw_value * 16 +
                                                 kh * kw_value * 16 + kw * 16 + g;

                                *(output_ptr + output_idx) = *(input_ptr + input_idx);
                            }
                        }
                    }
                }
            }
        }
    } else {
        ABORT_S() << "error: not supported reorder!";
    }
}

// reorder weight layout from NCHW to Goihw8g
static void weight_reorder_Goihw8g(Tensor<X86>& input,
                                   Tensor<X86>& output) {
    Shape shape = input.shape();
    int g_value = shape[0], oc_value = shape[1], ic_value = shape[1], kh_value = shape[2],
        kw_value = shape[3];

    if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_FLOAT) {
        float* output_ptr = static_cast<float*>(output.mutable_data());
        const float* input_ptr = static_cast<const float*>(input.data());

#pragma omp parallel for collapse(6) schedule(static)

        for (int g_idx = 0; g_idx < g_value / 8; ++g_idx) {
            for (int oc_idx = 0; oc_idx < oc_value; ++oc_idx) {
                for (int ic_idx = 0; ic_idx < ic_value; ++ic_idx) {
                    for (int kh = 0; kh < kh_value; ++kh) {
                        for (int kw = 0; kw < kw_value; ++kw) {
                            for (int g = 0; g < 8; ++g) {
                                int input_idx = (g_idx * 8 + g) * oc_value * ic_value * kh_value * kw_value +
                                                oc_idx * ic_value * kh_value * kw_value +
                                                ic_idx * kh_value * kw_value +
                                                kh * kw_value + kw;
                                int output_idx = g_idx * oc_value * ic_value * kh_value * kw_value * 8 +
                                                 oc_idx * ic_value * kh_value * kw_value * 8 +
                                                 ic_idx * kh_value * kw_value * 8 +
                                                 kh * kw_value * 8 + kw * 8 + g;

                                *(output_ptr + output_idx) = *(input_ptr + input_idx);
                            }
                        }
                    }
                }
            }
        }
    } else if (input.get_dtype() == AK_INT8 && output.get_dtype() == AK_INT8) {
        char* output_ptr = static_cast<char*>(output.mutable_data());
        const char* input_ptr = static_cast<const char*>(input.data());

#pragma omp parallel for collapse(6) schedule(static)

        for (int g_idx = 0; g_idx < g_value / 8; ++g_idx) {
            for (int oc_idx = 0; oc_idx < oc_value; ++oc_idx) {
                for (int ic_idx = 0; ic_idx < ic_value; ++ic_idx) {
                    for (int kh = 0; kh < kh_value; ++kh) {
                        for (int kw = 0; kw < kw_value; ++kw) {
                            for (int g = 0; g < 8; ++g) {
                                int input_idx = (g_idx * 8 + g) * oc_value * ic_value * kh_value * kw_value +
                                                oc_idx * ic_value * kh_value * kw_value +
                                                ic_idx * kh_value * kw_value +
                                                kh * kw_value + kw;
                                int output_idx = g_idx * oc_value * ic_value * kh_value * kw_value * 8 +
                                                 oc_idx * ic_value * kh_value * kw_value * 8 +
                                                 ic_idx * kh_value * kw_value * 8 +
                                                 kh * kw_value * 8 + kw * 8 + g;

                                *(output_ptr + output_idx) = *(input_ptr + input_idx);
                            }
                        }
                    }
                }
            }
        }
    } else {
        ABORT_S() << "error: not supported reorder!";
    }
}

// reorder bias layout from NCHW to 1C11
static void bias_reorder_nchw(const Tensor<X86>& input,
                              Tensor<X86>& output,
                              const std::vector<float>& scale) {
    Shape shape = input.shape();
    int n = shape[0], c = shape[1], h = shape[2], w = shape[3];

    if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_INT32) {
        int* output_ptr = static_cast<int*>(output.mutable_data());
        const float* input_ptr = static_cast<const float*>(input.data());

#pragma omp parallel for collapse(4) schedule(static)

        for (int n_idx = 0; n_idx < n; ++n_idx) {
            for (int c_idx = 0; c_idx < c; ++c_idx) {
                for (int h_idx = 0; h_idx < h; ++h_idx) {
                    for (int w_idx = 0; w_idx < w; ++w_idx) {
                        int input_idx = n_idx * c * h * w +
                                        c_idx * h * w +
                                        h_idx * w + w_idx;
                        int output_idx = n_idx * c * h * w +
                                         c_idx * h * w +
                                         h_idx * w + w_idx;

                        float scale_v = scale[c_idx];
                        *(output_ptr + output_idx) = (*(input_ptr + input_idx)) * scale_v;
                    }
                }
            }
        }
    } else if (input.get_dtype() == AK_FLOAT && output.get_dtype() == AK_FLOAT) {
        float* output_ptr = static_cast<float*>(output.mutable_data());
        const float* input_ptr = static_cast<const float*>(input.data());
        CHECK(scale.size()>0);
#pragma omp parallel for collapse(4) schedule(static)

        for (int n_idx = 0; n_idx < n; ++n_idx) {
            for (int c_idx = 0; c_idx < c; ++c_idx) {
                for (int h_idx = 0; h_idx < h; ++h_idx) {
                    for (int w_idx = 0; w_idx < w; ++w_idx) {
                        int input_idx = n_idx * c * h * w +
                                        c_idx * h * w +
                                        h_idx * w + w_idx;
                        int output_idx = n_idx * c * h * w +
                                         c_idx * h * w +
                                         h_idx * w + w_idx;
                        float scale_v = scale[c_idx];
                        *(output_ptr + output_idx) = (*(input_ptr + input_idx)) * scale_v;
                    }
                }
            }
        }
    }else if (input.get_dtype() == AK_INT32 && output.get_dtype() == AK_INT32) {
        int* output_ptr = static_cast<int*>(output.mutable_data());
        const int* input_ptr = static_cast<const int*>(input.data());

#pragma omp parallel for collapse(4) schedule(static)

        for (int n_idx = 0; n_idx < n; ++n_idx) {
            for (int c_idx = 0; c_idx < c; ++c_idx) {
                for (int h_idx = 0; h_idx < h; ++h_idx) {
                    for (int w_idx = 0; w_idx < w; ++w_idx) {
                        int input_idx = n_idx * c * h * w +
                                        c_idx * h * w +
                                        h_idx * w + w_idx;
                        int output_idx = n_idx * c * h * w +
                                         c_idx * h * w +
                                         h_idx * w + w_idx;

                        *(output_ptr + output_idx) = *(input_ptr + input_idx);
                    }
                }
            }
        }
    } else {
        ABORT_S() << "error: not supported convert!";
    }

}

// reorder input layout from NCHW(oc, ic, kh, kw) to nChwc8
inline void input_reorder_nChwc8(Tensor<X86>& input,
                                 Tensor<X86>& output) {
    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    CHECK_EQ(output.get_dtype(), AK_FLOAT) << "only support float type";
    Shape shape = input.valid_shape();
    int n_value = shape.num(), c_value = shape.channel(), h_value = shape.height(), w_value = shape.width();

    Shape new_shape({n_value, utils::round_up(c_value, 8) / 8, h_value, w_value, 8}, Layout_NCHW_C8);

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
#pragma omp parallel for collapse(5) schedule(static)

    for (int n = 0; n < n_value; ++n) {
        for (int c_idx = 0; c_idx < new_shape[1]; ++c_idx) {
            for (int h = 0; h < h_value; ++h) {
                for (int w = 0; w < w_value; ++w) {
                    for (int c = 0; c < 8; ++c) {
                        int input_idx = n * c_value * h_value * w_value + (c_idx * 8 + c) * h_value * w_value +
                                        h * w_value + w;
                        int output_idx = n * new_shape[1] * h_value * w_value * 8 + c_idx * h_value * w_value * 8 +
                                         h * w_value * 8 + w * 8 + c;

                        *(output_ptr + output_idx) = ((c_idx * 8 + c) < c_value) ? *(input_ptr + input_idx) : 0;
                    }
                }
            }
        }
    }
}

// reorder input layout from nchw_c8 to NCHW
inline void reorder_nchwc8_nchw(Tensor<X86>& input,
                                Tensor<X86>& output) {

    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    CHECK_EQ(output.get_dtype(), AK_FLOAT) << "only support float type";
    Shape shape = output.valid_shape();
    int n_value = shape[0];
    int c_value = shape[1];
    int h_value = shape[2];
    int w_value = shape[3];
    Shape shape_input = input.valid_shape();
    int c_round_div8 = shape_input[1];

    if (input.get_layout() == Layout_NCHW_C8R) {
        c_round_div8 = (shape_input.channel() + 7) / 8;
    }

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
#pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < n_value; ++n) {
        for (int c = 0; c < c_value; ++c) {
            for (int h = 0; h < h_value; ++h) {
                for (int w = 0; w < w_value; ++w) {
                    int round_c = c / 8;
                    int remainder_c = c % 8;
                    int input_idx = n * c_round_div8 * h_value * w_value * 8 + round_c * h_value * w_value * 8 +
                                    h * w_value * 8 + w * 8 + remainder_c;
                    int output_idx = n * c_value * h_value * w_value + c * h_value * w_value  +
                                     h * w_value  + w ;

                    *(output_ptr + output_idx) = input_ptr[input_idx];
                }
            }
        }
    }

}

// reorder output layout from NCHW(oc, ic, kh, kw) to nChwc8
inline void output_reorder_nChwc8(Tensor<X86>& input,
                                  Tensor<X86>& output) {

    input_reorder_nChwc8(input, output);
}


inline void weight_padding_nhwc(Tensor<X86>* input, Tensor<X86>* output) {
    CHECK_EQ(input->get_dtype(),AK_INT8);
    CHECK_EQ(output->get_dtype(),AK_INT8);
    Shape shape = input->shape();
    Shape shape_padding = output->shape();
    int oc_value = shape[0], ic_value = shape[1], kh_value = shape[2], kw_value = shape[3];
    int oc_padding = shape_padding[0], ic_padding = shape_padding[1];;

    char* output_ptr = static_cast<char*>(output->mutable_data());
    const char* input_ptr = static_cast<const char*>(input->data());

#pragma omp parallel for collapse(4) schedule(static)

    for (int oc = 0; oc < oc_padding; ++oc) {
        for (int ic = 0; ic < ic_padding; ++ic) {
            for (int kh = 0; kh < kh_value; ++kh) {
                for (int kw = 0; kw < kw_value; ++kw) {
                    int input_idx = oc * ic_value * kh_value * kw_value +
                                    ic * kh_value * kw_value +
                                    kh * kw_value + kw;
                    int output_idx = oc * ic_padding * kh_value * kw_value +
                                     ic * kh_value * kw_value +
                                     kh * kw_value + kw;

                    if (oc < oc_value && ic < ic_value) {
                        *(output_ptr + output_idx) = (*(input_ptr + input_idx));
                    } else {
                        *(output_ptr + output_idx) = 0;
                    }
                }
            }
        }
    }
}



} // namespace saber
} // namespace anakin



#endif // X86_UTILS_H
