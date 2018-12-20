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
#include <assert.h>

#include <type_traits>

#include "saber/core/common.h"
#include "saber/core/tensor.h"
#include "saber/funcs/saber_util.h"

namespace anakin {
namespace saber {

#define UNUSED(x) ((void)x)
#define MAYBE_UNUSED(x) UNUSED(x)

#ifdef _WIN32
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

namespace utils {


/* a bunch of std:: analogues to be compliant with any msvs version
 *
 * Rationale: msvs c++ (and even some c) headers contain special pragma that
 * injects msvs-version check into object files in order to abi-mismatches
 * during the static linking. This makes sense if e.g. std:: objects are passed
 * through between application and library, which is not the case for mkl-dnn
 * (since there is no any c++-rt dependent stuff, ideally...). */

/* SFINAE helper -- analogue to std::enable_if */
class VectorPrint {
public:
    template <typename Dtype>
    static void print_float(Dtype* target) {
        float* f = (float*)target;
        printf("size = %d\n", sizeof(Dtype));

        for (int i = 0; i < sizeof(Dtype) / sizeof(float); i++) {
            printf(" %f ,", f[i]);
        }

        printf("\n");
    }
};
template<typename opTensor>
static inline void try_expand_clean_tensor(opTensor& tensor, anakin::saber::Shape shape) {
    if (utils::try_expand_tensor(tensor, shape)) {
        memset(tensor.mutable_data(), 0, tensor.valid_size()* type_length(tensor.get_dtype()));
    };
}

class AlignedUtils {
public:
    template <typename Dtype>
    void aligned_last_dim(const Dtype* input, Dtype* output, int input_size, int ori_last_dim,
                          int aligned_dim) {
        for (int row = 0; row < input_size / ori_last_dim; row++) {
            for (int col = ori_last_dim; col < aligned_dim; col++) {
                output[row * aligned_dim + col] = static_cast<Dtype>(0);
            }
        }

        for (int i = 0; i < input_size; i++) {
            int row = i / ori_last_dim;
            int col = i % ori_last_dim;
            output[row * aligned_dim + col] = input[i];
        }
    }
    template <typename Dtype>
    void unaligned_last_dim(const Dtype* input, Dtype* output, int output_size, int ori_last_dim,
                            int aligned_dim) {
        for (int i = 0; i < output_size; i++) {
            int row = i / ori_last_dim;
            int col = i % ori_last_dim;
            output[i] = input[row * aligned_dim + col];
        }
    }

};

class SeqSortedseqTranseUtil {
public:
    SeqSortedseqTranseUtil(bool is_reverse = false, bool is_bi = false)
        : _is_reverse(is_reverse),
          _is_bi(is_bi) {};
    void print_vec(int* in, int size, const char* perfix) {
        for (int i = 0; i < size; i++) {
            printf("[%s] %d = %d\n", perfix, i, in[i]);
        }
    }
    template <typename Dtype>
    void seq_2_sorted_seq(const Dtype*  input, Dtype* output, int word_size) {
        //        _map_vec.resize(word_sum);
        int word_sum = _map_vec.size();
        //        std::cout << "word_sum = " << word_sum << std::endl;

        for (int ori_word_id = 0; ori_word_id < word_sum; ++ori_word_id) {
            //can param
            int word_start = ori_word_id * word_size;
            int maped_id = _map_vec[ori_word_id];
            int maped_start = maped_id * word_size;

            for (int word_vec_offset = 0; word_vec_offset < word_size; ++word_vec_offset) {
                //                std::cout<<maped_start + word_vec_offset<<" --> "<<word_start + word_vec_offset<<" , = "<<input[maped_start + word_vec_offset]<<std::endl;

                output[maped_start + word_vec_offset] = input[word_start + word_vec_offset];

            }
        }
    }
    template <typename Dtype>
    void hidden_2_sorted_hidden(const Dtype*  input, Dtype* output, int hidden_size) {
        //        _map_vec.resize(word_sum);
        int batch_size = _length_index.size();
        //        std::cout << "word_sum = " << word_sum << std::endl;

        for (int ori_word_id = 0; ori_word_id < batch_size; ++ori_word_id) {
            //can param
            int word_start = ori_word_id * hidden_size;
            int maped_id = _length_index[ori_word_id];
            int maped_start = maped_id * hidden_size;

            for (int word_vec_offset = 0; word_vec_offset < hidden_size; ++word_vec_offset) {
                //                std::cout<<maped_start + word_vec_offset<<" --> "<<word_start + word_vec_offset<<" , = "<<input[maped_start + word_vec_offset]<<std::endl;

                output[word_start + word_vec_offset] = input[maped_start + word_vec_offset];

            }
        }
    }
    template <typename Dtype>
    void sorted_seq_2_seq(const Dtype* input, Dtype* output, int hidden_size) {
        int word_sum = _map_vec.size();

        for (int ori_word_id = 0; ori_word_id < word_sum; ori_word_id++) {
            //can param
            int word_start = ori_word_id * hidden_size;
            int maped_id = _map_vec[ori_word_id];
            int maped_start = maped_id * hidden_size;

            for (int word_vec_offset = 0; word_vec_offset < hidden_size; word_vec_offset++) {
                //            std::cout<<ori_word_id+word_vec_offset<<" -> "<<maped_start+word_vec_offset<<std::endl;
                output[word_start + word_vec_offset] = input[maped_start + word_vec_offset];
            }
        }
    }
    template <typename Dtype>
    void sorted_seq_2_seq(const Dtype* input, Dtype* output, int hidden_size,
                          int alligned_hidden_size) {
        int word_sum = _map_vec.size();

        for (int ori_word_id = 0; ori_word_id < word_sum; ori_word_id++) {
            //can param
            int word_start = ori_word_id * hidden_size;
            int maped_id = _map_vec[ori_word_id];
            int maped_start = maped_id * alligned_hidden_size;

            for (int word_vec_offset = 0; word_vec_offset < hidden_size; word_vec_offset++) {
                //            std::cout<<ori_word_id+word_vec_offset<<" -> "<<maped_start+word_vec_offset<<std::endl;
                output[word_start + word_vec_offset] = input[maped_start + word_vec_offset];
            }
        }
    }
    /**
     * return whether need to transform
     * @param offset_vec
     * @param emit_offset_vec
     * @param emit_length
     * @return
     */
    bool get_sorted_map(std::vector<int>& offset_vec,
                        std::vector<int>& emit_offset_vec, int& emit_length) {
        int batch_size = offset_vec.size() - 1;
        int word_sum = offset_vec[offset_vec.size() - 1];
        std::vector<int>length_vec(batch_size);
        _length_index.resize(batch_size);

        if (batch_size == 1) {
            emit_length = offset_vec[1] - offset_vec[0];
            emit_offset_vec.resize(emit_length + 1);

            for (int i = 0; i <= emit_length; i++) {
                emit_offset_vec[i] = i;
            }

            return false;
        }

        int max_len = 0;

        for (int i = 0; i < offset_vec.size() - 1; ++i) {
            int len = offset_vec[i + 1] - offset_vec[i];
            max_len = max_len > len ? max_len : len;
            length_vec[i] = len;
            _length_index[i] = i;
        }

        emit_length = max_len;

        if (max_len == 1) {
            emit_offset_vec.push_back(0);
            emit_offset_vec.push_back(emit_length * batch_size);
            return false;
        }

        std::sort(_length_index.begin(), _length_index.end(), [&length_vec](int i1, int i2) {
            return length_vec[i1] > length_vec[i2];
        });

        emit_offset_vec.resize(max_len + 1);
        _map_vec.resize(word_sum);

        int target_word_id = 0;
        std::vector<int> length_vec_cnt = length_vec;
        int last_batch_size = batch_size;
        for (int word_id_in_seq = 0; word_id_in_seq < max_len; word_id_in_seq++) {
            emit_offset_vec[word_id_in_seq] = target_word_id;

            for (int batch_id = 0; batch_id < last_batch_size; batch_id++) {
                int old_batch_id = _length_index[batch_id];

                if (length_vec_cnt[old_batch_id] > 0) {
                    int inner_word_id_in_seq = word_id_in_seq;

                    if (_is_reverse) {
                        inner_word_id_in_seq = length_vec[old_batch_id] - 1 - word_id_in_seq;
                    }

                    int old_word_id = offset_vec[old_batch_id] + inner_word_id_in_seq;
                    _map_vec[old_word_id] = target_word_id;
                    //                    printf("map %d -> %d\n",old_word_id,target_word_id);
                    length_vec_cnt[old_batch_id]--;
                    target_word_id++;
                } else {
                    last_batch_size--;
                    break;
                }
            }
        }

        //        print_vec(_map_vec.data(),word_sum,"map");
        emit_offset_vec[max_len] = word_sum;
        return true;
    }


private:
    //    std::vector<int> _length_vec;
    std::vector<int> _length_index;
    std::vector<int> _map_vec;
    bool _is_reverse;
    bool _is_bi;

};

inline int round_up(int k, int c) {
    return ((k + c - 1) / c) * c;
}

inline int div_up(int k, int c) {
    return (k + c - 1) / c;
}

/* analogue std::conditional */
template <bool, typename, typename> struct conditional {};
template <typename T, typename F> struct conditional<true, T, F> {
    typedef T type;
};
template <typename T, typename F> struct conditional<false, T, F> {
    typedef F type;
};

template <bool, typename, bool, typename, typename> struct conditional3 {};
template <typename T, typename FT, typename FF>
struct conditional3<true, T, false, FT, FF> {
    typedef T type;
};
template <typename T, typename FT, typename FF>
struct conditional3<false, T, true, FT, FF> {
    typedef FT type;
};
template <typename T, typename FT, typename FF>
struct conditional3<false, T, false, FT, FF> {
    typedef FF type;
};

template <bool, typename U, U, U> struct conditional_v {};
template <typename U, U t, U f> struct conditional_v<true, U, t, f> {
    static constexpr U value = t;
};
template <typename U, U t, U f> struct conditional_v<false, U, t, f> {
    static constexpr U value = f;
};

template<typename T>
inline const T& min(const T& a, const T& b) {
    return a < b ? a : b;
}

template<typename T>
inline const T& max(const T& a, const T& b) {
    return a > b ? a : b;
}

template <typename T>
inline typename std::remove_reference<T>::type zero() {
    auto zero = typename std::remove_reference<T>::type();
    return zero;
}

template <typename T, typename P>
inline bool everyone_is(T val, P item) {
    return val == item;
}
template <typename T, typename P, typename... Args>
inline bool everyone_is(T val, P item, Args... item_others) {
    return val == item && everyone_is(val, item_others...);
}

template <typename T, typename P>
inline bool one_of(T val, P item) {
    return val == item;
}
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename... Args>
inline bool any_null(Args... ptrs) {
    return one_of(nullptr, ptrs...);
}

inline bool implication(bool cause, bool effect) {
    return !cause || effect;
}

template<typename T>
inline void array_copy(T* dst, const T* src, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        dst[i] = src[i];
    }
}

template<typename T>
inline bool array_cmp(const T* a1, const T* a2, size_t size) {
    for (size_t i = 0; i < size; ++i) if (a1[i] != a2[i]) {
            return false;
        }

    return true;
}

template<typename T, typename U>
inline void array_set(T* arr, const U& val, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        arr[i] = static_cast<T>(val);
    }
}

namespace product_impl {

template<size_t> struct int2type {};

template <typename T>
constexpr int product_impl(const T* arr, int2type<0>) {
    return arr[0];
}

template <typename T, size_t num>
inline T product_impl(const T* arr, int2type<num>) {
    return arr[0] * product_impl(arr + 1, int2type < num - 1 > ());
}
}

template <size_t num, typename T>
inline T array_product(const T* arr) {
    return product_impl::product_impl(arr, product_impl::int2type < num - 1 > ());
}

template<typename T, typename R = T>
inline R array_product(const T* arr, size_t size) {
    R prod = 1;

    for (size_t i = 0; i < size; ++i) {
        prod *= arr[i];
    }

    return prod;
}

template <typename T, typename U>
inline typename std::remove_reference<T>::type div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template <typename T, typename U>
inline typename std::remove_reference<T>::type rnd_up(const T a, const U b) {
    return div_up(a, b) * b;
}

template <typename T, typename U>
inline typename std::remove_reference<T>::type rnd_dn(const T a, const U b) {
    return (a / b) * b;
}

template <typename T, typename U, typename V>
inline U this_block_size(const T offset, const U max, const V block_size) {
    assert(offset < max);
    // TODO (Roma): can't use nstl::max() due to circular dependency... we
    // need to fix this
    const T block_boundary = offset + block_size;

    if (block_boundary > max) {
        return max - offset;
    } else {
        return block_size;
    }
}

template <typename Telem, size_t Tdims>
struct array_offset_calculator {
    template <typename... Targs>
    array_offset_calculator(Telem* base, Targs... Fargs) : _dims{ Fargs... } {
        _base_ptr = base;
    }

    template <typename... Targs>
    inline Telem& operator()(Targs... Fargs) {
        return *(_base_ptr + _offset(1, Fargs...));
    }

private:
    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t element) {
        return element;
    }

    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t theta, size_t element) {
        return element + (_dims[dimension] * theta);
    }

    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t theta, size_t element,
                          Targs... Fargs) {
        size_t t_prime = element + (_dims[dimension] * theta);
        return _offset(dimension + 1, t_prime, Fargs...);
    }

    Telem* _base_ptr;
    const int _dims[Tdims];
};

} // namespace utils


inline void* zmalloc(size_t size, int alignment) {
    void* ptr = NULL;

#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
    int rc = ptr ? 0 : -1;
#else
    int rc = ::posix_memalign(&ptr, alignment, size);
#endif

    return (rc == 0) ? ptr : NULL;
}

inline void zfree(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    ::free(p);
#endif
}

struct c_compatible {
    enum { default_alignment = 4096 };

    static void* operator new (size_t sz) {
        return zmalloc(sz, default_alignment);
    }

    static void* operator new (size_t sz, void* p) {
        UNUSED(sz);
        return p;
    }

    static void* operator new[](size_t sz) {
        return zmalloc(sz, default_alignment);
    }

    static void operator delete (void* p) {
        zfree(p);
    }

    static void operator delete[](void* p) {
        zfree(p);
    }
};

inline void yield_thread() { }

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
                    for (int oc = 0; oc < 8; ++oc){
                        for (int ic = 0; ic < 8; ++ic)
                        {
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

    Shape new_shape({utils::round_up(oc_value, 8)/8, ic_value, kh_value, kw_value, 8}, Layout_NCHW_C8);

    if(oc_value % 8 != 0) {
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
                        *(output_ptr + output_idx) = ((oc_idx * 8 + oc) < oc_value)? *(input_ptr + input_idx) : 0;
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
static void bias_reorder_nchw(Tensor<X86>& input,
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
    } else if (input.get_dtype() == AK_INT32 && output.get_dtype() == AK_INT32) {
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
    int n_value = shape[0], c_value = shape[1], h_value = shape[2], w_value = shape[3];

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
    int c_round_div8=shape_input[1];
    if(input.get_layout()==Layout_NCHW_C8R) {
        c_round_div8 = (shape_input.channel() + 7) / 8;
    }

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
#pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < n_value; ++n) {
        for (int c = 0; c < c_value; ++c) {
            for (int h = 0; h < h_value; ++h) {
                for (int w = 0; w < w_value; ++w) {
                    int round_c=c/8;
                    int remainder_c=c%8;
                    int input_idx = n * c_round_div8 * h_value * w_value*8 + round_c * h_value * w_value*8 +
                                    h * w_value*8 + w*8+remainder_c;
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

template<typename T> struct is_integral
{ static constexpr bool value = false; };

template<> struct is_integral<int32_t> { static constexpr bool value = true; };
template<> struct is_integral<int16_t> { static constexpr bool value = true; };
template<> struct is_integral<int8_t> { static constexpr bool value = true; };
template<> struct is_integral<uint8_t> { static constexpr bool value = true; };

template <typename data_t, typename acc_t>
inline typename std::enable_if<!is_integral<data_t>::value,
       typename std::remove_reference<data_t>::type>::type
saturate(const acc_t &x) { return x; }

template <typename data_t, typename acc_t>
inline typename std::enable_if<is_integral<data_t>::value,
       typename std::remove_reference<data_t>::type>::type
saturate(const acc_t &x) {
    acc_t v = x;
    if (v < (acc_t)std::numeric_limits<data_t>::lowest())
        v = (acc_t)std::numeric_limits<data_t>::lowest();
    if (v > (acc_t)std::numeric_limits<data_t>::max())
        v = (acc_t)std::numeric_limits<data_t>::max();
    return (typename std::remove_reference<data_t>::type)v;
}

template <typename out_t>
inline out_t round_and_saturate(float f, round_mode rmode) {
    switch (rmode) {
    case round_mode::nearest: f = nearbyintf(f); break;
    case round_mode::down: f = floorf(f); break;
    }
    return saturate<out_t>(f);
}

/* Quantization with beta == 0 */
template <typename in_t, typename out_t> struct qz_b0 {
    out_t operator()(in_t in, float alpha, round_mode rmode)
    { return round_and_saturate<out_t>(alpha * in, rmode); }
};

inline size_t datatype_size(DataType data_type) {
    switch (data_type) {
    case AK_FLOAT:
        return sizeof(float);

    case AK_INT32:
        return sizeof(int32_t);

    case AK_HALF:
        return sizeof(int16_t);

    case AK_INT8:
        return sizeof(int8_t);

    case AK_UINT8:
        return sizeof(uint8_t);

    case AK_INVALID:
    default:
        assert(!"unknown data_type");
    }

    return 0;
}

/** returns floor(log2(v)), aka the position of the leftmost non-0 bit */
inline int ilog2q(size_t v) {
    if (v == 0)
        return -1;

    int p = 0;
#   define CP(pw) do { if (v >= (1ull << pw)) { v >>= pw; p += pw; } } while(0)
    CP(32); CP(16); CP(8); CP(4); CP(2); CP(1);
#   undef CP
    return p;
}

struct scratchpad_t {
    virtual ~scratchpad_t() {}
    virtual char *get() const = 0;
};


} // namespace saber
} // namespace anakin



#endif // X86_UTILS_H
