/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#ifndef X86_UTILS_H
#define X86_UTILS_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "saber/core/tensor.h"

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
template<bool expr, class T = void> struct enable_if {};
template<class T> struct enable_if<true, T> { typedef T type; };

/* analogue std::conditional */
template <bool, typename, typename> struct conditional {};
template <typename T, typename F> struct conditional<true, T, F>
{ typedef T type; };
template <typename T, typename F> struct conditional<false, T, F>
{ typedef F type; };

template <bool, typename, bool, typename, typename> struct conditional3 {};
template <typename T, typename FT, typename FF>
struct conditional3<true, T, false, FT, FF> { typedef T type; };
template <typename T, typename FT, typename FF>
struct conditional3<false, T, true, FT, FF> { typedef FT type; };
template <typename T, typename FT, typename FF>
struct conditional3<false, T, false, FT, FF> { typedef FF type; };

template <bool, typename U, U, U> struct conditional_v {};
template <typename U, U t, U f> struct conditional_v<true, U, t, f>
{ static constexpr U value = t; };
template <typename U, U t, U f> struct conditional_v<false, U, t, f>
{ static constexpr U value = f; };

template <typename T> struct remove_reference { typedef T type; };
template <typename T> struct remove_reference<T&> { typedef T type; };
template <typename T> struct remove_reference<T&&> { typedef T type; };

template<typename T>
inline const T& min(const T& a, const T& b) {
       return a < b ? a : b;
}

template<typename T>
inline const T& max(const T& a, const T& b) {
       return a > b ? a : b;
}

template <typename T>
inline T&& forward(typename utils::remove_reference<T>::type &t)
{ return static_cast<T&&>(t); }
template <typename T>
inline T&& forward(typename utils::remove_reference<T>::type &&t)
{ return static_cast<T&&>(t); }

template <typename T>
inline typename remove_reference<T>::type zero()
{ auto zero = typename remove_reference<T>::type(); return zero; }

template <typename T, typename P>
inline bool everyone_is(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool everyone_is(T val, P item, Args... item_others) {
    return val == item && everyone_is(val, item_others...);
}

template <typename T, typename P>
inline bool one_of(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename... Args>
inline bool any_null(Args... ptrs) { return one_of(nullptr, ptrs...); }

inline bool implication(bool cause, bool effect) { return !cause || effect; }

template<typename T>
inline void array_copy(T *dst, const T *src, size_t size) {
    for (size_t i = 0; i < size; ++i) dst[i] = src[i];
}

template<typename T>
inline bool array_cmp(const T *a1, const T *a2, size_t size) {
    for (size_t i = 0; i < size; ++i) if (a1[i] != a2[i]) return false;
    return true;
}

template<typename T, typename U>
inline void array_set(T *arr, const U& val, size_t size) {
    for (size_t i = 0; i < size; ++i) arr[i] = static_cast<T>(val);
}

namespace product_impl {

template<size_t> struct int2type{};

template <typename T>
constexpr int product_impl(const T *arr, int2type<0>) {
    return arr[0];
}

template <typename T, size_t num>
inline T product_impl(const T *arr, int2type<num>) {
    return arr[0] * product_impl(arr + 1, int2type<num - 1>()); }
}

template <size_t num, typename T>
inline T array_product(const T *arr) {
    return product_impl::product_impl(arr, product_impl::int2type<num-1>());
}

template<typename T, typename R = T>
inline R array_product(const T *arr, size_t size) {
    R prod = 1;
    for (size_t i = 0; i < size; ++i) {
        prod *= arr[i];
    }
    return prod;
}

template <typename T, typename U>
inline typename remove_reference<T>::type div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template <typename T, typename U>
inline typename remove_reference<T>::type rnd_up(const T a, const U b) {
    return div_up(a, b) * b;
}

template <typename T, typename U>
inline typename remove_reference<T>::type rnd_dn(const T a, const U b) {
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
    }
    else {
        return block_size;
    }
}


template <typename T, typename U>
inline void balance211(T n, U team, U tid, T &n_start, T &n_end) {
    T n_min = 1;
    T &n_my = n_end;
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else if(n_min == 1) {
        // team = T1 + T2
        // n = T1*n1 + T2*n2  (n1 - n2 = 1)
        T n1 = div_up(n, (T)team);
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_my = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }
    n_end += n_start;
}

template<typename T>
inline T nd_iterator_init(T start) { return start; }
template<typename T, typename U, typename W, typename... Args>
inline T nd_iterator_init(T start, U &x, const W &X, Args &&... tuple) {
    start = nd_iterator_init(start, utils::forward<Args>(tuple)...);
    x = start % X;
    return start / X;
}

inline bool nd_iterator_step() { return true; }
template<typename U, typename W, typename... Args>
inline bool nd_iterator_step(U &x, const W &X, Args &&... tuple) {
    if (nd_iterator_step(utils::forward<Args>(tuple)...) ) {
        x = (x + 1) % X;
        return x == 0;
    }
    return false;
}

template<typename U, typename W, typename Y>
inline bool nd_iterator_jump(U &cur, const U end, W &x, const Y &X)
{
    U max_jump = end - cur;
    U dim_jump = X - x;
    if (dim_jump <= max_jump) {
        x = 0;
        cur += dim_jump;
        return true;
    } else {
        cur += max_jump;
        x += max_jump;
        return false;
    }
}

template<typename U, typename W, typename Y, typename... Args>
inline bool nd_iterator_jump(U &cur, const U end, W &x, const Y &X,
        Args &&... tuple) {
    if (nd_iterator_jump(cur, end, utils::forward<Args>(tuple)...)) {
        x = (x + 1) % X;
        return x == 0;
    }
    return false;
}

template <typename Telem, size_t Tdims>
struct array_offset_calculator {
    template <typename... Targs>
    array_offset_calculator(Telem *base, Targs... Fargs) : _dims{ Fargs... } {
        _base_ptr = base;
    }

    template <typename... Targs>
    inline Telem &operator()(Targs... Fargs) {
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

    Telem *_base_ptr;
    const int _dims[Tdims];
};

} // namespace utils


inline void *zmalloc(size_t size, int alignment) {
  void *ptr = NULL;

#ifdef _WIN32
  ptr = _aligned_malloc(size, alignment);
  int rc = ptr ? 0 : -1;
#else
  int rc = ::posix_memalign(&ptr, alignment, size);
#endif

  return (rc == 0) ? ptr : NULL;
}

inline void zfree(void *p) {
#ifdef _WIN32
  _aligned_free(p);
#else
  ::free(p);
#endif
}

struct c_compatible {
    enum { default_alignment = 4096 };

    static void *operator new(size_t sz) {
        return zmalloc(sz, default_alignment);
    }

    static void *operator new(size_t sz, void *p) {
        UNUSED(sz);
        return p;
    }

    static void *operator new[](size_t sz) {
        return zmalloc(sz, default_alignment);
    }

    static void operator delete(void *p) {
        zfree(p);
    }

    static void operator delete[](void *p) {
        zfree(p);
    }
};

inline void yield_thread() { }

// reorder weight layout from NCHW(oc, ic, kh, kw) to OIhw16i16o
inline void weight_reorder_OIhw16i16o(Tensor<X86, AK_FLOAT, NCHW> &input, Tensor<X86, AK_FLOAT, NCHW> &output) {
     Shape shape = input.shape();
     int oc_value = shape[0], ic_value = shape[1], kh_value = shape[2], kw_value = shape[3];
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
                                               ic_idx * kh_value* kw_value * 16 * 16 +
                                               kh * kw_value * 16 * 16 +
                                               kw * 16 * 16 + ic * 16 + oc;

                              *(output.mutable_data() + output_idx) = *(input.data() + input_idx);
                         }
                     }
                 }
             }
        }
     }
}

// reorder weight layout from NCHW(oc, ic, kh, kw) to OIhwi16o
inline void weight_reorder_OIhwi16o(Tensor<X86, AK_FLOAT, NCHW> &input, Tensor<X86, AK_FLOAT, NCHW> &output) {
    Shape shape = input.shape();
    #pragma omp parallel for collapse(5) schedule(static)
    for (int oc_idx = 0; oc_idx < shape[0] / 16; ++oc_idx) {
        for (int kh = 0; kh < shape[2]; ++kh ) {
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

                        *(output.mutable_data() + output_idx) = *(input.data() + input_idx);
                    }
                }
            }
        }
    }
}


// reorder weight layout from NCHW(oc, ic, kh, kw) to OIhwi8o
inline void weight_reorder_OIhwi8o(Tensor<X86, AK_FLOAT, NCHW> &input, Tensor<X86, AK_FLOAT, NCHW> &output) {
    Shape shape = input.shape();

    #pragma omp parallel for collapse(5) schedule(static)
    for (int oc_idx = 0; oc_idx < shape[0] / 8; ++oc_idx) {
        for (int kh = 0; kh < shape[2]; ++kh) {
            for (int kw = 0; kw < shape[3]; ++kw) {
                for (int ic = 0; ic < shape[1]; ++ic) {
                    for (int oc = 0; oc < 8; ++oc) {
                        int input_idx = (oc_idx * 8 + oc) * shape[1] * shape[2] * shape[3] +
                                        ic * shape[2] * shape[3] + 
                                        kh * shape[3] + kw;
                        int output_idx = oc_idx * shape[2] * shape[3] * shape[1] * 8 +
                                         kh * shape[3] * shape[1] * 8 +
                                         kw * shape[1] * 8 +
                                         ic * 8 + oc;

                        *(output.mutable_data() + output_idx) = *(input.data() + input_idx);
                    }
                }
            }
        }
    }
}

// reorder weight layout from NCHW to Goihw16g
static void weight_reorder_Goihw16g(Tensor<X86, AK_FLOAT, NCHW> &input, Tensor<X86, AK_FLOAT, NCHW> &output){
     Shape shape = input.shape();
     int g_value = shape[0], oc_value = shape[1], ic_value = shape[1], kh_value = shape[2], kw_value = shape[3];
#pragma omp parallel for collapse(6) schedule(static)
     for (int g_idx = 0; g_idx < g_value/16; ++g_idx) {
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

                             *(output.mutable_data() + output_idx) = *(input.data() + input_idx);
                         }
                     }
                 }
             }
         }
     }
}

inline size_t datatype_size(DataType data_type) {
    switch (data_type) {
        case AK_FLOAT: return sizeof(float);
        case AK_INT32: return sizeof(int32_t);
        case AK_INT16: return sizeof(int16_t);
        case AK_INT8: return sizeof(int8_t);
        case AK_UINT8: return sizeof(uint8_t);
        case AK_INVALID:
        default: assert(!"unknown data_type");
    }
    return 0;
}

} // namespace saber
} // namespace anakin

#if defined(_OPENMP)
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_num_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline int omp_in_parallel() { return 0; }
#endif

#endif // X86_UTILS_H

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
