#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_MATH_ACTIVATION_FUNCTIONS_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_MATH_ACTIVATION_FUNCTIONS_H
#include <math.h>
#include <string>
#include "saber/saber_types.h"
#include "utils/logger/logger.h"
#include <immintrin.h>

namespace anakin{
namespace saber {
namespace math {

template <typename T>
void sigmoid(size_t len, T *x, T *y) {
    for (size_t i = 0; i < len; i++) {
        y[i] = 1. / (1. + exp(-x[i]));
    }
}

template <typename T>
void parallel_sigmoid(size_t len, T *x, T *y) {
    #pragma omp parallel for
    for (size_t i = 0; i < len; i++) {
        y[i] = 1. / (1. + exp(-x[i]));
    }
}

template <typename T>
void relu(size_t len, T *x, T *y) {
    for (size_t i = 0; i < len; i++) {
        y[i] = x[i] < 0 ? 0 : x[i];
    }
}

template <typename T>
void parallel_relu(size_t len, T *x, T *y) {
    #pragma omp parallel for
    for (size_t i = 0; i < len; i++) {
        y[i] = x[i] < 0 ? 0 : x[i];
    }
}

template <typename T>
void tanh(size_t len, T *x, T *y) {
    for (size_t i = 0; i < len; i++) {
        T e_x = exp(2 * x[i]);
        y[i] = (e_x - 1) / (e_x + 1);
    }
}

template <typename T>
void parallel_tanh(size_t len, T *x, T *y) {
    #pragma omp parallel for
    for (size_t i = 0; i < len; i++) {
        T e_x = exp(2 * x[i]);
        y[i] = (e_x - 1) / (e_x + 1);
    }
}

template <typename T>
void stanh(size_t len, T *x, T *y) {
    for (size_t i = 0; i < len; i++) {
        T e_x = exp(4. * x[i] / 3.);
        y[i] = 1.7159 * (e_x - 1) / (e_x + 1);
    }
}

template <typename T>
void parallel_stanh(size_t len, T *x, T *y) {
    #pragma omp parallel for
    for (size_t i = 0; i < len; i++) {
        T e_x = exp(4. * x[i] / 3.);
        y[i] = 1.7159 * (e_x - 1) / (e_x + 1);
    }
}

template <typename T>
void identity(size_t len, T *x, T *y) {
    for (size_t i = 0; i < len; i++) {
        y[i] = x[i];
    }
}

template <typename T>
void parallel_identity(size_t len, T *x, T *y) {
    #pragma omp parallel for
    for (size_t i = 0; i < len; i++) {
        y[i] = x[i];
    }
}

__m256 Relu(const __m256 a);
__m256 Sigmoid(const __m256 a);
__m256 Tanh(const __m256 a);
__m256 Identity(const __m256 a);

template <typename T>
struct Active {
  typedef void (*Act)(size_t, T*, T*);
  typedef T (*Act_m256)(T);
};

static Active<float>::Act k_act_float[] = {
    nullptr,
    &sigmoid<float>,
    &relu<float>,
    &tanh<float>,
    nullptr,
    nullptr,
    &identity<float>,
    nullptr,
    nullptr,
    &stanh<float>
    };

static Active<float>::Act k_parallel_act_float[] = {
    nullptr,
    &parallel_sigmoid<float>,
    &parallel_relu<float>,
    &parallel_tanh<float>,
    nullptr,
    nullptr,
    &parallel_identity<float>,
    nullptr,
    nullptr,
    &parallel_stanh<float>
    };

static Active<__m256>::Act_m256 k_act_avx[] = {
    nullptr,
    &Sigmoid,
    &Relu,
    &Tanh,
    nullptr,
    nullptr,
    &Identity,
    nullptr,
    nullptr,
    nullptr
};

inline void activation(size_t len, float *src, float *dst, int index) {
    auto *func = k_act_float[index];
    if (!func) {
        LOG(ERROR) << "activation not implemented!";
    }
    func(len, src, dst);
}

inline void parallel_activation(size_t len, float *src, float *dst, int index) {
    auto *func = k_parallel_act_float[index];
    if (!func) {
        LOG(ERROR) << "activation not implemented!";
    }
    func(len, src, dst);
}

inline __m256 avx_activation(__m256 a, int index) {
    return k_act_avx[index](a);
}

}  // namespace math
}  // namespace saber
}  // namespace anakin
#endif  //ANAKIN_SABER_FUNCS_IMPL_X86_MATH_ACTIVATION_FUNCTIONS_H
