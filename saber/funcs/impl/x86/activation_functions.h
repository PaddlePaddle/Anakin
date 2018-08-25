//
//#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_MATH_ACTIVATION_FUNCTIONS_H
//#define ANAKIN_SABER_FUNCS_IMPL_X86_MATH_ACTIVATION_FUNCTIONS_H
//#include <math.h>
//#include <string>
//#include "saber/saber_types.h"
//#include "utils/logger/logger.h"
//#ifdef __AVX2__
//#include "saber/funcs/impl/x86/saber_avx2_math.h"
//#endif
//
//namespace anakin {
//namespace saber {
//namespace math {
//
//template <typename T>
//static void sigmoid(size_t len, T *x, T *y) {
//    for (size_t i = 0; i < len; i++) {
//        y[i] = 1. / (1. + exp(-x[i]));
//    }
//}
//
//template <typename T>
//static void parallel_sigmoid(size_t len, T *x, T *y) {
//#pragma omp parallel for
//    for (size_t i = 0; i < len; i++) {
//        y[i] = 1. / (1. + exp(-x[i]));
//    }
//}
//
//template <typename T>
//static static void relu(size_t len, T *x, T *y) {
//    for (size_t i = 0; i < len; i++) {
//        y[i] = x[i] < 0 ? 0 : x[i];
//    }
//}
//
//template <typename T>
//static void parallel_relu(size_t len, T *x, T *y) {
//#pragma omp parallel for
//    for (size_t i = 0; i < len; i++) {
//        y[i] = x[i] < 0 ? 0 : x[i];
//    }
//}
//
//template <typename T>
//static void tanh(size_t len, T *x, T *y) {
//    for (size_t i = 0; i < len; i++) {
//        T e_x = exp(2 * x[i]);
//        y[i] = (e_x - 1) / (e_x + 1);
//    }
//}
//
//template <typename T>
//static void parallel_tanh(size_t len, T *x, T *y) {
//#pragma omp parallel for
//    for (size_t i = 0; i < len; i++) {
//        T e_x = exp(2 * x[i]);
//        y[i] = (e_x - 1) / (e_x + 1);
//    }
//}
//
//template <typename T>
//static void stanh(size_t len, T *x, T *y) {
//    for (size_t i = 0; i < len; i++) {
//        T e_x = exp(4. * x[i] / 3.);
//        y[i] = 1.7159 * (e_x - 1) / (e_x + 1);
//    }
//}
//
//template <typename T>
//static void parallel_stanh(size_t len, T *x, T *y) {
//#pragma omp parallel for
//    for (size_t i = 0; i < len; i++) {
//        T e_x = exp(4. * x[i] / 3.);
//        y[i] = 1.7159 * (e_x - 1) / (e_x + 1);
//    }
//}
//
//template <typename T>
//static void identity(size_t len, T *x, T *y) {
//    for (size_t i = 0; i < len; i++) {
//        y[i] = x[i];
//    }
//}
//
//template <typename T>
//static void parallel_identity(size_t len, T *x, T *y) {
//#pragma omp parallel for
//    for (size_t i = 0; i < len; i++) {
//        y[i] = x[i];
//    }
//}
//
//template <typename T>
//struct Active {
//    typedef void (*Act)(size_t, T*, T*);
//    typedef T (*Act_m256)(T);
//};
//
//static Active<float>::Act k_act_float[] = {
//        nullptr,
//        &sigmoid<float>,
//        &relu<float>,
//        &tanh<float>,
//        nullptr,
//        nullptr,
//        &identity<float>,
//        &sigmoid<float>,
//        &tanh<float>,
//        &stanh<float>
//};
//
//static Active<float>::Act k_parallel_act_float[] = {
//        nullptr,
//        &parallel_sigmoid<float>,
//        &parallel_relu<float>,
//        &parallel_tanh<float>,
//        nullptr,
//        nullptr,
//        &parallel_identity<float>,
//        &parallel_sigmoid<float>,
//        &parallel_tanh<float>,
//        &parallel_stanh<float>
//};
//
//static inline void activation(size_t len, float *src, float *dst, int index) {
//    auto *func = k_act_float[index];
//    if (!func) {
//                LOG(ERROR) << "activation not implemented!";
//    }
//    func(len, src, dst);
//}
//
//static inline void parallel_activation(size_t len, float *src, float *dst, int index) {
//    auto *func = k_parallel_act_float[index];
//    if (!func) {
//                LOG(ERROR) << "activation not implemented!";
//    }
//    func(len, src, dst);
//}
//
//#ifdef __AVX2__
//static inline __m256 Exp(__m256 a) { return exp256_ps(a); }
//
//static inline __m256 Relu(const __m256 a) {
//    __m256 tmp = _mm256_set1_ps(0.0f);
//    return _mm256_max_ps(a, tmp);
//}
//
//static inline __m256 Sigmoid(const __m256 a) {
//    __m256 tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), a);
//    tmp = Exp(tmp);
//    tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);
//    tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp);
//    return tmp;
//}
//
//static inline __m256 Tanh(const __m256 a) {
//    __m256 tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), a);
//    tmp = Exp(tmp);
//    return _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(2.0f),
//                                       _mm256_add_ps(_mm256_set1_ps(1.0f), tmp)),
//                         _mm256_set1_ps(1.0f));
//}
//
//static inline __m256 Identity(const __m256 a) { return a; }
//
//static Active<__m256>::Act_m256 k_act_avx[] = {
//        nullptr,
//        &Sigmoid,
//        &Relu,
//        &Tanh,
//        nullptr,
//        nullptr,
//        &Identity,
//        &Sigmoid,
//        &Tanh,
//        nullptr
//};
//static inline __m256 avx_activation(__m256 a, int index) {
//    return k_act_avx[index](a);
//}
//#endif
//}  // namespace math
//}  // namespace saber
//}  // namespace anakin
//#endif  //ANAKIN_SABER_FUNCS_IMPL_X86_MATH_ACTIVATION_FUNCTIONS_H
