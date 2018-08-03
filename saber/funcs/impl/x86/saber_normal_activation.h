
#ifndef ANAKIN_SABER_NORMAL_ACTIVATION_H
#define ANAKIN_SABER_NORMAL_ACTIVATION_H

#include "saber_types.h"

namespace anakin {

namespace saber {


#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0

template<typename Dtype>
inline Dtype InValidAct(Dtype a) {
    CHECK_EQ(0, 1) << "InValidAct";
}

template<typename Dtype>
inline Dtype Sigmoid(const Dtype a) {
    return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-a));
}

template<typename Dtype>
inline Dtype Sigmoid_fluid(const Dtype a) {
    const Dtype min = SIGMOID_THRESHOLD_MIN;
    const Dtype max = SIGMOID_THRESHOLD_MAX;
    Dtype tmp = (a < min) ? min : ((a > max) ? max : a);
    return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-tmp));
}

template<typename Dtype>
inline Dtype Tanh_fluid(const Dtype a) {
    Dtype tmp = -2.0 * a;
    tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
    return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

template<typename Dtype>
inline Dtype Tanh(const Dtype a) {
    Dtype tmp = -2.0 * a;
    return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

template<typename Dtype>
inline Dtype Relu(const Dtype a) {
    return a > static_cast<Dtype>(0.0) ? a : static_cast<Dtype>(0.0);
}

template<typename Dtype>
inline Dtype Identity(const Dtype a) {
    return a;
}

#ifdef __AVX2__
#include "saber_avx2_math.h"

template<>
inline __m256 Relu<__m256>(const __m256 a) {
    __m256 tmp = _mm256_set1_ps(0.0f);
    return _mm256_max_ps(a, tmp);
}

template<>
inline __m256 Sigmoid_fluid<__m256>(const __m256 a) {
    __m256 max = _mm256_set1_ps(SIGMOID_THRESHOLD_MAX);
    __m256 min = _mm256_set1_ps(SIGMOID_THRESHOLD_MIN);
    __m256 tmp = _mm256_max_ps(a, min);
    tmp = _mm256_min_ps(tmp, max);
    tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), tmp);
    tmp = exp256_ps_fma(tmp);
    tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);
    tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp);
    return tmp;
}

template<>
inline __m256 Sigmoid<__m256>(const __m256 a) {
    __m256 tmp = a;
    tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), tmp);
    tmp = exp256_ps_fma(tmp);
    tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);
    tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp);
    return tmp;
}


template<>
inline __m256 Tanh_fluid<__m256>(const __m256 a) {
    __m256 max = _mm256_set1_ps(EXP_MAX_INPUT);
    __m256 tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), a);
    tmp = _mm256_min_ps(tmp, max);
    tmp = exp256_ps_fma(tmp);
    return _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(2.0f),
                                       _mm256_add_ps(_mm256_set1_ps(1.0f), tmp)),
                         _mm256_set1_ps(1.0f));
}

template<>
inline __m256 Tanh<__m256>(const __m256 a) {
    __m256 tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), a);
    tmp = exp256_ps_fma(tmp);
    return _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(2.0f),
                                       _mm256_add_ps(_mm256_set1_ps(1.0f), tmp)),
                         _mm256_set1_ps(1.0f));
}

#endif

#ifdef __AVX512F__
#include "saber_avx512_math.h"

template<>
inline __m512 Relu<__m512>(const __m512 a) {
    __m512 tmp = _mm512_set1_ps(0.0f);
    return _mm512_max_ps(a, tmp);
}

template<>
inline __m512 Sigmoid_fluid<__m512>(const __m512 a) {
    __m512 max = _mm512_set1_ps(SIGMOID_THRESHOLD_MAX);
    __m512 min = _mm512_set1_ps(SIGMOID_THRESHOLD_MIN);
    __m512 tmp = _mm512_max_ps(a, min);
    tmp = _mm512_min_ps(tmp, max);
    tmp = _mm512_sub_ps(_mm512_set1_ps(0.0f), tmp);
    tmp = exp512_ps_fma(tmp);
    tmp = _mm512_add_ps(_mm512_set1_ps(1.0f), tmp);
    tmp = _mm512_div_ps(_mm512_set1_ps(1.0f), tmp);
    return tmp;
}

template<>
inline __m512 Sigmoid<__m512>(const __m512 a) {
    __m512  tmp = a;
    tmp = _mm512_sub_ps(_mm512_set1_ps(0.0f), tmp);
    tmp = exp512_ps_fma(tmp);
    tmp = _mm512_add_ps(_mm512_set1_ps(1.0f), tmp);
    tmp = _mm512_div_ps(_mm512_set1_ps(1.0f), tmp);
    return tmp;
}

template<>
inline __m512 Sigmoid_fast<__m512>(const __m512 a) {
    __m512  tmp = a;
    tmp = _mm512_sub_ps(_mm512_set1_ps(0.0f), tmp);
    tmp = exp512_ps_fma(tmp);
    tmp = _mm512_add_ps(_mm512_set1_ps(1.0f), tmp);
    tmp = _mm512_div_ps(_mm512_set1_ps(1.0f), tmp);
    return tmp;
}

template<>
inline __m512 Tanh_fluid<__m512>(const __m512 a) {
    __m512 max = _mm512_set1_ps(EXP_MAX_INPUT);
    __m512 tmp = _mm512_mul_ps(_mm512_set1_ps(-2.0f), a);
    tmp = _mm512_min_ps(tmp, max);
    tmp = exp512_ps_fma(tmp);
    return _mm512_sub_ps(_mm512_div_ps(_mm512_set1_ps(2.0f),
                                       _mm512_add_ps(_mm512_set1_ps(1.0f), tmp)),
                         _mm512_set1_ps(1.0f));
}

template<>
inline __m512 Tanh<__m512>(const __m512 a) {
    __m512 tmp = _mm512_mul_ps(_mm512_set1_ps(-2.0f), a);
    tmp = exp512_ps_fma(tmp);
    return _mm512_sub_ps(_mm512_div_ps(_mm512_set1_ps(2.0f),
                                       _mm512_add_ps(_mm512_set1_ps(1.0f), tmp)),
                         _mm512_set1_ps(1.0f));
}

#endif

template<typename Dtype>
struct ACTIVATION {
    typedef Dtype(*Act)(const Dtype);
};

template<typename Dtype>
inline typename ACTIVATION<Dtype>::Act Activate_inner(ActiveType type) {
    static typename ACTIVATION<Dtype>::Act vec[9] = {&InValidAct<Dtype>, &Sigmoid < Dtype >, &Relu < Dtype >,
                                                     &Tanh < Dtype >,
                                                     &InValidAct<Dtype>, &InValidAct<Dtype>,
                                                     &Identity < Dtype >, &Sigmoid_fluid < Dtype >,
                                                     &Tanh_fluid < Dtype >
                                                    };
    return vec[type];
}

}
}
#endif //ANAKIN_SABER_NORMAL_ACTIVATION_H
