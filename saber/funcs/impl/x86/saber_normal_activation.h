
#ifndef ANAKIN_SABER_NORMAL_ACTIVATION_H
#define ANAKIN_SABER_NORMAL_ACTIVATION_H

#include "saber_types.h"
#include <cmath>


#include "saber_avx512_math.h"
#include "saber_avx2_math.h"
#include "saber_sse_math.h"

namespace anakin {

namespace saber {


template<typename Dtype>
inline Dtype InValidAct(Dtype a) {
    return 0;
}

template<typename Dtype>
inline Dtype Sigmoid(const Dtype a) {
    return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + exp(-a));
}


template<typename Dtype>
inline Dtype Tanh(const Dtype a) {
    Dtype tmp = static_cast<Dtype>(-2.0) * a;
    return (static_cast<Dtype>(2.0) / (static_cast<Dtype>(1.0) + exp(tmp))) - static_cast<Dtype>(1.0);
}

template<typename Dtype>
inline Dtype Relu(const Dtype a) {
    return a > static_cast<Dtype>(0.0) ? a : static_cast<Dtype>(0.0);
}

template<typename Dtype>
inline Dtype Identity(const Dtype a) {
    return a;
}

#if defined(__SSE4_2__) and defined(__FMA__)

template<>
inline __m128 InValidAct<__m128>(const __m128 a) {
    return _mm_set1_ps(0.0f);
}


template<>
inline __m128 Relu<__m128>(const __m128 a) {
    __m128 tmp = _mm_set1_ps(0.0f);
    return _mm_max_ps(a, tmp);
}


template<>
inline __m128 Sigmoid<__m128>(const __m128 a) {
    __m128 tmp = a;
    tmp = _mm_sub_ps(_mm_set1_ps(0.0f), tmp);
    tmp = exp128_ps_fma(tmp);
    tmp = _mm_add_ps(_mm_set1_ps(1.0f), tmp);
    tmp = _mm_div_ps(_mm_set1_ps(1.0f), tmp);
    return tmp;
}


template<>
inline __m128 Tanh<__m128>(const __m128 a) {
    __m128 tmp = _mm_mul_ps(_mm_set1_ps(-2.0f), a);
    tmp = exp128_ps_fma(tmp);
    return _mm_sub_ps(_mm_div_ps(_mm_set1_ps(2.0f),
                                 _mm_add_ps(_mm_set1_ps(1.0f), tmp)),
                      _mm_set1_ps(1.0f));
}


#endif




#if defined(__AVX2__) and defined(__FMA__)

template<>
inline __m256 InValidAct<__m256>(const __m256 a) {
    return _mm256_set1_ps(0.0f);
}

template<>
inline __m256 Relu<__m256>(const __m256 a) {
    __m256 tmp = _mm256_set1_ps(0.0f);
    return _mm256_max_ps(a, tmp);
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
inline __m256 Tanh<__m256>(const __m256 a) {
    __m256 tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), a);
    tmp = exp256_ps_fma(tmp);
    return _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(2.0f),
                                       _mm256_add_ps(_mm256_set1_ps(1.0f), tmp)),
                         _mm256_set1_ps(1.0f));
}

#endif


#if defined(__AVX512F__)

template<>
inline __m512 InValidAct<__m512>(const __m512 a) {
    return _mm512_set1_ps(0.0f);
}

template<>
inline __m512 Relu<__m512>(const __m512 a) {
    __m512 tmp = _mm512_set1_ps(0.0f);
    return _mm512_max_ps(a, tmp);
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
    static typename ACTIVATION<Dtype>::Act vec[7] = {&InValidAct<Dtype>, &Sigmoid < Dtype >, &Relu < Dtype >,
                                                     &Tanh < Dtype >,
                                                     &InValidAct<Dtype>, &InValidAct<Dtype>,
                                                     &Identity < Dtype >
                                                    };
    return vec[type];
}

template<typename Dtype>
static inline Dtype Activate_inner(Dtype value,ActiveType type) {
    static typename ACTIVATION<Dtype>::Act vec[7] = {&InValidAct<Dtype>, &Sigmoid < Dtype >, &Relu < Dtype >,
                                                     &Tanh < Dtype >,
                                                     &InValidAct<Dtype>, &InValidAct<Dtype>,
                                                     &Identity < Dtype >
    };
    return vec[type](value);
}

}
}
#endif //ANAKIN_SABER_NORMAL_ACTIVATION_H
