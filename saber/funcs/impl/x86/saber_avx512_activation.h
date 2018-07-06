//
// Created by Liu,Junjie(SYS) on 2018/6/7.
//

#ifndef ANAKIN_SABER_AVX512_ACTIVATION_H
#define ANAKIN_SABER_AVX512_ACTIVATION_H

#include "saber_avx512_math.h"

inline __m512 Relu(const __m512 a) {
    __m512 tmp = _mm512_set1_ps(0.0f);
    return _mm512_max_ps(a, tmp);
}
inline __m512 InValidAct(__m512 a) {
    CHECK_EQ(0,1)<<"InValidAct";
}

inline __m512 Exp_fast(__m512 a) {
    return exp512_ps_fma(a);
}

inline __m512 Exp(__m512 a) {
    return exp512_ps(a);
}

inline __m512 Relu(const __m512 a) {
    __m512 tmp = _mm512_set1_ps(0.0f);
    return _mm512_max_ps(a, tmp);
}

inline __m512 Sigmoid_fluid(const __m512 a) {
    __m512 max = _mm512_set1_ps(SIGMOID_THRESHOLD_MAX);
    __m512 min = _mm512_set1_ps(SIGMOID_THRESHOLD_MIN);
    __m512 tmp = _mm512_max_ps(a, min);
    tmp = _mm512_min_ps(tmp, max);
    tmp = _mm512_sub_ps(_mm512_set1_ps(0.0f), tmp);
    tmp = Exp(tmp);
    tmp = _mm512_add_ps(_mm512_set1_ps(1.0f), tmp);
    tmp = _mm512_div_ps(_mm512_set1_ps(1.0f), tmp);
    return tmp;
}

inline __m512 Sigmoid(const __m512 a) {
    __m512  tmp = a;
    tmp = _mm512_sub_ps(_mm512_set1_ps(0.0f), tmp);
    tmp = Exp(tmp);
    tmp = _mm512_add_ps(_mm512_set1_ps(1.0f), tmp);
    tmp = _mm512_div_ps(_mm512_set1_ps(1.0f), tmp);
    return tmp;
}

inline __m512 Sigmoid_fast(const __m512 a) {
    __m512  tmp = a;
    tmp = _mm512_sub_ps(_mm512_set1_ps(0.0f), tmp);
    tmp = Exp_fast(tmp);
    tmp = _mm512_add_ps(_mm512_set1_ps(1.0f), tmp);
    tmp = _mm512_div_ps(_mm512_set1_ps(1.0f), tmp);
    return tmp;
}

inline __m512 Tanh_fluid(const __m512 a) {
    __m512 max = _mm512_set1_ps(EXP_MAX_INPUT);
    __m512 tmp = _mm512_mul_ps(_mm512_set1_ps(-2.0f), a);
    tmp = _mm512_min_ps(tmp, max);
    tmp = Exp(tmp);
    return _mm512_sub_ps(_mm512_div_ps(_mm512_set1_ps(2.0f),
                                       _mm512_add_ps(_mm512_set1_ps(1.0f), tmp)),
                         _mm512_set1_ps(1.0f));
}

inline __m512 Tanh(const __m512 a) {
    __m512 tmp = _mm512_mul_ps(_mm512_set1_ps(-2.0f), a);
    tmp = Exp(tmp);
    return _mm512_sub_ps(_mm512_div_ps(_mm512_set1_ps(2.0f),
                                       _mm512_add_ps(_mm512_set1_ps(1.0f), tmp)),
                         _mm512_set1_ps(1.0f));
}

inline __m512 Tanh_fast(const __m512 a) {
    __m512 tmp = _mm512_mul_ps(_mm512_set1_ps(-2.0f), a);
    tmp = Exp_fast(tmp);
    return _mm512_sub_ps(_mm512_div_ps(_mm512_set1_ps(2.0f),
                                       _mm512_add_ps(_mm512_set1_ps(1.0f), tmp)),
                         _mm512_set1_ps(1.0f));
}

inline __m512 Identity(const __m512 a) {
    return a;
}


static __m512 (*act_func[])(__m512)= {&InValidAct, &Sigmoid_fast, &Relu, &Tanh, &InValidAct, \
                                        & InValidAct, &Identity, &Sigmoid_fluid, &Tanh_fluid
};
#endif //ANAKIN_SABER_AVX512_ACTIVATION_H
