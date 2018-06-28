

#ifndef ANAKIN_SABER_AVX2_ACTIVATION_H
#define ANAKIN_SABER_AVX2_ACTIVATION_H

#include "saber_avx2_math.h"
#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0

static inline __m256 _mm256_expfaster_ps(const __m256 &a) {

    const __m256 C1 = _mm256_set1_ps(1064872507.1541044f);
    const __m256 C2 = _mm256_set1_ps(12102203.161561485f);

    return _mm256_castsi256_ps(_mm256_cvttps_epi32(_mm256_fmadd_ps(C2, a, C1)));
}

inline __m256 InValidAct(__m256 a) {
    CHECK_EQ(0,1)<<"InValidAct";
}

inline __m256 Exp_fast(__m256 a) {
    return _mm256_expfaster_ps(a);
}

inline __m256 Exp(__m256 a) {
    return exp256_ps_fma(a);
}

inline __m256 Relu(const __m256 a) {
    __m256 tmp = _mm256_set1_ps(0.0f);
    return _mm256_max_ps(a, tmp);
}

inline __m256 Sigmoid_fluid(const __m256 a) {
    __m256 max = _mm256_set1_ps(SIGMOID_THRESHOLD_MAX);
    __m256 min = _mm256_set1_ps(SIGMOID_THRESHOLD_MIN);
    __m256 tmp = _mm256_max_ps(a, min);
    tmp = _mm256_min_ps(tmp, max);
    tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), tmp);
    tmp = Exp(tmp);
    tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);
    tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp);
    return tmp;
}

inline __m256 Sigmoid(const __m256 a) {
    __m256  tmp = a;
    tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), tmp);
    tmp = Exp(tmp);
    tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);
    tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp);
    return tmp;
}

inline __m256 Sigmoid_fast(const __m256 a) {
    __m256  tmp = a;
    tmp = _mm256_sub_ps(_mm256_set1_ps(0.0f), tmp);
    tmp = Exp_fast(tmp);
    tmp = _mm256_add_ps(_mm256_set1_ps(1.0f), tmp);
    tmp = _mm256_div_ps(_mm256_set1_ps(1.0f), tmp);
    return tmp;
}

inline __m256 Tanh_fluid(const __m256 a) {
    __m256 max = _mm256_set1_ps(EXP_MAX_INPUT);
    __m256 tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), a);
    tmp = _mm256_min_ps(tmp, max);
    tmp = Exp(tmp);
    return _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(2.0f),
                                       _mm256_add_ps(_mm256_set1_ps(1.0f), tmp)),
                         _mm256_set1_ps(1.0f));
}

inline __m256 Tanh(const __m256 a) {
    __m256 tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), a);
    tmp = Exp(tmp);
    return _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(2.0f),
                                       _mm256_add_ps(_mm256_set1_ps(1.0f), tmp)),
                         _mm256_set1_ps(1.0f));
}

inline __m256 Tanh_fast(const __m256 a) {
    __m256 tmp = _mm256_mul_ps(_mm256_set1_ps(-2.0f), a);
    tmp = Exp_fast(tmp);
    return _mm256_sub_ps(_mm256_div_ps(_mm256_set1_ps(2.0f),
                                       _mm256_add_ps(_mm256_set1_ps(1.0f), tmp)),
                         _mm256_set1_ps(1.0f));
}

inline __m256 Identity(const __m256 a) {
    return a;
}
__m256 (*act_func[])(__m256)= {&InValidAct, &Sigmoid, &Relu, &Tanh, &InValidAct, \
                                        & InValidAct, &Identity, &Sigmoid_fluid, &Tanh_fluid
};

#endif //ANAKIN_SABER_AVX2_ACTIVATION_H
