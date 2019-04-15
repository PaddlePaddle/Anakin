
#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_AVX2_EXPAND_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_AVX2_EXPAND_H
#if defined(__AVX2__) and defined(__FMA__)
#include <immintrin.h>

namespace anakin {
namespace saber {

inline __v8si _m256_continue_mask_v8si(const int& x) {
    static __v8si map[9] = {
        { 0,  0,  0,  0,  0,  0,  0,  0},
        {-1,  0,  0,  0,  0,  0,  0,  0},
        {-1, -1,  0,  0,  0,  0,  0,  0},
        {-1, -1, -1,  0,  0,  0,  0,  0},
        {-1, -1, -1, -1,  0,  0,  0,  0},
        {-1, -1, -1, -1, -1,  0,  0,  0},
        {-1, -1, -1, -1, -1, -1,  0,  0},
        {-1, -1, -1, -1, -1, -1, -1,  0},
        {-1, -1, -1, -1, -1, -1, -1, -1}
    };
    return map[x];
}

inline __m256 _m256_continue_mask_m256(const int& x) {
    static __m256 map[9] = {
        { 0,  0,  0,  0,  0,  0,  0,  0},
        {-1,  0,  0,  0,  0,  0,  0,  0},
        {-1, -1,  0,  0,  0,  0,  0,  0},
        {-1, -1, -1,  0,  0,  0,  0,  0},
        {-1, -1, -1, -1,  0,  0,  0,  0},
        {-1, -1, -1, -1, -1,  0,  0,  0},
        {-1, -1, -1, -1, -1, -1,  0,  0},
        {-1, -1, -1, -1, -1, -1, -1,  0},
        {-1, -1, -1, -1, -1, -1, -1, -1}
    };
    return map[x];
}

inline __m256i _m256_continue_mask_m256i(const int& x) {
    return (__m256i)_m256_continue_mask_v8si(x);
}

#define MAX(a,b)a>b?a:b
inline float _m256_self_sum(const __m256& x) {
    float temp0 = x[0] + x[1];
    float temp1 = x[2] + x[3];
    float temp2 = x[4] + x[5];
    float temp3 = x[6] + x[7];
    temp0 += temp1;
    temp2 += temp3;
    temp0 += temp2;
    return temp0;
}


inline float _m256_self_max(const __m256& x) {
    float temp0 = MAX(x[0], x[1]);
    float temp1 = MAX(x[2], x[3]);
    float temp2 = MAX(x[4], x[5]);
    float temp3 = MAX(x[6], x[7]);
    temp0 = MAX(temp0, temp1);
    temp2 = MAX(temp2, temp3);
    temp0 = MAX(temp0, temp2);
    return temp0;
}

inline float _m256_max_array(const float* in, int length) {
    __m256 max_vec = _mm256_set1_ps(-1e32);
    int round_length =  length/8*8;
    int remainder = length % 8;
    for (int j = 0; j < round_length; j += 8) {
        __m256 temp_in = _mm256_loadu_ps(&in[j]);
        max_vec = _mm256_max_ps(temp_in, max_vec);
    }

    if (remainder > 0) {
        __m256i _vec_mask = _m256_continue_mask_m256i(remainder);
        __m256 temp_in = _mm256_maskload_ps(&in[round_length], _vec_mask);
        __m256  _vec_mask_m256 = _m256_continue_mask_m256(remainder);
        max_vec = _mm256_blendv_ps(max_vec, _mm256_max_ps(temp_in, max_vec), _vec_mask_m256);
    }
    return _m256_self_max(max_vec);
}


}

}
#endif


#endif //ANAKIN_SABER_FUNCS_IMPL_X86_AVX2_EXPAND_H
