
#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_SABER_AVX512_FUNCS_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_SABER_AVX512_FUNCS_H

#if defined(__AVX512F__)
#include "saber_normal_activation.h"
namespace anakin {

namespace saber {

void avx512_vector_sigmoid(const float* in, int length, float* out) {
    const int simd_length = 16;
    int remainder = length % simd_length;
    int round_length = length / simd_length * simd_length;

#pragma omp parallel for schedule(static)

    for (int i = 0; i < length; i += simd_length) {
        __m512 temp = Sigmoid(_mm512_loadu_ps(&in[i]));
        _mm512_storeu_ps(&out[i], temp);
    }

    if (remainder > 0) {
        __mmask16 vec_mask = 0xffff;
        vec_mask = vec_mask >> (simd_length - remainder);
        __m512 temp;
        temp = _mm512_mask_loadu_ps(temp, vec_mask, &in[round_length]);
        _mm512_mask_storeu_ps(&out[round_length], vec_mask, Sigmoid(temp));
    }
};

}
}
#endif

#endif //ANAKIN_SABER_AVX512_FUNCS_H
