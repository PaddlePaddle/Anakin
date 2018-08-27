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

#ifndef ANAKIN_SABER_AVX512_MATH_H
#define ANAKIN_SABER_AVX512_MATH_H

#if defined(__AVX512F__)
#include <immintrin.h>
static inline __m512 _mm512_expfaster_ps(const __m512& a) {

    const __m512 C1 = _mm512_set1_ps(1064872507.1541044f);
    const __m512 C2 = _mm512_set1_ps(12102203.161561485f);

    return _mm512_castsi512_ps(_mm512_cvttps_epi32(_mm512_fmadd_ps(C2, a, C1)));
}

inline __m512 exp512_ps_fma(__m512 x) {
    __m512 tmp = _mm512_setzero_ps(), fx;
    __m512i imm0;
    __m512 one = _mm512_set1_ps(1.f);
    __m512 _ps512_exp_hi = _mm512_set1_ps(88.3762626647949f);
    __m512 _ps512_exp_lo = _mm512_set1_ps(-88.3762626647949f);
    x = _mm512_min_ps(x, _ps512_exp_hi);
    x = _mm512_max_ps(x, _ps512_exp_lo);

    __m512 _ps512_cephes_LOG2EF = _mm512_set1_ps(1.44269504088896341f);
    fx = _mm512_mul_ps(x, _ps512_cephes_LOG2EF);
    __m512 _ps512_0p5 = _mm512_set1_ps(0.5);
    fx = _mm512_add_ps(fx, _ps512_0p5);

    tmp = _mm512_floor_ps(fx);

    //TODO:check _mm512_cmp_ps_mask _mm512_cmp_ps
    __mmask16 mask_16 = _mm512_cmp_ps_mask(tmp, fx, _CMP_GT_OS);
    __m512 zero = _mm512_setzero_ps();
    __m512 mask = _mm512_mask_add_ps(zero, mask_16, zero, one);

    //    __m512 mask = _mm512_cmp_ps(tmp, fx, _CMP_GT_OS);
    //    mask = _mm512_and_ps(mask, one);
    fx = _mm512_sub_ps(tmp, mask);

    __m512 _ps512_cephes_exp_C1 = _mm512_set1_ps(0.693359375f);
    __m512 _ps512_cephes_exp_C2 = _mm512_set1_ps(-2.12194440E-4f);
    tmp = _mm512_mul_ps(fx, _ps512_cephes_exp_C1);
    __m512 z = _mm512_mul_ps(fx, _ps512_cephes_exp_C2);
    x = _mm512_sub_ps(x, tmp);
    x = _mm512_sub_ps(x, z);
    z = _mm512_mul_ps(x, x);

    __m512 _ps512_cephes_exp_p0 = _mm512_set1_ps(1.9875691500E-4f);
    __m512 _ps512_cephes_exp_p1 = _mm512_set1_ps(1.3981999507E-3f);
    __m512 _ps512_cephes_exp_p2 = _mm512_set1_ps(8.3334519073E-3f);
    __m512 _ps512_cephes_exp_p3 = _mm512_set1_ps(4.1665795894E-2f);
    __m512 _ps512_cephes_exp_p4 = _mm512_set1_ps(1.6666665459E-1f);
    __m512 _ps512_cephes_exp_p5 = _mm512_set1_ps(5.0000001201E-1f);
    __m512 y = _ps512_cephes_exp_p0;
    y = _mm512_fmadd_ps(y, x, _ps512_cephes_exp_p1);
    y = _mm512_fmadd_ps(y, x, _ps512_cephes_exp_p2);
    y = _mm512_fmadd_ps(y, x, _ps512_cephes_exp_p3);
    y = _mm512_fmadd_ps(y, x, _ps512_cephes_exp_p4);
    y = _mm512_fmadd_ps(y, x, _ps512_cephes_exp_p5);
    y = _mm512_fmadd_ps(y, z, x);
    y = _mm512_add_ps(y, one);
    /* build 2^n */
    imm0 = _mm512_cvttps_epi32(fx);
    // another two AVX2 instructions
    __m512i _pi32_512_0x7f = _mm512_set1_epi32(0x7f);
    imm0 = _mm512_add_epi32(imm0, _pi32_512_0x7f);
    imm0 = _mm512_slli_epi32(imm0, 23);
    __m512 pow2n = _mm512_castsi512_ps(imm0);
    y = _mm512_mul_ps(y, pow2n);
    return y;
}
#endif
#endif //ANAKIN_SABER_AVX512_MATH_H