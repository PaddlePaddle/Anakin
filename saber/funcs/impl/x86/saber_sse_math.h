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

#ifndef ANAKIN_SABER_SSE_MATH_H
#define ANAKIN_SABER_SSE_MATH_H

#include <immintrin.h>

static inline __m128 exp128_ps_fma(__m128 x) {
    __m128 tmp = _mm_setzero_ps(), fx;
    __m128i imm0;
    __m128 one=_mm_set1_ps(1.f);
    __m128 _ps128_exp_hi=_mm_set1_ps(88.3762626647949f);
    __m128 _ps128_exp_lo=_mm_set1_ps(-88.3762626647949f);
    x = _mm_min_ps(x, _ps128_exp_hi);
    x = _mm_max_ps(x, _ps128_exp_lo);

    __m128 _ps128_cephes_LOG2EF=_mm_set1_ps(1.44269504088896341f);
    fx = _mm_mul_ps(x, _ps128_cephes_LOG2EF);
    __m128 _ps128_0p5=_mm_set1_ps(0.5);
    fx = _mm_add_ps(fx, _ps128_0p5);

    tmp = _mm_floor_ps(fx);

    __m128 mask = _mm_cmp_ps(tmp, fx, _CMP_GT_OS);
    mask = _mm_and_ps(mask, one);
    fx = _mm_sub_ps(tmp, mask);

    __m128 _ps128_cephes_exp_C1=_mm_set1_ps(0.693359375f);
    __m128 _ps128_cephes_exp_C2=_mm_set1_ps(-2.12194440E-4f);
    tmp = _mm_mul_ps(fx, _ps128_cephes_exp_C1);
    __m128 z = _mm_mul_ps(fx, _ps128_cephes_exp_C2);
    x = _mm_sub_ps(x, tmp);
    x = _mm_sub_ps(x, z);
    z = _mm_mul_ps(x, x);

    __m128 _ps128_cephes_exp_p0=_mm_set1_ps(1.9875691500E-4f);
    __m128 _ps128_cephes_exp_p1=_mm_set1_ps(1.3981999507E-3f);
    __m128 _ps128_cephes_exp_p2=_mm_set1_ps(8.3334519073E-3f);
    __m128 _ps128_cephes_exp_p3=_mm_set1_ps(4.1665795894E-2f);
    __m128 _ps128_cephes_exp_p4=_mm_set1_ps(1.6666665459E-1f);
    __m128 _ps128_cephes_exp_p5=_mm_set1_ps(5.0000001201E-1f);
    __m128 y = _ps128_cephes_exp_p0;
    y = _mm_fmadd_ps(y, x,_ps128_cephes_exp_p1);
    y = _mm_fmadd_ps(y, x,_ps128_cephes_exp_p2);
    y = _mm_fmadd_ps(y, x,_ps128_cephes_exp_p3);
    y = _mm_fmadd_ps(y, x,_ps128_cephes_exp_p4);
    y = _mm_fmadd_ps(y, x,_ps128_cephes_exp_p5);
    y = _mm_fmadd_ps(y, z,x);
    y = _mm_add_ps(y, one);
    /* build 2^n */
    imm0 = _mm_cvttps_epi32(fx);
    // another two AVX2 instructions
    __m128i _pi32_128_0x7f=_mm_set1_epi32(0x7f);
    imm0 = _mm_add_epi32(imm0, _pi32_128_0x7f);
    imm0 = _mm_slli_epi32(imm0, 23);
    __m128 pow2n = _mm_castsi128_ps(imm0);
    y = _mm_mul_ps(y, pow2n);
    return y;
}

#endif