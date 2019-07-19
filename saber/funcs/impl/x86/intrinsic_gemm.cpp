#include "intrinsic_gemm.h"

#include <emmintrin.h>
#include <mmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <x86intrin.h>
namespace anakin {

namespace saber {
#if defined(__AVX2__)
inline void block8x8_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {
    //printf("block8x8_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;
    const int8_t* pa4 = pa0 + 4 * lda;
    const int8_t* pa5 = pa0 + 5 * lda;
    const int8_t* pa6 = pa0 + 6 * lda;
    const int8_t* pa7 = pa0 + 7 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;
    const int8_t* pb4 = pb0 + 4 * ldb;
    const int8_t* pb5 = pb0 + 5 * ldb;
    const int8_t* pb6 = pb0 + 6 * ldb;
    const int8_t* pb7 = pb0 + 7 * ldb;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;
    int* pc4 = c + 4 * ldc;
    int* pc5 = c + 5 * ldc;
    int* pc6 = c + 6 * ldc;
    int* pc7 = c + 7 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma1_l;
    __m256i ma2_l;
    __m256i ma3_l;
    __m256i ma4_l;
    __m256i ma5_l;
    __m256i ma6_l;
    __m256i ma7_l;
    __m256i ma0_h;
    __m256i ma1_h;
    __m256i ma2_h;
    __m256i ma3_h;
    __m256i ma4_h;
    __m256i ma5_h;
    __m256i ma6_h;
    __m256i ma7_h;

    __m256i mb0_l;
    __m256i mb1_l;
    __m256i mb2_l;
    __m256i mb3_l;
    __m256i mb4_l;
    __m256i mb5_l;
    __m256i mb6_l;
    __m256i mb7_l;
    __m256i mb0_h;
    __m256i mb1_h;
    __m256i mb2_h;
    __m256i mb3_h;
    __m256i mb4_h;
    __m256i mb5_h;
    __m256i mb6_h;
    __m256i mb7_h;

    __m256i mc0;
    __m256i mc1;
    __m256i mc2;
    __m256i mc3;
    __m256i mc4;
    __m256i mc5;
    __m256i mc6;
    __m256i mc7;
    __m256i mc8;
    __m256i mc9;
    __m256i mc10;
    __m256i mc11;
    __m256i mc12;
    __m256i mc13;
    __m256i mc14;
    __m256i mc15;

    _mm_prefetch((char*) pa0, _MM_HINT_T0);
    _mm_prefetch((char*) pa1, _MM_HINT_T0);
    _mm_prefetch((char*) pa2, _MM_HINT_T0);
    _mm_prefetch((char*) pa3, _MM_HINT_T0);
    _mm_prefetch((char*) pa4, _MM_HINT_T0);
    _mm_prefetch((char*) pa5, _MM_HINT_T0);
    _mm_prefetch((char*) pa6, _MM_HINT_T0);
    _mm_prefetch((char*) pa7, _MM_HINT_T0);

    _mm_prefetch((char*) pb0, _MM_HINT_T0);
    _mm_prefetch((char*) pb1, _MM_HINT_T0);
    _mm_prefetch((char*) pb2, _MM_HINT_T0);
    _mm_prefetch((char*) pb3, _MM_HINT_T0);
    _mm_prefetch((char*) pb4, _MM_HINT_T0);
    _mm_prefetch((char*) pb5, _MM_HINT_T0);
    _mm_prefetch((char*) pb6, _MM_HINT_T0);
    _mm_prefetch((char*) pb7, _MM_HINT_T0);

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    __m256i sum8 = _mm256_setzero_si256();
    __m256i sum9 = _mm256_setzero_si256();
    __m256i sum10 = _mm256_setzero_si256();
    __m256i sum11 = _mm256_setzero_si256();
    __m256i sum12 = _mm256_setzero_si256();
    __m256i sum13 = _mm256_setzero_si256();
    __m256i sum14 = _mm256_setzero_si256();
    __m256i sum15 = _mm256_setzero_si256();

    __m256i sum16 = _mm256_setzero_si256();
    __m256i sum17 = _mm256_setzero_si256();
    __m256i sum18 = _mm256_setzero_si256();
    __m256i sum19 = _mm256_setzero_si256();
    __m256i sum20 = _mm256_setzero_si256();
    __m256i sum21 = _mm256_setzero_si256();
    __m256i sum22 = _mm256_setzero_si256();
    __m256i sum23 = _mm256_setzero_si256();

    __m256i sum24 = _mm256_setzero_si256();
    __m256i sum25 = _mm256_setzero_si256();
    __m256i sum26 = _mm256_setzero_si256();
    __m256i sum27 = _mm256_setzero_si256();
    __m256i sum28 = _mm256_setzero_si256();
    __m256i sum29 = _mm256_setzero_si256();
    __m256i sum30 = _mm256_setzero_si256();
    __m256i sum31 = _mm256_setzero_si256();

    __m256i sum32 = _mm256_setzero_si256();
    __m256i sum33 = _mm256_setzero_si256();
    __m256i sum34 = _mm256_setzero_si256();
    __m256i sum35 = _mm256_setzero_si256();
    __m256i sum36 = _mm256_setzero_si256();
    __m256i sum37 = _mm256_setzero_si256();
    __m256i sum38 = _mm256_setzero_si256();
    __m256i sum39 = _mm256_setzero_si256();

    __m256i sum40 = _mm256_setzero_si256();
    __m256i sum41 = _mm256_setzero_si256();
    __m256i sum42 = _mm256_setzero_si256();
    __m256i sum43 = _mm256_setzero_si256();
    __m256i sum44 = _mm256_setzero_si256();
    __m256i sum45 = _mm256_setzero_si256();
    __m256i sum46 = _mm256_setzero_si256();
    __m256i sum47 = _mm256_setzero_si256();

    __m256i sum48 = _mm256_setzero_si256();
    __m256i sum49 = _mm256_setzero_si256();
    __m256i sum50 = _mm256_setzero_si256();
    __m256i sum51 = _mm256_setzero_si256();
    __m256i sum52 = _mm256_setzero_si256();
    __m256i sum53 = _mm256_setzero_si256();
    __m256i sum54 = _mm256_setzero_si256();
    __m256i sum55 = _mm256_setzero_si256();

    __m256i sum56 = _mm256_setzero_si256();
    __m256i sum57 = _mm256_setzero_si256();
    __m256i sum58 = _mm256_setzero_si256();
    __m256i sum59 = _mm256_setzero_si256();
    __m256i sum60 = _mm256_setzero_si256();
    __m256i sum61 = _mm256_setzero_si256();
    __m256i sum62 = _mm256_setzero_si256();
    __m256i sum63 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //the 0 row
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));

        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        sum0 = _mm256_add_epi32(mc0, sum0);

        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mb1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb1 + 16)));
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma0_h, mb1_h));
        sum1 = _mm256_add_epi32(mc1, sum1);

        mb2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        mb2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb2 + 16)));
        mc2 = _mm256_madd_epi16(ma0_l, mb2_l);
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma0_h, mb2_h));
        sum2 = _mm256_add_epi32(mc2, sum2);

        mb3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));
        mb3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb3 + 16)));
        mc3 = _mm256_madd_epi16(ma0_l, mb3_l);
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma0_h, mb3_h));
        sum3 = _mm256_add_epi32(mc3, sum3);

        mb4_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb4));
        mb4_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb4 + 16)));
        mc4 = _mm256_madd_epi16(ma0_l, mb4_l);
        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma0_h, mb4_h));
        sum4 = _mm256_add_epi32(mc4, sum4);

        mb5_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb5));
        mb5_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb5 + 16)));
        mc5 = _mm256_madd_epi16(ma0_l, mb5_l);
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma0_h, mb5_h));
        sum5 = _mm256_add_epi32(mc5, sum5);

        mb6_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb6));
        mb6_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb6 + 16)));
        mc6 = _mm256_madd_epi16(ma0_l, mb6_l);
        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma0_h, mb6_h));
        sum6 = _mm256_add_epi32(mc6, sum6);

        mb7_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb7));
        mb7_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb7 + 16)));
        mc7 = _mm256_madd_epi16(ma0_l, mb7_l);
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma0_h, mb7_h));
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        ma1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa1 + 16)));

        mc8 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc8 = _mm256_add_epi32(mc8, _mm256_madd_epi16(ma1_h, mb0_h));
        sum8 = _mm256_add_epi32(mc8, sum8);

        mc9 = _mm256_madd_epi16(ma1_l, mb1_l);
        mc9 = _mm256_add_epi32(mc9, _mm256_madd_epi16(ma1_h, mb1_h));
        sum9 = _mm256_add_epi32(mc9, sum9);

        mc10 = _mm256_madd_epi16(ma1_l, mb2_l);
        mc10 = _mm256_add_epi32(mc10, _mm256_madd_epi16(ma1_h, mb2_h));
        sum10 = _mm256_add_epi32(mc10, sum10);

        mc11 = _mm256_madd_epi16(ma1_l, mb3_l);
        mc11 = _mm256_add_epi32(mc11, _mm256_madd_epi16(ma1_h, mb3_h));
        sum11 = _mm256_add_epi32(mc11, sum11);

        mc12 = _mm256_madd_epi16(ma1_l, mb4_l);
        mc12 = _mm256_add_epi32(mc12, _mm256_madd_epi16(ma1_h, mb4_h));
        sum12 = _mm256_add_epi32(mc12, sum12);

        mc13 = _mm256_madd_epi16(ma1_l, mb5_l);
        mc13 = _mm256_add_epi32(mc13, _mm256_madd_epi16(ma1_h, mb5_h));
        sum13 = _mm256_add_epi32(mc13, sum13);

        mc14 = _mm256_madd_epi16(ma1_l, mb6_l);
        mc14 = _mm256_add_epi32(mc14, _mm256_madd_epi16(ma1_h, mb6_h));
        sum14 = _mm256_add_epi32(mc14, sum14);

        mc15 = _mm256_madd_epi16(ma1_l, mb7_l);
        mc15 = _mm256_add_epi32(mc15, _mm256_madd_epi16(ma1_h, mb7_h));
        sum15 = _mm256_add_epi32(mc15, sum15);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));
        ma2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa2 + 16)));

        mc0 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma2_h, mb0_h));
        sum16 = _mm256_add_epi32(mc0, sum16);

        mc1 = _mm256_madd_epi16(ma2_l, mb1_l);
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma2_h, mb1_h));
        sum17 = _mm256_add_epi32(mc1, sum17);

        mc2 = _mm256_madd_epi16(ma2_l, mb2_l);
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma2_h, mb2_h));
        sum18 = _mm256_add_epi32(mc2, sum18);

        mc3 = _mm256_madd_epi16(ma2_l, mb3_l);
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma2_h, mb3_h));
        sum19 = _mm256_add_epi32(mc3, sum19);

        mc4 = _mm256_madd_epi16(ma2_l, mb4_l);
        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma2_h, mb4_h));
        sum20 = _mm256_add_epi32(mc4, sum20);

        mc5 = _mm256_madd_epi16(ma2_l, mb5_l);
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma2_h, mb5_h));
        sum21 = _mm256_add_epi32(mc5, sum21);

        mc6 = _mm256_madd_epi16(ma2_l, mb6_l);
        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma2_h, mb6_h));
        sum22 = _mm256_add_epi32(mc6, sum22);

        mc7 = _mm256_madd_epi16(ma2_l, mb7_l);
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma2_h, mb7_h));
        sum23 = _mm256_add_epi32(mc7, sum23);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));
        ma3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa3 + 16)));

        mc8 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc8 = _mm256_add_epi32(mc8, _mm256_madd_epi16(ma3_h, mb0_h));
        sum24 = _mm256_add_epi32(mc8, sum24);

        mc9 = _mm256_madd_epi16(ma3_l, mb1_l);
        mc9 = _mm256_add_epi32(mc9, _mm256_madd_epi16(ma3_h, mb1_h));
        sum25 = _mm256_add_epi32(mc9, sum25);

        mc10 = _mm256_madd_epi16(ma3_l, mb2_l);
        mc10 = _mm256_add_epi32(mc10, _mm256_madd_epi16(ma3_h, mb2_h));
        sum26 = _mm256_add_epi32(mc10, sum26);

        mc11 = _mm256_madd_epi16(ma3_l, mb3_l);
        mc11 = _mm256_add_epi32(mc11, _mm256_madd_epi16(ma3_h, mb3_h));
        sum27 = _mm256_add_epi32(mc11, sum27);

        mc12 = _mm256_madd_epi16(ma3_l, mb4_l);
        mc12 = _mm256_add_epi32(mc12, _mm256_madd_epi16(ma3_h, mb4_h));
        sum28 = _mm256_add_epi32(mc12, sum28);

        mc13 = _mm256_madd_epi16(ma3_l, mb5_l);
        mc13 = _mm256_add_epi32(mc13, _mm256_madd_epi16(ma3_h, mb5_h));
        sum29 = _mm256_add_epi32(mc13, sum29);

        mc14 = _mm256_madd_epi16(ma3_l, mb6_l);
        mc14 = _mm256_add_epi32(mc14, _mm256_madd_epi16(ma3_h, mb6_h));
        sum30 = _mm256_add_epi32(mc14, sum30);

        mc15 = _mm256_madd_epi16(ma3_l, mb7_l);
        mc15 = _mm256_add_epi32(mc15, _mm256_madd_epi16(ma3_h, mb7_h));
        sum31 = _mm256_add_epi32(mc15, sum31);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa4));
        ma4_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa4 + 16)));

        mc0 = _mm256_madd_epi16(ma4_l, mb0_l);
        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma4_h, mb0_h));
        sum32 = _mm256_add_epi32(mc0, sum32);

        mc1 = _mm256_madd_epi16(ma4_l, mb1_l);
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma4_h, mb1_h));
        sum33 = _mm256_add_epi32(mc1, sum33);

        mc2 = _mm256_madd_epi16(ma4_l, mb2_l);
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma4_h, mb2_h));
        sum34 = _mm256_add_epi32(mc2, sum34);

        mc3 = _mm256_madd_epi16(ma4_l, mb3_l);
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma4_h, mb3_h));
        sum35 = _mm256_add_epi32(mc3, sum35);

        mc4 = _mm256_madd_epi16(ma4_l, mb4_l);
        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma4_h, mb4_h));
        sum36 = _mm256_add_epi32(mc4, sum36);

        mc5 = _mm256_madd_epi16(ma4_l, mb5_l);
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma4_h, mb5_h));
        sum37 = _mm256_add_epi32(mc5, sum37);

        mc6 = _mm256_madd_epi16(ma4_l, mb6_l);
        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma4_h, mb6_h));
        sum38 = _mm256_add_epi32(mc6, sum38);

        mc7 = _mm256_madd_epi16(ma4_l, mb7_l);
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma4_h, mb7_h));
        sum39 = _mm256_add_epi32(mc7, sum39);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa5));
        ma5_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa5 + 16)));

        mc8 = _mm256_madd_epi16(ma5_l, mb0_l);
        mc8 = _mm256_add_epi32(mc8, _mm256_madd_epi16(ma5_h, mb0_h));
        sum40 = _mm256_add_epi32(mc8, sum40);

        mc9 = _mm256_madd_epi16(ma5_l, mb1_l);
        mc9 = _mm256_add_epi32(mc9, _mm256_madd_epi16(ma5_h, mb1_h));
        sum41 = _mm256_add_epi32(mc9, sum41);

        mc10 = _mm256_madd_epi16(ma5_l, mb2_l);
        mc10 = _mm256_add_epi32(mc10, _mm256_madd_epi16(ma5_h, mb2_h));
        sum42 = _mm256_add_epi32(mc10, sum42);

        mc11 = _mm256_madd_epi16(ma5_l, mb3_l);
        mc11 = _mm256_add_epi32(mc11, _mm256_madd_epi16(ma5_h, mb3_h));
        sum43 = _mm256_add_epi32(mc11, sum43);

        mc12 = _mm256_madd_epi16(ma5_l, mb4_l);
        mc12 = _mm256_add_epi32(mc12, _mm256_madd_epi16(ma5_h, mb4_h));
        sum44 = _mm256_add_epi32(mc12, sum44);

        mc13 = _mm256_madd_epi16(ma5_l, mb5_l);
        mc13 = _mm256_add_epi32(mc13, _mm256_madd_epi16(ma5_h, mb5_h));
        sum45 = _mm256_add_epi32(mc13, sum45);

        mc14 = _mm256_madd_epi16(ma5_l, mb6_l);
        mc14 = _mm256_add_epi32(mc14, _mm256_madd_epi16(ma5_h, mb6_h));
        sum46 = _mm256_add_epi32(mc14, sum46);

        mc15 = _mm256_madd_epi16(ma5_l, mb7_l);
        mc15 = _mm256_add_epi32(mc15, _mm256_madd_epi16(ma5_h, mb7_h));
        sum47 = _mm256_add_epi32(mc15, sum47);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa6));
        ma6_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa6 + 16)));

        mc0 = _mm256_madd_epi16(ma6_l, mb0_l);
        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma6_h, mb0_h));
        sum48 = _mm256_add_epi32(mc0, sum48);

        mc1 = _mm256_madd_epi16(ma6_l, mb1_l);
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma6_h, mb1_h));
        sum49 = _mm256_add_epi32(mc1, sum49);

        mc2 = _mm256_madd_epi16(ma6_l, mb2_l);
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma6_h, mb2_h));
        sum50 = _mm256_add_epi32(mc2, sum50);

        mc3 = _mm256_madd_epi16(ma6_l, mb3_l);
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma6_h, mb3_h));
        sum51 = _mm256_add_epi32(mc3, sum51);

        mc4 = _mm256_madd_epi16(ma6_l, mb4_l);
        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma6_h, mb4_h));
        sum52 = _mm256_add_epi32(mc4, sum52);

        mc5 = _mm256_madd_epi16(ma6_l, mb5_l);
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma6_h, mb5_h));
        sum53 = _mm256_add_epi32(mc5, sum53);

        mc6 = _mm256_madd_epi16(ma6_l, mb6_l);
        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma6_h, mb6_h));
        sum54 = _mm256_add_epi32(mc6, sum54);

        mc7 = _mm256_madd_epi16(ma6_l, mb7_l);
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma6_h, mb7_h));
        sum55 = _mm256_add_epi32(mc7, sum55);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa7));
        ma7_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa7 + 16)));

        mc8 = _mm256_madd_epi16(ma7_l, mb0_l);
        mc8 = _mm256_add_epi32(mc8, _mm256_madd_epi16(ma7_h, mb0_h));
        sum56 = _mm256_add_epi32(mc8, sum56);

        mc9 = _mm256_madd_epi16(ma7_l, mb1_l);
        mc9 = _mm256_add_epi32(mc9, _mm256_madd_epi16(ma7_h, mb1_h));
        sum57 = _mm256_add_epi32(mc9, sum57);

        mc10 = _mm256_madd_epi16(ma7_l, mb2_l);
        mc10 = _mm256_add_epi32(mc10, _mm256_madd_epi16(ma7_h, mb2_h));
        sum58 = _mm256_add_epi32(mc10, sum58);

        mc11 = _mm256_madd_epi16(ma7_l, mb3_l);
        mc11 = _mm256_add_epi32(mc11, _mm256_madd_epi16(ma7_h, mb3_h));
        sum59 = _mm256_add_epi32(mc11, sum59);

        mc12 = _mm256_madd_epi16(ma7_l, mb4_l);
        mc12 = _mm256_add_epi32(mc12, _mm256_madd_epi16(ma7_h, mb4_h));
        sum60 = _mm256_add_epi32(mc12, sum60);

        mc13 = _mm256_madd_epi16(ma7_l, mb5_l);
        mc13 = _mm256_add_epi32(mc13, _mm256_madd_epi16(ma7_h, mb5_h));
        sum61 = _mm256_add_epi32(mc13, sum61);

        mc14 = _mm256_madd_epi16(ma7_l, mb6_l);
        mc14 = _mm256_add_epi32(mc14, _mm256_madd_epi16(ma7_h, mb6_h));
        sum62 = _mm256_add_epi32(mc14, sum62);

        mc15 = _mm256_madd_epi16(ma7_l, mb7_l);
        mc15 = _mm256_add_epi32(mc15, _mm256_madd_epi16(ma7_h, mb7_h));
        sum63 = _mm256_add_epi32(mc15, sum63);

        _mm_prefetch((char*) pa0 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pa1 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pa2 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pa3 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pa4 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pa5 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pa6 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pa7 + 32, _MM_HINT_T0);

        _mm_prefetch((char*) pb0 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pb1 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pb2 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pb3 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pb4 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pb5 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pb6 + 32, _MM_HINT_T0);
        _mm_prefetch((char*) pb7 + 32, _MM_HINT_T0);

        pa0 += 32;
        pa1 += 32;
        pa2 += 32;
        pa3 += 32;
        pa4 += 32;
        pa5 += 32;
        pa6 += 32;
        pa7 += 32;

        pb0 += 32;
        pb1 += 32;
        pb2 += 32;
        pb3 += 32;
        pb4 += 32;
        pb5 += 32;
        pb6 += 32;
        pb7 += 32;
    }

    //leftover
    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mb2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        mb3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));
        mb4_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb4));
        mb5_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb5));
        mb6_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb6));
        mb7_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb7));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma0_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma0_l, mb3_l);
        mc4 = _mm256_madd_epi16(ma0_l, mb4_l);
        mc5 = _mm256_madd_epi16(ma0_l, mb5_l);
        mc6 = _mm256_madd_epi16(ma0_l, mb6_l);
        mc7 = _mm256_madd_epi16(ma0_l, mb7_l);

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);
        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));

        mc0 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma1_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma1_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb3_l);
        mc4 = _mm256_madd_epi16(ma1_l, mb4_l);
        mc5 = _mm256_madd_epi16(ma1_l, mb5_l);
        mc6 = _mm256_madd_epi16(ma1_l, mb6_l);
        mc7 = _mm256_madd_epi16(ma1_l, mb7_l);

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);
        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);
        sum12 = _mm256_add_epi32(mc4, sum12);
        sum13 = _mm256_add_epi32(mc5, sum13);
        sum14 = _mm256_add_epi32(mc6, sum14);
        sum15 = _mm256_add_epi32(mc7, sum15);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));

        mc0 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma2_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma2_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma2_l, mb3_l);
        mc4 = _mm256_madd_epi16(ma2_l, mb4_l);
        mc5 = _mm256_madd_epi16(ma2_l, mb5_l);
        mc6 = _mm256_madd_epi16(ma2_l, mb6_l);
        mc7 = _mm256_madd_epi16(ma2_l, mb7_l);

        sum16 = _mm256_add_epi32(mc0, sum16);
        sum17 = _mm256_add_epi32(mc1, sum17);
        sum18 = _mm256_add_epi32(mc2, sum18);
        sum19 = _mm256_add_epi32(mc3, sum19);
        sum20 = _mm256_add_epi32(mc4, sum20);
        sum21 = _mm256_add_epi32(mc5, sum21);
        sum22 = _mm256_add_epi32(mc6, sum22);
        sum23 = _mm256_add_epi32(mc7, sum23);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));

        mc0 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma3_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma3_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma3_l, mb3_l);
        mc4 = _mm256_madd_epi16(ma3_l, mb4_l);
        mc5 = _mm256_madd_epi16(ma3_l, mb5_l);
        mc6 = _mm256_madd_epi16(ma3_l, mb6_l);
        mc7 = _mm256_madd_epi16(ma3_l, mb7_l);

        sum24 = _mm256_add_epi32(mc0, sum24);
        sum25 = _mm256_add_epi32(mc1, sum25);
        sum26 = _mm256_add_epi32(mc2, sum26);
        sum27 = _mm256_add_epi32(mc3, sum27);
        sum28 = _mm256_add_epi32(mc4, sum28);
        sum29 = _mm256_add_epi32(mc5, sum29);
        sum30 = _mm256_add_epi32(mc6, sum30);
        sum31 = _mm256_add_epi32(mc7, sum31);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa4));

        mc0 = _mm256_madd_epi16(ma4_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma4_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma4_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma4_l, mb3_l);
        mc4 = _mm256_madd_epi16(ma4_l, mb4_l);
        mc5 = _mm256_madd_epi16(ma4_l, mb5_l);
        mc6 = _mm256_madd_epi16(ma4_l, mb6_l);
        mc7 = _mm256_madd_epi16(ma4_l, mb7_l);

        sum32 = _mm256_add_epi32(mc0, sum32);
        sum33 = _mm256_add_epi32(mc1, sum33);
        sum34 = _mm256_add_epi32(mc2, sum34);
        sum35 = _mm256_add_epi32(mc3, sum35);
        sum36 = _mm256_add_epi32(mc4, sum36);
        sum37 = _mm256_add_epi32(mc5, sum37);
        sum38 = _mm256_add_epi32(mc6, sum38);
        sum39 = _mm256_add_epi32(mc7, sum39);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa5));

        mc0 = _mm256_madd_epi16(ma5_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma5_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma5_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma5_l, mb3_l);
        mc4 = _mm256_madd_epi16(ma5_l, mb4_l);
        mc5 = _mm256_madd_epi16(ma5_l, mb5_l);
        mc6 = _mm256_madd_epi16(ma5_l, mb6_l);
        mc7 = _mm256_madd_epi16(ma5_l, mb7_l);

        sum40 = _mm256_add_epi32(mc0, sum40);
        sum41 = _mm256_add_epi32(mc1, sum41);
        sum42 = _mm256_add_epi32(mc2, sum42);
        sum43 = _mm256_add_epi32(mc3, sum43);
        sum44 = _mm256_add_epi32(mc4, sum44);
        sum45 = _mm256_add_epi32(mc5, sum45);
        sum46 = _mm256_add_epi32(mc6, sum46);
        sum47 = _mm256_add_epi32(mc7, sum47);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa6));

        mc0 = _mm256_madd_epi16(ma6_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma6_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma6_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma6_l, mb3_l);
        mc4 = _mm256_madd_epi16(ma6_l, mb4_l);
        mc5 = _mm256_madd_epi16(ma6_l, mb5_l);
        mc6 = _mm256_madd_epi16(ma6_l, mb6_l);
        mc7 = _mm256_madd_epi16(ma6_l, mb7_l);

        sum48 = _mm256_add_epi32(mc0, sum48);
        sum49 = _mm256_add_epi32(mc1, sum49);
        sum50 = _mm256_add_epi32(mc2, sum50);
        sum51 = _mm256_add_epi32(mc3, sum51);
        sum52 = _mm256_add_epi32(mc4, sum52);
        sum53 = _mm256_add_epi32(mc5, sum53);
        sum54 = _mm256_add_epi32(mc6, sum54);
        sum55 = _mm256_add_epi32(mc7, sum55);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa7));

        mc0 = _mm256_madd_epi16(ma7_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma7_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma7_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma7_l, mb3_l);
        mc4 = _mm256_madd_epi16(ma7_l, mb4_l);
        mc5 = _mm256_madd_epi16(ma7_l, mb5_l);
        mc6 = _mm256_madd_epi16(ma7_l, mb6_l);
        mc7 = _mm256_madd_epi16(ma7_l, mb7_l);

        sum56 = _mm256_add_epi32(mc0, sum56);
        sum57 = _mm256_add_epi32(mc1, sum57);
        sum58 = _mm256_add_epi32(mc2, sum58);
        sum59 = _mm256_add_epi32(mc3, sum59);
        sum60 = _mm256_add_epi32(mc4, sum60);
        sum61 = _mm256_add_epi32(mc5, sum61);
        sum62 = _mm256_add_epi32(mc6, sum62);
        sum63 = _mm256_add_epi32(mc7, sum63);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;
        pa4 += 16;
        pa5 += 16;
        pa6 += 16;
        pa7 += 16;

        pb0 += 16;
        pb1 += 16;
        pb2 += 16;
        pb3 += 16;
        pb4 += 16;
        pb5 += 16;
        pb6 += 16;
        pb7 += 16;
    }

    if (0x08 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb1));
        mb2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb2));
        mb3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb3));
        mb4_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb4));
        mb5_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb5));
        mb6_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb6));
        mb7_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb7));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma0_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma0_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma0_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma0_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma0_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma0_l, mb7_l);

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);
        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa1));

        mc0 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma1_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma1_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma1_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma1_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma1_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma1_l, mb7_l);

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);
        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);
        sum12 = _mm256_add_epi32(mc4, sum12);
        sum13 = _mm256_add_epi32(mc5, sum13);
        sum14 = _mm256_add_epi32(mc6, sum14);
        sum15 = _mm256_add_epi32(mc7, sum15);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa2));

        mc0 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma2_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma2_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma2_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma2_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma2_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma2_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma2_l, mb7_l);

        sum16 = _mm256_add_epi32(mc0, sum16);
        sum17 = _mm256_add_epi32(mc1, sum17);
        sum18 = _mm256_add_epi32(mc2, sum18);
        sum19 = _mm256_add_epi32(mc3, sum19);
        sum20 = _mm256_add_epi32(mc4, sum20);
        sum21 = _mm256_add_epi32(mc5, sum21);
        sum22 = _mm256_add_epi32(mc6, sum22);
        sum23 = _mm256_add_epi32(mc7, sum23);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa3));

        mc0 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma3_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma3_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma3_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma3_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma3_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma3_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma3_l, mb7_l);

        sum24 = _mm256_add_epi32(mc0, sum24);
        sum25 = _mm256_add_epi32(mc1, sum25);
        sum26 = _mm256_add_epi32(mc2, sum26);
        sum27 = _mm256_add_epi32(mc3, sum27);
        sum28 = _mm256_add_epi32(mc4, sum28);
        sum29 = _mm256_add_epi32(mc5, sum29);
        sum30 = _mm256_add_epi32(mc6, sum30);
        sum31 = _mm256_add_epi32(mc7, sum31);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa4));

        mc0 = _mm256_mullo_epi32(ma4_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma4_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma4_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma4_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma4_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma4_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma4_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma4_l, mb7_l);

        sum32 = _mm256_add_epi32(mc0, sum32);
        sum33 = _mm256_add_epi32(mc1, sum33);
        sum34 = _mm256_add_epi32(mc2, sum34);
        sum35 = _mm256_add_epi32(mc3, sum35);
        sum36 = _mm256_add_epi32(mc4, sum36);
        sum37 = _mm256_add_epi32(mc5, sum37);
        sum38 = _mm256_add_epi32(mc6, sum38);
        sum39 = _mm256_add_epi32(mc7, sum39);


        //the 5 row
        ma5_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa5));

        mc0 = _mm256_mullo_epi32(ma5_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma5_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma5_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma5_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma5_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma5_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma5_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma5_l, mb7_l);

        sum40 = _mm256_add_epi32(mc0, sum40);
        sum41 = _mm256_add_epi32(mc1, sum41);
        sum42 = _mm256_add_epi32(mc2, sum42);
        sum43 = _mm256_add_epi32(mc3, sum43);
        sum44 = _mm256_add_epi32(mc4, sum44);
        sum45 = _mm256_add_epi32(mc5, sum45);
        sum46 = _mm256_add_epi32(mc6, sum46);
        sum47 = _mm256_add_epi32(mc7, sum47);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa6));

        mc0 = _mm256_mullo_epi32(ma6_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma6_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma6_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma6_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma6_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma6_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma6_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma6_l, mb7_l);

        sum48 = _mm256_add_epi32(mc0, sum48);
        sum49 = _mm256_add_epi32(mc1, sum49);
        sum50 = _mm256_add_epi32(mc2, sum50);
        sum51 = _mm256_add_epi32(mc3, sum51);
        sum52 = _mm256_add_epi32(mc4, sum52);
        sum53 = _mm256_add_epi32(mc5, sum53);
        sum54 = _mm256_add_epi32(mc6, sum54);
        sum55 = _mm256_add_epi32(mc7, sum55);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa7));

        mc0 = _mm256_mullo_epi32(ma7_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma7_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma7_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma7_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma7_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma7_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma7_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma7_l, mb7_l);

        sum56 = _mm256_add_epi32(mc0, sum56);
        sum57 = _mm256_add_epi32(mc1, sum57);
        sum58 = _mm256_add_epi32(mc2, sum58);
        sum59 = _mm256_add_epi32(mc3, sum59);
        sum60 = _mm256_add_epi32(mc4, sum60);
        sum61 = _mm256_add_epi32(mc5, sum61);
        sum62 = _mm256_add_epi32(mc6, sum62);
        sum63 = _mm256_add_epi32(mc7, sum63);

        pa0 += 8;
        pa1 += 8;
        pa2 += 8;
        pa3 += 8;
        pa4 += 8;
        pa5 += 8;
        pa6 += 8;
        pa7 += 8;

        pb0 += 8;
        pb1 += 8;
        pb2 += 8;
        pb3 += 8;
        pb4 += 8;
        pb5 += 8;
        pb6 += 8;
        pb7 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga4[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga5[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga6[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga7[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb4[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb5[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb6[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb7[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];
            ga1[i] = pa1[i];
            ga2[i] = pa2[i];
            ga3[i] = pa3[i];
            ga4[i] = pa4[i];
            ga5[i] = pa5[i];
            ga6[i] = pa6[i];
            ga7[i] = pa7[i];

            gb0[i] = pb0[i];
            gb1[i] = pb1[i];
            gb2[i] = pb2[i];
            gb3[i] = pb3[i];
            gb4[i] = pb4[i];
            gb5[i] = pb5[i];
            gb6[i] = pb6[i];
            gb7[i] = pb7[i];
        }

        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb1));
        mb2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb2));
        mb3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb3));
        mb4_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb4));
        mb5_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb5));
        mb6_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb6));
        mb7_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb7));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma0_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma0_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma0_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma0_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma0_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma0_l, mb7_l);

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);
        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga1));

        mc0 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma1_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma1_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma1_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma1_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma1_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma1_l, mb7_l);

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);
        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);
        sum12 = _mm256_add_epi32(mc4, sum12);
        sum13 = _mm256_add_epi32(mc5, sum13);
        sum14 = _mm256_add_epi32(mc6, sum14);
        sum15 = _mm256_add_epi32(mc7, sum15);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga2));

        mc0 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma2_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma2_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma2_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma2_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma2_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma2_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma2_l, mb7_l);

        sum16 = _mm256_add_epi32(mc0, sum16);
        sum17 = _mm256_add_epi32(mc1, sum17);
        sum18 = _mm256_add_epi32(mc2, sum18);
        sum19 = _mm256_add_epi32(mc3, sum19);
        sum20 = _mm256_add_epi32(mc4, sum20);
        sum21 = _mm256_add_epi32(mc5, sum21);
        sum22 = _mm256_add_epi32(mc6, sum22);
        sum23 = _mm256_add_epi32(mc7, sum23);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga3));

        mc0 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma3_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma3_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma3_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma3_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma3_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma3_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma3_l, mb7_l);

        sum24 = _mm256_add_epi32(mc0, sum24);
        sum25 = _mm256_add_epi32(mc1, sum25);
        sum26 = _mm256_add_epi32(mc2, sum26);
        sum27 = _mm256_add_epi32(mc3, sum27);
        sum28 = _mm256_add_epi32(mc4, sum28);
        sum29 = _mm256_add_epi32(mc5, sum29);
        sum30 = _mm256_add_epi32(mc6, sum30);
        sum31 = _mm256_add_epi32(mc7, sum31);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga4));

        mc0 = _mm256_mullo_epi32(ma4_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma4_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma4_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma4_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma4_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma4_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma4_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma4_l, mb7_l);

        sum32 = _mm256_add_epi32(mc0, sum32);
        sum33 = _mm256_add_epi32(mc1, sum33);
        sum34 = _mm256_add_epi32(mc2, sum34);
        sum35 = _mm256_add_epi32(mc3, sum35);
        sum36 = _mm256_add_epi32(mc4, sum36);
        sum37 = _mm256_add_epi32(mc5, sum37);
        sum38 = _mm256_add_epi32(mc6, sum38);
        sum39 = _mm256_add_epi32(mc7, sum39);


        //the 5 row
        ma5_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga5));

        mc0 = _mm256_mullo_epi32(ma5_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma5_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma5_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma5_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma5_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma5_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma5_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma5_l, mb7_l);

        sum40 = _mm256_add_epi32(mc0, sum40);
        sum41 = _mm256_add_epi32(mc1, sum41);
        sum42 = _mm256_add_epi32(mc2, sum42);
        sum43 = _mm256_add_epi32(mc3, sum43);
        sum44 = _mm256_add_epi32(mc4, sum44);
        sum45 = _mm256_add_epi32(mc5, sum45);
        sum46 = _mm256_add_epi32(mc6, sum46);
        sum47 = _mm256_add_epi32(mc7, sum47);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga6));

        mc0 = _mm256_mullo_epi32(ma6_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma6_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma6_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma6_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma6_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma6_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma6_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma6_l, mb7_l);

        sum48 = _mm256_add_epi32(mc0, sum48);
        sum49 = _mm256_add_epi32(mc1, sum49);
        sum50 = _mm256_add_epi32(mc2, sum50);
        sum51 = _mm256_add_epi32(mc3, sum51);
        sum52 = _mm256_add_epi32(mc4, sum52);
        sum53 = _mm256_add_epi32(mc5, sum53);
        sum54 = _mm256_add_epi32(mc6, sum54);
        sum55 = _mm256_add_epi32(mc7, sum55);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga7));

        mc0 = _mm256_mullo_epi32(ma7_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma7_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma7_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma7_l, mb3_l);
        mc4 = _mm256_mullo_epi32(ma7_l, mb4_l);
        mc5 = _mm256_mullo_epi32(ma7_l, mb5_l);
        mc6 = _mm256_mullo_epi32(ma7_l, mb6_l);
        mc7 = _mm256_mullo_epi32(ma7_l, mb7_l);

        sum56 = _mm256_add_epi32(mc0, sum56);
        sum57 = _mm256_add_epi32(mc1, sum57);
        sum58 = _mm256_add_epi32(mc2, sum58);
        sum59 = _mm256_add_epi32(mc3, sum59);
        sum60 = _mm256_add_epi32(mc4, sum60);
        sum61 = _mm256_add_epi32(mc5, sum61);
        sum62 = _mm256_add_epi32(mc6, sum62);
        sum63 = _mm256_add_epi32(mc7, sum63);
    }

    //store
    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum2 = _mm256_hadd_epi32(sum2, sum3);
    sum0 = _mm256_hadd_epi32(sum0, sum2);

    sum4 = _mm256_hadd_epi32(sum4, sum5);
    sum6 = _mm256_hadd_epi32(sum6, sum7);
    sum4 = _mm256_hadd_epi32(sum4, sum6);

    sum0 = _mm256_add_epi32(_mm256_permute2x128_si256(sum0, sum4, 0x20),
                            _mm256_permute2x128_si256(sum0, sum4, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1] = _mm256_extract_epi32(sum0, 1);
    pc0[2] = _mm256_extract_epi32(sum0, 2);
    pc0[3] = _mm256_extract_epi32(sum0, 3);
    pc0[4] = _mm256_extract_epi32(sum0, 4);
    pc0[5] = _mm256_extract_epi32(sum0, 5);
    pc0[6] = _mm256_extract_epi32(sum0, 6);
    pc0[7] = _mm256_extract_epi32(sum0, 7);

    //the 1 row
    sum8 = _mm256_hadd_epi32(sum8, sum9);
    sum10 = _mm256_hadd_epi32(sum10, sum11);
    sum8 = _mm256_hadd_epi32(sum8, sum10);

    sum12 = _mm256_hadd_epi32(sum12, sum13);
    sum14 = _mm256_hadd_epi32(sum14, sum15);
    sum12 = _mm256_hadd_epi32(sum12, sum14);

    sum8 = _mm256_add_epi32(_mm256_permute2x128_si256(sum8, sum12, 0x20),
                            _mm256_permute2x128_si256(sum8, sum12, 0x31));

    pc1[0] = _mm256_extract_epi32(sum8, 0);
    pc1[1] = _mm256_extract_epi32(sum8, 1);
    pc1[2] = _mm256_extract_epi32(sum8, 2);
    pc1[3] = _mm256_extract_epi32(sum8, 3);
    pc1[4] = _mm256_extract_epi32(sum8, 4);
    pc1[5] = _mm256_extract_epi32(sum8, 5);
    pc1[6] = _mm256_extract_epi32(sum8, 6);
    pc1[7] = _mm256_extract_epi32(sum8, 7);

    //the 2 row
    sum16 = _mm256_hadd_epi32(sum16, sum17);
    sum18 = _mm256_hadd_epi32(sum18, sum19);
    sum16 = _mm256_hadd_epi32(sum16, sum18);

    sum20 = _mm256_hadd_epi32(sum20, sum21);
    sum22 = _mm256_hadd_epi32(sum22, sum23);
    sum20 = _mm256_hadd_epi32(sum20, sum22);

    sum16 = _mm256_add_epi32(_mm256_permute2x128_si256(sum16, sum20, 0x20),
                             _mm256_permute2x128_si256(sum16, sum20, 0x31));

    pc2[0] = _mm256_extract_epi32(sum16, 0);
    pc2[1] = _mm256_extract_epi32(sum16, 1);
    pc2[2] = _mm256_extract_epi32(sum16, 2);
    pc2[3] = _mm256_extract_epi32(sum16, 3);
    pc2[4] = _mm256_extract_epi32(sum16, 4);
    pc2[5] = _mm256_extract_epi32(sum16, 5);
    pc2[6] = _mm256_extract_epi32(sum16, 6);
    pc2[7] = _mm256_extract_epi32(sum16, 7);

    //the 3 row
    sum24 = _mm256_hadd_epi32(sum24, sum25);
    sum26 = _mm256_hadd_epi32(sum26, sum27);
    sum24 = _mm256_hadd_epi32(sum24, sum26);

    sum28 = _mm256_hadd_epi32(sum28, sum29);
    sum30 = _mm256_hadd_epi32(sum30, sum31);
    sum28 = _mm256_hadd_epi32(sum28, sum30);

    sum24 = _mm256_add_epi32(_mm256_permute2x128_si256(sum24, sum28, 0x20),
                             _mm256_permute2x128_si256(sum24, sum28, 0x31));

    pc3[0] = _mm256_extract_epi32(sum24, 0);
    pc3[1] = _mm256_extract_epi32(sum24, 1);
    pc3[2] = _mm256_extract_epi32(sum24, 2);
    pc3[3] = _mm256_extract_epi32(sum24, 3);
    pc3[4] = _mm256_extract_epi32(sum24, 4);
    pc3[5] = _mm256_extract_epi32(sum24, 5);
    pc3[6] = _mm256_extract_epi32(sum24, 6);
    pc3[7] = _mm256_extract_epi32(sum24, 7);

    //the 4 row
    sum32 = _mm256_hadd_epi32(sum32, sum33);
    sum34 = _mm256_hadd_epi32(sum34, sum35);
    sum32 = _mm256_hadd_epi32(sum32, sum34);

    sum36 = _mm256_hadd_epi32(sum36, sum37);
    sum38 = _mm256_hadd_epi32(sum38, sum39);
    sum36 = _mm256_hadd_epi32(sum36, sum38);

    sum32 = _mm256_add_epi32(_mm256_permute2x128_si256(sum32, sum36, 0x20),
                             _mm256_permute2x128_si256(sum32, sum36, 0x31));

    pc4[0] = _mm256_extract_epi32(sum32, 0);
    pc4[1] = _mm256_extract_epi32(sum32, 1);
    pc4[2] = _mm256_extract_epi32(sum32, 2);
    pc4[3] = _mm256_extract_epi32(sum32, 3);
    pc4[4] = _mm256_extract_epi32(sum32, 4);
    pc4[5] = _mm256_extract_epi32(sum32, 5);
    pc4[6] = _mm256_extract_epi32(sum32, 6);
    pc4[7] = _mm256_extract_epi32(sum32, 7);

    //the 5 row
    sum40 = _mm256_hadd_epi32(sum40, sum41);
    sum42 = _mm256_hadd_epi32(sum42, sum43);
    sum40 = _mm256_hadd_epi32(sum40, sum42);

    sum44 = _mm256_hadd_epi32(sum44, sum45);
    sum46 = _mm256_hadd_epi32(sum46, sum47);
    sum44 = _mm256_hadd_epi32(sum44, sum46);

    sum40 = _mm256_add_epi32(_mm256_permute2x128_si256(sum40, sum44, 0x20),
                             _mm256_permute2x128_si256(sum40, sum44, 0x31));

    pc5[0] = _mm256_extract_epi32(sum40, 0);
    pc5[1] = _mm256_extract_epi32(sum40, 1);
    pc5[2] = _mm256_extract_epi32(sum40, 2);
    pc5[3] = _mm256_extract_epi32(sum40, 3);
    pc5[4] = _mm256_extract_epi32(sum40, 4);
    pc5[5] = _mm256_extract_epi32(sum40, 5);
    pc5[6] = _mm256_extract_epi32(sum40, 6);
    pc5[7] = _mm256_extract_epi32(sum40, 7);

    //the 6 row
    sum48 = _mm256_hadd_epi32(sum48, sum49);
    sum50 = _mm256_hadd_epi32(sum50, sum51);
    sum48 = _mm256_hadd_epi32(sum48, sum50);

    sum52 = _mm256_hadd_epi32(sum52, sum53);
    sum54 = _mm256_hadd_epi32(sum54, sum55);
    sum52 = _mm256_hadd_epi32(sum52, sum54);

    sum48 = _mm256_add_epi32(_mm256_permute2x128_si256(sum48, sum52, 0x20),
                             _mm256_permute2x128_si256(sum48, sum52, 0x31));

    pc6[0] = _mm256_extract_epi32(sum48, 0);
    pc6[1] = _mm256_extract_epi32(sum48, 1);
    pc6[2] = _mm256_extract_epi32(sum48, 2);
    pc6[3] = _mm256_extract_epi32(sum48, 3);
    pc6[4] = _mm256_extract_epi32(sum48, 4);
    pc6[5] = _mm256_extract_epi32(sum48, 5);
    pc6[6] = _mm256_extract_epi32(sum48, 6);
    pc6[7] = _mm256_extract_epi32(sum48, 7);

    //the 7 row
    sum56 = _mm256_hadd_epi32(sum56, sum57);
    sum58 = _mm256_hadd_epi32(sum58, sum59);
    sum56 = _mm256_hadd_epi32(sum56, sum58);

    sum60 = _mm256_hadd_epi32(sum60, sum61);
    sum62 = _mm256_hadd_epi32(sum62, sum63);
    sum60 = _mm256_hadd_epi32(sum60, sum62);

    sum56 = _mm256_add_epi32(_mm256_permute2x128_si256(sum56, sum60, 0x20),
                             _mm256_permute2x128_si256(sum56, sum60, 0x31));

    pc7[0] = _mm256_extract_epi32(sum56, 0);
    pc7[1] = _mm256_extract_epi32(sum56, 1);
    pc7[2] = _mm256_extract_epi32(sum56, 2);
    pc7[3] = _mm256_extract_epi32(sum56, 3);
    pc7[4] = _mm256_extract_epi32(sum56, 4);
    pc7[5] = _mm256_extract_epi32(sum56, 5);
    pc7[6] = _mm256_extract_epi32(sum56, 6);
    pc7[7] = _mm256_extract_epi32(sum56, 7);
}

inline void block8x4_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int32_t stride) {
    //printf("block8x4_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;
    const int8_t* pa4 = pa0 + 4 * lda;
    const int8_t* pa5 = pa0 + 5 * lda;
    const int8_t* pa6 = pa0 + 6 * lda;
    const int8_t* pa7 = pa0 + 7 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;
    int* pc4 = c + 4 * ldc;
    int* pc5 = c + 5 * ldc;
    int* pc6 = c + 6 * ldc;
    int* pc7 = c + 7 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma1_l;
    __m256i ma2_l;
    __m256i ma3_l;
    __m256i ma4_l;
    __m256i ma5_l;
    __m256i ma6_l;
    __m256i ma7_l;
    __m256i ma0_h;
    __m256i ma1_h;
    __m256i ma2_h;
    __m256i ma3_h;
    __m256i ma4_h;
    __m256i ma5_h;
    __m256i ma6_h;
    __m256i ma7_h;

    __m256i mb0_l;
    __m256i mb1_l;
    __m256i mb2_l;
    __m256i mb3_l;
    __m256i mb0_h;
    __m256i mb1_h;
    __m256i mb2_h;
    __m256i mb3_h;

    __m256i mc0;
    __m256i mc1;
    __m256i mc2;
    __m256i mc3;
    __m256i mc4;
    __m256i mc5;
    __m256i mc6;
    __m256i mc7;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    __m256i sum8 = _mm256_setzero_si256();
    __m256i sum9 = _mm256_setzero_si256();
    __m256i sum10 = _mm256_setzero_si256();
    __m256i sum11 = _mm256_setzero_si256();
    __m256i sum12 = _mm256_setzero_si256();
    __m256i sum13 = _mm256_setzero_si256();
    __m256i sum14 = _mm256_setzero_si256();
    __m256i sum15 = _mm256_setzero_si256();

    __m256i sum16 = _mm256_setzero_si256();
    __m256i sum17 = _mm256_setzero_si256();
    __m256i sum18 = _mm256_setzero_si256();
    __m256i sum19 = _mm256_setzero_si256();
    __m256i sum20 = _mm256_setzero_si256();
    __m256i sum21 = _mm256_setzero_si256();
    __m256i sum22 = _mm256_setzero_si256();
    __m256i sum23 = _mm256_setzero_si256();

    __m256i sum24 = _mm256_setzero_si256();
    __m256i sum25 = _mm256_setzero_si256();
    __m256i sum26 = _mm256_setzero_si256();
    __m256i sum27 = _mm256_setzero_si256();
    __m256i sum28 = _mm256_setzero_si256();
    __m256i sum29 = _mm256_setzero_si256();
    __m256i sum30 = _mm256_setzero_si256();
    __m256i sum31 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));

        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mb1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb1 + 16)));

        mb2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        mb2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb2 + 16)));

        mb3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));
        mb3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb3 + 16)));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma0_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma0_l, mb3_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma0_h, mb1_h));
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma0_h, mb2_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma0_h, mb3_h));

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        ma1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa1 + 16)));

        mc4 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma1_l, mb1_l);
        mc6 = _mm256_madd_epi16(ma1_l, mb2_l);
        mc7 = _mm256_madd_epi16(ma1_l, mb3_l);

        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma1_h, mb0_h));
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma1_h, mb1_h));
        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma1_h, mb2_h));
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma1_h, mb3_h));

        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));
        ma2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa2 + 16)));

        mc0 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma2_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma2_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma2_l, mb3_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma2_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma2_h, mb1_h));
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma2_h, mb2_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma2_h, mb3_h));

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);
        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));
        ma3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa3 + 16)));

        mc4 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma3_l, mb1_l);
        mc6 = _mm256_madd_epi16(ma3_l, mb2_l);
        mc7 = _mm256_madd_epi16(ma3_l, mb3_l);

        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma3_h, mb0_h));
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma3_h, mb1_h));
        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma3_h, mb2_h));
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma3_h, mb3_h));

        sum12 = _mm256_add_epi32(mc4, sum12);
        sum13 = _mm256_add_epi32(mc5, sum13);
        sum14 = _mm256_add_epi32(mc6, sum14);
        sum15 = _mm256_add_epi32(mc7, sum15);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa4));
        ma4_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa4 + 16)));

        mc0 = _mm256_madd_epi16(ma4_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma4_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma4_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma4_l, mb3_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma4_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma4_h, mb1_h));
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma4_h, mb2_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma4_h, mb3_h));

        sum16 = _mm256_add_epi32(mc0, sum16);
        sum17 = _mm256_add_epi32(mc1, sum17);
        sum18 = _mm256_add_epi32(mc2, sum18);
        sum19 = _mm256_add_epi32(mc3, sum19);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa5));
        ma5_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa5 + 16)));

        mc4 = _mm256_madd_epi16(ma5_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma5_l, mb1_l);
        mc6 = _mm256_madd_epi16(ma5_l, mb2_l);
        mc7 = _mm256_madd_epi16(ma5_l, mb3_l);

        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma5_h, mb0_h));
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma5_h, mb1_h));
        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma5_h, mb2_h));
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma5_h, mb3_h));

        sum20 = _mm256_add_epi32(mc4, sum20);
        sum21 = _mm256_add_epi32(mc5, sum21);
        sum22 = _mm256_add_epi32(mc6, sum22);
        sum23 = _mm256_add_epi32(mc7, sum23);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa6));
        ma6_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa6 + 16)));

        mc0 = _mm256_madd_epi16(ma6_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma6_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma6_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma6_l, mb3_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma6_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma6_h, mb1_h));
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma6_h, mb2_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma6_h, mb3_h));

        sum24 = _mm256_add_epi32(mc0, sum24);
        sum25 = _mm256_add_epi32(mc1, sum25);
        sum26 = _mm256_add_epi32(mc2, sum26);
        sum27 = _mm256_add_epi32(mc3, sum27);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa7));
        ma7_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa7 + 16)));

        mc4 = _mm256_madd_epi16(ma7_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma7_l, mb1_l);
        mc6 = _mm256_madd_epi16(ma7_l, mb2_l);
        mc7 = _mm256_madd_epi16(ma7_l, mb3_l);

        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma7_h, mb0_h));
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma7_h, mb1_h));
        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma7_h, mb2_h));
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma7_h, mb3_h));

        sum28 = _mm256_add_epi32(mc4, sum28);
        sum29 = _mm256_add_epi32(mc5, sum29);
        sum30 = _mm256_add_epi32(mc6, sum30);
        sum31 = _mm256_add_epi32(mc7, sum31);

        pa0 += 32;
        pa1 += 32;
        pa2 += 32;
        pa3 += 32;
        pa4 += 32;
        pa5 += 32;
        pa6 += 32;
        pa7 += 32;

        pb0 += 32;
        pb1 += 32;
        pb2 += 32;
        pb3 += 32;
    }

    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mb2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        mb3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma0_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma0_l, mb3_l);

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));

        mc0 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma1_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma1_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb3_l);

        sum4 = _mm256_add_epi32(mc0, sum4);
        sum5 = _mm256_add_epi32(mc1, sum5);
        sum6 = _mm256_add_epi32(mc2, sum6);
        sum7 = _mm256_add_epi32(mc3, sum7);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));

        mc0 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma2_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma2_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma2_l, mb3_l);

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);
        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));

        mc0 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma3_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma3_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma3_l, mb3_l);

        sum12 = _mm256_add_epi32(mc0, sum12);
        sum13 = _mm256_add_epi32(mc1, sum13);
        sum14 = _mm256_add_epi32(mc2, sum14);
        sum15 = _mm256_add_epi32(mc3, sum15);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa4));

        mc0 = _mm256_madd_epi16(ma4_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma4_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma4_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma4_l, mb3_l);

        sum16 = _mm256_add_epi32(mc0, sum16);
        sum17 = _mm256_add_epi32(mc1, sum17);
        sum18 = _mm256_add_epi32(mc2, sum18);
        sum19 = _mm256_add_epi32(mc3, sum19);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa5));
        ma5_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa5 + 16)));

        mc0 = _mm256_madd_epi16(ma5_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma5_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma5_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma5_l, mb3_l);

        sum20 = _mm256_add_epi32(mc0, sum20);
        sum21 = _mm256_add_epi32(mc1, sum21);
        sum22 = _mm256_add_epi32(mc2, sum22);
        sum23 = _mm256_add_epi32(mc3, sum23);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa6));
        ma6_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa6 + 16)));

        mc0 = _mm256_madd_epi16(ma6_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma6_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma6_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma6_l, mb3_l);

        sum24 = _mm256_add_epi32(mc0, sum24);
        sum25 = _mm256_add_epi32(mc1, sum25);
        sum26 = _mm256_add_epi32(mc2, sum26);
        sum27 = _mm256_add_epi32(mc3, sum27);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa7));
        ma7_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa7 + 16)));

        mc0 = _mm256_madd_epi16(ma7_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma7_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma7_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma7_l, mb3_l);

        sum28 = _mm256_add_epi32(mc0, sum28);
        sum29 = _mm256_add_epi32(mc1, sum29);
        sum30 = _mm256_add_epi32(mc2, sum30);
        sum31 = _mm256_add_epi32(mc3, sum31);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;
        pa4 += 16;
        pa5 += 16;
        pa6 += 16;
        pa7 += 16;

        pb0 += 16;
        pb1 += 16;
        pb2 += 16;
        pb3 += 16;
    }

    if (0x08 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb1));
        mb2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb2));
        mb3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb3));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma0_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma0_l, mb3_l);

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa1));

        mc0 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma1_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma1_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb3_l);

        sum4 = _mm256_add_epi32(mc0, sum4);
        sum5 = _mm256_add_epi32(mc1, sum5);
        sum6 = _mm256_add_epi32(mc2, sum6);
        sum7 = _mm256_add_epi32(mc3, sum7);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa2));

        mc0 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma2_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma2_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma2_l, mb3_l);

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);
        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa3));

        mc0 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma3_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma3_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma3_l, mb3_l);

        sum12 = _mm256_add_epi32(mc0, sum12);
        sum13 = _mm256_add_epi32(mc1, sum13);
        sum14 = _mm256_add_epi32(mc2, sum14);
        sum15 = _mm256_add_epi32(mc3, sum15);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa4));

        mc0 = _mm256_mullo_epi32(ma4_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma4_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma4_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma4_l, mb3_l);

        sum16 = _mm256_add_epi32(mc0, sum16);
        sum17 = _mm256_add_epi32(mc1, sum17);
        sum18 = _mm256_add_epi32(mc2, sum18);
        sum19 = _mm256_add_epi32(mc3, sum19);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa5));

        mc0 = _mm256_mullo_epi32(ma5_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma5_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma5_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma5_l, mb3_l);

        sum20 = _mm256_add_epi32(mc0, sum20);
        sum21 = _mm256_add_epi32(mc1, sum21);
        sum22 = _mm256_add_epi32(mc2, sum22);
        sum23 = _mm256_add_epi32(mc3, sum23);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa6));

        mc0 = _mm256_mullo_epi32(ma6_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma6_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma6_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma6_l, mb3_l);

        sum24 = _mm256_add_epi32(mc0, sum24);
        sum25 = _mm256_add_epi32(mc1, sum25);
        sum26 = _mm256_add_epi32(mc2, sum26);
        sum27 = _mm256_add_epi32(mc3, sum27);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa7));

        mc0 = _mm256_mullo_epi32(ma7_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma7_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma7_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma7_l, mb3_l);

        sum28 = _mm256_add_epi32(mc0, sum28);
        sum29 = _mm256_add_epi32(mc1, sum29);
        sum30 = _mm256_add_epi32(mc2, sum30);
        sum31 = _mm256_add_epi32(mc3, sum31);

        pa0 += 8;
        pa1 += 8;
        pa2 += 8;
        pa3 += 8;
        pa4 += 8;
        pa5 += 8;
        pa6 += 8;
        pa7 += 8;

        pb0 += 8;
        pb1 += 8;
        pb2 += 8;
        pb3 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga4[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga5[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga6[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga7[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];
            ga1[i] = pa1[i];
            ga2[i] = pa2[i];
            ga3[i] = pa3[i];
            ga4[i] = pa4[i];
            ga5[i] = pa5[i];
            ga6[i] = pa6[i];
            ga7[i] = pa7[i];

            gb0[i] = pb0[i];
            gb1[i] = pb1[i];
            gb2[i] = pb2[i];
            gb3[i] = pb3[i];
        }

        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb1));
        mb2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb2));
        mb3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb3));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma0_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma0_l, mb3_l);

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga1));

        mc0 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma1_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma1_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb3_l);

        sum4 = _mm256_add_epi32(mc0, sum4);
        sum5 = _mm256_add_epi32(mc1, sum5);
        sum6 = _mm256_add_epi32(mc2, sum6);
        sum7 = _mm256_add_epi32(mc3, sum7);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga2));

        mc0 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma2_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma2_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma2_l, mb3_l);

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);
        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga3));

        mc0 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma3_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma3_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma3_l, mb3_l);

        sum12 = _mm256_add_epi32(mc0, sum12);
        sum13 = _mm256_add_epi32(mc1, sum13);
        sum14 = _mm256_add_epi32(mc2, sum14);
        sum15 = _mm256_add_epi32(mc3, sum15);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga4));

        mc0 = _mm256_mullo_epi32(ma4_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma4_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma4_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma4_l, mb3_l);

        sum16 = _mm256_add_epi32(mc0, sum16);
        sum17 = _mm256_add_epi32(mc1, sum17);
        sum18 = _mm256_add_epi32(mc2, sum18);
        sum19 = _mm256_add_epi32(mc3, sum19);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga5));

        mc0 = _mm256_mullo_epi32(ma5_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma5_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma5_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma5_l, mb3_l);

        sum20 = _mm256_add_epi32(mc0, sum20);
        sum21 = _mm256_add_epi32(mc1, sum21);
        sum22 = _mm256_add_epi32(mc2, sum22);
        sum23 = _mm256_add_epi32(mc3, sum23);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga6));

        mc0 = _mm256_mullo_epi32(ma6_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma6_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma6_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma6_l, mb3_l);

        sum24 = _mm256_add_epi32(mc0, sum24);
        sum25 = _mm256_add_epi32(mc1, sum25);
        sum26 = _mm256_add_epi32(mc2, sum26);
        sum27 = _mm256_add_epi32(mc3, sum27);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga7));

        mc0 = _mm256_mullo_epi32(ma7_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma7_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma7_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma7_l, mb3_l);

        sum28 = _mm256_add_epi32(mc0, sum28);
        sum29 = _mm256_add_epi32(mc1, sum29);
        sum30 = _mm256_add_epi32(mc2, sum30);
        sum31 = _mm256_add_epi32(mc3, sum31);
    }

    //store
    __m256i zero = _mm256_setzero_si256();

    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum2 = _mm256_hadd_epi32(sum2, sum3);
    sum0 = _mm256_hadd_epi32(sum0, sum2);

    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1 * stride] = _mm256_extract_epi32(sum0, 1);
    pc0[2 * stride] = _mm256_extract_epi32(sum0, 2);
    pc0[3 * stride] = _mm256_extract_epi32(sum0, 3);

    //the 1 row
    sum4 = _mm256_hadd_epi32(sum4, sum5);
    sum6 = _mm256_hadd_epi32(sum6, sum7);
    sum4 = _mm256_hadd_epi32(sum4, sum6);

    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum4, 0);
    pc1[1 * stride] = _mm256_extract_epi32(sum4, 1);
    pc1[2 * stride] = _mm256_extract_epi32(sum4, 2);
    pc1[3 * stride] = _mm256_extract_epi32(sum4, 3);

    //the 2 row
    sum8 = _mm256_hadd_epi32(sum8, sum9);
    sum10 = _mm256_hadd_epi32(sum10, sum11);
    sum8 = _mm256_hadd_epi32(sum8, sum10);
    sum8 = _mm256_add_epi32(sum8, _mm256_permute2x128_si256(sum8, zero, 0x31));

    pc2[0] = _mm256_extract_epi32(sum8, 0);
    pc2[1 * stride] = _mm256_extract_epi32(sum8, 1);
    pc2[2 * stride] = _mm256_extract_epi32(sum8, 2);
    pc2[3 * stride] = _mm256_extract_epi32(sum8, 3);

    //the 3 row
    sum12 = _mm256_hadd_epi32(sum12, sum13);
    sum14 = _mm256_hadd_epi32(sum14, sum15);
    sum12 = _mm256_hadd_epi32(sum12, sum14);
    sum12 = _mm256_add_epi32(sum12, _mm256_permute2x128_si256(sum12, zero, 0x31));
    pc3[0] = _mm256_extract_epi32(sum12, 0);
    pc3[1 * stride] = _mm256_extract_epi32(sum12, 1);
    pc3[2 * stride] = _mm256_extract_epi32(sum12, 2);
    pc3[3 * stride] = _mm256_extract_epi32(sum12, 3);

    //the 4 row
    sum16 = _mm256_hadd_epi32(sum16, sum17);
    sum18 = _mm256_hadd_epi32(sum18, sum19);
    sum16 = _mm256_hadd_epi32(sum16, sum18);
    sum16 = _mm256_add_epi32(sum16, _mm256_permute2x128_si256(sum16, zero, 0x31));
    pc4[0] = _mm256_extract_epi32(sum16, 0);
    pc4[1 * stride] = _mm256_extract_epi32(sum16, 1);
    pc4[2 * stride] = _mm256_extract_epi32(sum16, 2);
    pc4[3 * stride] = _mm256_extract_epi32(sum16, 3);

    //the 5 row
    sum20 = _mm256_hadd_epi32(sum20, sum21);
    sum22 = _mm256_hadd_epi32(sum22, sum23);
    sum20 = _mm256_hadd_epi32(sum20, sum22);
    sum20 = _mm256_add_epi32(sum20, _mm256_permute2x128_si256(sum20, zero, 0x31));
    pc5[0] = _mm256_extract_epi32(sum20, 0);
    pc5[1 * stride] = _mm256_extract_epi32(sum20, 1);
    pc5[2 * stride] = _mm256_extract_epi32(sum20, 2);
    pc5[3 * stride] = _mm256_extract_epi32(sum20, 3);

    //the 6 row
    sum24 = _mm256_hadd_epi32(sum24, sum25);
    sum26 = _mm256_hadd_epi32(sum26, sum27);
    sum24 = _mm256_hadd_epi32(sum24, sum26);
    sum24 = _mm256_add_epi32(sum24, _mm256_permute2x128_si256(sum24, zero, 0x31));
    pc6[0] = _mm256_extract_epi32(sum24, 0);
    pc6[1 * stride] = _mm256_extract_epi32(sum24, 1);
    pc6[2 * stride] = _mm256_extract_epi32(sum24, 2);
    pc6[3 * stride] = _mm256_extract_epi32(sum24, 3);

    //the 7 row
    sum28 = _mm256_hadd_epi32(sum28, sum29);
    sum30 = _mm256_hadd_epi32(sum30, sum31);
    sum28 = _mm256_hadd_epi32(sum28, sum30);
    sum28 = _mm256_add_epi32(sum28, _mm256_permute2x128_si256(sum28, zero, 0x31));
    pc7[0] = _mm256_extract_epi32(sum28, 0);
    pc7[1 * stride] = _mm256_extract_epi32(sum28, 1);
    pc7[2 * stride] = _mm256_extract_epi32(sum28, 2);
    pc7[3 * stride] = _mm256_extract_epi32(sum28, 3);

}

inline void block8x2_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int32_t stride) {
    //printf("block8x2_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;
    const int8_t* pa4 = pa0 + 4 * lda;
    const int8_t* pa5 = pa0 + 5 * lda;
    const int8_t* pa6 = pa0 + 6 * lda;
    const int8_t* pa7 = pa0 + 7 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;
    int* pc4 = c + 4 * ldc;
    int* pc5 = c + 5 * ldc;
    int* pc6 = c + 6 * ldc;
    int* pc7 = c + 7 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma1_l;
    __m256i ma2_l;
    __m256i ma3_l;
    __m256i ma4_l;
    __m256i ma5_l;
    __m256i ma6_l;
    __m256i ma7_l;
    __m256i ma0_h;
    __m256i ma1_h;
    __m256i ma2_h;
    __m256i ma3_h;
    __m256i ma4_h;
    __m256i ma5_h;
    __m256i ma6_h;
    __m256i ma7_h;

    __m256i mb0_l;
    __m256i mb1_l;
    __m256i mb0_h;
    __m256i mb1_h;

    __m256i mc0;
    __m256i mc1;
    __m256i mc2;
    __m256i mc3;
    __m256i mc4;
    __m256i mc5;
    __m256i mc6;
    __m256i mc7;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    __m256i sum8 = _mm256_setzero_si256();
    __m256i sum9 = _mm256_setzero_si256();
    __m256i sum10 = _mm256_setzero_si256();
    __m256i sum11 = _mm256_setzero_si256();
    __m256i sum12 = _mm256_setzero_si256();
    __m256i sum13 = _mm256_setzero_si256();
    __m256i sum14 = _mm256_setzero_si256();
    __m256i sum15 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));

        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mb1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb1 + 16)));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma0_h, mb1_h));

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        ma1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa1 + 16)));

        mc2 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb1_l);

        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma1_h, mb0_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma1_h, mb1_h));

        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));
        ma2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa2 + 16)));

        mc4 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma2_l, mb1_l);

        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma2_h, mb0_h));
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma2_h, mb1_h));

        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));
        ma3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa3 + 16)));

        mc6 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc7 = _mm256_madd_epi16(ma3_l, mb1_l);

        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma3_h, mb0_h));
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma3_h, mb1_h));

        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa4));
        ma4_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa4 + 16)));

        mc0 = _mm256_madd_epi16(ma4_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma4_l, mb1_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma4_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma4_h, mb1_h));

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa5));
        ma5_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa5 + 16)));

        mc2 = _mm256_madd_epi16(ma5_l, mb0_l);
        mc3 = _mm256_madd_epi16(ma5_l, mb1_l);

        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma5_h, mb0_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma5_h, mb1_h));

        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa6));
        ma6_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa6 + 16)));

        mc4 = _mm256_madd_epi16(ma6_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma6_l, mb1_l);

        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma6_h, mb0_h));
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma6_h, mb1_h));

        sum12 = _mm256_add_epi32(mc4, sum12);
        sum13 = _mm256_add_epi32(mc5, sum13);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa7));
        ma7_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa7 + 16)));

        mc6 = _mm256_madd_epi16(ma7_l, mb0_l);
        mc7 = _mm256_madd_epi16(ma7_l, mb1_l);

        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma7_h, mb0_h));
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma7_h, mb1_h));

        sum14 = _mm256_add_epi32(mc6, sum14);
        sum15 = _mm256_add_epi32(mc7, sum15);

        pa0 += 32;
        pa1 += 32;
        pa2 += 32;
        pa3 += 32;
        pa4 += 32;
        pa5 += 32;
        pa6 += 32;
        pa7 += 32;

        pb0 += 32;
        pb1 += 32;
    }

    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));

        mc2 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb1_l);

        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));

        mc4 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma2_l, mb1_l);

        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));

        mc6 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc7 = _mm256_madd_epi16(ma3_l, mb1_l);

        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa4));

        mc0 = _mm256_madd_epi16(ma4_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma4_l, mb1_l);

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa5));

        mc2 = _mm256_madd_epi16(ma5_l, mb0_l);
        mc3 = _mm256_madd_epi16(ma5_l, mb1_l);

        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa6));

        mc4 = _mm256_madd_epi16(ma6_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma6_l, mb1_l);

        sum12 = _mm256_add_epi32(mc4, sum12);
        sum13 = _mm256_add_epi32(mc5, sum13);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa7));

        mc6 = _mm256_madd_epi16(ma7_l, mb0_l);
        mc7 = _mm256_madd_epi16(ma7_l, mb1_l);

        sum14 = _mm256_add_epi32(mc6, sum14);
        sum15 = _mm256_add_epi32(mc7, sum15);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;
        pa4 += 16;
        pa5 += 16;
        pa6 += 16;
        pa7 += 16;

        pb0 += 16;
        pb1 += 16;
    }

    if (0x08 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb1));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa1));

        mc2 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb1_l);

        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa2));

        mc4 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc5 = _mm256_mullo_epi32(ma2_l, mb1_l);

        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa3));

        mc6 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc7 = _mm256_mullo_epi32(ma3_l, mb1_l);

        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa4));

        mc0 = _mm256_mullo_epi32(ma4_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma4_l, mb1_l);

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa5));

        mc2 = _mm256_mullo_epi32(ma5_l, mb0_l);
        mc3 = _mm256_mullo_epi32(ma5_l, mb1_l);

        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa6));

        mc4 = _mm256_mullo_epi32(ma6_l, mb0_l);
        mc5 = _mm256_mullo_epi32(ma6_l, mb1_l);

        sum12 = _mm256_add_epi32(mc4, sum12);
        sum13 = _mm256_add_epi32(mc5, sum13);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa7));

        mc6 = _mm256_mullo_epi32(ma7_l, mb0_l);
        mc7 = _mm256_mullo_epi32(ma7_l, mb1_l);

        sum14 = _mm256_add_epi32(mc6, sum14);
        sum15 = _mm256_add_epi32(mc7, sum15);

        pa0 += 8;
        pa1 += 8;
        pa2 += 8;
        pa3 += 8;
        pa4 += 8;
        pa5 += 8;
        pa6 += 8;
        pa7 += 8;

        pb0 += 8;
        pb1 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga4[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga5[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga6[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga7[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];
            ga1[i] = pa1[i];
            ga2[i] = pa2[i];
            ga3[i] = pa3[i];
            ga4[i] = pa4[i];
            ga5[i] = pa5[i];
            ga6[i] = pa6[i];
            ga7[i] = pa7[i];

            gb0[i] = pb0[i];
            gb1[i] = pb1[i];
        }

        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb1));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga1));

        mc2 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb1_l);

        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga2));

        mc4 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc5 = _mm256_mullo_epi32(ma2_l, mb1_l);

        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga3));

        mc6 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc7 = _mm256_mullo_epi32(ma3_l, mb1_l);

        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga4));

        mc0 = _mm256_mullo_epi32(ma4_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma4_l, mb1_l);

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga5));

        mc2 = _mm256_mullo_epi32(ma5_l, mb0_l);
        mc3 = _mm256_mullo_epi32(ma5_l, mb1_l);

        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga6));

        mc4 = _mm256_mullo_epi32(ma6_l, mb0_l);
        mc5 = _mm256_mullo_epi32(ma6_l, mb1_l);

        sum12 = _mm256_add_epi32(mc4, sum12);
        sum13 = _mm256_add_epi32(mc5, sum13);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga7));

        mc6 = _mm256_mullo_epi32(ma7_l, mb0_l);
        mc7 = _mm256_mullo_epi32(ma7_l, mb1_l);

        sum14 = _mm256_add_epi32(mc6, sum14);
        sum15 = _mm256_add_epi32(mc7, sum15);
    }

    //store
    __m256i zero = _mm256_setzero_si256();

    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum0 = _mm256_hadd_epi32(sum0, zero);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1 * stride] = _mm256_extract_epi32(sum0, 1);

    //the 1 row
    sum2 = _mm256_hadd_epi32(sum2, sum3);
    sum2 = _mm256_hadd_epi32(sum2, zero);
    sum2 = _mm256_add_epi32(sum2, _mm256_permute2x128_si256(sum2, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum2, 0);
    pc1[1 * stride] = _mm256_extract_epi32(sum2, 1);

    //the 2 row
    sum4 = _mm256_hadd_epi32(sum4, sum5);
    sum4 = _mm256_hadd_epi32(sum4, zero);
    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    pc2[0] = _mm256_extract_epi32(sum4, 0);
    pc2[1 * stride] = _mm256_extract_epi32(sum4, 1);

    //the 3 row
    sum6 = _mm256_hadd_epi32(sum6, sum7);
    sum6 = _mm256_hadd_epi32(sum6, zero);
    sum6 = _mm256_add_epi32(sum6, _mm256_permute2x128_si256(sum6, zero, 0x31));

    pc3[0] = _mm256_extract_epi32(sum6, 0);
    pc3[1 * stride] = _mm256_extract_epi32(sum6, 1);

    //the 4 row
    sum8 = _mm256_hadd_epi32(sum8, sum9);
    sum8 = _mm256_hadd_epi32(sum8, zero);
    sum8 = _mm256_add_epi32(sum8, _mm256_permute2x128_si256(sum8, zero, 0x31));

    pc4[0] = _mm256_extract_epi32(sum8, 0);
    pc4[1 * stride] = _mm256_extract_epi32(sum8, 1);

    //the 5 row
    sum10 = _mm256_hadd_epi32(sum10, sum11);
    sum10 = _mm256_hadd_epi32(sum10, zero);
    sum10 = _mm256_add_epi32(sum10, _mm256_permute2x128_si256(sum10, zero, 0x31));

    pc5[0] = _mm256_extract_epi32(sum10, 0);
    pc5[1 * stride] = _mm256_extract_epi32(sum10, 1);

    //the 6 row
    sum12 = _mm256_hadd_epi32(sum12, sum13);
    sum12 = _mm256_hadd_epi32(sum12, zero);
    sum12 = _mm256_add_epi32(sum12, _mm256_permute2x128_si256(sum12, zero, 0x31));

    pc6[0] = _mm256_extract_epi32(sum12, 0);
    pc6[1 * stride] = _mm256_extract_epi32(sum12, 1);

    //the 7 row
    sum14 = _mm256_hadd_epi32(sum14, sum15);
    sum14 = _mm256_hadd_epi32(sum14, zero);
    sum14 = _mm256_add_epi32(sum14, _mm256_permute2x128_si256(sum14, zero, 0x31));

    pc7[0] = _mm256_extract_epi32(sum14, 0);
    pc7[1 * stride] = _mm256_extract_epi32(sum14, 1);
}

inline void block8x1_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int32_t stride) {
    //printf("block8x1_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;
    const int8_t* pa4 = pa0 + 4 * lda;
    const int8_t* pa5 = pa0 + 5 * lda;
    const int8_t* pa6 = pa0 + 6 * lda;
    const int8_t* pa7 = pa0 + 7 * lda;

    const int8_t* pb0 = b;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;
    int* pc4 = c + 4 * ldc;
    int* pc5 = c + 5 * ldc;
    int* pc6 = c + 6 * ldc;
    int* pc7 = c + 7 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma1_l;
    __m256i ma2_l;
    __m256i ma3_l;
    __m256i ma4_l;
    __m256i ma5_l;
    __m256i ma6_l;
    __m256i ma7_l;
    __m256i ma0_h;
    __m256i ma1_h;
    __m256i ma2_h;
    __m256i ma3_h;
    __m256i ma4_h;
    __m256i ma5_h;
    __m256i ma6_h;
    __m256i ma7_h;

    __m256i mb0_l;
    __m256i mb0_h;

    __m256i mc0;
    __m256i mc1;
    __m256i mc2;
    __m256i mc3;
    __m256i mc4;
    __m256i mc5;
    __m256i mc6;
    __m256i mc7;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        ma1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa1 + 16)));

        mc1 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma1_h, mb0_h));
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));
        ma2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa2 + 16)));

        mc2 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma2_h, mb0_h));
        sum2 = _mm256_add_epi32(mc2, sum2);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));
        ma3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa3 + 16)));

        mc3 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma3_h, mb0_h));
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa4));
        ma4_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa4 + 16)));

        mc4 = _mm256_madd_epi16(ma4_l, mb0_l);
        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma4_h, mb0_h));
        sum4 = _mm256_add_epi32(mc4, sum4);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa5));
        ma5_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa5 + 16)));

        mc5 = _mm256_madd_epi16(ma5_l, mb0_l);
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma5_h, mb0_h));
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa6));
        ma6_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa6 + 16)));

        mc6 = _mm256_madd_epi16(ma6_l, mb0_l);
        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma6_h, mb0_h));
        sum6 = _mm256_add_epi32(mc6, sum6);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa7));
        ma7_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa7 + 16)));

        mc7 = _mm256_madd_epi16(ma7_l, mb0_l);
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma7_h, mb0_h));
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 32;
        pa1 += 32;
        pa2 += 32;
        pa3 += 32;
        pa4 += 32;
        pa5 += 32;
        pa6 += 32;
        pa7 += 32;

        pb0 += 32;
    }

    //leftover
    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        mc1 = _mm256_madd_epi16(ma1_l, mb0_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));
        mc2 = _mm256_madd_epi16(ma2_l, mb0_l);
        sum2 = _mm256_add_epi32(mc2, sum2);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));
        mc3 = _mm256_madd_epi16(ma3_l, mb0_l);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa4));
        mc4 = _mm256_madd_epi16(ma4_l, mb0_l);
        sum4 = _mm256_add_epi32(mc4, sum4);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa5));
        mc5 = _mm256_madd_epi16(ma5_l, mb0_l);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa6));
        mc6 = _mm256_madd_epi16(ma6_l, mb0_l);
        sum6 = _mm256_add_epi32(mc6, sum6);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa7));
        mc7 = _mm256_madd_epi16(ma7_l, mb0_l);
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;
        pa4 += 16;
        pa5 += 16;
        pa6 += 16;
        pa7 += 16;

        pb0 += 16;
    }

    if (0x08 & k_leftover) {
        //a
        __m256i ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));

        //b
        __m256i mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa1));
        mc1 = _mm256_mullo_epi32(ma1_l, mb0_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa2));
        mc2 = _mm256_mullo_epi32(ma2_l, mb0_l);
        sum2 = _mm256_add_epi32(mc2, sum2);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa3));
        mc3 = _mm256_mullo_epi32(ma3_l, mb0_l);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa4));
        mc4 = _mm256_mullo_epi32(ma4_l, mb0_l);
        sum4 = _mm256_add_epi32(mc4, sum4);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa5));
        mc5 = _mm256_mullo_epi32(ma5_l, mb0_l);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa6));
        mc6 = _mm256_mullo_epi32(ma6_l, mb0_l);
        sum6 = _mm256_add_epi32(mc6, sum6);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa7));
        mc7 = _mm256_mullo_epi32(ma7_l, mb0_l);
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 8;
        pa1 += 8;
        pa2 += 8;
        pa3 += 8;
        pa4 += 8;
        pa5 += 8;
        pa6 += 8;
        pa7 += 8;

        pb0 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga4[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga5[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga6[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga7[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];
            ga1[i] = pa1[i];
            ga2[i] = pa2[i];
            ga3[i] = pa3[i];
            ga4[i] = pa4[i];
            ga5[i] = pa5[i];
            ga6[i] = pa6[i];
            ga7[i] = pa7[i];

            gb0[i] = pb0[i];
        }

        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga1));
        mc1 = _mm256_mullo_epi32(ma1_l, mb0_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga2));
        mc2 = _mm256_mullo_epi32(ma2_l, mb0_l);
        sum2 = _mm256_add_epi32(mc2, sum2);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga3));
        mc3 = _mm256_mullo_epi32(ma3_l, mb0_l);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 4 row
        ma4_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga4));
        mc4 = _mm256_mullo_epi32(ma4_l, mb0_l);
        sum4 = _mm256_add_epi32(mc4, sum4);

        //the 5 row
        ma5_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga5));
        mc5 = _mm256_mullo_epi32(ma5_l, mb0_l);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 6 row
        ma6_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga6));
        mc6 = _mm256_mullo_epi32(ma6_l, mb0_l);
        sum6 = _mm256_add_epi32(mc6, sum6);

        //the 7 row
        ma7_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga7));
        mc7 = _mm256_mullo_epi32(ma7_l, mb0_l);
        sum7 = _mm256_add_epi32(mc7, sum7);
    }

    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, sum0, 0x81));
    sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 8));
    sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 4));
    pc0[0] = _mm256_extract_epi32(sum0, 0);

    sum1 = _mm256_add_epi32(sum1, _mm256_permute2x128_si256(sum1, sum1, 0x81));
    sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 8));
    sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 4));
    pc1[0] = _mm256_extract_epi32(sum1, 0);

    sum2 = _mm256_add_epi32(sum2, _mm256_permute2x128_si256(sum2, sum2, 0x81));
    sum2 = _mm256_add_epi32(sum2, _mm256_srli_si256(sum2, 8));
    sum2 = _mm256_add_epi32(sum2, _mm256_srli_si256(sum2, 4));
    pc2[0] = _mm256_extract_epi32(sum2, 0);

    sum3 = _mm256_add_epi32(sum3, _mm256_permute2x128_si256(sum3, sum3, 0x81));
    sum3 = _mm256_add_epi32(sum3, _mm256_srli_si256(sum3, 8));
    sum3 = _mm256_add_epi32(sum3, _mm256_srli_si256(sum3, 4));
    pc3[0] = _mm256_extract_epi32(sum3, 0);

    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, sum4, 0x81));
    sum4 = _mm256_add_epi32(sum4, _mm256_srli_si256(sum4, 8));
    sum4 = _mm256_add_epi32(sum4, _mm256_srli_si256(sum4, 4));
    pc4[0] = _mm256_extract_epi32(sum4, 0);

    sum5 = _mm256_add_epi32(sum5, _mm256_permute2x128_si256(sum5, sum5, 0x81));
    sum5 = _mm256_add_epi32(sum5, _mm256_srli_si256(sum5, 8));
    sum5 = _mm256_add_epi32(sum5, _mm256_srli_si256(sum5, 4));
    pc5[0] = _mm256_extract_epi32(sum5, 0);

    sum6 = _mm256_add_epi32(sum6, _mm256_permute2x128_si256(sum6, sum6, 0x81));
    sum6 = _mm256_add_epi32(sum6, _mm256_srli_si256(sum6, 8));
    sum6 = _mm256_add_epi32(sum6, _mm256_srli_si256(sum6, 4));
    pc6[0] = _mm256_extract_epi32(sum6, 0);

    sum7 = _mm256_add_epi32(sum7, _mm256_permute2x128_si256(sum7, sum7, 0x81));
    sum7 = _mm256_add_epi32(sum7, _mm256_srli_si256(sum7, 8));
    sum7 = _mm256_add_epi32(sum7, _mm256_srli_si256(sum7, 4));
    pc7[0] = _mm256_extract_epi32(sum7, 0);

}

inline void block4x8_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int32_t stride) {
    //printf("block8x4_kernel_avx2\n");
    block8x4_kernel_avx2(k, b, ldb, a, lda, c, stride, ldc);
}

inline void block4x4_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {
    //printf("block4x4_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma1_l;
    __m256i ma2_l;
    __m256i ma3_l;
    __m256i ma0_h;
    __m256i ma1_h;
    __m256i ma2_h;
    __m256i ma3_h;

    __m256i mb0_l;
    __m256i mb1_l;
    __m256i mb2_l;
    __m256i mb3_l;
    __m256i mb0_h;
    __m256i mb1_h;
    __m256i mb2_h;
    __m256i mb3_h;

    __m256i mc0;
    __m256i mc1;
    __m256i mc2;
    __m256i mc3;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    __m256i sum8 = _mm256_setzero_si256();
    __m256i sum9 = _mm256_setzero_si256();
    __m256i sum10 = _mm256_setzero_si256();
    __m256i sum11 = _mm256_setzero_si256();
    __m256i sum12 = _mm256_setzero_si256();
    __m256i sum13 = _mm256_setzero_si256();
    __m256i sum14 = _mm256_setzero_si256();
    __m256i sum15 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));

        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mb1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb1 + 16)));

        mb2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        mb2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb2 + 16)));

        mb3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));
        mb3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb3 + 16)));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma0_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma0_l, mb3_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma0_h, mb1_h));
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma0_h, mb2_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma0_h, mb3_h));

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        ma1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa1 + 16)));

        mc0 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma1_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma1_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb3_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma1_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma1_h, mb1_h));
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma1_h, mb2_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma1_h, mb3_h));

        sum4 = _mm256_add_epi32(mc0, sum4);
        sum5 = _mm256_add_epi32(mc1, sum5);
        sum6 = _mm256_add_epi32(mc2, sum6);
        sum7 = _mm256_add_epi32(mc3, sum7);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));
        ma2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa2 + 16)));

        mc0 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma2_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma2_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma2_l, mb3_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma2_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma2_h, mb1_h));
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma2_h, mb2_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma2_h, mb3_h));

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);
        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));
        ma3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa3 + 16)));

        mc0 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma3_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma3_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma3_l, mb3_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma3_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma3_h, mb1_h));
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma3_h, mb2_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma3_h, mb3_h));

        sum12 = _mm256_add_epi32(mc0, sum12);
        sum13 = _mm256_add_epi32(mc1, sum13);
        sum14 = _mm256_add_epi32(mc2, sum14);
        sum15 = _mm256_add_epi32(mc3, sum15);

        pa0 += 32;
        pa1 += 32;
        pa2 += 32;
        pa3 += 32;

        pb0 += 32;
        pb1 += 32;
        pb2 += 32;
        pb3 += 32;
    }

    //leftover
    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mb2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        mb3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma0_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma0_l, mb3_l);

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));

        mc0 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma1_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma1_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb3_l);

        sum4 = _mm256_add_epi32(mc0, sum4);
        sum5 = _mm256_add_epi32(mc1, sum5);
        sum6 = _mm256_add_epi32(mc2, sum6);
        sum7 = _mm256_add_epi32(mc3, sum7);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));

        mc0 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma2_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma2_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma2_l, mb3_l);

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);
        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));

        mc0 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma3_l, mb1_l);
        mc2 = _mm256_madd_epi16(ma3_l, mb2_l);
        mc3 = _mm256_madd_epi16(ma3_l, mb3_l);

        sum12 = _mm256_add_epi32(mc0, sum12);
        sum13 = _mm256_add_epi32(mc1, sum13);
        sum14 = _mm256_add_epi32(mc2, sum14);
        sum15 = _mm256_add_epi32(mc3, sum15);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;

        pb0 += 16;
        pb1 += 16;
        pb2 += 16;
        pb3 += 16;
    }

    if (0x08 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb1));
        mb2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb2));
        mb3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb3));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma0_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma0_l, mb3_l);

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa1));

        mc0 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma1_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma1_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb3_l);

        sum4 = _mm256_add_epi32(mc0, sum4);
        sum5 = _mm256_add_epi32(mc1, sum5);
        sum6 = _mm256_add_epi32(mc2, sum6);
        sum7 = _mm256_add_epi32(mc3, sum7);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa2));

        mc0 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma2_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma2_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma2_l, mb3_l);

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);
        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa3));

        mc0 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma3_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma3_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma3_l, mb3_l);

        sum12 = _mm256_add_epi32(mc0, sum12);
        sum13 = _mm256_add_epi32(mc1, sum13);
        sum14 = _mm256_add_epi32(mc2, sum14);
        sum15 = _mm256_add_epi32(mc3, sum15);

        pa0 += 8;
        pa1 += 8;
        pa2 += 8;
        pa3 += 8;

        pb0 += 8;
        pb1 += 8;
        pb2 += 8;
        pb3 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];
            ga1[i] = pa1[i];
            ga2[i] = pa2[i];
            ga3[i] = pa3[i];

            gb0[i] = pb0[i];
            gb1[i] = pb1[i];
            gb2[i] = pb2[i];
            gb3[i] = pb3[i];
        }

        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb1));
        mb2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb2));
        mb3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb3));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma0_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma0_l, mb3_l);

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga1));

        mc0 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma1_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma1_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb3_l);

        sum4 = _mm256_add_epi32(mc0, sum4);
        sum5 = _mm256_add_epi32(mc1, sum5);
        sum6 = _mm256_add_epi32(mc2, sum6);
        sum7 = _mm256_add_epi32(mc3, sum7);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga2));

        mc0 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma2_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma2_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma2_l, mb3_l);

        sum8 = _mm256_add_epi32(mc0, sum8);
        sum9 = _mm256_add_epi32(mc1, sum9);
        sum10 = _mm256_add_epi32(mc2, sum10);
        sum11 = _mm256_add_epi32(mc3, sum11);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga3));

        mc0 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma3_l, mb1_l);
        mc2 = _mm256_mullo_epi32(ma3_l, mb2_l);
        mc3 = _mm256_mullo_epi32(ma3_l, mb3_l);

        sum12 = _mm256_add_epi32(mc0, sum12);
        sum13 = _mm256_add_epi32(mc1, sum13);
        sum14 = _mm256_add_epi32(mc2, sum14);
        sum15 = _mm256_add_epi32(mc3, sum15);
    }

    //store
    __m256i zero = _mm256_setzero_si256();

    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum2 = _mm256_hadd_epi32(sum2, sum3);
    sum0 = _mm256_hadd_epi32(sum0, sum2);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1] = _mm256_extract_epi32(sum0, 1);
    pc0[2] = _mm256_extract_epi32(sum0, 2);
    pc0[3] = _mm256_extract_epi32(sum0, 3);

    //the 1 row
    sum4 = _mm256_hadd_epi32(sum4, sum5);
    sum6 = _mm256_hadd_epi32(sum6, sum7);
    sum4 = _mm256_hadd_epi32(sum4, sum6);
    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum4, 0);
    pc1[1] = _mm256_extract_epi32(sum4, 1);
    pc1[2] = _mm256_extract_epi32(sum4, 2);
    pc1[3] = _mm256_extract_epi32(sum4, 3);

    //the 2 row
    sum8 = _mm256_hadd_epi32(sum8, sum9);
    sum10 = _mm256_hadd_epi32(sum10, sum11);
    sum8 = _mm256_hadd_epi32(sum8, sum10);
    sum8 = _mm256_add_epi32(sum8, _mm256_permute2x128_si256(sum8, zero, 0x31));

    pc2[0] = _mm256_extract_epi32(sum8, 0);
    pc2[1] = _mm256_extract_epi32(sum8, 1);
    pc2[2] = _mm256_extract_epi32(sum8, 2);
    pc2[3] = _mm256_extract_epi32(sum8, 3);

    //the 3 row
    sum12 = _mm256_hadd_epi32(sum12, sum13);
    sum14 = _mm256_hadd_epi32(sum14, sum15);
    sum12 = _mm256_hadd_epi32(sum12, sum14);
    sum12 = _mm256_add_epi32(sum12, _mm256_permute2x128_si256(sum12, zero, 0x31));
    pc3[0] = _mm256_extract_epi32(sum12, 0);
    pc3[1] = _mm256_extract_epi32(sum12, 1);
    pc3[2] = _mm256_extract_epi32(sum12, 2);
    pc3[3] = _mm256_extract_epi32(sum12, 3);
}

inline void block4x2_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int stride) {
    //printf("block4x2_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma1_l;
    __m256i ma2_l;
    __m256i ma3_l;
    __m256i ma0_h;
    __m256i ma1_h;
    __m256i ma2_h;
    __m256i ma3_h;

    __m256i mb0_l;
    __m256i mb1_l;
    __m256i mb0_h;
    __m256i mb1_h;

    __m256i mc0;
    __m256i mc1;
    __m256i mc2;
    __m256i mc3;
    __m256i mc4;
    __m256i mc5;
    __m256i mc6;
    __m256i mc7;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));

        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mb1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb1 + 16)));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma0_h, mb1_h));

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        ma1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa1 + 16)));

        mc2 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb1_l);

        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma1_h, mb0_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma1_h, mb1_h));

        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));
        ma2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa2 + 16)));

        mc4 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma2_l, mb1_l);

        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma2_h, mb0_h));
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma2_h, mb1_h));

        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));
        ma3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa3 + 16)));

        mc6 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc7 = _mm256_madd_epi16(ma3_l, mb1_l);

        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma3_h, mb0_h));
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma3_h, mb1_h));

        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 32;
        pa1 += 32;
        pa2 += 32;
        pa3 += 32;

        pb0 += 32;
        pb1 += 32;
    }

    //leftover
    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));

        mc2 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb1_l);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));

        mc4 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma2_l, mb1_l);
        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));

        mc6 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc7 = _mm256_madd_epi16(ma3_l, mb1_l);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;

        pb0 += 16;
        pb1 += 16;
    }

    if (0x08 & k_leftover) {
        //a
        __m256i ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));

        //b
        __m256i mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));
        __m256i mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb1));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa1));

        mc2 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb1_l);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa2));

        mc4 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc5 = _mm256_mullo_epi32(ma2_l, mb1_l);
        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa3));

        mc6 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc7 = _mm256_mullo_epi32(ma3_l, mb1_l);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 8;
        pa1 += 8;
        pa2 += 8;
        pa3 += 8;

        pb0 += 8;
        pb1 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];
            ga1[i] = pa1[i];
            ga2[i] = pa2[i];
            ga3[i] = pa3[i];

            gb0[i] = pb0[i];
            gb1[i] = pb1[i];
        }

        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb1));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga1));

        mc2 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb1_l);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga2));

        mc4 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc5 = _mm256_mullo_epi32(ma2_l, mb1_l);
        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga3));

        mc6 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc7 = _mm256_mullo_epi32(ma3_l, mb1_l);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);
    }

    //store
    __m256i zero = _mm256_setzero_si256();

    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum0 = _mm256_hadd_epi32(sum0, zero);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1 * stride] = _mm256_extract_epi32(sum0, 1);

    //the 1 row
    sum2 = _mm256_hadd_epi32(sum2, sum3);
    sum2 = _mm256_hadd_epi32(sum2, zero);
    sum2 = _mm256_add_epi32(sum2, _mm256_permute2x128_si256(sum2, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum2, 0);
    pc1[1 * stride] = _mm256_extract_epi32(sum2, 1);

    //the 2 row
    sum4 = _mm256_hadd_epi32(sum4, sum5);
    sum4 = _mm256_hadd_epi32(sum4, zero);
    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    pc2[0] = _mm256_extract_epi32(sum4, 0);
    pc2[1 * stride] = _mm256_extract_epi32(sum4, 1);

    //the 3 row
    sum6 = _mm256_hadd_epi32(sum6, sum7);
    sum6 = _mm256_hadd_epi32(sum6, zero);
    sum6 = _mm256_add_epi32(sum6, _mm256_permute2x128_si256(sum6, zero, 0x31));

    pc3[0] = _mm256_extract_epi32(sum6, 0);
    pc3[1 * stride] = _mm256_extract_epi32(sum6, 1);
}

inline void block4x1_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int stride) {
    //printf("block4x1_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;

    const int8_t* pb0 = b;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma1_l;
    __m256i ma2_l;
    __m256i ma3_l;
    __m256i ma0_h;
    __m256i ma1_h;
    __m256i ma2_h;
    __m256i ma3_h;

    __m256i mb0_l;
    __m256i mb0_h;

    __m256i mc0;
    __m256i mc1;
    __m256i mc2;
    __m256i mc3;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        ma1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa1 + 16)));

        mc1 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma1_h, mb0_h));
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));
        ma2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa2 + 16)));

        mc2 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma2_h, mb0_h));
        sum2 = _mm256_add_epi32(mc2, sum2);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));
        ma3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa3 + 16)));

        mc3 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma3_h, mb0_h));
        sum3 = _mm256_add_epi32(mc3, sum3);

        pa0 += 32;
        pa1 += 32;
        pa2 += 32;
        pa3 += 32;

        pb0 += 32;
    }

    //leftover
    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        mc1 = _mm256_madd_epi16(ma1_l, mb0_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));
        mc2 = _mm256_madd_epi16(ma2_l, mb0_l);
        sum2 = _mm256_add_epi32(mc2, sum2);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));
        mc3 = _mm256_madd_epi16(ma3_l, mb0_l);
        sum3 = _mm256_add_epi32(mc3, sum3);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;

        pb0 += 16;
    }

    if (0x08 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa1));
        mc1 = _mm256_mullo_epi32(ma1_l, mb0_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa2));
        mc2 = _mm256_mullo_epi32(ma2_l, mb0_l);
        sum2 = _mm256_add_epi32(mc2, sum2);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa3));
        mc3 = _mm256_mullo_epi32(ma3_l, mb0_l);
        sum3 = _mm256_add_epi32(mc3, sum3);

        pa0 += 8;
        pa1 += 8;
        pa2 += 8;
        pa3 += 8;

        pb0 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];
            ga1[i] = pa1[i];
            ga2[i] = pa2[i];
            ga3[i] = pa3[i];

            gb0[i] = pb0[i];
        }

        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga1));
        mc1 = _mm256_mullo_epi32(ma1_l, mb0_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga2));
        mc2 = _mm256_mullo_epi32(ma2_l, mb0_l);
        sum2 = _mm256_add_epi32(mc2, sum2);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga3));
        mc3 = _mm256_mullo_epi32(ma3_l, mb0_l);
        sum3 = _mm256_add_epi32(mc3, sum3);
    }

    //store
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, sum0, 0x81));
    sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 8));
    sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 4));
    pc0[0] = _mm256_extract_epi32(sum0, 0);

    sum1 = _mm256_add_epi32(sum1, _mm256_permute2x128_si256(sum1, sum1, 0x81));
    sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 8));
    sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 4));
    pc1[0] = _mm256_extract_epi32(sum1, 0);

    sum2 = _mm256_add_epi32(sum2, _mm256_permute2x128_si256(sum2, sum2, 0x81));
    sum2 = _mm256_add_epi32(sum2, _mm256_srli_si256(sum2, 8));
    sum2 = _mm256_add_epi32(sum2, _mm256_srli_si256(sum2, 4));
    pc2[0] = _mm256_extract_epi32(sum2, 0);

    sum3 = _mm256_add_epi32(sum3, _mm256_permute2x128_si256(sum3, sum3, 0x81));
    sum3 = _mm256_add_epi32(sum3, _mm256_srli_si256(sum3, 8));
    sum3 = _mm256_add_epi32(sum3, _mm256_srli_si256(sum3, 4));
    pc3[0] = _mm256_extract_epi32(sum3, 0);
}

inline void block2x8_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int stride) {
    //printf("block2x8_kernel_avx2\n");
    block8x2_kernel_avx2(k, b, ldb, a, lda, c, stride, ldc);
}

inline void block2x4_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int stride) {
    //printf("block2x4_kernel_avx2\n");
    block4x2_kernel_avx2(k, b, ldb, a, lda, c, stride, ldc);

}

inline void block2x2_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc) {
    //printf("block2x2_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma1_l;
    __m256i ma0_h;
    __m256i ma1_h;

    __m256i mb0_l;
    __m256i mb1_l;
    __m256i mb0_h;
    __m256i mb1_h;

    __m256i mc0;
    __m256i mc1;
    __m256i mc2;
    __m256i mc3;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));

        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mb1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb1 + 16)));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        sum0 = _mm256_add_epi32(mc0, sum0);

        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma0_h, mb1_h));
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        ma1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa1 + 16)));

        mc2 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma1_h, mb0_h));
        sum2 = _mm256_add_epi32(mc2, sum2);

        mc3 = _mm256_madd_epi16(ma1_l, mb1_l);
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma1_h, mb1_h));
        sum3 = _mm256_add_epi32(mc3, sum3);

        pa0 += 32;
        pa1 += 32;
        pb0 += 32;
        pb1 += 32;
    }

    //leftover
    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));

        mc2 = _mm256_madd_epi16(ma1_l, mb0_l);
        sum2 = _mm256_add_epi32(mc2, sum2);

        mc3 = _mm256_madd_epi16(ma1_l, mb1_l);
        sum3 = _mm256_add_epi32(mc3, sum3);

        pa0 += 16;
        pa1 += 16;

        pb0 += 16;
        pb1 += 16;
    }

    if (0x08 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb1));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa1));

        mc2 = _mm256_mullo_epi32(ma1_l, mb0_l);
        sum2 = _mm256_add_epi32(mc2, sum2);

        mc3 = _mm256_mullo_epi32(ma1_l, mb1_l);
        sum3 = _mm256_add_epi32(mc3, sum3);

        pa0 += 8;
        pb0 += 8;
        pa1 += 8;
        pb1 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];
            ga1[i] = pa1[i];

            gb0[i] = pb0[i];
            gb1[i] = pb1[i];
        }

        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb1));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga1));

        mc2 = _mm256_mullo_epi32(ma1_l, mb0_l);
        sum2 = _mm256_add_epi32(mc2, sum2);

        mc3 = _mm256_mullo_epi32(ma1_l, mb1_l);
        sum3 = _mm256_add_epi32(mc3, sum3);
    }

    //store
    __m256i zero = _mm256_setzero_si256();

    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum0 = _mm256_hadd_epi32(sum0, zero);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1] = _mm256_extract_epi32(sum0, 1);

    //the 1 row
    sum2 = _mm256_hadd_epi32(sum2, sum3);
    sum2 = _mm256_hadd_epi32(sum2, zero);
    sum2 = _mm256_add_epi32(sum2, _mm256_permute2x128_si256(sum2, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum2, 0);
    pc1[1] = _mm256_extract_epi32(sum2, 1);
}

inline void block2x1_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int stride) {
    //printf("block2x1_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;

    const int8_t* pb0 = b;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma1_l;
    __m256i ma0_h;
    __m256i ma1_h;

    __m256i mb0_l;
    __m256i mb0_h;

    __m256i mc0;
    __m256i mc1;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        ma1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa1 + 16)));

        mc1 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma1_h, mb0_h));
        sum1 = _mm256_add_epi32(mc1, sum1);

        pa0 += 32;
        pa1 += 32;

        pb0 += 32;
    }

    //leftover
    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        mc1 = _mm256_madd_epi16(ma1_l, mb0_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        pa0 += 16;
        pa1 += 16;

        pb0 += 16;
    }

    if (0x08 & k_leftover) {
        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));

        //the 0 row
        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa1));
        mc1 = _mm256_mullo_epi32(ma1_l, mb0_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        pa0 += 8;
        pa1 += 8;

        pb0 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];
            ga1[i] = pa1[i];

            gb0[i] = pb0[i];
        }

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));

        //the 0 row
        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga1));
        mc1 = _mm256_mullo_epi32(ma1_l, mb0_l);
        sum1 = _mm256_add_epi32(mc1, sum1);
    }

    //store
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, sum0, 0x81));
    sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 8));
    sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 4));
    pc0[0] = _mm256_extract_epi32(sum0, 0);

    sum1 = _mm256_add_epi32(sum1, _mm256_permute2x128_si256(sum1, sum1, 0x81));
    sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 8));
    sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 4));
    pc1[0] = _mm256_extract_epi32(sum1, 0);
}

inline void block1x16_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c) {
    //printf("block1x16_kernel_avx2\n");
    const int8_t* pa0 = a;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;
    const int8_t* pb2 = pb0 + 2 * ldb;
    const int8_t* pb3 = pb0 + 3 * ldb;
    const int8_t* pb4 = pb0 + 4 * ldb;
    const int8_t* pb5 = pb0 + 5 * ldb;
    const int8_t* pb6 = pb0 + 6 * ldb;
    const int8_t* pb7 = pb0 + 7 * ldb;
    const int8_t* pb8 = pb0 + 8 * ldb;
    const int8_t* pb9 = pb0 + 9 * ldb;
    const int8_t* pb10 = pb0 + 10 * ldb;
    const int8_t* pb11 = pb0 + 11 * ldb;
    const int8_t* pb12 = pb0 + 12 * ldb;
    const int8_t* pb13 = pb0 + 13 * ldb;
    const int8_t* pb14 = pb0 + 14 * ldb;
    const int8_t* pb15 = pb0 + 15 * ldb;

    int* pc0 = c;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma0_h;

    __m256i mb0_l;
    __m256i mb1_l;
    __m256i mb2_l;
    __m256i mb3_l;
    __m256i mb4_l;
    __m256i mb5_l;
    __m256i mb6_l;
    __m256i mb7_l;
    __m256i mb0_h;
    __m256i mb1_h;
    __m256i mb2_h;
    __m256i mb3_h;
    __m256i mb4_h;
    __m256i mb5_h;
    __m256i mb6_h;
    __m256i mb7_h;
    __m256i mb8_l;
    __m256i mb9_l;
    __m256i mb10_l;
    __m256i mb11_l;
    __m256i mb12_l;
    __m256i mb13_l;
    __m256i mb14_l;
    __m256i mb15_l;
    __m256i mb8_h;
    __m256i mb9_h;
    __m256i mb10_h;
    __m256i mb11_h;
    __m256i mb12_h;
    __m256i mb13_h;
    __m256i mb14_h;
    __m256i mb15_h;

    __m256i mc0;
    __m256i mc1;
    __m256i mc2;
    __m256i mc3;
    __m256i mc4;
    __m256i mc5;
    __m256i mc6;
    __m256i mc7;
    __m256i mc8;
    __m256i mc9;
    __m256i mc10;
    __m256i mc11;
    __m256i mc12;
    __m256i mc13;
    __m256i mc14;
    __m256i mc15;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    __m256i sum8 = _mm256_setzero_si256();
    __m256i sum9 = _mm256_setzero_si256();
    __m256i sum10 = _mm256_setzero_si256();
    __m256i sum11 = _mm256_setzero_si256();
    __m256i sum12 = _mm256_setzero_si256();
    __m256i sum13 = _mm256_setzero_si256();
    __m256i sum14 = _mm256_setzero_si256();
    __m256i sum15 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //the 0 col
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 col
        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mb1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb1 + 16)));
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma0_h, mb1_h));
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 2 col
        mb2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        mb2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb2 + 16)));
        mc2 = _mm256_madd_epi16(ma0_l, mb2_l);
        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma0_h, mb2_h));
        sum2 = _mm256_add_epi32(mc2, sum2);

        //the 3 col
        mb3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));
        mb3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb3 + 16)));
        mc3 = _mm256_madd_epi16(ma0_l, mb3_l);
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma0_h, mb3_h));
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 4 col
        mb4_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb4));
        mb4_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb4 + 16)));
        mc4 = _mm256_madd_epi16(ma0_l, mb4_l);
        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma0_h, mb4_h));
        sum4 = _mm256_add_epi32(mc4, sum4);

        //the 5 col
        mb5_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb5));
        mb5_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb5 + 16)));
        mc5 = _mm256_madd_epi16(ma0_l, mb5_l);
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma0_h, mb5_h));
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 6 col
        mb6_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb6));
        mb6_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb6 + 16)));
        mc6 = _mm256_madd_epi16(ma0_l, mb6_l);
        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma0_h, mb6_h));
        sum6 = _mm256_add_epi32(mc6, sum6);

        //the 7 col
        mb7_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb7));
        mb7_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb7 + 16)));
        mc7 = _mm256_madd_epi16(ma0_l, mb7_l);
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma0_h, mb7_h));
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 8 col
        mb8_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb8));
        mb8_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb8 + 16)));
        mc8 = _mm256_madd_epi16(ma0_l, mb8_l);
        mc8 = _mm256_add_epi32(mc8, _mm256_madd_epi16(ma0_h, mb8_h));
        sum8 = _mm256_add_epi32(mc8, sum8);

        //the 9 col
        mb9_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb9));
        mb9_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb9 + 16)));
        mc9 = _mm256_madd_epi16(ma0_l, mb9_l);
        mc9 = _mm256_add_epi32(mc9, _mm256_madd_epi16(ma0_h, mb9_h));
        sum9 = _mm256_add_epi32(mc9, sum9);

        //the 10 col
        mb10_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb10));
        mb10_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb10 + 16)));
        mc10 = _mm256_madd_epi16(ma0_l, mb10_l);
        mc10 = _mm256_add_epi32(mc10, _mm256_madd_epi16(ma0_h, mb10_h));
        sum10 = _mm256_add_epi32(mc10, sum10);

        //the 11 col
        mb11_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb11));
        mb11_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb11 + 16)));
        mc11 = _mm256_madd_epi16(ma0_l, mb11_l);
        mc11 = _mm256_add_epi32(mc11, _mm256_madd_epi16(ma0_h, mb11_h));
        sum11 = _mm256_add_epi32(mc11, sum11);

        //the 12 col
        mb12_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb12));
        mb12_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb12 + 16)));
        mc12 = _mm256_madd_epi16(ma0_l, mb12_l);
        mc12 = _mm256_add_epi32(mc12, _mm256_madd_epi16(ma0_h, mb12_h));
        sum12 = _mm256_add_epi32(mc12, sum12);

        //the 13 col
        mb13_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb13));
        mb13_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb13 + 16)));
        mc13 = _mm256_madd_epi16(ma0_l, mb13_l);
        mc13 = _mm256_add_epi32(mc13, _mm256_madd_epi16(ma0_h, mb13_h));
        sum13 = _mm256_add_epi32(mc13, sum13);

        //the 14 col
        mb14_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb14));
        mb14_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb14 + 16)));
        mc14 = _mm256_madd_epi16(ma0_l, mb14_l);
        mc14 = _mm256_add_epi32(mc14, _mm256_madd_epi16(ma0_h, mb14_h));
        sum14 = _mm256_add_epi32(mc14, sum14);

        //the 15 col
        mb15_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb15));
        mb15_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb15 + 16)));
        mc15 = _mm256_madd_epi16(ma0_l, mb15_l);
        mc15 = _mm256_add_epi32(mc15, _mm256_madd_epi16(ma0_h, mb15_h));
        sum15 = _mm256_add_epi32(mc15, sum15);

        pa0 += 32;

        pb0 += 32;
        pb1 += 32;
        pb2 += 32;
        pb3 += 32;
        pb4 += 32;
        pb5 += 32;
        pb6 += 32;
        pb7 += 32;

        pb8 += 32;
        pb9 += 32;
        pb10 += 32;
        pb11 += 32;
        pb12 += 32;
        pb13 += 32;
        pb14 += 32;
        pb15 += 32;
    }

    //leftover
    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //the 0 col
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 col
        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 2 col
        mb2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb2));
        mc2 = _mm256_madd_epi16(ma0_l, mb2_l);
        sum2 = _mm256_add_epi32(mc2, sum2);

        //the 3 col
        mb3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb3));
        mc3 = _mm256_madd_epi16(ma0_l, mb3_l);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 4 col
        mb4_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb4));
        mc4 = _mm256_madd_epi16(ma0_l, mb4_l);
        sum4 = _mm256_add_epi32(mc4, sum4);

        //the 5 col
        mb5_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb5));
        mc5 = _mm256_madd_epi16(ma0_l, mb5_l);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 6 col
        mb6_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb6));
        mc6 = _mm256_madd_epi16(ma0_l, mb6_l);
        sum6 = _mm256_add_epi32(mc6, sum6);

        //the 7 col
        mb7_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb7));
        mc7 = _mm256_madd_epi16(ma0_l, mb7_l);
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 8 col
        mb8_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb8));
        mc8 = _mm256_madd_epi16(ma0_l, mb8_l);
        sum8 = _mm256_add_epi32(mc8, sum8);

        //the 9 col
        mb9_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb9));
        mc9 = _mm256_madd_epi16(ma0_l, mb9_l);
        sum9 = _mm256_add_epi32(mc9, sum9);

        //the 10 col
        mb10_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb10));
        mc10 = _mm256_madd_epi16(ma0_l, mb10_l);
        sum10 = _mm256_add_epi32(mc10, sum10);

        //the 11 col
        mb11_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb11));
        mc11 = _mm256_madd_epi16(ma0_l, mb11_l);
        sum11 = _mm256_add_epi32(mc11, sum11);

        //the 12 col
        mb12_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb12));
        mc12 = _mm256_madd_epi16(ma0_l, mb12_l);
        sum12 = _mm256_add_epi32(mc12, sum12);

        //the 13 col
        mb13_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb13));
        mc13 = _mm256_madd_epi16(ma0_l, mb13_l);
        sum13 = _mm256_add_epi32(mc13, sum13);

        //the 14 col
        mb14_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb14));
        mc14 = _mm256_madd_epi16(ma0_l, mb14_l);
        sum14 = _mm256_add_epi32(mc14, sum14);

        //the 15 col
        mb15_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb15));
        mc15 = _mm256_madd_epi16(ma0_l, mb15_l);
        sum15 = _mm256_add_epi32(mc15, sum15);

        pa0 += 16;

        pb0 += 16;
        pb1 += 16;
        pb2 += 16;
        pb3 += 16;
        pb4 += 16;
        pb5 += 16;
        pb6 += 16;
        pb7 += 16;

        pb8 += 16;
        pb9 += 16;
        pb10 += 16;
        pb11 += 16;
        pb12 += 16;
        pb13 += 16;
        pb14 += 16;
        pb15 += 16;
    }

    if (0x08 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));

        //the 0 col
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 col
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb1));
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 2 col
        mb2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb2));
        mc2 = _mm256_mullo_epi32(ma0_l, mb2_l);
        sum2 = _mm256_add_epi32(mc2, sum2);

        //the 3 col
        mb3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb3));
        mc3 = _mm256_mullo_epi32(ma0_l, mb3_l);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 4 col
        mb4_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb4));
        mc4 = _mm256_mullo_epi32(ma0_l, mb4_l);
        sum4 = _mm256_add_epi32(mc4, sum4);

        //the 5 col
        mb5_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb5));
        mc5 = _mm256_mullo_epi32(ma0_l, mb5_l);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 6 col
        mb6_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb6));
        mc6 = _mm256_mullo_epi32(ma0_l, mb6_l);
        sum6 = _mm256_add_epi32(mc6, sum6);

        //the 7 col
        mb7_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb7));
        mc7 = _mm256_mullo_epi32(ma0_l, mb7_l);
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 8 col
        mb8_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb8));
        mc8 = _mm256_mullo_epi32(ma0_l, mb8_l);
        sum8 = _mm256_add_epi32(mc8, sum8);

        //the 9 col
        mb9_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb9));
        mc9 = _mm256_mullo_epi32(ma0_l, mb9_l);
        sum9 = _mm256_add_epi32(mc9, sum9);

        //the 10 col
        mb10_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb10));
        mc10 = _mm256_mullo_epi32(ma0_l, mb10_l);
        sum10 = _mm256_add_epi32(mc10, sum10);

        //the 11 col
        mb11_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb11));
        mc11 = _mm256_mullo_epi32(ma0_l, mb11_l);
        sum11 = _mm256_add_epi32(mc11, sum11);

        //the 12 col
        mb12_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb12));
        mc12 = _mm256_mullo_epi32(ma0_l, mb12_l);
        sum12 = _mm256_add_epi32(mc12, sum12);

        //the 13 col
        mb13_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb13));
        mc13 = _mm256_mullo_epi32(ma0_l, mb13_l);
        sum13 = _mm256_add_epi32(mc13, sum13);

        //the 14 col
        mb14_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb14));
        mc14 = _mm256_mullo_epi32(ma0_l, mb14_l);
        sum14 = _mm256_add_epi32(mc14, sum14);

        //the 15 col
        mb15_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb15));
        mc15 = _mm256_mullo_epi32(ma0_l, mb15_l);
        sum15 = _mm256_add_epi32(mc15, sum15);

        pa0 += 8;

        pb0 += 8;
        pb1 += 8;
        pb2 += 8;
        pb3 += 8;
        pb4 += 8;
        pb5 += 8;
        pb6 += 8;
        pb7 += 8;

        pb8 += 8;
        pb9 += 8;
        pb10 += 8;
        pb11 += 8;
        pb12 += 8;
        pb13 += 8;
        pb14 += 8;
        pb15 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb4[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb5[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb6[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb7[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb8[8]  __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb9[8]  __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb10[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb11[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb12[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb13[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb14[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb15[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];

            gb0[i] = pb0[i];
            gb1[i] = pb1[i];
            gb2[i] = pb2[i];
            gb3[i] = pb3[i];
            gb4[i] = pb4[i];
            gb5[i] = pb5[i];
            gb6[i] = pb6[i];
            gb7[i] = pb7[i];

            gb8[i] = pb8[i];
            gb9[i] = pb9[i];
            gb10[i] = pb10[i];
            gb11[i] = pb11[i];
            gb12[i] = pb12[i];
            gb13[i] = pb13[i];
            gb14[i] = pb14[i];
            gb15[i] = pb15[i];
        }

        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));

        //the 0 col
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        //the 1 col
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb1));
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 2 col
        mb2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb2));
        mc2 = _mm256_mullo_epi32(ma0_l, mb2_l);
        sum2 = _mm256_add_epi32(mc2, sum2);

        //the 3 col
        mb3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb3));
        mc3 = _mm256_mullo_epi32(ma0_l, mb3_l);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 4 col
        mb4_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb4));
        mc4 = _mm256_mullo_epi32(ma0_l, mb4_l);
        sum4 = _mm256_add_epi32(mc4, sum4);

        //the 5 col
        mb5_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb5));
        mc5 = _mm256_mullo_epi32(ma0_l, mb5_l);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 6 col
        mb6_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb6));
        mc6 = _mm256_mullo_epi32(ma0_l, mb6_l);
        sum6 = _mm256_add_epi32(mc6, sum6);

        //the 7 col
        mb7_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb7));
        mc7 = _mm256_mullo_epi32(ma0_l, mb7_l);
        sum7 = _mm256_add_epi32(mc7, sum7);

        //the 8 col
        mb8_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb8));
        mc8 = _mm256_mullo_epi32(ma0_l, mb8_l);
        sum8 = _mm256_add_epi32(mc8, sum8);

        //the 9 col
        mb9_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb9));
        mc9 = _mm256_mullo_epi32(ma0_l, mb9_l);
        sum9 = _mm256_add_epi32(mc9, sum9);

        //the 10 col
        mb10_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb10));
        mc10 = _mm256_mullo_epi32(ma0_l, mb10_l);
        sum10 = _mm256_add_epi32(mc10, sum10);

        //the 11 col
        mb11_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb11));
        mc11 = _mm256_mullo_epi32(ma0_l, mb11_l);
        sum11 = _mm256_add_epi32(mc11, sum11);

        //the 12 col
        mb12_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb12));
        mc12 = _mm256_mullo_epi32(ma0_l, mb12_l);
        sum12 = _mm256_add_epi32(mc12, sum12);

        //the 13 col
        mb13_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb13));
        mc13 = _mm256_mullo_epi32(ma0_l, mb13_l);
        sum13 = _mm256_add_epi32(mc13, sum13);

        //the 14 col
        mb14_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb14));
        mc14 = _mm256_mullo_epi32(ma0_l, mb14_l);
        sum14 = _mm256_add_epi32(mc14, sum14);

        //the 15 col
        mb15_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb15));
        mc15 = _mm256_mullo_epi32(ma0_l, mb15_l);
        sum15 = _mm256_add_epi32(mc15, sum15);
    }

    //store
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, sum0, 0x81));
    sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 8));
    sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 4));
    pc0[0] = _mm256_extract_epi32(sum0, 0);

    sum1 = _mm256_add_epi32(sum1, _mm256_permute2x128_si256(sum1, sum1, 0x81));
    sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 8));
    sum1 = _mm256_add_epi32(sum1, _mm256_srli_si256(sum1, 4));
    pc0[1] = _mm256_extract_epi32(sum1, 0);

    sum2 = _mm256_add_epi32(sum2, _mm256_permute2x128_si256(sum2, sum2, 0x81));
    sum2 = _mm256_add_epi32(sum2, _mm256_srli_si256(sum2, 8));
    sum2 = _mm256_add_epi32(sum2, _mm256_srli_si256(sum2, 4));
    pc0[2] = _mm256_extract_epi32(sum2, 0);

    sum3 = _mm256_add_epi32(sum3, _mm256_permute2x128_si256(sum3, sum3, 0x81));
    sum3 = _mm256_add_epi32(sum3, _mm256_srli_si256(sum3, 8));
    sum3 = _mm256_add_epi32(sum3, _mm256_srli_si256(sum3, 4));
    pc0[3] = _mm256_extract_epi32(sum3, 0);

    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, sum4, 0x81));
    sum4 = _mm256_add_epi32(sum4, _mm256_srli_si256(sum4, 8));
    sum4 = _mm256_add_epi32(sum4, _mm256_srli_si256(sum4, 4));
    pc0[4] = _mm256_extract_epi32(sum4, 0);

    sum5 = _mm256_add_epi32(sum5, _mm256_permute2x128_si256(sum5, sum5, 0x81));
    sum5 = _mm256_add_epi32(sum5, _mm256_srli_si256(sum5, 8));
    sum5 = _mm256_add_epi32(sum5, _mm256_srli_si256(sum5, 4));
    pc0[5] = _mm256_extract_epi32(sum5, 0);

    sum6 = _mm256_add_epi32(sum6, _mm256_permute2x128_si256(sum6, sum6, 0x81));
    sum6 = _mm256_add_epi32(sum6, _mm256_srli_si256(sum6, 8));
    sum6 = _mm256_add_epi32(sum6, _mm256_srli_si256(sum6, 4));
    pc0[6] = _mm256_extract_epi32(sum6, 0);

    sum7 = _mm256_add_epi32(sum7, _mm256_permute2x128_si256(sum7, sum7, 0x81));
    sum7 = _mm256_add_epi32(sum7, _mm256_srli_si256(sum7, 8));
    sum7 = _mm256_add_epi32(sum7, _mm256_srli_si256(sum7, 4));
    pc0[7] = _mm256_extract_epi32(sum7, 0);

    sum8 = _mm256_add_epi32(sum8, _mm256_permute2x128_si256(sum8, sum8, 0x81));
    sum8 = _mm256_add_epi32(sum8, _mm256_srli_si256(sum8, 8));
    sum8 = _mm256_add_epi32(sum8, _mm256_srli_si256(sum8, 4));
    pc0[8] = _mm256_extract_epi32(sum8, 0);

    sum9 = _mm256_add_epi32(sum9, _mm256_permute2x128_si256(sum9, sum9, 0x81));
    sum9 = _mm256_add_epi32(sum9, _mm256_srli_si256(sum9, 8));
    sum9 = _mm256_add_epi32(sum9, _mm256_srli_si256(sum9, 4));
    pc0[9] = _mm256_extract_epi32(sum9, 0);

    sum10 = _mm256_add_epi32(sum10, _mm256_permute2x128_si256(sum10, sum10, 0x81));
    sum10 = _mm256_add_epi32(sum10, _mm256_srli_si256(sum10, 8));
    sum10 = _mm256_add_epi32(sum10, _mm256_srli_si256(sum10, 4));
    pc0[10] = _mm256_extract_epi32(sum10, 0);

    sum11 = _mm256_add_epi32(sum11, _mm256_permute2x128_si256(sum11, sum11, 0x81));
    sum11 = _mm256_add_epi32(sum11, _mm256_srli_si256(sum11, 8));
    sum11 = _mm256_add_epi32(sum11, _mm256_srli_si256(sum11, 4));
    pc0[11] = _mm256_extract_epi32(sum11, 0);

    sum12 = _mm256_add_epi32(sum12, _mm256_permute2x128_si256(sum12, sum12, 0x81));
    sum12 = _mm256_add_epi32(sum12, _mm256_srli_si256(sum12, 8));
    sum12 = _mm256_add_epi32(sum12, _mm256_srli_si256(sum12, 4));
    pc0[12] = _mm256_extract_epi32(sum12, 0);

    sum13 = _mm256_add_epi32(sum13, _mm256_permute2x128_si256(sum13, sum13, 0x81));
    sum13 = _mm256_add_epi32(sum13, _mm256_srli_si256(sum13, 8));
    sum13 = _mm256_add_epi32(sum13, _mm256_srli_si256(sum13, 4));
    pc0[13] = _mm256_extract_epi32(sum13, 0);

    sum14 = _mm256_add_epi32(sum14, _mm256_permute2x128_si256(sum14, sum14, 0x81));
    sum14 = _mm256_add_epi32(sum14, _mm256_srli_si256(sum14, 8));
    sum14 = _mm256_add_epi32(sum14, _mm256_srli_si256(sum14, 4));
    pc0[14] = _mm256_extract_epi32(sum14, 0);

    sum15 = _mm256_add_epi32(sum15, _mm256_permute2x128_si256(sum15, sum15, 0x81));
    sum15 = _mm256_add_epi32(sum15, _mm256_srli_si256(sum15, 8));
    sum15 = _mm256_add_epi32(sum15, _mm256_srli_si256(sum15, 4));
    pc0[15] = _mm256_extract_epi32(sum15, 0);
}

void block1x8_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int stride) {
    block8x1_kernel_avx2(k, b, ldb, a, lda, c, stride, ldc);
}

void block1x4_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int stride) {
    block4x1_kernel_avx2(k, b, ldb, a, lda, c, stride, ldc);
}

void block1x2_kernel_avx2(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int stride) {
    block2x1_kernel_avx2(k, b, ldb, a, lda, c, stride, ldc);
}

void block1x1_kernel_avx2(const int32_t k, const int8_t* a, const int8_t* b, int* c) {
    //printf("block1x1_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pb0 = b;

    int* pc0 = c;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma0_h;

    __m256i mb0_l;
    __m256i mb0_h;

    __m256i mc0;

    __m256i sum0 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        sum0 = _mm256_add_epi32(mc0, sum0);

        pa0 += 32;
        pb0 += 32;
    }

    //leftover
    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        pa0 += 16;
        pb0 += 16;
    }

    if (0x08 & k_leftover) {
        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));

        //the 0 row
        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);

        pa0 += 8;
        pb0 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];
            gb0[i] = pb0[i];
        }

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));

        //the 0 row
        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        sum0 = _mm256_add_epi32(mc0, sum0);
    }

    //store
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, sum0, 0x81));
    sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 8));
    sum0 = _mm256_add_epi32(sum0, _mm256_srli_si256(sum0, 4));
    pc0[0] = _mm256_extract_epi32(sum0, 0);
}

void chgemm_c_c_n_t_avx2(
    const int32_t m, const int32_t n, const int32_t k,
    const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb,
    int32_t* c, const int32_t ldc) {
    size_t m_block_size = 8;
    size_t mb = m / m_block_size;
    size_t m_leftover = m % m_block_size;

    //    LOG(INFO)<<"chgemm_c_c_n_t_avx2";
    //m>=8
    for (size_t i = 0; i < mb; ++i) {
        size_t n_block_size = 8;
        size_t nb = n / n_block_size;
        size_t n_leftover = n % n_block_size;

        //n=8
        for (size_t j = 0; j < nb; ++j) {
            block8x8_kernel_avx2(k, a + (i * m_block_size) * lda, lda,
                                 b + (j * n_block_size) * ldb, ldb,
                                 c + (i * m_block_size) * ldc + j * n_block_size, ldc);
        }

        //n=4
        if (n_leftover & 0x04) {
            block8x4_kernel_avx2(k, a + (i * m_block_size) * lda, lda,
                                 b + (nb * n_block_size) * ldb, ldb,
                                 c + (i * m_block_size) * ldc + nb * n_block_size, ldc, 1);
        }

        //n=2
        if (n_leftover & 0x02) {
            size_t n4 = n_leftover & 0x04 ? 4 : 0;
            block8x2_kernel_avx2(k, a + (i * m_block_size) * lda, lda,
                                 b + (nb * n_block_size + n4) * ldb, ldb,
                                 c + (i * m_block_size) * ldc + nb * n_block_size + n4, ldc, 1);
        }

        //n=1
        if (n_leftover & 0x01) {
            size_t n4 = n_leftover & 0x04 ? 4 : 0;
            size_t n2 = n_leftover & 0x02 ? 2 : 0;
            block8x1_kernel_avx2(k, a + (i * m_block_size) * lda, lda,
                                 b + (nb * n_block_size + n2 + n4) * ldb, ldb,
                                 c + (i * m_block_size) * ldc + nb * n_block_size + n2 + n4, ldc, 1);
        }
    }

    //m==4
    if (m_leftover & 0x04) {
        size_t n_block_size = 8;
        size_t nb = n / n_block_size;
        size_t n_leftover = n % n_block_size;

        //n=8
        for (size_t j = 0; j < nb; ++j) {
            block4x8_kernel_avx2(k, a + (mb * m_block_size) * lda, lda,
                                 b + (j * n_block_size) * ldb, ldb,
                                 c + (mb * m_block_size) * ldc + j * n_block_size, ldc, 1);
        }

        //n=4
        if (n_leftover & 0x04) {
            block4x4_kernel_avx2(k, a + (mb * m_block_size) * lda, lda,
                                 b + (nb * n_block_size) * ldb, ldb,
                                 c + (mb * m_block_size) * ldc + nb * n_block_size, ldc);
        }

        //n=2
        if (n_leftover & 0x02) {
            size_t n4 = n_leftover & 0x04 ? 4 : 0;
            block4x2_kernel_avx2(k, a + (mb * m_block_size) * lda, lda,
                                 b + (nb * n_block_size + n4) * ldb, ldb,
                                 c + (mb * m_block_size) * ldc + nb * n_block_size + n4, ldc, 1);
        }

        //n=1
        if (n_leftover & 0x01) {
            size_t n4 = n_leftover & 0x04 ? 4 : 0;
            size_t n2 = n_leftover & 0x02 ? 2 : 0;
            block4x1_kernel_avx2(k, a + (mb * m_block_size) * lda, lda,
                                 b + (nb * n_block_size + n4 + n2) * ldb, ldb,
                                 c + (mb * m_block_size) * ldc + nb * n_block_size + n4 + n2, ldc, 1);
        }
    }

    //m==2
    if (m_leftover & 0x02) {
        LOG(INFO) << "hello m_leftover";
        size_t n_block_size = 8;
        size_t nb = n / n_block_size;
        size_t n_leftover = n % n_block_size;

        size_t m4 = m_leftover & 0x04 ? 4 : 0;

        //n=8
        for (size_t j = 0; j < nb; ++j) {
            block2x8_kernel_avx2(k, a + (mb * m_block_size + m4) * lda, lda,
                                 b + (j * n_block_size) * ldb, ldb,
                                 c + (mb * m_block_size + m4) * ldc + j * n_block_size, ldc, 1);
        }

        //n=4
        if (n_leftover & 0x04) {
            block2x4_kernel_avx2(k, a + (mb * m_block_size + m4) * lda, lda,
                                 b + (nb * n_block_size) * ldb, ldb,
                                 c + (mb * m_block_size + m4) * ldc +
                                 nb * n_block_size, ldc, 1);
        }

        //n=2
        if (n_leftover & 0x02) {
            size_t n4 = n_leftover & 0x04 ? 4 : 0;
            block2x2_kernel_avx2(k, a + (mb * m_block_size + m4) * lda, lda,
                                 b + (nb * n_block_size + n4) * ldb, ldb,
                                 c + (mb * m_block_size + m4) * ldc +
                                 nb * n_block_size + n4, ldc);
            LOG(INFO) << "hello";
        }

        //n=1
        if (n_leftover & 0x01) {
            size_t n4 = n_leftover & 0x04 ? 4 : 0;
            size_t n2 = n_leftover & 0x02 ? 2 : 0;
            block2x1_kernel_avx2(k, a + (mb * m_block_size + m4) * lda, lda,
                                 b + (nb * n_block_size + n4 + n2) * ldb, ldb,
                                 c + (mb * m_block_size + m4) * ldc +
                                 nb * n_block_size + n4 + n2, ldc, 1);
        }
    }

    //m==1
    if (m_leftover & 0x01) {
        size_t n_block_size = 16;
        size_t nb = n / n_block_size;
        size_t n_leftover = n % n_block_size;

        size_t m4 = m_leftover & 0x04 ? 4 : 0;
        size_t m2 = m_leftover & 0x02 ? 2 : 0;

        //n=16
        for (size_t j = 0; j < nb; ++j) {
            block1x16_kernel_avx2(k, a + (mb * m_block_size + m4 + m2) * lda, lda,
                                  b + (j * n_block_size) * ldb, ldb,
                                  c + (mb * m_block_size + m4 + m2) * ldc + j * n_block_size);
        }

        //n=8
        if (n_leftover & 0x08) {
            block1x8_kernel_avx2(k, a + (mb * m_block_size + m4 + m2) * lda, lda,
                                 b + (nb * n_block_size) * ldb, ldb,
                                 c + (mb * m_block_size + m4 + m2) * ldc + nb * n_block_size, ldc, 1);
        }

        //n=4
        if (n_leftover & 0x04) {
            size_t n8 = n_leftover & 0x08 ? 8 : 0;
            block1x4_kernel_avx2(k, a + (mb * m_block_size + m4 + m2) * lda, lda,
                                 b + (nb * n_block_size + n8) * ldb, ldb,
                                 c + (mb * m_block_size + m4 + m2) * ldc + nb * n_block_size + n8, ldc, 1);
        }

        //n=2
        if (n_leftover & 0x02) {
            size_t n8 = n_leftover & 0x08 ? 8 : 0;
            size_t n4 = n_leftover & 0x04 ? 4 : 0;
            block1x2_kernel_avx2(k, a + (mb * m_block_size + m4 + m2) * lda, lda,
                                 b + (nb * n_block_size + n8 + n4) * ldb, ldb,
                                 c + (mb * m_block_size + m4 + m2) * ldc + nb * n_block_size + n8 + n4, ldc, 1);
        }

        //n=1
        if (n_leftover & 0x01) {
            size_t n8 = n_leftover & 0x08 ? 8 : 0;
            size_t n4 = n_leftover & 0x04 ? 4 : 0;
            size_t n2 = n_leftover & 0x02 ? 2 : 0;
            block1x1_kernel_avx2(k, a + (mb * m_block_size + m4 + m2) * lda,
                                 b + (nb * n_block_size + n8 + n4 + n2) * ldb,
                                 c + (mb * m_block_size + m4 + m2) * ldc + nb * n_block_size + n8 + n4 + n2);
        }
    }
}

template <>
SaberStatus IntrinsicGemm< char,  char, int >::init(
    const bool trans_a, const bool trans_b,
    const int m, const int n, const int k,
    Context<X86> ctx) {
    CHECK_EQ(trans_a, false) << "only support no trans";
    CHECK_EQ(trans_b, false) << "only support no trans";
    _lda = (!trans_a) ? k : m;
    _ldb = (!trans_b) ? k : n;
    _ldc = n;
    _m = m;
    _n = n;
    _k = k;
    _trans_a = trans_a ? 'T' : 'N';
    _trans_b = trans_b ? 'T' : 'N';
    return SaberSuccess;
}

inline void block4x2_kernel_avx2_me(
    const int32_t k, const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb, int* c, const int32_t ldc, const int stride) {
    //printf("block4x2_kernel_avx2\n");
    const int8_t* pa0 = a;
    const int8_t* pa1 = pa0 + 1 * lda;
    const int8_t* pa2 = pa0 + 2 * lda;
    const int8_t* pa3 = pa0 + 3 * lda;

    const int8_t* pb0 = b;
    const int8_t* pb1 = pb0 + 1 * ldb;

    int* pc0 = c;
    int* pc1 = c + 1 * ldc;
    int* pc2 = c + 2 * ldc;
    int* pc3 = c + 3 * ldc;

    size_t nk = k >> 5; // k / 32
    size_t k_leftover = k - (nk << 5); // k % 32

    __m256i ma0_l;
    __m256i ma1_l;
    __m256i ma2_l;
    __m256i ma3_l;
    __m256i ma0_h;
    __m256i ma1_h;
    __m256i ma2_h;
    __m256i ma3_h;

    __m256i mb0_l;
    __m256i mb1_l;
    __m256i mb0_h;
    __m256i mb1_h;

    __m256i mc0;
    __m256i mc1;
    __m256i mc2;
    __m256i mc3;
    __m256i mc4;
    __m256i mc5;
    __m256i mc6;
    __m256i mc7;

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();
    __m256i sum2 = _mm256_setzero_si256();
    __m256i sum3 = _mm256_setzero_si256();
    __m256i sum4 = _mm256_setzero_si256();
    __m256i sum5 = _mm256_setzero_si256();
    __m256i sum6 = _mm256_setzero_si256();
    __m256i sum7 = _mm256_setzero_si256();

    for (size_t k = 0; k < nk; ++k) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));
        ma0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa0 + 16)));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb0_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb0 + 16)));

        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));
        mb1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pb1 + 16)));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);

        mc0 = _mm256_add_epi32(mc0, _mm256_madd_epi16(ma0_h, mb0_h));
        mc1 = _mm256_add_epi32(mc1, _mm256_madd_epi16(ma0_h, mb1_h));

        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));
        ma1_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa1 + 16)));

        mc2 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb1_l);

        mc2 = _mm256_add_epi32(mc2, _mm256_madd_epi16(ma1_h, mb0_h));
        mc3 = _mm256_add_epi32(mc3, _mm256_madd_epi16(ma1_h, mb1_h));

        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));
        ma2_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa2 + 16)));

        mc4 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma2_l, mb1_l);

        mc4 = _mm256_add_epi32(mc4, _mm256_madd_epi16(ma2_h, mb0_h));
        mc5 = _mm256_add_epi32(mc5, _mm256_madd_epi16(ma2_h, mb1_h));

        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));
        ma3_h = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*)(pa3 + 16)));

        mc6 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc7 = _mm256_madd_epi16(ma3_l, mb1_l);

        mc6 = _mm256_add_epi32(mc6, _mm256_madd_epi16(ma3_h, mb0_h));
        mc7 = _mm256_add_epi32(mc7, _mm256_madd_epi16(ma3_h, mb1_h));

        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 32;
        pa1 += 32;
        pa2 += 32;
        pa3 += 32;

        pb0 += 32;
        pb1 += 32;
    }

    //leftover
    if (0x10 & k_leftover) {
        //a
        ma0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa0));

        //b
        mb0_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb0));
        mb1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pb1));

        //the 0 row
        mc0 = _mm256_madd_epi16(ma0_l, mb0_l);
        mc1 = _mm256_madd_epi16(ma0_l, mb1_l);
        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa1));

        mc2 = _mm256_madd_epi16(ma1_l, mb0_l);
        mc3 = _mm256_madd_epi16(ma1_l, mb1_l);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa2));

        mc4 = _mm256_madd_epi16(ma2_l, mb0_l);
        mc5 = _mm256_madd_epi16(ma2_l, mb1_l);
        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i*) pa3));

        mc6 = _mm256_madd_epi16(ma3_l, mb0_l);
        mc7 = _mm256_madd_epi16(ma3_l, mb1_l);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 16;
        pa1 += 16;
        pa2 += 16;
        pa3 += 16;

        pb0 += 16;
        pb1 += 16;
    }

    if (0x08 & k_leftover) {
        //a
        __m256i ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa0));

        //b
        __m256i mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb0));
        __m256i mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pb1));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa1));

        mc2 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb1_l);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa2));

        mc4 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc5 = _mm256_mullo_epi32(ma2_l, mb1_l);
        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) pa3));

        mc6 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc7 = _mm256_mullo_epi32(ma3_l, mb1_l);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);

        pa0 += 8;
        pa1 += 8;
        pa2 += 8;
        pa3 += 8;

        pb0 += 8;
        pb1 += 8;
    }

    size_t leftover = k_leftover & 0x07;

    if (leftover) {
        int8_t ga0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga2[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t ga3[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        int8_t gb0[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};
        int8_t gb1[8] __attribute__((aligned(16))) = {0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < leftover; ++i) {
            ga0[i] = pa0[i];
            ga1[i] = pa1[i];
            ga2[i] = pa2[i];
            ga3[i] = pa3[i];

            gb0[i] = pb0[i];
            gb1[i] = pb1[i];
        }

        //a
        ma0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga0));

        //b
        mb0_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb0));
        mb1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) gb1));

        //the 0 row
        mc0 = _mm256_mullo_epi32(ma0_l, mb0_l);
        mc1 = _mm256_mullo_epi32(ma0_l, mb1_l);
        sum0 = _mm256_add_epi32(mc0, sum0);
        sum1 = _mm256_add_epi32(mc1, sum1);

        //the 1 row
        ma1_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga1));

        mc2 = _mm256_mullo_epi32(ma1_l, mb0_l);
        mc3 = _mm256_mullo_epi32(ma1_l, mb1_l);
        sum2 = _mm256_add_epi32(mc2, sum2);
        sum3 = _mm256_add_epi32(mc3, sum3);

        //the 2 row
        ma2_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga2));

        mc4 = _mm256_mullo_epi32(ma2_l, mb0_l);
        mc5 = _mm256_mullo_epi32(ma2_l, mb1_l);
        sum4 = _mm256_add_epi32(mc4, sum4);
        sum5 = _mm256_add_epi32(mc5, sum5);

        //the 3 row
        ma3_l = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*) ga3));

        mc6 = _mm256_mullo_epi32(ma3_l, mb0_l);
        mc7 = _mm256_mullo_epi32(ma3_l, mb1_l);
        sum6 = _mm256_add_epi32(mc6, sum6);
        sum7 = _mm256_add_epi32(mc7, sum7);
    }

    //store
    __m256i zero = _mm256_setzero_si256();

    //the 0 row
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum0 = _mm256_hadd_epi32(sum0, zero);
    sum0 = _mm256_add_epi32(sum0, _mm256_permute2x128_si256(sum0, zero, 0x31));

    pc0[0] = _mm256_extract_epi32(sum0, 0);
    pc0[1 * stride] = _mm256_extract_epi32(sum0, 1);

    //the 1 row
    sum2 = _mm256_hadd_epi32(sum2, sum3);
    sum2 = _mm256_hadd_epi32(sum2, zero);
    sum2 = _mm256_add_epi32(sum2, _mm256_permute2x128_si256(sum2, zero, 0x31));

    pc1[0] = _mm256_extract_epi32(sum2, 0);
    pc1[1 * stride] = _mm256_extract_epi32(sum2, 1);

    //the 2 row
    sum4 = _mm256_hadd_epi32(sum4, sum5);
    sum4 = _mm256_hadd_epi32(sum4, zero);
    sum4 = _mm256_add_epi32(sum4, _mm256_permute2x128_si256(sum4, zero, 0x31));

    pc2[0] = _mm256_extract_epi32(sum4, 0);
    pc2[1 * stride] = _mm256_extract_epi32(sum4, 1);

    //the 3 row
    sum6 = _mm256_hadd_epi32(sum6, sum7);
    sum6 = _mm256_hadd_epi32(sum6, zero);
    sum6 = _mm256_add_epi32(sum6, _mm256_permute2x128_si256(sum6, zero, 0x31));

    pc3[0] = _mm256_extract_epi32(sum6, 0);
    pc3[1 * stride] = _mm256_extract_epi32(sum6, 1);
}
/**
 * b must packed
 */
inline void avx_s8s8s32_gemm_2x4_packed(
    const int32_t m, const int32_t n, const int32_t k,
    const int8_t* a, const int32_t lda,
    const int8_t* b, const int32_t ldb,
    int32_t* c, const int32_t ldc) {
    //    LOG(INFO)<<"my code";
    const int m_block = 4;
    const int n_block = 2;
    int mb = m / m_block;
    int nb = n / n_block;
    int m_remainder = m % m_block;
    int n_remainder = n % n_block;
    CHECK_EQ(m_remainder, 0) << "only support remainder = 0";
    CHECK_EQ(n_remainder, 0) << "only support remainder = 0";

    for (int mbi = 0; mbi < mb; mbi++) {
        for (int nbi = 0; nbi < nb; nbi++) {
            const int8_t* a_ptr = &a[mbi * m_block * lda];
            const int8_t* b_ptr = &b[nbi * n_block * ldb];
            int32_t* c_ptr = &c[mbi * m_block * n + nbi * n_block];
            block4x2_kernel_avx2_me(k, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, 1);
        }
    }
}
template <>
SaberStatus IntrinsicGemm< char,  char, int>::dispatch(
    const float alpha, const float beta,
    const  char* ptr_a, const  char* ptr_b, int* ptr_c) {
    CHECK(ptr_a != nullptr);
    CHECK(ptr_b != nullptr);
    CHECK(ptr_c != nullptr);
    //    LOG(INFO)<<"chgemm_c_c_n_t_avx2 dispatch";
    //    LOG(INFO)<<_m<<","<<_n<<","<<_k<<","<<","<<_lda<<","<<","<<_ldb<<","<<_ldc;
    chgemm_c_c_n_t_avx2(_m, _n, _k, (int8_t*)ptr_a, _lda, (int8_t*)ptr_b, _ldb, ptr_c, _ldc);
    //    LOG(INFO)<<"chgemm_c_c_n_t_avx2 end";
    //    avx_s8s8s32_gemm_2x4_packed(_m,_n,_k,ptr_a,_lda,ptr_b,_ldb,ptr_c,_ldc);
    //    exit(0);
    return SaberSuccess;
}
#else

template <>
SaberStatus IntrinsicGemm< char,  char, int >::init(
        const bool trans_a, const bool trans_b,
        const int m, const int n, const int k,
        Context<X86> ctx) {
    LOG(FATAL)<<"not impl";
    return SaberSuccess;
}

template <>
SaberStatus IntrinsicGemm< char,  char, int>::dispatch(
        const float alpha, const float beta,
        const  char* ptr_a, const  char* ptr_b, int* ptr_c) {
    LOG(FATAL)<<"not impl";
    return SaberSuccess;
}
#endif
}
}

