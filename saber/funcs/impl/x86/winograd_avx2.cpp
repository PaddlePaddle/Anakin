#include "saber/funcs/impl/x86/winograd_avx2.h"
#include "mkl_cblas.h"
#include "mkl_trans.h"
#include "tensor_op.h"
#include "saber/funcs/impl/x86/saber_avx2_expand.h"
namespace anakin {
namespace saber {

#if defined(__AVX2__) and defined(__FMA__)

/**
 * \brief transpose with arm neon optimization
 * @param data_out
 * @param data_in
 * @param w_in
 * @param h_in
 */
static void transpose(float* data_out, const float* data_in, int w_in, int h_in) {
    for (int j = 0; j < h_in; ++j) {
        for (int i = 0; i < w_in; ++i) {
            data_out[i * h_in + j] = data_in[j * w_in + i];
        }
    }
}

/**
* \brief winograd transform conv3x3 weights, f63
* this is done in op initialization or creation, only do once
* dout = G * g * GT, where G is the transform coeff, g is the input weights
* @param dout
* @param din
* @param ch_out
* @param ch_in
* @param work_space
*/
static void winograd_transform_weights(float* dout, const float* din, int ch_out, \
                                       int ch_in, float* work_space) {
    const float coeff[8][3] = {
        {      1.0f,         0.0f,       0.0f},
        { -2.0f / 9,    -2.0f / 9,  -2.0f / 9},
        { -2.0f / 9,     2.0f / 9,  -2.0f / 9},
        { 1.0f / 90,    1.0f / 45,  2.0f / 45},
        { 1.0f / 90,   -1.0f / 45,  2.0f / 45},
        {32.0f / 45,   16.0f / 45,  8.0f / 45},
        {32.0f / 45,  -16.0f / 45,  8.0f / 45},
        {      0.0f,         0.0f,       1.0f}
    };

    float* ptr_out = (float*)work_space;

    for (int i = 0; i < ch_out; i++) {
        for (int j = 0; j < ch_in; j++) {
            const float* kernel0 = static_cast<const float*>(din) + (i * ch_in + j) * 9;
            float* ptr_channel = ptr_out + (i * ch_in + j) * 64;

            //! transform kernel, transposed
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            //! h
            float tmp[8][3];

            for (int i = 0; i < 8; i++) {
                tmp[i][0] = k0[0] * coeff[i][0] + k0[1] * coeff[i][1] + k0[2] * coeff[i][2];
                tmp[i][1] = k1[0] * coeff[i][0] + k1[1] * coeff[i][1] + k1[2] * coeff[i][2];
                tmp[i][2] = k2[0] * coeff[i][0] + k2[1] * coeff[i][1] + k2[2] * coeff[i][2];
            }

            //! v
            for (int j = 0; j < 8; j++) {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 8; i++) {
                    ptr_channel[j * 8 + i] = tmpp[0] * coeff[i][0] + tmpp[1] * coeff[i][1] + \
                                             tmpp[2] * coeff[i][2];
                }
            }
        }
    }

    transpose(static_cast<float*>(dout), ptr_out, 64, ch_out * ch_in);
}


inline void transpose8_ps(__m256& row0, __m256& row1, __m256& row2, __m256& row3, __m256& row4,
                          __m256& row5, __m256& row6, __m256& row7) {
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(row0, row1);
    __t1 = _mm256_unpackhi_ps(row0, row1);
    __t2 = _mm256_unpacklo_ps(row2, row3);
    __t3 = _mm256_unpackhi_ps(row2, row3);
    __t4 = _mm256_unpacklo_ps(row4, row5);
    __t5 = _mm256_unpackhi_ps(row4, row5);
    __t6 = _mm256_unpacklo_ps(row6, row7);
    __t7 = _mm256_unpackhi_ps(row6, row7);
    __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
    __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
    __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
    __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
    __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
    __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
    __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
    __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
    row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

static inline void winograd_f6k3_output_inplace_avx2(
    __m256& m0,
    __m256& m1,
    __m256& m2,
    __m256& m3,
    __m256& m4,
    __m256& m5,
    __m256& m6,
    __m256& m7, const float& bias, const bool& with_relu) {



    const __m256 m_32p0 = _mm256_set1_ps(32.f);
    const __m256 m_16p0 = _mm256_set1_ps(16.f);
    const __m256 m_8p0 = _mm256_set1_ps(8.f);
    const __m256 m_4p0 = _mm256_set1_ps(4.f);
    const __m256 m_2p0 = _mm256_set1_ps(2.f);

    const __m256 m_0p5 = _mm256_set1_ps(0.5f);
    const __m256 m_0p25 = _mm256_set1_ps(0.25f);
    const __m256 m_0p125 = _mm256_set1_ps(0.125f);
    const __m256 m_0p0625 = _mm256_set1_ps(0.0625f);
    const __m256 m_0p03125 = _mm256_set1_ps(0.03125f);

    __m256 m1_add_m2 = m1 + m2;
    __m256 m1_sub_m2 = m1 - m2;
    __m256 m3_add_m4 = m3 + m4;
    __m256 m3_sub_m4 = m3 - m4;
    __m256 m5_add_m6 = m5 + m6;
    __m256 m5_sub_m6 = m5 - m6;

    // Finised with M[0-6] as **inputs** here.
    m0 = m0 + m1_add_m2 + m3_add_m4 + m5_add_m6;
    m2 = m1_add_m2 + m_4p0 * m3_add_m4 + m_0p25 * m5_add_m6;
    m4 = m1_add_m2 + m3_add_m4 * m_16p0 + m5_add_m6 * m_0p0625;
    m1 = m1_sub_m2 + m3_sub_m4 * m_2p0 + m5_sub_m6 * m_0p5;
    m3 = m1_sub_m2 + m3_sub_m4 * m_8p0 + m5_sub_m6 * m_0p125;
    m5 = m7 + m1_sub_m2 + m3_sub_m4 * m_32p0 + m5_sub_m6 * m_0p03125;
    m6 = _mm256_setzero_ps();
    m7 = _mm256_setzero_ps();

    transpose8_ps(m0, m1, m2, m3, m4, m5, m6, m7);

    m1_add_m2 = m1 + m2;
    m1_sub_m2 = m1 - m2;
    m3_add_m4 = m3 + m4;
    m3_sub_m4 = m3 - m4;
    m5_add_m6 = m5 + m6;
    m5_sub_m6 = m5 - m6;

    const __m256 bias_value = _mm256_set1_ps(bias);
    const __m256 m_0p0 = _mm256_setzero_ps();

    if (with_relu) {
        m0 = _mm256_max_ps(bias_value + m0 + m1_add_m2 + m3_add_m4 + m5_add_m6, m_0p0);
        m2 = _mm256_max_ps(bias_value + m1_add_m2 + m_4p0 * m3_add_m4 + m_0p25 * m5_add_m6, m_0p0);
        m4 = _mm256_max_ps(bias_value + m1_add_m2 + m3_add_m4 * m_16p0 + m5_add_m6 * m_0p0625, m_0p0);
        m1 = _mm256_max_ps(bias_value + m1_sub_m2 + m3_sub_m4 * m_2p0 + m5_sub_m6 * m_0p5, m_0p0);
        m3 = _mm256_max_ps(bias_value + m1_sub_m2 + m3_sub_m4 * m_8p0 + m5_sub_m6 * m_0p125, m_0p0);
        m5 = _mm256_max_ps(bias_value + m7 + m1_sub_m2 + m3_sub_m4 * m_32p0 + m5_sub_m6 * m_0p03125, m_0p0);
    } else {
        m0 = bias_value + m0 + m1_add_m2 + m3_add_m4 + m5_add_m6;
        m2 = bias_value + m1_add_m2 + m_4p0 * m3_add_m4 + m_0p25 * m5_add_m6;
        m4 = bias_value + m1_add_m2 + m3_add_m4 * m_16p0 + m5_add_m6 * m_0p0625;
        m1 = bias_value + m1_sub_m2 + m3_sub_m4 * m_2p0 + m5_sub_m6 * m_0p5;
        m3 = bias_value + m1_sub_m2 + m3_sub_m4 * m_8p0 + m5_sub_m6 * m_0p125;
        m5 = bias_value + m7 + m1_sub_m2 + m3_sub_m4 * m_32p0 + m5_sub_m6 * m_0p03125;
    }


}

static inline void winograd_f6k3_output_inplace_avx2_float_in(
    __m256& m0,
    __m256& m1,
    __m256& m2,
    __m256& m3,
    __m256& m4,
    __m256& m5,
    __m256& m6,
    __m256& m7, float* din, const float& bias, const bool& with_relu) {



    const __m256 m_32p0 = _mm256_set1_ps(32.f);
    const __m256 m_16p0 = _mm256_set1_ps(16.f);
    const __m256 m_8p0 = _mm256_set1_ps(8.f);
    const __m256 m_4p0 = _mm256_set1_ps(4.f);
    const __m256 m_2p0 = _mm256_set1_ps(2.f);

    const __m256 m_0p5 = _mm256_set1_ps(0.5f);
    const __m256 m_0p25 = _mm256_set1_ps(0.25f);
    const __m256 m_0p125 = _mm256_set1_ps(0.125f);
    const __m256 m_0p0625 = _mm256_set1_ps(0.0625f);
    const __m256 m_0p03125 = _mm256_set1_ps(0.03125f);

    m0 = _mm256_loadu_ps(&din[0 * 8]);
    m1 = _mm256_loadu_ps(&din[1 * 8]);
    m2 = _mm256_loadu_ps(&din[2 * 8]);
    m3 = _mm256_loadu_ps(&din[3 * 8]);
    m4 = _mm256_loadu_ps(&din[4 * 8]);
    m5 = _mm256_loadu_ps(&din[5 * 8]);
    m6 = _mm256_loadu_ps(&din[6 * 8]);
    m7 = _mm256_loadu_ps(&din[7 * 8]);

    __m256 m1_add_m2 = m1 + m2;
    __m256 m1_sub_m2 = m1 - m2;
    __m256 m3_add_m4 = m3 + m4;
    __m256 m3_sub_m4 = m3 - m4;
    __m256 m5_add_m6 = m5 + m6;
    __m256 m5_sub_m6 = m5 - m6;

    // Finised with M[0-6] as **inputs** here.
    m0 = m0 + m1_add_m2 + m3_add_m4 + m5_add_m6;
    m2 = m1_add_m2 + m_4p0 * m3_add_m4 + m_0p25 * m5_add_m6;
    m4 = m1_add_m2 + m3_add_m4 * m_16p0 + m5_add_m6 * m_0p0625;
    m1 = m1_sub_m2 + m3_sub_m4 * m_2p0 + m5_sub_m6 * m_0p5;
    m3 = m1_sub_m2 + m3_sub_m4 * m_8p0 + m5_sub_m6 * m_0p125;
    m5 = m7 + m1_sub_m2 + m3_sub_m4 * m_32p0 + m5_sub_m6 * m_0p03125;
    m6 = _mm256_setzero_ps();
    m7 = _mm256_setzero_ps();

    transpose8_ps(m0, m1, m2, m3, m4, m5, m6, m7);

    m1_add_m2 = m1 + m2;
    m1_sub_m2 = m1 - m2;
    m3_add_m4 = m3 + m4;
    m3_sub_m4 = m3 - m4;
    m5_add_m6 = m5 + m6;
    m5_sub_m6 = m5 - m6;

    const __m256 bias_value = _mm256_set1_ps(bias);
    const __m256 m_0p0 = _mm256_setzero_ps();

    if (with_relu) {
        m0 = _mm256_max_ps(bias_value + m0 + m1_add_m2 + m3_add_m4 + m5_add_m6, m_0p0);
        m2 = _mm256_max_ps(bias_value + m1_add_m2 + m_4p0 * m3_add_m4 + m_0p25 * m5_add_m6, m_0p0);
        m4 = _mm256_max_ps(bias_value + m1_add_m2 + m3_add_m4 * m_16p0 + m5_add_m6 * m_0p0625, m_0p0);
        m1 = _mm256_max_ps(bias_value + m1_sub_m2 + m3_sub_m4 * m_2p0 + m5_sub_m6 * m_0p5, m_0p0);
        m3 = _mm256_max_ps(bias_value + m1_sub_m2 + m3_sub_m4 * m_8p0 + m5_sub_m6 * m_0p125, m_0p0);
        m5 = _mm256_max_ps(bias_value + m7 + m1_sub_m2 + m3_sub_m4 * m_32p0 + m5_sub_m6 * m_0p03125, m_0p0);
    } else {
        m0 = bias_value + m0 + m1_add_m2 + m3_add_m4 + m5_add_m6;
        m2 = bias_value + m1_add_m2 + m_4p0 * m3_add_m4 + m_0p25 * m5_add_m6;
        m4 = bias_value + m1_add_m2 + m3_add_m4 * m_16p0 + m5_add_m6 * m_0p0625;
        m1 = bias_value + m1_sub_m2 + m3_sub_m4 * m_2p0 + m5_sub_m6 * m_0p5;
        m3 = bias_value + m1_sub_m2 + m3_sub_m4 * m_8p0 + m5_sub_m6 * m_0p125;
        m5 = bias_value + m7 + m1_sub_m2 + m3_sub_m4 * m_32p0 + m5_sub_m6 * m_0p03125;
    }

}

static inline void winograd_f6k3_input_inplace_avx2(
    __m256& m0,
    __m256& m1,
    __m256& m2,
    __m256& m3,
    __m256& m4,
    __m256& m5,
    __m256& m6,
    __m256& m7) {
    const __m256 m_5p25 = _mm256_set1_ps(5.25f);
    const __m256 m_4p25 = _mm256_set1_ps(4.25f);
    const __m256 m_4p0 = _mm256_set1_ps(4.f);
    const __m256 m_2p5 = _mm256_set1_ps(2.5f);
    const __m256 m_2p0 = _mm256_set1_ps(2.f);
    const __m256 m_1p25 = _mm256_set1_ps(1.25f);
    const __m256 m_0p5 = _mm256_set1_ps(0.5f);
    const __m256 m_0p25 = _mm256_set1_ps(0.25f);
    m0 = m0 - m6 + (m4 - m2) * m_5p25;
    m7 = m7 - m1 + (m3 - m5) * m_5p25;

    __m256 t1 = m2 + m6 - m4 * m_4p25;
    __m256 t2 = m1 + m5 - m3 * m_4p25;

    __m256 s1 = m4 * m_1p25;
    __m256 s2 = m3 * m_2p5;

    __m256 p1 = m6 + (m2 * m_0p25 - s1);
    __m256 p2 = m1 * m_0p5 - s2 + m5 * m_2p0;

    m3 = p1 + p2;
    m4 = p1 - p2;


    p1 = m6 + (m2 - s1) * m_4p0;
    p2 = m1 * m_2p0 - s2 + m5 * m_0p5;

    m5 = p1 + p2;
    m6 = p1 - p2;

    m1 = _mm256_add_ps(t1, t2);
    m2 = _mm256_sub_ps(t1, t2);

    transpose8_ps(m0, m1, m2, m3, m4, m5, m6, m7);

    m0 = m0 - m6 + (m4 - m2) * m_5p25;
    m7 = m7 - m1 + (m3 - m5) * m_5p25;

    t1 = m2 + m6 - m4 * m_4p25;
    t2 = m1 + m5 - m3 * m_4p25;

    s1 = m4 * m_1p25;
    s2 = m3 * m_2p5;

    p1 = m6 + (m2 * m_0p25 - s1);
    p2 = m1 * m_0p5 - s2 + m5 * m_2p0;

    m3 = p1 + p2;
    m4 = p1 - p2;


    p1 = m6 + (m2 - s1) * m_4p0;
    p2 = m1 * m_2p0 - s2 + m5 * m_0p5;

    m5 = p1 + p2;
    m6 = p1 - p2;

    m1 = _mm256_add_ps(t1, t2);
    m2 = _mm256_sub_ps(t1, t2);
}

static inline void winograd_f6k3_input_inplace_avx2(
    __m256& m0,
    __m256& m1,
    __m256& m2,
    __m256& m3,
    __m256& m4,
    __m256& m5,
    __m256& m6,
    __m256& m7, float* out) {
    const __m256 m_5p25 = _mm256_set1_ps(5.25f);
    const __m256 m_4p25 = _mm256_set1_ps(4.25f);
    const __m256 m_4p0 = _mm256_set1_ps(4.f);
    const __m256 m_2p5 = _mm256_set1_ps(2.5f);
    const __m256 m_2p0 = _mm256_set1_ps(2.f);
    const __m256 m_1p25 = _mm256_set1_ps(1.25f);
    const __m256 m_0p5 = _mm256_set1_ps(0.5f);
    const __m256 m_0p25 = _mm256_set1_ps(0.25f);
    m0 = m0 - m6 + (m4 - m2) * m_5p25;
    m7 = m7 - m1 + (m3 - m5) * m_5p25;

    __m256 t1 = m2 + m6 - m4 * m_4p25;
    __m256 t2 = m1 + m5 - m3 * m_4p25;

    __m256 s1 = m4 * m_1p25;
    __m256 s2 = m3 * m_2p5;

    __m256 p1 = m6 + (m2 * m_0p25 - s1);
    __m256 p2 = m1 * m_0p5 - s2 + m5 * m_2p0;

    m3 = p1 + p2;
    m4 = p1 - p2;


    p1 = m6 + (m2 - s1) * m_4p0;
    p2 = m1 * m_2p0 - s2 + m5 * m_0p5;

    m5 = p1 + p2;
    m6 = p1 - p2;

    m1 = _mm256_add_ps(t1, t2);
    m2 = _mm256_sub_ps(t1, t2);

    transpose8_ps(m0, m1, m2, m3, m4, m5, m6, m7);

    m0 = m0 - m6 + (m4 - m2) * m_5p25;
    m7 = m7 - m1 + (m3 - m5) * m_5p25;
    _mm256_storeu_ps(out + 0 * 8, m0);
    _mm256_storeu_ps(out + 7 * 8, m7);

    t1 = m2 + m6 - m4 * m_4p25;
    t2 = m1 + m5 - m3 * m_4p25;

    s1 = m4 * m_1p25;
    s2 = m3 * m_2p5;

    p1 = m6 + (m2 * m_0p25 - s1);
    p2 = m1 * m_0p5 - s2 + m5 * m_2p0;

    m3 = p1 + p2;
    m4 = p1 - p2;
    _mm256_storeu_ps(out + 3 * 8, m3);
    _mm256_storeu_ps(out + 4 * 8, m4);

    p1 = m6 + (m2 - s1) * m_4p0;
    p2 = m1 * m_2p0 - s2 + m5 * m_0p5;

    m5 = p1 + p2;
    m6 = p1 - p2;
    _mm256_storeu_ps(out + 5 * 8, m5);
    _mm256_storeu_ps(out + 6 * 8, m6);

    m1 = _mm256_add_ps(t1, t2);
    m2 = _mm256_sub_ps(t1, t2);
    _mm256_storeu_ps(out + 1 * 8, m1);
    _mm256_storeu_ps(out + 2 * 8, m2);
}

static void winograd_all_in_one(const float* din, float* dout, \
                                int num, int chout, int hout, int wout, \
                                int chin, int hin, int win, \
                                const float* weights, const float* bias, \
                                int pad_w, int pad_h, bool flag_bias, bool flag_relu, float* tmp_work_space) {
    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    //! transform input
    int tile_w = (wout + 5) / 6;
    int tile_h = (hout + 5) / 6;
    int size_tile = tile_h * tile_w;
    int size_trans_channel = 8 * 8 * size_tile;
    int max_ch = chin > chout ? chin : chout;

    for (int oc = 0; oc < chout; oc++) {

        for (int h = 0; h < tile_h; h++) {

            for (int w = 0; w < tile_w; w++) {
                __m256 result[8] = {_mm256_setzero_ps()};

                for (int ic = 0; ic < chin; ++ic) {
                    //! prepare data 8x8
                    //! row 8
                    __m256 data_in_tmp[8] = {_mm256_setzero_ps()};
                    const float* din_channel = din + ic * size_in_channel;

                    //memset(data_in_tmp[0], 0, sizeof(float) * 64);
                    for (int j = 0; j < 8; ++j) {
                        int start_row = h * 6 + j - pad_h;

                        if (start_row >= 0 && start_row < hin) {
                            for (int k = 0; k < 8; ++k) {
                                int start_col = w * 6 + k - pad_w;

                                if (start_col >= 0 && start_col < win) {
                                    data_in_tmp[j][k] = din_channel[start_row * win + start_col];
                                }
                            }
                        }
                    }

                    winograd_f6k3_input_inplace_avx2(data_in_tmp[0], data_in_tmp[1], data_in_tmp[2], data_in_tmp[3],
                                                     data_in_tmp[4],
                                                     data_in_tmp[5], data_in_tmp[6], data_in_tmp[7]);


                    //                    exit(0);
                    /////////////////////////////////////
                    for (int i = 0; i < 8; i++) {
                        int weights_index = oc * chin * 64 + ic * 64;
                        result[i] += data_in_tmp[i] * _mm256_loadu_ps(&weights[weights_index + i * 8]);
                    }
                }

                float bias_value = flag_bias ? bias[oc] : 0.f;
                //output
                winograd_f6k3_output_inplace_avx2(result[0], result[1], result[2], result[3], result[4],
                                                  result[5], result[6], result[7], bias_value, flag_relu);

                float* dout_channel = dout + oc * hout * wout;

                for (int j = 0; j < 6; ++j) {
                    int end_row = h * 6 + j;

                    if (end_row < hout) {
                        for (int k = 0; k < 6; ++k) {
                            int end_col = w * 6 + k;

                            if (end_col < wout) {
                                dout_channel[end_row * wout + end_col] = result[j][k];
                            }
                        }
                    }
                }
            }
        }
    }

}


static void conv_x86_winograd3x3_avx2_opt(const float* din, float* dout, \
        int num, int chout, int hout, int wout, \
        int chin, int hin, int win, \
        const float* weights, const float* bias, \
        int pad_w, int pad_h, bool flag_bias, bool flag_relu, float* tmp_work_space) {
    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    //! transform input
    int tile_w = (wout + 5) / 6;
    int tile_h = (hout + 5) / 6;
    int size_tile = tile_h * tile_w;
    int size_trans_channel = 8 * 8 * size_tile;
    int max_ch = chin > chout ? chin : chout;

    int m = chout;
    int n = size_tile;
    int k = chin;


    //! tmp data buffer for input transform
    float* tmp_data1 = tmp_work_space;
    //! tmp data buffer for dot mul
    float* tmp_data2 = tmp_data1 + size_trans_channel * max_ch;

    //SaberTimer<ARM> t1;
    //Context<ARM> ctx1;


    for (int i = 0; i < num; ++i) {

        const float* din_batch = static_cast<const float*>(din) + i * chin * size_in_channel;
        float* dout_batch = static_cast<float*>(dout) + i * chout * size_out_channel;

        //t1.start(ctx1);
        //! transform input Bt * data * B
#if 1
        #pragma omp parallel for schedule(static)

        for (int j = 0; j < chin; ++j) {

            const float* din_channel = din_batch + j * size_in_channel;
            float* data_trans_channel = tmp_data1 + j * size_trans_channel;

            for (int h = 0; h < tile_h; h++) {

                for (int w = 0; w < tile_w; w ++) {
                    //! prepare data 8x8
                    //! row 8
                    __m256 data_in_tmp[8] = {_mm256_setzero_ps()};

                    //memset(data_in_tmp[0], 0, sizeof(float) * 64);
                    for (int j = 0; j < 8; ++j) {
                        int start_row = h * 6 + j - pad_h;

                        if (start_row >= 0 && start_row < hin) {
                            int start_col = w * 6 - pad_w;

                            if (start_col >= 0) {
                                if (win - start_col >= 8) {
                                    data_in_tmp[j] = _mm256_loadu_ps(&din_channel[start_row * win + start_col]);
                                } else {
                                    int remainder = win - start_col;
                                    data_in_tmp[j] = _mm256_maskload_ps(&din_channel[start_row * win + start_col],
                                                                        _m256_continue_mask_m256i(remainder));
                                }
                            } else {
                                for (int k = 0; k < 8; ++k) {
                                    int start_col = w * 6 + k - pad_w;

                                    if (start_col >= 0 && start_col < win) {
                                        data_in_tmp[j][k] = din_channel[start_row * win + start_col];
                                    }
                                }
                            }

                        }
                    }

                    winograd_f6k3_input_inplace_avx2(data_in_tmp[0], data_in_tmp[1], data_in_tmp[2], data_in_tmp[3],
                                                     data_in_tmp[4],
                                                     data_in_tmp[5], data_in_tmp[6], data_in_tmp[7], data_trans_channel);

                    data_trans_channel += 64;
                }
            }
        }

#endif
        //! end of transform input

#if 1
        ////////////////////////////////////////////////////////////////////////////////
        //! dot mul
        //! transpose input, convert from ch_in * tile_h * tile_w * 64 to
        //! 64 * ch_in * tile_h * tile_w
        int hblock = 16;
        int m_round = hblock * ((chout + hblock - 1) / hblock);
        int stride_a = m_round * chin;
        int stride_b = chin * size_tile;
        int stride_c = chout * size_tile;
#if 1
        MKL_Somatcopy('R', 'T', stride_b, 64, 1.f, tmp_data1, 64, tmp_data2, stride_b);
#endif


        CBLAS_TRANSPOSE trans[1] = {CblasNoTrans};
        int m_array[1] = {chout};
        int n_array[1] = {size_tile};
        int k_array[1] = {chin};
        int lda_array[1] = {chin};
        int ldb_array[1] = {size_tile};
        int ldc_array[1] = {size_tile};
        float alpha_array[1] = {1.f};
        float beta_array[1] = {0.f};
        const float* ptr_a_array[64];
        const float* ptr_b_array[64];
        float* ptr_c_array[64];
        int group_size[1] = {64};

        for (int l = 0; l < 64; ++l) {
            ptr_a_array[l] = static_cast<const float*>(weights) + l * chout * chin;
            ptr_b_array[l] = tmp_data2 + l * stride_b;
            ptr_c_array[l] = tmp_data1 + l * stride_c;

        }


        cblas_sgemm_batch(CblasRowMajor, trans, trans, m_array, n_array, k_array, alpha_array, ptr_a_array,
                          lda_array, ptr_b_array, ldb_array, beta_array, ptr_c_array, ldc_array, 1, group_size);

        //! transpose output, convert from 64 * ch_out * tile_h * tile_w to
        //! ch_out * tile_h * tile_w * 64
#if 1
        MKL_Somatcopy('R', 'T', 64, stride_c, 1.f, tmp_data1, stride_c, tmp_data2, 64);
#endif
        //! end of dot mul
#endif

#if 1
        ///////////////////////////////////////////////////////////////////////////////
        //! transform output
        #pragma omp parallel for schedule(static)

        for (int i = 0; i < chout; ++i) {

            float bias_value = flag_bias ? static_cast<const float*>(bias)[i] : 0.f;
            float* dout_tmp = tmp_data2 + i * size_trans_channel;
            float* dout_channel = dout_batch + i * size_out_channel;

            for (int h = 0; h < tile_h; ++h) {
                for (int w = 0; w < tile_w; ++w) {

                    __m256 out_tmp[8];

                    winograd_f6k3_output_inplace_avx2_float_in(out_tmp[0], out_tmp[1], out_tmp[2], out_tmp[3],
                            out_tmp[4], out_tmp[5], out_tmp[6], out_tmp[7], dout_tmp, bias_value, flag_relu);
                    dout_tmp += 64;

                    for (int j = 0; j < 6; ++j) {
                        int end_row = h * 6 + j;

                        if (end_row < hout) {
                            int end_col = w * 6 ;

                            int remainder = std::min(wout - end_col, 6);
                            _mm256_maskstore_ps(&dout_channel[end_row * wout + end_col], _m256_continue_mask_m256i(remainder),
                                                out_tmp[j]);
                        }
                    }
                }
            }
        }

        //! end of transform output
#endif
    }
}

template <>
SaberStatus SaberConvWinogradAvx2<AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    ConvParam<X86>* conv_param = &param.conv_param;
    int batch_size = inputs[0]->num();
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    int kernel_h = conv_param->weight()->height();
    int kernel_w = conv_param->weight()->width();
    int in_stride = in_h * in_w;
    int out_stride = out_h * out_w;
    int group = conv_param->group;
    const float* weights_d = (const float*)conv_param->weight()->data();
    _winor_weights.re_alloc(Shape({8, 8, out_c, in_c}));
    Tensor<X86> trans_temp(Shape({8, 8, out_c, in_c}));
    float* trans_tmp_ptr = static_cast<float*>(trans_temp.mutable_data());

    winograd_transform_weights(static_cast<float*>(_winor_weights.mutable_data()),
    static_cast<float*>(conv_param->weight()->data()), out_c, in_c,
                               trans_tmp_ptr);


    int tile_w = (out_w + 5) / 6;
    int tile_h = (out_h + 5) / 6;
    int size_tile = tile_h * tile_w;
    int size_trans_channel = 8 * 8 * size_tile;
    int max_ch = in_c > out_c ? in_c : out_c;
    _winor_temp.re_alloc(Shape({1, 2, max_ch, size_trans_channel}));


    return SaberSuccess;
}

template <>
SaberStatus SaberConvWinogradAvx2<AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    LOG(INFO) << "SaberConvWinogradAvx2 init";
    return create(inputs, outputs, param, ctx);

}


template <>
SaberStatus SaberConvWinogradAvx2<AK_FLOAT>::dispatch(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param) {
    ConvParam<X86>* conv_param = &param.conv_param;
    int batch_size = inputs[0]->num();
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    int kernel_h = conv_param->weight()->height();
    int kernel_w = conv_param->weight()->width();
    int in_stride = in_h * in_w;
    int out_stride = out_h * out_w;
    int group = conv_param->group;
    int weight_size_per_group = (out_c / group) * (in_c / group) * kernel_h * kernel_w;
    const float* bias_ptr = nullptr;

    if (conv_param->bias() != nullptr && conv_param->bias()->valid_size() > 0) {
        bias_ptr = static_cast<const float*>(conv_param->bias()->data());
    }

    bool with_relu = conv_param->activation_param.active == Active_relu;


    const float* din = (const float*)inputs[0]->data();
    float* dout = (float*)outputs[0]->mutable_data();

    conv_x86_winograd3x3_avx2_opt(din, dout, batch_size, out_c, out_h, out_w, in_c, in_h, in_w,
                                  static_cast<const float*>(_winor_weights.data()),
                                  bias_ptr, conv_param->pad_w, conv_param->pad_h, bias_ptr != nullptr, with_relu,
                                  static_cast<float*>(_winor_temp.mutable_data()));

    return SaberSuccess;
}

#else
template <>
SaberStatus SaberConvWinogradAvx2<AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param, Context<X86>& ctx) {


    return SaberUnImplError;
}

template <>
SaberStatus SaberConvWinogradAvx2<AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);

}


template <>
SaberStatus SaberConvWinogradAvx2<AK_FLOAT>::dispatch(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param) {


    return SaberUnImplError;
}

#endif

}
}
