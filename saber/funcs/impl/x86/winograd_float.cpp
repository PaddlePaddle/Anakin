#include "saber/funcs/impl/x86/winograd_float.h"
#include "mkl_cblas.h"
#include "mkl_trans.h"
#ifdef USE_SGX
extern "C" void mkl_free_buffers();
#endif

namespace anakin {
namespace saber {

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

    float* ptr_out = work_space;

    for (int i = 0; i < ch_out; i++) {
        for (int j = 0; j < ch_in; j++) {
            const float* kernel0 = din + (i * ch_in + j) * 9;
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

static void winograd_transform_weights_oc_ic_64(float* dout, const float* din, int ch_out, \
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

    float* ptr_out = dout;

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

}
template <>
SaberStatus SaberConvWinogradFloat<AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
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

    winograd_transform_weights(static_cast<float*>(_winor_weights.mutable_data()), static_cast<const float*>(conv_param->weight()->data()), out_c, in_c,
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
SaberStatus SaberConvWinogradFloat<AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    LOG(INFO)<<"SaberConvWinogradFloat init";
    return create(inputs, outputs, param, ctx);

}

static void gemm(const bool trans_a, const bool transb, int m, int n, int k, const float alpha,
                 const float* a, const float* b, const float beta, float* c) {
    //    cout << "(" << m << "," << n << "," << k << ")" << endl;
    int lda = (!trans_a/* == CblasNoTrans*/) ? k : m;
    int ldb = (!transb/* == CblasNoTrans*/) ? n : k;
    CBLAS_TRANSPOSE cblas_transa =
        (!trans_a/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cblas_transb =
        (!transb/* == CblasNoTrans*/) ? CblasNoTrans : CblasTrans;
    cblas_sgemm(CblasRowMajor, cblas_transa, cblas_transb, m, n, k, alpha, a, k, b, n, beta, c, n);
#ifdef USE_SGX
    mkl_free_buffers();
#endif
};

static void print_hw(const float* in, int h, int w) {
    printf("\n");

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%f \t", in[i * w + j]);
        }

        printf("\n");
    }

}


/**
 * \brief winograd conv, transform input, f6x3
 * dout = BT * d * B, whrer B is the transform
 * BT = 1      0   -21/4       0     21/4        0   -1   0
 *      0      1       1   -17/4    -17/4        1    1   0
 *      0     -1       1    17/4    -17/4       -1    1   0
 *      0    1/2     1/4    -5/2     -5/4        2    1   0
 *      0   -1/2     1/4     5/2     -5/4       -2    1   0
 *      0      2       4    -5/2       -5      1/2    1   0
 *      0     -2       4     5/2       -5     -1/2    1   0
 *      0     -1       0    21/4        0    -21/4    0   1
 * @param dout
 * @param din
 */
inline  void transform_input_f6x6(float* dout, const float* din) {
    float tmp[8][8];

    //! BT * d
    for (int m = 0; m < 8; m++) {
        tmp[0][m] = din[0] - din[6] + (din[4] - din[2]) * 5.25f;
        tmp[7][m] = din[7] - din[1] + (din[3] - din[5]) * 5.25f;

        float tmp12a = din[2] + din[6] - din[4] * 4.25f;
        float tmp12b = din[1] + din[5] - din[3] * 4.25f;

        tmp[1][m] = tmp12a + tmp12b;
        tmp[2][m] = tmp12a - tmp12b;

        float tmp34a = din[6] + din[2] * 0.25f - din[4] * 1.25f;
        float tmp34b = din[1] * 0.5f - din[3] * 2.5f + din[5] * 2.f;

        tmp[3][m] = tmp34a + tmp34b;
        tmp[4][m] = tmp34a - tmp34b;

        float tmp56a = din[6] + (din[2] - din[4] * 1.25f) * 4.f;
        float tmp56b = din[1] * 2.f - din[3] * 2.5f + din[5] * 0.5f;

        tmp[5][m] = tmp56a + tmp56b;
        tmp[6][m] = tmp56a - tmp56b;

        din += 8;
    }

    for (int m = 0; m < 8; m++) {
        const float* tmp0 = tmp[m];

        dout[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25f;
        dout[7] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25f;

        float tmp12a = tmp0[2] + tmp0[6] - tmp0[4] * 4.25f;
        float tmp12b = tmp0[1] + tmp0[5] - tmp0[3] * 4.25f;

        dout[1] = tmp12a + tmp12b;
        dout[2] = tmp12a - tmp12b;

        float tmp34a = tmp0[6] + tmp0[2] * 0.25f - tmp0[4] * 1.25f;
        float tmp34b = tmp0[1] * 0.5f - tmp0[3] * 2.5f + tmp0[5] * 2.f;

        dout[3] = tmp34a + tmp34b;
        dout[4] = tmp34a - tmp34b;

        float tmp56a = tmp0[6] + (tmp0[2] - tmp0[4] * 1.25f) * 4.f;
        float tmp56b = tmp0[1] * 2.f - tmp0[3] * 2.5f + tmp0[5] * 0.5f;

        dout[5] = tmp56a + tmp56b;
        dout[6] = tmp56a - tmp56b;

        dout += 8;
    }
}

/**
 * \brief winograd conv, transform input, f6x3
 * dout = BT * d * B, whrer B is the transform
 * BT = 1      0   -21/4       0     21/4        0   -1   0
 *      0      1       1   -17/4    -17/4        1    1   0
 *      0     -1       1    17/4    -17/4       -1    1   0
 *      0    1/2     1/4    -5/2     -5/4        2    1   0
 *      0   -1/2     1/4     5/2     -5/4       -2    1   0
 *      0      2       4    -5/2       -5      1/2    1   0
 *      0     -2       4     5/2       -5     -1/2    1   0
 *      0     -1       0    21/4        0    -21/4    0   1
 * @param dout
 * @param din
 */
inline  void transform_input_f6x6_c8(float* dout, const float* din) {
    float tmp[8][8][8];

    //! BT * d
    for (int m = 0; m < 8; m++) {
        for (int i = 0; i < 8; i++) {
            tmp[0][m][i] = din[0] - din[6] + (din[4] - din[2]) * 5.25f;
            tmp[7][m][i] = din[7] - din[1] + (din[3] - din[5]) * 5.25f;

            float tmp12a = din[2] + din[6] - din[4] * 4.25f;
            float tmp12b = din[1] + din[5] - din[3] * 4.25f;

            tmp[1][m][i] = tmp12a + tmp12b;
            tmp[2][m][i] = tmp12a - tmp12b;

            float tmp34a = din[6] + din[2] * 0.25f - din[4] * 1.25f;
            float tmp34b = din[1] * 0.5f - din[3] * 2.5f + din[5] * 2.f;

            tmp[3][m][i] = tmp34a + tmp34b;
            tmp[4][m][i] = tmp34a - tmp34b;

            float tmp56a = din[6] + (din[2] - din[4] * 1.25f) * 4.f;
            float tmp56b = din[1] * 2.f - din[3] * 2.5f + din[5] * 0.5f;

            tmp[5][m][i] = tmp56a + tmp56b;
            tmp[6][m][i] = tmp56a - tmp56b;
            din += 8;
        }

    }

    for (int m = 0; m < 8; m++) {
        for (int i = 0; i < 8; i++) {
            const float* tmp0 = tmp[m][i];

            dout[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25f;
            dout[7] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25f;

            float tmp12a = tmp0[2] + tmp0[6] - tmp0[4] * 4.25f;
            float tmp12b = tmp0[1] + tmp0[5] - tmp0[3] * 4.25f;

            dout[1] = tmp12a + tmp12b;
            dout[2] = tmp12a - tmp12b;

            float tmp34a = tmp0[6] + tmp0[2] * 0.25f - tmp0[4] * 1.25f;
            float tmp34b = tmp0[1] * 0.5f - tmp0[3] * 2.5f + tmp0[5] * 2.f;

            dout[3] = tmp34a + tmp34b;
            dout[4] = tmp34a - tmp34b;

            float tmp56a = tmp0[6] + (tmp0[2] - tmp0[4] * 1.25f) * 4.f;
            float tmp56b = tmp0[1] * 2.f - tmp0[3] * 2.5f + tmp0[5] * 0.5f;

            dout[5] = tmp56a + tmp56b;
            dout[6] = tmp56a - tmp56b;

            dout += 8;
        }
    }
}


inline void transform_output_f6x6(float* output, const float* din, float bias) {
    float tmp[6][8];

    for (int m = 0; m < 8; m++) {
        float tmp024a = din[1] + din[2];
        float tmp135a = din[1] - din[2];

        float tmp024b = din[3] + din[4];
        float tmp135b = din[3] - din[4];

        float tmp024c = din[5] + din[6];
        float tmp135c = din[5] - din[6];

        tmp[0][m] = din[0] + tmp024a + tmp024b + tmp024c;
        tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 0.25f;
        tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c * 0.0625f;

        tmp[1][m] = tmp135a + tmp135b * 2 + tmp135c * 0.5f;
        tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 0.125f;
        tmp[5][m] = din[7] + tmp135a + tmp135b * 32 + tmp135c * 0.03125f;

        din += 8;
    }

    for (int m = 0; m < 6; m++) {
        const float* tmp0 = tmp[m];

        float tmp024a = tmp0[1] + tmp0[2];
        float tmp135a = tmp0[1] - tmp0[2];

        float tmp024b = tmp0[3] + tmp0[4];
        float tmp135b = tmp0[3] - tmp0[4];

        float tmp024c = tmp0[5] + tmp0[6];
        float tmp135c = tmp0[5] - tmp0[6];

        output[0] = bias + tmp0[0] + tmp024a + tmp024b + tmp024c;
        output[2] = bias + tmp024a + tmp024b * 4 + tmp024c * 0.25f;
        output[4] = bias + tmp024a + tmp024b * 16 + tmp024c * 0.0625f;

        output[1] = bias + tmp135a + tmp135b * 2 + tmp135c * 0.5f;
        output[3] = bias + tmp135a + tmp135b * 8 + tmp135c * 0.125f;
        output[5] = bias + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c * 0.03125f;

        output += 6;
    }
}

static void load_data_2_ic_th_tw_64_8(int pad_h, int pad_w, int tile_h, int tile_w, int chin,
                                      int hin,
                                      int win, const float* din_batch, float* dout) {
    int size_in_channel = win * hin * 8;
    int chin_div_up_8 = chin / 8;

    for (int ic = 0; ic < chin_div_up_8; ++ic) {
        for (int h = 0; h < tile_h; h++) {
            for (int w = 0; w < tile_w; w++) {

                const float* din_channel = din_batch + ic * size_in_channel;
                float* data_trans_channel = dout + ic * tile_h * tile_w * 64 * 8 + h * tile_w * 64 * 8 + w * 64 * 8;
                //! prepare data 8x8
                //! row 8
                float data_in_tmp[8][8][8] = {0.f};

                //memset(data_in_tmp[0], 0, sizeof(float) * 64);
                for (int j = 0; j < 8; ++j) {
                    int start_row = h * 6 + j - pad_h;

                    if (start_row >= 0 && start_row < hin) {
                        for (int k = 0; k < 8; ++k) {
                            int start_col = w * 6 + k - pad_w;

                            if (start_col >= 0 && start_col < win) {
                                for (int i = 0; i < 8; i++) {
                                    data_in_tmp[j][k][i] = din_channel[start_row * win * 8 + start_col * 8 + i];
                                }
                            }
                        }
                    }
                }

                //                        print_hw(&data_in_tmp[0][0],8,8);
                transform_input_f6x6(data_trans_channel, &data_in_tmp[0][0][0]);

                //                        print_hw(data_trans_channel,8,8);
                //                        exit(0);


            }
        }
    }
}





static void conv_x86_winograd3x3(const void *din, void *dout, \
                                 int num, int chout, int hout, int wout, \
                                 int chin, int hin, int win, \
                                 const void *weights, const void *bias, \
                                 int pad_w, int pad_h, bool flag_bias, bool flag_relu, float *tmp_work_space) {
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
                    float data_in_tmp[8][8] = {0.f};

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

                    transform_input_f6x6(data_trans_channel, &data_in_tmp[0][0]);
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
        //        transpose(tmp_data2, tmp_data1, 64, stride_b);


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
        //        transpose(tmp_data2, tmp_data1, stride_c, 64);
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

                    float out_tmp[6][6];

                    transform_output_f6x6(out_tmp[0], dout_tmp, bias_value);
                    dout_tmp += 64;

                    for (int j = 0; j < 6; ++j) {
                        int end_row = h * 6 + j;

                        if (end_row < hout) {
                            for (int k = 0; k < 6; ++k) {
                                int end_col = w * 6 + k;

                                if (end_col < wout) {
                                    if (flag_relu) {
                                        dout_channel[end_row * wout + end_col] = out_tmp[j][k] > 0.f ? out_tmp[j][k] : 0.f;
                                    } else {
                                        dout_channel[end_row * wout + end_col] = out_tmp[j][k];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        //! end of transform output
#endif
        //t1.end(ctx1);
        //LOG(INFO) << "winograd conv transform output time: " << t1.get_average_ms();
    }
}


template <>
SaberStatus SaberConvWinogradFloat<AK_FLOAT>::dispatch(const std::vector<Tensor<X86> *>& inputs,
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
        bias_ptr = static_cast<const float *>(conv_param->bias()->data());
    }

    bool with_relu = conv_param->activation_param.active == Active_relu;


    const float* din = (const float*)inputs[0]->data();
    float* dout = (float*)outputs[0]->mutable_data();

    conv_x86_winograd3x3(din, dout, batch_size, out_c, out_h, out_w, in_c, in_h, in_w,
                         static_cast<const float *>(_winor_weights.data()),
                         bias_ptr, conv_param->pad_w, conv_param->pad_h, bias_ptr != nullptr, with_relu,
                         static_cast<float *>(_winor_temp.mutable_data()));
    return SaberSuccess;
}

}
}
