#include "saber/funcs/impl/arm/impl/conv_arm_impl.h"

#ifdef USE_ARM_PLACE

#include "saber/funcs/timer.h"

namespace anakin{

namespace saber{

void transpose(float* data_out, const float* data_in, int w_in, int h_in);
void transform_input_f6x6(float* dout, const float* din);
void transform_output_f6x6(float* output, const float* din, float bias);
#if 0
ConvWinogradF63::ConvWinogradF63() {

}

ConvWinogradF63::~ConvWinogradF63() {

}

bool ConvWinogradF63::init(const size_t l1_cache, const size_t l2_cache, \
    const int chout, const int chin, const int hin, \
    const int win, const int threads) {

    return true;
}

bool ConvWinogradF63::operator()(const float *trans_weights, const float *din, \
    float *dout, void *workspace) {

    return true;
}
#endif
void conv_arm_winograd3x3(Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, Tensor<ARM, AK_FLOAT, NCHW>& tensor_in, \
    const float* weights, const float* bias, \
    int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
    int pad_w, int pad_h, bool flag_bias, bool flag_relu, Sgemm& gemmer, void* work_space) {
    int w_in = tensor_in.width();
    int h_in = tensor_in.height();
    int ch_in = tensor_in.channel();
    int num = tensor_in.num();

    int w_out = tensor_out.width();
    int h_out = tensor_out.height();
    int ch_out = tensor_out.channel();

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;

    //! transform input
    int tile_w = (w_out + 5) / 6;
    int tile_h = (h_out + 5) / 6;
    int size_tile = tile_h * tile_w;
    int size_trans_channel = 8 * 8 * size_tile;
    int max_ch = ch_in > ch_out? ch_in : ch_out;

    //! tmp data buffer for input transform
    float* tmp_data1 = (float*)work_space;
    //! tmp data buffer for dot mul
    float* tmp_data2 = tmp_data1 + size_trans_channel * max_ch;

    //SaberTimer<ARM> t1;
    //Context<ARM> ctx1;

    for (int i = 0; i < num; ++i) {

        const float* din_batch = tensor_in.data() + i * ch_in * size_in_channel;
        float* dout_batch = tensor_out.mutable_data() + i * ch_out * size_out_channel;

        //t1.start(ctx1);
        //! transform input Bt * data * B
#pragma omp parallel for
        for (int j = 0; j < ch_in; ++j) {

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
                        if (start_row >= 0 && start_row < h_in){
                            for (int k = 0; k < 8; ++k) {
                                int start_col = w * 6 + k - pad_w;
                                if (start_col >= 0 && start_col < w_in) {
                                    data_in_tmp[j][k] = din_channel[start_row * w_in + start_col];
                                }
                            }
                        }
                    }
                    transform_input_f6x6(data_trans_channel, data_in_tmp[0]);
                    data_trans_channel += 64;
                }
            }
        }
        //! end of transform input

#if 1
        ////////////////////////////////////////////////////////////////////////////////
        //! dot mul
        //! transpose input, convert from ch_in * tile_h * tile_w * 64 to
        //! 64 * ch_in * tile_h * tile_w
        int stride_a = ch_out * ch_in;
        int stride_b = ch_in * size_tile;
        int stride_c = ch_out * size_tile;
        transpose(tmp_data2, tmp_data1, 64, stride_b);

        //t1.end(ctx1);
        //LOG(INFO) << "winograd conv transform input time: " << t1.get_average_ms();

        //t1.clear();
        //t1.start(ctx1);

        //! gemm
//#pragma omp parallel for
        for (int l = 0; l < 64; ++l) {
            const float* ptr_a = weights + l * stride_a;
            const float* ptr_b = tmp_data2 + l * stride_b;
            float* ptr_c = tmp_data1 + l * stride_c;
            gemmer(ptr_a, ch_in, ptr_b, size_tile, ptr_c, size_tile, 1.f, 0.f, false);
        }

        //! transpose output, convert from 64 * ch_out * tile_h * tile_w to
        //! ch_out * tile_h * tile_w * 64
        transpose(tmp_data2, tmp_data1, stride_c, 64);
        //! end of dot mul
#endif
        //t1.end(ctx1);
        //LOG(INFO) << "winograd conv dot mul time: " << t1.get_average_ms();


        //t1.clear();
        //t1.start(ctx1);
#if 1
        ///////////////////////////////////////////////////////////////////////////////
        //! transform output
#pragma omp parallel for
        for (int i = 0; i < ch_out; ++i) {

            float bias_value = flag_bias? bias[i] : 0.f;
            float* dout_tmp = tmp_data2 + i * size_trans_channel;
            float* dout_channel = dout_batch + i * size_out_channel;

            for (int h = 0; h < tile_h; ++h) {
                for (int w = 0; w < tile_w; ++w) {

                    float out_tmp[6][6];

                    transform_output_f6x6(out_tmp[0], dout_tmp, bias_value);
                    dout_tmp += 64;

                    for (int j = 0; j < 6; ++j) {
                        int end_row = h * 6 + j;
                        if (end_row < h_out) {
                            for (int k = 0; k < 6; ++k) {
                                int end_col = w * 6 + k;
                                if (end_col < w_out){
                                    dout_channel[end_row * w_out + end_col] = out_tmp[j][k];
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

/**
 * \brief transpose with arm neon optimization
 * @param data_out
 * @param data_in
 * @param w_in
 * @param h_in
 */
void transpose(float* data_out, const float* data_in, int w_in, int h_in) {

    int nw = w_in >> 2;
    int nh = h_in >> 2;
#pragma omp parallel for
    for (int i = 0; i < nh; i++) {
        for (int j = 0; j < nw; j++) {
            const float *ptr = data_in + i * 4 * w_in + j * 4;
            float *outptr = data_out + j * 4 * h_in + i * 4;

            const float *in0 = ptr;
            const float *in1 = in0 + w_in;
            const float *in2 = in1 + w_in;
            const float *in3 = in2 + w_in;

            float *out0 = outptr;
            float *out1 = out0 + h_in;
            float *out2 = out1 + h_in;
            float *out3 = out2 + h_in;
#ifdef __aarch64__
#else
            asm(    "vld1.32 {d0, d1}, [%[in0]]    \n"
                    "vld1.32 {d2, d3}, [%[in1]]    \n"
                    "vld1.32 {d4, d5}, [%[in2]]    \n"
                    "vld1.32 {d6, d7}, [%[in3]]    \n"
                    "vtrn.32 q0, q1                \n"
                    "vtrn.32 q2, q3                \n"
                    "vswp d1, d4                   \n"
                    "vswp d3, d6                   \n"
                    "vst1.32 {d0, d1}, [%[out0]]   \n"
                    "vst1.32 {d2, d3}, [%[out1]]   \n"
                    "vst1.32 {d4, d5}, [%[out2]]   \n"
                    "vst1.32 {d6, d7}, [%[out3]]   \n"
            :
            : [out0] "r" (out0), [out1] "r" (out1), [out2] "r" (out2), [out3] "r" (out3),
            [in0] "r" (in0), [in1] "r" (in1), [in2] "r" (in2), [in3] "r" (in3)
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"
            );
#endif //__aarch64__
        }
    }
    //! process remains
    for (int i = 0; i < nw * 4; i++) {
        for (int j = nh * 4; j < h_in; j++) {
            const float *ptr = data_in + j * w_in + i;
            float *outptr = data_out + i * h_in + j;
            *outptr = *ptr;
        }
    }
    for (int i = nw * 4; i < w_in; i++) {
        for (int j = 0; j < h_in; j++) {
            const float *ptr = data_in + w_in * j + i;
            float *outptr = data_out + i * h_in + j;
            *outptr = *ptr;
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
void winograd_transform_weights(float* dout, const float* din, int ch_out, \
    int ch_in, void* work_space) {
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
                    ptr_channel[j*8 + i] = tmpp[0] * coeff[i][0] + tmpp[1] * coeff[i][1] + \
                        tmpp[2] * coeff[i][2];
                }
            }
        }
    }
    transpose(dout, ptr_out, 64, ch_out * ch_in);
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
void transform_input_f6x6(float* dout, const float* din) {
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
 * \brief winograd conv, transform output, f63
 * out = AT * din * A
 * AT = 1      1       1       1        1        1        1   0
 *      0      1      -1       2       -2      1/2     -1/2   0
 *      0      1       1       4        4      1/4      1/4   0
 *      0      1      -1       8       -8      1/8     -1/8   0
 *      0      1       1      16       16     1/16     1/16   0
 *      0      1      -1      32      -32     1/32    -1/32   1
 * @param output
 * @param din
 * @param bias
 */
void transform_output_f6x6(float* output, const float* din, float bias) {
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

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE