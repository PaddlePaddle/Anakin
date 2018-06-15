#include "saber/funcs/impl/arm/impl/conv_arm_impl.h"
#ifdef USE_ARM_PLACE

#include "saber/funcs/impl/arm/impl/sgemv_arm.h"

namespace anakin{

namespace saber{
/**
 * \brief neon implementation to add bias
 * @param tensor
 * @param bias
 * @param channel
 * @param channel_size
 */
void fill_bias(float* tensor, const float* bias, int channel, int channel_size) {

    float* data = tensor;

    for (int j = 0; j < channel; ++j) {
       float32x4_t vdata = vdupq_n_f32(bias[j]);
       int i = 0;
       for (; i < channel_size - 3; i += 4) {
           vst1q_f32(data + i, vdata);
       }
       for (; i < channel_size; i++) {
           data[i] = bias[j];
       }
       data += channel_size;
   }
}
/**
 * \brief basic direct convolution function
 */
void conv_arm_basic(Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, Tensor<ARM, AK_FLOAT, NCHW>& tensor_in, \
    const float* weights, const float* bias, \
    int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
    int pad_w, int pad_h, bool flag_bias, bool flag_relu, Sgemm& gemmer, void* work_space) {

    int w_in = tensor_in.width();
    int h_in = tensor_in.height();
    int ch_in = tensor_in.channel();
    int num_in = tensor_in.num();

    int w_out = tensor_out.width();
    int h_out = tensor_out.height();
    int ch_out = tensor_out.channel();

    const int size_kernel = kernel_h * kernel_w;

    int kernel_ext_w = (kernel_w - 1) * dila_w + 1;
    int kernel_ext_h = (kernel_h - 1) * dila_h + 1;

    const int ch_out_g = ch_out / group;
    const int ch_in_g = ch_in / group;
    const int size_in_channel = w_in * h_in;
    const int size_in_batch = size_in_channel * ch_in;
    const int size_out_channel = w_out * h_out;
    const int size_out_batch = size_out_channel * ch_out;

    //printf("extend kernel size: %d, %d\n", kernel_ext_w, kernel_ext_h);
    const float *data_in = tensor_in.data();
    float *outptr = tensor_out.mutable_data();

    for (int b = 0; b < num_in; ++b) {
        float *outptr_batch = outptr + b * size_out_batch;
        const float* data_in_batch = data_in + b * size_in_batch;
#pragma omp parallel for collapse(2)
        for (int g = 0; g < group; ++g) {
            for (int c = 0; c < ch_out_g; ++c) {
                const float *inptr_group = data_in_batch + g * ch_in_g * size_in_channel;
                float *outptr_ch = outptr_batch + (g * ch_out_g + c) * size_out_channel;
                const float *weight_ch = weights + (g * ch_out_g + c) * ch_in_g * size_kernel;

                float bias_value = flag_bias? bias[g * ch_out_g + c] : 0.f;
                fill_bias(outptr_ch, &bias_value, 1, w_out * h_out);

                for (int i = 0; i < h_out; ++i) {
                    for (int j = 0; j < w_out; ++j) {

                        const float *weight_ch_in = weight_ch;

                        int hstart = i * stride_h - pad_h;
                        int wstart = j * stride_w - pad_w;
                        int hend = std::min(hstart + kernel_ext_h, h_in);
                        int wend = std::min(wstart + kernel_ext_w, w_in);
                        hstart = std::max(hstart, 0);
                        wstart = std::max(wstart, 0);

                        int khstart = hend < kernel_ext_h? (kernel_ext_h - hend) / dila_h : 0;
                        int kwstart = wend < kernel_ext_w? (kernel_ext_w - wend) / dila_w : 0;

                        //printf("channel: %d, index: %d, %d, %d, %d, %d, %d\n", c, hstart, wstart, hend, wend, khstart, kwstart);
                        const float* inptr_ch = inptr_group + hstart * w_in + wstart;

                        for (int k = 0; k < ch_in_g; ++k) {
                            const float* inptr_kernel = inptr_ch;
                            int khidx = khstart;
                            for (int idxh = hstart; idxh < hend; idxh += dila_h, khidx++) {
                                const float* inptr_kernel_w = inptr_kernel;
                                int kwidx = kwstart;
                                for (int idxw = wstart; idxw < wend; idxw += dila_w, kwidx++) {
                                    outptr_ch[j] += weight_ch_in[khidx * kernel_w + kwidx] * inptr_kernel_w[0];
                                    inptr_kernel_w += dila_w;
                                }
                                inptr_kernel += dila_h * w_in;
                            }
                            inptr_ch += size_in_channel;
                            weight_ch_in += size_kernel;
                        }
                        if (flag_relu) {
                            outptr_ch[j] = outptr_ch[j] > 0? outptr_ch[j] : 0.f;
                        }
                    }
                    outptr_ch += w_out;
                }
            }
        }
    }
}

/**
 * \brief inline funcs used in im2col
 * @param a
 * @param b
 * @return
 */
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

/**
 * \brief normal im2col function for gemm conv
 * @tparam dtype
 * @param data_im
 * @param channels
 * @param height
 * @param width
 * @param kernel_size
 * @param pad
 * @param stride
 * @param data_col
 */
template <typename dtype>
void im2col(const dtype *data_im, const int channels, const int height, \
                   const int width, const int kernel_h, const int kernel_w, \
                   const int pad_h, const int pad_w, const int stride_h, const int stride_w, \
                   dtype *data_col) {
    const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -pad_w + kernel_col;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}

/**
 * \brief specify im2col for kernel size 1x1, and stride = 2
 * @param data_im
 * @param channels
 * @param height
 * @param width
 * @param data_col
 */
void im2col1x1s2(const float* data_im, const int channels, const int height, \
        const int width, float* data_col){
    float32x4x2_t vdin;
    int size = height * width;
    int width_out = (width - 1) / 2 + 1;
    int height_out = (height - 1) / 2 + 1;
    int size_out = width_out * height_out;
//#pragma omp parallel for
    for (int i = 0; i < channels; ++i) {
        float* dout = data_col + i * size_out;
        const float* din = data_im + i * size;
        for (int j = 0; j < height - 1; j += 2) {

            const float* dinh = din + j * width;
            int k = 0;
            for (; k < width - 7; k += 8) {
                vdin = vld2q_f32(dinh + k);
                vst1q_f32(dout, vdin.val[0]);
                dout += 4;
            }
            for (; k < width; k += 2) {
                *(dout++) = *(din + k);
            }
        }
    }
}

/**
 * \brief convolution function for kernel size 1x1, stride size 1, gemm implementation
 */
void conv1x1s1_gemm(Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, Tensor<ARM, AK_FLOAT, NCHW>& tensor_in, \
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

    int channel_size_out = w_out * h_out;
    int channel_size_in = w_in * h_in;

    const int m = ch_out;
    const int n = h_out * w_out;
    const int k = ch_in;

    //! use gemv when the output channel size = 1
    if (n == 1) {
        for (int b = 0; b < num; ++b) {
            float *data_out_batch = tensor_out.mutable_data() + b * ch_out * channel_size_out;
            const float* dB = tensor_in.data() + b * ch_in * channel_size_in;
            if (flag_bias){
                if (flag_relu) {
                    sgemv_bias_relu(false, m, k, weights, dB, data_out_batch, bias);
                } else {
                    sgemv_bias(false, m, k, weights, dB, data_out_batch, bias);
                }

            } else {
                if (flag_relu) {
                    sgemv_relu(false, m, k, weights, dB, data_out_batch);
                } else {
                    sgemv(false, m, k, weights, dB, data_out_batch);
                }
            }
        }

    } else {
        for (int b = 0; b < num; ++b) {
            // dC
            float *data_out_batch = tensor_out.mutable_data(b * ch_out * channel_size_out);
            const float* dB = tensor_in.data(b * ch_in * channel_size_in);
            float beta = 0.f;
            if (flag_bias){
                fill_bias(data_out_batch, bias, ch_out, w_out * h_out);
                beta = 1.f;
            }
            gemmer(weights, k, dB, n, data_out_batch, n, 1.f, beta, flag_relu);
        }
    }
}

/**
 * \brief convolution function for kernel size 3x3, stride size 2, gemm implementation
 */
void conv_im2col_gemm(Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, Tensor<ARM, AK_FLOAT, NCHW>& tensor_in, \
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

    const int m = ch_out;
    const int n = h_out * w_out;
    const int k = ch_in * kernel_h * kernel_w;

    int channel_size_out = w_out * h_out;
    int channel_size_in = w_in * h_in;
    for (int b = 0; b < num; ++b) {
        // dC
        float *data_out_batch = tensor_out.mutable_data(b * ch_out * channel_size_out);
        float *data_in_batch = tensor_in.mutable_data(b * ch_in * channel_size_in);

        float* dB = (float*)work_space;
        if (kernel_w == 1 && pad_w == 0) {
            im2col1x1s2(data_in_batch, ch_in, h_in, w_in, dB);
        } else {
            im2col(data_in_batch, ch_in, h_in, w_in, kernel_h, kernel_w, \
            pad_h, pad_w, stride_h, stride_w, dB);
        }

        float beta = 0.f;
        if (flag_bias) {
            fill_bias(data_out_batch, bias, ch_out, w_out * h_out);
            beta = 1.f;
        }

        gemmer(weights, k, dB, n, data_out_batch, n, 1.f, beta, flag_relu);
    }
}

void conv_3x3s1_direct(Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, Tensor<ARM, AK_FLOAT, NCHW>& tensor_in, \
    const float* weights, const float* bias, \
    int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
    int pad_w, int pad_h, bool flag_bias, bool flag_relu, Sgemm& gemmer, void* work_space) {
    //! 3x3s1 convolution, implemented by direct algorithm
    //! pad is done implicit
    const float zero[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    //! for 4x6 convolution window
    const int right_pad_idx[4] = {3, 2, 1, 0};

    int w_in = tensor_in.width();
    int h_in = tensor_in.height();
    int ch_in = tensor_in.channel();
    int num = tensor_in.num();

    int w_out = tensor_out.width();
    int h_out = tensor_out.height();
    int ch_out = tensor_out.channel();

    const float* din = tensor_in.data();
    float* dout = tensor_out.mutable_data();

    int size_in_channel = w_in * h_in;
    int size_out_channel = w_out * h_out;
    int w_stride = ch_in * 9;

    int tile_w = (w_in + 3) >> 2;
    int tile_h = (h_in + 1) >> 1;
    int w_in_twice = w_in << 1;
    int cnt_col = tile_w - 2;

    int size_pad_right = 1 + (tile_w << 2) - w_in;
    int size_pad_bottom = 1 + (tile_h << 1) - h_in;

    int cremain = ch_out - (ch_out >> 1) << 1;

    uint32x4_t vmask_rp = vcgeq_s32(vld1q_s32(right_pad_idx), vdupq_n_s32(size_pad_right));
    unsigned int pmask_rp[4];
    vst1q_u32(pmask_rp, vmask_rp);
    int right_pad_sub = (size_pad_right - 1) * sizeof(float);

    for (int n = 0; n < num; ++n) {
        const float *din_batch = din + n * ch_in * size_in_channel;
        float *dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for
        for (int c = 0; c < ch_out - 1; c += 2) {

            float* dout_c0 = dout_batch + c * size_out_channel;
            float* dout_c1 = dout_c0 + size_out_channel;

            if (flag_bias) {
                fill_bias(dout_c0, &bias[c], 1, size_out_channel);
                fill_bias(dout_c1, &bias[c + 1], 1, size_out_channel);
            } else {
                fill_bias(dout_c0, zero, 1, size_out_channel);
                fill_bias(dout_c1, zero, 1, size_out_channel);
            }

            //float* dout_c2 = dout_c1 + size_out_channel;
            //float* dout_c3 = dout_c2 + size_out_channel;

            const float* wc0 = weights + c * w_stride;
            const float* wc1 = wc0 + w_stride;

            //const float* wc2 = wc0 + w_stride;
            //const float* wc3 = wc0 + w_stride;

            for (int i = 0; i < ch_in; ++i) {

                const float *din_channel = din_batch + i * size_in_channel;

                const float* wcin0 = wc0 + i * 9;
                const float* wcin1 = wc1 + i * 9;
                float32x4_t wr00 = vld1q_f32(wcin0);
                float32x4_t wr01 = vld1q_f32(wcin0 + 3);
                float32x4_t wr02 = vld1q_f32(wcin0 + 6);

                float32x4_t wr10 = vld1q_f32(wcin1);
                float32x4_t wr11 = vld1q_f32(wcin1 + 3);
                float32x4_t wr12 = vld1q_f32(wcin1 + 6);

                float *doutc0r0 = dout_c0;
                float *doutc0r1 = doutc0r0 + w_out;

                float *doutc1r0 = dout_c1;
                float *doutc1r1 = doutc1r0 + w_out;

                const float *dr0 = din_channel;
                const float *dr1 = dr0 + w_in;
                const float *dr2 = dr1 + w_in;
                const float *dr3 = dr2 + w_in;

                const float *din0_ptr = dr0;
                const float *din1_ptr = dr1;
                const float *din2_ptr = dr2;
                const float *din3_ptr = dr3;

                float* ptr_zero = const_cast<float*>(zero);

                //! deal with top pad
                int h = 0;
                {
                    //! process
                    if (1) {
#ifdef __aarch64__
                        // todo
#else
                        int cnt = cnt_col;

                        float tmp1[4];
                        float* ptr1 = tmp1;
                        float tmp2[4];
                        float* ptr2 = tmp2;

                        asm volatile(
                        //! process left pad
                        "pld [%[doutc0r0], #192]                @ preload data\n"
                                "pld [%[doutc0r1], #192]                @ preload data\n"
                                "vld1.32  {d26-d27}, [%[doutc0r0]]      @ load dout0r0\n"
                                "pld [%[din0_ptr], #192]                @ preload data\n"
                                "pld [%[din1_ptr], #192]                @ preload data\n"
                                "pld [%[din2_ptr], #192]                @ preload data\n"
                                "vld1.32  {d28-d29}, [%[doutc0r1]]      @ load dout0r1\n"
                                "pld [%[doutc1r0], #192]                @ preload data\n"
                                "pld [%[doutc1r1], #192]                @ preload data\n"

                                //! 1st row
                                "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r1\n"
                                "pld [%[din0_ptr], #192]                @ preload data\n"
                                "vld1.32  {d16-d17}, [%[doutc1r0]]      @ load dout1r0\n"
                                "vld1.32  {d18-d19}, [%[doutc1r1]]      @ load dout1r1\n"

                                "vmla.f32 q13, q10, %e[wr01][1]         @ mul weight0 11, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr00][1]         @ mul weight0 01, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr11][1]          @ mul weight1 11, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr10][1]          @ mul weight1 01, out1r1\n"

                                "vext.32  q12, q10, q11, #1             @ shift left r1\n"
                                "vmla.f32 q13, q12, %f[wr01][0]         @ mul weight0 12, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr00][0]         @ mul weight0 02, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr11][0]          @ mul weight1 12, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr10][0]          @ mul weight1 02, out1r1\n"

                                "vmov.u32 q15, #0                       @ dump zero\n"
                                "vext.32  q12, q15, q10, #3             @ shift right r1\n"
                                "vmla.f32 q13, q12, %e[wr01][0]         @ mul weight0 10, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr00][0]         @ mul weight0 00, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr11][0]          @ mul weight1 10, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr10][0]          @ mul weight1 00, out1r1\n"

                                //! 2nd row
                                "vld1.32  {d20-d22}, [%[din1_ptr]]!     @ load din r2\n"
                                "pld [%[din1_ptr], #192]                @ preload data\n"
                                "vmla.f32 q13, q10, %e[wr02][1]         @ mul weight0 21, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr01][1]         @ mul weight0 11, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr12][1]          @ mul weight1 21, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr11][1]          @ mul weight1 11, out1r1\n"

                                "vext.32  q12, q10, q11, #1             @ shift left r2\n"
                                "vmla.f32 q13, q12, %f[wr02][0]         @ mul weight0 22, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr01][0]         @ mul weight0 12, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr12][0]          @ mul weight1 22, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr11][0]          @ mul weight1 12, out1r1\n"

                                "vext.32  q12, q15, q10, #3             @ shift right r2\n"
                                "vmla.f32 q13, q12, %e[wr02][0]         @ mul weight0 20, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr01][0]         @ mul weight0 10, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr12][0]          @ mul weight1 20, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr11][0]          @ mul weight1 10, out1r1\n"

                                //! 3rd row
                                "vld1.32  {d20-d22}, [%[din2_ptr]]!     @ load din r3\n"
                                "pld [%[din2_ptr], #192]                @ preload data\n"
                                "vmla.f32 q14, q10, %e[wr02][1]         @ mul weight0 21, out0r1\n"
                                "vmla.f32 q9, q10, %e[wr12][1]          @ mul weight1 21, out1r1\n"

                                "vext.32  q12, q10, q11, #1             @ shift left r3\n"
                                "vmla.f32 q14, q12, %f[wr02][0]         @ mul weight0 22, out0r1\n"
                                "vmla.f32 q9, q12, %f[wr12][0]          @ mul weight1 22, out1r1\n"

                                "vext.32  q12, q15, q10, #3               @ shift right r3\n"
                                "vmla.f32 q14, q12, %e[wr02][0]            @ mul weight0 20, out0r1\n"
                                "vmla.f32 q9, q12, %e[wr12][0]            @ mul weight1 20, out1r1\n"

                                "vst1.32  {d26-d27}, [%[doutc0r0]]!    @ store result, add pointer\n"
                                "pld [%[doutc0r0], #192]                @ preload data\n"
                                "vst1.32  {d28-d29}, [%[doutc0r1]]!    @ store result, add pointer\n"
                                "pld [%[doutc0r1], #192]                @ preload data\n"

                                "vst1.32  {d16-d17}, [%[doutc1r0]]!    @ store result, add pointer\n"
                                "pld [%[doutc1r0], #192]                @ preload data\n"
                                "vst1.32  {d18-d19}, [%[doutc1r1]]!    @ store result, add pointer\n"
                                "pld [%[doutc1r1], #192]                @ preload data\n"

                                "sub %[din0_ptr], #12                   @ 1pad + 2 float data overlap\n"
                                "sub %[din1_ptr], #12                   @ 1pad + 2 float data overlap\n"
                                "sub %[din2_ptr], #12                   @ 1pad + 2 float data overlap\n"

                                //! process mid cols
                                "cmp %[cnt], #1                         @ check whether has mid cols\n"
                                "blt  start_top_right                   @ jump to main loop start point\n"
                                "start_top_mid:                         @ main loop start point\n"

                                "vld1.32  {d26-d27}, [%[doutc0r0]]       @ load dout0r0\n"
                                "vld1.32  {d28-d29}, [%[doutc0r1]]       @ load dout0r1\n"

                                //! 1st row
                                "vld1.32  {d20-d22}, [%[din0_ptr]]!      @ load din r1\n"
                                "pld [%[din0_ptr], #192]                @ preload data\n"
                                "vmla.f32 q13, q10, %e[wr01][0]           @ mul weight0 10, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr00][0]           @ mul weight0 00, out0r1\n"
                                "vld1.32  {d16-d17}, [%[doutc1r0]]        @ load dout1r0\n"
                                "vld1.32  {d18-d19}, [%[doutc1r1]]        @ load dout1r1\n"
                                "vmla.f32 q8, q10, %e[wr11][0]           @ mul weight1 10, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr10][0]           @ mul weight1 00, out1r1\n"

                                "vext.32  q12, q10, q11, #1                 @ shift left r1\n"
                                "vmla.f32 q13, q12, %e[wr01][1]              @ mul weight0 11, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr00][1]              @ mul weight0 01, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr11][1]            @ mul weight1 11, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr10][1]            @ mul weight1 01, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r1\n"
                                "vmla.f32 q13, q12, %f[wr01][0]            @ mul weight0 12, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr00][0]            @ mul weight0 02, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr11][0]            @ mul weight1 12, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr10][0]            @ mul weight1 02, out1r1\n"

                                //! 2nd row
                                "vld1.32  {d20-d22}, [%[din1_ptr]]!       @ load din r2\n"
                                "pld [%[din1_ptr], #192]                @ preload data\n"
                                "vmla.f32 q13, q10, %e[wr02][0]           @ mul weight0 20, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr01][0]           @ mul weight0 10, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr12][0]           @ mul weight1 20, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr11][0]           @ mul weight1 10, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r2\n"
                                "vmla.f32 q13, q12, %e[wr02][1]            @ mul weight0 21, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr01][1]            @ mul weight0 11, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr12][1]            @ mul weight1 21, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr11][1]            @ mul weight1 11, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r2\n"
                                "vmla.f32 q13, q12, %f[wr02][0]            @ mul weight0 22, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr01][0]            @ mul weight0 12, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr12][0]            @ mul weight1 22, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr11][0]            @ mul weight1 12, out1r1\n"

                                //! 3rd row
                                "vld1.32  {d20-d22}, [%[din2_ptr]]!      @ load din r3\n"
                                "pld [%[din2_ptr], #192]                @ preload data\n"
                                "vmla.f32 q14, q10, %e[wr02][0]           @ mul weight0 20, out0r1\n"
                                "vmla.f32 q9, q10, %e[wr12][0]           @ mul weight1 20, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r3\n"
                                "vmla.f32 q14, q12, %e[wr02][1]            @ mul weight0 21, out0r1\n"
                                "vmla.f32 q9, q12, %e[wr12][1]            @ mul weight1 21, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r3\n"
                                "vmla.f32 q14, q12, %f[wr02][0]            @ mul weight0 22, out0r1\n"
                                "vmla.f32 q9, q12, %f[wr12][0]            @ mul weight1 22, out1r1\n"

                                "vst1.32  {d26-d27}, [%[doutc0r0]]!    @ store result, add pointer\n"
                                "pld [%[doutc0r0], #192]                @ preload data\n"
                                "vst1.32  {d28-d29}, [%[doutc0r1]]!    @ store result, add pointer\n"
                                "pld [%[doutc0r1], #192]                @ preload data\n"

                                "vst1.32  {d16-d17}, [%[doutc1r0]]!    @ store result, add pointer\n"
                                "pld [%[doutc1r0], #192]                @ preload data\n"
                                "vst1.32  {d18-d19}, [%[doutc1r1]]!    @ store result, add pointer\n"
                                "pld [%[doutc1r1], #192]                @ preload data\n"

                                "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                                "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"
                                "sub %[din2_ptr], #8                    @ 2 float data overlap with previous data\n"
                                "subs %[cnt], #1                        @ loop count minus 1\n"
                                "bne    start_top_mid                   @ jump to main loop start point\n"

                                //! process right pad
                                "start_top_right:                       @ right pad entry\n"
                                "vld1.32  {d26-d27}, [%[doutc0r0]]       @ load dout0r0\n"
                                "vld1.32  {d28-d29}, [%[doutc0r1]]       @ load dout0r1\n"

                                //! 1st row
                                "vld1.32  {d20-d22}, [%[din0_ptr]]!      @ load din r1\n"
                                "vbif d21, d31, %e[vmask_rp]             @ bit select, deal with right pad\n"
                                "vbif d22, d31, %f[vmask_rp]             @ bit select, deal with right pad\n"
                                "vmla.f32 q13, q10, %e[wr01][0]           @ mul weight0 10, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr00][0]           @ mul weight0 00, out0r1\n"
                                "vld1.32  {d16-d17}, [%[doutc1r0]]        @ load dout1r0\n"
                                "vld1.32  {d18-d19}, [%[doutc1r1]]        @ load dout1r1\n"
                                "vmla.f32 q8, q10, %e[wr11][0]           @ mul weight1 10, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr10][0]           @ mul weight1 00, out1r1\n"

                                "vext.32  q12, q10, q11, #1                 @ shift left r1\n"
                                "vmla.f32 q13, q12, %e[wr01][1]              @ mul weight0 11, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr00][1]              @ mul weight0 01, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr11][1]            @ mul weight1 11, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr10][1]            @ mul weight1 01, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r1\n"
                                "vmla.f32 q13, q12, %f[wr01][0]            @ mul weight0 12, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr00][0]            @ mul weight0 02, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr11][0]            @ mul weight1 12, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr10][0]            @ mul weight1 02, out1r1\n"

                                //! 2nd row
                                "vld1.32  {d20-d22}, [%[din1_ptr]]!       @ load din r2\n"
                                "vbif d21, d31, %e[vmask_rp]             @ bit select, deal with right pad\n"
                                "vbif d22, d31, %f[vmask_rp]             @ bit select, deal with right pad\n"
                                "vmla.f32 q13, q10, %e[wr02][0]           @ mul weight0 20, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr01][0]           @ mul weight0 10, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr12][0]           @ mul weight1 20, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr11][0]           @ mul weight1 10, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r2\n"
                                "vmla.f32 q13, q12, %e[wr02][1]            @ mul weight0 21, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr01][1]            @ mul weight0 11, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr12][1]            @ mul weight1 21, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr11][1]            @ mul weight1 11, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r2\n"
                                "vmla.f32 q13, q12, %f[wr02][0]            @ mul weight0 22, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr01][0]            @ mul weight0 12, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr12][0]            @ mul weight1 22, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr11][0]            @ mul weight1 12, out1r1\n"

                                //! 3rd row
                                "vld1.32  {d20-d22}, [%[din2_ptr]]!      @ load din r3\n"
                                "vbif d21, d31, %e[vmask_rp]             @ bit select, deal with right pad\n"
                                "vbif d22, d31, %f[vmask_rp]             @ bit select, deal with right pad\n"
                                "vmla.f32 q14, q10, %e[wr02][0]           @ mul weight0 20, out0r1\n"
                                "vmla.f32 q9, q10, %e[wr12][0]           @ mul weight1 20, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r3\n"
                                "vmla.f32 q14, q12, %e[wr02][1]            @ mul weight0 21, out0r1\n"
                                "vmla.f32 q9, q12, %e[wr12][1]            @ mul weight1 21, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r3\n"
                                "vmla.f32 q14, q12, %f[wr02][0]            @ mul weight0 22, out0r1\n"
                                "vmla.f32 q9, q12, %f[wr12][0]            @ mul weight1 22, out1r1\n"

                                "vld1.32  {d20-d21}, [%[doutc0r0]]       @ load dout0r0\n"
                                "vld1.32  {d22-d23}, [%[doutc0r1]]       @ load dout0r1\n"

                                "vmvn.32  q12, q15                      @ \n"
                                "vext.32  q15, q12, %q[vmask_rp], #3    @ shift mask right 1\n"
                                "vbif q13, q10, q15                      @ bit select\n"
                                "vbif q14, q11, q15                      @ bit select\n"

                                "vst1.32  {d26-d27}, [%[doutc0r0]]!    @ store result, add pointer\n"
                                "vst1.32  {d28-d29}, [%[doutc0r1]]!    @ store result, add pointer\n"

                                "vld1.32  {d20-d21}, [%[doutc1r0]]       @ load dout1r0\n"
                                "vld1.32  {d22-d23}, [%[doutc1r1]]       @ load dout1r1\n"

                                "vbif q8, q10, q15                      @ bit select\n"
                                "vbif q9, q11, q15                      @ bit select\n"

                                "vst1.32  {d16-d17}, [%[doutc1r0]]!    @ store result, add pointer\n"
                                "vst1.32  {d18-d19}, [%[doutc1r1]]!    @ store result, add pointer\n"

                                "sub %[doutc0r0], %[doutc0r0], %[right_pad_sub] @ sub \n"
                                "sub %[doutc0r1], %[doutc0r1], %[right_pad_sub] @ sub \n"
                                "sub %[doutc1r0], %[doutc1r0], %[right_pad_sub] @ sub \n"
                                "sub %[doutc1r1], %[doutc1r1], %[right_pad_sub] @ sub \n"

                        :[doutc0r0] "+r"(doutc0r0), [doutc0r1] "+r"(doutc0r1), \
                            [doutc1r0] "+r" (doutc1r0), [doutc1r1] "+r" (doutc1r1),\
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [cnt] "+r"(cnt)
                        :[wr00] "w"(wr00), [wr01] "w"(wr01), [wr02] "w"(wr02), \
                            [wr10] "w"(wr10), [wr11] "w"(wr11), [wr12] "w"(wr12), \
                            [vmask_rp] "w" (vmask_rp), [right_pad_sub] "r" (right_pad_sub)
                        :"q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                        );
#endif //__aarch64__
                    }
                    //! after process, increase pointer
                    doutc0r0 += w_out;
                    doutc0r1 = doutc0r0 + w_out;
                    doutc1r0 += w_out;
                    doutc1r1 = doutc1r0 + w_out;

                    dr0 = dr1;
                    dr1 = dr2;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                } //! end of process top row


                //! process mid row
                for (h = 1; h < tile_h - 1; h++) {
                    din0_ptr = dr0;
                    din1_ptr = dr1;
                    din2_ptr = dr2;
                    din3_ptr = dr3;

                    {
#ifdef __aarch64__
                        // todo
#else
                        int cnt = cnt_col;
                        asm volatile (
                        //! process left pad
                        "pld [%[doutc0r0], #192]                @ preload data\n"
                                "pld [%[doutc0r1], #192]                @ preload data\n"
                                "vld1.32  {d26-d27}, [%[doutc0r0]]      @ load dout0r0\n"
                                "pld [%[din0_ptr], #192]                @ preload data\n"
                                "pld [%[din1_ptr], #192]                @ preload data\n"
                                "pld [%[din2_ptr], #192]                @ preload data\n"
                                "pld [%[din3_ptr], #192]                @ preload data\n"
                                "vld1.32  {d28-d29}, [%[doutc0r1]]      @ load dout0r1\n"
                                "pld [%[doutc1r0], #192]                @ preload data\n"
                                "pld [%[doutc1r1], #192]                @ preload data\n"

                                //! 1st row
                                "vld1.32  {d20-d22}, [%[din0_ptr]]!     @ load din r0\n"
                                "pld [%[din0_ptr], #192]                @ preload data\n"
                                "vmla.f32 q13, q10, %e[wr00][1]         @ mul weight0 01, out0r0\n"
                                "vld1.32  {d16-d17}, [%[doutc1r0]]      @ load dout1r0\n"
                                "vld1.32  {d18-d19}, [%[doutc1r1]]      @ load dout1r1\n"
                                "vmla.f32 q8, q10, %e[wr10][1]          @ mul weight1 01, out1r0\n"

                                "vext.32  q12, q10, q11, #1                 @ shift left r0\n"
                                "vmla.f32 q13, q12, %f[wr00][0]              @ mul weight0 02, out0r0\n"
                                "vmla.f32 q8, q12, %f[wr10][0]              @ mul weight1 02, out1r0\n"

                                "vmov.u32 q15, #0                         @ dump zero\n"
                                "vext.32  q12, q15, q10, #3               @ shift right r1\n"
                                "vmla.f32 q13, q12, %e[wr00][0]            @ mul weight0 00, out0r0\n"
                                "vmla.f32 q8, q12, %e[wr10][0]            @ mul weight1 00, out1r0\n"

                                //! 2nd row
                                "vld1.32  {d20-d22}, [%[din1_ptr]]!       @ load din r1\n"
                                "pld [%[din1_ptr], #192]                @ preload data\n"
                                "vmla.f32 q13, q10, %e[wr01][1]           @ mul weight0 11, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr00][1]           @ mul weight0 01, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr11][1]           @ mul weight1 11, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr10][1]           @ mul weight1 01, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r1\n"
                                "vmla.f32 q13, q12, %f[wr01][0]            @ mul weight0 12, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr00][0]            @ mul weight0 02, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr11][0]            @ mul weight1 12, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr10][0]            @ mul weight1 02, out1r1\n"

                                "vext.32  q12, q15, q10, #3               @ shift right r1\n"
                                "vmla.f32 q13, q12, %e[wr01][0]            @ mul weight0 10, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr00][0]            @ mul weight0 00, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr11][0]            @ mul weight1 10, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr10][0]            @ mul weight1 00, out1r1\n"

                                //! 3rd row
                                "vld1.32  {d20-d22}, [%[din2_ptr]]!       @ load din r2\n"
                                "pld [%[din2_ptr], #192]                @ preload data\n"
                                "vmla.f32 q13, q10, %e[wr02][1]           @ mul weight0 21, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr01][1]           @ mul weight0 11, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr12][1]           @ mul weight1 21, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr11][1]           @ mul weight1 11, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r2\n"
                                "vmla.f32 q13, q12, %f[wr02][0]            @ mul weight0 22, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr01][0]            @ mul weight0 12, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr12][0]            @ mul weight1 22, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr11][0]            @ mul weight1 12, out1r1\n"

                                "vext.32  q12, q15, q10, #3               @ shift right r2\n"
                                "vmla.f32 q13, q12, %e[wr02][0]            @ mul weight0 20, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr01][0]            @ mul weight0 10, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr12][0]            @ mul weight1 20, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr11][0]            @ mul weight1 10, out1r1\n"

                                //! 4rd row
                                "vld1.32  {d20-d22}, [%[din3_ptr]]!      @ load din r3\n"
                                "pld [%[din3_ptr], #192]                @ preload data\n"
                                "vmla.f32 q14, q10, %e[wr02][1]           @ mul weight0 21, out0r1\n"
                                "vmla.f32 q9, q10, %e[wr12][1]           @ mul weight1 21, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r3\n"
                                "vmla.f32 q14, q12, %f[wr02][0]            @ mul weight0 22, out0r1\n"
                                "vmla.f32 q9, q12, %f[wr12][0]            @ mul weight1 22, out1r1\n"

                                "vext.32  q12, q15, q10, #3               @ shift right r3\n"
                                "vmla.f32 q14, q12, %e[wr02][0]            @ mul weight0 20, out0r1\n"
                                "vmla.f32 q9, q12, %e[wr12][0]            @ mul weight1 20, out1r1\n"

                                "vst1.32  {d26-d27}, [%[doutc0r0]]!    @ store result, add pointer\n"
                                "pld [%[doutc0r0], #192]                @ preload data\n"
                                "vst1.32  {d28-d29}, [%[doutc0r1]]!    @ store result, add pointer\n"
                                "pld [%[doutc0r1], #192]                @ preload data\n"

                                "vst1.32  {d16-d17}, [%[doutc1r0]]!    @ store result, add pointer\n"
                                "pld [%[doutc1r0], #192]                @ preload data\n"
                                "vst1.32  {d18-d19}, [%[doutc1r1]]!    @ store result, add pointer\n"
                                "pld [%[doutc1r1], #192]                @ preload data\n"

                                "sub %[din0_ptr], #12                   @ 1pad + 2 float data overlap\n"
                                "sub %[din1_ptr], #12                   @ 1pad + 2 float data overlap\n"
                                "sub %[din2_ptr], #12                   @ 1pad + 2 float data overlap\n"
                                "sub %[din3_ptr], #12                   @ 1pad + 2 float data overlap\n"

                                //! process mid cols
                                "cmp %[cnt], #1                         @ check whether has mid cols\n"
                                "blt  start_mid_right                   @ jump to main loop start point\n"
                                "start_mid_mid:                         @ main loop start point\n"

                                "vld1.32  {d26-d27}, [%[doutc0r0]]       @ load dout0r0\n"
                                "vld1.32  {d28-d29}, [%[doutc0r1]]       @ load dout0r1\n"

                                //! 1st row
                                "vld1.32  {d20-d22}, [%[din0_ptr]]!      @ load din r0\n"
                                "pld [%[din0_ptr], #192]                @ preload data\n"
                                "vmla.f32 q13, q10, %e[wr00][0]           @ mul weight0 00, out0r0\n"
                                "vld1.32  {d16-d17}, [%[doutc1r0]]        @ load dout1r0\n"
                                "vld1.32  {d18-d19}, [%[doutc1r1]]        @ load dout1r1\n"
                                "vmla.f32 q8, q10, %e[wr10][0]           @ mul weight1 00, out1r0\n"

                                "vext.32  q12, q10, q11, #1                 @ shift left r0\n"
                                "vmla.f32 q13, q12, %e[wr00][1]              @ mul weight0 01, out0r0\n"
                                "vmla.f32 q8, q12, %e[wr10][1]              @ mul weight1 01, out1r0\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r0\n"
                                "vmla.f32 q13, q12, %f[wr00][0]            @ mul weight0 02, out0r0\n"
                                "vmla.f32 q8, q12, %f[wr10][0]            @ mul weight1 02, out1r0\n"

                                //! 2nd row
                                "vld1.32  {d20-d22}, [%[din1_ptr]]!       @ load din r1\n"
                                "pld [%[din1_ptr], #192]                @ preload data\n"
                                "vmla.f32 q13, q10, %e[wr01][0]           @ mul weight0 10, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr00][0]           @ mul weight0 00, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr11][0]           @ mul weight1 10, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr10][0]           @ mul weight1 00, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r1\n"
                                "vmla.f32 q13, q12, %e[wr01][1]            @ mul weight0 11, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr00][1]            @ mul weight0 01, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr11][1]            @ mul weight1 11, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr10][1]            @ mul weight1 01, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r1\n"
                                "vmla.f32 q13, q12, %f[wr01][0]            @ mul weight0 12, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr00][0]            @ mul weight0 02, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr11][0]            @ mul weight1 12, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr10][0]            @ mul weight1 02, out1r1\n"

                                //! 3rd row
                                "vld1.32  {d20-d22}, [%[din2_ptr]]!       @ load din r2\n"
                                "pld [%[din2_ptr], #192]                @ preload data\n"
                                "vmla.f32 q13, q10, %e[wr02][0]           @ mul weight0 20, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr01][0]           @ mul weight0 10, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr12][0]           @ mul weight1 20, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr11][0]           @ mul weight1 10, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r2\n"
                                "vmla.f32 q13, q12, %e[wr02][1]            @ mul weight0 21, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr01][1]            @ mul weight0 11, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr12][1]            @ mul weight1 21, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr11][1]            @ mul weight1 11, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift right r2\n"
                                "vmla.f32 q13, q12, %f[wr02][0]            @ mul weight0 22, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr01][0]            @ mul weight0 12, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr12][0]            @ mul weight1 22, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr11][0]            @ mul weight1 12, out1r1\n"

                                //! 4rd row
                                "vld1.32  {d20-d22}, [%[din3_ptr]]!      @ load din r3\n"
                                "pld [%[din3_ptr], #192]                @ preload data\n"
                                "vmla.f32 q14, q10, %e[wr02][0]           @ mul weight0 20, out0r1\n"
                                "vmla.f32 q9, q10, %e[wr12][0]           @ mul weight1 20, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r3\n"
                                "vmla.f32 q14, q12, %e[wr02][1]            @ mul weight0 21, out0r1\n"
                                "vmla.f32 q9, q12, %e[wr12][1]            @ mul weight1 21, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r3\n"
                                "vmla.f32 q14, q12, %f[wr02][0]            @ mul weight0 22, out0r1\n"
                                "vmla.f32 q9, q12, %f[wr12][0]            @ mul weight1 22, out1r1\n"

                                "vst1.32  {d26-d27}, [%[doutc0r0]]!    @ store result, add pointer\n"
                                "pld [%[doutc0r0], #192]                @ preload data\n"
                                "vst1.32  {d28-d29}, [%[doutc0r1]]!    @ store result, add pointer\n"
                                "pld [%[doutc0r1], #192]                @ preload data\n"

                                "vst1.32  {d16-d17}, [%[doutc1r0]]!    @ store result, add pointer\n"
                                "pld [%[doutc1r0], #192]                @ preload data\n"
                                "vst1.32  {d18-d19}, [%[doutc1r1]]!    @ store result, add pointer\n"
                                "pld [%[doutc1r1], #192]                @ preload data\n"

                                "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                                "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"
                                "sub %[din2_ptr], #8                    @ 2 float data overlap with previous data\n"
                                "sub %[din3_ptr], #8                    @ 2 float data overlap with previous data\n"
                                "subs %[cnt], #1                        @ loop count minus 1\n"
                                "bne    start_mid_mid                   @ jump to main loop start point\n"

                                //! process right pad
                                "start_mid_right:                       @ right pad entry\n"

                                "vld1.32  {d26-d27}, [%[doutc0r0]]       @ load dout0r0\n"
                                "vld1.32  {d28-d29}, [%[doutc0r1]]       @ load dout0r1\n"

                                //! 1st row
                                "vld1.32  {d20-d22}, [%[din0_ptr]]!      @ load din r0\n"
                                "vbif d21, d31, %e[vmask_rp]             @ bit select, deal with right pad\n"
                                "vbif d22, d31, %f[vmask_rp]             @ bit select, deal with right pad\n"
                                "vmla.f32 q13, q10, %e[wr00][0]           @ mul weight0 00, out0r0\n"
                                "vld1.32  {d16-d17}, [%[doutc1r0]]        @ load dout1r0\n"
                                "vld1.32  {d18-d19}, [%[doutc1r1]]        @ load dout1r1\n"
                                "vmla.f32 q8, q10, %e[wr10][0]           @ mul weight1 00, out1r0\n"

                                "vext.32  q12, q10, q11, #1                 @ shift left r0\n"
                                "vmla.f32 q13, q12, %e[wr00][1]              @ mul weight0 01, out0r0\n"
                                "vmla.f32 q8, q12, %e[wr10][1]              @ mul weight1 01, out1r0\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r0\n"
                                "vmla.f32 q13, q12, %f[wr00][0]            @ mul weight0 02, out0r0\n"
                                "vmla.f32 q8, q12, %f[wr10][0]            @ mul weight1 02, out1r0\n"

                                //! 2nd row
                                "vld1.32  {d20-d22}, [%[din1_ptr]]!       @ load din r1\n"
                                "vbif d21, d31, %e[vmask_rp]             @ bit select, deal with right pad\n"
                                "vbif d22, d31, %f[vmask_rp]             @ bit select, deal with right pad\n"
                                "vmla.f32 q13, q10, %e[wr01][0]           @ mul weight0 10, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr00][0]           @ mul weight0 00, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr11][0]           @ mul weight1 10, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr10][0]           @ mul weight1 00, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r1\n"
                                "vmla.f32 q13, q12, %e[wr01][1]            @ mul weight0 11, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr00][1]            @ mul weight0 01, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr11][1]            @ mul weight1 11, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr10][1]            @ mul weight1 01, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r1\n"
                                "vmla.f32 q13, q12, %f[wr01][0]            @ mul weight0 12, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr00][0]            @ mul weight0 02, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr11][0]            @ mul weight1 12, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr10][0]            @ mul weight1 02, out1r1\n"

                                //! 3rd row
                                "vld1.32  {d20-d22}, [%[din2_ptr]]!       @ load din r2\n"
                                "vbif d21, d31, %e[vmask_rp]             @ bit select, deal with right pad\n"
                                "vbif d22, d31, %f[vmask_rp]             @ bit select, deal with right pad\n"
                                "vmla.f32 q13, q10, %e[wr02][0]           @ mul weight0 20, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr01][0]           @ mul weight0 10, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr12][0]           @ mul weight1 20, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr11][0]           @ mul weight1 10, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r2\n"
                                "vmla.f32 q13, q12, %e[wr02][1]            @ mul weight0 21, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr01][1]            @ mul weight0 11, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr12][1]            @ mul weight1 21, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr11][1]            @ mul weight1 11, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift right r2\n"
                                "vmla.f32 q13, q12, %f[wr02][0]            @ mul weight0 22, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr01][0]            @ mul weight0 12, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr12][0]            @ mul weight1 22, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr11][0]            @ mul weight1 12, out1r1\n"

                                //! 4rd row
                                "vld1.32  {d20-d22}, [%[din3_ptr]]!      @ load din r3\n"
                                "vbif d21, d31, %e[vmask_rp]             @ bit select, deal with right pad\n"
                                "vbif d22, d31, %f[vmask_rp]             @ bit select, deal with right pad\n"
                                "vmla.f32 q14, q10, %e[wr02][0]           @ mul weight0 20, out0r1\n"
                                "vmla.f32 q9, q10, %e[wr12][0]           @ mul weight1 20, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r3\n"
                                "vmla.f32 q14, q12, %e[wr02][1]            @ mul weight0 21, out0r1\n"
                                "vmla.f32 q9, q12, %e[wr12][1]            @ mul weight1 21, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r3\n"
                                "vmla.f32 q14, q12, %f[wr02][0]            @ mul weight0 22, out0r1\n"
                                "vmla.f32 q9, q12, %f[wr12][0]            @ mul weight1 22, out1r1\n"

                                "vld1.32  {d20-d21}, [%[doutc0r0]]       @ load dout0r0\n"
                                "vld1.32  {d22-d23}, [%[doutc0r1]]       @ load dout0r1\n"

                                "vmvn.32  q12, q15                      @ \n"
                                "vext.32  q15, q12, %q[vmask_rp], #3    @ shift mask right 1\n"
                                "vbif q13, q10, q15                      @ bit select\n"
                                "vbif q14, q11, q15                      @ bit select\n"

                                "vst1.32  {d26-d27}, [%[doutc0r0]]!    @ store result, add pointer\n"

                                "vst1.32  {d28-d29}, [%[doutc0r1]]!    @ store result, add pointer\n"

                                "vld1.32  {d20-d21}, [%[doutc1r0]]       @ load dout1r0\n"
                                "vld1.32  {d22-d23}, [%[doutc1r1]]       @ load dout1r1\n"

                                "vbif q8, q10, q15                      @ bit select\n"
                                "vbif q9, q11, q15                      @ bit select\n"

                                "vst1.32  {d16-d17}, [%[doutc1r0]]!    @ store result, add pointer\n"
                                "vst1.32  {d18-d19}, [%[doutc1r1]]!    @ store result, add pointer\n"

                                "sub %[doutc0r0], %[doutc0r0], %[right_pad_sub] @ sub \n"
                                "sub %[doutc0r1], %[doutc0r1], %[right_pad_sub] @ sub \n"
                                "sub %[doutc1r0], %[doutc1r0], %[right_pad_sub] @ sub \n"
                                "sub %[doutc1r1], %[doutc1r1], %[right_pad_sub] @ sub \n"

                        :[doutc0r0] "+r"(doutc0r0), [doutc0r1] "+r"(doutc0r1), \
                            [doutc1r0] "+r" (doutc1r0), [doutc1r1] "+r" (doutc1r1),\
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [din3_ptr] "+r"(din3_ptr), \
                            [cnt] "+r"(cnt)
                        :[wr00] "w"(wr00), [wr01] "w"(wr01), [wr02] "w"(wr02), \
                            [wr10] "w"(wr10), [wr11] "w"(wr11), [wr12] "w"(wr12), \
                            [vmask_rp] "w" (vmask_rp), [right_pad_sub] "r" (right_pad_sub)
                        :"q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                        );
#endif //__aarch64__
                    }
                    doutc0r0 += w_out;
                    doutc0r1 = doutc0r0 + w_out;
                    doutc1r0 += w_out;
                    doutc1r1 = doutc1r0 + w_out;

                    dr0 = dr2;
                    dr1 = dr3;
                    dr2 = dr1 + w_in;
                    dr3 = dr2 + w_in;
                } //! end of processing mid rows

                //! deal with bottom pad
                if (1) {

                    din0_ptr = dr0;
                    din1_ptr = dr1;
                    if (size_pad_bottom == 2){
                        din2_ptr = ptr_zero;
                    } else {
                        din2_ptr = dr2;
                    }
                    // process
                    {
#ifdef __aarch64__
                        // todo
#else
                        int cnt = cnt_col;
                        asm volatile (
                        //! process left pad
                        "pld [%[doutc0r0], #192]                @ preload data\n"
                                "pld [%[doutc0r1], #192]                @ preload data\n"
                                "pld [%[doutc1r0], #192]                @ preload data\n"
                                "pld [%[doutc1r1], #192]                @ preload data\n"
                                "vld1.32  {d26-d27}, [%[doutc0r0]]       @ load dout0r0\n"
                                "pld [%[din0_ptr], #192]                @ preload data\n"
                                "pld [%[din1_ptr], #192]                @ preload data\n"
                                "pld [%[din2_ptr], #192]                @ preload data\n"
                                "vld1.32  {d28-d29}, [%[doutc0r1]]       @ load dout0r1\n"

                                //! 1st row
                                "vld1.32  {d20-d22}, [%[din0_ptr]]!      @ load din r0\n"
                                "vmla.f32 q13, q10, %e[wr00][1]           @ mul weight0 01, out0r0\n"
                                "vld1.32  {d16-d17}, [%[doutc1r0]]        @ load dout1r0\n"
                                "vld1.32  {d18-d19}, [%[doutc1r1]]        @ load dout1r1\n"
                                "vmla.f32 q8, q10, %e[wr10][1]           @ mul weight1 01, out1r0\n"

                                "pld [%[din0_ptr], #192]                @ preload data\n"

                                "vext.32  q12, q10, q11, #1                 @ shift left r0\n"
                                "vmla.f32 q13, q12, %f[wr00][0]              @ mul weight0 02, out0r0\n"
                                "vmla.f32 q8, q12, %f[wr10][0]              @ mul weight1 02, out1r0\n"

                                "vmov.u32 q15, #0                         @ dump zero\n"
                                "vext.32  q12, q15, q10, #3               @ shift right r1\n"
                                "vmla.f32 q13, q12, %e[wr00][0]            @ mul weight0 00, out0r0\n"
                                "vmla.f32 q8, q12, %e[wr10][0]            @ mul weight1 00, out1r0\n"

                                //! 2nd row
                                "vld1.32  {d20-d22}, [%[din1_ptr]]!       @ load din r1\n"
                                "vmla.f32 q13, q10, %e[wr01][1]           @ mul weight0 11, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr00][1]           @ mul weight0 01, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr11][1]           @ mul weight1 11, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr10][1]           @ mul weight1 01, out1r1\n"

                                "pld [%[din1_ptr], #192]                @ preload data\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r1\n"
                                "vmla.f32 q13, q12, %f[wr01][0]            @ mul weight0 12, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr00][0]            @ mul weight0 02, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr11][0]            @ mul weight1 12, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr10][0]            @ mul weight1 02, out1r1\n"

                                "vext.32  q12, q15, q10, #3               @ shift right r1\n"
                                "vmla.f32 q13, q12, %e[wr01][0]            @ mul weight0 10, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr00][0]            @ mul weight0 00, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr11][0]            @ mul weight1 10, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr10][0]            @ mul weight1 00, out1r1\n"

                                //! 3rd row
                                "vld1.32  {d20-d22}, [%[din2_ptr]]       @ load din r2\n"
                                "vmla.f32 q13, q10, %e[wr02][1]           @ mul weight0 21, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr01][1]           @ mul weight0 11, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr12][1]           @ mul weight1 21, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr11][1]           @ mul weight1 11, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r2\n"
                                "vmla.f32 q13, q12, %f[wr02][0]            @ mul weight0 22, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr01][0]            @ mul weight0 12, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr12][0]            @ mul weight1 22, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr11][0]            @ mul weight1 12, out1r1\n"

                                "vext.32  q12, q15, q10, #3               @ shift right r2\n"
                                "vmla.f32 q13, q12, %e[wr02][0]            @ mul weight0 20, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr01][0]            @ mul weight0 10, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr12][0]            @ mul weight1 20, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr11][0]            @ mul weight1 10, out1r1\n"


                                "vst1.32  {d26-d27}, [%[doutc0r0]]!    @ store result, add pointer\n"
                                "vst1.32  {d16-d17}, [%[doutc1r0]]!    @ store result, add pointer\n"

                                "pld [%[doutc0r0], #192]                @ preload data\n"
                                "pld [%[doutc1r0], #192]                @ preload data\n"

                                "sub %[din0_ptr], #12                   @ 1pad + 2 float data overlap\n"
                                "sub %[din1_ptr], #12                   @ 1pad + 2 float data overlap\n"

                                "cmp %[bot_pad],  #2  @ check if bottom pad is 2\n"
                                "beq    conv3x3_bot_mid @ jump to next block\n"
                                "vst1.32  {d28-d29}, [%[doutc0r1]]!    @ store result, add pointer\n"
                                "vst1.32  {d18-d19}, [%[doutc1r1]]!    @ store result, add pointer\n"
                                "add %[din2_ptr], #12                   @ 1pad + 2 float data overlap\n"
                                "pld [%[din2_ptr], #192]                @ preload data\n"
                                "pld [%[doutc0r1], #192]                @ preload data\n"
                                "pld [%[doutc1r1], #192]                @ preload data\n"

                                //! process mid cols
                                "cmp %[cnt], #1                         @ check whether has mid cols\n"
                                "blt  conv3x3_bot_right                   @ jump to main loop start point\n"
                                "conv3x3_bot_mid:                         @ main loop start point\n"

                                "vld1.32  {d26-d27}, [%[doutc0r0]]       @ load dout0r0\n"
                                "vld1.32  {d28-d29}, [%[doutc0r1]]       @ load dout0r1\n"

                                //! 1st row
                                "vld1.32  {d20-d22}, [%[din0_ptr]]!      @ load din r0\n"
                                "vmla.f32 q13, q10, %e[wr00][0]           @ mul weight0 00, out0r0\n"
                                "vld1.32  {d16-d17}, [%[doutc1r0]]        @ load dout1r0\n"
                                "vld1.32  {d18-d19}, [%[doutc1r1]]        @ load dout1r1\n"
                                "vmla.f32 q8, q10, %e[wr10][0]           @ mul weight1 00, out1r0\n"

                                "pld [%[din0_ptr], #192]                @ preload data\n"

                                "vext.32  q12, q10, q11, #1                 @ shift left r0\n"
                                "vmla.f32 q13, q12, %e[wr00][1]              @ mul weight0 01, out0r0\n"
                                "vmla.f32 q8, q12, %e[wr10][1]              @ mul weight1 01, out1r0\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r0\n"
                                "vmla.f32 q13, q12, %f[wr00][0]            @ mul weight0 02, out0r0\n"
                                "vmla.f32 q8, q12, %f[wr10][0]            @ mul weight1 02, out1r0\n"

                                //! 2nd row
                                "vld1.32  {d20-d22}, [%[din1_ptr]]!       @ load din r1\n"
                                "vmla.f32 q13, q10, %e[wr01][0]           @ mul weight0 10, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr00][0]           @ mul weight0 00, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr11][0]           @ mul weight1 10, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr10][0]           @ mul weight1 00, out1r1\n"

                                "pld [%[din1_ptr], #192]                @ preload data\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r1\n"
                                "vmla.f32 q13, q12, %e[wr01][1]            @ mul weight0 11, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr00][1]            @ mul weight0 01, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr11][1]            @ mul weight1 11, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr10][1]            @ mul weight1 01, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r1\n"
                                "vmla.f32 q13, q12, %f[wr01][0]            @ mul weight0 12, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr00][0]            @ mul weight0 02, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr11][0]            @ mul weight1 12, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr10][0]            @ mul weight1 02, out1r1\n"

                                //! 3rd row
                                "vld1.32  {d20-d22}, [%[din2_ptr]]       @ load din r2\n"
                                "vmla.f32 q13, q10, %e[wr02][0]           @ mul weight0 20, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr01][0]           @ mul weight0 10, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr12][0]           @ mul weight1 20, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr11][0]           @ mul weight1 10, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r2\n"
                                "vmla.f32 q13, q12, %e[wr02][1]            @ mul weight0 21, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr01][1]            @ mul weight0 11, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr12][1]            @ mul weight1 21, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr11][1]            @ mul weight1 11, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift right r2\n"
                                "vmla.f32 q13, q12, %f[wr02][0]            @ mul weight0 22, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr01][0]            @ mul weight0 12, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr12][0]            @ mul weight1 22, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr11][0]            @ mul weight1 12, out1r1\n"

                                "vst1.32  {d26-d27}, [%[doutc0r0]]!    @ store result, add pointer\n"
                                "vst1.32  {d16-d17}, [%[doutc1r0]]!    @ store result, add pointer\n"
                                "pld [%[doutc0r0], #192]                @ preload data\n"
                                "pld [%[doutc1r0], #192]                @ preload data\n"

                                "sub %[din0_ptr], #8                    @ 2 float data overlap with previous data\n"
                                "sub %[din1_ptr], #8                    @ 2 float data overlap with previous data\n"

                                "cmp %[bot_pad],  #2                    @ check if bottom pad is 2\n"
                                "beq    end_bot_mid                     @ jump to check point\n"

                                "vst1.32  {d28-d29}, [%[doutc0r1]]!     @ store result, add pointer\n"
                                "vst1.32  {d18-d19}, [%[doutc1r1]]!     @ store result, add pointer\n"
                                "pld [%[doutc0r1], #192]                @ preload data\n"
                                "pld [%[doutc1r1], #192]                @ preload data\n"
                                "add %[din2_ptr], #16                   @ point to 4 data ahead\n"
                                "pld [%[din2_ptr], #192]                @ preload data\n"

                                "end_bot_mid:  @ check point\n"
                                "subs %[cnt], #1                        @ loop count minus 1\n"
                                "bne    conv3x3_bot_mid                   @ jump to main loop start point\n"

                                //! process right pad
                                "conv3x3_bot_right:                       @ right pad entry\n"

                                "vld1.32  {d26-d27}, [%[doutc0r0]]       @ load dout0r0\n"
                                "vld1.32  {d28-d29}, [%[doutc0r1]]       @ load dout0r1\n"

                                //! 1st row
                                "vld1.32  {d20-d22}, [%[din0_ptr]]!      @ load din r0\n"
                                "vbif d21, d31, %e[vmask_rp]             @ bit select, deal with right pad\n"
                                "vbif d22, d31, %f[vmask_rp]             @ bit select, deal with right pad\n"
                                "vmla.f32 q13, q10, %e[wr00][0]           @ mul weight0 00, out0r0\n"
                                "vld1.32  {d16-d17}, [%[doutc1r0]]        @ load dout1r0\n"
                                "vld1.32  {d18-d19}, [%[doutc1r1]]        @ load dout1r1\n"
                                "vmla.f32 q8, q10, %e[wr10][0]           @ mul weight1 00, out1r0\n"

                                "vext.32  q12, q10, q11, #1                 @ shift left r0\n"
                                "vmla.f32 q13, q12, %e[wr00][1]              @ mul weight0 01, out0r0\n"
                                "vmla.f32 q8, q12, %e[wr10][1]              @ mul weight1 01, out1r0\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r0\n"
                                "vmla.f32 q13, q12, %f[wr00][0]            @ mul weight0 02, out0r0\n"
                                "vmla.f32 q8, q12, %f[wr10][0]            @ mul weight1 02, out1r0\n"

                                //! 2nd row
                                "vld1.32  {d20-d22}, [%[din1_ptr]]!       @ load din r1\n"
                                "vbif d21, d31, %e[vmask_rp]             @ bit select, deal with right pad\n"
                                "vbif d22, d31, %f[vmask_rp]             @ bit select, deal with right pad\n"
                                "vmla.f32 q13, q10, %e[wr01][0]           @ mul weight0 10, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr00][0]           @ mul weight0 00, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr11][0]           @ mul weight1 10, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr10][0]           @ mul weight1 00, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r1\n"
                                "vmla.f32 q13, q12, %e[wr01][1]            @ mul weight0 11, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr00][1]            @ mul weight0 01, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr11][1]            @ mul weight1 11, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr10][1]            @ mul weight1 01, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift left r1\n"
                                "vmla.f32 q13, q12, %f[wr01][0]            @ mul weight0 12, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr00][0]            @ mul weight0 02, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr11][0]            @ mul weight1 12, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr10][0]            @ mul weight1 02, out1r1\n"

                                //! 3rd row
                                "vld1.32  {d20-d22}, [%[din2_ptr]]       @ load din r2\n"
                                "vbif d21, d31, %e[vmask_rp]             @ bit select, deal with right pad\n"
                                "vbif d22, d31, %f[vmask_rp]             @ bit select, deal with right pad\n"
                                "vmla.f32 q13, q10, %e[wr02][0]           @ mul weight0 20, out0r0\n"
                                "vmla.f32 q14, q10, %e[wr01][0]           @ mul weight0 10, out0r1\n"
                                "vmla.f32 q8, q10, %e[wr12][0]           @ mul weight1 20, out1r0\n"
                                "vmla.f32 q9, q10, %e[wr11][0]           @ mul weight1 10, out1r1\n"

                                "vext.32  q12, q10, q11, #1               @ shift left r2\n"
                                "vmla.f32 q13, q12, %e[wr02][1]            @ mul weight0 21, out0r0\n"
                                "vmla.f32 q14, q12, %e[wr01][1]            @ mul weight0 11, out0r1\n"
                                "vmla.f32 q8, q12, %e[wr12][1]            @ mul weight1 21, out1r0\n"
                                "vmla.f32 q9, q12, %e[wr11][1]            @ mul weight1 11, out1r1\n"

                                "vext.32  q12, q10, q11, #2               @ shift right r2\n"
                                "vmla.f32 q13, q12, %f[wr02][0]            @ mul weight0 22, out0r0\n"
                                "vmla.f32 q14, q12, %f[wr01][0]            @ mul weight0 12, out0r1\n"
                                "vmla.f32 q8, q12, %f[wr12][0]            @ mul weight1 22, out1r0\n"
                                "vmla.f32 q9, q12, %f[wr11][0]            @ mul weight1 12, out1r1\n"

                                "vld1.32  {d20-d21}, [%[doutc0r0]]       @ load dout0r0\n"
                                "vld1.32  {d22-d23}, [%[doutc1r0]]       @ load dout0r1\n"

                                "vmvn.32  q12, q15                      @ \n"
                                "vext.32  q15, q12, %q[vmask_rp], #3    @ shift mask right 1\n"
                                "vbif q13, q10, q15                     @ bit select\n"
                                "vbif q8, q11, q15                      @ bit select\n"

                                "vst1.32  {d26-d27}, [%[doutc0r0]]      @ store result, add pointer\n"
                                "vst1.32  {d16-d17}, [%[doutc1r0]]      @ store result, add pointer\n"

                                "cmp %[bot_pad],  #2  @ check if bottom pad is 2\n"
                                "beq    end_conv3x3s1 @ jump to end point\n"

                                "vld1.32  {d20-d21}, [%[doutc0r1]]       @ load dout0r0\n"
                                "vld1.32  {d22-d23}, [%[doutc1r1]]       @ load dout0r1\n"

                                "vbif q14, q10, q15                     @ bit select\n"
                                "vbif q9, q11, q15                      @ bit select\n"

                                "vst1.32  {d28-d29}, [%[doutc0r1]]      @ store result, add pointer\n"
                                "vst1.32  {d18-d19}, [%[doutc1r1]]      @ store result, add pointer\n"
                                "end_conv3x3s1:  @ end\n"
                        :[doutc0r0] "+r"(doutc0r0), [doutc0r1] "+r"(doutc0r1), \
                            [doutc1r0] "+r" (doutc1r0), [doutc1r1] "+r" (doutc1r1),\
                            [din0_ptr] "+r"(din0_ptr), [din1_ptr] "+r"(din1_ptr), \
                            [din2_ptr] "+r"(din2_ptr), [cnt] "+r"(cnt)
                        :[wr00] "w"(wr00), [wr01] "w"(wr01), [wr02] "w"(wr02), \
                            [wr10] "w"(wr10), [wr11] "w"(wr11), [wr12] "w"(wr12), \
                            [vmask_rp] "w" (vmask_rp), [bot_pad] "r"(size_pad_bottom)
                        :"q8", "q9", "q10", \
                            "q11", "q12", "q13", "q14", "q15"
                        );
#endif //__aarch64__
                    }
                } // end of processing bottom pad
            } // end of processing channels
        } //end of processing output channel
        if (cremain > 0) {
            for (int c = 0; c < cremain; ++c) {

                int cidx = ch_out - cremain + c;
                float* dout_c = dout_batch + cidx * size_out_channel;

                if (flag_bias) {
                    fill_bias(dout_c, &bias[cidx], 1, size_out_channel);
                } else {
                    fill_bias(dout_c, zero, 1, size_out_channel);
                }

                const float* wc0 = weights + cidx * w_stride;

                for (int i = 0; i < ch_in; ++i) {
                    const float* din_channel = din_batch + i * size_in_channel;
                    for (int h = 0; h < h_out; ++h) {

                        int hstart = h - pad_h;
                        int hend = hstart + 3;
                        hstart = std::max(hstart, 0);
                        hend = std::min(hend, h_in);

                        int khstart = hend < kernel_h? kernel_h - hend : 0;

                        float* dout_row = dout_c + h * w_out;

                        for (int w = 0; w < w_out; ++w) {
                            int wstart = w - pad_w;
                            int wend = wstart + 3;
                            wstart = std::max(wstart, 0);
                            wend = std::min(wend, w_in);
                            int kwstart = wend < kernel_w? kernel_w - wend : 0;

                            for (int kh = hstart; kh < hend; ++kh) {
                                for (int kw = wstart; kw < wend; ++kw) {
                                    dout_row[w] += din_channel[kh * w_in + kw] * \
                                        wc0[(khstart + kh - hstart) * 3 + kwstart + kw - wstart];
                                }
                            }
                        }
                    }
                    wc0 += 9;
                }
            }
        } // end of remain out channel

    } // end of processing batchs
}

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE