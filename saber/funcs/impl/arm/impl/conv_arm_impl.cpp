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

    const int m = ch_out / group;
    const int n = h_out * w_out;
    const int k = ch_in / group;

    int weights_size_per_group = ch_out * ch_in / (group * group);

        //! use gemv when the output channel size = 1
    if (n == 1) {
        for (int b = 0; b < num; ++b) {
            for (int g = 0; g < group; ++g) {
                float* dout_group = tensor_out.mutable_data() + (b * ch_out + g * m) * channel_size_out;
                const float* din_group = tensor_in.data() + (b * ch_in + g * k)* channel_size_in;
                const float* weights_group = weights + g * weights_size_per_group;
                const float* bias_group = bias + g * m;
                if (flag_bias){
                    if (flag_relu) {
                        sgemv_bias_relu(false, m, k, weights_group, din_group, dout_group, bias_group);
                    } else {
                        sgemv_bias(false, m, k, weights_group, din_group, dout_group, bias_group);
                    }

                } else {
                    if (flag_relu) {
                        sgemv_relu(false, m, k, weights_group, din_group, dout_group);
                    } else {
                        sgemv(false, m, k, weights_group, din_group, dout_group);
                    }
                }
            }
        }

    } else {
        for (int b = 0; b < num; ++b) {
            // dC
            for (int g = 0; g < group; ++g) {
                float* dout_group = tensor_out.mutable_data() + (b * ch_out + g * m) * channel_size_out;
                const float* din_group = tensor_in.data() + (b * ch_in + g * k) * channel_size_in;
                const float* weights_group = weights + g * weights_size_per_group;
                const float* bias_group = bias + g * m;
                float beta = 0.f;
                if (flag_bias) {
                    fill_bias(dout_group, bias_group, m, w_out * h_out);
                    beta = 1.f;
                }
                gemmer(weights_group, k, din_group, n, dout_group, n, 1.f, beta, flag_relu);
            }

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

    const int m = ch_out / group;
    const int n = h_out * w_out;
    const int k = ch_in * kernel_h * kernel_w / group;

    const int chin_per_group = ch_in / group;

    int channel_size_out = w_out * h_out;
    int channel_size_in = w_in * h_in;

    int weights_size_per_group = ch_out * ch_in * kernel_w * kernel_h / (group * group);
        
    for (int b = 0; b < num; ++b) {
        // dC
        for (int g = 0; g < group; ++g) {
            float* dout_group = tensor_out.mutable_data() + (b * ch_out + g * m) * channel_size_out;
            const float* din_group = tensor_in.data() + (b * ch_in + g * chin_per_group) * channel_size_in;
            const float* weights_group = weights + g * weights_size_per_group;
            const float* bias_group = bias + g * m;
            float* dB = (float*)work_space;
            if (kernel_w == 1 && pad_w == 0) {
                im2col1x1s2(din_group, chin_per_group, h_in, w_in, dB);
            } else {
                im2col(din_group, chin_per_group, h_in, w_in, kernel_h, kernel_w, \
                    pad_h, pad_w, stride_h, stride_w, dB);
            }
            float beta = 0.f;
            if (flag_bias) {
                fill_bias(dout_group, bias_group, m, w_out * h_out);
                beta = 1.f;
            }

            gemmer(weights_group, k, dB, n, dout_group, n, 1.f, beta, flag_relu);
        }
    }
}

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE