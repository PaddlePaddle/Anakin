#include "saber/lite/funcs/neon/impl/conv_arm_impl.h"
#ifdef USE_ARM_PLACE
#include "saber/lite/funcs/neon/impl/sgemv_arm.h"
#include "saber/lite/funcs/neon/impl/sgemv_arm_int8.h"
#include "saber/lite/funcs/neon/impl/sgemm_conv.h"
#include "saber/lite/funcs/neon/impl/sgemm_prepacked_int8.h"

namespace anakin{

namespace saber{

namespace lite{

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

void fill_bias_int8(int* tensor, const int* bias, int channel, int channel_size) {

    int* data = tensor;

    for (int j = 0; j < channel; ++j) {
        int32x4_t vdata = vdupq_n_s32(bias[j]);
        int i = 0;
        for (; i < channel_size - 3; i += 4) {
            vst1q_s32(data + i, vdata);
        }
        for (; i < channel_size; i++) {
            data[i] = bias[j];
        }
        data += channel_size;
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
template <typename Dtype>
void im2col(const Dtype* data_im, const int channels, const int height, const int width, \
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w, \
    const int stride_h, const int stride_w, const int dilation_h, const int dilation_w, \
    Dtype* data_col) {

    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
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
void compute_offset(int* idx_out, int h, int w, int kernel_h, int kernel_w, int height, int width, int pad_h, int pad_w, int dilation_h, int dilation_w) {
    int idx_h[kernel_h];
    int idx_w[kernel_w];
    for (int i = 0; i < kernel_h; ++i) {
        idx_h[i] = h - pad_h + i * dilation_h;
    }
    for (int i = 0; i < kernel_w; ++i) {
        idx_w[i] = w - pad_w + i * dilation_w;
    }

    for (int k_h = 0; k_h < kernel_h; ++k_h) {
        for (int k_w = 0; k_w < kernel_w; ++k_w) {
            idx_out[k_h * kernel_w + k_w] = (idx_h[k_h] >= 0 && idx_w[k_w] >= 0 && idx_h[k_h] < height && idx_w[k_w] < width) ? idx_h[k_h] * width + idx_w[k_w] : -1;
        }
    }
}
template <typename Dtype>
void im2col3x3(const Dtype* data_im, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_col, const int* idx) {

    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int kernel_stride = kernel_h * kernel_w;
    int in_channel_stride = height * width;
    const int* idx_out = idx;
    Dtype* data_col_ptr = data_col;

    bool flag_continue = false;
    if (dilation_h == 1 && dilation_w == 1) {
        flag_continue = true;
    }

    for (int o = 0; o < output_h * output_w; o += 1) {
        const Dtype* data_im_ptr = data_im;

        //int* idx_out_d = idx_out;

        int idx_out_d0 = idx_out[0];
        int idx_out_d1 = idx_out[1];
        int idx_out_d2 = idx_out[2];
        int idx_out_d3 = idx_out[3];
        int idx_out_d4 = idx_out[4];
        int idx_out_d5 = idx_out[5];
        int idx_out_d6 = idx_out[6];
        int idx_out_d7 = idx_out[7];
        int idx_out_d8 = idx_out[8];

        for (int i = 0; i < channels; i += 1) {

            if (idx_out_d0 >= 0 && idx_out_d2 >= 0 && idx_out_d6 >= 0 && idx_out_d8 >= 0) {
                if (flag_continue) {
                    memcpy(data_col_ptr, data_im_ptr + idx_out_d0, kernel_w * sizeof(Dtype));
                    memcpy(data_col_ptr + kernel_w, data_im_ptr + idx_out_d3, kernel_w * sizeof(Dtype));
                    memcpy(data_col_ptr + kernel_w + kernel_w, data_im_ptr + idx_out_d6, kernel_w * sizeof(Dtype));
                } else {
                    data_col_ptr[0] = data_im_ptr[idx_out_d0];
                    data_col_ptr[1] = data_im_ptr[idx_out_d1];
                    data_col_ptr[2] = data_im_ptr[idx_out_d2];
                    data_col_ptr[3] = data_im_ptr[idx_out_d3];
                    data_col_ptr[4] = data_im_ptr[idx_out_d4];
                    data_col_ptr[5] = data_im_ptr[idx_out_d5];
                    data_col_ptr[6] = data_im_ptr[idx_out_d6];
                    data_col_ptr[7] = data_im_ptr[idx_out_d7];
                    data_col_ptr[8] = data_im_ptr[idx_out_d8];
                }
            } else {
                data_col_ptr[0] = (idx_out_d0 < 0) ? 0 : data_im_ptr[idx_out_d0];
                data_col_ptr[1] = (idx_out_d1 < 0) ? 0 : data_im_ptr[idx_out_d1];
                data_col_ptr[2] = (idx_out_d2 < 0) ? 0 : data_im_ptr[idx_out_d2];
                data_col_ptr[3] = (idx_out_d3 < 0) ? 0 : data_im_ptr[idx_out_d3];
                data_col_ptr[4] = (idx_out_d4 < 0) ? 0 : data_im_ptr[idx_out_d4];
                data_col_ptr[5] = (idx_out_d5 < 0) ? 0 : data_im_ptr[idx_out_d5];
                data_col_ptr[6] = (idx_out_d6 < 0) ? 0 : data_im_ptr[idx_out_d6];
                data_col_ptr[7] = (idx_out_d7 < 0) ? 0 : data_im_ptr[idx_out_d7];
                data_col_ptr[8] = (idx_out_d8 < 0) ? 0 : data_im_ptr[idx_out_d8];
            }
            data_im_ptr += height * width;
            data_col_ptr += kernel_stride;
        }
        //data_col_ptr += channels * kernel_stride;
        //idx_out += kernel_stride * 2;
        idx_out += kernel_stride;
    }
}

/**
 * \brief convolution function for kernel size 1x1, stride size 1, gemm implementation
 */
void conv1x1s1_gemm(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, void* work_space, const void* idx_ptr) {

    int channel_size_out = wout * hout;
    int channel_size_in = win * hin;

    const int m = chout / group;
    const int n = hout * wout;
    const int k = chin / group;

    int hblock = get_hblock(ctx->get_arch());
    int m_roundup = hblock * ((m + hblock - 1) / hblock);
    int weights_size_per_group = m * k;
    if (n > 1) {
        weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
    }

    //int weights_size_per_group = m_roundup * k;//chout * chin / (group * group);
    //! use gemv when the output channel size = 1
    for (int b = 0; b < num; ++b) {
        // dC
        for (int g = 0; g < group; ++g) {
            float* dout_group = static_cast<float*>(dout) + (b * chout + g * m) * channel_size_out;
            const float* din_group = static_cast<const float*>(din) + (b * chin + g * k) * channel_size_in;
            const float* weights_group = static_cast<const float*>(weights) + g * weights_size_per_group;
            const float* bias_group = static_cast<const float*>(bias) + g * m;

            if (n == 1) {
                if (flag_bias) {
                    if (flag_relu) {
                        sgemv_bias_relu(false, m, k, weights_group, din_group, dout_group, bias_group);
                    } else {
                        sgemv_bias(false, m, k, weights_group, din_group, dout_group, bias_group);
                    }
                } else  {
                    if (flag_relu) {
                        sgemv_relu(false, m, k, weights_group, din_group, dout_group);
                    } else {
                        sgemv(false, m, k, weights_group, din_group, dout_group);
                    }
                }
            } else {
                sgemm_prepack(weights_group, din_group, bias_group, dout_group, m, n, k, flag_bias, flag_relu, false, ctx);
            }
        }

    }
}


void conv1x1s1_gemm_int8(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr) {
    int channel_size_out = wout * hout;
    int channel_size_in = win * hin;

    const int m = chout / group;
    const int n = hout * wout;
    const int k = chin / group;

    int hblock = get_hblock_int8(ctx->get_arch());
    int m_roundup = hblock * ((m + hblock - 1) / hblock);
    int weights_size_per_group = m * k;
    if (n > 1) {
        weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
    }

    //! use gemv when the output channel size = 1
    for (int b = 0; b < num; ++b) {
        // dC
        for (int g = 0; g < group; ++g) {
            int* dout_group = static_cast<int*>(dout) + (b * chout + g * m) * channel_size_out;
            const char* din_group = static_cast<const char*>(din) + (b * chin + g * k) * channel_size_in;
            const char* weights_group = static_cast<const char*>(weights) + g * weights_size_per_group;
            const int* bias_group = static_cast<const int*>(bias) + g * m;

            if (n == 1) {
                if (flag_bias) {
                    if (flag_relu) {
                        //sgemv_bias_relu(false, m, k, weights_group, din_group, dout_group, bias_group);
                    } else {
                        //sgemv_bias(false, m, k, weights_group, din_group, dout_group, bias_group);
                    }
                } else  {
                    if (flag_relu) {
                        //sgemv_relu(false, m, k, weights_group, din_group, dout_group);
                    } else {
                        //sgemv(false, m, k, weights_group, din_group, dout_group);
                    }
                }
            } else {
                sgemm_prepack_int8(weights_group, din_group, bias_group, dout_group, m, n, k, flag_bias, flag_relu, false, ctx);
            }
        }

    }
}

/**
 * \brief convolution function for kernel size 3x3, stride size 2, gemm implementation
 */
void conv_im2col_gemm(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, void* work_space, const void* idx_ptr) {

    const int m = chout / group;
    const int n = hout * wout;
    const int k = chin * kernel_h * kernel_w / group;
    const int chin_per_group = chin / group;
    int channel_size_out = wout * hout;
    int channel_size_in = win * hin;

    int hblock = get_hblock(ctx->get_arch());
    int m_roundup = hblock * ((m + hblock - 1) / hblock);
    int weights_size_per_group = m * k;
    if (n > 1) {
        weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
    }

    bool flag_im2col2 = (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && n > 1);

    float* tmp_work_space = static_cast<float*>(ctx->get_work_space()) + ctx->l2_cache_size() / sizeof(float);

    //! use gemv when the output channel size = 1
    for (int b = 0; b < num; ++b) {
        // dC
        for (int g = 0; g < group; ++g) {
            float* dout_group = static_cast<float*>(dout) + (b * chout + g * m) * channel_size_out;
            const float* din_group = static_cast<const float*>(din) + (b * chin + g * chin_per_group) * channel_size_in;
            const float* weights_group = static_cast<const float*>(weights) + g * weights_size_per_group;
            const float* bias_group = static_cast<const float*>(bias) + g * m;
            float* dB = tmp_work_space;

            if (flag_im2col2) {
                const int* idx = (const int*)idx_ptr;
                im2col3x3(din_group, chin_per_group, hin, win, kernel_h, kernel_w, \
                    pad_h, pad_w, stride_h, stride_w, dila_h, dila_w, dB, idx);

            } else {
                im2col(din_group, chin_per_group, hin, win, kernel_h, kernel_w, \
                    pad_h, pad_w, stride_h, stride_w, dila_h, dila_w, dB);
            }
            if (n == 1) {
                if (flag_bias) {
                    if (flag_relu) {
                        sgemv_bias_relu(false, m, k, weights_group, dB, dout_group, bias_group);
                    } else {
                        sgemv_bias(false, m, k, weights_group, dB, dout_group, bias_group);
                    }
                } else {
                    if (flag_relu) {
                        sgemv_relu(false, m, k, weights_group, dB, dout_group);
                    } else {
                        sgemv(false, m, k, weights_group, dB, dout_group);
                    }
                }
            } else {
                sgemm_prepack(weights_group, dB, bias_group, dout_group, m, n, k, flag_bias, flag_relu, flag_im2col2, ctx);
            }
        }
    }
}

void conv_im2col_gemm_int8(const void* din, void* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          const void* weights, const void* bias, \
                          int group, int kernel_w, int kernel_h, int stride_w, int stride_h, int dila_w, int dila_h, \
                          int pad_w, int pad_h, bool flag_bias, bool flag_relu, Context* ctx, \
                          void* work_space, const void* idx_ptr) {
    const int m = chout / group;
    const int n = hout * wout;
    const int k = chin * kernel_h * kernel_w / group;
    const int chin_per_group = chin / group;
    int channel_size_out = wout * hout;
    int channel_size_in = win * hin;

    int hblock = get_hblock_int8(ctx->get_arch());
    int m_roundup = hblock * ((m + hblock - 1) / hblock);
    int weights_size_per_group = m * k;
    if (n > 1) {
        weights_size_per_group = ((m_roundup * k + 15) / 16) * 16;
    }

    bool flag_im2col2 = (kernel_h == 3 && kernel_w == 3 && stride_h == 1 && stride_w == 1 && n > 1);

    char* tmp_work_space = static_cast<char*>(ctx->get_work_space()) + ctx->l2_cache_size();

    //! use gemv when the output channel size = 1
    for (int b = 0; b < num; ++b) {
        // dC
        for (int g = 0; g < group; ++g) {
            int* dout_group = static_cast<int*>(dout) + (b * chout + g * m) * channel_size_out;
            const char* din_group = static_cast<const char*>(din) + (b * chin + g * chin_per_group) * channel_size_in;
            const char* weights_group = static_cast<const char*>(weights) + g * weights_size_per_group;
            const int* bias_group = static_cast<const int*>(bias) + g * m;
            char* dB = tmp_work_space;

            if (flag_im2col2) {
                const int* idx = (const int*)idx_ptr;
                im2col3x3(din_group, chin_per_group, hin, win, kernel_h, kernel_w, \
                    pad_h, pad_w, stride_h, stride_w, dila_h, dila_w, dB, idx);

            } else {
                im2col(din_group, chin_per_group, hin, win, kernel_h, kernel_w, \
                    pad_h, pad_w, stride_h, stride_w, dila_h, dila_w, dB);
            }
            if (n == 1) {
                const signed char* db_ptr = reinterpret_cast<const signed char*> (dB);
                const signed char* wei_ptr = reinterpret_cast<const signed char*> (weights_group);
                if (flag_bias) {
                    if (flag_relu) {
                       sgemv_bias_relu_int8(false, m, k, wei_ptr, db_ptr, dout_group, bias_group);
                    } else {
                       sgemv_bias_int8(false, m, k, wei_ptr, db_ptr, dout_group, bias_group);
                    }
                } else {
                    if (flag_relu) {
                       sgemv_relu_int8(false, m, k, wei_ptr, db_ptr, dout_group);
                    } else {
                       sgemv_int8(false, m, k, wei_ptr, db_ptr, dout_group);
                    }
                }
            } else {
                sgemm_prepack_int8(weights_group, dB, bias_group, dout_group, m, n, k, flag_bias, flag_relu, flag_im2col2, ctx);
            }
        }
    }
}

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE
