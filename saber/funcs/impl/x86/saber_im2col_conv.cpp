
#include "saber/funcs/impl/x86/saber_im2col_conv.h"

namespace anakin {
namespace saber {

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                const int dilation_h, const int dilation_w,
                Dtype* data_col) {

    const int output_h = (height + 2 * pad_h -
                          (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
                          (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
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

template <>
SaberStatus SaberIm2colConv<AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86>*>& outputs,
      ConvEltwiseParam<X86>& param, Context<X86>& ctx) {
    ConvParam<X86>* conv_param = &(param.conv_param);
    this->_ctx = &ctx;
    int in_c = inputs[0]->channel();
    int in_h = inputs[0]->height();
    int in_w = inputs[0]->width();
    int out_c = outputs[0]->channel();
    int out_h = outputs[0]->height();
    int out_w = outputs[0]->width();
    int kernel_h = conv_param->weight()->height();
    int kernel_w = conv_param->weight()->width();

    int slice_size = in_c * kernel_h * kernel_w * out_h * out_w / conv_param->group;
    Shape _im2col_shape({slice_size}, Layout_W);
    _im2col_tensor.reshape(_im2col_shape);

    int out_stride = out_h * out_w;
    _gemm.init(false, false, out_c / conv_param->group, out_stride, in_c / conv_param->group * kernel_h * kernel_w,
               *(this->_ctx));

    return SaberSuccess;
}

template <>
SaberStatus SaberIm2colConv<AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param, Context<X86>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}
template <>
SaberStatus SaberIm2colConv<AK_FLOAT>::dispatch(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param) {
    ConvParam<X86>* conv_param = &(param.conv_param);
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

    const float* din = (const float*)inputs[0]->data();
    float* dout = (float*)outputs[0]->mutable_data();
    const float* weights_d = (const float*)conv_param->weight()->data();


    bool flag_bias = (conv_param->bias()->valid_size() > 0);
    bool flag_relu = conv_param->activation_param.has_active;
    const float* bias = flag_bias ? (const float*)conv_param->bias()->data() : nullptr;
    const float negative_slope = conv_param->activation_param.negative_slope;

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < group; j++) {
            im2col_cpu(din, in_c / group, in_h, in_w, kernel_h, kernel_w, conv_param->pad_h, conv_param->pad_w,
                       conv_param->stride_h, conv_param->stride_w, conv_param->dilation_h, conv_param->dilation_w,
                       (float*)_im2col_tensor.mutable_data());

            _gemm.dispatch(1.f, 0.f, weights_d + j * weight_size_per_group, (const float*)_im2col_tensor.data(),
                           dout);

            din += in_c / group * in_stride;
            dout += out_c / group * out_stride;
        }
    }

    if (flag_bias && !flag_relu) {
        float* output = (float*)outputs[0]->mutable_data();
        int id = 0;

        for (int i = 0; i < batch_size; i++) {
            for (int oc = 0; oc < out_c; ++oc) {
                for (int inner_id = 0; inner_id < out_stride; ++inner_id, ++id) {
                    output[id] += bias[oc];
                }
            }
        }
    } else if (!flag_bias && flag_relu) {
        float* output = (float*)outputs[0]->mutable_data();
        int id = 0;

        for (int i = 0; i < batch_size; i++) {
            for (int oc = 0; oc < out_c; ++oc) {
                for (int inner_id = 0; inner_id < out_stride; ++inner_id, ++id) {
                    if (output[id] < 0) {
                        output[id] = output[id] * negative_slope;
                    }
                }
            }
        }
    } else if (flag_bias && flag_relu) {
        float *output = (float*)outputs[0]->mutable_data();
        int id = 0;
        for (int i = 0; i < batch_size; i++) {
            for (int oc = 0; oc < out_c; ++oc) {
                for (int inner_id = 0; inner_id < out_stride; ++inner_id, ++id) {
                    float temp = output[id];
                    temp += bias[oc];
                    if (temp < 0) {
                        temp = temp * negative_slope;
                    }

                    output[id] = temp;
                }
            }
        }
    }

    return SaberSuccess;
}
}
}
