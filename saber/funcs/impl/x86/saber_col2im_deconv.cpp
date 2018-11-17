
#include "saber/funcs/impl/x86/saber_col2im_deconv.h"

namespace anakin {
namespace saber {

void fill_bias_relu(float* tensor, const float* bias, int channel, int channel_size,
                    bool flag_relu) {
    float* data = tensor;
    for (int j = 0; j < channel; ++j) {
        for (int i = 0; i < channel_size; i++) {
            data[i] += bias[j];
            if (flag_relu) {
                data[i] = data[i] > 0 ? data[i] : 0.f;
            }
        }
        data += channel_size;
    }
}

void fill_relu(float* tensor, int channel, int channel_size,
               bool flag_relu) {
    float* data = tensor;
    for (int j = 0; j < channel; ++j) {
        for (int i = 0; i < channel_size; i++) {
            if (flag_relu) {
                data[i] = data[i] > 0 ? data[i] : 0.f;
            }
        }
        data += channel_size;
    }
}


inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void col2im(const Dtype* data_col, const int channels,
            const int height, const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            Dtype* data_im) {

    memset(data_im, 0, height * width * channels * sizeof(Dtype));
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;

    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;

                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        data_col += output_w;
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;

                        for (int output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                data_im[input_row * width + input_col] += *data_col;
                            }
                            data_col++;
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
SaberStatus SaberCol2ImDeconv<AK_FLOAT>::create(const std::vector<Tensor<X86> *>& inputs,
                                              std::vector<Tensor<X86>*>& outputs,
                                              ConvParam<X86> &param, Context<X86>&ctx) {
    this->_ctx = &ctx;

    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chin = inputs[0]->channel();
    int chout = outputs[0]->channel();

    int _kw = param.weight()->width();
    int _kh = param.weight()->height();

    int _m = chout * _kw * _kh / param.group;
    int _n = hin * win;
    int _k = chin / param.group;

    if (chin != chout || param.group != chin) {
        CHECK_EQ(chin % param.group, 0) << "input channel or group size error";
        CHECK_EQ(chout % param.group, 0) << "output channel or group size error";
    }
    Shape workspace_shape({1, 1, 1, param.group* _m * _n});
    workspace_tensor.re_alloc(workspace_shape, AK_FLOAT);

    _gemm.init(true, false, _m, _n, _k, *(this->_ctx));
    return SaberSuccess;
}

template <>
SaberStatus SaberCol2ImDeconv<AK_FLOAT>::init(const std::vector<Tensor<X86> *>& inputs,
                                            std::vector<Tensor<X86>*>& outputs,
                                            ConvParam<X86> &param, Context<X86>&ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberCol2ImDeconv<AK_FLOAT>::dispatch(const std::vector<Tensor<X86> *>& inputs,
                                                std::vector<Tensor<X86>*>& outputs,
                                                ConvParam<X86> &param) {
    bool bias_term = param.bias() != nullptr && param.bias()->valid_size() > 0;
    int win = inputs[0]->width();
    int hin = inputs[0]->height();
    int chin = inputs[0]->channel();
    int num = inputs[0]->num();
    int wout = outputs[0]->width();
    int hout = outputs[0]->height();
    int chout = outputs[0]->channel();

    int _kw = param.weight()->width();
    int _kh = param.weight()->height();

    int _m = chout * _kw * _kh / param.group;
    int _n = hin * win;
    int _k = chin / param.group;

    int group = param.group;
    int group_size_in = win * hin * chin / group;
    int group_size_out = wout * hout * chout / group;
    int group_size_coldata = _m * _n;
    int group_size_weights = chin * chout * _kw * _kh / (group * group);
    bool flag_1x1s1p1 = (_kw == 1) && (_kh == 1) && (param.stride_h == 1) && \
                        (param.stride_w == 1) && (param.pad_w == 1) && (param.pad_h == 1) && \
                        (param.dilation_w == 1) && (param.dilation_h == 1);

    bool with_relu = (param.activation_param.active == Active_relu);
    const float* din = static_cast<const float*>(inputs[0]->data());
    float* dout = static_cast<float*>(outputs[0]->mutable_data());
    const float* weights = static_cast<const float*>(param.weight()->data());
    float* workspace_ptr = static_cast<float*>(workspace_tensor.mutable_data());

    for (int i = 0; i < num; ++i) {
        const float* din_batch = din + i * chin * hin * win;
        float* dout_batch = dout + i * chout * hout * wout;

        float* col_data = workspace_ptr;
        for (int g = 0; g < param.group; ++g) {
            const float* din_group = din_batch + g * group_size_in;
            const float* weights_group = weights + g * group_size_weights;
            float* coldata_group = col_data + g * group_size_coldata;
            _gemm.dispatch(1.f, 0.f, weights_group, din_group, coldata_group);
        }
        col2im(col_data, chout, hout, wout, _kh, _kw, param.pad_h, param.pad_w, \
               param.stride_h, param.stride_w, param.dilation_h, param.dilation_w, \
               dout_batch);

        //! add bias
        if (bias_term) {
            fill_bias_relu(dout_batch, static_cast<const float*>(param.bias()->data()), chout, wout * hout,
                           with_relu);
        } else {
            fill_relu(dout_batch, chout, wout * hout,
                      with_relu);
        }
    }
}
}
}
