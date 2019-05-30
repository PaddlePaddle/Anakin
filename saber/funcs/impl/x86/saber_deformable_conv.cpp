#include "saber/funcs/impl/x86/saber_deformable_conv.h"

namespace anakin {
namespace saber {

template <>
SaberStatus SaberDeformableConv2D<X86, AK_FLOAT>::create(
    const std::vector<Tensor<X86> *>& inputs,
    std::vector<Tensor<X86> *>& outputs,
    DeformableConvParam<X86>& param, Context<X86>& ctx) {
    
    this->_ctx = &ctx;
    return SaberSuccess;
}

template <>
SaberStatus SaberDeformableConv2D<X86, AK_FLOAT>::init(
    const std::vector<Tensor<X86> *>& inputs,
    std::vector<Tensor<X86> *>& outputs,
    DeformableConvParam<X86>& param, Context<X86>& ctx) {

    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

static inline float deformable_bilinear(const float* bottom_data, const int data_width,
                          const int height, const int width, float h, float w) {
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;
    if (h_low >= height - 1) {
        h_high = h_low = height - 1;
        h = (float) h_low;
    } else {
        h_high = h_low + 1;
    }

    if (w_low >= width - 1) {
        w_high = w_low = width - 1;
        w = (float) w_low;
    } else {
        w_high = w_low + 1;
    }
    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh;
    float hw = 1 - lw;
    float v1 = bottom_data[h_low * data_width + w_low];
    float v2 = bottom_data[h_low * data_width + w_high];
    float v3 = bottom_data[h_high * data_width + w_low];
    float v4 = bottom_data[h_high * data_width + w_high];
    float w1 = hh * hw;
    float w2 = hh * lw;
    float w3 = lh * hw;
    float w4 = lh * lw;
    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

template <>
SaberStatus SaberDeformableConv2D<X86, AK_FLOAT>::dispatch(
    const std::vector<Tensor<X86> *>& inputs,
    std::vector<Tensor<X86> *>& outputs,
    DeformableConvParam<X86>& param) {
    
    const float* in_data = (const float*)inputs[0]->data();
    const float* offset_data = (const float*)inputs[1]->data();
    float* out_data = (float*)outputs[0]->mutable_data();
    bool with_bias = param.bias()->size() > 0;

    const int out_num = outputs[0]->num();
    const int out_channels = outputs[0]->channel();
    const int out_h = outputs[0]->height();
    const int out_w = outputs[0]->width();

    const int in_num = inputs[0]->num();
    const int in_channels = inputs[0]->channel();
    const int in_h = inputs[0]->height();
    const int in_w = inputs[0]->width();

    const int out_c_group = out_channels / param.group;
    const int in_c_group = in_channels / param.group;

    const int kernel_h = param.weight()->height();
    const int kernel_w = param.weight()->width();
    const int stride_h = param.stride_h;
    const int stride_w = param.stride_w;
    const int pad_h = param.pad_h;
    const int pad_w = param.pad_w;
    const int dilation_h = param.dilation_h;
    const int dilation_w = param.dilation_w;

    const float alpha = 1.f;
    const float beta = 0.f;
    
    const float* weights_data = (const float*)param.weight()->data();
    const float* bias_data = (const float*)param.bias()->data();

    for (int n = 0; n < out_num; ++n) {
        for (int g = 0; g < param.group; ++g) {
            for (int oc = 0; oc < out_c_group; ++oc) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {

                        int out_idx = n * param.group * out_c_group * out_h * out_w 
                                    + g * out_c_group * out_h * out_w
                                    + oc * out_h * out_w 
                                    + oh * out_w 
                                    + ow;

                        float bias_d = with_bias ? bias_data[g * out_c_group + oc] : 0.f;
                        out_data[out_idx] = bias_d + out_data[out_idx] * beta;

                        for (int ic = 0; ic < in_c_group; ++ic) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    
                                    const float* offset_data_ptr = offset_data
                                                + n * param.group * 2 * kernel_h * kernel_w * out_h * out_w
                                                + g * 2 * kernel_h * kernel_w * out_h * out_w;
                                    const int data_offset_h_ptr = ((2 * (kh * kernel_w + kw))
                                                            * out_h + oh) * out_w + ow;
                                    const int data_offset_w_ptr = ((2 * (kh * kernel_w + kw) + 1)
                                                            * out_h + oh) * out_w + ow;
                                    const float offset_h = offset_data_ptr[data_offset_h_ptr];
                                    const float offset_w = offset_data_ptr[data_offset_w_ptr];
                                    
                                    float val = 0.f;

                                    const float iw = ow * stride_w - pad_w + kw * dilation_w + offset_w;
                                    const float ih = oh * stride_h - pad_h + kh * dilation_h + offset_h;
                                    
                                    if (iw >= 0 && iw < in_w && ih >= 0 && ih < in_h) {

                                        const float map_h = kh * dilation_h + offset_h;
                                        const float map_w = kw * dilation_w + offset_w;
                                        const int cur_height = in_h - (oh * stride_h - pad_h);
                                        const int cur_width = in_w - (ow * stride_w - pad_w);

                                        const float* in_data_offset = in_data 
                                                                + n * param.group * in_c_group * in_h * in_w
                                                                + (g * in_c_group + ic) * in_h * in_w 
                                                                + (oh * stride_h - pad_h) * in_w 
                                                                + (ow * stride_w - pad_w);

                                        val = deformable_bilinear(in_data_offset, 
                                                                in_w, cur_height, cur_width, map_h, map_w);
                
                                        int widx = g * out_c_group * in_c_group * kernel_h * kernel_w
                                                + oc * in_c_group * kernel_h * kernel_w
                                                + ic * kernel_h * kernel_w
                                                + kh * kernel_w
                                                + kw;

                                        out_data[out_idx] += alpha * val * weights_data[widx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return SaberSuccess;
}

template class SaberDeformableConv2D<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberDeformableConv2D, DeformableConvParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberDeformableConv2D, DeformableConvParam, X86, AK_INT8);
    
}
}