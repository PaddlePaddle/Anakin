#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_conv_kernel.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_conv.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/funcs/impl/x86/saber_normal_activation.h"
#include "debug.h"

namespace anakin {
namespace saber {

using namespace jit;

using jit_conv_ker_t = void (*)(jit_conv_call_t*);

inline void jit_conv_ker_pipeline(jit_conv_ker_t ker, jit_conv_call_t& p,
                                  const void* src, const void* dst,
                                  const void* filt, const void* bias,
                                  int channel, int kh_padding) {
#define PIPELINE(field) \
    do { \
        p.field = p.field ## _prf; \
        p.field ## _prf = field; \
    } while (0)

    PIPELINE(src);
    PIPELINE(dst);
    PIPELINE(filt);
    PIPELINE(bias);
    PIPELINE(channel);
    PIPELINE(kh_padding);

    if (p.src) {
        ker(&p);
    }
}

template <>
SaberStatus JitAvx2Conv<AK_FLOAT>::check_conf(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param) {

    ConvParam<X86>* conv_param = &param.conv_param;
    const Tensor<X86>* weights = conv_param->weight();
    const Tensor<X86>* bias = conv_param->bias();
    const jit_conv_conf_t jcp = kernel->jcp;
    Tensor<X86>* input = inputs[0];
    Tensor<X86>* output = outputs[0];

    // check format
    LayoutType input_layout = inputs[0]->get_layout();
    LayoutType output_layout = outputs[0]->get_layout();
    bool is_layout_ok = (input_layout == Layout_NCHW || input_layout == Layout_NCHW_C8
                         || input_layout == Layout_NCHW_C8R)
                        && (output_layout == Layout_NCHW || output_layout == Layout_NCHW_C8
                            || output_layout == Layout_NCHW_C8R);

    if (!is_layout_ok) {
        LOG(FATAL) << "wrong format layout " << inputs[0]->get_layout() << "," << outputs[0]->get_layout();
        return SaberUnImplError;
    }

    // check param
    bool param_ok = true
                    && jcp.t_pad == conv_param->pad_h
                    && jcp.l_pad == conv_param->pad_w
                    && jcp.stride_h == conv_param->stride_h
                    && jcp.stride_w == conv_param->stride_w
                    && jcp.dilate_h == conv_param->dilation_h - 1
                    && jcp.dilate_w == conv_param->dilation_w - 1;
//    LOG(INFO) << "jcp.t_pad " << jcp.t_pad << "," << conv_param->pad_h;
    // check shape
    bool shape_ok = true
                    && jcp.kh == weights->height()
                    && jcp.kw == weights->width()
                    && jcp.ngroups == 1
                    && jcp.mb == input->num()
                    && jcp.ic == utils::round_up(input->channel(), 8)
                    && jcp.ih == input->height()
                    && jcp.iw == input->width()
                    && jcp.oc == utils::round_up(output->channel(), 8)
                    && jcp.oh == output->height()
                    && jcp.ow == output->width();

    if (param_ok && shape_ok) {
        return SaberSuccess;
    } else {
        LOG(INFO) << "param or shape changed, re-init kernel";
        return SaberNotInitialized;
    }
}

template<>
SaberStatus JitAvx2Conv<AK_FLOAT>::create(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param, Context<X86>& ctx) {
    DLOG(INFO) << "input layout " << inputs[0]->get_layout() << " , output layout " <<
               outputs[0]->get_layout();
    SaberStatus status = SaberSuccess;
    ConvParam<X86>* conv_param = &param.conv_param;
    ActivationParam<X86>* act_param = nullptr;
    const Tensor<X86>* weights = conv_param->weight();
    Tensor<X86>* input = inputs[0];
    Tensor<X86>* output = outputs[0];

    // check conf
    if (kernel) {
        status = check_conf(inputs, outputs, param);

        if (status != SaberNotInitialized) {
            return status;
        }
    }

    // init conf
    conf.src_fmt = input->get_layout();
    conf.ngroups = 1;
    conf.mb = input->num();
    conf.ic = input->channel();
    conf.ih = input->height();
    conf.iw = input->width();

    conf.oc = output->channel();
    conf.oh = output->height();
    conf.ow = output->width();

    if (input->get_layout() == Layout_NCHW_C8R) {
        conf.ic = utils::round_up(input->channel(), 8);
        conf.src_fmt = Layout_NCHW_C8;
        DLOG(INFO) << "input->get_layout == Layout_NCHW_C8R";
    }

    if (output->get_layout() == Layout_NCHW_C8R) {
        conf.oc = utils::round_up(output->channel(), 8);
    }

    DLOG(INFO) << "oc = " << conf.oc << ", ic = " << conf.ic;

    conf.kh = weights->height();
    conf.kw = weights->width();
    conf.stride_h = conv_param->stride_h;
    conf.stride_w = conv_param->stride_w;
    conf.t_pad = conv_param->pad_h;
    conf.l_pad = conv_param->pad_w;
    conf.dilate_h = conv_param->dilation_h <= 0 ? 0 : (conv_param->dilation_h - 1);
    conf.dilate_w = conv_param->dilation_w <= 0 ? 0 : (conv_param->dilation_w - 1);

    conf.with_sum = false;
   
    if (param.eltwise_param.has_eltwise){
        conf.with_sum = true;    
    }
    conf.with_bias = (conv_param->bias() != NULL)&&(conv_param->bias()->valid_size()>0);
    conf.with_relu = conv_param->activation_param.has_active;

    if (conf.with_relu) {
        act_param = &(conv_param->activation_param);
        conf.relu_negative_slope = act_param->negative_slope;
    }

    status = jit_avx2_conv_act_kernel::init_conf(conf);

    if (status == SaberSuccess) {
        if (kernel != nullptr) {
            delete kernel;
            kernel = nullptr;
        }

        kernel = new jit_avx2_conv_act_kernel(this->conf);
    } else {
        return SaberUnImplError;
    }

    // reorder weights
    Tensor<X86>* weights_reorder = conv_param->mutable_weight();

    weights_internal.reset(new Tensor<X86>(weights_reorder->valid_shape()));

    if (inputs[0]->get_layout() == Layout_NCHW) {
        weight_reorder_OIhwi8o(*weights_reorder, *weights_internal);
    } else if (inputs[0]->get_layout() == Layout_NCHW_C8
               || inputs[0]->get_layout() == Layout_NCHW_C8R) {
        weight_reorder_OIhw8i8o(*weights_reorder, *weights_internal);
    }

    if (conf.with_bias) {
        Shape bias_s({1, conf.oc, 1, 1}, Layout_NCHW);
        bias_internal.reset(new Tensor<X86>(bias_s));
        bias_internal->set_shape(conv_param->bias()->valid_shape(), bias_s);
        bias_internal->copy_from(*conv_param->bias());
    }

    if (outputs[0]->get_layout() == Layout_NCHW) {
        Shape shape = outputs[0]->valid_shape();
        int n_value = shape[0], c_value = shape[1], h_value = shape[2], w_value = shape[3];
        Shape new_shape({n_value, utils::round_up(c_value, 8) / 8, h_value, w_value, 8}, Layout_NCHW_C8);
        _temp_output.reshape(new_shape);
    }

    return SaberSuccess;
}

template <>
SaberStatus JitAvx2Conv<AK_FLOAT>::init(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param, Context<X86>& ctx) {

    ConvParam<X86>* conv_param = &param.conv_param;
    LayoutType input_layout = inputs[0]->get_layout();
    LayoutType output_layout = outputs[0]->get_layout();
    bool is_layout_ok = (input_layout == Layout_NCHW || input_layout == Layout_NCHW_C8
                         || input_layout == Layout_NCHW_C8R)
                        && (output_layout == Layout_NCHW || output_layout == Layout_NCHW_C8
                            || output_layout == Layout_NCHW_C8R);

    if (!is_layout_ok) {
        LOG(FATAL) << "wrong format layout " << inputs[0]->get_layout() << "," << outputs[0]->get_layout();
        return SaberUnImplError;
    }


    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

void conv_basic_check(Tensor<X86>& tensor_in, Tensor<X86>& tensor_out,
                      const float* weights, const float* bias, int group,
                      int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
                      int pad_w, int pad_h, bool flag_bias, bool flag_relu, float beta = 0.f) {

    auto src_data = reinterpret_cast<const float*>(tensor_in.data());
    auto dst_data_ref = reinterpret_cast<float*>(tensor_out.mutable_data());
    Tensor<X86> bk;
    bk.re_alloc(tensor_out.valid_shape(), AK_FLOAT);
    bk.copy_from(tensor_out);
    auto weights_data = weights;
    bool with_bias = flag_bias;
    auto bias_data = bias;

    int in_num = tensor_out.num();
    int out_channels = tensor_out.channel();
    int out_h = tensor_out.height();
    int out_w = tensor_out.width();

    int in_channel = tensor_in.channel();
    int in_h = tensor_in.height();
    int in_w = tensor_in.width();
    int out_c_group = out_channels / group;
    int in_c_group = in_channel / group;
    #pragma omp parallel for num_threads(8) collapse(5) schedule(static)

    for (int n = 0; n < in_num; ++n) {
        for (int g = 0; g < group; ++g) {
            for (int oc = 0; oc < out_c_group; ++oc) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        int out_idx = n * group * out_c_group * out_h * out_w + g * out_c_group * out_h * out_w
                                      + oc * out_h * out_w + oh * out_w + ow;
                        float bias_d = with_bias ? (float)(bias_data[g * out_c_group + oc]) : 0.f;
                        dst_data_ref[out_idx] = bias_d + dst_data_ref[out_idx] * beta;

                        for (int ic = 0; ic < in_c_group; ++ic) {
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    int iw = ow * stride_w - pad_w + kw * (dilation_w);
                                    int ih = oh * stride_h - pad_h + kh * (dilation_h);

                                    if (iw < 0 || iw >= in_w) {
                                        continue;
                                    }

                                    if (ih < 0 || ih >= in_h) {
                                        continue;
                                    }

                                    int iidx = n * in_channel * in_h * in_w
                                               + g * in_c_group * in_h * in_w
                                               + ic * in_h * in_w
                                               + ih * in_w
                                               + iw;
                                    int widx = g * out_c_group * in_c_group * kernel_h * kernel_w
                                               + oc * in_c_group * kernel_h * kernel_w
                                               + ic * kernel_h * kernel_w
                                               + kh * kernel_w
                                               + kw;

                                    dst_data_ref[out_idx]
                                    += src_data[iidx]
                                       * weights_data[widx];
                                }
                            }
                        }

                        if (flag_relu) {
                            dst_data_ref[out_idx] = dst_data_ref[out_idx] > 0.f ? dst_data_ref[out_idx] : 0.f;
                        }
                    }
                }
            }
        }
    }
}

static inline void conv_basic_check_nchwc(const float* src_data, float* dst_data_ref, int in_num,
        int in_channel, int in_h, int in_w,
        int out_channels, int out_h, int out_w,
        const float* weights, const float* bias,
        int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
        int pad_w, int pad_h, bool flag_bias, bool flag_relu) {

    //    #pragma omp parallel for num_threads(8) collapse(5) schedule(static)
    int in_channel_div8 = utils::div_up(in_channel, 8);
    int out_channel_div8 = utils::div_up(out_channels, 8);

    for (int n = 0; n < in_num; ++n) {
        for (int oc = 0; oc < out_channel_div8; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int out_idx =   n  * out_channel_div8 * out_h * out_w * 8
                                    + oc * out_h * out_w * 8
                                    + oh * out_w * 8
                                    + ow * 8;
                    float result[8] = {0.f};

                    if (flag_bias) {
                        for (int i = 0; i < 8; i++) {
                            result[i] = bias[oc * 8 + i];
                        }
                    }

                    for (int ic = 0; ic < in_channel_div8; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int iw = ow * stride_w - pad_w + kw * (dilation_w);
                                int ih = oh * stride_h - pad_h + kh * (dilation_h);

                                if (iw < 0 || iw >= in_w) {
                                    continue;
                                }

                                if (ih < 0 || ih >= in_h) {
                                    continue;
                                }

                                for (int inner_oc = 0; inner_oc < 8; inner_oc++) {
                                    for (int inner_ic = 0; inner_ic < 8; inner_ic++) {

                                        int iidx = n * in_channel_div8 * in_h * in_w * 8
                                                   + ic * in_h * in_w * 8
                                                   + ih * in_w * 8
                                                   + iw * 8 + inner_ic;
                                        int widx = oc * in_channel_div8 * kernel_h * kernel_w * 8 * 8
                                                   + ic * kernel_h * kernel_w * 8 * 8
                                                   + kh * kernel_w * 8 * 8
                                                   + kw * 8 * 8
                                                   + inner_ic * 8 + inner_oc;

                                        result[inner_oc]
                                        += src_data[iidx]
                                           * weights[widx];

                                    }
                                }
                            }
                        }
                    }

                    for (int inner_oc = 0; inner_oc < 8; inner_oc++) {
                        if (flag_relu) {
                            dst_data_ref[out_idx + inner_oc] = result[inner_oc] > 0.f ? result[inner_oc] : 0.f;
                        } else {
                            dst_data_ref[out_idx + inner_oc] = result[inner_oc];
                        }
                    }

                }
            }
        }
    }
}
#if defined(__AVX2__) and defined(__FMA__)
static inline void conv_basic_check_nchwc_avx2(const float* src_data, float* dst_data_ref,
        int in_num,
        int in_channel, int in_h, int in_w,
        int out_channels, int out_h, int out_w,
        const float* weights, const float* bias,
        int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
        int pad_w, int pad_h, bool flag_bias, bool flag_relu) {

    //    #pragma omp parallel for num_threads(8) collapse(5) schedule(static)
    int in_channel_div8 = utils::div_up(in_channel, 8);
    int out_channel_div8 = utils::div_up(out_channels, 8);

    for (int n = 0; n < in_num; ++n) {
        for (int oc = 0; oc < out_channel_div8; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int out_idx =   n  * out_channel_div8 * out_h * out_w * 8
                                    + oc * out_h * out_w * 8
                                    + oh * out_w * 8
                                    + ow * 8;
                    __m256 result = _mm256_setzero_ps();

                    if (flag_bias) {
                        result = _mm256_loadu_ps(bias + oc * 8);
                    }

                    for (int ic = 0; ic < in_channel_div8; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int iw = ow * stride_w - pad_w + kw * (dilation_w);
                                int ih = oh * stride_h - pad_h + kh * (dilation_h);

                                if (iw < 0 || iw >= in_w) {
                                    continue;
                                }

                                if (ih < 0 || ih >= in_h) {
                                    continue;
                                }

                                const float* inpute_base = src_data + n * in_channel_div8 * in_h * in_w * 8
                                                           + ic * in_h * in_w * 8
                                                           + ih * in_w * 8
                                                           + iw * 8;
                                __m256 input_8 = _mm256_loadu_ps(inpute_base);
                                //                                LOG(INFO)<<":::"<<ih<<","<<iw<<","<<ic;
                                //                                printf_intrin_var(input_8);
                                const float* weight_base = weights + oc * in_channel_div8 * kernel_h * kernel_w * 8 * 8
                                                           + ic * kernel_h * kernel_w * 8 * 8
                                                           + kh * kernel_w * 8 * 8
                                                           + kw * 8 * 8;

                                for (int inner_ic = 0; inner_ic < 8; inner_ic++) {
                                    __m256 weight_8 = _mm256_loadu_ps(weight_base + inner_ic * 8);
                                    __m256 base = _mm256_set1_ps(input_8[inner_ic]);
                                    result = _mm256_fmadd_ps(base, weight_8, result);
                                    //                                    printf_intrin_var(input_8);
                                    //                                    printf_intrin_var(weight_8);
                                    //                                    printf_intrin_var(result);
                                    //                                    LOG(INFO)<<"-------";
                                }
                            }
                        }
                    }

                    if (flag_relu) {
                        _mm256_storeu_ps(&dst_data_ref[out_idx ], _mm256_max_ps(_mm256_setzero_ps(), result));
                    } else {
                        _mm256_storeu_ps(&dst_data_ref[out_idx ], result);
                    }

                    //                    exit(0);
                }
            }
        }
    }
}

static inline void conv_basic_check_nchwc_avx2_conv_1x1(const float* src_data, float* dst_data_ref,
        int in_num,
        int in_channel, int in_h, int in_w,
        int out_channels, int out_h, int out_w,
        const float* weights, const float* bias,
        int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
        int pad_w, int pad_h, bool flag_bias, bool flag_relu) {

    //    #pragma omp parallel for num_threads(8) collapse(5) schedule(static)
    int in_channel_div8 = utils::div_up(in_channel, 8);
    int out_channel_div8 = utils::div_up(out_channels, 8);

    for (int n = 0; n < in_num; ++n) {
        for (int oc = 0; oc < out_channel_div8; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int out_idx =   n  * out_channel_div8 * out_h * out_w * 8
                                    + oc * out_h * out_w * 8
                                    + oh * out_w * 8
                                    + ow * 8;
                    __m256 result = _mm256_setzero_ps();

                    if (flag_bias) {
                        result = _mm256_loadu_ps(bias + oc * 8);
                    }

                    for (int ic = 0; ic < in_channel_div8; ++ic) {
                        const float* weight_base = weights + oc * in_channel_div8 * kernel_h * kernel_w * 8 * 8
                                                   + ic * 8 * 8;
                        int iw = ow;
                        int ih = oh;

                        const float* inpute_base = src_data + n * in_channel_div8 * in_h * in_w * 8
                                                   + ic * in_h * in_w * 8
                                                   + ih * in_w * 8
                                                   + iw * 8;
                        __m256 input_8 = _mm256_loadu_ps(inpute_base);

                        for (int inner_ic = 0; inner_ic < 8; inner_ic++) {
                            __m256 weight_8 = _mm256_loadu_ps(weight_base + inner_ic * 8);
                            __m256 base = _mm256_set1_ps(input_8[inner_ic]);
                            result = _mm256_fmadd_ps(base, weight_8, result);
                        }
                    }

                    if (flag_relu) {
                        _mm256_storeu_ps(&dst_data_ref[out_idx ], _mm256_max_ps(_mm256_setzero_ps(), result));
                    } else {
                        _mm256_storeu_ps(&dst_data_ref[out_idx ], result);
                    }

                    //                    exit(0);
                }
            }
        }
    }
}

static inline void conv_basic_check_nchwc_avx2_h4(const float* src_data, float* dst_data_ref,
        int in_num,
        int in_channel, int in_h, int in_w,
        int out_channels, int out_h, int out_w,
        const float* weights, const float* bias,
        int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h,
        int pad_w, int pad_h, bool flag_bias, bool flag_relu) {

    //    #pragma omp parallel for num_threads(8) collapse(5) schedule(static)
    int in_channel_div8 = utils::div_up(in_channel, 8);
    int out_channel_div8 = utils::div_up(out_channels, 8);

    for (int n = 0; n < in_num; ++n) {
        for (int oc = 0; oc < out_channel_div8; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w / 4; ++ow) {
                    int out_idx =   n  * out_channel_div8 * out_h * out_w * 8
                                    + oc * out_h * out_w * 8
                                    + oh * out_w * 8;

                    __m256 result[4];

                    if (flag_bias) {
                        result[0] = result[1] = result[2] = result[3] = _mm256_loadu_ps(bias + oc * 8);
                    } else {
                        result[0] = result[1] = result[2] = result[3] = _mm256_setzero_ps();
                    }

                    for (int ic = 0; ic < in_channel_div8; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                const float* weight_base =
                                    weights + oc * in_channel_div8 * kernel_h * kernel_w * 8 * 8
                                    + ic * kernel_h * kernel_w * 8 * 8
                                    + kh * kernel_w * 8 * 8
                                    + kw * 8 * 8;
                                __m256 weights_8[8];

                                for (int inner_ic = 0; inner_ic < 8; inner_ic++) {
                                    weights_8[inner_ic] = _mm256_loadu_ps(weight_base + inner_ic * 8);
                                }

                                for (int inner_ow = 0; inner_ow < 4; inner_ow++) {
                                    int iw = (ow * 4 + inner_ow) * stride_w - pad_w + kw * (dilation_w);
                                    int ih = oh * stride_h - pad_h + kh * (dilation_h);

                                    if (iw < 0 || iw >= in_w) {
                                        continue;
                                    }

                                    if (ih < 0 || ih >= in_h) {
                                        continue;
                                    }

                                    const float* inpute_base = src_data + n * in_channel_div8 * in_h * in_w * 8
                                                               + ic * in_h * in_w * 8
                                                               + ih * in_w * 8
                                                               + iw * 8;
                                    //                                LOG(INFO)<<":::"<<ih<<","<<iw<<","<<ic;
                                    //                                printf_intrin_var(input_8);

                                    for (int inner_ic = 0; inner_ic < 8; inner_ic++) {
                                        __m256 base = _mm256_set1_ps(inpute_base[inner_ic]);
                                        result[inner_ow] = _mm256_fmadd_ps(base, weights_8[inner_ic], result[inner_ow]);
                                        //                                    printf_intrin_var(input_8);
                                        //                                    printf_intrin_var(weight_8);
                                        //                                    printf_intrin_var(result);
                                        //                                    LOG(INFO)<<"-------";
                                    }
                                }
                            }
                        }
                    }

                    for (int inner_ow = 0; inner_ow < 4; inner_ow++) {
                        if (flag_relu) {
                            _mm256_storeu_ps(&dst_data_ref[out_idx + (ow * 4 + inner_ow) * 8],
                                             _mm256_max_ps(_mm256_setzero_ps(), result[inner_ow]));
                        } else {
                            _mm256_storeu_ps(&dst_data_ref[out_idx + (ow * 4 + inner_ow) * 8], result[inner_ow]);
                        }
                    }

                    //                    exit(0);
                }
            }
        }
    }
}
#endif

template <>
SaberStatus JitAvx2Conv<AK_FLOAT>::dispatch(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param) {


    ConvParam<X86>* conv_param = &param.conv_param;

    bool with_bias=(conv_param->bias() != NULL)&&(conv_param->bias()->valid_size()>0);


    const float* ptr_src = reinterpret_cast<const float*>(inputs[0]->data());
    const float* ptr_weights = reinterpret_cast<const float*>(weights_internal->data());
    const float* ptr_bias = (conv_param->bias() != NULL)&&(conv_param->bias()->valid_size()>0) ? reinterpret_cast<const float*>(bias_internal->data()) : nullptr;
    float* ptr_dst = nullptr;

    //    if(inputs[0]->get_layout()==Layout_NCHW_C8R&&outputs[0]->get_layout()==Layout_NCHW_C8R){
    ////        Shape in_nchw=inputs[0]->valid_shape();
    ////        in_nchw.set_layout_without_shape(Layout_NCHW);
    ////        Tensor<X86> temp_in(in_nchw);
    ////        Shape out_nchw=outputs[0]->valid_shape();
    ////        out_nchw.set_layout_without_shape(Layout_NCHW);
    ////        Tensor<X86> temp_out(out_nchw);
    ////        reorder_nchwc8_nchw(*inputs[0],temp_in);
    ////        conv_basic_check(temp_in,temp_out, static_cast<float*>(conv_param->weight()->data()),
    ////            static_cast<float*>(conv_param->bias()->data()),conv_param->group,conv_param->weight()->width(),
    ////            conv_param->weight()->height(),conv_param->stride_w,conv_param->stride_h,conv_param->dilation_w,
    ////            conv_param->dilation_h,conv_param->pad_w,conv_param->pad_h,conv_param->bias()!=nullptr,
    ////                         conv_param->activation_param.active==Active_relu,0);
    ////        input_reorder_nChwc8(temp_out,*outputs[0]);
    //
    ////        LOG(INFO)<<inputs[0]->valid_shape()<<",out = "<<outputs[0]->valid_shape();
    ////        weight_reorder_nchw2nchw8o8i(*conv_param->mutable_weight(),*weights_internal);
    //        conv_basic_check_nchwc_avx2_h4(ptr_src,reinterpret_cast<float*>(outputs[0]->mutable_data()),inputs[0]->num(),inputs[0]->channel(),inputs[0]->height(),
    //                               inputs[0]->width(),outputs[0]->channel(),outputs[0]->height(),outputs[0]->width(),
    //                               ptr_weights,ptr_bias,conv_param->weight()->width(),
    //                               conv_param->weight()->height(),conv_param->stride_w,conv_param->stride_h,conv_param->dilation_w,
    //                               conv_param->dilation_h,conv_param->pad_w,conv_param->pad_h,conv_param->bias()!=nullptr, conv_param->activation_param.active==Active_relu);
    //        return SaberSuccess;
    //
    //    }


    if (outputs[0]->get_layout() == Layout_NCHW) {
        ptr_dst = reinterpret_cast<float*>(_temp_output.mutable_data());
    } else {
        ptr_dst = reinterpret_cast<float*>(outputs[0]->mutable_data());
    }

    DLOG(INFO) << "input layout " << inputs[0]->get_layout() << " , output layout " <<
               outputs[0]->get_layout() << "," << anakin_get_thread_num() << "," << anakin_get_num_threads() << "::" <<
               conf.with_relu << "," << conf.with_bias;
    const auto& jcp = kernel->jcp;

    int ocb_work = utils::div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const size_t work_amount = jcp.mb * jcp.ngroups * ocb_work * jcp.oh;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);
        int icbb = 0;

        while (icbb < jcp.nb_ic) {
            int icb_step = jcp.nb_ic_blocking;
            int icb_step_rem = jcp.nb_ic - icbb;

            if (icb_step_rem < jcp.nb_ic_blocking_max) {
                icb_step = icb_step_rem;
            }

            size_t n{0}, g{0}, ocbb{0}, oh{0};
            nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work, oh, jcp.oh);

            for (size_t iwork = start; iwork < end; ++iwork) {
                int ocb = ocbb * jcp.nb_oc_blocking;
                int ocb_num = jcp.nb_oc_blocking;

                for (int icb = icbb; icb < icbb + icb_step; ++icb) {
                    jit_conv_call_t par_conv;
                    par_conv.flags = 0;
                    const int ij = oh * jcp.stride_h;
                    const int i_t_overflow = utils::max(0, jcp.t_pad - ij);
                    const int i_b_overflow = utils::max(jcp.ih, ij
                                                        + (jcp.kh - 1) * (jcp.dilate_h + 1) - jcp.t_pad + 1) - jcp.ih;

                    const size_t _oc = g * jcp.nb_oc + ocb;
                    const size_t _ic = g * jcp.nb_ic + icb;

                    const int src_ic = jcp.ic == 3 ? 0 : _ic;
                    const int wgt_ic = jcp.ic == 3 ? 0 : icb;

                    const int ih = utils::max(ij - jcp.t_pad + utils::div_up(i_t_overflow,
                                              (jcp.dilate_h + 1)) * (jcp.dilate_h + 1), 0);

                    par_conv.src = (jcp.src_fmt == Layout_NCHW) ? ptr_src + n * jcp.ic * jcp.ih * jcp.iw +
                                   src_ic * jcp.ih * jcp.iw + ih * jcp.iw :
                                   ptr_src + n * jcp.ic * jcp.ih * jcp.iw + src_ic * jcp.ih * jcp.iw * 8
                                   + ih * jcp.iw * 8;

                    par_conv.dst = ptr_dst + n * jcp.oc * jcp.oh * jcp.ow + _oc * jcp.oh * jcp.ow * 8
                                   + oh * jcp.ow * 8;

                    const int wh = utils::div_up(i_t_overflow, (jcp.dilate_h + 1));

                    par_conv.filt = (jcp.src_fmt == Layout_NCHW) ? ptr_weights + ocb * jcp.kh * jcp.kw * jcp.ic * 8 +
                                    wh * jcp.kw * jcp.ic * 8 + wgt_ic * 8 :
                                    ptr_weights + ocb * jcp.ic * jcp.kh * jcp.kw * 8 +
                                    wgt_ic * jcp.kh * jcp.kw * 8 * 8 + wh * jcp.kw * 8 * 8;

                    if (icb == 0) {
                        if (with_bias) {
                            par_conv.bias = ptr_bias +  _oc * 8;
                        }

                        par_conv.flags |= FLAG_IC_FIRST;
                    }

                    if (jcp.with_relu && icb + 1 == jcp.nb_ic) {
                        par_conv.flags |= FLAG_IC_LAST;
                    }

                    par_conv.oc_blocks = utils::min(ocb + ocb_num, jcp.nb_oc) - ocb;
                    par_conv.kw_padding = 0;

                    const int kh_padding = jcp.kh -
                                           utils::div_up(i_t_overflow, (jcp.dilate_h + 1)) -
                                           utils::div_up(i_b_overflow, (jcp.dilate_h + 1));
                    par_conv.kh_padding = utils::max(0, kh_padding);

                    kernel->jit_ker(&par_conv);
                }

                nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work, oh, jcp.oh);
            }

            icbb += icb_step;
        }
    };

    #pragma omp parallel
    {
        ker(anakin_get_thread_num(), anakin_get_num_threads());
    }

    if (outputs[0]->get_layout() == Layout_NCHW) {
        reorder_nchwc8_nchw(_temp_output, *outputs[0]);
    }

    return SaberSuccess;
}

template class JitAvx2Conv<AK_FLOAT>;


} // namespace saber
} // namespace anakin
