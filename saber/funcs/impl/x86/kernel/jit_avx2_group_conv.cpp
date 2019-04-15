#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_group_conv_kernel.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_group_conv.h"
#include "saber/funcs/impl/x86/x86_utils.h"
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
SaberStatus JitAvx2GroupConv<AK_FLOAT>::check_conf(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param) {

    ConvParam<X86>* conv_param = &(param.conv_param);
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

    // check shape
    bool shape_ok = true
                    && jcp.kh == weights->height()
                    && jcp.kw == weights->width()
                    && jcp.ngroups == conv_param->group
                    && jcp.mb == input->num()
                    && jcp.ic == input->channel() / conv_param->group
                    && jcp.ih == input->height()
                    && jcp.iw == input->width()
                    && jcp.oc == output->channel() / conv_param->group
                    && jcp.oh == output->height()
                    && jcp.ow == output->width();

    if (param_ok && shape_ok) {
        return SaberSuccess;
    } else {
        LOG(INFO) << "param or shape changed, re-init kernel";
        return SaberNotInitialized;
    }
}

template <>
SaberStatus JitAvx2GroupConv<AK_FLOAT>::create(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param, Context<X86>& ctx) {
    SaberStatus status = SaberSuccess;
    ConvParam<X86>* conv_param = &(param.conv_param);
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
    conf.ngroups = conv_param->group;
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


    conf.kh = weights->height();
    conf.kw = weights->width();
    conf.stride_h = conv_param->stride_h;
    conf.stride_w = conv_param->stride_w;
    conf.t_pad = conv_param->pad_h;
    conf.l_pad = conv_param->pad_w;
    conf.dilate_h = conv_param->dilation_h <= 0 ? 0 : (conv_param->dilation_h - 1);
    conf.dilate_w = conv_param->dilation_w <= 0 ? 0 : (conv_param->dilation_w - 1);

    conf.with_bias = (conv_param->bias() != NULL)&&(conv_param->bias()->valid_size()>0);
    conf.with_relu = conv_param->activation_param.has_active;
    conf.with_sum = false;

    if (conf.with_relu) {
        act_param = &(conv_param->activation_param);
        conf.relu_negative_slope = act_param->negative_slope;
    }

    status = jit_avx2_group_conv_act_kernel::init_conf(conf);

    if (status == SaberSuccess) {
        if (kernel != nullptr) {
            delete kernel;
            kernel = nullptr;
        }

        kernel = new jit_avx2_group_conv_act_kernel(this->conf);
    } else {
        return SaberUnImplError;
    }

    // reorder weights
    Shape weights_s({conf.oc, conf.ic, conf.kh, conf.kw}, Layout_NCHW);
    Tensor<X86>* weights_reorder = conv_param->mutable_weight();

    weights_internal.clear();

    for (int i = 0; i < conf.ngroups; i++) {
        Tensor<X86> weights_temp(static_cast<float*>(weights_reorder->data()) + i * weights_s.count(),
                                 X86(), 0, weights_s, AK_FLOAT);
        weights_internal.push_back(std::make_shared<Tensor<X86> >(weights_s));

        if (inputs[0]->get_layout() == Layout_NCHW) {
            weight_reorder_OIhwi8o(weights_temp, *(weights_internal.back()));
        } else if (inputs[0]->get_layout() == Layout_NCHW_C8
                   || inputs[0]->get_layout() == Layout_NCHW_C8R) {
            weight_reorder_OIhw8i8o(weights_temp, *(weights_internal.back()));
        }
    }
    LOG(INFO)<<"ready to init bias "<<conf.with_bias;
    if (conf.with_bias) {
        LOG(INFO)<<"init bias";
        Shape bias_s({1, conf.oc * conf.ngroups, 1, 1}, Layout_NCHW);
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
SaberStatus JitAvx2GroupConv<AK_FLOAT>::init(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param, Context<X86>& ctx) {
    ConvParam<X86>* conv_param = &(param.conv_param);
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

template <>
SaberStatus JitAvx2GroupConv<AK_FLOAT>::dispatch(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param) {

    ConvParam<X86>* conv_param = &(param.conv_param);
    bool with_bias=(conv_param->bias() != NULL)&&(conv_param->bias()->valid_size()>0);

    const float* ptr_src = reinterpret_cast<const float*>(inputs[0]->data());
    const float* ptr_bias = with_bias ? reinterpret_cast<const float*>(bias_internal->data()) : nullptr;

    float* ptr_dst = nullptr;

    if (outputs[0]->get_layout() == Layout_NCHW) {
        ptr_dst = reinterpret_cast<float*>(_temp_output.mutable_data());
    } else {
        ptr_dst = reinterpret_cast<float*>(outputs[0]->mutable_data());
    }


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
                const float* ptr_weights = reinterpret_cast<const float*>(weights_internal[g]->data());

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

                    par_conv.src = (jcp.src_fmt == Layout_NCHW) ? ptr_src + n * jcp.ngroups * jcp.ic * jcp.ih * jcp.iw
                                   + src_ic * 8 * jcp.ih * jcp.iw + ih * jcp.iw : ptr_src +
                                   n * jcp.ngroups * jcp.ic * jcp.ih * jcp.iw + src_ic * jcp.ih * jcp.iw * 8
                                   + ih * jcp.iw * 8;

                    par_conv.dst = ptr_dst + n * jcp.ngroups * jcp.oc * jcp.oh * jcp.ow + _oc * jcp.oh * jcp.ow * 8
                                   + oh * jcp.ow * 8;

                    const int wh = utils::div_up(i_t_overflow, (jcp.dilate_h + 1));

                    par_conv.filt = (jcp.src_fmt == Layout_NCHW) ? ptr_weights + ocb * jcp.kh * jcp.kw * jcp.ic * 8 +
                                    wh * jcp.kw * jcp.ic * 8 + wgt_ic * 8 : ptr_weights + ocb * jcp.ic * jcp.kh * jcp.kw * 8
                                    + wgt_ic * jcp.kh * jcp.kw * 8 * 8 + wh * jcp.kw * 8 * 8;

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

template class JitAvx2GroupConv<AK_FLOAT>;


} // namespace saber
} // namespace anakin
