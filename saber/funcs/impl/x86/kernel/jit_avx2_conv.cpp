#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_conv_kernel.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_conv.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {

using namespace jit;

using jit_conv_ker_t = void (*)(jit_conv_call_t *);

inline void jit_conv_ker_pipeline(jit_conv_ker_t ker, jit_conv_call_t &p,
                                  const void *src, const void *dst,
                                  const void *filt, const void *bias,
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
        ConvEltwiseParam<X86> &param) {

    ConvParam<X86> *conv_param = &param.conv_param;
    const Tensor<X86> *weights = conv_param->weight();
    const Tensor<X86> *bias = conv_param->bias();
    const jit_conv_conf_t jcp = kernel->jcp;
    Tensor<X86> *input = inputs[0];
    Tensor<X86> *output = outputs[0];

    // check format
    if (((inputs[0]->get_layout() != Layout_NCHW) && (
          inputs[0]->get_layout() != Layout_NCHW_C8))
       || (outputs[0]->get_layout() != Layout_NCHW_C8)
        || (weights->get_layout() != Layout_NCHW)) 
    {
        LOG(ERROR) << "wrong format";
        return SaberUnImplError;
    }

    // check param
    bool param_ok = true
                    && jcp.t_pad == conv_param->pad_h
                    && jcp.l_pad == conv_param->pad_w
                    && jcp.stride_h == conv_param->stride_h
                    && jcp.stride_w == conv_param->stride_w
                    && jcp.dilate_h == conv_param->dilation_h
                    && jcp.dilate_w == conv_param->dilation_w;

    // check shape
    bool shape_ok = true
                    && jcp.kh == weights->height()
                    && jcp.kw == weights->width()
                    && jcp.ngroups == 1
                    && jcp.mb == input->num()
                    && jcp.ic == input->channel()
                    && jcp.ih == input->height()
                    && jcp.iw == input->width()
                    && jcp.oc == output->channel()
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
        ConvEltwiseParam<X86> &param, Context<X86> &ctx) {
    SaberStatus status = SaberSuccess;
    ConvParam<X86> *conv_param = &param.conv_param;
    ActivationParam<X86> *act_param = nullptr;
    const Tensor<X86> *weights = conv_param->weight();
    Tensor<X86> *input = inputs[0];
    Tensor<X86> *output = outputs[0];
    // check conf
    if (kernel) {
        status = check_conf(inputs, outputs, param);
        if(status != SaberNotInitialized) {
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

    conf.kh = weights->height();
    conf.kw = weights->width();
    conf.stride_h = conv_param->stride_h;
    conf.stride_w = conv_param->stride_w;
    conf.t_pad = conv_param->pad_h;
    conf.l_pad = conv_param->pad_w;
    conf.dilate_h = conv_param->dilation_h <= 0 ? 0 : (conv_param->dilation_h - 1);
    conf.dilate_w = conv_param->dilation_w <= 0 ? 0 : (conv_param->dilation_w - 1);

    conf.with_bias = (conv_param->bias()!= NULL);
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
    Tensor<X86> *weights_reorder = conv_param->mutable_weight();

    weights_internal.reset(new Tensor<X86>(weights_reorder->valid_shape()));

    if (inputs[0]->get_layout() == Layout_NCHW) {
        weight_reorder_OIhwi8o(*weights_reorder, *weights_internal);
    } else if (inputs[0]->get_layout() == Layout_NCHW_C8) {
        weight_reorder_OIhw8i8o(*weights_reorder, *weights_internal);
    }

    if (conf.with_bias) {
        Shape bias_s({1,conf.oc,1,1}, Layout_NCHW); 
        bias_internal.reset(new Tensor<X86>(bias_s));
        bias_internal->set_shape(conv_param->bias()->valid_shape(), bias_s);
        bias_internal->copy_from(*conv_param->bias());
    }

    return SaberSuccess;
}

template <>
SaberStatus JitAvx2Conv<AK_FLOAT>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86> &param, Context<X86> &ctx) {

    ConvParam<X86> *conv_param = &param.conv_param;
    if (((inputs[0]->get_layout() != Layout_NCHW) && (
        inputs[0]->get_layout() != Layout_NCHW_C8))
        || (outputs[0]->get_layout() != Layout_NCHW_C8)
        || (conv_param->weight()->get_layout() != Layout_NCHW)) {

        LOG(ERROR) << "wrong format";
        return SaberUnImplError;
    }

    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus JitAvx2Conv<AK_FLOAT>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86> &param) {

    ConvParam<X86> *conv_param = &param.conv_param;
    const Tensor<X86> *bias = conv_param->bias();

    const float *ptr_src = reinterpret_cast<const float*>(inputs[0]->data());
    const float *ptr_weights = reinterpret_cast<const float*>(weights_internal->data());
    const float *ptr_bias = bias? reinterpret_cast<const float*>(bias_internal->data()) : nullptr;
    auto ptr_dst = reinterpret_cast<float*>(outputs[0]->mutable_data());
    const auto &jcp = kernel->jcp;

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

                    par_conv.src = (jcp.src_fmt == Layout_NCHW)? ptr_src + n * jcp.ic * jcp.ih * jcp.iw + 
                                   src_ic * jcp.ih * jcp.iw + ih * jcp.iw : 
                                   ptr_src + n * jcp.ic * jcp.ih * jcp.iw + src_ic * jcp.ih * jcp.iw * 8 
                                   + ih * jcp.iw * 8;
                                   
                    par_conv.dst = ptr_dst + n * jcp.oc * jcp.oh * jcp.ow + _oc * jcp.oh * jcp.ow * 8 
                                   + oh * jcp.ow * 8;
                    
                    const int wh = utils::div_up(i_t_overflow, (jcp.dilate_h + 1));

                    par_conv.filt = ptr_weights + ocb * jcp.ic * jcp.kh * jcp.kw * 8 * 8 + 
                                    wgt_ic * jcp.kh * jcp.kw * 8 * 8 + wh * jcp.kw * 8 * 8;

                    if (icb == 0) {
                        if (bias) {
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

    return SaberSuccess;
}

template class JitAvx2Conv<AK_FLOAT>;


} // namespace saber
} // namespace anakin
