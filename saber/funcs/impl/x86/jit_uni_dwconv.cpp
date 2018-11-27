#include "saber/funcs/impl/x86/jit_uni_dwconv.h"
#include "saber/funcs/impl/x86/jit_call_conf.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include <iostream>

namespace anakin {
namespace saber {

using namespace jit;

template <>
SaberStatus JitUniDWConv<AK_FLOAT>::check_conf(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvParam<X86> &param) {
    ConvParam<X86> *conv_param = &(param);
    const Tensor<X86> *weights = conv_param->weight();
    const Tensor<X86> *bias = conv_param->bias();
    const jit_conv_conf_t jcp = kernel->jcp;
    Tensor<X86> *input = inputs[0];
    Tensor<X86> *output = outputs[0];

    // check format
//    if (!(std::is_same<LayOutType_in, NCHW_C16>::value &&
//          std::is_same<LayOutType_out, NCHW_C16>::value &&
//          std::is_same<LayOutType_op, NCHW>::value &&
//          inDtype == AK_FLOAT)) {
//                LOG(ERROR) << "wrong format";
//        return SaberUnImplError;
//    }
    if ((inputs[0]->get_layout() != Layout_NCHW_C16)
        || (outputs[0]->get_layout() != Layout_NCHW_C16)
        || (param.weight()->get_layout() != Layout_NCHW)) {

                LOG(ERROR) << "wrong format";
        return SaberUnImplError;
    }

    // check param
    bool param_ok = true
                    && jcp.t_pad == conv_param->pad_h
                    && jcp.l_pad == conv_param->pad_w
                    && jcp.b_pad == conv_param->pad_h
                    && jcp.r_pad == conv_param->pad_w
                    && jcp.stride_h == conv_param->stride_h
                    && jcp.stride_w == conv_param->stride_w
                    && jcp.dilate_h == conv_param->dilation_h
                    && jcp.dilate_w == conv_param->dilation_w;

    // check shape
    bool shape_ok = true
                    && jcp.kh == weights->height()
                    && jcp.kw == weights->width()
                    && jcp.ngroups == weights->num()
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

template <>
SaberStatus JitUniDWConv<AK_FLOAT>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvParam<X86> &param,
        Context<X86> &ctx) {
    SaberStatus status;
    ConvParam<X86> *conv_param = &(param);
    ActivationParam<X86> *act_param = nullptr;
    const Tensor<X86> *weights = conv_param->weight();
    const Tensor<X86> *bias = conv_param->bias();
    Tensor<X86> *input = inputs[0];
    Tensor<X86> *output = outputs[0];

    // check conf
    if (kernel) {
        status = check_conf(inputs, outputs, param);
        if (status != SaberNotInitialized) {
            return status;
        }
    }

    // init conf
    conf.ngroups = weights->num();
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
    conf.b_pad = conv_param->pad_h;
    conf.r_pad = conv_param->pad_w;
    conf.dilate_h = conv_param->dilation_h;
    conf.dilate_w = conv_param->dilation_w;

    conf.with_bias = (bias != NULL);
    conf.with_relu = param.activation_param.has_active;
    if (conf.with_relu) {
        act_param = &(param.activation_param);
        conf.relu_negative_slope = static_cast<float>(act_param->negative_slope);
    }

    conf.is_dw = (conf.oc / conf.ngroups == weights->channel());
    bool ok = true
              && conf.oc == conf.ngroups
              && conf.ic == conf.ngroups
              && conf.is_dw;
    if (!ok) {
                LOG(ERROR) << "dw conv init fail, return UnImplError";
        return SaberUnImplError;
    }

    status = jit_uni_dwconv_kernel_f32<avx512_common>::init_conf(conf);
    if (status == SaberSuccess) {
        if (kernel != nullptr) {
            delete kernel;
            kernel = nullptr;
        }
        kernel = new jit_uni_dwconv_kernel_f32<avx512_common>(conf);
    } else {
        return SaberUnImplError;
    }

    // reorder weights
    Tensor<X86> *weights_reorder = conv_param->mutable_weight();
    weights_internal.reset(new Tensor<X86>(weights_reorder->valid_shape()));
    weight_reorder_Goihw16g(*weights_reorder, *weights_internal);

    return status;
}

template <>
SaberStatus JitUniDWConv<AK_FLOAT>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvParam<X86> &param,
        Context<X86> &ctx) {

//    if (!(std::is_same<LayOutType_in, NCHW_C16>::value &&
//          std::is_same<LayOutType_out, NCHW_C16>::value &&
//          std::is_same<LayOutType_op, NCHW>::value &&
//          OpDtype == AK_FLOAT)) {
//        return SaberUnImplError;
//    }
    if ((inputs[0]->get_layout() != Layout_NCHW_C16)
        || (outputs[0]->get_layout() != Layout_NCHW_C16)
        || (param.weight()->get_layout() != Layout_NCHW)) {

            LOG(ERROR) << "wrong format";
        return SaberUnImplError;
    }
    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus JitUniDWConv<AK_FLOAT>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvParam<X86> &param) {

    ConvParam<X86> *conv_param = &(param);
    const Tensor<X86> *bias = conv_param->bias();

    const float *ptr_src = reinterpret_cast<const float*>(inputs[0]->data());
    const float *ptr_weights = reinterpret_cast<const float*>(weights_internal->data());
    const float *ptr_bias = reinterpret_cast<const float*>(bias->data());
    auto ptr_dst = reinterpret_cast<float*>(outputs[0]->mutable_data());

    const auto &jcp = kernel->jcp;

    int dil_h = jcp.dilate_h + 1;
    int dil_w = jcp.dilate_w + 1;
    int str_h = jcp.stride_h;
    int str_w = jcp.stride_w;

    int MB = jcp.mb;
    int chb_work = utils::div_up(jcp.nb_ch, jcp.nb_ch_blocking);
    const size_t work_amount = MB * chb_work * jcp.oh;

    auto kernel_params = [&](int ur_w_step, int ow, int oh, int ih, int kh,
                             int kh_padding, int ch, int ch_num, int n) {
        jit_conv_call_t par_conv;

        const int i_l_overflow = utils::max(0, (jcp.l_pad - ow * str_w));
        const int i_r_overflow = utils::max(jcp.iw, (ow * str_w
                                                     + (jcp.kw - 1)*dil_w - jcp.l_pad + 1)) - jcp.iw;

        const int iw = utils::max((ow*str_w - jcp.l_pad
                                   + utils::div_up(i_l_overflow, dil_w)*dil_w), 0);
        const int kw = utils::div_up(i_l_overflow, dil_w);

        const int kw_padding = jcp.kw - utils::div_up(i_l_overflow, dil_w)
                               - utils::div_up(i_r_overflow, dil_w);

        par_conv.src = ptr_src + n * jcp.ic * jcp.iw * jcp.ih + ch * jcp.iw * jcp.ih * 16 + ih * jcp.iw * 16 + iw * 16;
        par_conv.dst = ptr_dst + n * jcp.oc * jcp.ow * jcp.oh + ch * jcp.ow * jcp.oh * 16 + oh * jcp.ow * 16 + ow * 16;

        //par_conv.filt = &weights[weights_d.blk_off(ch, 0, 0, kh, kw)];
        par_conv.filt = ptr_weights + (ch * jcp.kh * jcp.kw + kh * jcp.kw + kw) *16;
        if (bias) {
            par_conv.bias = ptr_bias + ch * jcp.ch_block;
        }

        par_conv.kh_padding = (size_t)utils::max(0, kh_padding);
        par_conv.kw_padding = (size_t)utils::max(0, kw_padding);

        par_conv.ur_w = (size_t)ur_w_step;

        par_conv.ch_blocks = utils::min(ch + ch_num, jcp.nb_ch) - ch;

        return par_conv;
    };

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        utils::balance211(work_amount, nthr, ithr, start, end);

        size_t n{0}, chb{0}, oh{0};
        utils::nd_iterator_init(start, n, MB, chb, chb_work, oh, jcp.oh);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int ch = chb * jcp.nb_ch_blocking;
            int ch_num = jcp.nb_ch_blocking;

            const int i_t_overflow = utils::max(0, (int)(jcp.t_pad - oh*str_h));
            const int i_b_overflow = utils::max(jcp.ih,
                                                (int)(oh*str_h + (jcp.kh - 1)*dil_h - jcp.t_pad + 1)) - jcp.ih;

            const int ih = utils::max((int)(oh*str_h - jcp.t_pad
                                            + utils::div_up(i_t_overflow, dil_h)*dil_h), 0);
            const int kh = utils::div_up(i_t_overflow, dil_h);
            const int kh_padding = jcp.kh - utils::div_up(i_t_overflow, dil_h)
                                   - utils::div_up(i_b_overflow, dil_h);

            // left border
            int ow = 0;
            int l_border = utils::min(utils::div_up(jcp.l_pad, str_w), jcp.ow);
            int ur_w_step = 1;
            for (; ow < l_border; ow++) {
                jit_conv_call_t par_conv = kernel_params(ur_w_step, ow, oh, ih,
                                                         kh, kh_padding, ch, ch_num, n);

                kernel->jit_ker(&par_conv);
            }

            // main loop
            ur_w_step = (jcp.iw - (jcp.kw - 1)*dil_w + jcp.l_pad - 1) / jcp.stride_w - ow + 1;
            if (ur_w_step > 0) {
                jit_conv_call_t par_conv = kernel_params(ur_w_step, ow, oh, ih,
                                                         kh, kh_padding, ch, ch_num, n);

                kernel->jit_ker(&par_conv);

                ow += ur_w_step;
            }

            // right border
            ur_w_step = 1;
            for (; ow < jcp.ow; ow++) {
                jit_conv_call_t par_conv = kernel_params(ur_w_step, ow, oh, ih,
                                                         kh, kh_padding, ch, ch_num, n);

                kernel->jit_ker(&par_conv);
            }

            utils::nd_iterator_step(n, MB, chb, chb_work, oh, jcp.oh);
        }
    };

#pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }

    return SaberSuccess;
}

template class JitUniDWConv<AK_FLOAT>;
} // namespace saber
} // namespace anakin
