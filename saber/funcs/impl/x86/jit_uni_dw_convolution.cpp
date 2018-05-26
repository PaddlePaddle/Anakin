#include "saber/funcs/impl/x86/jit_uni_dw_convolution.h"
#include "saber/funcs/impl/x86/jit_call_conf.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include <iostream>

namespace anakin {
namespace saber {

using namespace jit;

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus JitUniDWConvolution<OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<inTensor*>& inputs,
        std::vector<outTensor*>& outputs,
        ConvActiveParam<opTensor> &param,
        Context<X86> &ctx) {

    if (!(std::is_same<LayOutType_in, NCHW_C16>::value &&
         std::is_same<LayOutType_out, NCHW_C16>::value &&
         std::is_same<LayOutType_op, NCHW>::value &&
         OpDtype == AK_FLOAT )) {
        return SaberUnImplError;
    }
    
    // get context of uni_dw_convolution
    this->_ctx = ctx;

    ConvParam<opTensor> *conv_param = &(param.conv_param);
    ActivationParam<opTensor> *act_param = &(param.activation_param);

    const opTensor *weights = conv_param->weight();
    const opTensor *bias = conv_param->bias();

    Shape src_shape(inputs[0]->shape());
    Shape dst_shape(outputs[0]->shape());
    Shape weights_shape(weights->shape());

    conf.ngroups = weights_shape[0];
    conf.mb = src_shape[0];
    if (src_shape.dims() == 5) {
        conf.ic = src_shape[1] * src_shape[4];
    }
    else {
        conf.ic = src_shape[1];
    }
    conf.ih = src_shape[2];
    conf.iw = src_shape[3];

    if (src_shape.dims() == 5) {
       conf.oc = dst_shape[1] * dst_shape[4];
    } else {
       conf.oc = dst_shape[1];
    }
    conf.oh = dst_shape[2];
    conf.ow = dst_shape[3];

    conf.kh = weights_shape[2];
    conf.kw = weights_shape[3];

    conf.stride_h = conv_param->stride_h;
    conf.stride_w = conv_param->stride_w;
    conf.t_pad = conv_param->pad_h;
    conf.l_pad = conv_param->pad_w;
    conf.b_pad = conv_param->pad_h;
    conf.r_pad = conv_param->pad_w;
    conf.dilate_h = conv_param->dilation_h;
    conf.dilate_w = conv_param->dilation_w;

    conf.with_bias = (bias != NULL);
    conf.with_relu = param.has_active;
    if (conf.with_relu) {
        conf.relu_negative_slope = static_cast<float>(act_param->negative_slope);
    }

    conf.is_dw = conf.oc / conf.ngroups == weights_shape[1];
    bool ok = true
        && conf.oc == conf.ngroups
        && conf.ic == conf.ngroups
        && conf.is_dw;
    if (!ok) {
        LOG(ERROR) << "dw conv init fail, return UnImplError";
        return SaberUnImplError;
    }

    SaberStatus status = jit_uni_dw_conv_kernel_f32<avx512_common>::init_conf(conf);
    if (status == SaberSuccess) {
        return create(inputs, outputs, param, ctx);
    } else {
        return SaberUnImplError;
    }
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus JitUniDWConvolution<OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<inTensor*>& inputs,
        std::vector<outTensor*>& outputs,
        ConvActiveParam<opTensor> &param,
        Context<X86> &ctx) {

    kernel_ = new jit_uni_dw_conv_kernel_f32<avx512_common>(conf);
    ConvParam<opTensor> *conv_param = &(param.conv_param);
    opTensor *weights = conv_param->mutable_weight();
    weights_internal.reset(new opTensor(weights->shape()));
    weight_reorder_OIhwi16o(*weights, *weights_internal);
    return SaberSuccess;
}

template <DataType OpDtype ,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus JitUniDWConvolution<OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<inTensor*>& inputs,
        std::vector<outTensor*>& outputs,
        ConvActiveParam<opTensor> &param) {
    
    ConvParam<opTensor> *conv_param = &(param.conv_param);
    const opTensor *bias = conv_param->bias();

    const dtype *ptr_src = reinterpret_cast<const dtype*>(inputs[0]->get_buf()->get_data());
    const dtype *ptr_weights = reinterpret_cast<const dtype*>(weights_internal->get_buf()->get_data());
    const dtype *ptr_bias = reinterpret_cast<const dtype*>(bias->get_buf()-> get_data());
    auto ptr_dst = reinterpret_cast<dtype*>(outputs[0]->mutable_data());

    const auto &jcp = kernel_->jcp;

    int dil_h = jcp.dilate_h + 1;
    int dil_w = jcp.dilate_w + 1;
    int str_h = jcp.stride_h;
    int str_w = jcp.stride_w;

    int MB = jcp.mb;
    int chb_work = utils::div_up(jcp.nb_ch, jcp.nb_ch_blocking);
    const size_t work_amount = MB * chb_work * jcp.oh;

    auto kernel_params = [&](int ur_w_step, int ow, int oh, int ih, int kh,
            int kh_padding, int ch, int ch_num, int n) {
        jit_conv_call_t par_conv = {};

        const int i_l_overflow = utils::max(0, (jcp.l_pad - ow * str_w));
        const int i_r_overflow = utils::max(jcp.iw, (ow * str_w
                                + (jcp.kw - 1)*dil_w - jcp.l_pad + 1)) - jcp.iw;

        const int iw = utils::max((ow*str_w - jcp.l_pad
                                + utils::div_up(i_l_overflow, dil_w)*dil_w), 0);
        const int kw = utils::div_up(i_l_overflow, dil_w);

        const int kw_padding = jcp.kw - utils::div_up(i_l_overflow, dil_w)
                                - utils::div_up(i_r_overflow, dil_w);

        par_conv.src = ptr_src + n * jcp.ic * jcp.iw * jcp.ih + ch * jcp.iw * jcp.ih * 16 + ih * jcp.iw * 16 + iw * 16;
        par_conv.src = ptr_dst + n * jcp.oc * jcp.ow * jcp.oh + ch * jcp.iw * jcp.ih * 16 + oh * jcp.ow * 16 + ow * 16;

        //par_conv.filt = &weights[weights_d.blk_off(ch, 0, 0, kh, kw)];
        par_conv.filt = ptr_weights + ch * jcp.ngroups * jcp.kh * jcp.kw + kh * jcp.kw * 16 + kw *16;
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

                kernel_->jit_ker(&par_conv);
            }

            // main loop
            ur_w_step = (jcp.iw - (jcp.kw - 1)*dil_w + jcp.l_pad - 1)
                / jcp.stride_w - ow + 1;
            if (ur_w_step > 0) {
                jit_conv_call_t par_conv = kernel_params(ur_w_step, ow, oh, ih,
                                            kh, kh_padding, ch, ch_num, n);

                kernel_->jit_ker(&par_conv);

                ow += ur_w_step;
            }

            // right border
            ur_w_step = 1;
            for (; ow < jcp.ow; ow++) {
                jit_conv_call_t par_conv = kernel_params(ur_w_step, ow, oh, ih,
                                            kh, kh_padding, ch, ch_num, n);

                kernel_->jit_ker(&par_conv);
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

template class JitUniDWConvolution<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW_C8>;
template class JitUniDWConvolution<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW_C16>;
template class JitUniDWConvolution<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C16, NCHW_C16>;

} // namespace saber
} // namespace anakin
