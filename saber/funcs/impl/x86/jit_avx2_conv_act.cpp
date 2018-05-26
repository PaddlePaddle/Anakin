#include "saber/funcs/impl/x86/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_conv_act_kernel.h"
#include "saber/funcs/impl/x86/jit_avx2_conv_act.h"

#include "x86_utils.h"

namespace anakin {
namespace saber {

using namespace jit;

using jit_conv_ker_t = void (*)(jit_conv_call_t *);

inline void jit_conv_ker_pipeline(jit_conv_ker_t ker, jit_conv_call_t &p,
        const void *src, const void *dst, const void *filt, const void *bias,
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

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus JitAvx2ConvAct<OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<inTensor*>& inputs,
        std::vector<outTensor*>& outputs,
        ConvActiveParam<opTensor> &param, Context<X86> &ctx) {
    this->_ctx = ctx;
    ConvParam<opTensor> *conv_param = &(param.conv_param);
    ActivationParam<opTensor> *act_param = &(param.activation_param);

    const opTensor *weights = conv_param->weight();
    Shape src_shape(inputs[0]->shape());
    Shape dst_shape(outputs[0]->shape());
    Shape weights_shape(weights->shape());

    // const bool with_groups = false;
    conf.ngroups = 1;
    conf.mb = src_shape[0];
    conf.ic = src_shape[1];
    conf.ih = src_shape[2];
    conf.iw = src_shape[3];

    conf.oc = dst_shape[1] * 8;
    conf.oh = dst_shape[2];
    conf.ow = dst_shape[3];

    conf.kh = weights_shape[2];
    conf.kw = weights_shape[3];
    conf.stride_h = conv_param -> stride_h;
    conf.stride_w = conv_param -> stride_w;
    conf.t_pad = conv_param -> pad_h;
    conf.l_pad = conv_param -> pad_w;
    conf.dilate_h = conv_param -> dilation_h;
    conf.dilate_w = conv_param -> dilation_w;
    conf.with_relu = param.has_active;
    conf.with_bias = !(conv_param -> bias() == NULL);
    
    if (conf.with_relu) {
        conf.relu_negative_slope = static_cast<float>(act_param->negative_slope);
    }

    if (!(std::is_same<LayOutType_in, NCHW>::value &&
          std::is_same<LayOutType_out, NCHW_C8>::value &&
          std::is_same<LayOutType_op, NCHW>::value)) {
        return SaberUnImplError;
    }

    SaberStatus status = jit_avx2_conv_act_kernel::init_conf(conf);
    if (status == SaberSuccess) {
        return create(inputs, outputs, param, ctx);
    } else {
        return SaberUnImplError;
    }
}
template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus JitAvx2ConvAct<OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<inTensor*>& inputs,
        std::vector<outTensor*>& outputs,
        ConvActiveParam<opTensor> &param, Context<X86> &ctx) {
    kernel_ = new jit_avx2_conv_act_kernel(this->conf);

    ConvParam<opTensor> *conv_param = &(param.conv_param);
    opTensor *weights = conv_param->mutable_weight();
    weights_internal.reset(new opTensor(weights->shape()));
    weight_reorder_OIhwi8o(*weights, *weights_internal);

    return SaberSuccess;
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus JitAvx2ConvAct<OpDtype, inDtype, outDtype,
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

    int ocb_work = utils::div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    // int gb_work = jcp.nb_g;
    const size_t work_amount = jcp.mb * jcp.ngroups * ocb_work * jcp.oh;
    // const size_t work_amount_dw = jcp.mb * gb_work * jcp.oh;

    auto ker = [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        utils::balance211(work_amount, nthr, ithr, start, end);

        int icbb = 0;
        while (icbb < jcp.nb_ic) {
            int icb_step = jcp.nb_ic_blocking;
            int icb_step_rem = jcp.nb_ic - icbb;
            if (icb_step_rem < jcp.nb_ic_blocking_max) {
                icb_step = icb_step_rem;
            }

            size_t n{0}, g{0}, ocbb{0}, oh{0};
            utils::nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work, oh, jcp.oh);
            for (size_t iwork = start; iwork < end; ++iwork) {
                int ocb = ocbb * jcp.nb_oc_blocking;
                int ocb_num = jcp.nb_oc_blocking;

                for (int icb = icbb; icb < icbb + icb_step; ++icb) {
                    jit_conv_call_t par_conv = {};

                    const int ij = oh * jcp.stride_h;
                    const int i_t_overflow = saber::utils::max(0, jcp.t_pad - ij);
                    const int i_b_overflow = saber::utils::max(jcp.ih, ij + (jcp.kh - 1) * (jcp.dilate_h + 1) - jcp.t_pad + 1) - jcp.ih;

                    const size_t _oc = g * jcp.nb_oc + ocb;
                    const size_t _ic = g * jcp.nb_ic + icb;

                    const int src_ic = jcp.ic == 3 ? 0 : _ic;
                    const int wgt_ic = jcp.ic == 3 ? 0 : icb;
                    const int ih = saber::utils::max(ij - jcp.t_pad + utils::div_up(i_t_overflow,
                                                                                    (jcp.dilate_h + 1)) * (jcp.dilate_h + 1), 0);

                    par_conv.src = ptr_src + n * jcp.ic * jcp.iw * jcp.ih + src_ic * jcp.iw * jcp.ih + ih * jcp.iw;
                    par_conv.dst = ptr_dst + n * jcp.oc * jcp.ow * jcp.oh + _oc * jcp.ow * jcp.oh * 8 + oh * jcp.ow * 8;

                    const int wh = utils::div_up(i_t_overflow, (jcp.dilate_h + 1)); 

                    par_conv.filt = ptr_weights + ocb * jcp.kh * jcp.kw * jcp.ic * 8 + wgt_ic * 8 + wh * jcp.kw * jcp.ic * 8;

                    if (icb == 0) {
                        if (bias) {
                            par_conv.bias = ptr_bias +  _oc * 8;
                        }
                        par_conv.flags |= 1 << 4;
                    }

                    if (jcp.with_relu && icb + 1 == jcp.nb_ic) {
                        par_conv.flags |= 1 << 5;
                    }

                    par_conv.oc_blocks = saber::utils::min(ocb + ocb_num, jcp.nb_oc) - ocb;

                    par_conv.kw_padding = 0;
                    const int kh_padding = jcp.kh -
                                           utils::div_up(i_t_overflow, (jcp.dilate_h + 1)) -
                                           utils::div_up(i_b_overflow, (jcp.dilate_h + 1));
                    par_conv.kh_padding = saber::utils::max(0, kh_padding);
                    kernel_->jit_ker(&par_conv);
                }
                utils::nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work,
                                        oh, jcp.oh);
            }
            icbb += icb_step;
        }
    };

    #pragma omp parallel
    {
        ker(omp_get_thread_num(), omp_get_num_threads());
    }

    return SaberSuccess;
}

template class JitAvx2ConvAct<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW_C8>;
template class JitAvx2ConvAct<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW_C16>;
template class JitAvx2ConvAct<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C16, NCHW_C16>;


} // namespace saber
} // namespace anakin
