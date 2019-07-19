#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/kernel/jit_avx2_deconv.h"
#include "x86_utils.h"
#include "tensor_op.h"

namespace anakin {
namespace saber {

using namespace jit;

using jit_deconv_ker_t = void (*)(jit_deconv_call_t*);

inline void jit_deconv_ker_pipeline(jit_deconv_ker_t ker, jit_deconv_call_t& p,
                                    const void* src, const void* dst, const void* filt,
                                    const void* bias, int channel, int kh_padding) {

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

    if (p.src&&ker) {
        ker(&p);
    }else{

    }
}

template <>
SaberStatus JitAvx2Deconv<AK_FLOAT>::check_conf(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvParam<X86>& param) {

    ConvParam<X86>* conv_param = &(param);
    const Tensor<X86>* weights = conv_param->weight();
    const jit_deconv_conf_t jcp = kernel->jcp;
    Tensor<X86>* input = outputs[0];
    Tensor<X86>* output = inputs[0];

    // check param
    bool param_ok = true
                    && jcp.t_pad == conv_param->pad_h
                    && jcp.l_pad == conv_param->pad_w
                    && jcp.stride_h == conv_param->stride_h
                    && jcp.stride_w == conv_param->stride_w;

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

template <>
SaberStatus JitAvx2Deconv<AK_FLOAT>::create(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvParam<X86>& param, Context<X86>& ctx) {

    SaberStatus status = SaberSuccess;
    ConvParam<X86>* conv_param = &(param);
    ActivationParam<X86>* act_param = nullptr;
    const Tensor<X86>* weights = conv_param->weight();
    Tensor<X86>* input = outputs[0];
    Tensor<X86>* output = inputs[0];

    // check conf
    if (kernel) {
        status = check_conf(inputs, outputs, param);

        if (status != SaberNotInitialized) {
            LOG(INFO) << "check_conf != SaberNotInitialized";
            return status;
        }
    }

    // init conf
    conf.src_fmt = input->get_layout();

    if (input->get_layout() == Layout_NCHW_C8R) {
        conf.src_fmt = Layout_NCHW_C8;
    }

    conf.ngroups = 1;

    conf.ndims = input->dims();
    conf.mb = input->num();

    // swap param
    conf.ic = input->channel();
    conf.ih = input->height();
    conf.iw = input->width();

    conf.oc = output->channel();
    conf.oc_without_padding = conf.oc;
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

    conf.with_bias = (conv_param->bias() != nullptr && conv_param->bias()->valid_size() > 0);
    conf.with_relu = conv_param->activation_param.has_active;
    conf.with_sum = false;

    if (conf.with_relu) {
        return SaberUnImplError;
    }

    if (conf.dilate_h != 0 || conf.dilate_w != 0) {
        return SaberUnImplError;
    }

    if (conf.with_relu) {
        act_param = &(conv_param->activation_param);
        conf.relu_negative_slope = act_param->negative_slope;
    }

    status = jit_avx2_deconv_act_kernel::init_conf(conf);

    if (status == SaberSuccess) {
        if (kernel != nullptr) {
            delete kernel;
            kernel = nullptr;
        }

        kernel = new jit_avx2_deconv_act_kernel(this->conf);
    } else {
        return SaberUnImplError;
    }

    // reorder weights
    Tensor<X86>* weights_reorder = conv_param->mutable_weight();

    weights_internal.reset(new Tensor<X86>(weights_reorder->valid_shape()));

    if (conf.src_fmt == Layout_NCHW_C8) {
        weight_reorder_OIhw8o8i(*weights_reorder, *weights_internal);
    }

    return SaberSuccess;
}

template <>
SaberStatus JitAvx2Deconv<AK_FLOAT>::init(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvParam<X86>& param, Context<X86>& ctx) {

    ConvParam<X86>* conv_param = &(param);

    if ((inputs[0]->get_layout() != Layout_NCHW_C8R)
            || (outputs[0]->get_layout() != Layout_NCHW_C8R)
            || (conv_param->weight()->get_layout() != Layout_NCHW)) {
        LOG(FATAL) << "data layout is not supported " << inputs[0]->get_layout() << "," <<
                   outputs[0]->get_layout();
        return SaberUnImplError;
    }

    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus JitAvx2Deconv<AK_FLOAT>::dispatch(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvParam<X86>& param) {
    using namespace std;
    ConvParam<X86>* conv_param = &(param);
    const Tensor<X86>* bias = conv_param->bias();

    auto diff_src = reinterpret_cast<const float*>(outputs[0]->data());
    auto weights = reinterpret_cast<const float*>(weights_internal->data());
    auto diff_dst = reinterpret_cast<const float*>(inputs[0]->data());
    const float* diff_bias = (bias != nullptr
                              && bias->valid_size() > 0) ? reinterpret_cast<const float*>(bias->data()) : nullptr;

    const auto& jcp = kernel->jcp;


    size_t diff_src_h_stride = jcp.iw * jcp.ic_block;
    size_t diff_src_C_stride = jcp.ih * jcp.iw * jcp.ic_block;
    size_t diff_src_n_stride = jcp.ih * jcp.iw * jcp.ic;
    size_t diff_dst_h_stride = jcp.ow * jcp.oc_block;
    size_t diff_dst_C_stride = jcp.oh * jcp.ow * jcp.oc_block;
    size_t diff_dst_n_stride = jcp.oh * jcp.ow * jcp.oc;
    size_t wht_h_stride = jcp.kw * jcp.ic_block * jcp.oc_block;
    size_t wht_ic_stride = jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block;
    size_t wht_oc_stride = jcp.kh * jcp.kw * jcp.ic * jcp.oc_block;
    size_t wht_g_stride = wht_oc_stride / jcp.ngroups;

    bool is_fast_path = jcp.dilate_h == 0 && jcp.stride_h == 1;

    auto ker = [&](const int ithr, const int nthr) {
        int start{0}, end{0}, start_copy;
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int work_amount = jcp.ngroups * jcp.mb * ic_chunks * jcp.ih;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        jit_deconv_call_t par_deconv;
        par_deconv.src_prf = nullptr;
        par_deconv.dst_prf = nullptr;
        par_deconv.filt_prf = nullptr;
        par_deconv.bias_prf = nullptr;
        par_deconv.kh_padding_prf = 0;
        par_deconv.channel_prf = 0;

        for (int ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            int n{0}, g{0}, icc{0}, ih_s{0};

            nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, icc, ic_chunks, ih_s, jcp.ih);

            while (start < end) {
                int icb = icc * jcp.nb_ic_blocking;
                int g_icb = g * jcp.nb_ic + icb;
                int g_ocb = g * jcp.nb_oc;

                int work_rem = end - start;
                int ih_e = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;

                auto diff_src_w = diff_src + n * diff_src_n_stride + g_icb *
                                  diff_src_C_stride; //diff_src_d.blk_off(n, g_icb);
                auto diff_dst_w = diff_dst + n * diff_dst_n_stride + (g_ocb + ocb_l2) * diff_dst_C_stride;
                auto wht_w = weights + g * wht_g_stride + ocb_l2 * wht_oc_stride + icb * wht_ic_stride;
                auto bias_w = diff_bias ? diff_bias + g_icb * jcp.ic_block : nullptr;

                for (int ocb = ocb_l2;
                        ocb < utils::min(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2); ++ocb) {
                    for (int ij = ih_s; ij < ih_e; ++ij) {
                        int oj, k_len, k_lo;

                        if (is_fast_path) { // dilate == 0 && stride == 1
                            int i_t_overflow = utils::max(0, jcp.kh - 1 - ij
                                                          - jcp.t_pad);
                            int i_b_overflow = utils::max(0, jcp.kh - jcp.ih + ij
                                                          - jcp.b_pad);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow;
                        } else {
                            int i_t_overflow = utils::max(0, (jcp.kh - 1 - ij
                                                              - jcp.t_pad) / jcp.stride_h);
                            int i_b_overflow = utils::max(0, (jcp.kh - jcp.ih + ij
                                                              - jcp.b_pad) / jcp.stride_h);
                            int overflow_kh_hi = jcp.kh - 1 - std::abs((jcp.ih - 1
                                                 + jcp.b_pad - ij) % jcp.stride_h);
                            int overflow_kh_lo = (ij + jcp.t_pad)
                                                 % jcp.stride_h;

                            k_len = (overflow_kh_hi - overflow_kh_lo)
                                    / jcp.stride_h + 1 - i_t_overflow
                                    - i_b_overflow;
                            k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                            oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                        }

                        assert(k_len >= 0);

                        jit_deconv_ker_pipeline(kernel->jit_ker, par_deconv,
                                                diff_src_w + ij * diff_src_h_stride,
                                                diff_dst_w + oj * diff_dst_h_stride,
                                                wht_w + k_lo * wht_h_stride,
                                                bias_w, ocb, k_len);
                    }

                    diff_dst_w += diff_dst_C_stride;
                    wht_w += wht_oc_stride;
                }

                nd_iterator_jump(start, end, n, jcp.mb, g, jcp.ngroups, icc, ic_chunks, ih_s, jcp.ih);
            }
        }

        jit_deconv_ker_pipeline(kernel->jit_ker, par_deconv,
                                diff_src, diff_dst, weights, 0, 0, 1);
    };

    #pragma omp parallel
    {
        ker(anakin_get_thread_num(), anakin_get_num_threads());
    }

    return SaberSuccess;
}

template class JitAvx2Deconv<AK_FLOAT>;
} // namespace saber
} // namespace anakin
