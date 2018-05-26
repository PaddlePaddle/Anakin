#include <iostream>

#include "saber/funcs/impl/x86/jit_avx512_conv_act.h"
#include "saber/funcs/impl/x86/jit_call_conf.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {

using namespace jit;

using jit_conv_ker_t = void (*)(jit_conv_call_t *);

inline bool is_1stconv(const jit_conv_conf_t &jcp) {
      return utils::one_of(jcp.ic, 1, 3);
}

inline void jit_conv_ker_pipeline(jit_conv_ker_t ker, jit_conv_call_t &p,
       const void *src, const void *dst, const void *filt, const void *bias,
       int channel, int kh_padding)
{
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
SaberStatus JitAvx512ConvAct<OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
        const std::vector<inTensor*>& inputs,
        std::vector<outTensor*>& outputs,
        ConvActiveParam<opTensor> &param,
        Context<X86> &ctx) {
    // get context of avx512_conv_act
    this->_ctx = ctx;
    ConvParam<opTensor> *conv_param = &(param.conv_param);
    ActivationParam<opTensor> *act_param = &(param.activation_param);

    const opTensor *weights = conv_param->weight();

    Shape src_shape(inputs[0]->shape());
    Shape dst_shape(outputs[0]->shape());
    Shape weights_shape(weights->shape());

    const bool with_groups = false;
    conf.ngroups = with_groups ? weights_shape[0] : 1;

    conf.mb = src_shape[0];
    if (src_shape.dims() == 5) {
        conf.ic = src_shape[1] * src_shape[4] / conf.ngroups;
    }
    else {
        conf.ic = src_shape[1] / conf.ngroups;
    }
    conf.ih = src_shape[2];
    conf.iw = src_shape[3];

    if (dst_shape.dims() == 5) {
       conf.oc = dst_shape[1] * dst_shape[4] / conf.ngroups;
    } else {
       conf.oc = dst_shape[1] / conf.ngroups;
    }
    conf.oh = dst_shape[2];
    conf.ow = dst_shape[3];

    conf.kh = weights_shape[2];
    conf.kw = weights_shape[3];
    conf.stride_h = conv_param->stride_h;
    conf.stride_w = conv_param->stride_w;
    conf.t_pad = conv_param->pad_h;
    conf.l_pad = conv_param->pad_w;
    conf.dilate_h = conv_param->dilation_h;
    conf.dilate_w = conv_param->dilation_w;

    conf.with_relu = param.has_active;
    if (conf.with_relu) {
        conf.relu_negative_slope = static_cast<float>(act_param->negative_slope);
    }
    conf.with_bias = !(conv_param->bias() == NULL);
    conf.is_1stconv = is_1stconv(conf);

    // check memory layout
    if (conf.is_1stconv) {
        if (!(std::is_same<LayOutType_in, NCHW>::value &&
              std::is_same<LayOutType_out, NCHW_C16>::value &&
              std::is_same<LayOutType_op, NCHW>::value )) {
            return SaberUnImplError;
        }
    } else {
        if (!(std::is_same<LayOutType_in, NCHW_C16>::value &&
              std::is_same<LayOutType_out, NCHW_C16>::value &&
              std::is_same<LayOutType_op, NCHW>::value)) {
            return SaberUnImplError;
        }
    }

    SaberStatus status = jit_conv_act_kernel::init_conf(conf);
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
SaberStatus JitAvx512ConvAct<OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
        const std::vector<inTensor*>& inputs,
        std::vector<outTensor*>& outputs,
        ConvActiveParam<opTensor> &param,
        Context<X86> &ctx) {
    kernel_ = new jit_conv_act_kernel(conf);

    ConvParam<opTensor> *conv_param = &(param.conv_param);
    opTensor *weights = conv_param->mutable_weight();
    weights_internal.reset(new opTensor(weights->shape()));
    if (std::is_same<LayOutType_in, NCHW>::value) {
        weight_reorder_OIhwi16o(*weights, *weights_internal);
    } else if (std::is_same<LayOutType_in, NCHW_C16>::value) {
        weight_reorder_OIhw16i16o(*weights, *weights_internal);
    }

    return SaberSuccess;
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus JitAvx512ConvAct<OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
        const std::vector<inTensor*>& inputs,
        std::vector<outTensor*>& outputs,
        ConvActiveParam<opTensor> &param) {

    ConvParam<opTensor> *conv_param = &(param.conv_param);
    const opTensor *bias = conv_param->bias();

    const dtype *ptr_src = reinterpret_cast<const dtype*>(
            inputs[0]->get_buf()->get_data());
    const dtype *ptr_weights = reinterpret_cast<const dtype*>(
            weights_internal->get_buf()->get_data());
    const dtype *ptr_bias = reinterpret_cast<const dtype*>(
            bias->get_buf()-> get_data());
    auto ptr_dst = reinterpret_cast<dtype*>(outputs[0]->mutable_data());

    const auto &jcp = kernel_->jcp;

    #pragma omp parallel
    {
        int ithr = omp_get_thread_num(), nthr = omp_get_num_threads();
        int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
        int start, end, start_copy;
        int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh;
        utils::balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        jit_conv_call_t par_conv = { 0 };
        size_t src_h_stride = jcp.iw * jcp.ic_block;
        size_t src_c_stride = jcp.ih * jcp.iw * jcp.ic_block;
        size_t dst_h_stride = jcp.ow * jcp.oc_block;
        size_t wht_h_stride = jcp.kw * jcp.ic_block * jcp.oc_block;
        size_t wht_ic_stride = jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block;

        if (jcp.is_1stconv) {
            src_h_stride = jcp.iw;
            src_c_stride = jcp.ih * jcp.iw;
            wht_ic_stride = jcp.oc_block;
        }

        for (int icb_l2 = 0; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            int n{0}, g{0}, occ{0}, oh_s{0};
            if (jcp.loop_order == conv_loop_order_t::loop_cgn) {
                utils::nd_iterator_init(start, occ, oc_chunks,
                                        g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
            }
            else if (jcp.loop_order == conv_loop_order_t::loop_gnc) {
                utils::nd_iterator_init(start,
                                        g, jcp.ngroups,
                                        n, jcp.mb,
                                        occ, oc_chunks,
                                        oh_s, jcp.oh);
            }

            while (start < end) {
                int ocb = occ * jcp.nb_oc_blocking;
                int g_ocb = g * jcp.nb_oc + ocb;
                int g_oc = g_ocb * jcp.oc_block;
                int g_icb = g * jcp.nb_ic;

                int work_rem = end - start;
                int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
                int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

                size_t bias_blk_off = g_oc;
                size_t dst_blk_off = n * jcp.oc * jcp.oh * jcp.ow +
                                     (g_ocb * jcp.oh * jcp.ow + oh_s * jcp.ow) * jcp.oc_block;
                size_t src_blk_off = n * jcp.ic * jcp.ih * jcp.iw +
                                     (g_icb + icb_l2) * jcp.ih * jcp.iw * jcp.ic_block
                                     + ih_s * jcp.iw * jcp.ic_block;
                size_t weight_blk_off= ocb * jcp.ic * jcp.kh * jcp.kw * jcp.oc_block +
                                       icb_l2 * jcp.kh * jcp.kw * jcp.oc_block * jcp.ic_block;

                if (jcp.is_1stconv) {
                   src_blk_off = n * jcp.ic * jcp.ih * jcp.iw + ih_s * jcp.iw;
                   weight_blk_off = ocb * jcp.ic * jcp.kh * jcp.kw * jcp.oc_block;
                }

                auto bias_w = ptr_bias ? ptr_bias + bias_blk_off : 0;
                auto dst_w = ptr_dst + dst_blk_off;
                auto src_w = ptr_src + src_blk_off;
                auto wht_w = ptr_weights + weight_blk_off;

                for (int icb = icb_l2;
                     icb < utils::min(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2); ++icb) {
                    auto src_c = src_w;
                    auto dst_c = dst_w;
                    for (int oj = oh_s, ij = ih_s;
                        oj < oh_e; ++oj, ij += jcp.stride_h) {

                        int i_t_overflow = -utils::min(0, ij);
                        int i_b_overflow = utils::max(jcp.ih, ij + jcp.kh) - jcp.ih;
                        int kh_padding = utils::max(0, jcp.kh - i_t_overflow - i_b_overflow);

                        jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                                              src_c + i_t_overflow * src_h_stride,
                                              dst_c, wht_w + i_t_overflow * wht_h_stride,
                                              bias_w, icb, kh_padding);

                        src_c += src_h_stride * jcp.stride_h;
                        dst_c += dst_h_stride;
                    }
                    src_w += src_c_stride;
                    wht_w += wht_ic_stride;
                }

                if (jcp.loop_order == conv_loop_order_t::loop_cgn) {
                       utils::nd_iterator_jump(start, end,
                                               occ, oc_chunks, g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
                } else if (jcp.loop_order == conv_loop_order_t::loop_gnc) {
                       utils::nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, occ, oc_chunks, oh_s, jcp.oh);
                }
            }
        }

        jit_conv_ker_pipeline(kernel_->jit_ker, par_conv,
                              ptr_src, ptr_dst, ptr_weights, ptr_bias, 0, 0);

    }

    return SaberSuccess;
}

template class JitAvx512ConvAct<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C16, NCHW_C16>;
template class JitAvx512ConvAct<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW_C16>;
template class JitAvx512ConvAct<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW_C8>;

} // namespace saber
} // namespace anakin
