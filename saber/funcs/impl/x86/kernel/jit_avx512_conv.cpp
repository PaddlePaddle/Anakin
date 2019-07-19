#include <iostream>

#include "saber/funcs/impl/x86/kernel/jit_avx512_conv.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "tensor_op.h"
namespace anakin {
namespace saber {

using namespace jit;

using jit_conv_ker_t = void (*)(jit_conv_call_t*);

inline void jit_conv_ker_pipeline(jit_conv_ker_t ker, jit_conv_call_t& p,
                                  const void* src, const void* dst, const void* filt, const void* bias,
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
SaberStatus JitAvx512Conv<AK_FLOAT>::check_conf(
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
    conf.is_1stconv = input_layout == Layout_NCHW
                      && utils::one_of(input->channel(), 1, 3); //utils::one_of(input->channel(), 1, 3);


    if (conf.is_1stconv) {
        bool is_layout_ok = (input_layout == Layout_NCHW)
                            && (output_layout == Layout_NHWC || output_layout == Layout_NCHW_C16
                                || output_layout == Layout_NCHW_C16R || output_layout == Layout_NCHW)
                            && weights->get_layout() == Layout_NCHW;

        if (!is_layout_ok) {
            LOG(FATAL) << "1stconv wrong format ";
            return SaberUnImplError;
        }
    } else {
        bool is_layout_ok = (input_layout == Layout_NCHW_C16 || input_layout == Layout_NCHW_C16R) &&
                            (output_layout == Layout_NCHW_C16 || output_layout == Layout_NCHW_C16R) &&
                            (conv_param->weight()->get_layout() == Layout_NCHW);

        if (!is_layout_ok) {
            LOG(FATAL) << "wrong format";
            return SaberUnImplError;
        }
    }

    // check param
    bool param_ok = true
                    && jcp.t_pad == conv_param->pad_h
                    && jcp.l_pad == conv_param->pad_w
                    && jcp.stride_h == conv_param->stride_h
                    && jcp.stride_w == conv_param->stride_w
                    && jcp.dilate_h == (conv_param->dilation_h <= 0 ? 0 : (conv_param->dilation_h - 1))
                    && jcp.dilate_w == (conv_param->dilation_w <= 0 ? 0 : (conv_param->dilation_w - 1));

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
SaberStatus JitAvx512Conv<AK_FLOAT>::create(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param,
    Context<X86>& ctx) {
    SaberStatus status;
    ConvParam<X86>* conv_param = &(param.conv_param);
    ActivationParam<X86>* act_param = nullptr;
    const Tensor<X86>* weights = conv_param->weight();
    Tensor<X86>* output = outputs[0];
    Tensor<X86>* input = inputs[0];

    // check conf
    if (kernel) {
        status = check_conf(inputs, outputs, param);

        if (status != SaberNotInitialized) {
            return status;
        }
    }

    // init conf
    const bool with_groups = false;
    conf.ngroups = with_groups ? weights->num() : 1;

    conf.mb = input->num();
    conf.ic = input->channel() / conf.ngroups;
    conf.ih = input->height();
    conf.iw = input->width();

    conf.oc = output->channel() / conf.ngroups;
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

    conf.with_relu = conv_param->activation_param.has_active;

    if (conf.with_relu) {
        act_param = &(conv_param->activation_param);
        conf.relu_negative_slope = static_cast<float>(act_param->negative_slope);
    }

    conf.with_bias = (conv_param->bias() != nullptr && conv_param->bias()->valid_size() > 0);

    conf.with_sum = false;
    conf.dst_dt = output->get_dtype();

    if (outputs[0]->get_layout() == Layout_NHWC) {
        conf.output_nhwc = true;
    } else {
        conf.output_nhwc = false;
    }

    status = jit_conv_act_kernel::init_conf(conf);

    if (status == SaberSuccess) {
        if (kernel != nullptr) {
            delete kernel;
            kernel = nullptr;
        }

        kernel = new jit_conv_act_kernel(conf);
    } else {
        LOG(FATAL) << "jit_conv_act_kernel SaberUnImplError ";
        return SaberUnImplError;
    }

    // reorder weights
    Tensor<X86>* weights_reorder = conv_param->mutable_weight();
    weights_internal.reset(new Tensor<X86>(weights_reorder->valid_shape()));

    if (inputs[0]->get_layout() == Layout_NCHW) {
        weight_reorder_OIhwi16o(*weights_reorder, *weights_internal);
    } else if (inputs[0]->get_layout() == Layout_NCHW_C16
               || inputs[0]->get_layout() == Layout_NCHW_C16R) {
        weight_reorder_OIhw16i16o(*weights_reorder, *weights_internal);
    } else {
        LOG(FATAL) << "unsupport ";
    }

    if (output[0].get_dtype() == AK_UINT8) {
        CHECK(output[0].get_scale().size() > 0);
        float scale = 1.f / (output[0].get_scale()[0] * (127.f / 255.f));
        utils::ScaleUtils::scale_fp32_fp32(*weights_internal, scale);

        if ((conv_param->bias() != nullptr && conv_param->bias()->valid_size() > 0)) {
            utils::try_expand_tensor(bias_internal, conv_param->bias()->valid_shape());
            bias_internal.copy_from(*conv_param->bias());
            utils::ScaleUtils::scale_fp32_fp32(bias_internal, scale);
        }
    }

    if (output->get_layout() == Layout_NCHW) {
        utils::try_expand_tensor(_inner_tensor, Shape({output->num(), conf.oc, conf.oh, conf.ow}, Layout_NCHW_C16R));
        DLOG(INFO) << "try_expand_tensor " << _inner_tensor.valid_size() << "," << conf.oc << "," << conf.oh
                   << "," << conf.ow;
    }

    return SaberSuccess;
}

template <>
SaberStatus JitAvx512Conv<AK_FLOAT>::init(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param,
    Context<X86>& ctx) {
    SaberStatus ret = SaberSuccess;
    ConvParam<X86>* conv_param = &(param.conv_param);
    Tensor<X86>* input = inputs[0];

    LayoutType input_layout = inputs[0]->get_layout();
    LayoutType output_layout = outputs[0]->get_layout();
    conf.is_1stconv = input_layout == Layout_NCHW; //utils::one_of(input->channel(), 1, 3);

    if (conf.is_1stconv) {
        bool is_layout_ok = (input_layout == Layout_NCHW)
                            && (output_layout == Layout_NHWC || output_layout == Layout_NCHW_C16
                                || output_layout == Layout_NCHW_C16R || output_layout == Layout_NCHW)
                            && conv_param->weight()->get_layout() == Layout_NCHW;

        if (!is_layout_ok) {
            LOG(FATAL) << "1stconv wrong format ";
            return SaberUnImplError;
        }
    } else {
        bool is_layout_ok = (input_layout == Layout_NCHW_C16 || input_layout == Layout_NCHW_C16R) &&
                            (output_layout == Layout_NCHW_C16 || output_layout == Layout_NCHW_C16R) &&
                            (conv_param->weight()->get_layout() == Layout_NCHW);

        if (!is_layout_ok) {
            LOG(FATAL) << "wrong format";
            return SaberUnImplError;
        }
    }

    this->_ctx = &ctx;
    ret = create(inputs, outputs, param, ctx);

    if (ret != SaberSuccess) {
        LOG(FATAL) << "create failed";
        return ret;
    }

    return ret;
}

template <>
SaberStatus JitAvx512Conv<AK_FLOAT>::dispatch_nchw_c16(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param) {


    ConvParam<X86>* conv_param = &(param.conv_param);
    const Tensor<X86>* bias = conv_param->bias();
    const DataType type = outputs[0]->get_dtype();

    const float* ptr_src = reinterpret_cast<const float*>(inputs[0]->data());
    const float* ptr_weights = reinterpret_cast<const float*>(weights_internal->data());
    const float* ptr_bias = reinterpret_cast<const float*>(bias->data());
    DLOG(INFO) << "outputs " << outputs.size() << "," << outputs[0]->valid_shape();
    auto ptr_dst = static_cast<float*>(outputs[0]->mutable_data());

    const auto& jcp = kernel->jcp;
    DLOG(INFO) << "dispatch_nchw_c16 " << jcp.is_1stconv << "," << jcp.output_nhwc;
    #pragma omp parallel
    {
        int ithr = anakin_get_thread_num(), nthr = anakin_get_num_threads();
        int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
        int start, end, start_copy;
        int work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_t();
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
                nd_iterator_init(start, occ, oc_chunks, g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
            } else if (jcp.loop_order == conv_loop_order_t::loop_gnc) {
                nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, occ, oc_chunks, oh_s, jcp.oh);
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
                                     (g_icb + icb_l2) * jcp.ih * jcp.iw * jcp.ic_block + ih_s * jcp.iw * jcp.ic_block;
                size_t weight_blk_off = ocb * jcp.ic * jcp.kh * jcp.kw * jcp.oc_block +
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

                        jit_conv_ker_pipeline(kernel->jit_ker, par_conv,
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
                    nd_iterator_jump(start, end,
                                     occ, oc_chunks, g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
                } else if (jcp.loop_order == conv_loop_order_t::loop_gnc) {
                    nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, occ, oc_chunks, oh_s, jcp.oh);
                }
            }
        }

        jit_conv_ker_pipeline(kernel->jit_ker, par_conv,
                              ptr_src, ptr_dst, ptr_weights, ptr_bias, 0, 0);

    }

    return SaberSuccess;
}

template <>
SaberStatus JitAvx512Conv<AK_FLOAT>::dispatch_nhwc(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param) {

    CHECK(outputs[0]->get_dtype() == AK_UINT8);
    ConvParam<X86>* conv_param = &(param.conv_param);
    const Tensor<X86>* bias = conv_param->bias();
    const DataType type = outputs[0]->get_dtype();

    const float* ptr_src = reinterpret_cast<const float*>(inputs[0]->data());
    const float* ptr_weights = reinterpret_cast<const float*>(weights_internal->data());
    const float* ptr_bias = reinterpret_cast<const float*>(bias_internal.data());

    auto ptr_dst = static_cast<uint8_t*>(outputs[0]->mutable_data());

    const auto& jcp = kernel->jcp;
    DLOG(INFO) << "dispatch_nhwc " << jcp.is_1stconv << "," << jcp.output_nhwc;
    #pragma omp parallel
    {
        int ithr = anakin_get_thread_num(), nthr = anakin_get_num_threads();
        int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
        int start, end, start_copy;
        int work_amount = jcp.mb * jcp.ngroups * jcp.oh;

        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_call_t();
        size_t src_h_stride = jcp.iw * jcp.ic_block;
        size_t src_c_stride = jcp.ih * jcp.iw * jcp.ic_block;
        size_t dst_h_stride = jcp.ow * jcp.oc;
        size_t wht_h_stride = jcp.kw * jcp.ic_block * jcp.oc_block;
        size_t wht_ic_stride = jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block;

        if (jcp.is_1stconv) {
            src_h_stride = jcp.iw;
            src_c_stride = jcp.ih * jcp.iw;
            wht_ic_stride = jcp.oc_block;
        } else {
            LOG(FATAL) << "not support";
        }

        for (int icb_l2 = 0; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            int n{0}, g{0}, oh_s{0};

            if (jcp.loop_order == conv_loop_order_t::loop_cgn) {
                nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
            } else if (jcp.loop_order == conv_loop_order_t::loop_gnc) {
                nd_iterator_init(start, g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
            }

            while (start < end) {
                for (int occ = 0; occ < oc_chunks; occ++) {
                    int ocb = occ * jcp.nb_oc_blocking;
                    int g_ocb = g * jcp.nb_oc + ocb;
                    int g_oc = g_ocb * jcp.oc_block;
                    int g_icb = g * jcp.nb_ic;

                    int work_rem = end - start;
                    int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
                    int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

                    size_t bias_blk_off = g_oc;
                    size_t dst_blk_off = n * jcp.oh * jcp.ow * jcp.oc + oh_s * jcp.ow * jcp.oc + g_ocb * jcp.oc_block;
                    size_t src_blk_off = n * jcp.ic * jcp.ih * jcp.iw +
                                         (g_icb + icb_l2) * jcp.ih * jcp.iw * jcp.ic_block + ih_s * jcp.iw * jcp.ic_block;
                    size_t weight_blk_off = ocb * jcp.ic * jcp.kh * jcp.kw * jcp.oc_block +
                                            icb_l2 * jcp.kh * jcp.kw * jcp.oc_block * jcp.ic_block;

                    if (jcp.is_1stconv) {
                        src_blk_off = n * jcp.ic * jcp.ih * jcp.iw + ih_s * jcp.iw;
                        weight_blk_off = ocb * jcp.ic * jcp.kh * jcp.kw * jcp.oc_block;
                    } else {
                        LOG(FATAL) << "not support";
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

                            jit_conv_ker_pipeline(kernel->jit_ker, par_conv,
                                                  src_c + i_t_overflow * src_h_stride,
                                                  dst_c, wht_w + i_t_overflow * wht_h_stride,
                                                  bias_w, icb, kh_padding);
                            src_c += src_h_stride * jcp.stride_h;
                            dst_c += dst_h_stride;
                        }

                        src_w += src_c_stride;
                        wht_w += wht_ic_stride;
                    }
                }

                if (jcp.loop_order == conv_loop_order_t::loop_cgn) {
                    nd_iterator_jump(start, end,
                                     g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
                } else if (jcp.loop_order == conv_loop_order_t::loop_gnc) {
                    nd_iterator_jump(start, end, g, jcp.ngroups, n, jcp.mb, oh_s, jcp.oh);
                }
            }
        }

        jit_conv_ker_pipeline(kernel->jit_ker, par_conv,
                              ptr_src, ptr_dst, ptr_weights, ptr_bias, 0, 0);

    }

    return SaberSuccess;
}
template <>
SaberStatus JitAvx512Conv<AK_FLOAT>::dispatch(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvEltwiseParam<X86>& param) {

    const auto& jcp = kernel->jcp;

    if (outputs[0]->get_layout() == Layout_NCHW) {
        std::vector<Tensor<X86>*> temp_tensor_vec;
        temp_tensor_vec.push_back(&_inner_tensor);
        //        print_tensor(*inputs[0]);
        dispatch_nchw_c16(inputs, temp_tensor_vec, param);
        //        LOG(INFO)<<"dispatch_nchw_c16 finish";
        //        print_tensor(_inner_tensor);
        utils::reorder_nchwc_nchw(_inner_tensor, *outputs[0]);
        return SaberSuccess;
    } else if (jcp.output_nhwc) {
        return dispatch_nhwc(inputs, outputs, param);
    } else {
        return dispatch_nchw_c16(inputs, outputs, param);
    }

}

// template class JitAvx512Conv<AK_FLOAT>;

} // namespace saber
} // namespace anakin
