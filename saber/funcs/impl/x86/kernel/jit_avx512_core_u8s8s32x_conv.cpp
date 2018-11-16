#include "saber/funcs/impl/x86/kernel/jit_avx512_core_u8s8s32x_conv.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "anakin_thread.h"

namespace anakin {
namespace saber {

using namespace jit;

SaberStatus JitAvx512U8S8S32XConv::init(const std::vector<Tensor<X86>*> &inputs,
                                        std::vector<Tensor<X86>*> &outputs,
                                        ConvEltwiseParam<X86> &param,
                                        Context<X86> &ctx) {
    this->_ctx = &ctx;
    ConvParam<X86> *conv_param = &(param.conv_param);
    const Tensor<X86> *weights = conv_param->weight();
    Shape wgt_shape(weights->shape());
    bool depthwise = (conv_param->group > 1) && (wgt_shape[1] == 1);

    // reorder weights
    // TODO check weights, do scale or not?
    Tensor<X86> *weights_reorder = conv_param->mutable_weight();
    if (weights_internal_ != nullptr) {
        delete weights_internal_;
        weights_internal_ = nullptr;
    }
    weights_internal_ = new Tensor<X86>(weights_reorder->shape(), AK_INT8);
    weights_internal_->set_scale(weights_reorder->get_scale());
    if (depthwise) {
        weight_reorder_Goihw16g(*weights_reorder, *weights_internal_);
    } else if (conv_param->group == 1) {
        weight_reorder_OIhw4i16o4i(*weights_reorder, *weights_internal_, weights_reorder->get_scale());
    } else {
        return SaberUnImplError;
    }

    return create(inputs, outputs, param, ctx);
}

SaberStatus JitAvx512U8S8S32XConv::create(const std::vector<Tensor<X86>*> &inputs,
                                          std::vector<Tensor<X86>*> &outputs,
                                          ConvEltwiseParam<X86> &param,
                                          Context<X86> &ctx) {
    SaberStatus status = SaberSuccess;

    ConvParam<X86> *conv_param = &(param.conv_param);
    jit_conv_conf_t jcp;

    status = init_conf(jcp, inputs, outputs, param);
    if (status != SaberSuccess) {
        return status;
    }

    // TODO check bias, do scale or not?
    Tensor<X86> *bias_src = conv_param->mutable_bias();
    if (bias_internal_ != nullptr) {
        delete bias_internal_;
        bias_internal_ = nullptr;
    }
    if (bias_src != nullptr) {
        bias_internal_ = new Tensor<X86>(bias_src->shape(), AK_INT32);
        bias_internal_->set_scale(bias_src->get_scale());
        bias_reorder_nchw(*bias_src, *bias_internal_, bias_src->get_scale());
    }

    float scale_in = inputs[0]->get_scale()[0];
    float scale_out = outputs[0]->get_scale()[0];
    auto scale_w = weights_internal_->get_scale();
    std::vector<float>().swap(scale_);
    for (int i = 0; i < scale_w.size(); i++) {
        this->scale_.push_back((scale_w[i] * scale_in) / scale_out);
    }

    return status;
}

SaberStatus JitAvx512U8S8S32XConv::dispatch(const std::vector<Tensor<X86>*> &inputs,
                                            std::vector<Tensor<X86>*> &outputs,
                                            ConvEltwiseParam<X86> &param) {
    ConvParam<X86> *conv_param = &(param.conv_param);
    const Tensor<X86> *bias = conv_param->bias();

    // check input and output data type, do scale or not
    CHECK_EQ(inputs[0]->get_dtype(), AK_UINT8) << "only support uint8 input type";
    const unsigned char *ptr_src = reinterpret_cast<const unsigned char*>(inputs[0]->data());
    const char *ptr_weights = reinterpret_cast<const char*>(weights_internal_->data());
    const int32_t *ptr_bias = nullptr;
    if (bias_internal_ != nullptr) {
        ptr_bias = reinterpret_cast<const int32_t*>(bias_internal_->data());
    }
    char *ptr_dst = reinterpret_cast<char *>(outputs[0]->mutable_data());
    int dst_type_size = type_length(outputs[0]->get_dtype());

    const auto &jcp = kernel_->jcp;
    const auto oscale = scale_;

    parallel(0, [&](const int ithr, const int nthr) {
        int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int nb_groups = jcp.nb_ch;
        int group_block = jcp.ch_block;

        int start{0}, end{0};
        int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh;
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_call_t();

        size_t src_h_stride = jcp.iw * jcp.ic;
        size_t dst_h_stride = jcp.ow * jcp.oc;
        size_t wht_h_stride = jcp.kw * jcp.ic_block * jcp.oc_block;
        size_t wht_ic_stride = jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block;
        if (jcp.is_dw) {
            src_h_stride = jcp.iw * jcp.ic * jcp.ngroups;
            dst_h_stride = jcp.ow * jcp.oc * jcp.ngroups;
            wht_h_stride = jcp.kw * jcp.ch_block;
            wht_ic_stride = jcp.kh * jcp.kw * jcp.ch_block;
        }

        int n{0}, gb{0}, occ{0}, oh_s{0};
        if (jcp.loop_order == loop_cgn) {
            utils::nd_iterator_init(start, occ, oc_chunks, gb, nb_groups, n, jcp.mb, oh_s, jcp.oh);
        } else if (jcp.loop_order == loop_gnc) {
            utils::nd_iterator_init(start, gb, nb_groups, n, jcp.mb, occ, oc_chunks, oh_s, jcp.oh);
        } else if (jcp.loop_order == loop_ngc) {
            utils::nd_iterator_init(start, n, jcp.mb, gb, nb_groups, occ, oc_chunks, oh_s, jcp.oh);
        } else {
            assert(!"unsupported loop order");
        }

        while (start < end) {
            int ocb = occ * jcp.nb_oc_blocking;
            int g = gb * group_block;
            int g_oc = (g * jcp.nb_oc + ocb) * jcp.oc_block;

            int g_ic = g * jcp.nb_ic * jcp.oc_block;

            int work_rem = end - start;
            int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

            size_t bias_blk_off = g_oc;
            size_t dst_blk_off = n * jcp.oc * jcp.oh * jcp.ow +
                                 oh_s * jcp.ow * jcp.oc + g_oc;
            size_t src_blk_off = n * jcp.ic * jcp.ih * jcp.iw +
                                 ih_s * jcp.iw * jcp.ic + g_ic;
            size_t weight_blk_off = ocb * jcp.ic * jcp.kh * jcp.kw * jcp.oc_block;
            if (jcp.is_dw) {
                dst_blk_off = n * nb_groups *jcp.oh * jcp.ow * jcp.ch_block + g_oc + oh_s * jcp.ow * nb_groups * jcp.ch_block;
                src_blk_off = n * nb_groups *jcp.ih * jcp.iw * jcp.ch_block + g_ic + ih_s * jcp.iw * nb_groups * jcp.ch_block;
                weight_blk_off =  gb * jcp.kh * jcp.kw * jcp.ch_block + ocb * jcp.kh * jcp.kw * jcp.ch_block;
            }
            auto bias_w = ptr_bias ? ptr_bias + bias_blk_off : 0;
            auto dst_w = ptr_dst + dst_blk_off * dst_type_size;
            auto src_w = ptr_src + src_blk_off;
            auto wht_w = ptr_weights + weight_blk_off;

            for (int oj = oh_s, ij = ih_s;
                 oj < oh_e; ++oj, ij += jcp.stride_h) {
                int dilate_h = jcp.dilate_h + 1;
                int i_t_overflow = utils::div_up(utils::max(0, -ij), dilate_h);
                int i_b_overflow = utils::div_up(utils::max(0, ij - jcp.ih + (jcp.kh - 1) * dilate_h + 1),
                                                 dilate_h);
                int kh_padding = utils::max(0,
                                            jcp.kh - i_t_overflow - i_b_overflow);

                p.src = src_w + i_t_overflow * dilate_h * src_h_stride;
                p.dst = dst_w;
                p.filt = wht_w + i_t_overflow * wht_h_stride;
                p.bias = bias_w;
                p.oc_blocks = jcp.is_dw ? gb : ocb;
                p.kh_padding = kh_padding;
                p.scales = &oscale[jcp.is_oc_scale * g_oc];
                kernel_->jit_ker(&p);

                src_w += src_h_stride * jcp.stride_h;
                dst_w += dst_h_stride * dst_type_size;
            }

            if (jcp.loop_order == loop_cgn) {
                utils::nd_iterator_jump(start, end, occ, oc_chunks, gb, nb_groups, n,
                                        jcp.mb, oh_s, jcp.oh);
            } else if (jcp.loop_order == loop_gnc) {
                utils::nd_iterator_jump(start, end, gb, nb_groups, n, jcp.mb, occ,
                                        oc_chunks, oh_s, jcp.oh);
            } else if (jcp.loop_order == loop_ngc) {
                utils::nd_iterator_jump(start, end, n, jcp.mb, gb, nb_groups, occ,
                                        oc_chunks, oh_s, jcp.oh);
            } else {
                assert(!"unsupported loop order");
            }
        }
    });

    return SaberSuccess;
}

SaberStatus JitAvx512U8S8S32XConv::init_conf(jit_conv_conf_t &jcp,
                                             const std::vector<Tensor<X86>*> &inputs,
                                             std::vector<Tensor<X86>*> &outputs,
                                             ConvEltwiseParam<X86> &param) {
    SaberStatus status;
    ConvParam<X86> *conv_param = &(param.conv_param);
    EltwiseParam<X86> *eltwise_param = &(param.eltwise_param);
    ActivationParam<X86> *act_param = &(conv_param->activation_param);
    const Tensor<X86> *weights = conv_param->weight();
    const Tensor<X86> *bias = conv_param->bias();
    Tensor<X86> *input = inputs[0];
    Tensor<X86> *output = outputs[0];
    Shape src_shape(input->shape());
    Shape dst_shape(output->shape());
    Shape wgt_shape(weights->shape());

    // init conf
    const bool with_groups = (conv_param->group > 1);
    jcp.ngroups = with_groups ? conv_param->group : 1;

    jcp.mb = src_shape[0];
    jcp.ic = src_shape[3]/jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.ih = src_shape[1];
    jcp.iw = src_shape[2];
    jcp.oc = dst_shape[3]/jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.oh = dst_shape[1];
    jcp.ow = dst_shape[2];

    jcp.kh = wgt_shape[2];
    jcp.kw = wgt_shape[3];

    jcp.stride_h = conv_param->stride_h;
    jcp.stride_w = conv_param->stride_w;
    jcp.t_pad = conv_param->pad_h;
    jcp.l_pad = conv_param->pad_w;
    jcp.b_pad = conv_param->pad_h;
    jcp.r_pad = conv_param->pad_w;
    jcp.dilate_h = conv_param->dilation_h <= 0 ? 0 : (conv_param->dilation_h - 1);
    jcp.dilate_w = conv_param->dilation_w <= 0 ? 0 : (conv_param->dilation_w - 1);

    if (bias != nullptr) {
        jcp.bia_dt = bias->get_dtype();
    }
    jcp.dst_dt = output->get_dtype();
    jcp.rm = conv_param->rm;
    jcp.ur_h = 1;

    jcp.with_bias = (bias != NULL);
    jcp.with_relu = conv_param->activation_param.has_active;
    if (jcp.with_relu) {
        jcp.relu_negative_slope = static_cast<float>(act_param->negative_slope);
    }

    jcp.is_dw = with_groups && (jcp.ic == 1);

    jcp.with_sum = eltwise_param->has_eltwise && (eltwise_param->operation == Eltwise_sum);
    if (jcp.with_sum) {
        jcp.sum_scale = eltwise_param->coeff[1];
    }

    status = jit_avx512_core_u8s8s32x_fwd_kernel::init_conf(jcp);
    if (status == SaberSuccess) {
        if (kernel_ != nullptr) {
            delete kernel_;
            kernel_ = nullptr;
        }
        kernel_ = new jit_avx512_core_u8s8s32x_fwd_kernel(jcp);
    } else {
        return SaberUnImplError;
    }

    const int nthreads = omp_get_max_threads();
    ws_per_thread_ = jcp.oh * jcp.ow * jcp.oc;
    ws_ = (int *)zmalloc(nthreads * ws_per_thread_ * sizeof(int), 4096);
    if (!ws_) {
        LOG(ERROR) << "workspace allocation failed";
        delete kernel_;
        kernel_ = nullptr;
        return SaberOutOfMem;
    }
    return SaberSuccess;
}

SaberStatus JitAvx512U8S8S32XConv::check_conf(const jit_conv_conf_t &jcp,
                                              const std::vector<Tensor<X86>*> &inputs,
                                              std::vector<Tensor<X86>*> &outputs,
                                              ConvEltwiseParam<X86> &param) {
    return SaberSuccess;
}

} // namespace saber
} // namespace anakin
