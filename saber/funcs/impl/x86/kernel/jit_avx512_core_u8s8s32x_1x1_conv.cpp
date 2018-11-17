#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_core_u8s8s32x_1x1_conv.h"

namespace anakin {
namespace saber {

using namespace jit;

void JitAvx512u8s8s32xConv1x1::prepare_rtus(const std::vector<Tensor<X86>*>& inputs,
        jit_1x1_conv_conf_t& conf) {
    bool rtus_applicable = true &&
                           (conf.stride_h != 1 || conf.stride_w != 1) &&
                           (inputs[0]->get_layout() == Layout_NCHW_C16 || inputs[0]->get_layout() == Layout_NCHW_C8);

    rtus_applicable = rtus_applicable &&
                      conf.t_pad == 0 && conf.l_pad == 0 &&
                      conf.oh * conf.stride_h == conf.ih &&
                      conf.ow * conf.stride_w == conf.iw;

    // LOG(ERROR) << "rtus applicable:" << rtus_applicable;
    if (rtus_applicable) {
        this->reduce_src = true;
        conf.stride_h = conf.stride_w = 1;
        conf.ih = conf.oh;
        conf.iw = conf.ow;
    }

    return;
}


template <typename T, typename U>
void balance2D(U nthr, U ithr, T ny, T& ny_start, T& ny_end,
               T nx, T& nx_start, T& nx_end, T nx_divider) {
    const T grp_size = utils::div_up(nthr, nx_divider);
    const T grp_count = utils::div_up(nthr, grp_size);

    T grp = ithr / grp_size;
    T grp_ithr = ithr % grp_size;
    T grp_nthr = grp_size;
    T first_grps = nthr % grp_count;

    if (first_grps > 0 && grp >= first_grps) {
        ithr -= first_grps * grp_size;
        grp_nthr--;
        grp = ithr / grp_nthr + first_grps;
        grp_ithr = ithr % grp_nthr;
    }

    utils::balance211(nx, grp_count, grp, nx_start, nx_end);
    utils::balance211(ny, grp_nthr, grp_ithr, ny_start, ny_end);
}

SaberStatus JitAvx512u8s8s32xConv1x1::init(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param,
        Context<X86>& ctx) {
    this->_ctx = &ctx;
    ConvParam<X86>* conv_param = &(param.conv_param);
    const Tensor<X86>* weights = conv_param->weight();

    if (!(inputs[0]->get_layout() == Layout_NHWC &&
            outputs[0]->get_layout() == Layout_NHWC &&
            weights->get_layout() == Layout_NCHW)) {
        return SaberUnImplError;
    }

    // reorder weights
    Tensor<X86>* weights_reorder = conv_param->mutable_weight();

    if (weights_internal_ != nullptr) {
        delete weights_internal_;
    }

    weights_internal_ = new Tensor<X86>(weights_reorder->shape(), AK_INT8);
    weights_internal_->set_scale(weights_reorder->get_scale());
    weight_reorder_OIhw4i16o4i(*weights_reorder, *weights_internal_, weights_reorder->get_scale());

    return create(inputs, outputs, param, ctx);
}

SaberStatus JitAvx512u8s8s32xConv1x1::create(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param,
        Context<X86>& ctx) {
    SaberStatus status;
    ConvParam<X86>* conv_param = &(param.conv_param);
    EltwiseParam<X86>* eltwise_param = &(param.eltwise_param);
    ActivationParam<X86>* act_param = &(conv_param->activation_param);
    const Tensor<X86>* weights = conv_param->weight();
    const Tensor<X86>* bias = conv_param->bias();
    Tensor<X86>* input = inputs[0];
    Tensor<X86>* output = outputs[0];
    Shape src_shape(input->shape());
    Shape dst_shape(output->shape());
    Shape wgt_shape(weights->shape());


    // check conf
    if (kernel_) {
        status = check_conf(inputs, outputs, param);

        if (status != SaberNotInitialized) {
            return status;
        }
    }

    // init conf
    const bool with_groups = (conv_param->group > 1);
    conf.ngroups = with_groups ? weights->num() : 1;

    conf.mb = src_shape[0];
    conf.ic = wgt_shape[1];
    conf.ih = src_shape[1];
    conf.iw = src_shape[2];

    conf.oc = wgt_shape[0];
    conf.oh = dst_shape[1];
    conf.ow = dst_shape[2];
    conf.oc_without_padding = conf.oc;
    conf.ic_without_padding = conf.ic;

    conf.kh = wgt_shape[2];
    conf.kw = wgt_shape[3];
    conf.stride_h = conv_param->stride_h;
    conf.stride_w = conv_param->stride_w;
    conf.t_pad = conv_param->pad_h;
    conf.l_pad = conv_param->pad_w;

    conf.with_relu = act_param->has_active;

    if (conf.with_relu) {
        conf.relu_negative_slope = static_cast<float>(act_param->negative_slope);
    }

    conf.with_sum = eltwise_param->has_eltwise && (eltwise_param->operation == Eltwise_sum);

    if (conf.with_sum) {
        conf.sum_scale = eltwise_param->coeff[1];
    }

    conf.with_bias = (bias != NULL);

    if (bias != nullptr) {
        conf.bia_dt = bias->get_dtype();
    }

    conf.dst_dt = output->get_dtype();
    conf.typesize_in = type_length(input->get_dtype());
    conf.typesize_out = type_length(output->get_dtype());
    conf.typesize_acc = sizeof(int32_t);
    conf.typesize_bia = conf.with_bias ? type_length(conf.bia_dt) : 0;
    conf.rm = conv_param->rm;

    prepare_rtus(inputs, conf);

    conv_d.n = src_shape[0];
    conv_d.ic = wgt_shape[1];
    conv_d.ih = src_shape[1];
    conv_d.iw = src_shape[2];
    conv_d.oc = wgt_shape[0];
    conv_d.oh = dst_shape[1];
    conv_d.ow = dst_shape[2];
    conv_d.t_pad = conv_param->pad_h;
    conv_d.l_pad = conv_param->pad_w;
    conv_d.stride_h = conv_param->stride_h;
    conv_d.stride_w = conv_param->stride_w;

    status = jit_avx512_core_u8s8s32x_conv1x1_kernel::init_conf(conf, conv_d, omp_get_max_threads(),
             reduce_src);

    if (status == SaberSuccess) {
        if (kernel_ != nullptr) {
            delete kernel_;
            kernel_ = nullptr;
        }

        kernel_ = new jit_avx512_core_u8s8s32x_conv1x1_kernel(conf);
    } else {
        return SaberUnImplError;
    }

    if (reduce_src) {
        init_rtus_driver<uint8_t>(&rtus_driver_, conf, conv_d, ws_per_thread_, &scratch_);
    }

    // bias reorder
    Tensor<X86>* bias_src = conv_param->mutable_bias();

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

    return SaberSuccess;
}

SaberStatus JitAvx512u8s8s32xConv1x1::dispatch(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param) {
    ConvParam<X86>* conv_param = &(param.conv_param);
    const Tensor<X86>* bias = conv_param->bias();

    // check input and output data type, do scale or not
    CHECK_EQ(inputs[0]->get_dtype(), AK_UINT8) << "only support uint8 input type";
    const unsigned char* ptr_src = reinterpret_cast<const unsigned char*>(inputs[0]->data());
    const char* ptr_weights = reinterpret_cast<const char*>(weights_internal_->data());
    const int32_t* ptr_bias = nullptr;

    if (bias_internal_ != nullptr) {
        ptr_bias = reinterpret_cast<const int32_t*>(bias_internal_->data());
    }

    char* ptr_dst = reinterpret_cast<char*>(outputs[0]->mutable_data());
    int dst_type_size = type_length(outputs[0]->get_dtype());

    const auto& jcp = kernel_->jcp;
    const auto& oscales = scale_;
    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    const int stride_h = conv_param->stride_h;
    const int stride_w = conv_param->stride_w;
    const int pad_t = conv_param->pad_h;
    const int pad_l = conv_param->pad_w;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    #pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        int nthr = omp_get_num_threads();

        auto p = jit_1x1_conv_call_t();

        auto rp = rtus_driver_t::call_params_t();

        const int nb_oc = jcp.nb_load;
        const int os_block = jcp.bcast_block;
        // LOG(INFO) << "saber [nb_oc, nb_ic, nb_ic_blocking, os_block, load_grp_count] is [" << jcp.nb_load << ", " << jcp.nb_reduce << ", " << jcp.nb_reduce_blocking
        //                                                                                    << ", " << jcp.bcast_block << ", " << jcp.load_grp_count;

        int bcast_start{ 0 }, bcast_end{ 0 }, ocb_start{ 0 }, ocb_end{ 0 };
        balance2D(nthr, ithr, work_amount, bcast_start, bcast_end,
                  jcp.nb_load, ocb_start, ocb_end, jcp.load_grp_count);

        auto init_bcast = [&](int iwork, int& n, int& g, int& bcast_step,
        int& oh, int& ow, int& ih, int& iw) {
            int osb{0};
            nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
                             jcp.nb_bcast);
            bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                              jcp.nb_bcast_blocking_max);
            bcast_step = utils::min(bcast_step, bcast_end - iwork);

            const int os = osb * os_block;
            oh = os / jcp.ow;
            ow = os % jcp.ow;

            ih = utils::max(oh * stride_h - pad_t, 0);
            iw = utils::max(ow * stride_w - pad_l, 0);
            rp.iw_start = iw;

            p.bcast_dim = this_block_size(os, jcp.os, bcast_step * os_block);
            rp.os = p.bcast_dim;
        };

        auto init_load = [&](int ocb, int& load_step) {
            load_step = step(jcp.nb_load_blocking, ocb_end - ocb,
                             jcp.nb_load_blocking_max);
            p.load_dim = this_block_size(ocb * jcp.oc_block,
                                         ocb_end * jcp.oc_block, load_step * jcp.oc_block);

            if (ocb + load_step >= nb_oc) {
                p.first_last_flag |= FLAG_OC_LAST;
            } else {
                p.first_last_flag &= ~FLAG_OC_LAST;
            }
        };

        auto init_reduce = [&]() {
            p.reduce_dim = this_block_size(0, jcp.ic, jcp.ic);
            rp.icb = p.reduce_dim / jcp.reduce_block;
        };

        auto inner_ker = [&](int ocb, int n, int g, int oh, int ow,
        int ih, int iw) {
            const int icb = 0; // Start from the first IC block
            const int _ocb = g * nb_oc + ocb;
            const int _icb = g;

            //const size_t dst_off = dst_d.blk_off(n, _ocb * jcp.oc_block, oh, ow);
            const size_t dst_off = n * jcp.oc * jcp.oh * jcp.ow + oh * jcp.ow * jcp.oc
                                   + ow * jcp.oc + _ocb * jcp.oc_block;
            const size_t wei_off = ocb * jcp.ic * jcp.kh * jcp.kw * jcp.oc_block
                                   + icb * jcp.kh * jcp.kw * jcp.oc_block * jcp.ic_block;

            // p.output_data = &ptr_dst[dst_off];
            p.output_data = ptr_dst + dst_off * dst_type_size;
            // p.load_data = &weights[conf_.with_groups()
            //    ? weights_d.blk_off(g, ocb, icb)
            //    : weights_d.blk_off(ocb, icb)];
            p.load_data = &ptr_weights[wei_off];
            p.bias_data = &ptr_bias[_ocb * jcp.oc_block];
            p.scales = &oscales[jcp.is_oc_scale * _ocb * jcp.oc_block];

            if (reduce_src) {
                rp.ws = scratch_ + ithr * ws_per_thread_
                        + _icb * jcp.is * jcp.ic_block;

                if (ocb == ocb_start) {
                    // rp.src = src + src_d.blk_off(n, _icb * jcp.ic_block, ih, iw);
                    rp.src = ptr_src + n * jcp.ic * jcp.ih * jcp.iw +
                             + ih * jcp.iw * jcp.ic + iw * jcp.ic + _icb * jcp.ic_block;
                    rtus_driver_->ker_(&rp);
                }

                p.bcast_data = rp.ws;
            } else {
                // p.bcast_data = src + src_d.blk_off(n, _icb * jcp.ic_block, ih, iw);
                p.bcast_data = ptr_src + n * jcp.ic * jcp.ih * jcp.iw +
                               + ih * jcp.iw * jcp.ic + iw * jcp.ic + _icb * jcp.ic_block;;
            }

            kernel_->jit_ker(&p);
        };

        if (jcp.loop_order == loop_rlb) {
            init_reduce();
            int ocb = ocb_start;

            while (ocb < ocb_end) {
                int load_step = 0;
                init_load(ocb, load_step);
                int iwork = bcast_start;

                while (iwork < bcast_end) {
                    int n = 0;
                    int g = 0;
                    int bcast_step = 0;
                    int oh = 0;
                    int ow = 0;
                    int ih = 0;
                    int iw = 0;
                    init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                    inner_ker(ocb, n, g, oh, ow, ih, iw);
                    iwork += bcast_step;
                }

                ocb += load_step;
            }
        } else if (jcp.loop_order == loop_lbr) {
            int ocb = ocb_start;

            while (ocb < ocb_end) {
                int load_step = 0;
                init_load(ocb, load_step);
                int iwork = bcast_start;

                while (iwork < bcast_end) {
                    int n = 0;
                    int g = 0;
                    int bcast_step = 0;
                    int oh = 0;
                    int ow = 0;
                    int ih = 0;
                    int iw = 0;
                    init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                    init_reduce();
                    inner_ker(ocb, n, g, oh, ow, ih, iw);
                    iwork += bcast_step;
                }

                ocb += load_step;
            }
        } else if (jcp.loop_order == loop_rbl) {
            init_reduce();
            int iwork = bcast_start;

            while (iwork < bcast_end) {
                int n = 0;
                int g = 0;
                int bcast_step = 0;
                int oh = 0;
                int ow = 0;
                int ih = 0;
                int iw = 0;
                init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                int ocb = ocb_start;

                while (ocb < ocb_end) {
                    int load_step = 0;
                    init_load(ocb, load_step);
                    inner_ker(ocb, n, g, oh, ow, ih, iw);
                    ocb += load_step;
                }

                iwork += bcast_step;
            }
        } else if (jcp.loop_order == loop_blr) {
            int iwork = bcast_start;

            while (iwork < bcast_end) {
                int n = 0;
                int g = 0;
                int bcast_step = 0;
                int oh = 0;
                int ow = 0;
                int ih = 0;
                int iw = 0;
                init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                int ocb = ocb_start;

                while (ocb < ocb_end) {
                    int load_step = 0;
                    init_load(ocb, load_step);
                    init_reduce();
                    inner_ker(ocb, n, g, oh, ow, ih, iw);
                    ocb += load_step;
                }

                iwork += bcast_step;
            }
        } else {
            assert(!"unsupported loop order");
        }
    }

    return SaberSuccess;
}

SaberStatus JitAvx512u8s8s32xConv1x1::check_conf(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param) {
    ConvParam<X86>* conv_param = &(param.conv_param);
    const Tensor<X86>* weights = conv_param->weight();
    const jit_1x1_conv_conf_t jcp = kernel_->jcp;

    // check format
    if (!(inputs[0]->get_layout() == Layout_NHWC &&
            outputs[0]->get_layout() == Layout_NHWC &&
            weights->get_layout() == Layout_NCHW)) {
        LOG(ERROR) << "wrong format";
        return SaberUnImplError;
    }

    // check param
    bool param_ok = true &&
                    jcp.t_pad == conv_param->pad_h &&
                    jcp.l_pad == conv_param->pad_w &&
                    jcp.stride_h == conv_param->stride_h &&
                    jcp.stride_w == conv_param->stride_w;

#if 0
    // check shape
    bool shape_ok = true &&
                    jcp.kh == weights->height() &&
                    jcp.kw == weights->width() &&
                    jcp.ngroups == 1 &&
                    jcp.mb == input->num() &&
                    jcp.ic == input->channel() &&
                    jcp.ih == input->height() &&
                    jcp.iw == input->width() &&
                    jcp.oc == output->channel() &&
                    jcp.oh == output->height() &&
                    jcp.ow == output->width();

    if (param_ok && shape_ok) {
        return SaberSuccess;
    } else {
        LOG(ERROR) << "param or shape changed, re-init kernel";
        return SaberNotInitialized;
    }

#endif
    return SaberSuccess;
}

} // namespace saber
} // namespace anakin
