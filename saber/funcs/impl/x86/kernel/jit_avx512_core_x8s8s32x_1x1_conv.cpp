#include "saber/funcs/impl/x86/anakin_thread.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_core_x8s8s32x_1x1_conv.h"
#include "debug.h"
namespace anakin {
namespace saber {

using namespace jit;

void JitAvx512x8s8s32xConv1x1::prepare_rtus(const std::vector<Tensor<X86>*>& inputs,
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



SaberStatus JitAvx512x8s8s32xConv1x1::init(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param,
        Context<X86>& ctx) {
    LOG(INFO) << "init JitAvx512x8s8s32xConv1x1";
    this->_ctx = &ctx;
    ConvParam<X86>* conv_param = &(param.conv_param);
    const Tensor<X86>* weights = conv_param->weight();
    Tensor<X86>* input = inputs[0];

    if (!(inputs[0]->get_layout() == Layout_NHWC &&
            outputs[0]->get_layout() == Layout_NHWC &&
            weights->get_layout() == Layout_NCHW)) {
        return SaberUnImplError;
    }

    // padding weights
    Tensor<X86>* weights_orig = conv_param->mutable_weight();

    if (weights_orig->get_dtype() == AK_FLOAT) {
        _weights_scale.re_alloc(weights_orig->valid_shape(), AK_INT8);
        utils::ScaleUtils::scale_conv_weights_to_nchw_host(_weights_scale, *conv_param->weight());
        weights_orig = &_weights_scale;
    }

    int ch_blk = 16;
    Shape shape = weights_orig->valid_shape();
    int oc_value = shape[0];
    int ic_value = shape[1];
    int kh_value = shape[2];
    int kw_value = shape[3];
    int oc_padding = utils::rnd_up(oc_value, ch_blk);
    int ic_padding = utils::rnd_up(ic_value, ch_blk);
    Shape shape_padding({oc_padding, ic_padding, kh_value, kw_value}, weights_orig->get_layout());

    if (weights_padding_ != nullptr) {
        delete weights_padding_;
        weights_padding_ = nullptr;
    }

    weights_padding_ = new Tensor<X86>(shape_padding, AK_INT8);
    weights_padding_->set_scale(weights_orig->get_scale());
    weight_padding_nhwc(weights_orig, weights_padding_);

    if (input->get_dtype() == AK_INT8) {
        if (compensation_ != nullptr) {
            delete compensation_;
            compensation_ = nullptr;
        }

        compensation_ = (int32_t*)zmalloc(sizeof(int32_t) * oc_padding, 4096);
        memset(compensation_, 0, sizeof(int32_t) * oc_padding);
        float temp = 0;
        int32_t offset = 0;
        int32_t sum = 0;
        char* ptr_weights_padding = reinterpret_cast<char*>(weights_padding_->data());

        for (size_t i = 0; i < oc_padding; i++) {
            for (size_t j = 0; j < ic_padding; j++) {
                for (size_t h = 0; h < kh_value; h++) {
                    for (size_t w = 0; w < kw_value; w++) {
                        offset = i * ic_padding * kh_value * kw_value +
                                 j * kh_value * kw_value +
                                 h * kw_value +
                                 w;

                        if (!mayiuse(avx512_core_vnni)) {
                            ptr_weights_padding[offset] =
                                saturate<int8_t>(nearbyintf(ptr_weights_padding[offset] / 2.0f));
                        }

                        sum += ptr_weights_padding[offset];
                    }
                }
            }

            compensation_[i] = -128 * sum;
            sum = 0;
        }
    }

    // reorder weights
    if (weights_internal_ != nullptr) {
        delete weights_internal_;
        weights_internal_ = nullptr;
    }

    weights_internal_ = new Tensor<X86>(weights_padding_->valid_shape(), AK_INT8);
    weights_internal_->set_scale(weights_padding_->get_scale());
    weight_reorder_OIhw4i16o4i(*weights_padding_, *weights_internal_, weights_padding_->get_scale());

    // bias reorder
    Tensor<X86>* bias_src = conv_param->mutable_bias();

    if (bias_src != nullptr && bias_src->valid_size() > 0) {
        CHECK_EQ(bias_src->get_dtype(), AK_FLOAT);
        bias_internal_ = new Tensor<X86>(bias_src->valid_shape(), AK_FLOAT);
        auto weights_scale = weights_orig->get_scale();
        float in_scale = 1.f;
        CHECK_GT(input->get_scale().size(), 0) << "only support input scale size > 0";

        if (input->get_scale().size() > 0) {
            in_scale = input->get_scale()[0];
        }

        std::vector<float> scale_vec(bias_src->valid_size());

        if (inputs[0]->get_dtype() == AK_UINT8) {
            for (int i = 0; i < bias_src->valid_size(); i++) {
                scale_vec[i] = (1.f / (weights_scale[i] * in_scale * (127.f / 255.f)));
            }
        } else if (inputs[0]->get_dtype() == AK_INT8) {
            for (int i = 0; i < bias_src->valid_size(); i++) {
                scale_vec[i] = (1.f / (weights_scale[i] * in_scale));
            }
        } else {
            LOG(FATAL) << "not support input dtype " << inputs[0]->get_dtype();
        }

        bias_internal_->set_scale(scale_vec);
        bias_reorder_nchw(*bias_src, *bias_internal_, scale_vec);
    }

    EltwiseParam<X86>* elt_param = &param.eltwise_param;
    bool with_sum = elt_param->has_eltwise && (elt_param->operation == Eltwise_sum);


    if (with_sum) {
        CHECK_EQ(outputs[0]->get_scale().size(), 1);
        float out_scale = outputs[0]->get_scale()[0];
        DataType be_added_type = param.conv_param.beta_type;
        DataType output_dtype = outputs[0]->get_dtype();
        CHECK(be_added_type == AK_INT8 || be_added_type == AK_UINT8);

        if (be_added_type == AK_INT8 && output_dtype == AK_UINT8) {
            _sum_scale = param.conv_param.beta * (255.f/127.f)/ out_scale;
        } else if (be_added_type == AK_UINT8 && output_dtype == AK_INT8) {
            _sum_scale = param.conv_param.beta * (127.f/255.f) / out_scale;
        } else if (be_added_type == AK_UINT8 && output_dtype == AK_UINT8){
            _sum_scale = param.conv_param.beta / out_scale;
        } else if (be_added_type == AK_INT8 && output_dtype == AK_INT8){
            _sum_scale = param.conv_param.beta / out_scale;
        } else {
            LOG(FATAL)<<"not support type "<<be_added_type<<":"<<output_dtype;
        }
    }

    return create(inputs, outputs, param, ctx);
}

SaberStatus JitAvx512x8s8s32xConv1x1::create(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param,
        Context<X86>& ctx) {
    SaberStatus status;
    ConvParam<X86>* conv_param = &(param.conv_param);
    EltwiseParam<X86>* eltwise_param = &(param.eltwise_param);
    ActivationParam<X86>* act_param = &(conv_param->activation_param);
    const Tensor<X86>* weights = conv_param->weight();
    char* ptr_weights_internal = reinterpret_cast<char*>(weights_internal_->data());
    const Tensor<X86>* bias = conv_param->bias();
    Tensor<X86>* input = inputs[0];
    Tensor<X86>* output = outputs[0];
    Shape src_shape(input->valid_shape());
    Shape dst_shape(output->valid_shape());
    Shape wgt_shape(weights->valid_shape());


    // check conf
    if (kernel_) {
        status = check_conf(inputs, outputs, param);

        if (status != SaberNotInitialized) {
            return status;
        }
    }

    // init conf
    conf = jit_1x1_conv_conf_t();
    const bool with_groups = (conv_param->group > 1);
    conf.ngroups = with_groups ? weights->num() : 1;

    conf.mb = src_shape[0];
    conf.ic = src_shape[3];
    conf.ih = src_shape[1];
    conf.iw = src_shape[2];

    conf.oc = dst_shape[3];
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
        conf.sum_scale = _sum_scale;
        conf.sum_dt = param.conv_param.beta_type;
        conf.with_relu = eltwise_param->activation_param.has_active;
        conf.relu_negative_slope = eltwise_param->activation_param.negative_slope;
    }

    conf.with_bias = (bias != NULL && bias->valid_size() > 0);

    if (conf.with_bias) {
        conf.bia_dt = bias->get_dtype();
    }

    DLOG(INFO) << "with bias " << conf.with_bias << ",dt = " << conf.bia_dt;
    conf.src_dt = input->get_dtype();
    conf.signed_input = false;

    if (conf.src_dt == AK_INT8) {
        conf.signed_input = true;
    }

    conf.dst_dt = output->get_dtype();
    conf.typesize_in = type_length(input->get_dtype());
    conf.typesize_out = type_length(output->get_dtype());
    conf.typesize_acc = sizeof(int32_t);
    conf.typesize_bia = conf.with_bias ? type_length(conf.bia_dt) : 0;
    conf.rm = conv_param->rm;

    prepare_rtus(inputs, conf);

    conv_d.n = src_shape[0];
    conv_d.ic = src_shape[3];
    conv_d.ih = src_shape[1];
    conv_d.iw = src_shape[2];
    conv_d.oc = dst_shape[3];
    conv_d.oh = dst_shape[1];
    conv_d.ow = dst_shape[2];
    conv_d.t_pad = conv_param->pad_h;
    conv_d.l_pad = conv_param->pad_w;
    conv_d.stride_h = conv_param->stride_h;
    conv_d.stride_w = conv_param->stride_w;

    status = jit_avx512_core_x8s8s32x_conv1x1_kernel::init_conf(conf, conv_d, anakin_get_max_threads(),
             reduce_src);

    if (status == SaberSuccess) {
        if (kernel_ != nullptr) {
            delete kernel_;
            kernel_ = nullptr;
        }

        kernel_ = new jit_avx512_core_x8s8s32x_conv1x1_kernel(conf);
    } else {
        return SaberUnImplError;
    }

    if (reduce_src) {
        init_rtus_driver<uint8_t>(&rtus_driver_, conf, conv_d, ws_per_thread_, &scratch_);
    }


    float scale_in = inputs[0]->get_scale()[0];
    float scale_out = 1.f;
    if (outputs[0]->get_dtype() == AK_INT8 || outputs[0]->get_dtype() == AK_UINT8) {
        scale_out = outputs[0]->get_scale()[0];
    }
    auto scale_w = weights_internal_->get_scale();
    std::vector<float>().swap(scale_);

    if (inputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_dtype() == AK_INT8) {
        for (int i = 0; i < scale_w.size(); i++) {
            this->scale_.push_back((scale_w[i] * scale_in) / scale_out);
        }
    } else if (inputs[0]->get_dtype() == AK_UINT8 && outputs[0]->get_dtype() == AK_UINT8) {
        for (int i = 0; i < scale_w.size(); i++) {
            this->scale_.push_back((scale_w[i] * scale_in * (127.f / 255.f)) / (scale_out * (127.f / 255.f)));
        }
    } else if (inputs[0]->get_dtype() == AK_UINT8 && outputs[0]->get_dtype() == AK_INT8) {
        for (int i = 0; i < scale_w.size(); i++) {
            this->scale_.push_back((scale_w[i] * scale_in * (127.f / 255.f)) / (scale_out));
        }
    }else if (inputs[0]->get_dtype() == AK_UINT8 && outputs[0]->get_dtype() == AK_FLOAT) {
        for (int i = 0; i < scale_w.size(); i++) {
            this->scale_.push_back((scale_w[i] * scale_in * (127.f / 255.f)));
        }
    } else {
        LOG(FATAL) << "can`t cal scale for dtype " << inputs[0]->get_dtype() << "," <<
                   outputs[0]->get_dtype();
    }

    if (conf.signed_input) {
        if (conf.ver != ver_vnni) {
            if (local_scales_ != nullptr) {
                delete local_scales_;
                local_scales_ = nullptr;
            }

            size_t scales_size = scale_.size();
            local_scales_ = (float*)zmalloc(sizeof(float) * scales_size, 4096);

            if (local_scales_ == nullptr) {
                return SaberOutOfMem;
            }


            for (size_t i = 0; i < scales_size; i++) {
                local_scales_[i] = scale_[i] * 2;
            }
        }
    }

    return SaberSuccess;
}

SaberStatus JitAvx512x8s8s32xConv1x1::dispatch(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param) {
    ConvParam<X86>* conv_param = &(param.conv_param);
    const Tensor<X86>* bias = conv_param->bias();
    //    CHECK_EQ(inputs[0]->get_dtype(), AK_UINT8) << "only support uint8 input";
    // check input and output data type, do scale or not
    const unsigned char* ptr_src = reinterpret_cast<const unsigned char*>(inputs[0]->data());
    const char* ptr_weights = reinterpret_cast<const char*>(weights_internal_->data());
    const float* ptr_bias = nullptr;

    if (bias_internal_ != nullptr && bias_internal_->valid_size() > 0) {
        ptr_bias = reinterpret_cast<const float*>(bias_internal_->data());
    }

    char* ptr_dst = reinterpret_cast<char*>(outputs[0]->mutable_data());
    int dst_type_size = type_length(outputs[0]->get_dtype());

    const auto& jcp = kernel_->jcp;
    const auto& oscales = scale_;
    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;
    int offset = jcp.ngroups * (jcp.oc / jcp.oc_block) * (jcp.ic / jcp.ic_block)
                 * jcp.oc_block * jcp.ic_block;
    const int32_t* compensation = (jcp.signed_input) ? compensation_ : nullptr;
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
        int ithr = anakin_get_thread_num();
        int nthr = anakin_get_num_threads();

        auto p = jit_1x1_conv_call_t();

        auto rp = rtus_driver_t::call_params_t();

        const int nb_oc = jcp.nb_load;
        const int os_block = jcp.bcast_block;

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
            const size_t dst_off = n * jcp.oc_without_padding * jcp.oh * jcp.ow + oh * jcp.ow *
                                   jcp.oc_without_padding
                                   + ow * jcp.oc_without_padding + _ocb * jcp.oc_block;
            const size_t wei_off = ocb * jcp.ic * jcp.kh * jcp.kw * jcp.oc_block
                                   + icb * jcp.kh * jcp.kw * jcp.oc_block * jcp.ic_block;

            p.output_data = ptr_dst + dst_off * dst_type_size;
            // p.load_data = &weights[conf_.with_groups()
            //    ? weights_d.blk_off(g, ocb, icb)
            //    : weights_d.blk_off(ocb, icb)];
            p.load_data = &ptr_weights[wei_off];
            p.bias_data = &ptr_bias[_ocb * jcp.oc_block];
            p.compensation = (jcp.signed_input)
                             ? &compensation[_ocb * jcp.oc_block] : 0;
            p.scales = (jcp.signed_input && jcp.ver != ver_vnni)
                       ? &local_scales_[jcp.is_oc_scale * _ocb * jcp.oc_block]
                       : &oscales[jcp.is_oc_scale * _ocb * jcp.oc_block];

            if (reduce_src) {
                rp.ws = scratch_ + ithr * ws_per_thread_
                        + _icb * jcp.is * jcp.ic_block;

                if (ocb == ocb_start) {
                    // rp.src = src + src_d.blk_off(n, _icb * jcp.ic_block, ih, iw);
                    rp.src = ptr_src + n * jcp.ic_without_padding * jcp.ih * jcp.iw +
                             + ih * jcp.iw * jcp.ic_without_padding + iw * jcp.ic_without_padding + _icb * jcp.ic_block;
                    rtus_driver_->ker_(&rp);
                }

                p.bcast_data = rp.ws;
            } else {
                // p.bcast_data = src + src_d.blk_off(n, _icb * jcp.ic_block, ih, iw);
                p.bcast_data = ptr_src + n * jcp.ic_without_padding * jcp.ih * jcp.iw +
                               + ih * jcp.iw * jcp.ic_without_padding + iw * jcp.ic_without_padding + _icb * jcp.ic_block;;
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

SaberStatus JitAvx512x8s8s32xConv1x1::check_conf(const std::vector<Tensor<X86>*>& inputs,
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
