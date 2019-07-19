#include "saber/funcs/impl/x86/kernel/jit_avx512_core_x8s8s32x_conv.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/funcs/impl/x86/anakin_thread.h"

#include "debug.h"
#include "tensor_op.h"
namespace anakin {
namespace saber {

using namespace jit;

SaberStatus JitAvx512X8S8S32XConv::init(const std::vector<Tensor<X86>*>& inputs,
                                        std::vector<Tensor<X86>*>& outputs,
                                        ConvEltwiseParam<X86>& param,
                                        Context<X86>& ctx) {
    this->_ctx = &ctx;
    ConvParam<X86>* conv_param = &(param.conv_param);
    const Tensor<X86>* weights = conv_param->weight();
    Shape wgt_shape(weights->valid_shape());
    Tensor<X86>* input = inputs[0];
    bool depthwise = (conv_param->group > 1) && (wgt_shape[1] == 1);
    bool is_depthwise_int8 = depthwise&& inputs[0]->get_dtype()==AK_INT8;

//    if ((input->get_dtype() == AK_INT8) && depthwise) {
//        LOG(FATAL) << "depthwise conv is not supported if input is int8!";
//        return SaberUnImplError;
//    }

    CHECK_GT(inputs[0]->get_scale().size(), 0) << "input scale must >0";
    CHECK(outputs[0]->get_scale().size() > 0 || outputs[0]->get_dtype() == AK_FLOAT) << "output scale must >0";

    // padding weights
    Tensor<X86>* weights_orig = conv_param->mutable_weight();

    ///transe///
    if (weights_orig->get_dtype() == AK_FLOAT) {
        _weights_scale.re_alloc(weights_orig->valid_shape(), AK_INT8);
        utils::ScaleUtils::scale_conv_weights_to_nchw_host(_weights_scale, *conv_param->weight());
        weights_orig = &_weights_scale;
    }

    const Tensor<X86>* bias_src = conv_param->bias();

    // TODO check bias, do scale or not?
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

    ///transe///

    int ch_blk = 16;
    Shape shape = weights_orig->valid_shape();

    if (depthwise) {
        int g_value = shape[0];
        int c_value = shape[1];
        int kh_value = shape[2];
        int kw_value = shape[3];
        int g_padding = utils::rnd_up(g_value, ch_blk);
        Shape shape_padding({g_padding, c_value, kh_value, kw_value}, weights_orig->get_layout());

        if (weights_padding_ != nullptr) {
            delete weights_padding_;
            weights_padding_ = nullptr;
        }

        weights_padding_ = new Tensor<X86>(shape_padding, weights_orig->get_dtype());
        weights_padding_->set_scale(weights_orig->get_scale());
        weight_padding_nhwc(weights_orig, weights_padding_);
    } else if (conv_param->group == 1) {
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

        weights_padding_ = new Tensor<X86>(shape_padding, weights_orig->get_dtype());
        weights_padding_->set_scale(weights_orig->get_scale());
        weight_padding_nhwc(weights_orig, weights_padding_);
    } else {
        LOG(FATAL) << "not impl";
        return SaberUnImplError;
    }

    if (input->get_dtype() == AK_INT8 && !is_depthwise_int8) {
        if (compensation_ != nullptr) {
            delete compensation_;
            compensation_ = nullptr;
        }

        int oc_value = shape[0];
        int ic_value = shape[1];
        int kh_value = shape[2];
        int kw_value = shape[3];
        int oc_padding = utils::rnd_up(oc_value, ch_blk);
        int ic_padding = utils::rnd_up(ic_value, ch_blk);

        compensation_ = (int32_t*)zmalloc(sizeof(int32_t) * oc_padding, 4096);
        memset(compensation_, 0, sizeof(int32_t) * oc_padding);
        float sum = 0;
        float temp = 0;
        int32_t offset = 0;
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
    // TODO check weights, do scale or not?
    if (weights_internal_ != nullptr) {
        delete weights_internal_;
        weights_internal_ = nullptr;
    }

    weights_internal_ = new Tensor<X86>(weights_padding_->valid_shape(), AK_INT8);
    weights_internal_->set_scale(weights_padding_->get_scale());

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

    if (depthwise) {
        weight_reorder_Goihw16g(*weights_padding_, *weights_internal_);
    } else if (conv_param->group == 1) {
        weight_reorder_OIhw4i16o4i(*weights_padding_, *weights_internal_, weights_padding_->get_scale());
    } else {
        return SaberUnImplError;
    }

    return create(inputs, outputs, param, ctx);
}

SaberStatus JitAvx512X8S8S32XConv::create(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param,
        Context<X86>& ctx) {
    SaberStatus status = SaberSuccess;

    ConvParam<X86>* conv_param = &(param.conv_param);
    jit_conv_conf_t jcp = jit_conv_conf_t();

    status = init_conf(jcp, inputs, outputs, param);

    if (status != SaberSuccess) {
        LOG(FATAL) << "create failed";
        return status;
    }

    float scale_in = inputs[0]->get_scale()[0];
    float scale_out = 1.f;

    if (outputs[0]->get_scale().size() > 0 && outputs[0]->get_dtype() != AK_FLOAT) {
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
    }else if (inputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_dtype() == AK_UINT8) {
        for (int i = 0; i < scale_w.size(); i++) {
            this->scale_.push_back((scale_w[i] * scale_in) / (scale_out*(127.f/255.f)));
        }
    } else if (inputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_dtype() == AK_FLOAT) {
        for (int i = 0; i < scale_w.size(); i++) {
            this->scale_.push_back((scale_w[i] * scale_in));
        }
    } else {
        LOG(FATAL) << "can`t cal scale for dtype " << inputs[0]->get_dtype() << "," <<
                           outputs[0]->get_dtype();
    }

    if (jcp.signed_input) {
        if (jcp.ver != ver_vnni) {
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

    return status;
}

SaberStatus JitAvx512X8S8S32XConv::dispatch(const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param) {
    ConvParam<X86>* conv_param = &(param.conv_param);
    const Tensor<X86>* bias = conv_param->bias();
    CHECK_EQ(inputs[0]->get_layout(), Layout_NHWC) << "only support nhwc";

    if ((inputs[0]->get_dtype() != AK_UINT8) &&
            (inputs[0]->get_dtype() != AK_INT8)) {
        LOG(ERROR) << "only support uint8 and int8 input type!";
        return SaberUnImplError;
    }

    const unsigned char* ptr_src = reinterpret_cast<const unsigned char*>(inputs[0]->data());
    const char* ptr_weights = reinterpret_cast<const char*>(weights_internal_->data());
    const float* ptr_bias = nullptr;

    if (bias_internal_ != nullptr) {
        ptr_bias = reinterpret_cast<const float*>(bias_internal_->data());
    }

    char* ptr_dst = reinterpret_cast<char*>(outputs[0]->mutable_data());

    int dst_type_size = type_length(outputs[0]->get_dtype());
    const auto& jcp = kernel_->jcp;
    const auto oscales = scale_;//std::vector<float>({1.f});//scale_;

    int32_t* compensation = (jcp.signed_input) ? compensation_ : nullptr;

    parallel(0, [&](const int ithr, const int nthr) {
        int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;

        int nb_groups = jcp.nb_ch;
        int group_block = jcp.ch_block;

        int start{0}, end{0};
        int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh;
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_call_t();

        size_t src_h_stride = jcp.iw * jcp.ic_without_padding;
        size_t dst_h_stride = jcp.ow * jcp.oc_without_padding;
        size_t wht_h_stride = jcp.kw * jcp.ic_block * jcp.oc_block;

        if (jcp.is_dw) {
            src_h_stride = jcp.iw * jcp.ic_without_padding * jcp.ngroups;
            dst_h_stride = jcp.ow * jcp.oc_without_padding * jcp.ngroups;
            wht_h_stride = jcp.kw * jcp.ch_block;
        }

        int n{0}, gb{0}, occ{0}, oh_s{0};

        if (jcp.loop_order == loop_cgn) {
            nd_iterator_init(start, occ, oc_chunks, gb, nb_groups, n, jcp.mb, oh_s, jcp.oh);
        } else if (jcp.loop_order == loop_gnc) {
            nd_iterator_init(start, gb, nb_groups, n, jcp.mb, occ, oc_chunks, oh_s, jcp.oh);
        } else if (jcp.loop_order == loop_ngc) {
            nd_iterator_init(start, n, jcp.mb, gb, nb_groups, occ, oc_chunks, oh_s, jcp.oh);
        } else {
            assert(!"unsupported loop order");
        }

        while (start < end) {
            int ocb = occ * jcp.nb_oc_blocking;
            int g = gb * group_block;
            int g_oc = (g * jcp.nb_oc + ocb) * jcp.oc_block;

            int g_ic = g * jcp.nb_ic * jcp.ic_block;

            int work_rem = end - start;
            int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

            size_t bias_blk_off = g_oc;
            size_t dst_blk_off = n * jcp.oc_without_padding * jcp.oh * jcp.ow +
                                 oh_s * jcp.ow * jcp.oc_without_padding + g_oc;
            size_t src_blk_off = n * jcp.ic_without_padding * jcp.ih * jcp.iw +
                                 ih_s * jcp.iw * jcp.ic_without_padding + g_ic;
            size_t weight_blk_off = ocb * jcp.ic * jcp.kh * jcp.kw * jcp.oc_block;

            if (jcp.is_dw) {
                dst_blk_off = n * jcp.oh * jcp.ow * jcp.ngroups + g_oc + oh_s * jcp.ow * jcp.ngroups;
                src_blk_off = n * jcp.ih * jcp.iw * jcp.ngroups + g_ic + ih_s * jcp.iw * jcp.ngroups;
                weight_blk_off =  gb * jcp.kh * jcp.kw * jcp.ch_block + ocb * jcp.kh * jcp.kw * jcp.ch_block;
            }

            auto bias_w = ptr_bias ? ptr_bias + bias_blk_off : 0;

            auto dst_w = ptr_dst + dst_blk_off * dst_type_size;
            auto src_w = ptr_src + src_blk_off;
            auto wht_w = ptr_weights + weight_blk_off;
            auto scales = (jcp.signed_input && jcp.ver != ver_vnni)
                          ? &local_scales_[jcp.is_oc_scale * g_oc]
                          : &oscales[jcp.is_oc_scale * g_oc];

            for (int oj = oh_s, ij = ih_s;
                    oj < oh_e; ++oj, ij += jcp.stride_h) {
                int dilate_h = jcp.dilate_h + 1;
                int i_t_overflow = utils::min(jcp.kh, utils::div_up(utils::max(0, -ij), dilate_h));
                int i_b_overflow = utils::min(jcp.kh, utils::div_up(utils::max(0,
                                              ij - jcp.ih + (jcp.kh - 1) * dilate_h + 1),
                                              dilate_h));
                int kh_padding = utils::max(0, jcp.kh - i_t_overflow - i_b_overflow);

                size_t wei_stride = (!jcp.signed_input) ? i_t_overflow * wht_h_stride : 0;

                p.src = src_w + i_t_overflow * dilate_h * src_h_stride;
                p.dst = dst_w;
                p.filt = wht_w + + wei_stride;
                p.bias = bias_w;
                p.oc_blocks = jcp.is_dw ? gb : ocb;
                p.kh_padding = kh_padding;
                p.compensation = (jcp.signed_input) ? &compensation[g_oc] : nullptr;
                p.scales = scales;
                p.t_overflow = i_t_overflow;
                p.b_overflow = i_b_overflow;
                kernel_->jit_ker(&p);

                src_w += src_h_stride * jcp.stride_h;
                dst_w += dst_h_stride * dst_type_size;
            }

            if (jcp.loop_order == loop_cgn) {
                nd_iterator_jump(start, end, occ, oc_chunks, gb, nb_groups, n,
                                 jcp.mb, oh_s, jcp.oh);
            } else if (jcp.loop_order == loop_gnc) {
                nd_iterator_jump(start, end, gb, nb_groups, n, jcp.mb, occ,
                                 oc_chunks, oh_s, jcp.oh);
            } else if (jcp.loop_order == loop_ngc) {
                nd_iterator_jump(start, end, n, jcp.mb, gb, nb_groups, occ,
                                 oc_chunks, oh_s, jcp.oh);
            } else {
                assert(!"unsupported loop order");
            }
        }
    });

    return SaberSuccess;
}

SaberStatus JitAvx512X8S8S32XConv::init_conf(jit_conv_conf_t& jcp,
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param) {
    SaberStatus status;
    ConvParam<X86>* conv_param = &(param.conv_param);
    EltwiseParam<X86>* eltwise_param = &(param.eltwise_param);
    ActivationParam<X86>* act_param = &(conv_param->activation_param);
    const Tensor<X86>* weights = weights_internal_;
    const Tensor<X86>* bias = bias_internal_;
    Tensor<X86>* input = inputs[0];
    Tensor<X86>* output = outputs[0];
    Shape src_shape(input->valid_shape());
    Shape dst_shape(output->valid_shape());
    Shape wgt_shape(weights->valid_shape());

    // init conf
    const bool with_groups = (conv_param->group > 1);
    jcp.ngroups = with_groups ? conv_param->group : 1;

    jcp.mb = src_shape.num();
    jcp.ic = src_shape.channel() / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.ih = src_shape.height();
    jcp.iw = src_shape.width();
    jcp.oc = dst_shape.channel() / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.oh = dst_shape.height();
    jcp.ow = dst_shape.width();

    jcp.kh = wgt_shape.height();
    jcp.kw = wgt_shape.width();

    jcp.stride_h = conv_param->stride_h;
    jcp.stride_w = conv_param->stride_w;
    jcp.t_pad = conv_param->pad_h;
    jcp.l_pad = conv_param->pad_w;
    jcp.b_pad = conv_param->pad_h;
    jcp.r_pad = conv_param->pad_w;
    jcp.dilate_h = conv_param->dilation_h <= 0 ? 0 : (conv_param->dilation_h - 1);
    jcp.dilate_w = conv_param->dilation_w <= 0 ? 0 : (conv_param->dilation_w - 1);

    if (bias != nullptr && bias->valid_size() > 0) {
        jcp.bia_dt = bias->get_dtype();
    }

    DLOG(INFO) << "bia dt = " << jcp.bia_dt;
    jcp.dst_dt = output->get_dtype();
    jcp.rm = conv_param->rm;
    jcp.ur_h = 1;

    jcp.with_bias = (bias != nullptr && bias->valid_size() > 0);
    jcp.with_relu = conv_param->activation_param.has_active;

    if (jcp.with_relu) {
        jcp.relu_negative_slope = static_cast<float>(act_param->negative_slope);
    }
    jcp.is_dw = with_groups && (jcp.ic == 1);
    jcp.is_dw_int8 = jcp.is_dw&& (input->get_dtype() == AK_INT8);
    jcp.signed_input = (input->get_dtype() == AK_INT8 && !jcp.is_dw_int8) ? true : false;
    DLOG(INFO) << "ak signed = " << jcp.signed_input;


    if (jcp.is_dw && jcp.signed_input) {
        LOG(ERROR) << "depthwise conv is not supported if input is int8!";
        return SaberUnImplError;
    }

    jcp.with_sum = eltwise_param->has_eltwise && (eltwise_param->operation == Eltwise_sum);

    if (jcp.with_sum) {
        jcp.sum_scale = _sum_scale;
        jcp.sum_dt = param.conv_param.beta_type;
        jcp.with_relu = eltwise_param->activation_param.has_active;
        jcp.relu_negative_slope = eltwise_param->activation_param.negative_slope;
    }

    status = jit_avx512_core_x8s8s32x_fwd_kernel::init_conf(jcp);

    if (status == SaberSuccess) {
        if (kernel_ != nullptr) {
            delete kernel_;
            kernel_ = nullptr;
        }

        kernel_ = new jit_avx512_core_x8s8s32x_fwd_kernel(jcp);
    } else {
        LOG(FATAL) << "SaberUnImplError";
        return SaberUnImplError;
    }


    const int nthreads = anakin_get_max_threads();
    ws_per_thread_ = jcp.oh * jcp.ow * jcp.oc;
    ws_ = (int*)zmalloc(nthreads * ws_per_thread_ * sizeof(int), 4096);

    if (!ws_) {
        LOG(FATAL) << "workspace allocation failed";
        delete kernel_;
        kernel_ = nullptr;
        return SaberOutOfMem;
    }

    return SaberSuccess;
}

SaberStatus JitAvx512X8S8S32XConv::check_conf(const jit_conv_conf_t& jcp,
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param) {
    return SaberSuccess;
}

} // namespace saber
} // namespace anakin
