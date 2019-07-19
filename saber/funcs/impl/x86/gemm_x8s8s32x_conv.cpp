#include "saber/funcs/impl/x86/gemm_x8s8s32x_conv.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/core/tensor_op.h"
#include "mkl_cblas.h"
#include "anakin_thread.h"
#include "debug.h"
namespace anakin {
namespace saber {

using namespace jit;

SaberStatus GemmX8S8S32XConv::init(const std::vector<Tensor<X86>*>& inputs,
                                   std::vector<Tensor<X86>*>& outputs,
                                   ConvEltwiseParam<X86>& param,
                                   Context<X86>& ctx) {
    SaberStatus status = SaberUnImplError;
    ConvParam<X86>* conv_param = &(param.conv_param);

    this->_ctx = &ctx;
    jcp = jit_conv_conf_t();

    status = check_conf(jcp, inputs, outputs, param);

    if (status != SaberSuccess) {
        return status;
    }

    status = init_conf(jcp, inputs, outputs, param);

    if (status != SaberSuccess) {
        return status;
    }

    _acc_tensor.re_alloc(Shape({1, 1, 1, jcp.os* jcp.oc * jcp.nthr}), AK_INT32);
    _col_tensor.re_alloc(Shape({1, 1, 1, jcp.im2col_sz * jcp.nthr}), AK_UINT8);
    _offset_tensor.re_alloc(Shape({1, 1, 1, 1}), AK_INT32);
    return create(inputs, outputs, param, ctx);
}

SaberStatus GemmX8S8S32XConv::create(const std::vector<Tensor<X86>*>& inputs,
                                     std::vector<Tensor<X86>*>& outputs,
                                     ConvEltwiseParam<X86>& param,
                                     Context<X86>& ctx) {
    this->_ctx = &ctx;
    ConvParam<X86>* conv_param = &(param.conv_param);
    auto status = init_conf(jcp, inputs, outputs, param);

    if (status != SaberSuccess) {
        return status;
    }

    Tensor<X86>* weights_orig = conv_param->mutable_weight();

    if (weights_orig->get_dtype() == AK_FLOAT) {
        _weights_scale.re_alloc(weights_orig->valid_shape(), AK_INT8);
        utils::ScaleUtils::scale_conv_weights_to_nchw_host(_weights_scale, *conv_param->weight());
        weights_orig = &_weights_scale;
    }

    CHECK(weights_orig != nullptr);

    if (weights_internal_ != nullptr) {
        delete weights_internal_;
        weights_internal_ = nullptr;
    }

    weights_internal_ = new Tensor<X86>(weights_orig->shape(), AK_INT8);
    weights_internal_->set_scale(weights_orig->get_scale());
    weight_reorder_goihw2hwigo(weights_orig, weights_internal_);

    Tensor<X86>* bias_src = conv_param->mutable_bias();

    if (bias_internal_ != nullptr) {
        delete bias_internal_;
        bias_internal_ = nullptr;
    }

    if (bias_src != nullptr && bias_src->valid_size() > 0) {
        Tensor<X86>* input = inputs[0];
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

    utils::try_expand_tensor(_acc_tensor, jcp.os * jcp.oc * jcp.nthr);
    fill_tensor_const(_acc_tensor, 0);
    acc_ = (int32_t*)_acc_tensor.mutable_data();

    if (acc_ == nullptr) {
        return SaberOutOfMem;
    }

    utils::try_expand_tensor(_col_tensor, jcp.im2col_sz * jcp.nthr);
    fill_tensor_const(_col_tensor, 0);
    col_ = (uint8_t*)_col_tensor.mutable_data();

    if (col_ == nullptr) {
        return SaberOutOfMem;
    }

    if (jcp.signed_input) {
        utils::try_expand_tensor(_offset_tensor, jcp.ngroups * jcp.oc);
        fill_tensor_const(_offset_tensor, 0);
        offset_c_ = (int32_t*)_offset_tensor.mutable_data();

        if (offset_c_ == nullptr) {
            return SaberOutOfMem;
        }

        compute_c_offset(jcp, reinterpret_cast<const int8_t*>(weights_internal_->data()), offset_c_);
    } else {
        utils::try_expand_tensor(_offset_tensor, 1);
        fill_tensor_const(_offset_tensor, 0);
        offset_c_ = (int32_t*)_offset_tensor.mutable_data();

        if (offset_c_ == nullptr) {
            return SaberOutOfMem;
        }
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
    } else if (inputs[0]->get_dtype() == AK_UINT8 && outputs[0]->get_dtype() == AK_FLOAT) {
        for (int i = 0; i < scale_w.size(); i++) {
            this->scale_.push_back((scale_w[i] * scale_in * (127.f / 255.f)));
        }
    } else if (inputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_dtype() == AK_UINT8) {
        for (int i = 0; i < scale_w.size(); i++) {
            this->scale_.push_back((scale_w[i] * scale_in) / (scale_out * (127.f / 255.f)));
        }
    } else if (inputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_dtype() == AK_FLOAT) {
        for (int i = 0; i < scale_w.size(); i++) {
            this->scale_.push_back((scale_w[i] * scale_in));
        }
    } else {
        LOG(FATAL) << "can`t cal scale for dtype " << inputs[0]->get_dtype() << "," <<
                   outputs[0]->get_dtype();
    }

    return SaberSuccess;
}

template <typename InputDtype, typename OutputDtype>
SaberStatus GemmX8S8S32XConv::sub_dispatch(const std::vector<Tensor<X86> *>& inputs,
        std::vector<Tensor<X86> *>& outputs,
        ConvEltwiseParam<X86>& param) {
    ConvParam<X86>* conv_param = &(param.conv_param);
    const Tensor<X86>* bias = conv_param->bias();
    const Tensor<X86>* wei = conv_param->mutable_weight();

    const float* ptr_bias = nullptr;
    const auto oscale = scale_;
    auto* ptr_src = reinterpret_cast<const InputDtype*>(inputs[0]->data());
    auto* ptr_weights = reinterpret_cast<const int8_t*>(weights_internal_->data());
    auto* ptr_dst = reinterpret_cast<OutputDtype*>(outputs[0]->mutable_data());

    if (bias_internal_ != nullptr) {
        ptr_bias = reinterpret_cast<const float*>(bias_internal_->data());
    }

    if (((wei->shape())[0] != 1) || ((wei->shape())[1] != 1)) {
        wei = weights_internal_;
        ptr_weights = reinterpret_cast<const int8_t*>(wei->data());
    }

    const size_t work_amount = jcp.ngroups * jcp.mb;
    const size_t src_mb_stride = jcp.ngroups * jcp.ih * jcp.iw * jcp.ic;
    const size_t src_g_stride  = jcp.ic;
    const size_t wei_g_stride  = (jcp.ngroups > 1) ? jcp.oc : 0;
    const size_t dst_mb_stride = jcp.ngroups * jcp.oh * jcp.ow * jcp.oc;
    const size_t dst_g_stride  = jcp.oc;
    const size_t dst_os_stride = jcp.oc * jcp.ngroups;
    const bool do_relu = jcp.with_relu;
    const int32_t ithr = 0;
    const int32_t nthr = 1;
    //    parallel(jcp.nthr, [&](const int32_t ithr, const int32_t nthr) {
    auto col = col_ + (ptrdiff_t) ithr * jcp.im2col_sz;
    auto acc = acc_ + (ptrdiff_t) ithr * jcp.os * jcp.oc;

    int32_t n = 0, g = 0;
    size_t start = 0, end = 0;
    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups);

    for (auto iwork = start; iwork < end; ++iwork) {
        auto src = ptr_src + n * src_mb_stride + g * src_g_stride;
        auto wei = ptr_weights + g * wei_g_stride;
        auto dst = ptr_dst + n * dst_mb_stride + g * dst_g_stride;

        if (jcp.need_im2col) {
            im2col_u8(jcp, (const uint8_t*)src, col);
        }

        auto M = jcp.oc;
        auto K = jcp.ks * jcp.ic;
        auto N = jcp.os;
        int8_t offset_a = 0, offset_b = 0;

        if (jcp.signed_input) {
            cblas_gemm_s8u8s32(CblasColMajor, CblasNoTrans, CblasNoTrans,
                               CblasColOffset, M, N, K, 1.f, wei, M * jcp.ngroups,
                               offset_a, jcp.need_im2col ? col : (const uint8_t*)src, K, offset_b,
                               0.f, acc, M, offset_c_ + g * jcp.oc);
        } else {
            cblas_gemm_s8u8s32(CblasColMajor, CblasNoTrans, CblasNoTrans,
                               CblasFixOffset, M, N, K, 1.f, wei, M * jcp.ngroups,
                               offset_a, jcp.need_im2col ? col : (const uint8_t*)src, K, offset_b,
                               0.f, acc, M, offset_c_);
        }


        for (auto os = 0; os < jcp.os; ++os) {
            for (auto oc = 0; oc < jcp.oc; ++oc) {
                auto acc_off = os * jcp.oc + oc;
                auto g_oc = g * jcp.oc + oc;

                auto d = (float) acc[acc_off];

                if (jcp.with_bias) {
                    d += *(ptr_bias + g_oc);
                }

                d *= oscale[g_oc];

                if (do_relu && d < 0) {
                    d = 0;
                }

                auto dst_off = os * dst_os_stride + oc;

                if (std::is_same<OutputDtype, float>::value) {
                    dst[dst_off] = d;
                } else {
                    dst[dst_off] = (OutputDtype) nearbyintf(d);
                }
            }
        }

        nd_iterator_step(n, jcp.mb, g, jcp.ngroups);
    }

    //    });
    return SaberSuccess;
}
SaberStatus GemmX8S8S32XConv::dispatch(const std::vector<Tensor<X86>*>& inputs,
                                       std::vector<Tensor<X86>*>& outputs,
                                       ConvEltwiseParam<X86>& param) {
    DLOG(INFO) << "dispatch GemmX8S8S32XConv";

    if (inputs[0]->get_dtype() == AK_UINT8 && outputs[0]->get_dtype() == AK_FLOAT) {
        return this->template sub_dispatch<uint8_t, float>(inputs, outputs, param);
    } else if (inputs[0]->get_dtype() == AK_UINT8 && outputs[0]->get_dtype() == AK_UINT8) {
        return this->template sub_dispatch<uint8_t, uint8_t>(inputs, outputs, param);
    } else if (inputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_dtype() == AK_UINT8) {
        return this->template sub_dispatch<int8_t, uint8_t>(inputs, outputs, param);
    } else if (inputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_dtype() == AK_INT8) {
        return this->template sub_dispatch<int8_t, int8_t>(inputs, outputs, param);
    } else if (inputs[0]->get_dtype() == AK_INT8 && outputs[0]->get_dtype() == AK_FLOAT) {
        return this->template sub_dispatch<int8_t, float>(inputs, outputs, param);
    } else {
        LOG(FATAL) << "not support";
        return SaberSuccess;
    }
}

SaberStatus GemmX8S8S32XConv::check_conf(const jit_conv_conf_t& jcp,
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86>& param) {
    ConvParam<X86>* conv_param = &(param.conv_param);
    ActivationParam<X86>* act_param = &(conv_param->activation_param);
    Tensor<X86> const* weights = conv_param->weight();
    Tensor<X86> const* bias = conv_param->bias();
    Tensor<X86> const* input = inputs[0];
    Tensor<X86>* output = outputs[0];
    Shape src_shape = input->shape();
    Shape dst_shape = output->shape();
    Shape wgt_shape = weights->shape();
    auto group = conv_param->group;

    CHECK(input != nullptr);
    CHECK(output != nullptr);
    CHECK(weights != nullptr);

    if (weights_internal_ != nullptr) {
        delete weights_internal_;
        weights_internal_ = nullptr;
    }

    if (bias_internal_ != nullptr) {
        delete bias_internal_;
        bias_internal_ = nullptr;
    }

    auto ic_check = src_shape[3] % group;
    auto oc_check = dst_shape[3] % group;

    if ((group > 1) & ((ic_check + oc_check) > 0)) {
        LOG(ERROR) << "invalid input_channel or output_channel";
        return SaberInvalidValue;
    }

    return SaberSuccess;
}

SaberStatus GemmX8S8S32XConv::init_conf(jit_conv_conf_t& jcp,
                                        const std::vector<Tensor<X86>*>& inputs,
                                        std::vector<Tensor<X86>*>& outputs,
                                        ConvEltwiseParam<X86>& param) {
    SaberStatus status = SaberSuccess;
    ConvParam<X86>* conv_param = &(param.conv_param);
    ActivationParam<X86>* act_param = &(conv_param->activation_param);
    Tensor<X86> const* weights = conv_param->weight();
    Tensor<X86> const* bias = conv_param->bias();
    Tensor<X86> const* input = inputs[0];
    Tensor<X86>* output = outputs[0];
    Shape src_shape = input->shape();
    Shape dst_shape = output->shape();
    Shape wgt_shape = weights->shape();

    jcp.signed_input = (input->get_dtype() == AK_INT8) ? true : false;
    jcp.ngroups = conv_param->group;
    jcp.mb = src_shape[0];
    jcp.ih = src_shape[1];
    jcp.iw = src_shape[2];
    jcp.ic = src_shape[3] / jcp.ngroups;
    jcp.oh = dst_shape[1];
    jcp.ow = dst_shape[2];
    jcp.oc = dst_shape[3] / jcp.ngroups;
    jcp.kh = wgt_shape[2];
    jcp.kw = wgt_shape[3];
    jcp.is = jcp.ih * jcp.iw;
    jcp.os = jcp.oh * jcp.ow;
    jcp.ks = jcp.kh * jcp.kw;
    jcp.stride_h = conv_param->stride_h;
    jcp.stride_w = conv_param->stride_w;
    jcp.t_pad    = conv_param->pad_h;
    jcp.l_pad    = conv_param->pad_w;
    jcp.b_pad    = conv_param->pad_h;
    jcp.r_pad    = conv_param->pad_w;
    jcp.dilate_h = conv_param->dilation_h;
    jcp.dilate_w = conv_param->dilation_w;
    jcp.rm       = conv_param->rm;
    jcp.ur_h     = 1;
    jcp.im2col_sz   = (ptrdiff_t)jcp.ic * jcp.ks * jcp.os;
    jcp.need_im2col = !(jcp.oh == jcp.ih &&
                        jcp.ow == jcp.iw &&
                        jcp.ks == 1 &&
                        jcp.ngroups == 1 &&
                        jcp.signed_input == false);

    auto mb_ngroup = jcp.mb * jcp.ngroups;
    auto omp_max_threads = anakin_get_max_threads();
    auto omp_mb_ngroup_threads = mb_ngroup < omp_max_threads ?
                                 mb_ngroup :
                                 omp_max_threads;

    if (jcp.mb != 1) {
        jcp.nthr = omp_mb_ngroup_threads;
    } else {
        jcp.nthr = mb_ngroup > omp_max_threads / 2 ?
                   omp_mb_ngroup_threads : 1;
    }

    im2col_u8_method = 1;

    if (jcp.kh * jcp.kw != 1 && jcp.mb != 1) {
        im2col_u8_method = 2;
    }

    jcp.with_bias = (bias != NULL && bias->valid_size() > 0);
    jcp.with_relu = conv_param->activation_param.has_active;

    if (jcp.with_relu) {
        jcp.relu_negative_slope = static_cast<float>(act_param->negative_slope);
    }

    return SaberSuccess;
}

SaberStatus GemmX8S8S32XConv::weight_reorder_goihw2hwigo(Tensor<X86>* in,
        Tensor<X86>* out) {
    auto src = reinterpret_cast<const int8_t*>(in->data());
    auto dst = reinterpret_cast<int8_t*>(out->mutable_data());

    if ((src == nullptr) || (dst == nullptr)) {
        LOG(ERROR) << "invalid empty pointer";
        return SaberInvalidValue;
    }

    Shape shape = in->shape();
    auto oc_value = shape[0];
    auto ic_value = shape[1];
    auto kh_value = shape[2];
    auto kw_value = shape[3];
    auto src_index = 0, dst_index = 0;


    for (auto oc = 0; oc < oc_value; oc++) {
        for (auto ic = 0; ic < ic_value; ic++) {
            for (auto kh = 0; kh < kh_value; kh++) {
                for (auto kw = 0; kw < kw_value; kw++) {
                    src_index = ((oc * ic_value + ic) * kh_value + kh) * kw_value + kw;
                    dst_index = ((kh * kw_value + kw) * ic_value + ic) * oc_value + oc;
                    dst[dst_index] = src[src_index];
                }
            }
        }
    }


    return SaberSuccess;
}

SaberStatus GemmX8S8S32XConv::compute_c_offset(const jit_conv_conf_t& jcp,
        const int8_t* src,
        int32_t* dst) {
    if (src == nullptr || dst == nullptr) {
        LOG(FATAL) << "invalid empty pointer";
        return SaberInvalidValue;
    }

    auto g_value = jcp.ngroups;
    auto oc_value = jcp.oc;
    auto ks_value = jcp.ks;
    auto ic_value = jcp.ic;

    auto k_value = ks_value * ic_value,
         g_oc_value = g_value * oc_value;

    for (auto k = 0; k < k_value; ++k) {
        #pragma omp simd

        for (auto g_oc = 0; g_oc < g_oc_value; ++g_oc) {
            auto src_index = k * g_oc_value + g_oc;
            dst[g_oc] += -128 * src[src_index];
        }
    }

    return SaberSuccess;
}


SaberStatus GemmX8S8S32XConv::im2col_u8(const jit_conv_conf_t& jcp,
                                        const unsigned char* im,
                                        unsigned char* col) {
    auto jcp_oh = jcp.oh;
    auto jcp_ow = jcp.ow;
    auto jcp_kh = jcp.kh;
    auto jcp_kw = jcp.kw;
    auto jcp_t_pad = jcp.t_pad;
    auto jcp_l_pad = jcp.l_pad;
    auto jcp_stride_h = jcp.stride_h;
    auto jcp_stride_w = jcp.stride_w;
    auto jcp_ic = jcp.ic;
    auto jcp_ngroups = jcp.ngroups;

    switch (im2col_u8_method) {
    case 1:
        parallel_nd(jcp.oh, jcp.ow, [&](int32_t oh, int32_t ow) {
            for (auto kh = 0; kh < jcp.kh; ++kh) {
                const auto ih = oh * jcp.stride_h - jcp.t_pad + kh * jcp.dilate_h;

                for (auto kw = 0; kw < jcp.kw; ++kw) {
                    const auto iw = ow * jcp.stride_w - jcp.l_pad + kw * jcp.dilate_w;

                    const size_t col_idx = (((oh * jcp.ow + ow) * jcp.kh + kh) * jcp.kw + kw) * jcp.ic;
                    const size_t im_idx = (ih * jcp.iw + iw) * jcp.ngroups * jcp.ic;
                    #pragma omp simd

                    for (auto ic = 0; ic < jcp.ic; ++ic) {
                        if (iw < 0 || iw >= jcp.iw || ih < 0 || ih >= jcp.ih) {
                            if (jcp.signed_input) {
                                col[col_idx + ic] = 128;
                            } else {
                                col[col_idx + ic] = 0;
                            }
                        } else {
                            col[col_idx + ic] = jcp.signed_input ?
                                                128 + im[im_idx + ic] :
                                                im[im_idx + ic];
                        }
                    }
                }
            }
        });

        break;

    case 2:
        #pragma omp parallel for collapse(2) num_threads(jcp.nthr)
        for (auto oh = 0; oh < jcp.oh; ++oh) {
            for (auto ow = 0; ow < jcp.ow; ++ow) {
                for (auto kh = 0; kh < jcp.kh; ++kh) {
                    const auto ih = oh * jcp.stride_h - jcp.t_pad + kh * jcp.dilate_h;

                    for (auto kw = 0; kw < jcp.kw; ++kw) {
                        const auto iw = ow * jcp.stride_w - jcp.l_pad + kw * jcp.dilate_w;

                        const auto col_idx = (((oh * jcp.ow + ow) * jcp.kh + kh) * jcp.kw + kw) * jcp.ic;
                        const auto im_idx = (ih * jcp.iw + iw) * jcp.ngroups * jcp.ic;
                        #pragma omp simd

                        for (auto ic = 0; ic < jcp.ic; ++ic) {
                            if (iw < 0 || iw >= jcp.iw || ih < 0 || ih >= jcp.ih) {
                                if (jcp.signed_input) {
                                    col[col_idx + ic] = 128;
                                } else {
                                    col[col_idx + ic] = 0;
                                }
                            } else {
                                col[col_idx + ic] = jcp.signed_input ?
                                                    128 + im[im_idx + ic] :
                                                    im[im_idx + ic];
                            }
                        }
                    }
                }
            }
        }

        break;
    }

    return SaberSuccess;
}

} // namespace saber
} // namespace anakin
