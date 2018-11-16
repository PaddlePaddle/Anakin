#include "saber/funcs/impl/x86/gemm_u8s8s32x_conv.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "mkl_cblas.h"
#include "anakin_thread.h"

namespace anakin {
namespace saber {

using namespace jit;

SaberStatus GemmU8S8S32XConv::init(const std::vector<Tensor<X86>*> &inputs,
                                   std::vector<Tensor<X86>*> &outputs,
                                   ConvEltwiseParam<X86> &param,
                                   Context<X86> &ctx) {
    ConvParam<X86> *conv_param = &(param.conv_param);
    this->_ctx = &ctx;

    Tensor<X86> *weights_reorder = conv_param->mutable_weight();
    if (weights_reorder == nullptr || weights_reorder->mutable_data() == nullptr) {
        return SaberInvalidValue;
    }
    if (weights_internal_ != nullptr) {
        delete weights_internal_;
        weights_internal_ = nullptr;
    }
    weights_internal_ = new Tensor<X86>(weights_reorder->shape(), AK_INT8);
    weights_internal_->set_scale(weights_reorder->get_scale());
    weight_reorder_oihw2hwio(weights_reorder, weights_internal_);

    return create(inputs, outputs, param, ctx);
}

SaberStatus GemmU8S8S32XConv::create(const std::vector<Tensor<X86>*> &inputs,
                                     std::vector<Tensor<X86>*> &outputs,
                                     ConvEltwiseParam<X86> &param,
                                     Context<X86> &ctx) {
    SaberStatus status = SaberSuccess;
    ConvParam<X86> *conv_param = &(param.conv_param);

    status = init_conf(jcp, inputs, outputs, param);
    if (status != SaberSuccess) {
        return status;
    }

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

SaberStatus GemmU8S8S32XConv::dispatch(const std::vector<Tensor<X86>*> &inputs,
                                       std::vector<Tensor<X86>*> &outputs,
                                       ConvEltwiseParam<X86> &param) {
    ConvParam<X86> *conv_param = &(param.conv_param);
    const Tensor<X86> *bias = conv_param->bias();
    Tensor<X86> *wei = conv_param->mutable_weight();

    CHECK_EQ(inputs[0]->get_dtype(), AK_UINT8) << "only support uint8 input type";
    const unsigned char *ptr_src = reinterpret_cast<const unsigned char*>(inputs[0]->data());
    const char *ptr_weights = reinterpret_cast<const char*>(weights_internal_->data());
    unsigned char *ptr_dst = reinterpret_cast<unsigned char *>(outputs[0]->mutable_data());
    const int32_t *ptr_bias = nullptr;
    int dst_type_size = type_length(outputs[0]->get_dtype());
    const auto oscale = scale_;

    if (bias_internal_ != nullptr) {
        ptr_bias = reinterpret_cast<const int32_t*>(bias_internal_->data());
    }

    if (((wei->shape())[0] != 1) || ((wei->shape())[1] != 1)) {
        wei = weights_internal_;
        ptr_weights = reinterpret_cast<const char*>(wei->data());
    }

    const size_t work_amount = jcp.ngroups * jcp.mb;
    const size_t src_mb_stride = jcp.ngroups * jcp.ih * jcp.iw * jcp.ic;
    const size_t src_g_stride  = jcp.ic;
    const size_t wei_g_stride  = (jcp.is_dw || jcp.ngroups > 1) ? jcp.oc : 0;
    const size_t dst_mb_stride = jcp.ngroups * jcp.oh * jcp.ow * jcp.oc;
    const size_t dst_g_stride  = jcp.oc;
    const size_t dst_os_stride = jcp.oc * jcp.ngroups;
    const bool do_relu = jcp.with_relu;

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        unsigned char *col = col_ + (ptrdiff_t) ithr * jcp.im2col_sz;
        int32_t *acc = acc_ + (ptrdiff_t) ithr * jcp.os * jcp.oc;

        int n{0}, g{0};
        size_t start = 0, end = 0;
        utils::balance211 (work_amount, nthr, ithr, start, end);
        utils::nd_iterator_init (start, n, jcp.mb, g, jcp.ngroups);

        for (size_t iwork = start; iwork < end; ++iwork) {
            const unsigned char *src = ptr_src + n * src_mb_stride + g * src_g_stride;
            const char *wei = ptr_weights + g * wei_g_stride;
            unsigned char *dst = ptr_dst + n * dst_mb_stride + g * dst_g_stride;

            if (jcp.need_im2col) {
                im2col_u8 (jcp, src, col);
            }

            const int M = jcp.oc;
            const int K = jcp.ks * jcp.ic;
            const int N = jcp.os;
            const int8_t off_a = 0, off_b = 0;
            const int32_t off_c = 0;

            cblas_gemm_s8u8s32 (CblasColMajor, CblasNoTrans, CblasNoTrans,
                                CblasFixOffset, M, N, K, 1., wei, M * jcp.ngroups,
                                off_a, jcp.need_im2col ? col : src, K, off_b, 0., acc,
                                M, (const int *) &off_c);

            #pragma omp parallel for collapse(2)
            for (int os = 0; os < jcp.os; ++os) {
                for (int oc = 0; oc < jcp.oc; ++oc) {
                    size_t acc_off = os * jcp.oc + oc;

                    float d = (float) acc[acc_off];
                    if (jcp.with_bias) {
                        d += *(ptr_bias + g * jcp.oc + oc);
                    }

                    d *= oscale[g * jcp.oc + oc];
                    if (do_relu)
                        d = (d < 0) ? 0 : d;
                    const size_t dst_off = os * dst_os_stride + oc;
                    dst[dst_off] = (uint8_t) nearbyintf(d);
                }
            }

            utils::nd_iterator_step (n, jcp.mb, g, jcp.ngroups);
        }
    });

    return SaberSuccess;
}

SaberStatus GemmU8S8S32XConv::init_conf(jit_conv_conf_t &jcp,
                                        const std::vector<Tensor<X86>*> &inputs,
                                        std::vector<Tensor<X86>*> &outputs,
                                        ConvEltwiseParam<X86> &param) {
    SaberStatus status = SaberSuccess;
    ConvParam<X86> *conv_param = &(param.conv_param);
    ActivationParam<X86> *act_param = &(conv_param->activation_param);
    const Tensor<X86> *weights = conv_param->weight();
    const Tensor<X86> *bias = conv_param->bias();
    Tensor<X86> *input = inputs[0];
    Tensor<X86> *output = outputs[0];
    Shape src_shape;
    Shape dst_shape;
    Shape wgt_shape;

    if ((input == nullptr) ||
        (output == nullptr) ||
        (weights == nullptr)) {
        return SaberInvalidValue;
    }

    src_shape = input->shape();
    dst_shape = output->shape();
    wgt_shape = weights->shape();

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
    jcp.im2col_sz   = (ptrdiff_t)jcp.ic * jcp.ks * jcp.os;
    jcp.need_im2col = !(jcp.oh == jcp.ih &&
                        jcp.ow == jcp.iw &&
                        jcp.ks == 1 &&
                        jcp.ngroups == 1);
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
    jcp.is_dw    = ((wgt_shape[1] == 1) &&
                    (dst_shape[3] == src_shape[3]));

    // TODO remove this logic once group convolution enabled
    if (jcp.ngroups > 1 && !jcp.is_dw) {
        return SaberUnImplError;
    }

    jcp.nthr = omp_get_max_threads();
    if (!(jcp.ic == 1 &&
          jcp.oc == 1 &&
          jcp.ngroups != 1) &&
        !(jcp.os / jcp.nthr < 64 &&
          jcp.mb != 1)) {
        jcp.nthr = 1;
    }

    jcp.with_bias = (bias != NULL);
    jcp.with_relu = conv_param->activation_param.has_active;
    if (jcp.with_relu) {
        jcp.relu_negative_slope = static_cast<float>(act_param->negative_slope);
    }

    size_t col_size = (size_t) jcp.im2col_sz * sizeof (unsigned char);
    size_t acc_size = (size_t) jcp.os * jcp.oc * sizeof (int32_t);
    acc_ = (int32_t *) zmalloc(acc_size * jcp.nthr, 4096);
    if (acc_ == nullptr) {
        return SaberOutOfMem;
    }

    col_ = (unsigned char *) zmalloc(col_size * jcp.nthr, 4096);
    if (col_ == nullptr) {
        zfree(acc_);
        acc_ = nullptr;
        return SaberOutOfMem;
    }
    memset(col_, 0, col_size * jcp.nthr);

    return SaberSuccess;
}

SaberStatus GemmU8S8S32XConv::check_conf(const jit_conv_conf_t &jcp,
                                         const std::vector<Tensor<X86>*> &inputs,
                                         std::vector<Tensor<X86>*> &outputs,
                                         ConvEltwiseParam<X86> &param) {
    return SaberSuccess;
}

SaberStatus GemmU8S8S32XConv::im2col_u8(const jit_conv_conf_t &jcp,
                                        const unsigned char* im,
                                        unsigned char* col) {
    int num_thr = (jcp.mb != 1) ? omp_get_max_threads() : 1;
    MAYBE_UNUSED(num_thr);
    #pragma omp parallel for collapse(2) num_threads(num_thr)
    for (int oh = 0; oh < jcp.oh; ++oh) {
        for (int ow = 0; ow < jcp.ow; ++ow) {
            for (int kh = 0; kh < jcp.kh; ++kh) {
                const int ih = oh * jcp.stride_h -
                               jcp.t_pad + kh * jcp.dilate_h;
                if (ih < 0 || ih >= jcp.ih) { 
                    continue;
                }

                for (int kw = 0; kw < jcp.kw; ++kw) {
                    const int iw = ow * jcp.stride_w -
                                   jcp.l_pad + kw * jcp.dilate_w;
                    if (iw < 0 || iw >= jcp.iw) {
                        continue;
                    }

                    const size_t col_idx = (((oh * jcp.ow + ow) * jcp.kh + kh) * jcp.kw + kw) *
                                           jcp.ic;
                    const size_t im_idx = (ih * jcp.iw + iw) * jcp.ngroups * jcp.ic;
                    #pragma omp simd
                    for (int ic = 0; ic < jcp.ic; ++ic) {
                        col[col_idx + ic] = im[im_idx + ic];
                    }
                }
            }
        }
    }

    return SaberSuccess;
}

SaberStatus GemmU8S8S32XConv::weight_reorder_oihw2hwio(Tensor<X86>* in,
                                                       Tensor<X86>* out) {
    if (in == nullptr || out == nullptr) {
        LOG(ERROR) << "invalid input or output weight tensor!";
        return SaberInvalidValue;
    }

    Shape shape = in->shape();
    int oc_value = shape[0];
    int ic_value = shape[1];
    int kh_value = shape[2];
    int kw_value = shape[3];
    int src_index =0;
    int dst_index = 0;

    if ((oc_value == 1) && (ic_value == 1)) {
        return SaberSuccess;
    }

    int8_t *src = (int8_t *)in->mutable_data();
    int8_t *dst = (int8_t *)out->mutable_data();

    if ((src == nullptr) || (dst == nullptr)) {
        LOG(ERROR) << "invalid input or output  weight tensor!";
        return SaberInvalidValue;
    }

    #pragma omp parallel for collapse(4)
    for (int oc = 0; oc < oc_value; oc++) {
        for (int ic = 0; ic < ic_value; ic++) {
            for (int kh = 0; kh < kh_value; kh++) {
                for (int kw = 0; kw < kw_value; kw++) {
                    src_index = oc * ic_value * kh_value * kw_value +
                                ic * kh_value * kw_value +
                                kh * kw_value +
                                kw;
                    dst_index = kh * kw_value * ic_value * oc_value +
                                kw * ic_value * oc_value +
                                ic * oc_value +
                                oc;
                    dst[dst_index] = src[src_index];
                }
            }
        }
    }

    return SaberSuccess;
}
} // namespace saber
} // namespace anakin
