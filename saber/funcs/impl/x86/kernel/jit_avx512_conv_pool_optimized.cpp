#include "saber/funcs/impl/x86/kernel/jit_avx512_conv_pool_optimized.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/funcs/impl/x86/saber_pooling.h"
#include "anakin_thread.h"

namespace anakin {
namespace saber {

using namespace jit;

SaberStatus JitAvx512ConvPoolOptimized::init(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvPoolingParam<X86>& param,
    Context<X86>& ctx) {
    SaberStatus ret = SaberUnImplError;

    this->_ctx = &ctx;
    ConvParam<X86>* conv_param = &(param.conv_param);
    PoolingParam<X86> pool_param = param.pooling_param;
    Tensor<X86>* input = inputs[0];
    const Tensor<X86>* weights = conv_param->weight();
    Shape wgt_shape(weights->shape());
    bool depthwise = (conv_param->group > 1) && (wgt_shape[1] == 1);
    auto out_scale = outputs[0]->get_scale();

    // check do partial pool with conv
    if (pool_param.window_w != pool_param.stride_w ||
            pool_param.window_w >= 3 ||
            pool_param.pooling_type != Pooling_max) {
        return ret;
    }

    // reorder weights
    // TODO check weights, do scale or not?
    Tensor<X86>* weights_reorder = conv_param->mutable_weight();

    if (weights_reorder->get_dtype() == AK_FLOAT) {
        _weights_scale.re_alloc(weights_reorder->valid_shape(), AK_INT8);
        utils::ScaleUtils::scale_conv_weights_to_nchw_host(_weights_scale, *conv_param->weight());
        weights_reorder = &_weights_scale;
    }

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

    const Tensor<X86>* bias_src = conv_param->bias();

    // TODO check bias, do scale or not?
    if (bias_src != nullptr && bias_src->valid_size() > 0) {
        bias_internal_ = new Tensor<X86>(bias_src->shape(), AK_INT32);
        auto weights_scale = weights_reorder->get_scale();
        float in_scale = 1.f;

        if (input->get_scale().size() > 0) {
            in_scale = input->get_scale()[0];
        }

        std::vector<float> scale_vec(bias_src->valid_size());

        //TODO:we need scale in this?
        for (int i = 0; i < bias_src->valid_size(); i++) {
            scale_vec[i] = outputs[0]->get_scale()[0] / weights_scale[i] / in_scale;
        }

        bias_internal_->set_scale(scale_vec);
        bias_reorder_nchw(*bias_src, *bias_internal_, scale_vec);
    }

    ///transe///

    // prepare buf
    ret = prepare_buf(outputs[0]->valid_shape(), pool_param, out_scale);

    if (ret != SaberSuccess) {
        return ret;
    }

    // init pool op
    if (this->pool_impl_) {
        delete this->pool_impl_;
    }

    this->pool_impl_ = new SaberPooling<X86, AK_INT8>;
    pool_param.window_w = 1;
    pool_param.stride_w = 1;
    ret = this->pool_impl_->init(buf_, outputs, pool_param, ctx);

    if (ret != SaberSuccess) {
        LOG(INFO) << "init pool impl error";
        return ret;
    }

    // do create
    ret = this->create(inputs, outputs, param, ctx);

    return ret;
}

SaberStatus JitAvx512ConvPoolOptimized::create(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvPoolingParam<X86>& param,
    Context<X86>& ctx) {
    SaberStatus ret = SaberUnImplError;

    auto out_scale = outputs[0]->get_scale();
    PoolingParam<X86> pool_param = param.pooling_param;

    // check do partial pool with conv
    if (pool_param.window_w != pool_param.stride_w ||
            pool_param.window_w >= 3 ||
            pool_param.pooling_type != Pooling_max) {
        return ret;
    }

    // make sure the buff was allocated successfully
    ret = prepare_buf(outputs[0]->valid_shape(), pool_param, out_scale);

    if (ret != SaberSuccess) {
        return ret;
    }

    // create conv act op
    ret = this->create_conv(inputs, buf_, param, ctx);

    if (ret != SaberSuccess) {
        return ret;
    }

    pool_param.window_w = 1;
    pool_param.stride_w = 1;
    // create pooling op
    ret = this->create_pool(buf_, outputs, pool_param, ctx);
    return ret;
}

SaberStatus JitAvx512ConvPoolOptimized::create_conv(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvPoolingParam<X86>& param,
    Context<X86>& ctx) {
    SaberStatus status;
    jit_conv_conf_t jcp;

    ConvParam<X86>* conv_param = &(param.conv_param);
    ActivationParam<X86>* act_param = &(conv_param->activation_param);
    PoolingParam<X86> pool_param = param.pooling_param;
    jcp.pool_kw = pool_param.window_w;
    jcp.pool_alg = pool_param.pooling_type;
    jcp.with_partial_pool = true;

    const Tensor<X86>* weights = conv_param->weight();
    const Tensor<X86>* bias = conv_param->bias();
    Tensor<X86>* input = inputs[0];
    Tensor<X86>* output = outputs[0];
    Shape src_shape(input->shape());
    Shape dst_shape(output->shape());
    Shape wgt_shape(weights->shape());

    // init conf
    const bool with_groups = (conv_param->group > 1);
    jcp.ngroups = with_groups ? conv_param->group : 1;

    jcp.mb = src_shape[0];
    jcp.ic = src_shape[3] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.ih = src_shape[1];
    jcp.iw = src_shape[2];
    jcp.oc = dst_shape[3] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.oh = dst_shape[1];

    // set conv's output width
    jcp.ow = dst_shape[2] * jcp.pool_kw;

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

    jcp.with_bias = (bias != nullptr && bias->valid_size() > 0);
    jcp.with_relu = conv_param->activation_param.has_active;

    if (jcp.with_relu) {
        jcp.relu_negative_slope = static_cast<float>(act_param->negative_slope);
    }

    jcp.is_dw = with_groups && (jcp.ic == 1);

    jcp.is_oc_scale = 1;
    jcp.with_sum = false;

    status = jit_avx512_core_u8s8s32x_conv_act_pool_kernel::init_conf(jcp);

    if (status == SaberSuccess) {
        if (conv_kernel_ != nullptr) {
            delete conv_kernel_;
            conv_kernel_ = nullptr;
        }

        conv_kernel_ = new jit_avx512_core_u8s8s32x_conv_act_pool_kernel(jcp);
    } else {
        return SaberUnImplError;
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

SaberStatus JitAvx512ConvPoolOptimized::create_pool(
    std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    PoolingParam<X86>& param,
    Context<X86>& ctx) {
    return this->pool_impl_->create(inputs, outputs, param, ctx);
}

SaberStatus JitAvx512ConvPoolOptimized::dispatch(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvPoolingParam<X86>& param) {

    SaberStatus ret = SaberSuccess;

    ConvParam<X86> conv_param(param.conv_param);
    PoolingParam<X86> pool_param = param.pooling_param;
    pool_param.window_w = 1;
    pool_param.stride_w = 1;

    ret = this->dispatch_conv(inputs, buf_, param);

    if (ret != SaberSuccess) {
        return ret;
    }

    /*for (int i = 0; i < buf_[0]->valid_size(); i++) {
        LOG(INFO) << (int) ((unsigned char *) buf_[0]->data())[i];
    }*/

    ret = this->dispatch_pool(buf_, outputs, pool_param);
    return ret;
}

SaberStatus JitAvx512ConvPoolOptimized::dispatch_conv(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    ConvPoolingParam<X86>& param) {
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

    const auto& jcp = conv_kernel_->jcp;
    const auto oscale = scale_;

    parallel(0, [&](const int ithr, const int nthr) {
        int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
        //        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int nb_groups = jcp.nb_ch;
        int group_block = jcp.ch_block;

        int start{0}, end{0};
        int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh;
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_call_t();

        size_t src_h_stride = jcp.iw * jcp.ic;
        size_t dst_h_stride = (jcp.ow / jcp.pool_kw) * jcp.oc;
        size_t wht_h_stride = jcp.kw * jcp.ic_block * jcp.oc_block;
        size_t wht_ic_stride = jcp.kh * jcp.kw * jcp.ic_block * jcp.oc_block;

        if (jcp.is_dw) {
            src_h_stride = jcp.iw * jcp.ic * jcp.ngroups;
            dst_h_stride = (jcp.ow / jcp.pool_kw) * jcp.oc * jcp.ngroups;
            wht_h_stride = jcp.kw * jcp.ch_block;
            wht_ic_stride = jcp.kh * jcp.kw * jcp.ch_block;
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

            int g_ic = g * jcp.nb_ic * jcp.oc_block;

            int work_rem = end - start;
            int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
            int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;

            size_t bias_blk_off = g_oc;
            size_t dst_blk_off = n * jcp.oc * jcp.oh * (jcp.ow / jcp.pool_kw) +
                                 oh_s * (jcp.ow / jcp.pool_kw) * jcp.oc + g_oc;
            size_t src_blk_off = n * jcp.ic * jcp.ih * jcp.iw +
                                 ih_s * jcp.iw * jcp.ic + g_ic;
            size_t weight_blk_off = ocb * jcp.ic * jcp.kh * jcp.kw * jcp.oc_block;

            if (jcp.is_dw) {
                dst_blk_off = n * nb_groups * jcp.oh * (jcp.ow / jcp.pool_kw) * jcp.ch_block + g_oc + oh_s *
                              (jcp.ow / jcp.pool_kw) * nb_groups * jcp.ch_block;
                src_blk_off = n * nb_groups * jcp.ih * jcp.iw * jcp.ch_block + g_ic + ih_s * jcp.iw * nb_groups *
                              jcp.ch_block;
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
                conv_kernel_->jit_ker(&p);

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

SaberStatus JitAvx512ConvPoolOptimized::dispatch_pool(
    std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    PoolingParam<X86>& param) {

    return this->pool_impl_->dispatch(inputs, outputs, param);
}

SaberStatus JitAvx512ConvPoolOptimized::prepare_buf(Shape pool_shape, PoolingParam<X86>& pool_param,
        std::vector<float> scale) {
    SaberStatus ret = SaberMemAllocFailed;

    // calculate the shape of buf, we do the w dim's pooling in conv kernel.
    Shape buf_shape({pool_shape[0],
                     (pool_shape[1] - 1) * pool_param.stride_h + pool_param.window_h - 2 * pool_param.pad_h,
                     pool_shape[2],
                     pool_shape[3]}, Layout_NHWC);

    // make sure allocate buf is successfully
    if (buf_.size() > 0 && buf_[0]->valid_shape() == buf_shape) {
        return SaberSuccess;
    }

    // release buf first
    release_buf();

    // allocate the buf according to the shape
    return allocate_buf(buf_shape, scale);
}

SaberStatus JitAvx512ConvPoolOptimized::allocate_buf(Shape buf_shape, std::vector<float> scale) {
    SaberStatus ret = SaberMemAllocFailed;

    Tensor<X86>* b_info = new Tensor<X86>(buf_shape, AK_UINT8);

    if (b_info) {
        b_info->set_scale(scale);
        buf_.push_back(b_info);
        ret = SaberSuccess;
    }

    return ret;
}

void JitAvx512ConvPoolOptimized::release_buf() {

    for (int i = 0; i < this->buf_.size(); i++) {
        delete buf_[i];
        buf_[i] = nullptr;
    }

    std::vector<Tensor<X86> *> ().swap(buf_);
    return;
}

} // namespace saber
} // namespace anakin
