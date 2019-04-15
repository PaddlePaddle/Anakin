#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/funcs/impl/x86/kernel/jit_avx512_conv1x1.h"

namespace anakin {
namespace saber {

using namespace jit;

inline void set_default_strides(jit_strides_t strides, const jit_dims_t dims,
                                int ndims, const int *perm = nullptr) {
    int id_perm[JIT_TENSOR_MAX_DIMS] = { 0 };
    for (int i = 0; i < ndims; ++i) {
        id_perm[i] = i;
    }

    if (perm == nullptr) {
        perm = id_perm;
    }

    strides[perm[ndims - 1]] = 1;
    for (int d = 1; d < ndims; ++d) {
        const int prev_idx = perm[ndims - d];
        const int curr_idx = perm[ndims - 1 - d];

        strides[curr_idx] = dims[curr_idx] == 0
                            ? 1
                            : strides[prev_idx] * utils::max(1, dims[prev_idx]);
    }
}

SaberStatus fill_contiguous_blocked(jit_dims_t md_dims,
                                    const int ndims, const jit_dims_t block_dims,
                                    const int perm[], jit_strides_t strides) {
    int unrolled_dims[2 * JIT_TENSOR_MAX_DIMS];
    int unrolled_strides[2 * JIT_TENSOR_MAX_DIMS];
    for (int d = 0; d < ndims; ++d) {
        unrolled_dims[d] = md_dims[d] / block_dims[d];
        unrolled_dims[ndims + d] = block_dims[d];
    }
    set_default_strides(unrolled_strides, unrolled_dims, 2 * ndims, perm);
    utils::array_copy(strides, &unrolled_strides[0], ndims);
    return SaberSuccess;
}

SaberStatus fill_nChw16c(jit_dims_t md_dims, int ndims, jit_strides_t strides) {
    const jit_dims_t block_dims = { 1, 16, 1, 1 };
    const int perm[] = {
            0, 1, 2, 3,
            4, 5, 6, 7 };
    return fill_contiguous_blocked(md_dims, ndims, block_dims, perm, strides);
}

SaberStatus fill_gOIhw16i16o(jit_dims_t md_dims, int ndims, jit_strides_t strides) {
    const jit_dims_t block_dims = { 1, 16, 16, 1, 1 };
    const int perm[] = {
            0, 1, 2, 3, 4,
            5, 7, 6, 8, 9 };
    return fill_contiguous_blocked(md_dims, ndims, block_dims, perm, strides);
}

SaberStatus fill_OIhw16i16o(jit_dims_t md_dims, int ndims, jit_strides_t strides) {
    const jit_dims_t block_dims = { 16, 16, 1, 1 };
    const int perm[] = {
            0, 1, 2, 3,
            5, 4, 6, 7 };
    return fill_contiguous_blocked(md_dims, ndims, block_dims, perm, strides);
}

int shape_to_jit_dim(jit_dims_t& md_dims, const Shape &shape) {
    for (int i = 0; i < shape.dims(); i++)
        md_dims[i] = shape[i];
    return shape.dims();
}

struct memory_block_t {
    jit_dims_t md_dims;
    jit_strides_t strides;

    template<typename ...Args> inline size_t blk_off(Args... args) {
        return _blk_off<sizeof...(args), Args...>(args...);
    }

    template<int ORIG_LEN, typename ...Void>
    inline size_t _blk_off() {
        return 0;
    }

    template<int ORIG_LEN, typename T, typename ...Args>
    inline size_t _blk_off(T xc, Args ...args) {
        constexpr int dc = ORIG_LEN - sizeof...(args)-1;
        return size_t(xc) * strides[dc]
               + _blk_off<ORIG_LEN, Args...>(args...);
    }

    memory_block_t(LayoutType layout_type, Shape &shape) {
        int ndims = 0;
        if (layout_type == Layout_NCHW_C16R) {
            ndims = 4;
        }
        else if (layout_type == Layout_GOIHW16I16O) {
            ndims = 5;
        }
        else if (layout_type == Layout_OIHW16I16O) {
            ndims = 4;
        }

        shape_to_jit_dim(md_dims, shape);
        if (layout_type == Layout_NCHW_C16R) {
            fill_nChw16c(md_dims, ndims, strides);
        }
        else if (layout_type == Layout_GOIHW16I16O) {
            fill_gOIhw16i16o(md_dims, ndims, strides);
        }
        else if (layout_type == Layout_OIHW16I16O) {
            fill_OIhw16i16o(md_dims, ndims, strides);
        }
    }
};


template <>
void JitAvx512Conv1x1<AK_FLOAT>::prepare_rtus() {
    bool rtus_applicable = true &&
                           (conf.stride_h != 1 || conf.stride_w != 1);

    rtus_applicable = rtus_applicable &&
                      conf.t_pad == 0 && conf.l_pad == 0 &&
                      conf.oh * conf.stride_h == conf.ih &&
                      conf.ow * conf.stride_w == conf.iw;

    // LOG(ERROR) << "rtus applicable:" << rtus_applicable;

    if (rtus_applicable) {
        this->reduce_src = true;
        this->conf.stride_h = this->conf.stride_w = 1;
        this->conf.ih = this->conf.oh;
        this->conf.iw = this->conf.ow;
    }

    return;
}


template <>
SaberStatus JitAvx512Conv1x1<AK_FLOAT>::check_conf(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86> &param) {

    ConvParam<X86> *conv_param = &(param.conv_param);
    const Tensor<X86> *weights = conv_param->weight();
    const Tensor<X86> *bias = conv_param->bias();
    const jit_1x1_conv_conf_t jcp = kernel->jcp;
    Tensor<X86> *input = inputs[0];
    Tensor<X86> *output = outputs[0];
    LayoutType input_layout = inputs[0]->get_layout();
    LayoutType output_layout = outputs[0]->get_layout();

    if ((inputs[0]->get_layout() != Layout_NCHW_C16R)
        || (outputs[0]->get_layout() != Layout_NCHW_C16R)
        || (conv_param->weight()->get_layout() != Layout_NCHW)) {

        LOG(FATAL) << "wrong format";
        return SaberUnImplError;
    }

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
                    && jcp.ic == utils::round_up(input->channel(), 16)
                    && jcp.ih == input->height()
                    && jcp.iw == input->width()
                    && jcp.oc == utils::round_up(output->channel(), 16)
                    && jcp.oh == output->height()
                    && jcp.ow == output->width();

    if (param_ok && shape_ok) {
        return SaberSuccess;
    } else {
        LOG(FATAL) << "param or shape changed, re-init kernel";
        return SaberNotInitialized;
    }

}

template <>
SaberStatus JitAvx512Conv1x1<AK_FLOAT>::create(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86> &param, Context<X86> &ctx) {
    ConvParam<X86> *conv_param = &(param.conv_param);
    ActivationParam<X86> *act_param = nullptr;
    SaberStatus status;
    const Tensor<X86> *weights = conv_param->weight();
    Tensor<X86> *input = inputs[0];
    Tensor<X86> *output = outputs[0];

    // check conf
    if (kernel) {
        status = check_conf(inputs, outputs, param);
        if(status != SaberNotInitialized) {
            return status;
        }
    }

    // init conf
    const bool with_groups = false;
    conf.ngroups = with_groups ? weights->num() : 1;

    conf.mb = input->num();
    conf.ic = utils::round_up(input->channel(), 16)  / conf.ngroups;
    conf.ih = input->height();
    conf.iw = input->width();

    conf.oc = utils::round_up(output->channel(), 16) / conf.ngroups;
    conf.oh = output->height();
    conf.ow = output->width();

    conf.kh = weights->height();
    conf.kw = weights->width();
    conf.stride_h = conv_param->stride_h;
    conf.stride_w = conv_param->stride_w;
    conf.t_pad = conv_param->pad_h;
    conf.l_pad = conv_param->pad_w;

    conf.with_relu = conv_param->activation_param.has_active;
    if (conf.with_relu) {
        act_param = &(conv_param->activation_param);
        conf.relu_negative_slope = static_cast<float>(act_param->negative_slope);
    }
    conf.with_bias = (conv_param->bias() != nullptr&&conv_param->bias()->valid_size()>0);

    conv_d.n = input->num();
    conv_d.ic = input->channel() / conf.ngroups;
    conv_d.ih = input->height();
    conv_d.iw = input->width();
    conv_d.oc = output->channel() / conf.ngroups;
    conv_d.oh = output->height();
    conv_d.ow = output->width();
    conv_d.t_pad = conv_param->pad_h;
    conv_d.l_pad = conv_param->pad_w;
    conv_d.stride_h = conv_param->stride_h;
    conv_d.stride_w = conv_param->stride_w;

    prepare_rtus();

    status = jit_avx512_common_1x1_conv_kernel::init_conf(conf, conv_d, anakin_get_max_threads(), reduce_src);
    if (status == SaberSuccess) {
        if (kernel != nullptr) {
            delete kernel;
            kernel = nullptr;
        }
        kernel = new jit_avx512_common_1x1_conv_kernel(this->conf);
    } else {
        return SaberUnImplError;
    }

    if (reduce_src) {
        init_rtus_driver<float>(&rtus_driver, conf, conv_d, ws_per_thread, &scratch);
    }

    // reorder weights
    Tensor<X86> *weights_reorder = conv_param->mutable_weight();
    weights_internal.reset(new Tensor<X86>(weights_reorder->valid_shape()));
    weight_reorder_OIhw16i16o(*weights_reorder, *weights_internal);

    return SaberSuccess;
}

template <>
SaberStatus JitAvx512Conv1x1<AK_FLOAT>::init(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86> &param, Context<X86> &ctx) {
    ConvParam<X86> *conv_param = &(param.conv_param);


    if ((inputs[0]->get_layout() != Layout_NCHW_C16R)
        || (outputs[0]->get_layout() != Layout_NCHW_C16R)
        || (conv_param->weight()->get_layout() != Layout_NCHW)) {

                LOG(ERROR) << "wrong format";
        return SaberUnImplError;
    }
    CHECK_EQ(conv_param->pad_w,0)<<"pad must == 0";
    CHECK_EQ(conv_param->pad_h,0)<<"pad must == 0";

    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus JitAvx512Conv1x1<AK_FLOAT>::dispatch(
        const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        ConvEltwiseParam<X86> &param) {

    ConvParam<X86> *conv_param = &(param.conv_param);
//    ActivationParam<X86> *act_param = &(param.activation_param);
    const Tensor<X86> *weights = conv_param->weight();
    const Tensor<X86> *bias = conv_param->bias();

    const float *ptr_src = reinterpret_cast<const float *>(inputs[0]->data());
    const float *ptr_bias = reinterpret_cast<const float *>(bias->data());
    float *ptr_dst = reinterpret_cast<float *>(outputs[0]->mutable_data());
    const float *ptr_weights = reinterpret_cast<const float *>(weights_internal->data());

    const auto &jcp = kernel->jcp;

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    const int stride_h = conv_param->stride_h;
    const int stride_w = conv_param->stride_w;
    const int pad_t = conv_param->pad_h;
    const int pad_l = conv_param->pad_w;
    Shape weights_shape = weights->valid_shape();
    memory_block_t weights_d(Layout_OIHW16I16O, weights_shape);

    Shape src_d_adjust(inputs[0]->valid_shape());
    src_d_adjust[1] *= 16;
    Shape dst_d_adjust(outputs[0]->valid_shape());
    dst_d_adjust[1] *= 16;
    memory_block_t dst_d(outputs[0]->get_layout(), dst_d_adjust);
    memory_block_t src_d(inputs[0]->get_layout(), src_d_adjust);

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

#pragma omp parallel
    {
        int ithr = anakin_get_thread_num(), nthr = anakin_get_num_threads();

        jit_1x1_conv_call_t p;

        rtus_driver_t::call_params_t rp;

        const int nb_oc = jcp.nb_load;
        const int nb_ic = jcp.nb_reduce;
        const int nb_ic_blocking = jcp.nb_reduce_blocking;
        const int os_block = jcp.bcast_block;

        int bcast_start{ 0 }, bcast_end{ 0 }, ocb_start{ 0 }, ocb_end{ 0 };
        balance2D(nthr, ithr, work_amount, bcast_start, bcast_end,
                  jcp.nb_load, ocb_start, ocb_end, jcp.load_grp_count);

        auto init_bcast = [&](int iwork, int &n, int &g, int &bcast_step,
                              int &oh, int &ow, int &ih, int &iw) {
            int osb{ 0 };
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

            p.bcast_dim = utils::this_block_size(os, jcp.os,
                                                 bcast_step * os_block);
            rp.os = p.bcast_dim;
        };

        auto init_load = [&](int ocb, int &load_step) {
            load_step = step(jcp.nb_load_blocking, ocb_end - ocb,
                             jcp.nb_load_blocking_max);
            p.load_dim = utils::this_block_size(ocb * jcp.oc_block,
                                                ocb_end * jcp.oc_block, load_step * jcp.oc_block);
        };

        auto init_reduce = [&](int icb) {
            const int nb_ic_blocking_step =
                    utils::min(icb + nb_ic_blocking, nb_ic) - icb;
            p.reduce_pos_flag = 0
                                | (icb == 0 ? FLAG_REDUCE_FIRST : 0)
                                | (icb + nb_ic_blocking_step >= nb_ic
                                   ? FLAG_REDUCE_LAST : 0);

            p.reduce_dim = utils::this_block_size(icb * jcp.ic_block,
                                                  jcp.ic, nb_ic_blocking_step * jcp.ic_block);
            rp.icb = p.reduce_dim / jcp.reduce_block;
        };

        auto inner_ker = [&](int ocb, int icb, int n, int g, int oh, int ow,
                             int ih, int iw) {
            const int _ocb = g * nb_oc + ocb;
            const size_t dst_off = dst_d.blk_off(n, _ocb, oh, ow);

            p.output_data = &ptr_dst[dst_off];
            p.bias_data = &ptr_bias[_ocb * jcp.oc_block];
            p.load_data = &ptr_weights[conv_param->group > 1 ?
                                       weights_d.blk_off(g, ocb, icb) :
                                       weights_d.blk_off(ocb, icb)];

            const int _icb = g * nb_ic + icb;

            if (reduce_src) {
                rp.ws = scratch + ithr * ws_per_thread
                        + _icb * jcp.is * jcp.ic_block;
                if (ocb == ocb_start) {
                    rp.src = ptr_src + src_d.blk_off(n, _icb, ih, iw);
                    rtus_driver->ker_(&rp);
                }
                p.bcast_data = rp.ws;
            } else {
                p.bcast_data = ptr_src + src_d.blk_off(n, _icb, ih, iw);
            }

            kernel->jit_ker(&p);
        };

        if (jcp.loop_order == loop_rlb) {
            for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                init_reduce(icb);
                int ocb = ocb_start;
                while (ocb < ocb_end) {
                    int load_step;
                    init_load(ocb, load_step);
                    int iwork = bcast_start;
                    while (iwork < bcast_end) {
                        int n, g, bcast_step, oh, ow, ih, iw;
                        init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                        inner_ker(ocb, icb, n, g, oh, ow, ih, iw);
                        iwork += bcast_step;
                    }
                    ocb += load_step;
                }
            }
        } else if (jcp.loop_order == loop_lbr) {
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, load_step);
                int iwork = bcast_start;
                while (iwork < bcast_end) {
                    int n, g, bcast_step, oh, ow, ih, iw;
                    init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                    for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                        init_reduce(icb);
                        inner_ker(ocb, icb, n, g, oh, ow, ih, iw);
                    }
                    iwork += bcast_step;
                }
                ocb += load_step;
            }
        } else if (jcp.loop_order == loop_rbl) {
            for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                init_reduce(icb);
                int iwork = bcast_start;
                while (iwork < bcast_end) {
                    int n, g, bcast_step, oh, ow, ih, iw;
                    init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                    int ocb = ocb_start;
                    while (ocb < ocb_end) {
                        int load_step;
                        init_load(ocb, load_step);
                        inner_ker(ocb, icb, n, g, oh, ow, ih, iw);
                        ocb += load_step;
                    }
                    iwork += bcast_step;
                }
            }
        } else if (jcp.loop_order == loop_blr) {
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, oh, ow, ih, iw;
                init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                int ocb = ocb_start;
                while (ocb < ocb_end) {
                    int load_step;
                    init_load(ocb, load_step);
                    for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                        init_reduce(icb);
                        inner_ker(ocb, icb, n, g, oh, ow, ih, iw);
                    }
                    ocb += load_step;
                }
                iwork += bcast_step;
            }
        }
        else {
            assert(!"unsupported loop order");
        }
    }

    return SaberSuccess;
}

template class JitAvx512Conv1x1<AK_FLOAT>;

} // namespace saber
} // namespace anakin
