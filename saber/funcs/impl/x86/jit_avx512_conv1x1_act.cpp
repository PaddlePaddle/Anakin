#include "saber/funcs/impl/x86/jit_call_conf.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/funcs/impl/x86/jit_avx512_conv1x1_act.h"

namespace anakin {
namespace saber {

using namespace jit;

inline void set_default_strides(jit_strides_t strides, const jit_dims_t dims,
    int ndims, const int *perm = NULL) {
    int id_perm[JIT_TENSOR_MAX_DIMS] = { 0 };
    for (int i = 0; i < ndims; ++i)
        id_perm[i] = i;
    if (perm == NULL)
        perm = id_perm;

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

int shape_to_jit_dim(jit_dims_t& md_dims, const Shape &shape)
{
    for (int i = 0; i < shape.dims(); i++)
        md_dims[i] = shape[i];
    return shape.dims();
}

template <DataType Dtype, typename LayoutType>
struct memory_block_t
{
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
    memory_block_t(const Shape &shape) {
        int ndims = 0;
        if (typeid(LayoutType) == typeid(NCHW_C16))
        {
            ndims = 4;
        }
        else if (typeid(LayoutType) == typeid(GOIHW16I16O))
        {
            ndims = 5;
        }
        else if (typeid(LayoutType) == typeid(OIHW16I16O))
        {
            ndims = 4;
        }
            //assert(shape.dims == ndims);
        shape_to_jit_dim(md_dims, shape);
        if (typeid(LayoutType) == typeid(NCHW_C16))
        {
            fill_nChw16c(md_dims, ndims, strides);
        }
        else if (typeid(LayoutType) == typeid(GOIHW16I16O))
        {
            fill_gOIhw16i16o(md_dims, ndims, strides);
        }
        else if (typeid(LayoutType) == typeid(OIHW16I16O))
        {
            fill_OIhw16i16o(md_dims, ndims, strides);
        }
    }
};


void rtus_prepare(reduce_to_unit_stride_t&rtus_,
    conv_1x1_desc *conv_d) {
    // Src Format = memory_format::nChw16c
    bool rtus_applicable = true
        && (conv_d->strides[0] != 1 || conv_d->strides[1] != 1);
    for (int d = 2; d < 4; ++d) {
        /* TODO: relax these conditions (by improving reducer) */
        rtus_applicable = rtus_applicable
            && conv_d->padding[0][d - 2] == 0
            && conv_d->dst_d[d] * conv_d->strides[d - 2] == conv_d->src_d[d];
    }
    if (rtus_applicable) {
        rtus_.reduce_src_ = true;
        rtus_.conv_d_ = conv_d;
        rtus_.conv_d_->strides[0] = rtus_.conv_d_->strides[1] = 1;
        utils::array_set(rtus_.conv_d_->padding[0], 0, 2);
        utils::array_set(rtus_.conv_d_->padding[1], 0, 2);
        int ic = rtus_.conv_d_->src_d[1];
        for (int i = 0; i < rtus_.conv_d_->dst_d_dims; i++)
        {
            rtus_.conv_d_->src_d[i] = rtus_.conv_d_->dst_d[i];
        }
        rtus_.conv_d_->src_d[1] = ic;
        fill_nChw16c(rtus_.conv_d_->src_d, 4, rtus_.src_dstrides);
    }
}


template <typename T, typename U>
void balance2D(U nthr, U ithr, T ny, T &ny_start, T &ny_end,
    T nx, T &nx_start, T &nx_end, T nx_divider)
{
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

/* convolution forward */
template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus JitAvx512Conv1x1Act<OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::init(
    const std::vector<inTensor*>& inputs,
    std::vector<outTensor*>& outputs,
    ConvActiveParam<opTensor> &param, Context<X86> &ctx) {
    this->_ctx = &ctx;
    // TODO: type checking
    // src = dst = nChw16c 
    // weight = group ? gOIhw16i16o : OIhw16i16o

    if (typeid(LayOutType_in) != typeid(NCHW_C16)) {
        return SaberUnImplError;
	}
    conv_d_.src_d_dims = shape_to_jit_dim(conv_d_.src_d, inputs[0]->shape());
    conv_d_.dst_d_dims = shape_to_jit_dim(conv_d_.dst_d, outputs[0]->shape());
    conv_d_.padding[0][0] = param.conv_param.pad_h;
    conv_d_.padding[0][1] = param.conv_param.pad_w;
    conv_d_.strides[0] = param.conv_param.stride_h;
    conv_d_.strides[1] = param.conv_param.stride_w;
    rtus_prepare(rtus_, &conv_d_);
    SaberStatus status;
    status = kernel_->init_conf(this->jcp_, conv_d_,
        param.conv_param.weight()->shape(),
        param.conv_param.group,
        param.conv_param.dilation_h, param.conv_param.dilation_w,
        param.has_active,
        param.activation_param.has_negative_slope() ? param.activation_param.negative_slope : 0.0,
        omp_get_max_threads(),
        param.conv_param.bias() != NULL,
        rtus_.reduce_src_);

    if (status != SaberSuccess) {
        return status;
	}

    if (!kernel_) {
        kernel_ = new jit::jit_avx512_common_1x1_conv_kernel(this->jcp_);
	}

    init_rtus_driver(&rtus_driver_, rtus_, jcp_, ws_per_thread_, &scratch_,
        inputs[0]->shape(), param.conv_param.stride_h, param.conv_param.stride_w);
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus JitAvx512Conv1x1Act<OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::create(
    const std::vector<inTensor*>& inputs,
    std::vector<outTensor*>& outputs,
    ConvActiveParam<opTensor> &param, Context<X86> &ctx) {

    return SaberSuccess;
}

template <DataType OpDtype,
        DataType inDtype,
        DataType outDtype,
        typename LayOutType_op,
        typename LayOutType_in,
        typename LayOutType_out>
SaberStatus JitAvx512Conv1x1Act<OpDtype, inDtype, outDtype,
        LayOutType_op, LayOutType_in, LayOutType_out>::dispatch(
    const std::vector<inTensor*>& inputs,
    std::vector<outTensor*>& outputs,
    ConvActiveParam<opTensor> &param) {

    ConvParam<opTensor> *conv_param = &(param.conv_param);
    ActivationParam<opTensor> *act_param = &(param.activation_param);
    const opTensor *weights = conv_param->weight();
    const opTensor *bias = conv_param->bias();

    const dtype *ptr_src = reinterpret_cast<const dtype *>(inputs[0]->get_buf()->get_data());
    const dtype *ptr_weights = reinterpret_cast<const dtype *>(weights->get_buf()->get_data());
    const dtype *ptr_bias = reinterpret_cast<const dtype *>(bias->get_buf()->get_data());
    auto ptr_dst = reinterpret_cast<dtype *>(outputs[0]->mutable_data());
    const auto &jcp = kernel_->jcp;

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    const int stride_h = conv_param->stride_h;
    const int stride_w = conv_param->stride_w;
    const int pad_t = conv_param->pad_h;
    const int pad_l = conv_param->pad_w;
    memory_block_t<OpDtype, OIHW16I16O> weights_d(weights->shape()); //TODO: Hard code
    memory_block_t<outDtype, LayOutType_out> dst_d(outputs[0]->shape());
    memory_block_t<inDtype, LayOutType_in> src_d(inputs[0]->shape());

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    #pragma omp parallel
    {
        int ithr = omp_get_thread_num(), nthr = omp_get_num_threads();

        jit::jit_1x1_conv_call_t p = {};

        jit::rtus_driver_t::call_params_t rp = {};

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
            jit::nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
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

            p.bcast_dim = jit::this_block_size(os, jcp.os,
                bcast_step * os_block);
            rp.os = p.bcast_dim;
        };

        auto init_load = [&](int ocb, int &load_step) {
            load_step = step(jcp.nb_load_blocking, ocb_end - ocb,
                jcp.nb_load_blocking_max);
            p.load_dim = jit::this_block_size(ocb * jcp.oc_block,
                ocb_end * jcp.oc_block, load_step * jcp.oc_block);
        };

        auto init_reduce = [&](int icb) {
            const int nb_ic_blocking_step =
                utils::min(icb + nb_ic_blocking, nb_ic) - icb;
            p.reduce_pos_flag = 0
                | (icb == 0 ? FLAG_REDUCE_FIRST : 0)
                | (icb + nb_ic_blocking_step >= nb_ic
                    ? FLAG_REDUCE_LAST : 0);

            p.reduce_dim = jit::this_block_size(icb * jcp.ic_block,
                jcp.ic, nb_ic_blocking_step * jcp.ic_block);
            rp.icb = p.reduce_dim / jcp.reduce_block;
        };

        auto inner_ker = [&](int ocb, int icb, int n, int g, int oh, int ow,
            int ih, int iw) {
            const int _ocb = g * nb_oc + ocb;
            const size_t dst_off = dst_d.blk_off(n, _ocb, oh, ow);

            p.output_data = &ptr_dst[dst_off];
            p.bias_data = &ptr_bias[_ocb * jcp.oc_block];
            p.load_data = &ptr_weights[conv_param->group > -1
                ? weights_d.blk_off(g, ocb, icb)
                : weights_d.blk_off(ocb, icb)];

            const int _icb = g * nb_ic + icb;

            if (rtus_.reduce_src_) {
                rp.ws = scratch_ + ithr * ws_per_thread_
                    + _icb * jcp.is * jcp.ic_block;
                if (ocb == ocb_start) {
                    rp.src = ptr_src + src_d.blk_off(n, _icb, ih, iw);
                    rtus_driver_->ker_(&rp);
                }
                p.bcast_data = rp.ws;
            } else
                p.bcast_data = ptr_src + src_d.blk_off(n, _icb, ih, iw);

            kernel_->jit_ker(&p);
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

template class JitAvx512Conv1x1Act<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW_C16, NCHW_C16, NCHW_C16>;
template class JitAvx512Conv1x1Act<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW_C16, NCHW_C16>;
template class JitAvx512Conv1x1Act<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW_C16>;
template class JitAvx512Conv1x1Act<AK_FLOAT, AK_FLOAT, AK_FLOAT, NCHW, NCHW, NCHW_C8>;
} // namespace saber
} // namespace anakin
