#include "saber/funcs/impl/x86/saber_pooling.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_pool_kernel_f32.h"

namespace anakin {
namespace saber {

using namespace jit;

template <>
SaberStatus SaberPooling<X86, AK_FLOAT>::init_conf(
    jit_pool_conf_t& jpp, const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    PoolingParam<X86>& param) {
    //**/this function only use for avx512
    using namespace utils;

    Shape src_shape(inputs[0]->shape());
    Shape dst_shape(outputs[0]->shape());
    const int simd_w = 16;
    const int ndims = 4;

    jpp.ndims = ndims;
    jpp.mb = src_shape[0];
    jpp.c = src_shape[1] * 16;
    jpp.id = (ndims == 5) ? src_shape[2] : 1;
    jpp.ih = src_shape[ndims - 2];
    jpp.iw = src_shape[ndims - 1];
    jpp.od = (ndims == 5) ? dst_shape[2] : 1;
    jpp.oh = dst_shape[ndims - 2];
    jpp.ow = dst_shape[ndims - 1];

    jpp.stride_d = 1;
    jpp.stride_h = param.stride_h;
    jpp.stride_w = param.stride_w;
    jpp.kd = 1;
    jpp.kh = param.window_h;
    jpp.kw = param.window_w;

    jpp.f_pad = 0;
    jpp.t_pad = param.pad_h;
    jpp.l_pad = param.pad_w;

    jpp.alg = param.pooling_type;

    jpp.ind_dt = AK_FLOAT;

    jpp.simple_alg = false;

    jpp.c_block = simd_w;

    jpp.nb_c = jpp.c / jpp.c_block;

    if (jpp.alg == Pooling_max) {
        jpp.ur_w = 16;
    } else {
        jpp.ur_w = 24;
    }

    if (jpp.ow < jpp.ur_w) {
        jpp.ur_w = jpp.ow;
    }

    if (jpp.l_pad > jpp.ur_w) {
        return SaberUnImplError;
    }

    jpp.ur_w_tail = jpp.ow % jpp.ur_w;

    if (jit_uni_pool_kernel_f32<avx512_common>::init_conf(jpp)) {
        return SaberSuccess;
    } else {
        return SaberUnImplError;
    }

}

template <>
SaberStatus SaberPooling<X86, AK_FLOAT>::create(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    PoolingParam<X86>& param,
    Context<X86>& ctx) {
    if (mayiuse(avx512_common)) {
        jit_pool_conf_t jpp_;

        if (init_conf(jpp_, inputs, outputs, param) != SaberSuccess) {
            return SaberUnImplError;
        }

        _kernel = new jit_uni_pool_kernel_f32<avx512_common>(jpp_);
    } else {}

    return SaberSuccess;
}

template <>
SaberStatus SaberPooling<X86, AK_FLOAT>::init(
    const std::vector<Tensor<X86>*>& inputs,
    std::vector<Tensor<X86>*>& outputs,
    PoolingParam<X86>& param, Context<X86>& ctx) {

    this->_ctx = &ctx;

    return create(inputs, outputs, param, ctx);
}

template <>
SaberStatus SaberPooling<X86, AK_FLOAT>
::dispatch(const std::vector<Tensor<X86>*>& inputs,
           std::vector<Tensor<X86>*>& outputs,
           PoolingParam<X86>& param) {

    const float* src = static_cast<const float*>(inputs[0]->data());
    float* dst = static_cast<float*>(outputs[0]->mutable_data());

    //if (mayiuse(avx512_common)) {
    if (false) {
        //avx512 use jit
        const auto& jpp = _kernel->jpp;

        auto ker = [&](int n, int b_c, int oh) {
            jit_pool_call_t arg;

            const int ij = oh * jpp.stride_h;
            const int i_t_overflow = std::max(0, jpp.t_pad - ij);
            const int i_b_overflow = std::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
            const int ih = std::max(ij - jpp.t_pad, 0);

            // TODO verify the calulation
            int index = n * jpp.ih * jpp.iw * jpp.c + b_c * jpp.iw * jpp.ih * jpp.c_block  + ih * jpp.iw *
                        jpp.c_block;
            arg.src = &src[index];
            index = n * jpp.oh * jpp.ow * jpp.c + b_c * jpp.ow * jpp.oh * jpp.c_block + oh * jpp.ow *
                    jpp.c_block;
            arg.dst = &dst[index];

            arg.oh = (oh == 0);
            arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
            arg.kh_padding_shift = i_t_overflow * jpp.kw;
            arg.kw_padding = 0;
            arg.ker_area_h = (float)(jpp.kh -
                                     std::max(0, oh * jpp.stride_h - jpp.t_pad + jpp.kh - jpp.ih) -
                                     std::max(0, jpp.t_pad - oh * jpp.stride_h));
            (*_kernel)(&arg);
        };

        #pragma omp parallel for collapse(3) schedule(static)

        for (int n = 0; n < jpp.mb; ++n) {
            for (int b_c = 0; b_c < jpp.nb_c; ++b_c) {
                for (int oh = 0; oh < jpp.oh; ++oh) {
                    ker(n, b_c, oh);
                }
            }
        }
    } else {
        //x86 common code
        int in_n = inputs[0]->num();
        int in_c = inputs[0]->channel();
        int in_h = inputs[0]->height();
        int in_w = inputs[0]->width();
        int size_in_n = in_c * in_h * in_w;
        int size_in_c = in_h * in_w;

        int out_h = outputs[0]->height();
        int out_w = outputs[0]->width();
        int size_out_n = in_c * out_h * out_w;
        int size_out_c = out_h * out_w;

        for (int ind_n = 0; ind_n < in_n; ++ind_n) {
            for (int ind_c = 0; ind_c < in_c; ++ind_c) {
                for (int ind_h = 0; ind_h < out_h; ++ind_h) {
                    int sh = ind_h * param.stride_h;
                    int eh = sh + param.window_h;

                    sh = (sh - param.pad_h) < 0 ? 0 : sh - param.pad_h;
                    eh = (eh - param.pad_h) > in_h ? in_h : eh - param.pad_h;


                    for (int ind_w = 0; ind_w < out_w; ++ind_w) {
                        int sw = ind_w * param.stride_w;
                        int ew = sw + param.window_w;

                        sw = (sw - param.pad_w) < 0 ? 0 : sw - param.pad_w;
                        ew = (ew - param.pad_w) > in_w ? in_w : ew - param.pad_w;


                        float result;

                        int dst_ind = ind_n * size_out_n + ind_c * size_out_c + ind_h * out_w + ind_w;

                        for (int kh = sh; kh < eh; ++kh) {
                            for (int kw = sw; kw < ew; ++kw) {
                                int src_ind = ind_n * size_in_n + ind_c * size_in_c + kh * in_w + kw;

                                if (kh == sh && kw == sw) {
                                    result = src[src_ind];
                                } else {
                                    if (param.pooling_type == Pooling_max) {
                                        result = result >= src[src_ind] ? result : src[src_ind];
                                    }

                                    if (param.pooling_type == Pooling_average_include_padding) {
                                        result += src[src_ind];
                                    }

                                    if (param.pooling_type == Pooling_average_exclude_padding) {
                                        result += src[src_ind];
                                    }
                                }

                            }
                        }

                        if (param.pooling_type == Pooling_average_include_padding) {
                            
                            int bh = param.window_h;
                            int bw = param.window_w;
                            if (ew == in_w)
                            {
                                bw = sw + param.window_w >= in_w + param.pad_w ? in_w + param.pad_w : sw + param.window_w;
                                bw -=sw;
                            }
                            if (eh == in_h)
                            {
                                bh = sh + param.window_h >= in_h + param.pad_h ? in_h + param.pad_h: sh + param.window_h;
                                bh -= sh;
                            }
                            result /= bh * bw;

                        }

                        if (param.pooling_type == Pooling_average_exclude_padding) {
                            result /= (ew - sw) * (eh - sh);
                        }

                        dst[dst_ind] = result;
                        //LOG(INFO)<<"saber:"<<dst_ind<<"re:"<<result;

                    }
                }
            }

        }
    }

    return SaberSuccess;
}
template class SaberPooling<X86, AK_FLOAT>;
DEFINE_OP_TEMPLATE(SaberPooling, PoolingParam, X86, AK_HALF);
DEFINE_OP_TEMPLATE(SaberPooling, PoolingParam, X86, AK_INT8);
}
} // namespace anakin