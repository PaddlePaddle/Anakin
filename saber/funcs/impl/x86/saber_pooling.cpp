#include "saber/funcs/impl/x86/saber_pooling.h"
#include "saber/funcs/impl/x86/kernel/jit_uni_pool_kernel_f32.h"


namespace anakin{
namespace saber {

using namespace jit;

template class SaberPooling<X86, AK_FLOAT>;

template <>
SaberStatus SaberPooling<X86, AK_FLOAT>::init_conf(
        jit_pool_conf_t &jpp, const std::vector<Tensor<X86>*>& inputs,
        std::vector<Tensor<X86>*>& outputs,
        PoolingParam<X86> &param) {
        
        using namespace utils;
        
        Shape src_shape(inputs[0]->shape());
        Shape dst_shape(outputs[0]->shape());
        bool ok = true
        && mayiuse(avx512_common)
        //          && std::is_same<LayOutType_in, NCHW_C16>::value
        //          && std::is_same<LayOutType_op, NCHW>::value
        && one_of(param.pooling_type, Pooling_max,
                  Pooling_average_include_padding,
                  Pooling_average_exclude_padding);
        if (!ok) {
            return SaberUnImplError;
        }
        
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
        PoolingParam<X86> &param,
        Context<X86> &ctx){
    
    jit_pool_conf_t jpp_;
    if(init_conf(jpp_, inputs, outputs, param) != SaberSuccess) {
        return SaberUnImplError;
    }
    _kernel = new jit_uni_pool_kernel_f32<avx512_common>(jpp_);
    return SaberSuccess;
}

template <>
SaberStatus SaberPooling<X86, AK_FLOAT>::init(
            const std::vector<Tensor<X86>*>& inputs,
            std::vector<Tensor<X86>*>& outputs,
            PoolingParam<X86> &param, Context<X86> &ctx){
    
        this->_ctx = &ctx;
    
        return create(inputs, outputs, param, ctx);
}
    
template <>
SaberStatus SaberPooling<X86, AK_FLOAT>
    ::dispatch(const std::vector<Tensor<X86>*>& inputs,
                  std::vector<Tensor<X86>*>& outputs,
                  PoolingParam<X86> &param){
        
    if (!mayiuse(avx512_common)) {
        return SaberUnImplError;
    }
    const float *src = (const float*)inputs[0]->data();
    float *dst = (float*)outputs[0]->mutable_data();
    
    const auto &jpp = _kernel->jpp;
    
    auto ker = [&](int n, int b_c, int oh) {
         jit_pool_call_t arg;
        
        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = std::max(0, jpp.t_pad - ij);
        const int i_b_overflow = std::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = std::max(ij - jpp.t_pad, 0);
        
        // TODO verify the calulation
        int index = n * jpp.ih * jpp.iw * jpp.c + b_c * jpp.iw * jpp.ih * jpp.c_block  + ih * jpp.iw * jpp.c_block;
        arg.src = &src[index];
        index = n * jpp.oh * jpp.ow * jpp.c + b_c * jpp.ow * jpp.oh * jpp.c_block + oh * jpp.ow * jpp.c_block;
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
    
    return SaberSuccess;
}
DEFINE_OP_TEMPLATE(SaberPooling, PoolingParam, X86, AK_INT16);
DEFINE_OP_TEMPLATE(SaberPooling, PoolingParam, X86, AK_INT8);
}
} // namespace anakin
