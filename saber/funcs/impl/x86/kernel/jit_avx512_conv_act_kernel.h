#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX512_CONV_ACT_KERNEL_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_KERMEL_JIT_AVX512_CONV_ACT_KERNEL_H

#include <iostream>
#include <stddef.h>

#include "jit_generator.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {
namespace jit {

struct jit_conv_act_kernel : public jit_generator {

public:
    jit_conv_act_kernel(jit_conv_conf_t ajcp) : jcp(ajcp) {
        generate();
        jit_ker = (void (*)(jit_conv_call_t *))getCode();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_conv_act_kernel);

    static SaberStatus init_conf(jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_t *);

private:
    using reg64_t = const Xbyak::Reg64;
    enum {
        typesize = sizeof(float),
        ker_reg_base_idx = 28,
    };

    reg64_t param = abi_param1;
    reg64_t reg_inp = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_out = r10;

    reg64_t reg_inp_prf = r11;
    reg64_t reg_ker_prf = r12;
    reg64_t reg_out_prf = r13;

    reg64_t aux_reg_inp = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_inp_prf = rsi;
    reg64_t aux_reg_ker_prf = rdx;

    reg64_t reg_channel = rsi;
    reg64_t reg_bias = rdx;

    reg64_t reg_kj = rax;
    reg64_t reg_relu_ns = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_tmp = rbp;

    reg64_t reg_ic_loop = rdx;
    reg64_t reg_inp_loop = rsi;

    reg64_t reg_init_flag = r13;
    reg64_t reg_bias_ptr = param;

    reg64_t aux_reg_ic = r12;
    reg64_t reg_binp = rax;
    reg64_t reg_bout = r11;
    reg64_t aux1_reg_inp = rbx;
    reg64_t aux_reg_out = abi_not_param1;

    inline Xbyak::Zmm zmm_ker(int i_ic) {
        assert(i_ic < 4);
        return Xbyak::Zmm(ker_reg_base_idx + i_ic);
    }

    inline Xbyak::Zmm zmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return Xbyak::Zmm(idx);
    }

    inline Xbyak::Zmm zmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Xbyak::Zmm(idx);
    }

    Xbyak::Reg64 imm_addr64 = r15;
    Xbyak::Xmm xmm_relu_ns = Xbyak::Xmm(30);
    Xbyak::Zmm zmm_relu_ns = Xbyak::Zmm(30);
    Xbyak::Zmm zmm_zero = Xbyak::Zmm(31);
    Xbyak::Zmm zmm_wei = Xbyak::Zmm(31);
    Xbyak::Zmm zmm_s32_tmp = Xbyak::Zmm(31);

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop_fma(int ur_w, int pad_l, int pad_r);
    inline void compute_loop_fma_core(int ur_w, int pad_l, int pad_r);
    inline void compute_loop_4fma(int ur_w, int pad_l, int pad_r);
    inline void compute_loop_4fma_1st(int ur_w, int pad_l, int pad_r);
    inline void compute_loop(int ur_w, int pad_l, int pad_r);

    void generate();

    inline void vpXdpwssd(Xbyak::Zmm zmm1, Xbyak::Zmm zmm2, reg64_t reg,
        int offset) {
            vpdpwssd(zmm1, zmm2, EVEX_compress_addr(reg, offset, true));
    }

    inline void vadd(Xbyak::Zmm zmm, reg64_t reg, int offset)   {
            vaddps(zmm, zmm, EVEX_compress_addr(reg, offset));
    }

    inline void vcmp(Xbyak::Opmask kmask,
        Xbyak::Zmm zmm_src1, Xbyak::Zmm zmm_src2, const unsigned char cmp) {
            vcmpps(kmask, zmm_src1, zmm_src2, cmp);
    }

    inline void vmul(Xbyak::Zmm zmm_dst, Xbyak::Opmask kmask,
                     Xbyak::Zmm zmm_src1, Xbyak::Zmm zmm_src2) {
            vmulps(zmm_dst | kmask, zmm_src1, zmm_src2);
    }

    inline int get_output_offset(int oi, int n_oc_block) {
        if (jcp.output_nhwc) {
            return jcp.typesize_out
                * (oi * jcp.oc + (n_oc_block * jcp.oc_block));
        } else {
            return jcp.typesize_out
                * (n_oc_block * jcp.oh * jcp.ow + oi) * jcp.oc_block;
        }
    }

    inline int get_input_offset(int ki, int ic, int oi, int pad_l) {
        int scale = 1;
        int iw_str = !jcp.is_1stconv ? jcp.ic_block : 1;
        int ic_str = !jcp.is_1stconv ? 1 : jcp.iw * jcp.ih;
        return jcp.typesize_in
            * ((ki + oi * jcp.stride_w - pad_l) * iw_str + scale * ic * ic_str);
    }

    inline int get_kernel_offset(int ki,int ic,int n_oc_block,int ker_number) {
        int scale = 1;
        return jcp.typesize_in * jcp.oc_block
            * (n_oc_block * jcp.nb_ic * jcp.ic_block * jcp.kh * jcp.kw
                    + (ic + ker_number) * scale + ki * jcp.ic_block);
    }

    inline int get_ow_start(int ki, int pad_l) {
        return utils::max(0, (pad_l - ki + jcp.stride_w - 1) / jcp.stride_w);
    }

    inline int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w - utils::max(0,
            (ki + pad_r - (jcp.kw - 1) + jcp.stride_w - 1) / jcp.stride_w);
    }

};


} // namespace jit
} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_KERMEL_JIT_AVX512_CONV_ACT_KERNEL_H
