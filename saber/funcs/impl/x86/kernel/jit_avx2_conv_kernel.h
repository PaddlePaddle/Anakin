#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_KERMEL_JIT_AVX2_CONV_KERNEL_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_KERMEL_JIT_AVX2_CONV_KERNEL_H

#include <iostream>
#include <stddef.h>

#include "jit_generator.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {
namespace jit {

struct jit_avx2_conv_act_kernel: public jit_generator {

    jit_avx2_conv_act_kernel(jit_conv_conf_t ajcp) : jcp(ajcp) {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_t *))this->getCode();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_conv_act_kernel);

    static SaberStatus init_conf(jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_t *);

private:
    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input = rax;
    reg64_t aux_reg_input = r8;
    reg64_t reg_kernel = rdx;
    reg64_t aux_reg_kernel = r9;
    reg64_t reg_output = rsi;
    reg64_t reg_bias = rbx;

    reg64_t kj = r10;
    reg64_t oi_iter = r11;
    reg64_t ki_iter = r12;
    reg64_t reg_kh = abi_not_param1;
    reg64_t reg_oc_blocks = r14;
    reg64_t imm_addr64 = r15;
    reg64_t reg_long_offt = r15;
    Xbyak::Reg32 reg_ci_flag = r13d;

    Xbyak::Xmm xmm_relu_ns = Xbyak::Xmm(13);
    Xbyak::Ymm ymm_relu_ns = Xbyak::Ymm(13);
    Xbyak::Ymm ymm_res_ns = Xbyak::Ymm(12);
    Xbyak::Ymm yzero = Xbyak::Ymm(15);
    Xbyak::Ymm ymask = Xbyak::Ymm(14);

    inline void oh_step_unroll_kw(int ur_w, int pad_l, int pad_r,
                                  int oc_blocks);
    inline void oh_step_nopad(int ur_w, int pad_l, int pad_r,
                              char pad_label, int oc_blocks, char oc_blocks_label);
    inline void width_blk_step(int ur_w, int pad_l, int pad_r,
                               char pad_label, int oc_blocks, char oc_blocks_label);
    inline void solve_common(int oc_blocks, char oc_blocks_label);

    void generate();
};

} // namespace jit
} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_KERMEL_JIT_AVX2_CONV_ACT_KERNEL_H
