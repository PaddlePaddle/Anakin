#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_UNI_DW_CONV_KERNEL_F32_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_KERMEL_JIT_UNI_DW_CONV_KERNEL_F32_H

#include "jit_generator.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include <iostream>
#include <stddef.h>

namespace anakin {
namespace saber {
namespace jit {


struct jit_uni_dwconv_kernel_f32 {

    jit_uni_dwconv_kernel_f32() {}

    jit_uni_dwconv_kernel_f32(jit_conv_conf_t ajcp): jcp(ajcp) {
    }

    jit_conv_conf_t jcp;

    virtual ~jit_uni_dwconv_kernel_f32() {}

    void (*jit_ker)(jit_conv_call_t *);
};


template <cpu_isa_t isa>
struct jit_dwconv_kernel_f32 : public jit_uni_dwconv_kernel_f32, public jit_generator {

public:
    jit_dwconv_kernel_f32(jit_conv_conf_t ajcp) : jit_uni_dwconv_kernel_f32(ajcp), jit_generator() {
        generate();
        jit_ker = (void (*)(jit_conv_call_t *))getCode();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_dwconv_kernel_f32);

    static SaberStatus init_conf(jit_conv_conf_t &jcp);


private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    const Xbyak::AddressFrame &vmmword = (isa == sse42)
                                         ? xword : (isa == avx2) ? yword : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;
    // dw convolution
    reg64_t reg_input = r8;
    reg64_t aux_reg_input = r9;
    reg64_t aux1_reg_input = r10;
    reg64_t reg_kernel = r11;
    reg64_t aux_reg_kernel = r12;
    reg64_t aux1_reg_kernel = r13;
    reg64_t reg_output = r14;
    reg64_t reg_bias = r15;
    reg64_t reg_kh = rax;
    reg64_t reg_kw = rbx;
    reg64_t iter_kh = rdx;
    reg64_t iter_kw = rsi;
    reg64_t reg_ur_w = rbp;
    reg64_t reg_ch_blocks = aux1_reg_input;
    reg64_t imm_addr64 = aux1_reg_input;
    inline Vmm get_ker_reg(int idx) { return Vmm(idx + 0); }
    inline Vmm get_src_reg(int idx) { return Vmm(idx + 1); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 4); }
    inline void load_src(int ur_ch_blocks, int ur_w);
    inline void apply_filter(int ur_ch_blocks, int ur_w);
    inline void apply_filter_unrolled(int ur_ch_blocks, int ur_w);
    inline void apply_activation(int ur_ch_blocks, int ur_w);
    inline void store_dst(int ur_ch_blocks, int ur_w);
    inline void loop_body(int ur_ch_blocks);
    // activation fusing
    Vmm vmm_mask = Vmm(0);
    Vmm vmm_res_ns = Vmm(1);
    Xbyak::Xmm xmm_relu_ns = Xbyak::Xmm(2);
    Vmm vmm_relu_ns = Vmm(2);
    Vmm vmm_zero = Vmm(3);
    const unsigned char _cmp_gt_os = 6;
    const unsigned char _cmp_lt_os = 1;
    void generate();
};


} // namespace jit
} // namespace saber
} // namespace anakin

#endif
