#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX512_CORE_X8S8S32X_CONV_KERNEL_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX512_CORE_X8S8S32X_CONV_KERNEL_H

#include <iostream>
#include <stddef.h>

#include "saber/funcs/impl/x86/kernel/jit_generator.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {
namespace jit {

struct jit_avx512_core_x8s8s32x_fwd_kernel : public jit_generator {
public:
    jit_avx512_core_x8s8s32x_fwd_kernel(jit_conv_conf_t ajcp) : jcp(ajcp) {
        generate();
        jit_ker = (void (*)(jit_conv_call_t *))getCode();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_x8s8s32x_conv_fwd_ker_t)

    jit_conv_conf_t jcp;
    static SaberStatus init_conf(jit_conv_conf_t &jcp);
    void (*jit_ker)(jit_conv_call_t *);

private:
    using reg64_t = const Xbyak::Reg64;
    using zmm_t = const Xbyak::Zmm;
    using xmm_t = const Xbyak::Xmm;
    enum {
        typesize = sizeof(float),
        ker_reg_base_idx = 28,
    };
    enum {
        no_last_block,
        last_ic_block,
        last_sp_block,
    };

    reg64_t reg_inp = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_out = r10;
    reg64_t aux_reg_inp = r11;
    reg64_t reg_ptr_sum_scale = r11;
    reg64_t aux_reg_ker = r12;
    reg64_t reg_scratch = r14;
    reg64_t reg_kj   = rax;
    reg64_t reg_overflow = rax;
    reg64_t reg_ptr_scales = rax;
    reg64_t reg_oi   = rbx;
    reg64_t reg_bias = rdx;
    reg64_t reg_compensation = reg_scratch;
    reg64_t reg_kh   = abi_not_param1;
    reg64_t param    = abi_param1;
    reg64_t reg_tmp = rbp;
    reg64_t imm_addr64 = r15;
    reg64_t reg_oc_blocks = rsi;
    reg64_t reg_icb = reg_bias;
    reg64_t reg_bias_alpha = reg_kh;

    Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);

    zmm_t zmm_tmp = zmm_t(28);
    zmm_t zmm_one = zmm_t(29);
    zmm_t zmm_scales = zmm_t(30);
    zmm_t zmm_shift = zmm_t(30);
    zmm_t zmm_zero = zmm_t(31);
    zmm_t zmm_wei = zmm_t(31);

    zmm_t zmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return zmm_t(idx);
    }
    xmm_t xmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < ker_reg_base_idx);
        return xmm_t(idx);
    }
    zmm_t zmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return zmm_t(idx);
    }
    zmm_t zmm_bias_alpha() {
        return zmm_t(jcp.nb_oc_blocking * jcp.ur_w);
    }
    xmm_t xmm_bias_alpha() {
        return xmm_t(jcp.nb_oc_blocking * jcp.ur_w);
    }
    int get_ow_start(int ki, int pad_l) {
        return std::max(0, utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }
    int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w - std::max(0, utils::div_up(pad_r - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1), jcp.stride_w));
    }
    bool maybe_relu(int position, const float *post_sum);
    void prepare_output(int ur_w);
    void store_output(int ur_w, int last_oc_block_flag);
    void compute_ker(int ur_w, int pad_l, int pad_r, int last_ic_block_flag, bool h_padded = false);
    void kh_loop(int ur_w, int pad_l, int pad_r, int last_ic_block_flag);
    void icb_loop(int ur_w, int pad_l, int pad_r, bool is_last_spatial_block);
    void generate();
    void cvt2ps(DataType type_in, zmm_t zmm_in, const Xbyak::Operand &op, bool mask_flag);
};

} // namespace jit
} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_JIT_AVX512_CORE_X8S8S32_CONV_ACT_KERNEL_H
