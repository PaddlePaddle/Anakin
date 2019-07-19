#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX2_DECONV_ACT_KERNEL_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX2_DECONV_ACT_KERNEL_H

#include "saber/funcs/impl/x86/kernel/jit_generator.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {
namespace jit {

struct jit_avx2_deconv_act_kernel : public jit_generator {

public:
    jit_avx2_deconv_act_kernel(jit_deconv_conf_t ajcp): jcp(ajcp)
    {
        this->generate();
        jit_ker = (void (*)(jit_deconv_call_t *))this->getCode();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_deconv_act_kernel);

    static SaberStatus init_conf(jit_deconv_conf_t &jcp);

    jit_deconv_conf_t jcp;
    void (*jit_ker)(jit_deconv_call_t *);
private:
    using reg64_t = const Xbyak::Reg64;
    enum {
        typesize = sizeof(float),
        ker_reg_base_idx = 14,
    };

    reg64_t param = abi_param1;
    reg64_t reg_dst = r8;
    reg64_t reg_ker = r9;
    reg64_t reg_src = r10;

    reg64_t reg_dst_prf = r11;
    reg64_t reg_ker_prf = r12;
    reg64_t reg_src_prf = r13;

    reg64_t aux_reg_dst = r14;
    reg64_t aux_reg_ker = r15;

    reg64_t aux_reg_dst_prf = rsi;
    reg64_t aux_reg_ker_prf = rdx;

    reg64_t aux_reg_dst_d_prf = r13;
    reg64_t aux_reg_dst_d = rbx;
    reg64_t aux_reg_ker_d_prf = abi_not_param1;
    reg64_t aux_reg_ker_d = r9;
    reg64_t reg_ki = r10;

    reg64_t reg_kj = rax;
    reg64_t reg_oi = rbx;
    reg64_t reg_kh = abi_not_param1;

    reg64_t reg_channel = rsi;

    reg64_t reg_bias = rdx;
    reg64_t reg_long_offt = r14;

    Xbyak::Ymm ymm_wei = Xbyak::Ymm(15);
    Xbyak::Ymm ymm_temp = Xbyak::Ymm(14);

    inline Xbyak::Ymm ymm_ker(int i_ic) {
        assert(i_ic < 2);
        return Xbyak::Ymm(ker_reg_base_idx + i_ic);
    }

    inline Xbyak::Ymm ymm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 15);
        return Xbyak::Ymm(idx);
    }

    inline Xbyak::Ymm ymm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        // print1(idx);
        assert(idx < ker_reg_base_idx);
        return Xbyak::Ymm(idx);
    }

    inline void vadd(Xbyak::Ymm ymm, const Xbyak::Operand& op) {
        vaddps(ymm, ymm, op);
    }

    inline int get_iw_start(int ki, int l_overflow)
    {
        int res = (jcp.iw - 1 + jcp.r_pad) % jcp.stride_w
                + l_overflow * jcp.stride_w
                - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return res;
    }

    inline int get_iw_end(int ur_w, int ki, int r_overflow)
    {
        if (utils::one_of(ur_w, jcp.iw, jcp.ur_w_tail))
            ur_w += utils::min(0, jcp.r_pad); // remove negative padding
        int res = (ur_w - 1 + jcp.l_pad) % jcp.stride_w
                + r_overflow * jcp.stride_w - ki * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return ur_w - res;
    }

    template<typename T>
    inline Xbyak::Address VEX_compress_addr(Xbyak::Reg64 base,
            T raw_offt, bool bcast = false)
    {
        using Xbyak::Ymm;
        using Xbyak::Reg64;
        using Xbyak::Address;
        using Xbyak::RegExp;

        assert(raw_offt <= INT_MAX);
        auto offt = static_cast<int>(raw_offt);

        int scale = 0;

        if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
            offt = offt - 2 * EVEX_max_8b_offt;
            scale = 1;
        } else if (3 * EVEX_max_8b_offt <= offt && offt < 5 * EVEX_max_8b_offt) {
            offt = offt - 4 * EVEX_max_8b_offt;
            scale = 2;
        }

        auto re = RegExp() + base + offt;
        if (scale)
            re = re + reg_EVEX_max_8b_offt * scale;

        if (bcast)
            return yword_b [re];
        else
            return yword [re];
    }

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop_fma(int ur_w, int l_overflow, int r_overflow);
    inline void compute_loop_fma_core(int ur_w, int l_overflow, int r_overflow);
    inline void compute_loop(int ur_w, int l_overflow, int r_overflow);
    void generate();
};
} // namespace jit
} // namespace saber
} // namespace anakin
#endif // ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX2_DECONV_ACT_KERNEL_H
