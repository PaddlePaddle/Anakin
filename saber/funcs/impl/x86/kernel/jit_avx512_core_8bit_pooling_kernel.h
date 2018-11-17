#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_JIT_AVX512_CORE_8BIT_POOLING_KERNEL_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_JIT_AVX512_CORE_8BIT_POOLING_KERNEL_H

#include <iostream>
#include <stddef.h>

#include "saber/funcs/impl/x86/kernel/jit_generator.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

struct jit_avx512_core_8bit_pooling_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_8bit_pooling_kernel)

    jit_avx512_core_8bit_pooling_kernel(const jit_pool_conf_t &jpp_) : jpp(jpp_) {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(getCode()));
    }

    Reg64 reg_ptr_src = r8;
    Reg64 reg_ptr_dst = r9;

    Reg64 ki = r10;
    Reg64 kj = r11;
    Reg64 reg_kw = r12;
    Reg64 reg_kh = r13;
    Reg64 c_iter = r14;

    Reg64 aux_reg_src_h = rax;
    Reg64 aux_reg_src_w = rbx;

    Reg64 reg_tmp = rdx;

    Reg64 reg_mask = r15;

    Opmask k_cmp_mask = Opmask(7);

    Opmask mask(int idx) {
        return Opmask(6 - idx);
    }

    Xmm xmm_tmp = Xmm(0);
    Xmm xmm_zeros = Xmm(31);
    Zmm vreg_tmp = Zmm(30);
    Zmm vreg_zeros = Zmm(31);

    size_t sizeof_src_dt() const {
        return datatype_size(jpp.src_dt);
    }
    size_t sizeof_dst_dt() const {
        return datatype_size(jpp.dst_dt);
    }

    /* max pooling */
    Zmm vreg_src(int idx) {
        return Zmm(idx);
    }

    Zmm vreg_dst(int idx) {
        return Zmm(jpp.ur_c + idx);
    }

    /* avg pooling */
    Zmm vreg_src_s32(int jj, int ll) {
        return Zmm(12*jj + ll);
    }

    Zmm vreg_dst_s32(int jj, int ll) {
        return Zmm(12*jj + ll + 4);
    }

    Zmm vreg_dst_f32(int jj, int ll) {
        return Zmm(12*jj + ll + 8);
    }

    void (*ker_)(const jit_pool_call_nhwc_t *);
    jit_pool_conf_t jpp;

    void init_tmp_reg();
    void init_mask();

    void load_src(int jj, int ll, int c_tail);
    void store_dst(int jj, int ll, int c_tail);

    void compute_avg_step(int ur_c, int c_tail);
    void compute_max_step(int ur_c, int c_tail);
    void compute_step(int ur_c, int c_tail);

    void compute_c_block();
    void generate();

    static SaberStatus init_conf(jit_pool_conf_t &jpp);

    void operator()(jit_pool_call_nhwc_t *arg) {ker_(arg);}
};

} // namespace jit
} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_JIT_AVX512_CORE_8BIT_POOLING_KERNEL_H
