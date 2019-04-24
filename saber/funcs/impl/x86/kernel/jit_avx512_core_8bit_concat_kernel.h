
#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX512_CORE_8BIT_CONCAT_KERNEL_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AVX512_CORE_8BIT_CONCAT_KERNEL_H

#include <iostream>
#include <stddef.h>
#include <float.h>
#include <math.h>

#include "saber/funcs/impl/x86/kernel/jit_generator.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/x86/x86_utils.h"

namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

struct jit_avx512_core_8bit_concat_kernel: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_8bit_concat_kernel)

    enum {
        USE_ZMM = 512,
        USE_YMM = 256,
        USE_XMM = 128,
    };

    Reg64 param = abi_param1;
    Reg64 reg_ptr_src = r8;
    Reg64 reg_ptr_src_i = r9;
    Reg64 reg_ptr_dst = r10;
    Reg64 reg_ptr_dst_i = r11;
    Reg64 reg_nb = r15;
    Reg64 reg_scale = r13;
    Reg64 reg_tail = r14;
    Reg64 reg_ninputs = rbx;

    Xmm xmm_src = Xmm(30);
    Xmm xmm_dst = Xmm(31);

    Zmm zmm_zero = Zmm(23);
    Zmm zmm_src_s32 = Zmm(26);
    Zmm zmm_dst_s32 = Zmm(27);
    Zmm zmm_dst_f32 = Zmm(28);
    Zmm zmm_scale = Zmm(25);
    Xmm xmm_scale = Xmm(25);
    Zmm zmm_scale_min = Zmm(24);
    Xmm xmm_scale_min = Xmm(24);

    Opmask mask(int idx) {
        return Opmask(6 - idx);
    }

    void compute_one_input_with_scale(int block_size);
    void compute_one_input_without_scale(int block_size);
    void (*ker_)(const jit_concat_call_t *);
    jit_concat_conf_t jpp;

    void generate();

    static SaberStatus init_conf(jit_concat_conf_t &jpp);

    jit_avx512_core_8bit_concat_kernel(const jit_concat_conf_t &jpp_)
           : jpp(jpp_) {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(getCode()));
    }

    void operator()(jit_concat_call_t *arg) {ker_(arg);}
};

}
}
}

#endif
