/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */


#ifndef CPU_JIT_UNI_POOL_KERNEL_F32_H
#define CPU_JIT_UNI_POOL_KERNEL_F32_H

#include <cfloat>

#include "saber/funcs/impl/x86/kernel/jit_generator.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "saber/funcs/impl/x86/x86_utils.h"
#include "saber/saber_types.h"


namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

struct jit_uni_pool_kernel_f32{

    jit_uni_pool_kernel_f32() {}

    jit_uni_pool_kernel_f32(jit_pool_conf_t ajpp): jpp(ajpp) {
    }

    jit_pool_conf_t jpp;

    virtual ~jit_uni_pool_kernel_f32() {}
    void operator()(jit_pool_call_t *arg) { jit_ker(arg); }

protected:
    void (*jit_ker)(jit_pool_call_t *);
};

template <cpu_isa_t isa>
struct jit_pool_kernel_f32: public jit_uni_pool_kernel_f32, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_pool_kernel_f32);

    jit_pool_kernel_f32(jit_pool_conf_t ajpp): jit_uni_pool_kernel_f32(ajpp), jit_generator() {
        this->generate();
        jit_ker = (decltype(jit_ker))this->getCode();
    }

    static bool init_conf(jit_pool_conf_t &jpp);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm, isa == avx2,
                                             Ymm, Zmm>::type;
    Xmm xreg(int idx) { return Xmm((isa == avx512_common ? 31 : 15) - idx); }
    Ymm yreg(int idx) { return Ymm(xreg(idx).getIdx()); }
    Vmm vreg(int idx) { return Vmm(xreg(idx).getIdx()); }

    const AddressFrame &vmmword = (isa == sse42) ? xword :
                                  (isa == avx2) ? yword : zword;

    Xmm vmm_mask = Xmm(0);
    Xmm xmm_ker_area_h = Xmm(2);
    Xmm xmm_one = Xmm(2);
    Xmm xmm_tmp = Xmm(3);

    Vmm vmm_ker_area_h = Vmm(2);
    Vmm vmm_one = Vmm(2);
    Vmm vmm_tmp = Vmm(3);

    Vmm vmm_k_offset = Vmm(1);

    Opmask k_index_mask = Opmask(6);
    Opmask k_store_mask = Opmask(7);

    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input      = r8;
    reg64_t aux_reg_input  = r9;
    reg64_t reg_index      = r10;
    reg64_t reg_output     = r12;
    reg64_t reg_arr_init   = r13;
    reg64_t dst_ptr        = abi_param1;

    reg64_t kj      = r14;
    reg64_t oi_iter = r15;
    reg64_t reg_kh  = rax;
    reg64_t reg_k_shift  = rbx;
    reg64_t tmp_gpr = abi_not_param1;
    reg64_t reg_ker_area_h = rdx;

    reg64_t zero_size = r15;
    reg64_t ki = r12;
    reg64_t aux_reg_input_d = r8;

    Xbyak::Reg32 reg_shuf_mask = esi;

    int prev_kw;

    void maybe_recalculate_divisor(int jj, int ur_w, int pad_l, int pad_r);
    void avg_step(int ur_w, int pad_l, int pad_r, const char *kh_label);
    void max_step_fwd(int ur_w, int pad_l, int pad_r, const char *kh_label);
    void max_step_bwd(int ur_w, int pad_l, int pad_r, const char *kh_label);

    void maybe_zero_diff_src();

    void step(int ur_w, int pad_l, int pad_r, const char *kh_label) {
        if (jpp.alg == Pooling_max) {
            max_step_fwd(ur_w, pad_l, pad_r, kh_label);
        }
        else {
            avg_step(ur_w, pad_l, pad_r, kh_label);
        }
    }

    void step_high_half(int ur_w, int pad_l, int pad_r, const char *kh_label) {
        add(reg_input, sizeof(float) * 4);
        add(reg_output, sizeof(float) * 4);

        step(ur_w, pad_l, pad_r, kh_label);
    }

    void generate();
};

} // namespace jit
} // namespace saber
} // namespace anakin

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
