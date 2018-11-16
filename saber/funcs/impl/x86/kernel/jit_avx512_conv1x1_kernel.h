/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef ANAKIN_SABER_FUNCS_JIT_AVX512_CONV1X1_KERNEL_H
#define ANAKIN_SABER_FUNCS_JIT_AVX512_CONV1X1_KERNEL_H

#include "saber/funcs/impl/impl_base.h"
#include "saber/core/tensor.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "jit_uni_1x1_conv_utils.h"
#include "jit_generator.h"

namespace anakin {
namespace saber {
namespace jit {

struct jit_avx512_common_1x1_conv_kernel : public jit_generator {
    jit_avx512_common_1x1_conv_kernel(jit_1x1_conv_conf_t ajcp) : jcp(ajcp) {
        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_t *)) this->getCode();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_1x1_conv_kernel)

    static SaberStatus init_conf(jit_1x1_conv_conf_t &jcp, conv_1x1_desc &conv_d,
                                 int nthreads, bool reduce_src = false);

    jit_1x1_conv_conf_t jcp;
    void (*jit_ker)(jit_1x1_conv_call_t *);

private:
    using reg64_t = const Xbyak::Reg64;
    using zmm_t = const Xbyak::Zmm;
    using mask_t = const Xbyak::Opmask;

    reg64_t reg_bcast_data = r8;
    reg64_t reg_load_data = r10;
    reg64_t reg_output_data = r9;
    reg64_t aux_reg_bcast_data = r14;
    reg64_t aux1_reg_bcast_data = rbx;
    reg64_t aux_reg_load_data = r15;
    reg64_t imm_addr64 = aux_reg_load_data;
    reg64_t aux_reg_output_data = abi_not_param1;
    reg64_t reg_load_loop_work = rsi;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t bcast_loop_iter = rdx;
    reg64_t reduce_loop_iter = abi_param1;
    reg64_t reg_reduce_pos_flag = rax;
    reg64_t reg_output_stride = r13;
    reg64_t reg_bias_data = r12;
    reg64_t reg_relu_ns = r13;
    reg64_t reg_bcast_loop_work = aux1_reg_bcast_data;
    mask_t vmask = k7;

    Xbyak::Xmm xmm_relu_ns = Xbyak::Xmm(30);
    Xbyak::Zmm zmm_relu_ns = Xbyak::Zmm(30);
    Xbyak::Zmm zmm_zero = Xbyak::Zmm(31);
    Xbyak::Zmm vreg_bcast = Xbyak::Zmm(31);

    int bcast_loop_work_offt = 0;
    int stack_space_needed = 16;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);

    void generate();
    static void balance(jit_1x1_conv_conf_t &jcp, int nthreads);
};

} // namespace jit
} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_JIT_AVX512_CONV1X1_KERNEL_H
