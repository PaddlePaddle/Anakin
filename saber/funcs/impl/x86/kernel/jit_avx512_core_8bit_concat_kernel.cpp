#include <iostream>
#include <stddef.h>
#include "saber/funcs/impl/x86/kernel/jit_avx512_core_8bit_concat_kernel.h"

namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

void jit_avx512_core_8bit_concat_kernel::compute_one_input_with_scale(int block_size) {
    Label l_next_block;
    Label l_tail_block;
    Label l_end;

    uni_vpxor(zmm_zero, zmm_zero, zmm_zero);
    mov(reg_ptr_src_i, ptr[reg_ptr_src]);
    mov(reg_ptr_dst_i, reg_ptr_dst);

    cmp(reg_nb, 0);
    je(l_tail_block, T_NEAR);
    L(l_next_block); {
        vpmovzxbd(zmm_src_s32, ptr[reg_ptr_src_i]);
        vcvtdq2ps(zmm_dst_f32, zmm_src_s32);
        vfmadd132ps(zmm_dst_f32, zmm_zero, zword_b[reg_scale]);
        vcvtps2dq(zmm_dst_s32 | T_rn_sae, zmm_dst_f32);
        vpmovusdb(ptr[reg_ptr_dst_i], zmm_dst_s32);

        add(reg_ptr_src_i, block_size);
        add(reg_ptr_dst_i, block_size);
        dec(reg_nb);
        cmp(reg_nb, 0);
        jg(l_next_block, T_NEAR);
    }

    cmp(reg_tail, 0);
    je(l_end, T_NEAR);

    L(l_tail_block);
    {
        vpmovzxbd(zmm_src_s32 | mask(0), ptr[reg_ptr_src_i]);
        vcvtdq2ps(zmm_dst_f32, zmm_src_s32);
        vfmadd132ps(zmm_dst_f32, zmm_zero, zword_b[reg_scale]);
        vcvtps2dq(zmm_dst_s32 | T_rn_sae, zmm_dst_f32);
        vpmovusdb(ptr[reg_ptr_dst_i] ,zmm_dst_s32 | mask(0));
    }

    L(l_end);
}

void jit_avx512_core_8bit_concat_kernel::compute_one_input_without_scale(int block_size) {
    Label l_next_block;
    Label l_tail_block;
    Label l_end;

    uni_vpxor(zmm_zero, zmm_zero, zmm_zero);
    mov(reg_ptr_src_i, ptr[reg_ptr_src]);
    mov(reg_ptr_dst_i, reg_ptr_dst);

    cmp(reg_nb, 0);
    je(l_tail_block, T_NEAR);
    L(l_next_block); {
        vmovdqu8(zmm_src_s32, ptr[reg_ptr_src_i]);
        vmovdqu8(ptr[reg_ptr_dst_i], zmm_src_s32);

        add(reg_ptr_src_i, block_size);
        add(reg_ptr_dst_i, block_size);
        dec(reg_nb);
        cmp(reg_nb, 0);
        jg(l_next_block, T_NEAR);
    }

    cmp(reg_tail, 0);
    je(l_end, T_NEAR);

    L(l_tail_block); {
        vmovdqu8(zmm_src_s32 | mask(0), ptr[reg_ptr_src_i]);
        vmovdqu8(ptr[reg_ptr_dst_i] , zmm_src_s32 | mask(0));
    }

    L(l_end);
}

void jit_avx512_core_8bit_concat_kernel::generate() {
    preamble();

#   define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(jit_concat_call_t, field)])

    READ_PARAM(reg_ptr_src, src);
    READ_PARAM(reg_ptr_dst, dst);
#   undef READ_PARAM

    mov(reg_scale, (size_t)jpp.scales);
    for (int i = 0; i < jpp.n_inputs; i++) {
        mov(reg_tail, jpp.tail[i]);
        kmovq(mask(0), reg_tail);
        mov(reg_nb, jpp.nb_ic[i]);

        if (std::fabs(1.0f - jpp.scales[i]) > FLT_MIN) {
            compute_one_input_with_scale(jpp.block[i]);
        }
        else {
            compute_one_input_without_scale(jpp.block[i]);
        }

        add(reg_ptr_src, sizeof(unsigned char*));
        add(reg_ptr_dst, jpp.ic[i]);
        add(reg_scale, sizeof(float));
    }

    postamble();
}

SaberStatus jit_avx512_core_8bit_concat_kernel::init_conf(jit_concat_conf_t &jpp) {
    SaberStatus ret = SaberUnImplError;

    if (!mayiuse(avx512_core)) {
        return ret;
    }

    return SaberSuccess;
}

}
}
}
