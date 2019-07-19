#include <iostream>

#include "jit_uni_dwconv_kernel_f32.h"
#include "utils/logger/logger.h"

#define GET_OFF(field) offsetof(jit_conv_call_t, field)

namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

template <cpu_isa_t isa>
void jit_dwconv_kernel_f32<isa>::load_src(int ur_ch_blocks, int ur_w) {
    int repeats = isa == sse42 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int ow = 0; ow < ur_w; ow++) {
                Vmm vmm_acc = get_acc_reg(i*ur_ch_blocks*ur_w + ch*ur_w + ow);
                int b_off = ch*jcp.ch_block + i*4;
                if (this->jcp.with_bias) {
                    uni_vmovups(vmm_acc,
                                vmmword[reg_bias + b_off*sizeof(float)]);
                } else {
                    uni_vpxor(vmm_acc, vmm_acc, vmm_acc);
                }
                int o_off = ch*jcp.oh*jcp.ow*jcp.ch_block
                            + ow*jcp.ch_block + i*4;
                if (this->jcp.with_sum) {
                    uni_vaddps(vmm_acc, vmm_acc,
                               vmmword[reg_output + o_off*sizeof(float)]);
                }
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_dwconv_kernel_f32<isa>::apply_filter(
        int ur_ch_blocks, int ur_w) {
    int ch_blk = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    Label iter_exit_label;
    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);
    cmp(reg_kw, 0);
    je(iter_exit_label, T_NEAR);
    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label); {
        mov(iter_kw, reg_kw);
        mov(aux1_reg_input, aux_reg_input);
        mov(aux1_reg_kernel, aux_reg_kernel);
        Label kw_label;
        L(kw_label); {
            int repeats = isa == sse42 ? 2 : 1;
            for (int i = 0; i < repeats; i++) {
                for (int ch = 0; ch < ur_ch_blocks; ch++) {
                    int ker_off = ch * jcp.kh * jcp.kw * ch_blk + i * 4;
                    Vmm vmm_ker = get_ker_reg(0);
                    uni_vmovups(vmm_ker, ptr[aux1_reg_kernel + ker_off * sizeof(float)]);
                    for (int ow = 0; ow < ur_w; ow++) {
                        int inp_off = ch * jcp.ih * jcp.iw * ch_blk +
                                      ow * stride_w * ch_blk + i * 4;
                        Vmm vmm_src = get_src_reg(0);
                        uni_vmovups(vmm_src, ptr[aux1_reg_input + inp_off * sizeof(float)]);
                        Vmm vmm_acc = get_acc_reg(i * ur_ch_blocks * ur_w + ch * ur_w + ow);
                        uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
                    }
                }
            }
            add(aux1_reg_kernel, ch_blk * sizeof(float));
            add(aux1_reg_input, ch_blk * dilate_w * sizeof(float));
            dec(iter_kw);
            cmp(iter_kw, 0);
            jg(kw_label, T_NEAR);
        }
        add(aux_reg_kernel, jcp.kw * ch_blk * sizeof(float));
        add(aux_reg_input, jcp.iw * ch_blk * dilate_h * sizeof(float));
        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }
    L(iter_exit_label);
}

template <cpu_isa_t isa>
void jit_dwconv_kernel_f32<isa>::apply_filter_unrolled(int ur_ch_blocks, int ur_w) {
    int ch_blk = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    Label iter_exit_label;
    cmp(reg_kh, 0);
    je(iter_exit_label, T_NEAR);
    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label); {
        int repeats = isa == sse42 ? 2 : 1;
        for (int i = 0; i < repeats; i++) {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int kw = 0; kw < jcp.kw; kw++) {
                    int ker_off = ch * jcp.kh * jcp.kw * ch_blk + kw * ch_blk + i * 4;
                    Vmm vmm_ker = get_ker_reg(0);
                    uni_vmovups(vmm_ker, ptr[aux_reg_kernel + ker_off * sizeof(float)]);
                    for (int ow = 0; ow < ur_w; ow++) {
                        int inp_off = ch * jcp.ih * jcp.iw * ch_blk +
                                      ow * stride_w * ch_blk + kw * ch_blk * dilate_w + i * 4;
                        Vmm vmm_src = get_src_reg(0);
                        uni_vmovups(vmm_src, ptr[aux_reg_input +
                                                 inp_off * sizeof(float)]);
                        Vmm vmm_acc = get_acc_reg(i * ur_ch_blocks * ur_w +
                                                  ch * ur_w + ow);
                        uni_vfmadd231ps(vmm_acc, vmm_src, vmm_ker);
                    }
                }
            }
        }
        add(aux_reg_kernel, jcp.kw * ch_blk * sizeof(float));
        add(aux_reg_input, jcp.iw * ch_blk * dilate_h * sizeof(float));
        dec(iter_kh);
        cmp(iter_kh, 0);
        jg(kh_label, T_NEAR);
    }
    L(iter_exit_label);
}

template <cpu_isa_t isa>
void jit_dwconv_kernel_f32<isa>::apply_activation(int ur_ch_blocks, int ur_w) {
    if (this->jcp.with_relu) {
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);
        if (jcp.relu_negative_slope == 0) {
            vmm_relu_ns = vmm_zero;
        } else {
            mov(imm_addr64, float2int(jcp.relu_negative_slope));
            movq(xmm_relu_ns, imm_addr64);
            uni_vbroadcastss(vmm_relu_ns, xmm_relu_ns);
        }
        int repeats = isa == sse42 ? 2 : 1;
        for (int i = 0; i < repeats; i++) {
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int ow = 0; ow < ur_w; ow++) {
                    Vmm vmm_dst = get_acc_reg(i*ur_ch_blocks*ur_w
                                              + ch*ur_w + ow);
                    if (isa == sse42) {
                        pxor(vmm_mask, vmm_mask);
                        cmpps(vmm_mask, vmm_dst, _cmp_gt_os);
                        movups(vmm_res_ns, vmm_dst);
                        mulps(vmm_res_ns, vmm_relu_ns);
                        blendvps(vmm_dst, vmm_res_ns);
                    } else if (isa == avx2) {
                        vcmpgtps(vmm_mask, vmm_dst, vmm_zero);
                        vmulps(vmm_res_ns, vmm_relu_ns, vmm_dst);
                        vblendvps(vmm_dst, vmm_res_ns, vmm_dst, vmm_mask);
                    } else if (isa == avx512_common) {
                        Opmask kmask = Opmask(7);
                        vcmpps(kmask, vmm_dst, vmm_zero, _cmp_lt_os);
                        vmulps(vmm_dst | kmask, vmm_dst, vmm_relu_ns);
                    }
                }
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_dwconv_kernel_f32<isa>::store_dst(
        int ur_ch_blocks, int ur_w) {
    int ch_blk = jcp.ch_block;
    int repeats = isa == sse42 ? 2 : 1;
    for (int i = 0; i < repeats; i++) {
        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            for (int ow = 0; ow < ur_w; ow++) {
                int o_off = ch * jcp.oh * jcp.ow * ch_blk + ow * ch_blk + i * 4;
                Vmm vmm_dst = get_acc_reg(i * ur_ch_blocks * ur_w + ch * ur_w + ow);
                uni_vmovups(vmmword[reg_output + o_off * sizeof(float)], vmm_dst);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_dwconv_kernel_f32<isa>::loop_body(int ur_ch_blocks) {
    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;
    L(unrolled_w_label); {
        int ur_w = jcp.ur_w;
        cmp(reg_ur_w, ur_w);
        jl(tail_w_label, T_NEAR);
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
        load_src(ur_ch_blocks, ur_w);
        apply_filter_unrolled(ur_ch_blocks, ur_w);
        apply_activation(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w);
        add(reg_input, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, sizeof(float) * ur_w * jcp.ch_block);
        sub(reg_ur_w, ur_w);
        jmp(unrolled_w_label);
    }
    L(tail_w_label); {
        int ur_w = 1;
        cmp(reg_ur_w, ur_w);
        jl(exit_label, T_NEAR);
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
        load_src(ur_ch_blocks, ur_w);
        apply_filter(ur_ch_blocks, ur_w);
        apply_activation(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w);
        add(reg_input, sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w);
        add(reg_output, sizeof(float) * ur_w * jcp.ch_block);
        sub(reg_ur_w, ur_w);
        jmp(tail_w_label);
    }
    L(exit_label);
}

template <cpu_isa_t isa>
void jit_dwconv_kernel_f32<isa>::generate() {
    this->preamble();

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_kw, ptr[this->param1 + GET_OFF(kw_padding)]);
    mov(reg_ch_blocks, ptr[this->param1 + GET_OFF(ch_blocks)]);
    mov(reg_ur_w, ptr[this->param1 + GET_OFF(ur_w)]);
    Label ch_blocks_tail_label;
    Label exit_label;
    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;
    cmp(reg_ch_blocks, jcp.nb_ch_blocking);
    jne(ch_blocks_tail ? ch_blocks_tail_label : exit_label, T_NEAR);
    loop_body(jcp.nb_ch_blocking); // channel main loop
    if (ch_blocks_tail) {
        L(ch_blocks_tail_label);
        cmp(reg_ch_blocks, ch_blocks_tail);
        jne(exit_label, T_NEAR);
        loop_body(ch_blocks_tail); // channel tail loop
    }
    L(exit_label);
    this->postamble();
}


template <cpu_isa_t isa>
SaberStatus jit_dwconv_kernel_f32<isa>::init_conf(jit_conv_conf_t &jcp) {
    if (!mayiuse(isa) && isa == avx512_common) {
                LOG(ERROR) << "Init an AVX512 kernel in a non-avx512 machine is not permitted";
        return SaberUnImplError;
    }

    const int simd_w = isa == avx512_common ? 16 : 8;

    jcp.ur_w = isa == avx512_common ? 6 : isa == avx2 ? 4 : 3;

    jcp.ch_block = simd_w;
    jcp.nb_ch = jcp.oc / jcp.ch_block;
    jcp.nb_ch_blocking = isa == avx512_common ? 4 : isa == avx2 ? 3 : 2;
    if (jcp.nb_ch < jcp.nb_ch_blocking) {
        jcp.nb_ch_blocking = jcp.nb_ch;
    }

    return SaberSuccess;
}

template struct jit_dwconv_kernel_f32<avx512_common>;
template struct jit_dwconv_kernel_f32<avx2>;
}
}
}

