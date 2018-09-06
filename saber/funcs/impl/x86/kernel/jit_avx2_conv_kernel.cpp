#include <iostream>

#include "jit_avx2_conv_kernel.h"

#define GET_OFF(field) offsetof(jit_conv_call_t, field)

namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

inline void jit_avx2_conv_act_kernel::oh_step_unroll_kw(int ur_w,
                                                        int pad_l, int pad_r, int oc_blocks) {
    int iw = jcp.iw;
    int ih = jcp.ih;
    int kw = jcp.kw;
    int kh = jcp.kh;
    int nb_ic = jcp.nb_ic;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;
    // int g_blk = jcp.g_block;
    // bool dw = jcp.is_dw;

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = saber::utils::max(0, saber::utils::div_up(pad_l - ki * dilate_w, stride_w));
        int jj_end = ur_w - saber::utils::max(0, saber::utils::div_up(ki * dilate_w + pad_r - (kw - 1) * dilate_w, stride_w));
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                int inp_off;

                inp_off = ifm2 * ih * iw + (ki * dilate_w + jj * stride_w - pad_l);

                vbroadcastss(Ymm(oc_blocks * ur_w + jj), ptr[aux_reg_input + sizeof(float) * inp_off]);
            }

            for (int ii = 0; ii < oc_blocks; ii++) {
                int ker_off = ii * nb_ic * kh * kw * ic_blk * oc_blk +
                              ki * ic_blk * oc_blk + ifm2 * oc_blk;
                vmovups(ymm15, ptr[aux_reg_kernel + sizeof(float) * ker_off]);
                for (int jj = jj_start; jj < jj_end; jj++)
                    vfmadd231ps(Ymm(ur_w * ii + jj),
                                Ymm(oc_blocks * ur_w + jj), ymm15);
            }
        }
    }
}

inline void jit_avx2_conv_act_kernel::oh_step_nopad(int ur_w,
                                                    int pad_l, int pad_r, char pad_tag,
                                                    int oc_blocks, char oc_blocks_tag) {
    jit_tagged_label kw_label("kw", pad_tag, oc_blocks_tag);

    int iw = jcp.iw;
    int ih = jcp.ih;
    int kw = jcp.kw;
    int kh = jcp.kh;
    int nb_ic = jcp.nb_ic;
    int stride_w = jcp.stride_w;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;
    // int g_blk = jcp.g_block;
    // bool dw = jcp.is_dw;

    xor_(ki_iter, ki_iter);
    L(kw_label);
    {
        int jj_start = 0;
        int jj_end = ur_w;
        for (int ifm2 = 0; ifm2 < ic_blk; ifm2++) {
            for (int jj = jj_start; jj < jj_end; jj++) {
                int inp_off;
                // if (jcp.src_fmt == nchw)
                inp_off = ifm2 * ih * iw + (jj * stride_w - pad_l);
                // else
                //  inp_off = (jj * stride_w - pad_l) * (dw ? g_blk : ic_blk) + ifm2;
                //  if (dw)
                //    vmovups(Ymm(oc_blocks * ur_w + jj), ptr[aux_reg_input + sizeof(float) * inp_off]);
                // else
                vbroadcastss(Ymm(oc_blocks * ur_w + jj),
                             ptr[aux_reg_input + sizeof(float) * inp_off]);
            }
            for (int ii = 0; ii < oc_blocks; ii++) {
                int aux_kernel_offset = ii * nb_ic * kh * kw * ic_blk * oc_blk + ifm2 * oc_blk;
                vmovups(ymm15, ptr[aux_reg_kernel + sizeof(float) * aux_kernel_offset]);
                for (int jj = jj_start; jj < jj_end; jj++) {
                    vfmadd231ps(Ymm(ur_w * ii + jj), Ymm(oc_blocks * ur_w + jj), ymm15);
                }
            }
        }
        add(aux_reg_kernel, sizeof(float) * oc_blk * ic_blk);
        add(aux_reg_input, sizeof(float) * dilate_w);

        inc(ki_iter);
        cmp(ki_iter, kw);
        jl(kw_label, T_NEAR);
    }
}

inline void jit_avx2_conv_act_kernel::width_blk_step(int ur_w,
                                                     int pad_l, int pad_r, char pad_tag,
                                                     int oc_blocks, char oc_blocks_tag) {
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ow = jcp.ow;
    int oh = jcp.oh;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;
    // int g_blk = jcp.g_block;
    bool dw = jcp.is_dw;
    const int inp_mult = dilate_h;
    const int inp_off  = dilate_w;

    jit_tagged_label init_done_label("init", pad_tag, oc_blocks_tag);
    jit_tagged_label init_first_label("first", pad_tag, oc_blocks_tag);

    if (!jcp.with_sum) {
        if (dw) {
            jmp(init_first_label, T_NEAR);
        }
        test(reg_ci_flag, 1 << 4);
        jne(init_first_label, T_NEAR);
    }

    for (int ii = 0; ii < oc_blocks; ii++) {
        for (int jj = 0; jj < ur_w; jj++) {
            vmovups(Ymm(ur_w * ii + jj), yword[reg_output + sizeof(float) * (ii * oh * ow + jj) * oc_blk]);
        }
    }

    if (jcp.with_sum && jcp.with_bias) {
        if (!dw) {
            test(reg_ci_flag, 1 << 4);
            je(init_done_label, T_NEAR);
        }

        for (int ii = 0; ii < oc_blocks; ii++) {
            for (int jj = 0; jj < ur_w; jj++) {
                vaddps(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj),
                       yword[reg_bias + sizeof(float) * ii * oc_blk]);
            }
        }
    }

    jmp(init_done_label);

    L(init_first_label);
    if (this->jcp.with_bias) {
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                vmovups(Ymm(ur_w * ii + jj), yword[reg_bias + sizeof(float) * ii * oc_blk]);
    } else {
        for (int ii = 0; ii < oc_blocks; ii++)
            for (int jj = 0; jj < ur_w; jj++)
                vpxor(Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj), Ymm(ur_w * ii + jj));
    }

    L(init_done_label);

    mov(aux_reg_input, reg_input);
    mov(aux_reg_kernel, reg_kernel);

    Label skip_kh_loop;
    mov(kj, reg_kh);
    if (jcp.kh <= jcp.t_pad) {
        cmp(kj, 0);
        je(skip_kh_loop, T_NEAR);
    }

    jit_tagged_label kh_label("kh", pad_tag, oc_blocks_tag);

    L(kh_label);
    {
        if (jcp.kw >= 5 && pad_l == 0 && pad_r == 0) {
            oh_step_nopad(ur_w, pad_l, pad_r, pad_tag, oc_blocks,
                          oc_blocks_tag);
            sub(aux_reg_input, sizeof(float) * kw * inp_off);
            add(aux_reg_input, sizeof(float) * iw * inp_mult);
        } else {
            oh_step_unroll_kw(ur_w, pad_l, pad_r, oc_blocks);
            add(aux_reg_kernel, sizeof(float) * kw * oc_blk * ic_blk);
            add(aux_reg_input, sizeof(float) * iw * inp_mult);
        }

        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);

    jit_tagged_label done_label("done", pad_tag, oc_blocks_tag);
    jit_tagged_label regular_store_label("store", pad_tag, oc_blocks_tag);

    if (this->jcp.with_relu) {
        assert(oc_blocks * ur_w < 15);
        if (!dw) {
            test(reg_ci_flag, 1<<5);
            je(regular_store_label, T_NEAR);
        }
        vxorps(yzero, yzero, yzero);
        if (jcp.relu_negative_slope == 0) {
            ymm_relu_ns = yzero;
        } else {
            mov(imm_addr64, float2int(jcp.relu_negative_slope));
            movq(xmm_relu_ns, imm_addr64);
            uni_vbroadcastss(ymm_relu_ns, xmm_relu_ns);
        }

        for (int ii = 0; ii < oc_blocks; ii++) {
            for (int jj = 0; jj < ur_w; jj++) {
                const size_t o_off = (ii * oh * ow + jj) * oc_blk;
                Ymm reg_out = Ymm(ur_w * ii + jj);

                vcmpgtps(ymask, reg_out, yzero);
                vmulps(ymm_res_ns, ymm_relu_ns, reg_out);
                vblendvps(reg_out, ymm_res_ns, reg_out, ymask);
                vmovups(yword[reg_output + sizeof(float) * o_off], reg_out);
            }
        }

        jmp(done_label);
        L(regular_store_label);
    }

    for (int ii = 0; ii < oc_blocks; ii++) {
        for (int jj = 0; jj < ur_w; jj++) {
            const size_t o_off = (ii * oh * ow + jj) * oc_blk;
            Ymm reg_out = Ymm(ur_w * ii + jj);
            vmovups(yword[reg_output + sizeof(float) * o_off], reg_out);
        }
    }
    L(done_label);
}

inline void jit_avx2_conv_act_kernel::solve_common(
        int oc_blocks, char oc_blocks_tag) {
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int n_oi = jcp.ow / ur_w;
    int iw = jcp.iw;
    int kw = jcp.kw;
    // int ic_blk = jcp.ic_block;
    int oc_blk = jcp.oc_block;
    // int g_blk = jcp.g_block;
    int dilate_w = jcp.dilate_w + 1;
    int str_w = jcp.stride_w;
    // bool dw = jcp.is_dw;
    const int inp_mult =  1;

    int l_pad = jcp.l_pad;
    int r_pad = saber::utils::max(0, (int(jcp.ow) - 1) * str_w + (kw - 1) * dilate_w
                                     - (iw + l_pad - 1));
    int r_pad1 = (ur_w * n_oi - 1) * str_w + (kw - 1) * dilate_w
                 - (iw + l_pad - 1);
    if (r_pad1 > 0) n_oi--;

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0)
            width_blk_step(ur_w, l_pad, r_pad1,
                           'l', oc_blocks, oc_blocks_tag); // "lrpad"
        else
            width_blk_step(ur_w, l_pad, 0,
                           'l', oc_blocks, oc_blocks_tag); // "lpad"
        add(reg_input, sizeof(float) * (ur_w * str_w - l_pad) * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
    }

    jit_tagged_label ow_loop_label("ow", oc_blocks_tag);
    xor_(oi_iter, oi_iter);

    if (n_oi > 0) {
        L(ow_loop_label);

        width_blk_step(ur_w, 0, 0,
                       'm', oc_blocks, oc_blocks_tag); // "middle"
        add(reg_input, sizeof(float) * ur_w * str_w * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);

        inc(oi_iter);
        cmp(oi_iter, n_oi);
        jl(ow_loop_label, T_NEAR);
    }

    if (r_pad1 > 0 && n_oi >=0) {
        width_blk_step(ur_w, 0, r_pad1,
                       'r', oc_blocks, oc_blocks_tag); // "rpad"
        add(reg_input, sizeof(float) * ur_w * str_w * inp_mult);
        add(reg_output, sizeof(float) * ur_w * oc_blk);
    }

    if (ur_w_tail != 0)
        width_blk_step(ur_w_tail, 0, r_pad,
                       't', oc_blocks, oc_blocks_tag); // "tail"
}

void jit_avx2_conv_act_kernel::generate() {
    this->preamble();

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kernel, ptr[this->param1 + GET_OFF(filt)]);
    if (jcp.with_bias)
        mov(reg_bias, ptr[this->param1 + GET_OFF(bias)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_ci_flag, ptr[this->param1 + GET_OFF(flags)]);
    mov(reg_oc_blocks, ptr[this->param1 + GET_OFF(oc_blocks)]);

    const char *tail_label = ".tail";
    const char *exit_label = ".exit";

    if (jcp.is_dw) {
        solve_common(jcp.ic_block, '0');
        jmp(exit_label, T_NEAR);
    }

    int nb_oc_tail = jcp.nb_oc % jcp.nb_oc_blocking;
    cmp(reg_oc_blocks, jcp.nb_oc_blocking);
    jne(nb_oc_tail ? tail_label : exit_label, T_NEAR);

    solve_common(jcp.nb_oc_blocking, '0' + jcp.nb_oc_blocking);
    jmp(exit_label, T_NEAR);

    if (nb_oc_tail) {
        L(tail_label);
        cmp(reg_oc_blocks, nb_oc_tail);
        jne(exit_label, T_NEAR);
        solve_common(nb_oc_tail, '0' + nb_oc_tail);
    }
    L(exit_label);

    this->postamble();
}


SaberStatus jit_avx2_conv_act_kernel::init_conf(jit_conv_conf_t &jcp) {
    if (!mayiuse(avx2)) {
        LOG(ERROR) << "init a AVX2 kernel in a non-avx2 machine is not permitted";
        return SaberUnImplError;
    }

    bool with_groups = false;
    const bool flat = jcp.ic == 3;
    const bool depthwise = true
                           && with_groups
                           && jcp.oc == 1
                           && jcp.ic == 1;
    const bool mimo = !flat && !depthwise;
    jcp.is_dw = depthwise ? true : false;
    const int simd_w = 8;

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.ur_w = 3;

    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.nb_oc_blocking = 4; /* the optimal value for the kernel */

    bool args_ok = true
                   && (jcp.oc % simd_w == 0)
                   && jcp.l_pad <= jcp.ur_w
                   && utils::implication(jcp.kw > 7, (jcp.t_pad == 0 && jcp.l_pad == 0)
                                                     || (jcp.stride_w == 1 && jcp.stride_h == 1))
                   && utils::implication(mimo, jcp.ic % simd_w == 0);
    // && implication(depthwise, jcp.ngroups % simd_w == 0);
    if (!args_ok) {
        LOG(ERROR) << "arguments check failed";
        return SaberUnImplError;
    }

    int r_pad_no_tail = saber::utils::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                                             + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));

    if (r_pad_no_tail > jcp.ur_w) {
        /* recalculate ur_w, nb_oc_blocking and ur_w_tail */
        jcp.ur_w = r_pad_no_tail + 1;
        jcp.nb_oc_blocking = ((16 - 1) - jcp.ur_w) / jcp.ur_w;
        jcp.ur_w_tail = jcp.ow % jcp.ur_w;
        /* check again ... */
        r_pad_no_tail = saber::utils::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                                             + (jcp.kw - 1) * (jcp.dilate_w + 1) - (jcp.iw + jcp.l_pad - 1));
        if ((r_pad_no_tail > jcp.ur_w) || (jcp.ow < jcp.ur_w)) {
            LOG(ERROR) << "tail should not be greater than ur_w";
            return SaberUnImplError;
        }
    }

    if (jcp.l_pad > jcp.ur_w) {
        LOG(ERROR) << "pad should not be greater than ur_w";
        return SaberUnImplError;
    }

    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.g_block = simd_w;
    jcp.nb_g = jcp.ngroups / jcp.g_block;

    jcp.nb_ic_blocking = 12;
    jcp.nb_ic_blocking_max = 16;

    return SaberSuccess;
}

} // namespace jit
} // namespace saber
} // namespace anakin
