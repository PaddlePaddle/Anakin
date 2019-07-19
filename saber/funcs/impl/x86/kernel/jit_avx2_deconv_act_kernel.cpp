#include "saber/funcs/impl/x86/kernel/jit_avx2_deconv_act_kernel.h"
#define GET_OFF(field) offsetof(jit_deconv_call_t, field)

namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

void jit_avx2_deconv_act_kernel::prepare_output(int ur_w)
{
    int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        // vmovups();
        for (int j = 0; j < ur_w; j++) {
            Ymm ymm = ymm_out(j, k);
            vxorpd(ymm, ymm, ymm);
        }
    }
}

void jit_avx2_deconv_act_kernel::store_output(int ur_w)
{
    Label no_update_label;
    Label store_label;

    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    if (jcp.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    }

    cmp(reg_channel, 0);
    je(no_update_label, T_NEAR);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Ymm ymm = ymm_out(j, k);
            size_t aux_src_offset = (size_t)typesize
                * ((size_t)k * jcp.ih * jcp.iw + j) * jcp.ic_block;
            vadd(ymm, make_safe_addr(reg_src, aux_src_offset,
                        reg_long_offt));
        }
    }
    jmp(store_label, T_NEAR);

    L(no_update_label);
    if (jcp.with_bias) {
        for (int k = 0; k < jcp.nb_ic_blocking; k++) {
            int bias_offset = typesize * k * jcp.ic_block;
            for (int j = 0; j < ur_w; j++) {
                Ymm ymm = ymm_out(j, k);
                vadd(ymm, make_safe_addr(reg_bias, bias_offset, reg_long_offt));
            }
        }
    }

    L(store_label);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Ymm ymm = ymm_out(j, k);
            size_t aux_src_offset = (size_t)typesize
                * ((size_t)k * jcp.ih * jcp.iw + j) * jcp.ic_block;
            vmovups(make_safe_addr(reg_src, aux_src_offset,
                        reg_long_offt), ymm);
        }
    }

}

void jit_avx2_deconv_act_kernel::compute_loop_fma(
        int ur_w, int l_overflow, int r_overflow)
{
    Label kh_label;
    Label kd_label;
    Label skip_kd_loop;
    Label store_output_label;
    int kw = jcp.kw;
    int ow = jcp.ow;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int l_pad = jcp.l_pad;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;

    int ker_pipeline_depth = 1;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 15);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = oc_block * kw;
    int num_inp_prfs = ur_w * utils::min(kw, stride_w)
                       + utils::max(0, kw - stride_w);
    int num_prfs = num_ker_loads + num_inp_prfs;
    int num_fmas = num_ker_loads * ur_w / stride_w;
    int prf_inst_spacing = utils::max(1, num_fmas / num_prfs);
    int prf_inst_trigger = (num_fmas % prf_inst_spacing) / 2;

    prepare_output(ur_w);

    mov(aux_reg_dst, reg_dst);
    mov(aux_reg_ker, reg_ker);

    mov(aux_reg_dst_prf, reg_dst_prf);
    mov(aux_reg_ker_prf, reg_ker_prf);

    mov(reg_kj, reg_kh);

    cmp(reg_kj, 0);
    je(store_output_label, T_NEAR);

    L(kh_label); {
        for (int ki = 0; ki < kw; ki++) {
            for (int oc = 0; oc < oc_block; oc++) {
                int aux_kernel_offset = typesize * ((oc * oc_block
                + ki * ic_block * oc_block));
                vmovups(ymm_wei, make_safe_addr(aux_reg_ker, aux_kernel_offset, reg_long_offt));

                int jj_start = get_iw_start(ki, l_overflow);
                int jj_end = get_iw_end(ur_w, ki, r_overflow);
                assert(stride_w != 1
                        || jj_start == utils::max(0,
                            l_overflow - (kw - 1 - ki) * dilate_w));
                assert(stride_w != 1
                        || jj_end == ur_w - utils::max(0,
                            r_overflow - ki * dilate_w));

                for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                    assert((jj + l_pad - ki * dilate_w) % stride_w == 0);
                    int aux_dst_offset = typesize *
                        (((jj + l_pad - ki * dilate_w)
                                / stride_w) * jcp.oc_block + oc);
                    vbroadcastss(ymm_temp, ptr[aux_reg_dst + aux_dst_offset]);
                    vfmadd231ps(ymm_out(jj, 0), ymm_wei, ymm_temp);
                }
            }
        }

        add(aux_reg_ker, typesize * stride_h * kw * oc_block * ic_block);
        sub(aux_reg_dst, typesize * (jcp.dilate_h + 1) * ow * oc_block);
        add(aux_reg_ker_prf, typesize * stride_h * kw * oc_block * ic_block);
        sub(aux_reg_dst_prf, typesize * (jcp.dilate_h + 1) * ow * oc_block);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    L(store_output_label); {
        store_output(ur_w);
    }
}

void jit_avx2_deconv_act_kernel::compute_loop_fma_core(int ur_w, int l_overflow, int r_overflow) {
    int kw = jcp.kw;
    int ow = jcp.ow;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_ic_block = jcp.nb_ic_blocking;
    Label kh_label;
    Label skip_kh_loop;
    Label kd_label;
    Label skip_kd_loop;

    int shift_ker_ptr = typesize * kw * oc_block * ic_block;
    int shift_dst_ptr = typesize * (jcp.dilate_h + 1) * ow * oc_block;

    auto output_offset = [=](int oi, int oc, int ki) {
        return typesize *
            (((oi + jcp.l_pad - ki * dilate_w) / stride_w) * oc_block + oc);
    };
    auto kernel_offset = [=](int icb, int oc, int ki) {
        int blk_idx = icb * jcp.kh * jcp.kw + ki;
        int blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        int oc_offset = oc * jcp.oc_block;
        return typesize * (blk_offset + oc_offset);
    };

    prepare_output(ur_w);

    mov(aux_reg_dst, reg_dst);
    mov(aux_reg_ker, reg_ker);

    mov(reg_kj, reg_kh);

    cmp(reg_kj, 0);
    je(skip_kh_loop, T_NEAR);

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);
            for (int oc = 0; oc < oc_block; oc++) {
                if (jcp.kernel_kind == expl_bcast) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        int aux_output_offset = output_offset(jj, oc, ki);
                        vbroadcastss(ymm_inp(jj, nb_ic_block),
                            ptr[aux_reg_dst + aux_output_offset]);
                    }
                }
                for (int ii = 0; ii < nb_ic_block; ii++) {
                    int aux_kernel_offset = kernel_offset(ii, oc, ki);
                    if (jj_end - jj_start > 0) {
                        vmovups(ymm_wei, make_safe_addr(aux_reg_ker,
                                                        aux_kernel_offset, reg_long_offt));
                    }
                    for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                        if (jcp.kernel_kind == expl_bcast) {
                            vfmadd231ps(ymm_out(jj, ii),
                                        ymm_inp(jj, nb_ic_block), ymm_wei);
                        } else {
                            vbroadcastss(ymm_temp, ptr[aux_reg_dst + output_offset(jj, oc, ki)]);
                            vfmadd231ps(ymm_out(jj, ii), ymm_wei, ymm_temp);
                        }
                    }
                }
            }
        }
        add(aux_reg_ker, shift_ker_ptr);
        sub(aux_reg_dst, shift_dst_ptr);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    L(skip_kh_loop);
    store_output(ur_w);
}

inline void jit_avx2_deconv_act_kernel::compute_loop(
        int ur_w, int l_overflow, int r_overflow)
{

    if (jcp.ver == ver_fma)
        if (jcp.kernel_kind == embd_bcast && jcp.nb_ic_blocking == 1)
            compute_loop_fma(ur_w, l_overflow, r_overflow);
        else
            compute_loop_fma_core(ur_w, l_overflow, r_overflow);
    else
        assert("!unknown convolution version");
}

void jit_avx2_deconv_act_kernel::generate() {
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ur_w = jcp.ur_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    int dst_shift = jcp.typesize_in * (ur_w / stride_w) * ic_block;
    int src_shift = jcp.typesize_out * ur_w * oc_block;

    preamble();

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);

    mov(reg_kh, ptr[param + GET_OFF(kh_padding)]);
    mov(reg_src_prf, ptr[param + GET_OFF(src_prf)]);
    mov(reg_dst_prf, ptr[param + GET_OFF(dst_prf)]);
    mov(reg_ker_prf, ptr[param + GET_OFF(filt_prf)]);

    int l_overflow = utils::max(0, ((kw - 1) * dilate_w - jcp.l_pad) / stride_w);
    int r_overflow = utils::max(0, ((kw - 1) * dilate_w
                    - utils::max(0, jcp.r_pad)) / stride_w);
    int r_overflow1 = utils::max(0, ((kw - 1) * dilate_w
                    - utils::max(0, jcp.r_pad) - ur_w_tail) / stride_w);

    int n_oi = iw / ur_w;
    if (r_overflow1 > 0) n_oi--;

    if (ur_w == iw) {
        compute_loop(ur_w, l_overflow, r_overflow);
    } else if (n_oi == 0) {
        compute_loop(ur_w, l_overflow, r_overflow1);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);
        add(reg_src_prf, src_shift);
        add(reg_dst_prf, dst_shift);
        if (ur_w_tail != 0)
            compute_loop(ur_w_tail, 0, r_overflow);
    } else {
        xor_(reg_oi, reg_oi);
        if (l_overflow > 0) {
            compute_loop(ur_w, l_overflow, 0);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
            add(reg_src_prf, src_shift);
            add(reg_dst_prf, dst_shift);

            inc(reg_oi);
        }
        if ((l_overflow <= 0 && n_oi > 0)
            || (l_overflow > 0 && n_oi > 1)) {
            Label ow_loop_label;
            L(ow_loop_label); {
                compute_loop(ur_w, 0, 0);
                add(reg_src, src_shift);
                add(reg_dst, dst_shift);
                add(reg_src_prf, src_shift);
                add(reg_dst_prf, dst_shift);

                inc(reg_oi);
                cmp(reg_oi, n_oi);
                jl(ow_loop_label, T_NEAR);
            }
        }
        if (r_overflow1 > 0) {
            compute_loop(ur_w, 0, r_overflow1);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
            add(reg_src_prf, src_shift);
            add(reg_dst_prf, dst_shift);
        }
        if (ur_w_tail != 0) {
            compute_loop(ur_w_tail, 0, r_overflow);
        }
    }

    postamble();
}

SaberStatus jit_avx2_deconv_act_kernel::init_conf(jit_deconv_conf_t &jcp) {
    if (!mayiuse(avx2)) {
        LOG(ERROR) << "init a AVX2 kernel in a non-avx2 machine is not permitted";
        return SaberUnImplError;
    }

    unsigned int L1_cache_size = get_cache_size(1, true);

    const int simd_w = cpu_isa_traits<avx2>::vlen / sizeof(float);
    int ndims = jcp.ndims;

    jcp.r_pad = (jcp.ow - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
            - (jcp.iw + jcp.l_pad - 1);
    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);

    jcp.oc_block = simd_w;
    jcp.ic_block = simd_w;

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_w = jcp.stride_w;

    int regs = 14;
    if (jcp.iw <= regs) {
        jcp.ur_w = jcp.iw;
    } else {
        for (int ur_w = regs; ur_w > 0; --ur_w) {
            if (ur_w % jcp.stride_w == 0) {
                jcp.ur_w = ur_w;
                break;
            }
        }
    }

    int l_overflow = utils::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - jcp.l_pad) / jcp.stride_w);
    int r_overflow1 = utils::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - utils::max(0, jcp.r_pad) - jcp.iw % jcp.ur_w) / jcp.stride_w);
    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow1 > 0) n_oi--;

    if (mayiuse(avx2)) {
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);
    }
    else
        return SaberUnImplError;

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    bool large_code_size = (jcp.ur_w != jcp.ow)
         && ((l_overflow <= 0 && n_oi > 0) ||(l_overflow > 0 && n_oi > 1))
         && (r_overflow1 > 0) && (l_overflow > 0);
    if (large_code_size) {
        const int max_code_size = 12 * 1024;
        const int num_ops_per_reg = 3 + jcp.oc_block * jcp.kw;
        int mult = 1;
        if (l_overflow > 0) mult += 1;
        if (r_overflow1 > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs/2; --ur_w) {
            if ((ur_w / jcp.stride_w) * mult * num_ops_per_reg * 9.2
                    < max_code_size) {
                if (ur_w % jcp.stride_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
    }

    if (jcp.ver == ver_fma && mayiuse(avx2)) {
        int try_nb_ic_blocking = 2;
        unsigned int ker_inp_size = typesize * jcp.iw * jcp.ic_block
            * try_nb_ic_blocking * jcp.kh;
        unsigned int ker_out_size = typesize * jcp.ow * jcp.oc_block;
        unsigned int ker_wei_size = typesize * jcp.kh * jcp.kw * jcp.ic_block
            * jcp.oc_block * try_nb_ic_blocking;
        unsigned int ker_total_size = ker_inp_size + ker_out_size
            + ker_wei_size;
        if (!(jcp.kw == 1 || (jcp.kw == 5 && jcp.iw < 8)
            || (jcp.kw < 5 && ((jcp.iw <= 5 || (jcp.iw > 8 && jcp.iw <= 13))
            || ker_total_size > L1_cache_size )))
                || jcp.stride_h > 1) {
            jcp.kernel_kind = embd_bcast;
            jcp.ur_w = utils::min(jcp.iw, regs);
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (!(jcp.kw > 3 || (jcp.kw == 3 && ker_total_size < L1_cache_size
                && jcp.ow > 8)) && jcp.stride_h == 1)
                if (jcp.nb_ic % try_nb_ic_blocking == 0) {
                    jcp.nb_ic_blocking = try_nb_ic_blocking;
                    jcp.ur_w = 15 / (jcp.nb_ic_blocking + 1);
                    if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
                }
         } else {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_oc_blocking = 1;
            jcp.nb_ic_blocking = 4;
            if (jcp.nb_ic < jcp.nb_ic_blocking) jcp.nb_ic_blocking = jcp.nb_ic;
            if (jcp.nb_ic % jcp.nb_ic_blocking != 0)
                for (int i = jcp.nb_ic_blocking; i > 0; i--) {
                    if (jcp.nb_ic % i == 0) {
                        jcp.nb_ic_blocking = i;
                        break;
                    }
                }
            jcp.ur_w = 15 / (jcp.nb_ic_blocking + 1);
            if (jcp.iw < jcp.ur_w) jcp.ur_w = jcp.iw;
        }
    }
    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    if (l_overflow * jcp.stride_w > jcp.ur_w)
        return SaberUnImplError;
    int r_overflow_no_tail = utils::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - utils::max(0, jcp.r_pad) - jcp.ur_w_tail) / jcp.stride_w);
    if (r_overflow_no_tail * jcp.stride_w > jcp.ur_w)
        return SaberUnImplError;
    if ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
        return SaberUnImplError;

    jcp.nb_oc_L2 = jcp.nb_oc;

    return SaberSuccess;
}
} // namespace jit
} // namespace saber
} // namespace anakin
