#include <iostream>

#include "jit_avx512_conv_act_kernel.h"

#define GET_OFF(field) offsetof(jit_conv_call_t, field)
#define KNx_L2_EFFECTIVE_CAPACITY ((512 - 64) * 1024)

namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

static unsigned int L1_cache_size = get_cache_size(1, true);

static inline void pick_loop_order(jit_conv_conf_t &jcp) {
    // auto w = jcp.ow;
    // auto h = jcp.oh;
    switch (jcp.ver) {
    case ver_fma:
        jcp.loop_order = loop_cgn;
        break;
    default:
        assert(!"unsupported convolution version");
    }
}


void jit_conv_act_kernel::prepare_output(int ur_w) {
    for (int k = 0; k < jcp.nb_oc_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
            int aux_output_offset = get_output_offset(j, k);
            mic_prefetcht1(EVEX_compress_addr(reg_out_prf, aux_output_offset));
        }
    }
}


void jit_conv_act_kernel::store_output(int ur_w) {

    Label no_update_label;
    Label store_label;
    Label relu_label;

    mov(reg_channel, ptr[param1 + GET_OFF(channel)]);
    if (jcp.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    }

    if (!jcp.with_sum) {
        cmp(reg_channel, 0);
        je(no_update_label, T_NEAR);
    }

    for (int k = 0; k < jcp.nb_oc_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            int aux_output_offset = get_output_offset(j, k);
            vadd(zmm, reg_out, aux_output_offset);
        }
    }

    if (!jcp.with_sum) {
        jmp(relu_label, T_NEAR);
    } else {
        cmp(reg_channel, 0);
        jne(relu_label, T_NEAR);
    }


    L(no_update_label);
    if (jcp.with_bias) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            int bias_offset = typesize * k * jcp.oc_block;
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                vadd(zmm, reg_bias, bias_offset);
            }
            mic_prefetcht1(EVEX_compress_addr(reg_bias, bias_offset + 64));
        }
    }

    L(relu_label);
    if (jcp.with_relu) {
        vpxord(zmm_zero, zmm_zero, zmm_zero);
        if (jcp.relu_negative_slope == 0 || jcp.ver == ver_4vnni) {
            zmm_relu_ns = zmm_zero;
        } else {
            mov(imm_addr64, float2int(jcp.relu_negative_slope));
            vmovq(xmm_relu_ns, imm_addr64);
            vbroadcastss(zmm_relu_ns, xmm_relu_ns);
        }
        cmp(reg_channel, jcp.nb_ic - 1);
        jl(store_label, T_NEAR);
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            for (int j = 0; j < ur_w; j++) {
                Opmask kmask = Opmask(7);
                Zmm zmm = zmm_out(j, k);
                vcmp(kmask, zmm, zmm_zero, _cmp_lt_os);
                vmul(zmm, kmask, zmm, zmm_relu_ns);
            }
        }
    }

    L(store_label);
    for (int k = 0; k < jcp.nb_oc_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            int aux_output_offset
                   = jcp.typesize_out * (k * jcp.oh * jcp.ow + j) * jcp.oc_block;
            if (jcp.output_nhwc)
            {
                aux_output_offset = jcp.typesize_out * ((j) * jcp.oc + (k * jcp.oc_block));
            }

            if (jcp.dst_dt != AK_FLOAT) {
                vcvtps2dq(zmm_s32_tmp | T_rn_sae, zmm);
            }
            switch (jcp.dst_dt) {
                case AK_FLOAT:
                    vmovups(EVEX_compress_addr(reg_out, aux_output_offset), zmm);
                    break;
                case AK_INT32:
                    vmovups(EVEX_compress_addr(reg_out, aux_output_offset), zmm_s32_tmp);
                    break;
                case AK_INT8:
                    vpmovdb(EVEX_compress_addr(reg_out, aux_output_offset), zmm_s32_tmp);
                    break;
                case AK_UINT8:
                    vpmovusdb(EVEX_compress_addr(reg_out, aux_output_offset), zmm_s32_tmp);
                    break;
                default: assert(!"unsupported dst data type");
            }
            mic_prefetcht0(EVEX_compress_addr(reg_out_prf, aux_output_offset));
        }
    }
}


void jit_conv_act_kernel::compute_loop_fma_core(int ur_w,
            int pad_l, int pad_r) {
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    Label kh_label;
    Label skip_kh_loop;
    int shift_kernel_ptr = jcp.typesize_in * jcp.kw * jcp.oc_block
          * jcp.ic_block;
    int shift_input_ptr = jcp.typesize_in * jcp.iw
          * (!jcp.is_1stconv ? ic_block : 1);
    auto input_offset = [=](int oi, int ic, int ki) {
        return jcp.typesize_in * ((ki + oi * stride_w - pad_l) * ic_block + ic);
    };
    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);

    prepare_output(ur_w);

    mov(reg_kj, reg_kh);
    if (jcp.kh <= jcp.t_pad) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_ow_start(ki, pad_l);
            int jj_end = get_ow_end(ur_w, ki, pad_r);
            for (int ic = 0; ic < ic_block; ic++) {
                if (jcp.kernel_kind == expl_bcast) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        int aux_input_offset = input_offset(jj, ic, ki);
                        vbroadcastss(zmm_inp(jj, nb_oc_block),
                               ptr[aux_reg_inp + aux_input_offset]);
                    }
                }

                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int aux_kernel_offset = jcp.typesize_in
                         * (ii * jcp.nb_ic * jcp.kh * jcp.kw * ic_block
                         * oc_block + ki * ic_block * oc_block + ic * oc_block);
                    if (jj_end - jj_start > 0) {
                        vmovups(zmm_wei, EVEX_compress_addr(aux_reg_ker,
                                 aux_kernel_offset));
                    }
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        if (jcp.kernel_kind == expl_bcast) {
                            vfmadd231ps(zmm_out(jj, ii),
                                          zmm_inp(jj, nb_oc_block), zmm_wei);
                        } else {
                            vfmadd231ps(zmm_out(jj, ii), zmm_wei,
                                          EVEX_compress_addr(aux_reg_inp,
                                          input_offset(jj, ic, ki), true));
                        }
                    }
                }
            }
        }
        add(aux_reg_ker, shift_kernel_ptr);
        add(aux_reg_inp, shift_input_ptr);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);
    store_output(ur_w);
}


void jit_conv_act_kernel::compute_loop_fma(int ur_w, int pad_l, int pad_r) {
    bool prf_ker = true;
    bool prf_inp = true;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_oc_block = jcp.nb_oc_blocking;
    Label kh_label;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    int num_ker_loads = ic_block * nb_oc_block * kw;
    const int simd_w = 16;
    int num_ker_prfs = prf_ker ? num_ker_loads : 0;
    int num_inp_prfs = prf_inp ?
         ur_w * utils::min(kw, stride_w) + utils::max(0, kw - stride_w) :  0;
    if (jcp.is_1stconv && prf_inp) {
        num_inp_prfs = utils::div_up(num_inp_prfs, simd_w) * ic_block;
    }
    int num_prfs = num_ker_prfs + num_inp_prfs;
    int num_fmas = num_ker_loads * ur_w;
    int prf_inst_spacing
        = (prf_ker || prf_inp) ? utils::max(1, num_fmas / num_prfs) : 1;
    int prf_inst_trigger = (num_fmas % prf_inst_spacing) / 2;

    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);

    prepare_output(ur_w);

    mov(aux_reg_inp_prf, reg_inp_prf);
    mov(aux_reg_ker_prf, reg_ker_prf);
    mov(reg_kj, reg_kh);
    Label skip_kh_loop;
    if (jcp.kh <= jcp.t_pad) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }
    align(16);
    L(kh_label);
    {
        int step = 0;
        int ker_prfs = 0;
        for (int ki = 0; ki < kw; ki++) {
            for (int ic = 0; ic < ic_block; ic++) {
                int aux_kernel_offset = 0;
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        aux_kernel_offset = get_kernel_offset(ki, ic, 0, i);
                        vmovups(zmm_ker(i), EVEX_compress_addr(
                        aux_reg_ker, aux_kernel_offset));
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                            = (step + load_offset) % ker_pipeline_depth;
                    aux_kernel_offset = get_kernel_offset(ki,ic,0,load_offset);
                    vmovups(zmm_ker(ker_load_reg_idx),
                           EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                }

                bool ker_prf_inserted = false;
                Zmm zmm_kernel = zmm_ker(step % ker_pipeline_depth);
                int j_start = get_ow_start(ki, pad_l);
                int j_end = get_ow_end(ur_w, ki, pad_r);
                for (int j = j_start; j < j_end; j++) {
                    int aux_input_offset = get_input_offset(ki, ic, j, pad_l);
                    vfmadd231ps(zmm_out(j, 0), zmm_kernel,
                        EVEX_compress_addr(aux_reg_inp, aux_input_offset, true));

                    int fma_idx = step * ur_w + j;
                    int prf_slot_idx = fma_idx / prf_inst_spacing;
                    if (fma_idx % prf_inst_spacing == prf_inst_trigger) {
                        if (prf_ker && !ker_prf_inserted
                                    && ker_prfs < num_ker_prfs) {
                            int ker_prf_offset
                                        = jcp.typesize_in * ker_prfs * jcp.oc_block;
                            mic_prefetcht2(EVEX_compress_addr(
                                aux_reg_ker_prf, ker_prf_offset));
                            ker_prf_inserted = true;
                            ker_prfs++;
                        } else if (prf_inp) {
                            int inp_prf_idx = prf_slot_idx - ker_prfs;
                            if (inp_prf_idx < num_inp_prfs) {
                                int inp_prf_stride = utils::max(kw, stride_w);
                                int inp_prf_offset=0;
                                if (!jcp.is_1stconv) {
                                    inp_prf_offset
                                                 = ic_block * jcp.typesize_in
                                                 * ((inp_prf_idx / kw)
                                                 * inp_prf_stride
                                                 + (inp_prf_idx % kw));
                                } else {
                                    int ic_prf_stride = jcp.typesize_in*iw*ih;
                                    int iw_prf_stride = jcp.typesize_in*simd_w;
                                    inp_prf_offset = ((inp_prf_idx / ic_block)
                                                 * iw_prf_stride
                                                 + (inp_prf_idx % ic_block)
                                                 * ic_prf_stride);
                                }

                                mic_prefetcht0(EVEX_compress_addr(
                                             aux_reg_inp_prf, inp_prf_offset));
                            }
                        }
                    }
                }

                step++;
            }
        }
        add(aux_reg_ker, jcp.typesize_in * kw * oc_block * ic_block);
        if (prf_ker) {
            add(aux_reg_ker_prf, jcp.typesize_in * kw * oc_block * ic_block);
        }
        int inp_mul = !jcp.is_1stconv ? ic_block : 1;
        add(aux_reg_inp, jcp.typesize_in * iw * inp_mul);
        if (prf_inp) {
            add(aux_reg_inp_prf, jcp.typesize_in * iw * inp_mul);
        }

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    L(skip_kh_loop);
    store_output(ur_w);
}


void jit_conv_act_kernel::compute_loop(int ur_w, int pad_l, int pad_r) {

    if (jcp.ver == ver_fma){
        if (jcp.is_1stconv || mayiuse(avx512_mic)) {
            compute_loop_fma(ur_w, pad_l, pad_r);
        } else if (jcp.kernel_kind == embd_bcast && jcp.nb_oc_blocking == 1) {
            compute_loop_fma(ur_w, pad_l, pad_r);
        } else {
            compute_loop_fma_core(ur_w, pad_l, pad_r);
        }
    } else {
        assert(!"unknown convolution version");
    }
}


void jit_conv_act_kernel::generate() {
    int iw = jcp.iw;
    int ow = jcp.ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int inp_mult = !jcp.is_1stconv ? ic_block : 1;
    int inp_shift_pad = jcp.typesize_in * (ur_w * stride_w - l_pad) * inp_mult;
    int inp_shift = jcp.typesize_in * (ur_w * stride_w * inp_mult);
    int out_shift = jcp.typesize_out * (ur_w * oc_block);
    if (jcp.output_nhwc) {
        out_shift = jcp.typesize_out * (ur_w * jcp.oc);
    }
    preamble();

    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);
    mov(reg_ker_prf, ptr[param1 + GET_OFF(filt_prf)]);
    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    int r_pad = utils::max(0, (ow - 1) * stride_w + (kw - 1) - (iw + l_pad - 1));

    int n_oi = ow / ur_w;
    int r_pad1 = (ur_w * n_oi - 1) * stride_w + kw - 1 - (iw + l_pad - 1);
    if (r_pad1 > 0) n_oi--;


    if (ow == ur_w) {
        mov(reg_inp_prf, ptr[param1 + GET_OFF(src_prf)]);
        mov(reg_out_prf, ptr[param1 + GET_OFF(dst_prf)]);
        compute_loop(ur_w, l_pad, r_pad);
    } else {
        //TODO: potentially suboptimal
        mov(reg_inp_prf, reg_inp);
        mov(reg_out_prf, reg_out);

        if (n_oi == 0) {
            add(reg_inp_prf, inp_shift_pad);
            add(reg_out_prf, out_shift);
            compute_loop(ur_w, l_pad, r_pad1);
            add(reg_inp, inp_shift_pad);
            add(reg_out, out_shift);
            if (ur_w_tail != 0) {
                add(reg_inp_prf, inp_shift);
                add(reg_out_prf, out_shift);
                compute_loop(ur_w_tail, 0, r_pad);
            }
        } else {
            xor_(reg_oi, reg_oi);
            if (l_pad > 0) {
                add(reg_inp_prf, inp_shift_pad);
                add(reg_out_prf, out_shift);
                compute_loop(ur_w, l_pad, 0);
                add(reg_inp, inp_shift_pad);
                add(reg_out, out_shift);
                inc(reg_oi);
            }
            if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
               if (l_pad <= 0 && r_pad1 > 0)
                    n_oi--;
                    Label ow_loop_label;
                    L(ow_loop_label);
                    {
                        add(reg_inp_prf, inp_shift);
                        add(reg_out_prf, out_shift);
                        compute_loop(ur_w, 0, 0);
                        add(reg_inp, inp_shift);
                        add(reg_out, out_shift);
                        inc(reg_oi);
                        cmp(reg_oi, n_oi);
                        jl(ow_loop_label, T_NEAR);
                    }
            }
            if (r_pad1 > 0) {
                add(reg_inp_prf, inp_shift);
                add(reg_out_prf, out_shift);
                compute_loop(ur_w, 0, r_pad1);
                add(reg_inp, inp_shift);
                add(reg_out, out_shift);
            }
            if (ur_w_tail != 0) {
                add(reg_inp_prf, inp_shift);
                add(reg_out_prf, out_shift);
                compute_loop(ur_w_tail, 0, r_pad);
            }
        }
    }
    postamble();
}


SaberStatus jit_conv_act_kernel::init_conf(jit_conv_conf_t &jcp) {
    if (!mayiuse(avx512_common)) {
        LOG(ERROR) << "init a AVX512 kernel in non-avx512 machine is not permitted";
        return SaberUnImplError;
    }

    const int simd_w = cpu_isa_traits<avx512_common>::vlen / sizeof(float);
    const int regs = 28;

    jcp.ur_h = 1;
    jcp.oc_block = simd_w;
    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;

    if (mayiuse(avx512_common)) {
        jcp.ver = ver_fma;
        jcp.typesize_in = sizeof(float);
        jcp.typesize_out = sizeof(float);

        if (jcp.output_nhwc) {
            jcp.typesize_out = sizeof(char);
        }

        if (jcp.is_1stconv) {
            // TODO: fix & remove constraints below
            if (jcp.l_pad != 0 || jcp.r_pad != 0
                   || jcp.b_pad != 0 || jcp.t_pad != 0
                   || (jcp.kw < 7 && jcp.kh < 7))
                   jcp.ver = ver_fma;
        }
    }

    // set jcp.ur_w
    if (jcp.is_1stconv) {
        jcp.ur_w = utils::min(jcp.ow, regs);
    } else {
        for (int ur_w = regs; ur_w > 0; --ur_w) {
            if (jcp.ow % ur_w == 0) {
                jcp.ur_w = ur_w;
                break;
            }
        }
        if (jcp.ur_w == 1) {
            jcp.ur_w = utils::min(jcp.ow, regs);
        }
    }

    // TODO (Tanya): currenly applied to Segnet convolutions only.
    // Need to try for other topologies
    if (jcp.ow > 150 && jcp.ur_w < regs / 2) {
        jcp.ur_w = regs;
    }

    int n_oi = (jcp.ow / jcp.ur_w);
    int r_pad = (jcp.ur_w * n_oi - 1) * jcp.stride_w + jcp.kw - jcp.iw - jcp.l_pad;
    if (jcp.l_pad > 0 && r_pad > 0) {
        n_oi--;
    }

    bool large_code_size = jcp.ur_w != jcp.ow && jcp.l_pad > 0 && r_pad > 0 &&
                            ((jcp.l_pad <= 0 && n_oi > 0) || (jcp.l_pad > 0 && n_oi > 1));
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const int num_ops_per_reg = 6 + jcp.ic_block * jcp.kw;
        int mult = 1;
        if (jcp.l_pad > 0) {
            mult += 1;
        }
        if (r_pad > 0) {
            mult += 1;
        }
        for (int ur_w = jcp.ur_w; ur_w > regs / 2; --ur_w) {
            if (ur_w * mult * num_ops_per_reg * 9.0 < max_code_size) {
                jcp.ur_w = ur_w;
                break;
            }
        }
    }

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    if (jcp.ver == ver_fma && mayiuse(avx512_core)) {
        int try_nb_oc_blocking = 2;
        unsigned int ker_inp_size = jcp.typesize_in * (jcp.iw / jcp.stride_w)
              * jcp.ic_block * jcp.kh;
        unsigned int ker_out_size = typesize * jcp.ow * jcp.oc_block
              * try_nb_oc_blocking;
        unsigned int ker_wei_size = typesize * jcp.kh * jcp.kw * jcp.ic_block
              * jcp.oc_block * try_nb_oc_blocking;
        unsigned int ker_total_size = ker_inp_size + ker_out_size
              + ker_wei_size;

        if (jcp.mb == 1) {
            jcp.kernel_kind = embd_bcast;
        } else if (jcp.is_1stconv || jcp.kw > 3
              || ((jcp.kw == 3 && jcp.ow <= 28 && ker_total_size < L1_cache_size)
                 && !(jcp.kw == 3 && jcp.ow == 13 && jcp.ic >= 192)
                 && !(jcp.kw == 3 && jcp.ow == 28 && jcp.ic >= 512))
            ) {
            jcp.kernel_kind = embd_bcast;
            jcp.ur_w = utils::min(jcp.ow, regs);
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (ker_total_size < L1_cache_size && jcp.ow <= 8 && jcp.kh <= 3
                && jcp.kw <= 3) {
                if (jcp.nb_oc % try_nb_oc_blocking == 0 && !jcp.is_1stconv) {
                    jcp.nb_oc_blocking = try_nb_oc_blocking;
                    jcp.ur_w = 31 / (jcp.nb_oc_blocking + 1);
                    if (jcp.ow < jcp.ur_w)  jcp.ur_w = jcp.ow;
                }
            }
        } else {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_ic_blocking = 1;
            jcp.nb_oc_blocking = 4;
            if (jcp.nb_oc < jcp.nb_oc_blocking) {
                jcp.nb_oc_blocking = jcp.nb_oc;
            }
            if (jcp.nb_oc % jcp.nb_oc_blocking != 0) {
                for (int i = jcp.nb_oc_blocking; i > 0; i--) {
                    if (jcp.nb_oc % i == 0) {
                        jcp.nb_oc_blocking = i;
                        break;
                    }
                }
            }
            jcp.ur_w = 31 / (jcp.nb_oc_blocking + 1);
            if (jcp.ow < jcp.ur_w) {
                jcp.ur_w = jcp.ow;
            }
        }
    }

    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    bool args_ok = true &&
                   jcp.oc % simd_w == 0 &&
                   jcp.l_pad <= jcp.ur_w &&
                   utils::implication(!jcp.is_1stconv, jcp.ic % simd_w == 0) &&
                   jcp.dilate_h == 0 && jcp.dilate_w == 0;
    if (!args_ok) {
        LOG(ERROR) << "arguments check failed";
        return SaberUnImplError;
    }

    int r_pad_no_tail = utils::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w +
                                      jcp.kw - jcp.iw - jcp.l_pad);
    if (r_pad_no_tail > jcp.ur_w) {
        LOG(ERROR) << "tail should not be greater than ur_w";
        return SaberUnImplError;
    }
    if (jcp.output_nhwc && jcp.with_sum) {
        LOG(ERROR) << "no support output nhwc and with_sum simultaneously";
        return SaberUnImplError;
    }
    pick_loop_order(jcp);
    jcp.nb_ic_L2 = jcp.nb_ic;

    return SaberSuccess;
}


} // namespace jit
} // namespace saber
} // namespace anakin
