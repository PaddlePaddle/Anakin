#include "jit_uni_pool_kernel_f32.h"

namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_pool_call_t, field)

template <cpu_isa_t isa>
bool jit_pool_kernel_f32<isa>::init_conf(jit_pool_conf_t &jpp) {
    bool layout_c16 = (jpp.src_fmt == Layout_NCHW_C16||jpp.src_fmt==Layout_NCHW_C16R) && mayiuse(avx512_common);
    bool layout_c8 = (jpp.src_fmt == Layout_NCHW_C8||jpp.src_fmt ==Layout_NCHW_C8R) && mayiuse(avx2);
    bool ok = true && (layout_c16 || layout_c8);
    if (!ok) {
        return false;
    }

    int simd_w;
    if (layout_c16)
        simd_w = 16;
    else if (layout_c8)
        simd_w = 8;
    else
        return false;

    jpp.simple_alg = false;
    jpp.c_block = simd_w;
    jpp.nb_c = jpp.c / jpp.c_block;
    if (jpp.alg == Pooling_max) {
        jpp.ur_w = 16;
        if (layout_c8)
            jpp.ur_w = 4;
    } else {
        jpp.ur_w = 24;
        if (layout_c8)
            jpp.ur_w = 12;
    }

    if (jpp.ow < jpp.ur_w) {
        jpp.ur_w = jpp.ow;
    }
    if (jpp.l_pad > jpp.ur_w) {
        return false;
    }
    jpp.ur_w_tail = jpp.ow % jpp.ur_w;

    return true;
}

template <cpu_isa_t isa>
inline void jit_pool_kernel_f32<isa>::maybe_recalculate_divisor(int jj,
        int ur_w, int pad_l, int pad_r) {
    if (jpp.alg == Pooling_average_exclude_padding) {
        int kw = jpp.kw;
        int stride_w = jpp.stride_w;

        int non_zero_kw = kw;
        non_zero_kw -= std::max(0, pad_l - jj*stride_w);
        non_zero_kw -= std::max(0, pad_r - (ur_w - 1 - jj)*stride_w);

        if (non_zero_kw != prev_kw) {
            mov(tmp_gpr, float2int((float)non_zero_kw));
            movq(xmm_tmp, tmp_gpr);
            uni_vbroadcastss(vmm_tmp, xmm_tmp);
            uni_vmulps(vmm_tmp, vmm_tmp, vmm_ker_area_h);
            prev_kw = non_zero_kw;
        }
    }
}

template <cpu_isa_t isa>
inline void jit_pool_kernel_f32<isa>::avg_step(int ur_w, int pad_l,
        int pad_r, const char* kh_label) {

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    Label kd_label;

    for (int jj = 0; jj < ur_w; jj++) {
        uni_vpxor(vreg(jj), vreg(jj), vreg(jj));
    }

    mov(aux_reg_input, reg_input);

    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = std::max(0, pad_l - ki);
            int jj_end = ur_w - utils::div_up(std::max(0, ki + pad_r - (kw - 1)), stride_w);
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki + jj * stride_w - pad_l) * c_block;
                if (aux_input_offset > iw * c_block) {
                    continue;
                }
                int input_offset = sizeof(float)*aux_input_offset;
                uni_vaddps(vreg(jj), vreg(jj), ptr[aux_reg_input + input_offset]);
            }
        }
        add(aux_reg_input,  sizeof(float) * iw * c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }

    for (int jj = 0; jj < ur_w; jj++) {
        maybe_recalculate_divisor(jj, ur_w, pad_l, pad_r);
        uni_vdivps(vreg(jj), vreg(jj), vmm_tmp);
        uni_vmovups(vmmword[reg_output + sizeof(float)*jj*c_block], vreg(jj));
    }
}

template <cpu_isa_t isa>
inline void jit_pool_kernel_f32<isa>::max_step_fwd(int ur_w, int pad_l,
        int pad_r, const char *kh_label) {
    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    Label kd_label;

    mov(tmp_gpr, float2int(std::numeric_limits<float>::lowest()));
    movq(xmm_tmp, tmp_gpr);
    uni_vbroadcastss(vmm_tmp, xmm_tmp);

    for (int jj = 0; jj < ur_w; jj++) {
        uni_vmovups(vreg(jj), vmm_tmp);
    }

    mov(aux_reg_input, reg_input);

    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = std::max(0, pad_l - ki);
            int jj_end = ur_w
                - utils::div_up(std::max(0, ki + pad_r - (kw-1)), stride_w);
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)* c_block;
                if (aux_input_offset > iw * c_block)
                    continue;
                int input_offset = sizeof(float)*aux_input_offset;
                uni_vmovups(vreg(ur_w+jj), ptr[aux_reg_input + input_offset]);
                if (isa == sse42) {
                    movups(vmm_mask, vreg(jj));
                    cmpps(vmm_mask, vreg(ur_w+jj), _cmp_lt_os);
                    blendvps(vreg(jj), vreg(ur_w+jj));
                } else if (isa == avx2) {
                    vcmpps(vreg(3*ur_w+jj), vreg(jj), vreg(ur_w+jj),
                           _cmp_lt_os);
                    vblendvps(vreg(jj), vreg(jj), vreg(ur_w+jj),
                              vreg(3*ur_w+jj));
                } else {
                    vcmpps(k_store_mask, vreg(jj), vreg(ur_w+jj), _cmp_lt_os);
                    vblendmps(vreg(jj) | k_store_mask, vreg(jj), vreg(ur_w+jj));
                }
            }
        }
        add(aux_reg_input,  sizeof(float) * iw * c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }

    for (int jj = 0; jj < ur_w; jj++) {
        uni_vmovups(vmmword[reg_output + sizeof(float)*jj*c_block], vreg(jj));
    }
}

template <cpu_isa_t isa>
inline void jit_pool_kernel_f32<isa>::max_step_bwd(int ur_w, int pad_l,
        int pad_r, const char *kh_label) {
    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    Label kd_label;

    for (int jj = 0; jj < ur_w; jj++) {
        uni_vmovups(vreg(jj), ptr[reg_output + sizeof(float)*jj*c_block]);

        const size_t step_index = jj * c_block * datatype_size(jpp.ind_dt);
        if (jpp.ind_dt == AK_UINT8) {
            if (isa == sse42) {
                movd(xreg(ur_w+jj), ptr[reg_index + step_index]);
                pmovzxbd(vreg(ur_w+jj), xreg(ur_w+jj));
            } else {
                if (isa == avx2)
                    movq(xreg(ur_w+jj), ptr[reg_index + step_index]);
                else
                    vmovups(vreg(ur_w+jj) | k_index_mask,
                            ptr[reg_index + step_index]);
                vpmovzxbd(vreg(ur_w+jj), xreg(ur_w+jj));
            }
        } else {
            uni_vmovups(vreg(ur_w+jj), ptr[reg_index + step_index]);
        }
    }
    movq(xmm_tmp, reg_k_shift);
    uni_vpbroadcastd(vmm_k_offset, xmm_tmp);

    mov(aux_reg_input, reg_input);

    xor_(kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = std::max(0, pad_l - ki);
            int jj_end = ur_w - utils::div_up(std::max(0, ki + pad_r - (kw - 1)), stride_w);
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)* c_block;
                if (aux_input_offset > iw * c_block) {
                    continue;
                }
                int input_offset = sizeof(float)*aux_input_offset;
                uni_vmovups(vreg(2*ur_w+jj), ptr[aux_reg_input + input_offset]);
                if (isa == sse42) {
                    mov(dst_ptr, aux_reg_input);
                    add(dst_ptr, input_offset);

                    movups(vreg(3*ur_w+jj), vreg(ur_w+jj));
                    pcmpeqd(vreg(3*ur_w+jj), vmm_k_offset);
                    addps(vreg(2*ur_w+jj), vreg(jj));
                    maskmovdqu(vreg(2*ur_w+jj), vreg(3*ur_w+jj));
                } else if (isa == avx2) {
                    vpcmpeqd(vreg(3*ur_w+jj), vreg(ur_w+jj), vmm_k_offset);
                    vaddps(vreg(2*ur_w+jj), vreg(2*ur_w+jj), vreg(jj));
                    vmaskmovps(vmmword[aux_reg_input + input_offset],
                            vreg(3*ur_w+jj), vreg(2*ur_w+jj));
                } else {
                    vpcmpeqd(k_store_mask, vreg(ur_w+jj), vmm_k_offset);
                    vblendmps(vmm_tmp | k_store_mask | T_z, vreg(jj), vreg(jj));
                    vaddps(vreg(2*ur_w+jj), vreg(2*ur_w+jj), vmm_tmp);
                    vmovups(vmmword[aux_reg_input +
                        sizeof(float)*aux_input_offset], vreg(2*ur_w+jj));
                }
            }
            uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_one);
        }
        add(aux_reg_input,  sizeof(float) * iw * c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }
}

template <cpu_isa_t isa>
void jit_pool_kernel_f32<isa>::maybe_zero_diff_src() {
    assert(jpp.c_block * sizeof(float) % cpu_isa_traits<isa>::vlen == 0);
    Label l_skip, l_zero;

    auto reg_oh = tmp_gpr;
    mov(reg_oh, ptr[this->param1 + GET_OFF(oh)]);
    cmp(reg_oh, 0);
    jz(l_skip, T_NEAR);

    auto vzero = vmm_tmp;
    uni_vpxor(vzero, vzero, vzero);

    auto reg_off = tmp_gpr;
    xor_(reg_off, reg_off);

    L(l_zero);
    {
        const int dim = jpp.iw * jpp.c_block * sizeof(float);
        for (int i = 0; i < dim; i += cpu_isa_traits<isa>::vlen)
            uni_vmovups(ptr[reg_input + reg_off + i], vzero);
        add(reg_off, dim);
        cmp(reg_off, jpp.ih * dim);
        jl(l_zero, T_NEAR);
    }

    L(l_skip);
}

template <cpu_isa_t isa>
void jit_pool_kernel_f32<isa>::generate() {
    this->preamble();

    int ow = jpp.ow;
    int iw = jpp.iw;
    int kw = jpp.kw;
    int kh = jpp.kh;
    int ur_w = jpp.ur_w;
    int c_block = jpp.c_block;
    int stride_w = jpp.stride_w;
    int l_pad = jpp.l_pad;
    int ur_w_tail = jpp.ur_w_tail;

    int n_oi = ow / ur_w;

    prev_kw = 0;

    int vlen = cpu_isa_traits<isa>::vlen;

    mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_kh, ptr[this->param1 + GET_OFF(kh_padding)]);
    mov(reg_k_shift, ptr[this->param1 + GET_OFF(kh_padding_shift)]);
    mov(reg_ker_area_h, ptr[this->param1 + GET_OFF(ker_area_h)]);

    int r_pad  = std::max(0, ((ow - 1) * stride_w) + kw - 1 - (iw + l_pad - 1));
    int r_pad1 = (ur_w * n_oi - 1) * stride_w + kw - 1 - (iw + l_pad - 1);
    if (r_pad1 > 0) {
        n_oi--;
    }

    if (jpp.alg == Pooling_average_exclude_padding) {
        movq(xmm_ker_area_h, reg_ker_area_h);
        uni_vpbroadcastd(vmm_ker_area_h, xmm_ker_area_h);
    }

    if (jpp.alg == Pooling_average_include_padding) {
        mov(tmp_gpr, float2int((float)(kw * kh * jpp.kd)));
        movq(xmm_tmp, tmp_gpr);
        uni_vpbroadcastd(vmm_tmp, xmm_tmp);
    }

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0) {
            step(ur_w, l_pad, r_pad1, ".kh_loop_oimain_padwl");
        } else  {
            step(ur_w, l_pad, 0, ".kh_loop_oimain_padwl");
        }

        if (isa == sse42) {
            if (n_oi < 0 && r_pad1 > 0) {
                step_high_half(ur_w, l_pad, r_pad1,
                    ".kh_loop_oimain_padwl_high_half");
            } else  {
                step_high_half(ur_w, l_pad, 0,
                    ".kh_loop_oimain_padwl_high_half");
            }
        }

        if (isa == sse42) {
            add(reg_input, sizeof(float)*(ur_w*stride_w-l_pad)*c_block - vlen);
            add(reg_output, sizeof(float)*ur_w*c_block - vlen);
        } else {
            add(reg_input, sizeof(float)*(ur_w*stride_w - l_pad)*c_block);
            add(reg_output, sizeof(float)*ur_w*c_block);
        }
    }

    xor_(oi_iter, oi_iter);
    if (n_oi > 0) {
        L(".ow_loop"); {
            step(ur_w, 0, 0, ".kh_loop_oimain");

            if (isa == sse42) {
                step_high_half(ur_w, 0, 0, ".kh_loop_oimain_high_half");
            }

            if (isa == sse42) {
                add(reg_input, sizeof(float)*ur_w*stride_w*c_block - vlen);
                add(reg_output, sizeof(float)*ur_w*c_block - vlen);
            } else {
                add(reg_input, sizeof(float)*ur_w*stride_w*c_block);
                add(reg_output, sizeof(float)*ur_w*c_block);
            }

            inc(oi_iter);
            cmp(oi_iter, n_oi); jl(".ow_loop", T_NEAR);
        } L(".ow_loop_end");
    }

    if (r_pad1 > 0 && n_oi >= 0) {
        step(ur_w, 0, r_pad1, ".kh_loop_oimain_padwr");

        if (isa == sse42) {
            step_high_half(ur_w, 0, r_pad1, ".kh_loop_oimain_padwr_high_half");
        }

        if (isa == sse42) {
            add(reg_input, sizeof(float)*ur_w*stride_w*c_block - vlen);
            add(reg_output, sizeof(float)*ur_w*c_block - vlen);
        } else {
            add(reg_input, sizeof(float)*ur_w*stride_w*c_block);
            add(reg_output, sizeof(float)*ur_w*c_block);
        }
    }

    if (ur_w_tail != 0) {
        step(ur_w_tail, 0, r_pad, ".kh_loop_oitail");

        if (isa == sse42) {
            step_high_half(ur_w_tail, 0, r_pad, ".kh_loop_oitail_high_half");
        }
    }

    this->postamble();
}

template struct jit_pool_kernel_f32<avx512_common>;
template struct jit_pool_kernel_f32<avx2>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
