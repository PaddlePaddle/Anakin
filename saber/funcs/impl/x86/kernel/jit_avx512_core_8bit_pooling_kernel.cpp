#include <iostream>
#include <stddef.h>

#include "jit_avx512_core_8bit_pooling_kernel.h"

namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

void jit_avx512_core_8bit_pooling_kernel::load_src(int jj,
                                                   int ll,
                                                   int c_tail) {
    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case Pooling_max: {
            auto offset = jj * c_block * sizeof_src_dt();
            if (jj == ur_c - 1 && c_tail) {
                if (jpp.src_dt == AK_INT32) {
                    vmovups(vreg_src(jj) | mask(0),
                            ptr[aux_reg_src_w + offset]);
                } else {
                    vmovdqu8(vreg_src(jj) | mask(0),
                             ptr[aux_reg_src_w + offset]);
                }
            } else {
                vmovups(vreg_src(jj), ptr[aux_reg_src_w + offset]);
            }
            break;
        }
        case Pooling_average_include_padding:
        case Pooling_average_exclude_padding: {
            auto offset = (ll * (c_block / 4) + jj * c_block) * sizeof_src_dt();
            if (jj == jpp.ur_c - 1 && c_tail) {
                if (jpp.tail[ll]) {
                    switch (jpp.src_dt) {
                        case AK_INT32:
                            vmovups(vreg_src_s32(jj, ll) | mask(ll),
                                    ptr[aux_reg_src_w + offset]);
                            break;
                        case AK_INT8:
                            vpmovsxbd(vreg_src_s32(jj, ll) | mask(ll),
                                      ptr[aux_reg_src_w + offset]);
                            break;
                        case AK_UINT8:
                            vpmovzxbd(vreg_src_s32(jj, ll) | mask(ll),
                                      ptr[aux_reg_src_w + offset]);
                            break;
                        // case AK_FLOAT:
                        //    vmovups(vreg_src_s32(jj, ll) | mask(ll),
                        //            ptr[aux_reg_src_w + offset]);
                        //    break;
                        default:
                            assert(!"unsupported src data type");
                    }
                }
            } else {
                switch (jpp.src_dt) {
                    case AK_INT32:
                        vmovups(vreg_src_s32(jj, ll),
                                ptr[aux_reg_src_w + offset]);
                        break;
                    case AK_INT8:
                        vpmovsxbd(vreg_src_s32(jj, ll),
                                  ptr[aux_reg_src_w + offset]);
                        break;
                    case AK_UINT8:
                        vpmovzxbd(vreg_src_s32(jj, ll),
                                ptr[aux_reg_src_w + offset]);
                        break;
                    // case AK_FLOAT:
                    //   vmovups(vreg_src_s32(jj, ll),
                    //           ptr[aux_reg_src_w + offset]);
                    //   break;
                    default:
                        assert(!"unsupported src data type");
                }
            }
            break;
        }
        default:
            assert(!"unsupported algorithm");
    }
}

void jit_avx512_core_8bit_pooling_kernel::store_dst(int jj,
                                               int ll,
                                               int c_tail) {
    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case Pooling_max: {
            auto offset = jj * c_block * sizeof_dst_dt();
            if (jj == ur_c - 1 && c_tail) {
                if (jpp.dst_dt == AK_INT32) {
                    vmovups(ptr[reg_ptr_dst + offset],
                            vreg_dst(jj) | mask(0));
                } else{
                    vmovdqu8(ptr[reg_ptr_dst + offset],
                             vreg_dst(jj) | mask(0));
                }
            } else {
                vmovups(ptr[reg_ptr_dst + offset], vreg_dst(jj));
            }
            break;
        }
        case Pooling_average_include_padding:
        case Pooling_average_exclude_padding: {
            auto offset = (ll * (c_block / 4) + jj * c_block) * sizeof_dst_dt();
            if (jj == ur_c - 1 && c_tail) {
                if (jpp.tail[ll]) {
                    switch (jpp.dst_dt) {
                        case AK_INT32:
                            vmovups(ptr[reg_ptr_dst + offset],
                                    vreg_dst_s32(jj, ll) | mask(ll));
                            break;
                        case AK_INT8:
                            vpmovdb(ptr[reg_ptr_dst + offset],
                                    vreg_dst_s32(jj, ll) | mask(ll));
                            break;
                        case AK_UINT8:
                            vpmovusdb(ptr[reg_ptr_dst + offset],
                                      vreg_dst_s32(jj, ll) | mask(ll));
                            break;
                        case AK_FLOAT:
                            vmovups(ptr[reg_ptr_dst + offset],
                                    vreg_dst_f32(jj, ll) | mask(ll));
                            break;
                        default:
                            assert(!"unsupported dst data_type");
                    }
                }
            } else {
                switch (jpp.dst_dt) {
                    case AK_INT32:
                        vmovups(ptr[reg_ptr_dst + offset],
                                vreg_dst_s32(jj, ll));
                        break;
                    case AK_INT8:
                        vpmovdb(ptr[reg_ptr_dst + offset],
                                vreg_dst_s32(jj, ll));
                        break;
                    case AK_UINT8:
                        vpmovusdb(ptr[reg_ptr_dst + offset],
                                  vreg_dst_s32(jj, ll));
                        break;
                    case AK_FLOAT:
                        vmovups(ptr[reg_ptr_dst + offset],
                                vreg_dst_f32(jj, ll));
                        break;
                    default:
                        assert(!"unsuppotred dst data_type");
                }
            }
            break;
        }
        default:
            assert(!"unsupported pooling algorithm");
    }
}

void jit_avx512_core_8bit_pooling_kernel::compute_max_step(int ur_c,
                                                           int c_tail) {
    Label l_kw;
    Label l_kh;
    int iw = jpp.iw;
    int c = jpp.c;

    for (int jj = 0; jj < ur_c; jj++) {
        vmovups(vreg_dst(jj), vreg_tmp);
    }

    mov(aux_reg_src_h, reg_ptr_src);

    xor_(kj, kj);
    L(l_kh); {
        mov(aux_reg_src_w, aux_reg_src_h);
        xor_(ki, ki);
        L(l_kw); {
            for (int jj = 0; jj < ur_c; jj++) {
                load_src(jj, 0, c_tail);
                if (jpp.src_dt == AK_INT32) {
                    vpcmpd(k_cmp_mask, vreg_dst(jj), vreg_src(jj), _cmp_lt_os);
                    vpblendmd(vreg_dst(jj) | k_cmp_mask, vreg_dst(jj),
                              vreg_src(jj));
                } else {
                    if (jpp.src_dt == AK_INT8) {
                        vpcmpb(k_cmp_mask, vreg_dst(jj), vreg_src(jj),
                                _cmp_lt_os);
                    } else {
                        vpcmpub(k_cmp_mask, vreg_dst(jj), vreg_src(jj),
                                _cmp_lt_os);
                    }
                    vpblendmb(vreg_dst(jj) | k_cmp_mask, vreg_dst(jj),
                              vreg_src(jj));
                }
            }
            add(aux_reg_src_w, c * sizeof_src_dt());
            inc(ki);
            cmp(ki, reg_kw);
            jl(l_kw, T_NEAR);
        }
        add(aux_reg_src_h, iw * c * sizeof_src_dt());
        inc(kj);
        cmp(kj, reg_kh);
        jl(l_kh, T_NEAR);
    }

    for (int jj = 0; jj < ur_c; jj++) {
        store_dst(jj, 0, c_tail);
    }
}

void jit_avx512_core_8bit_pooling_kernel::compute_avg_step(int ur_c,
                                                           int c_tail) {
    Label l_kw;
    Label l_kh;
    int iw = jpp.iw;
    int c = jpp.c;
    int num_ll = 0;

    switch (jpp.src_dt) {
        case AK_INT32:
        case AK_FLOAT:
            num_ll = 1;
            break;
        case AK_INT8:
        case AK_UINT8:
            num_ll = 4;
            break;
        default:
            assert(!"unsuppotred src data_type");
    }

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < 4; ll++) {
            uni_vpxor(vreg_src_s32(jj, ll),
                      vreg_src_s32(jj, ll), vreg_src_s32(jj, ll));
            uni_vpxor(vreg_dst_s32(jj, ll),
                      vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll));
            uni_vpxor(vreg_dst_f32(jj, ll),
                      vreg_dst_f32(jj, ll), vreg_dst_f32(jj, ll));
        }
    }

    mov(aux_reg_src_h, reg_ptr_src);

    xor_(kj, kj);
    L(l_kh); {
        mov(aux_reg_src_w, aux_reg_src_h);
        xor_(ki, ki);
        L(l_kw); {
            for (int jj = 0; jj < ur_c; jj++) {
                for (int ll = 0; ll < num_ll; ll++) {
                    load_src(jj, ll, c_tail);
                    vpaddd(vreg_dst_s32(jj, ll),
                           vreg_dst_s32(jj, ll),
                           vreg_src_s32(jj, ll));
                }
            }
            add(aux_reg_src_w, c * sizeof_src_dt());
            inc(ki);
            cmp(ki, reg_kw);
            jl(l_kw, T_NEAR);
        }
        add(aux_reg_src_h, iw * c * sizeof_src_dt());
        inc(kj);
        cmp(kj, reg_kh);
        jl(l_kh, T_NEAR);
    }

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < num_ll; ll++) {
            if (jpp.src_dt != AK_FLOAT) {
                vcvtdq2ps(vreg_dst_f32(jj, ll), vreg_dst_s32(jj, ll));
            }
            vfmadd132ps(vreg_dst_f32(jj, ll), vreg_zeros, vreg_tmp);
            if (jpp.dst_dt == AK_UINT8 || jpp.dst_dt == AK_INT8) {
                vcvtps2dq(vreg_dst_s32(jj, ll) | T_rn_sae, vreg_dst_f32(jj, ll));
            }
            store_dst(jj, ll, c_tail);
        }
    }
}

void jit_avx512_core_8bit_pooling_kernel::compute_step(int ur_c,
                                                       int c_tail) {
    switch (jpp.alg) {
        case Pooling_max:
            compute_max_step(ur_c, c_tail);
            break;
        case Pooling_average_include_padding:
        case Pooling_average_exclude_padding:
            compute_avg_step(ur_c, c_tail);
            break;
        default: assert(!"unsupported pooling algorithm");
    }
}

void jit_avx512_core_8bit_pooling_kernel::compute_c_block() {
    Label l_main_loop;

    int nb_c = jpp.nb_c;
    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;
    int ur_c_tail = jpp.ur_c_tail;
    int c_steps = nb_c / ur_c;
    int c_tail = jpp.c_tail;

    xor_(c_iter, c_iter);
    if (c_steps > 0) {
        L(l_main_loop); {
            compute_step(ur_c, 0);
            add(reg_ptr_src, ur_c * c_block * sizeof_src_dt());
            add(reg_ptr_dst, ur_c * c_block * sizeof_dst_dt());
            inc(c_iter);
            cmp(c_iter, c_steps);
            jl(l_main_loop, T_NEAR);
        }
    }

    if (ur_c_tail != 0) {
        compute_step(ur_c_tail, c_tail);
    }
}

void jit_avx512_core_8bit_pooling_kernel::init_mask() {
    for (int i = 0; i < 4; i++) {
        mov(reg_mask, jpp.tail[i]);
        kmovq(mask(i), reg_mask);
    }
}

void jit_avx512_core_8bit_pooling_kernel::init_tmp_reg() {
    switch (jpp.alg) {
        case Pooling_average_include_padding:
        case Pooling_average_exclude_padding:
            mov(reg_tmp, ptr[abi_param1 + offsetof(jit_pool_call_nhwc_t, idivider)]);
            movq(xmm_tmp, reg_tmp);
            vpbroadcastd(vreg_tmp, xmm_tmp);
            break;
        case Pooling_max:
            switch (jpp.src_dt) {
                case AK_INT32:
                    mov(reg_tmp, std::numeric_limits<int32_t>::lowest());
                    break;
                case AK_INT8:
                    mov(reg_tmp, std::numeric_limits<int8_t>::lowest());
                    break;
                case AK_UINT8:
                    mov(reg_tmp, std::numeric_limits<uint8_t>::lowest());
                    break;
                default: assert(!"unsupported src data_type");
            }

            movq(xmm_tmp, reg_tmp);
            if (jpp.src_dt == AK_INT32)
                vpbroadcastd(vreg_tmp, xmm_tmp);
            else
                vpbroadcastb(vreg_tmp, xmm_tmp);
            break;
        default: assert(!"unsupported pooling algorithm");
    }

}

void jit_avx512_core_8bit_pooling_kernel::generate() {
    preamble();

    #define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(jit_pool_call_nhwc_t, field)])

    if (jpp.src_dt == AK_FLOAT) {
        READ_PARAM(reg_ptr_src, src_fp32);
    }
    else {
        READ_PARAM(reg_ptr_src, src_i8);
    }

    if (jpp.dst_dt == AK_FLOAT) {
        READ_PARAM(reg_ptr_dst, dst_fp32);
    }
    else {
        READ_PARAM(reg_ptr_dst, dst_i8);
    }

    READ_PARAM(reg_kw, kw_range);
    READ_PARAM(reg_kh, kh_range);

    #undef READ_PARAM

    init_tmp_reg();
    init_mask();

    uni_vpxor(vreg_zeros, vreg_zeros, vreg_zeros);

    compute_c_block();

    postamble();
}

SaberStatus jit_avx512_core_8bit_pooling_kernel::init_conf(jit_pool_conf_t &jpp) {
    SaberStatus ret = SaberUnImplError;

    bool ok = true &&
              mayiuse(avx512_common) &&
              jpp.src_fmt == Layout_NHWC &&
              jpp.kh <= jpp.ih &&
              jpp.kw <= jpp.iw;
    if (!ok) {
        return SaberUnImplError;
    }

    jpp.simple_alg = false;
    if (jpp.alg == Pooling_max) {
        jpp.ur_w = 16;
    } else {
        jpp.ur_w = 24;
    }

    if ((jpp.alg == Pooling_max) && (jpp.dst_dt == AK_FLOAT)) {
        LOG(FATAL) << "dst format (AK_FLOAT) and pooling type (Pooling_max): NOT supported!" ;
        return SaberUnImplError;
    }

    jpp.c_block = 64 / (jpp.src_dt == AK_INT32 ? 4 : 1);
    jpp.c_tail = jpp.c % jpp.c_block;
    jpp.nb_c = jpp.c / jpp.c_block;
    jpp.ur_c = 1;
    jpp.ur_c_tail = jpp.nb_c - (jpp.nb_c / jpp.ur_c)*jpp.ur_c + (jpp.c_tail != 0);

    size_t tail_mask = (1ULL << jpp.c_tail) - 1;

    switch (jpp.alg) {
        case Pooling_max:
            jpp.tail[0] = tail_mask;
            jpp.tail[1] = 0;
            jpp.tail[2] = 0;
            jpp.tail[3] = 0;
            break;
        case Pooling_average_include_padding:
        case Pooling_average_exclude_padding:
            jpp.tail[0] = tail_mask & 0xffff;
            for (size_t i = 1, m = tail_mask; i < 4; i++) {
                m = m >> 16;
                jpp.tail[i] = m & 0xffff;
            }
            break;
        default: return SaberUnImplError;
    }

    if (jpp.ow < jpp.ur_w) {
        jpp.ur_w = jpp.ow;
    }
    if (jpp.l_pad > jpp.ur_w) {
        return SaberUnImplError;
    }
    jpp.ur_w_tail = jpp.ow % jpp.ur_w;
    return SaberSuccess;
}

} // namespace jit
} // namespace saber
} // namespace anakin
