#include "jit_avx512_conv1x1_kernel.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"

#include "saber/funcs/impl/x86/x86_utils.h"
#include "jit_uni_1x1_conv_utils.h"

using namespace anakin::saber::utils;

namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;
#define GET_OFF(field) offsetof(jit_1x1_conv_call_t, field)

void jit_avx512_common_1x1_conv_kernel::bcast_loop(int load_loop_blk) {
    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_bcast_data, reg_bcast_data);

    mov(aux_reg_output_data, reg_output_data);
    mov(bcast_loop_iter, EVEX_compress_addr(rsp, bcast_loop_work_offt));

    Label bcast_loop;
    Label bcast_loop_tail;

    cmp(bcast_loop_iter, jcp.ur);
    jl(bcast_loop_tail, T_NEAR);

    L(bcast_loop); {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            reduce_loop(load_loop_blk, jcp.ur, i, false);
            if (i < num_substeps - 1) {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_substep);
            } else {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_step
                                         - (num_substeps - 1) * jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_step
                                         - (num_substeps - 1) * jcp.bcast_loop_output_substep);
            }
        }
        sub(bcast_loop_iter, jcp.bcast_block);
        cmp(bcast_loop_iter, jcp.bcast_block);
        jge(bcast_loop, T_NEAR);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        cmp(bcast_loop_iter, 0);
        jz(bcast_loop_tail_out, T_NEAR);
        reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
        L(bcast_loop_tail_out);
    }
}


void jit_avx512_common_1x1_conv_kernel::reduce_loop(int load_loop_blk,
                                                    int ur, int substep, bool wraparound) {
    auto vreg_load = [=](int i_load, int i_fma) {
        return Zmm(rnd_up(ur * load_loop_blk, jcp.fma_step)
                   + jcp.fma_step * i_load + i_fma);
    };

    auto vreg_accum = [=](int i_load, int i_ur) {
        return Zmm(i_ur * load_loop_blk + i_load);
    };

    auto bias_ptr = [=](int i_load) {
        return EVEX_compress_addr(reg_bias_data,
                                  jcp.typesize_out * jcp.oc_block * i_load);
    };

    auto bcast_ptr = [=](int i_reduce, int i_ur, bool bcast) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        int offt;

        assert(jcp.reduce_loop_unroll == jcp.reduce_block);
        offt = (i_reduce == jcp.reduce_loop_unroll)
               ? (jcp.bcast_dim + i_ur) * jcp.reduce_loop_unroll
               : i_ur * jcp.reduce_loop_unroll + i_reduce;

        return EVEX_compress_addr(aux_reg_bcast_data, jcp.typesize_in * offt,
                                  bcast);
    };

    auto load_ptr = [=](int i_reduce, int i_load) {
        int offt;
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;

        offt = (i_load * jcp.reduce_dim + u0) * jcp.load_block;
        return EVEX_compress_addr(aux_reg_load_data,
                                  u1 * jcp.reduce_loop_load_step
                                  + jcp.typesize_in * offt);
    };

    auto output_ptr = [=](int i_load, int i_ur) {
        return EVEX_compress_addr(aux_reg_output_data,
                                  (i_load * jcp.bcast_dim + i_ur) * jcp.load_block
                                  * jcp.typesize_out);
    };

    auto init = [=]() {
        Label init_done;
        Label init_zero;

        if (jcp.with_sum) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    mic_prefetcht1(output_ptr(i_load, i_ur));
                }
            }
        }

        if (jcp.with_bias) {
            test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            jz(init_zero, T_NEAR);

            for (int i_load = 0; i_load < load_loop_blk; i_load++)
                for (int i_ur = 0; i_ur < ur; ++i_ur)
                    vmovups(vreg_accum(i_load, i_ur), bias_ptr(i_load));
            jmp(init_done, T_NEAR);
        }

        L(init_zero);
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                vpxord(r, r, r);
            }
        }
        L(init_done);
    };

    auto vcmp = [=](Xbyak::Opmask kmask,
                    Xbyak::Zmm zmm_src1, Xbyak::Zmm zmm_src2, const unsigned char cmp) {
        if (jcp.ver == ver_4vnni)
            vpcmpd(kmask, zmm_src1, zmm_src2, cmp);
        else
            vcmpps(kmask, zmm_src1, zmm_src2, cmp);
    };

    auto vmul = [=](Xbyak::Zmm zmm_dst, Xbyak::Opmask kmask,
                    Xbyak::Zmm zmm_src1, Xbyak::Zmm zmm_src2) {
        if (jcp.ver == ver_4vnni)
            vpmulld(zmm_dst | kmask, zmm_src1, zmm_src2);
        else
            vmulps(zmm_dst | kmask, zmm_src1, zmm_src2);
    };

    auto vadd = [=](const Xmm& x1, const Xmm& x2, const Operand& op) {
        if (jcp.ver == ver_4vnni)
            vpaddd(x1, x2, op);
        else
            vaddps(x1, x2, op);
    };

    auto store = [=]() {

        Label store_noadd;
        if (!jcp.with_sum) {
            test(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            jnz(store_noadd, T_NEAR);
        }

        for (int i_ur = 0; i_ur < ur; ++i_ur)
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                auto r = vreg_accum(i_load, i_ur);
                vadd(r, r, output_ptr(i_load, i_ur));
            }

        L(store_noadd);
        if (jcp.with_relu) {
            assert(ur * load_loop_blk <= 30);

            Label store_norelu;
            test(reg_reduce_pos_flag, FLAG_REDUCE_LAST);
            jz(store_norelu, T_NEAR);

            vpxord(zmm_zero, zmm_zero, zmm_zero);
            if (jcp.relu_negative_slope == 0) {
                zmm_relu_ns = zmm_zero;
            } else {
                mov(imm_addr64, float2int((float)jcp.relu_negative_slope));
                vmovq(xmm_relu_ns, imm_addr64);
                vbroadcastss(zmm_relu_ns, xmm_relu_ns);
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur)
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    vcmp(vmask, vreg_accum(i_load, i_ur), zmm_zero,
                         _cmp_lt_os);
                    vmul(vreg_accum(i_load, i_ur), vmask,
                         vreg_accum(i_load, i_ur), zmm_relu_ns);
                }
            L(store_norelu);
        }

        auto store_output = [=](bool output_is_aligned) {
            for (int i_ur = 0; i_ur < ur; ++i_ur)
                for (int i_load = 0; i_load < load_loop_blk; ++i_load)
                    if (output_is_aligned && jcp.use_vmovntps)
                        vmovntps(output_ptr(i_load, i_ur),
                                 vreg_accum(i_load, i_ur));
                    else
                        vmovups(output_ptr(i_load, i_ur),
                                vreg_accum(i_load, i_ur));
        };

        Label unaligned_store, end_store;
        test(aux_reg_output_data, cpu_isa_traits<avx512_common>::vlen - 1);
        jnz(unaligned_store, T_NEAR);
        store_output(true);
        jmp(end_store, T_NEAR);
        L(unaligned_store); {
        store_output(false);
    }
        L(end_store);
    };

    auto prefetch_callback = [=](int ur, int i_reduce, int i_ur, int i_load,
                                 bool last_block, bool wraparound, int reduce_step) {
        bool pf_ker_l1 = true;
        bool pf_ker_l2 = wraparound;
        int n_ops = (jcp.reduce_loop_unroll / reduce_step) * ur * load_loop_blk;
        int i_op = (i_reduce / reduce_step) * ur * load_loop_blk +
                   i_ur * load_loop_blk + i_load;

        int n_pf_ker_l1 = pf_ker_l1 ? jcp.reduce_block : 0;
        int n_pf_ker_l2 = pf_ker_l2 && wraparound ? jcp.reduce_block : 0;
        int n_pf_out_l1 = jcp.use_vmovntps ? 0 : ur;

        int pf_inp_ops = n_ops / 2; // # of operations during which to pf input
        int pf_inp_trigger;

        pf_inp_trigger = max(1, pf_inp_ops / ur);

        int n_other_pf =
                load_loop_blk * (n_pf_ker_l1 + n_pf_ker_l2 + n_pf_out_l1);
        int n_other_pf_ops = n_ops - pf_inp_ops;
        int other_pf_trigger
                = n_other_pf ? max(1, n_other_pf_ops / n_other_pf) : 0;

        if (i_op < pf_inp_ops && i_op % pf_inp_trigger == 0) {
            // input prefetches have the highest priority b/c the
            // first iteration of the kernel block touches all the
            // cache lines
            int i_pf = i_op / pf_inp_trigger;
            auto pf_reg = wraparound && last_block
                          ? reg_bcast_data
                          : (last_block ? aux1_reg_bcast_data
                                        : aux_reg_bcast_data);
            int offt = i_pf;
            {
                offt += wraparound && last_block
                        ? 0
                        : (last_block ? jcp.ur : jcp.bcast_dim);
                offt *= jcp.reduce_block;
            }
            mic_prefetcht0(ptr[pf_reg + offt * jcp.typesize_in]);
        }
        else if (i_op >= pf_inp_ops && n_other_pf) {
            // remaining prefetches are spread among the rest of the
            // operations; prefetches for output take priority
            // TODO: spread L2 prefetches among L1 prefetches
            i_op -= pf_inp_ops;
            if (i_op % other_pf_trigger == 0) {
                int i_pf = i_op / (load_loop_blk * other_pf_trigger);
                if (i_pf < n_pf_ker_l2) {
                    int offt = (i_pf + (i_load + 1) * jcp.reduce_dim)
                               * jcp.load_block;
                    mic_prefetcht1(ptr[aux_reg_load_data
                                       + offt * jcp.typesize_in]);
                } else if (i_pf < n_pf_ker_l2 + n_pf_ker_l1) {
                    i_pf -= n_pf_ker_l2;
                    auto pf_reg = last_block ? reg_load_data
                                             : aux_reg_load_data;
                    int offt = (i_pf + i_load * jcp.reduce_dim
                                + (last_block
                                   ? (wraparound ? jcp.reduce_dim : 0)
                                   : jcp.reduce_block))
                               * jcp.load_block;
                    mic_prefetcht0(ptr[pf_reg + offt * jcp.typesize_in]);
                } else if (i_pf < n_pf_ker_l1 + n_pf_ker_l2 + n_pf_out_l1) {
                    i_pf -= n_pf_ker_l1 + n_pf_ker_l2;
                    int offt = i_pf * jcp.load_block;
                    mic_prefetcht0(ptr[aux_reg_output_data
                                       + offt * jcp.typesize_out]);
                }
            }
        }
    };

    auto fma_block = [=](bool last_block) {
        assert(jcp.reduce_loop_unroll % jcp.fma_step == 0);

        int reduce_step = jcp.fma_step;
        if (jcp.ver == ver_4vnni)
            reduce_step *= 2;

        for (int i_reduce = 0; i_reduce < jcp.reduce_loop_unroll;
             i_reduce += reduce_step) {
            int load_scale = (jcp.ver == ver_4vnni) ? 2 : 1;
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                // if transposed input data used and if spatial size is
                // not divided by transpose step (4) then for last reduce step
                // we should load only needed load_registers data
                // and clear remaining
                if (jcp.transpose_src && jcp.is % jcp.fma_step && last_block
                    && i_reduce == jcp.reduce_loop_unroll - reduce_step) {
                    Label load_all;
                    Label load_finish;
                    test(reg_reduce_pos_flag, FLAG_SP_LAST);
                    jz(load_all, T_NEAR);

                    const int n_loads = jcp.is % jcp.fma_step;
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        if (i_fma < n_loads)
                            vmovups(vreg_load(i_load, i_fma),
                                    load_ptr(i_reduce + load_scale * i_fma,
                                             i_load));
                        else
                            vpxord(vreg_load(i_load, i_fma),
                                   vreg_load(i_load, i_fma),
                                   vreg_load(i_load, i_fma));
                    }
                    jmp(load_finish);

                    L(load_all);
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        vmovups(vreg_load(i_load, i_fma),
                                load_ptr(i_reduce + load_scale * i_fma, i_load));
                    }
                    L(load_finish);
                } else {
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                        vmovups(vreg_load(i_load, i_fma),
                                load_ptr(i_reduce
                                         + load_scale * i_fma,
                                         i_load));
                    }
                }
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                if (jcp.ver == ver_avx512_core && jcp.expl_bcast
                    && load_loop_blk > 1)
                    vbroadcastss(vreg_bcast, bcast_ptr(i_reduce, i_ur, false));
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    if (jcp.ver == ver_4fma)
                        v4fmaddps(vreg_accum(i_load, i_ur),
                                  vreg_load(i_load, 0),
                                  bcast_ptr(i_reduce, i_ur, false));
                    else if (jcp.ver == ver_4vnni)
                        vp4dpwssd(vreg_accum(i_load, i_ur),
                                  vreg_load(i_load, 0),
                                  bcast_ptr(i_reduce, i_ur, false));
                    else if (jcp.ver == ver_avx512_core && jcp.expl_bcast
                             && load_loop_blk > 1)
                        vfmadd231ps(vreg_accum(i_load, i_ur),
                                    vreg_load(i_load, 0), vreg_bcast);
                    else
                        vfmadd231ps(vreg_accum(i_load, i_ur),
                                    vreg_load(i_load, 0),
                                    bcast_ptr(i_reduce, i_ur, true));
                    prefetch_callback(ur, i_reduce, i_ur, i_load,
                                      last_block, wraparound, reduce_step);
                }
            }
        }
    };
    Label reduce_loop;
    Label reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);

    mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    mov(reduce_loop_iter, reg_reduce_loop_work);
    sub(reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);

    L(reduce_loop); {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    fma_block(true);

    store();
}

void jit_avx512_common_1x1_conv_kernel::generate() {
    preamble();

    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(reg_load_data, ptr[param1 + GET_OFF(load_data)]);
    mov(reg_output_data, ptr[param1 + GET_OFF(output_data)]);

    sub(rsp, stack_space_needed);

    if (jcp.with_bias) {
        mov(reg_bias_data, ptr[param1 + GET_OFF(bias_data)]);
    }

    mov(reg_load_loop_work, ptr[param1 + GET_OFF(load_dim)]);
    mov(reg_bcast_loop_work, ptr[param1 + GET_OFF(bcast_dim)]);
    mov(EVEX_compress_addr(rsp, bcast_loop_work_offt), reg_bcast_loop_work);
    mov(reg_reduce_loop_work, ptr[param1 + GET_OFF(reduce_dim)]);
    mov(reg_reduce_pos_flag, ptr[param1 + GET_OFF(reduce_pos_flag)]);

    mov(reg_relu_ns, reinterpret_cast<size_t>(&jcp.relu_negative_slope));

    auto load_loop_body = [=](int load_loop_blk) {
        bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);

        add(reg_bias_data,
            load_loop_blk * jcp.load_block * jcp.typesize_out);
        add(reg_output_data,
            load_loop_blk * jcp.bcast_dim * jcp.load_block *
            jcp.typesize_out);

        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
    };

    const int simd_w = 16;

    Label load_loop_blk[7];

    static const int ur_cases_fma_embd_bcast[] = { 2, 4, 5, 8, 14, 32 };
    static const int ur_cases_fma_expl_bcast[] = { 2, 5, 6, 9, 14, 32 };
    static const int ur_cases_4fma[] = { 2, 4, 6, 12, 32 };

    const int size_ur_cases_fma
            = (jcp.ver == ver_avx512_core && jcp.expl_bcast) ?
              sizeof(ur_cases_fma_expl_bcast) :
              sizeof(ur_cases_fma_embd_bcast);
    const int size_ur_cases_4fma = sizeof(ur_cases_4fma);

    const int *ur_cases_fma = (jcp.ver == ver_avx512_core && jcp.expl_bcast) ?
                              ur_cases_fma_expl_bcast :
                              ur_cases_fma_embd_bcast;
    const int *ur_cases = (jcp.ver == ver_4fma || jcp.ver == ver_4vnni)
                          ? ur_cases_4fma : ur_cases_fma;
    const int num_ur_cases = (jcp.ver == ver_4fma || jcp.ver == ver_4vnni ?
                              size_ur_cases_4fma :
                              size_ur_cases_fma)
                             / sizeof(*ur_cases);

    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.ur <= ur_cases[ur_idx]) {
            cmp(reg_load_loop_work, simd_w * (label_idx + 1));
            jle(load_loop_blk[label_idx], T_NEAR);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        if (jcp.ur <= ur_cases[ur_idx]) {
            int label_idx = num_ur_cases - ur_idx - 1;
            L(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    cmp(reg_load_loop_work, 0);
                    je(load_loop_blk[num_ur_cases], T_NEAR);
                }
                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    cmp(reg_load_loop_work, 2 * label_idx * simd_w);
                    je(load_loop_blk[label_idx - 1], T_NEAR);
                }
                cmp(reg_load_loop_work, (label_idx + 1) * simd_w);
                jge(load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx > 0; --idx) {
                cmp(reg_load_loop_work, simd_w * (idx + 1));
                je(load_loop_blk[idx], T_NEAR);
            }
            if (ur_idx < num_ur_cases - 2) {
                cmp(reg_load_loop_work, simd_w);
                jle(load_loop_blk[0], T_NEAR);
            }
        }
    }
    L(load_loop_blk[num_ur_cases]);

    add(rsp, stack_space_needed);

    postamble();
}


SaberStatus jit_avx512_common_1x1_conv_kernel::init_conf(jit_1x1_conv_conf_t &jcp,
                                                         conv_1x1_desc &conv_d,
                                                         int nthreads,
                                                         bool reduce_src) {
    if (!mayiuse(avx512_common)) {
                LOG(ERROR) <<"init a AVX512 kernel in non-avx512 machine is not permitted";
        return SaberUnImplError;
    }

    // const bool with_groups = jcp.ngroups > 1;
    bool args_ok = true;

    const int simd_w = 16;
    args_ok = true &&
              jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0 &&
              jcp.t_pad == 0 && jcp.l_pad == 0 &&
              jcp.stride_w == 1 && jcp.stride_h == 1 &&
              jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) {
                LOG(ERROR) << "ic:" << jcp.ic << ", oc:" << jcp.oc << ", stride_h:" << jcp.stride_h << ", stride_w:" << jcp.stride_w << ", kh:" << jcp.kh << ", kw:" << jcp.kw << ", pad:" << jcp.t_pad;
        return SaberUnImplError;
    }

    jcp.os = jcp.oh * jcp.ow;
    jcp.is = jcp.ih * jcp.iw;
    jcp.tr_is = rnd_up(jcp.is, 4);

    jcp.ic_block = jcp.oc_block = simd_w;
    jcp.transpose_src = false;

    jcp.ver = (mayiuse(avx512_core)) ? ver_avx512_core : ver_fma;
    jcp.fma_step = 1;
    jcp.typesize_in = sizeof(float);
    jcp.typesize_out = sizeof(float);

    const int SMALL_SPATIAL = 10;
    const int BIG_SPATIAL = 28;
    const int BIG_REDUCE_DIM = 1024;
    const int BIG_LOAD_DIM = 256;

    int load_blocking{ 0 };
    int load_blocking_max{ 0 };
    int bcast_blocking{ 0 };
    int bcast_blocking_max{ 0 };
    int reduce_blocking{ 0 };
    int reduce_blocking_max{ 0 };

    jcp.load_grp_count = 1;

    // const int L1_capacity = get_cache_size(1, true) / sizeof(float);
    const int L2_size = get_cache_size(2, true) / sizeof(float);
    const int L2_capacity = (L2_size * 3) / 4;

    {
        jcp.reduce_dim = jcp.ic;
        jcp.reduce_block = jcp.ic_block;

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.is;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll * jcp.bcast_dim * jcp.typesize_in;

        jcp.reduce_loop_load_step = jcp.reduce_loop_unroll * jcp.load_block * jcp.typesize_in;
        jcp.load_loop_load_step = jcp.reduce_dim * jcp.load_block * jcp.typesize_in;

        // adjusting registry blocking
        int max_regs, min_regs, size_treshold, ur_step;
        const int spatial = jcp.oh;
        if (jcp.ver == ver_avx512_core && (2 * jcp.mb) / nthreads >= 1) {
            max_regs = 9;
            min_regs = 6;
            size_treshold = 14;
            ur_step = 1;
            jcp.expl_bcast = true;

            if (jcp.load_dim > 128 && jcp.load_dim < BIG_LOAD_DIM &&
                spatial > SMALL_SPATIAL && spatial < BIG_SPATIAL) {
                max_regs = 6;
                min_regs = 5;
            }
        } else {
            bool is4ops = (jcp.ver == ver_4fma || jcp.ver == ver_4vnni);
            max_regs = is4ops ? 28 : 30;
            min_regs = 9;
            size_treshold = is4ops ? 28 : 14;
            ur_step = is4ops ? 4 : 1;
            jcp.expl_bcast = false;
            jcp.use_vmovntps = true;
        }

        jcp.ur = 1;
        for (int ur_w = max_regs; ur_w >= min_regs; ur_w -= ur_step) {
            if ((spatial >= size_treshold && spatial % ur_w == 0)
                || (spatial < size_treshold && jcp.os % ur_w == 0)) {
                jcp.ur = ur_w;
                break;
            }
        }
        if (jcp.ur == 1) {
            jcp.ur = utils::min(max_regs, jcp.os);
            int os_tail = jcp.os % max_regs;
            for (int i = max_regs; i >= min_regs; i -= ur_step) {
                int i_tail = jcp.os % i;
                if (i_tail > os_tail || i_tail == 0) {
                    jcp.ur = i;
                    os_tail = i_tail;
                    if (i_tail == 0)
                        break;
                }
            }
        }

        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
                = jcp.reduce_loop_unroll * jcp.bcast_dim * jcp.typesize_in;

        jcp.bcast_block = jcp.ur;

        jcp.bcast_loop_output_step = jcp.ur * jcp.load_block * jcp.typesize_out;
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.reduce_block * jcp.typesize_in;
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_iter_step = jcp.load_block;


        jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

        int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);

        if (jcp.ver == ver_avx512_core && jcp.expl_bcast) {
            if (jcp.load_dim <= BIG_LOAD_DIM && spatial > SMALL_SPATIAL
                && spatial < BIG_SPATIAL)
                reduce_blocking = utils::min(jcp.reduce_dim, 80);
            else if (spatial > SMALL_SPATIAL)
                reduce_blocking = utils::min(jcp.reduce_dim, 512);
            else
                reduce_blocking = utils::min(jcp.reduce_dim, 256);
            if (jcp.mb > 1
                && (spatial >= 28 || (spatial >= 17 && jcp.mb > 112)))
                jcp.use_vmovntps = true;
            else
                jcp.use_vmovntps = false;
        } else {
            int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

            reduce_blocking = nb_reduce;
            if (spatial <= SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
                reduce_blocking = 16;
            else if (spatial > SMALL_SPATIAL
                     && jcp.reduce_dim >= BIG_REDUCE_DIM)
                reduce_blocking = 8;
            reduce_blocking = best_divider(nb_reduce, 1, reduce_blocking, true);
            reduce_blocking *= jcp.reduce_block;
        }

        if (reduce_blocking < jcp.reduce_dim) {
            jcp.use_vmovntps = false;
            jcp.loop_order = reduce_src ? loop_rbl : loop_rlb;
        }
        load_blocking = jcp.load_dim;

        jcp.load_grp_count = div_up(nthreads, jcp.mb * jcp.ngroups * nb_bcast);
        jcp.load_grp_count = best_divider(
                nthreads, jcp.load_grp_count, 2 * jcp.load_grp_count, false);

        if (jcp.ver == ver_avx512_core && jcp.expl_bcast && jcp.bcast_dim <= 64
            && jcp.load_dim * jcp.reduce_dim >= L2_size) {
            jcp.load_grp_count = max(jcp.load_grp_count, 4);
        }
        else if (jcp.bcast_dim <= 49 && jcp.mb <= nthreads
                 && jcp.load_dim > 512 && jcp.load_dim / jcp.reduce_dim >= 4) {
            jcp.load_grp_count = max(jcp.load_grp_count, 2);
            load_blocking = jcp.load_block;
        }

        bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                                div_up(nthreads, jcp.load_grp_count))
                         * jcp.bcast_block;
        bcast_blocking = min(jcp.bcast_dim, bcast_blocking);
        bcast_blocking = rnd_up(bcast_blocking, jcp.bcast_block);

        int space_for_bcast
                = (L2_capacity - /* kernel_size - */
                   2 * jcp.load_block * reduce_blocking
                   - jcp.ur * reduce_blocking - 3 * 1024);
        if (jcp.reduce_dim * jcp.bcast_dim > L2_capacity)
            space_for_bcast /= 2;

        int bcast_in_cache
                = max(jcp.bcast_block, space_for_bcast / reduce_blocking);
        bcast_blocking = min(
                bcast_blocking, rnd_dn(bcast_in_cache, jcp.bcast_block));

        load_blocking_max = load_blocking;
        bcast_blocking_max = bcast_blocking * 3 / 2;
        reduce_blocking_max = reduce_blocking;
    }

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);
    assert(reduce_blocking_max);
    assert(load_blocking % jcp.load_block == 0);
    assert(reduce_blocking % jcp.reduce_block == 0);
    assert(load_blocking_max % jcp.load_block == 0);
    assert(reduce_blocking_max % jcp.reduce_block == 0);

    assert(jcp.bcast_block % jcp.ur == 0);
    assert(jcp.reduce_dim % jcp.reduce_block == 0);

    jcp.ur_tail = jcp.bcast_dim % jcp.ur;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = load_blocking / jcp.load_block;
    jcp.nb_load_blocking_max = load_blocking_max / jcp.load_block;
    jcp.nb_reduce_blocking = reduce_blocking / jcp.reduce_block;
    jcp.nb_reduce_blocking_max = reduce_blocking_max / jcp.reduce_block;

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    return SaberSuccess;
}

void jit_avx512_common_1x1_conv_kernel::balance(jit_1x1_conv_conf_t &jcp,
                                                int nthreads) {
    if (nthreads < jcp.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        jcp.nthr = jcp.nthr_mb = jcp.nthr_g =
        jcp.nthr_oc_b = jcp.nthr_ic_b = 1;
        return;
    }
    const int nb_bcast = utils::div_up(jcp.bcast_dim, jcp.bcast_block);
    const int nb_load = div_up(jcp.load_dim, jcp.load_block);
    const int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    jcp.nthr_g = jcp.ngroups;
    const int nthr = nthreads / jcp.nthr_g;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level
        * optimizer tries to minimize memory consumption. few notes: (n1)
        * unclear why, but that essentially helps first convolution...
        *  (n2) assuming the reduction over minibatch is always there:
        *    - instead of 8 it should be 5 here (write ~= 2 read):
        *      kernel: temporal workspace 1 write
        *      reduction: 1 read from workspace and 1 write to the diff_wei
        *    - but experiments showed 8 works better than 5 or 6... */
        int bcast_koeff = 1;
        int load_koeff = 1;
        int output_koeff = 12;
        if (jcp.transpose_src) {
            bcast_koeff = 5;
            load_koeff = 1;
            output_koeff = 8;
        }
        return 0
               + bcast_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
                 * div_up(jcp.ngroups, jcp.nthr_g)
                 * div_up(nb_bcast, nthr_ic_b) * jcp.ic_block * jcp.reduce_block
                 / jcp.stride_h / jcp.stride_w /* (n1) */
               + load_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
                 * div_up(jcp.ngroups, jcp.nthr_g)
                 * div_up(nb_load, nthr_oc_b) * jcp.oc_block * jcp.reduce_block
               + output_koeff /* (n2) */
                 * div_up(jcp.ngroups, jcp.nthr_g) * div_up(nb_load, nthr_oc_b)
                 * div_up(nb_bcast, nthr_ic_b) * jcp.ic_block
                 * jcp.oc_block;
    };

    int nthr_mb = 1, nthr_oc_b = 1, nthr_ic_b = 1;
    int best_mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = min(nthr, jcp.mb * nb_reduce);
    for (nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = min(nthr_par, nb_load);
        for (nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            nthr_ic_b = min(nthr_par / nthr_oc_b, nb_bcast);
            int mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                jcp.nthr_mb = nthr_mb;
                jcp.nthr_oc_b = nthr_oc_b;
                jcp.nthr_ic_b = nthr_ic_b;
            }
        }
    }
    if (jcp.nthr_mb > nthreads / 2 && jcp.nthr_mb < nthreads)
        jcp.nthr_mb = min(jcp.mb, nthreads);

    jcp.nthr = jcp.nthr_mb * jcp.nthr_g * jcp.nthr_oc_b * jcp.nthr_ic_b;
    assert(jcp.nthr <= nthreads);
}


} // namespace jit
} // namespace saber
} // namespace anakin

