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

#ifndef ANAKIN_SABER_FUNCS_JIT_AVX512_RTUS_DRIVER_H
#define ANAKIN_SABER_FUNCS_JIT_AVX512_RTUS_DRIVER_H

#include "saber/core/tensor.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "jit_generator.h"
#include "jit_uni_1x1_conv_utils.h"


namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

struct rtus_driver_t : public jit_generator {

    struct call_params_t {
        const void *ws; /* reduced image (w/ strides = 1) */
        const void *src; /* source image (w/ non-unit strides) */
        size_t icb;
        size_t os;
        size_t iw_start;
    };

    void(*ker_)(const call_params_t *p);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(rtus_driver_t)

    /* cpu specific part */
    using Vmm = Xbyak::Zmm;
    Xbyak::Reg64 reg_ws = abi_param1;
    Xbyak::Reg64 reg_src = abi_not_param1;
    Xbyak::Reg64 reg_icb = rdx;
    Xbyak::Reg64 reg_os = r11;
    Xbyak::Reg64 reg_iw_start = r8;

    Xbyak::Reg64 reg_cur_os = rax;
    Xbyak::Reg64 reg_cur_iw = r9;
    Xbyak::Reg64 reg_cur_src = r10;

    int iw_, stride_w_;
    int src_step_h_, src_step_icb_, ws_step_icb_, vlen_, vlen_shift_;
    bool src_to_ws_;
    size_t typesize_;
    Vmm reg_zero;
    Vmm reg_v;

    rtus_driver_t(int iw, int stride_w, int src_step_h,
                  int src_step_icb, int ws_step_icb,
                  bool src_to_ws, size_t typesize)
            : iw_(iw), stride_w_(stride_w), src_step_h_(src_step_h)
            , src_step_icb_(src_step_icb), ws_step_icb_(ws_step_icb)
            , src_to_ws_(src_to_ws), typesize_(typesize) {
        vlen_ = cpu_isa_traits<avx512_common>::vlen;
        vlen_shift_ = cpu_isa_traits<avx512_common>::vlen_shift;
        if (typesize_ == 2) {
            vlen_ /= 2;
            vlen_shift_--;
        }

        reg_zero = Vmm(0);
        reg_v = Vmm(1);

        generate();
    }

    void loop_is() {
        mov(reg_cur_src, reg_src);
        mov(reg_cur_iw, reg_iw_start);
        mov(reg_cur_os, reg_os);

        Label is_loop, skip_h_step;
        L(is_loop);

        if (src_to_ws_) {
            vmovups(reg_v, ptr[reg_cur_src]);
            vmovups(ptr[reg_ws], reg_v);
        } else {
            vmovups(reg_v, ptr[reg_ws]);
            vmovups(ptr[reg_cur_src], reg_v);
            for (int w = 1; w < stride_w_; ++w)
                vmovups(ptr[reg_cur_src + w * vlen_], reg_zero);
        }

        add(reg_ws, vlen_);

        add(reg_cur_iw, stride_w_);
        add(reg_cur_src, stride_w_ * vlen_);

        cmp(reg_cur_iw, iw_);
        jl(skip_h_step);

        if (src_to_ws_) {
            add(reg_cur_src, (src_step_h_ - iw_) * vlen_);
        } else {
            Xbyak::Reg64 reg_cur_src_fin = reg_cur_iw; /* just reuse */
            mov(reg_cur_src_fin, reg_cur_src);
            add(reg_cur_src_fin, (src_step_h_ - iw_) * vlen_);
            Label ih_loop;
            L(ih_loop);

            for (int w = 0; w < stride_w_; ++w)
                vmovups(ptr[reg_cur_src + w * vlen_], reg_zero);

            add(reg_cur_src, stride_w_ * vlen_);
            cmp(reg_cur_src, reg_cur_src_fin);
            jl(ih_loop);
        }
        xor_(reg_cur_iw, reg_cur_iw);

        L(skip_h_step);

        sub(reg_cur_os, vlen_);
        jnz(is_loop);

        /* restore dst */
        sub(reg_ws, reg_os);
    }

    void generate() {
#if defined(_WIN32)
        assert(reg_src == abi_not_param1 && abi_not_param1 == rdi);
        push(rdi);
#endif

#define READ_PARAM(what) \
        mov(reg_ ## what, ptr[abi_param1 + offsetof(call_params_t, what)])
        READ_PARAM(src);
        READ_PARAM(icb);
        READ_PARAM(os);
        READ_PARAM(iw_start);

        assert(reg_ws == abi_param1);
        READ_PARAM(ws); /* reg_ws should always be read the last */
#undef  READ_PARAM

        shl(reg_os, vlen_shift_);

        if (!src_to_ws_) {
            uni_vpxor(reg_zero, reg_zero, reg_zero);
        }

        Label icb_loop;
        L(icb_loop);

        loop_is();

        add(reg_ws, ws_step_icb_ * vlen_);
        add(reg_src, src_step_icb_ * vlen_);

        dec(reg_icb);
        jnz(icb_loop, T_NEAR);

#if defined(_WIN32)
        pop(rdi);
#endif

        uni_vzeroupper();
        ret();
        this->ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(this->getCode()));
    }
};

template <typename Dtype>
inline void init_rtus_driver(rtus_driver_t **p_rtus_driver,
                             jit_1x1_conv_conf_t &jcp,
                             conv_1x1_desc &conv_d,
                             size_t &ws_per_thread,
                             Dtype **p_scratch) {
    const int max_threads = anakin_get_max_threads();
    size_t factor = 0;

    factor = jcp.nb_reduce;

    size_t typesize = sizeof(decltype(**p_scratch));

    ws_per_thread = factor * jcp.is * jcp.ic_block;
    *p_scratch = (Dtype*)zmalloc(max_threads * ws_per_thread * typesize, 64);

    const int ih = conv_d.ih;
    const int iw = conv_d.iw;

    const int src_step_h = conv_d.stride_h * iw;
    const int src_step_icb = ih * iw;
    const int ws_step_icb = jcp.is;

    const bool src_to_ws = true;

    *p_rtus_driver = new rtus_driver_t(iw, conv_d.stride_w, src_step_h,
                                       src_step_icb, ws_step_icb, src_to_ws, typesize);
}


} // namaespace jit
} // namespace saber
} // namespace anakin
#endif // ANAKIN_SABER_FUNCS_JIT_AVX512_RTUS_DRIVER_H
