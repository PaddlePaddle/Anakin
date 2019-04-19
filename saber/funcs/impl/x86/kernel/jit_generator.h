#pragma once

#include <type_traits>
#include <limits.h>

#define XBYAK64
#define XBYAK_NO_OP_NAMES
/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR
#ifdef USE_SGX
#undef XBYAK_USE_MMAP_ALLOCATOR
#endif

#include "anakin_config.h"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"
#include "x86_utils.h"
#include "anakin_thread.h"

#define DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_name)                      \
  const char *name() const override { return #jit_name; } \
  const char *source_file() const override { return __FILE__; }

namespace anakin {
namespace saber {
namespace jit {

static Xbyak::util::Cpu cpu;
typedef enum {
    isa_any,
    sse42,
    avx,
    avx2,
    avx512_common,
    avx512_core,
    avx512_core_vnni,
    avx512_mic,
    avx512_mic_4ops,
} cpu_isa_t;  // Instruction set architecture

template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

template <>
struct cpu_isa_traits<sse42> {
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 16;
};

template <>
struct cpu_isa_traits<avx2> {
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = 32;
    static constexpr int n_vregs = 16;
};

template <>
struct cpu_isa_traits<avx512_common> {
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = 64;
    static constexpr int n_vregs = 32;
};

template <>
struct cpu_isa_traits<avx512_core> : public cpu_isa_traits<avx512_common> {};

template <>
struct cpu_isa_traits<avx512_mic> : public cpu_isa_traits<avx512_common> {};

template <>
struct cpu_isa_traits<avx512_mic_4ops> : public cpu_isa_traits<avx512_common> {
};

static inline bool mayiuse(const cpu_isa_t cpu_isa) {
    using namespace Xbyak::util;

    switch (cpu_isa) {
    case sse42:
        return cpu.has(Cpu::tSSE42);

    case avx:
        return cpu.has(Cpu::tAVX);

    case avx2:
        return cpu.has(Cpu::tAVX2);

    case avx512_common:
        //        return false;//for can`t pass test of jit
        return cpu.has(Cpu::tAVX512F);

    case avx512_core:
        return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) &&
               cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);

    case avx512_core_vnni:
        return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512BW) &&
               cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ) &&
               cpu.has(Cpu::tAVX512_VNNI);

    case avx512_mic:
        return true && cpu.has(Cpu::tAVX512F) && cpu.has(Cpu::tAVX512CD) &&
               cpu.has(Cpu::tAVX512ER) && cpu.has(Cpu::tAVX512PF);

    case avx512_mic_4ops:
        return true && mayiuse(avx512_mic) && cpu.has(Cpu::tAVX512_4FMAPS) &&
               cpu.has(Cpu::tAVX512_4VNNIW);

    case isa_any:
        return true;
    }

    return false;
}

static inline int float2int(float x) {
    union {
        float vfloat;
        int vint;
    } cvt;
    cvt.vfloat = x;
    return cvt.vint;
}


inline unsigned int get_cache_size(int level, bool per_core = true) {
    unsigned int l = level - 1;

    // Currently, if XByak is not able to fetch the cache topology
    // we default to 32KB of L1, 512KB of L2 and 1MB of L3 per core.
    if (cpu.data_cache_levels == 0) {
        const int L1_cache_per_core = 32000;
        const int L2_cache_per_core = 512000;
        const int L3_cache_per_core = 1024000;
        int num_cores = per_core ? 1 : anakin_get_max_threads();

        switch (l) {
        case (0):
            return L1_cache_per_core * num_cores;

        case (1):
            return L2_cache_per_core * num_cores;

        case (2):
            return L3_cache_per_core * num_cores;

        default:
            return 0;
        }
    }

    if (l < cpu.data_cache_levels) {
        if (cpu.cores_sharing_data_cache[l] > 0) {
            return cpu.data_cache_size[l] /
                   (per_core ? cpu.cores_sharing_data_cache[l] : 1);
        } else {
            return cpu.data_cache_size[l];
        }
    } else {
        return 0;
    }
}

#ifdef XBYAK64
constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
    Xbyak::Operand::RBX,
    Xbyak::Operand::RBP,
    Xbyak::Operand::R12,
    Xbyak::Operand::R13,
    Xbyak::Operand::R14,
    Xbyak::Operand::R15,
#ifdef _WIN
    Xbyak::Operand::RDI,
    Xbyak::Operand::RSI,
#endif
};

constexpr Xbyak::Operand::Code common_save_gpr_regs[] = {
    Xbyak::Operand::RAX,
    Xbyak::Operand::RCX,
    Xbyak::Operand:: RDX,
    Xbyak::Operand:: RBX,
    Xbyak::Operand:: RSI,
    Xbyak::Operand:: RDI,
    Xbyak::Operand:: R8,
    Xbyak::Operand:: R9,
    Xbyak::Operand:: R10,
    Xbyak::Operand:: R11,
    Xbyak::Operand:: R12,
    Xbyak::Operand:: R13,
    Xbyak::Operand:: R14,
    Xbyak::Operand:: R15,
};

#ifdef _WIN
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RCX),
       abi_param2(Xbyak::Operand::RDX), abi_param3(Xbyak::Operand::R8),
       abi_param4(Xbyak::Operand::R9), abi_not_param1(Xbyak::Operand::RDI);
#else
static const Xbyak::Reg64 abi_param1(Xbyak::Operand::RDI),
       abi_param2(Xbyak::Operand::RSI), abi_param3(Xbyak::Operand::RDX),
       abi_param4(Xbyak::Operand::RCX), abi_not_param1(Xbyak::Operand::RCX);
#endif
#endif

template <typename T, typename U, typename... Us>
struct all_same : std::false_type {};

template <typename T, typename... Us>
struct all_same<T, T, Us...> : all_same<T, Us...> { };

template <typename T>
struct all_same<T, T> : std::true_type {};

template <size_t len = 64>
class jit_tagged_label_base {
public:
    enum { maxlen = len };
    template <size_t n, typename... Tags,
              typename = std::enable_if<all_same<char, Tags...>::value>>
    jit_tagged_label_base(const char (&base)[n], Tags... tags) {
        // XXX: This code is ugly but useful
        constexpr size_t ntags = sizeof...(tags);
        static_assert(n + ntags < maxlen, "resulting label may be too long");
        // paste tags first in case base has unexpected null chars
        paste_tags(tags...);

        for (size_t i = 0; i < n; i++) {
            label_name_[ntags + i] = base[i];
        }

        // don't assume that the base string is 0-terminated
        label_name_[ntags + n] = '\0';
    }
    operator const char* () const {
        return label_name_;
    }
    const char* c_str() const {
        return label_name_;
    }
private:
    char label_name_[maxlen];
    void paste_tags() { }
    template <typename... Tags>
    void paste_tags(char tag, Tags... tags) {
        label_name_[sizeof...(tags)] = tag;
        paste_tags(tags...);
    }
};

typedef jit_tagged_label_base<> jit_tagged_label;

extern "C" Xbyak::uint8 __jit_start;
extern "C" Xbyak::uint8 __jit_end;

class jit_generator : public Xbyak::CodeGenerator {
private:
    const size_t xmm_reg_numbers = 8;
    const size_t ymm_reg_numbers = 16;
    const size_t zmm_reg_numbers = 32;
    const size_t xmm_len = 16;
    const size_t ymm_len = 32;
    const size_t zmm_len = 64;
#ifdef _WIN
    const size_t xmm_to_preserve_start = 6;
    const size_t xmm_to_preserve = 10;
#else
    const size_t xmm_to_preserve_start = 0;
    const size_t xmm_to_preserve = 0;
#endif

    const size_t num_abi_save_gpr_regs =
        sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

    const size_t num_common_save_gpr_regs =
        sizeof(common_save_gpr_regs) / sizeof(common_save_gpr_regs[0]);

    const size_t size_of_abi_save_regs =
        num_abi_save_gpr_regs * rax.getBit() / 8 + xmm_to_preserve * xmm_len;

public:
    enum {
        _cmp_eq_oq = 0u,
        _cmp_lt_os = 1u,
        _cmp_le_os = 2u,
        _cmp_neq_uq = 4u,
        _cmp_nlt_us = 5u,
        _cmp_nle_us = 6u,
    };

    Xbyak::Reg64 param1 = abi_param1;
    const int EVEX_max_8b_offt = 0x200;
    const Xbyak::Reg64 reg_EVEX_max_8b_offt = rbp;

    inline size_t get_size_of_abi_save_regs() {
        return size_of_abi_save_regs;
    }

    void preamble() {
        if (xmm_to_preserve) {
            sub(rsp, xmm_to_preserve * xmm_len);

            for (size_t i = 0; i < xmm_to_preserve; ++i) {
                movdqu(ptr[rsp + i * xmm_len], Xbyak::Xmm(xmm_to_preserve_start + i));
            }
        }

        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i) {
            push(Xbyak::Reg64(abi_save_gpr_regs[i]));
        }

        if (mayiuse(avx512_common)) {
            mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);
        }
    }
    void save_common_regs() {
        if (mayiuse(avx512_core)) {
            sub(rsp, zmm_reg_numbers * zmm_len);

            for (size_t i = 0; i < zmm_reg_numbers; ++i) {
                vmovdqu32(ptr[rsp + i * zmm_len], Xbyak::Zmm(i));
            }
        } else if (mayiuse(avx)) {
            sub(rsp, ymm_reg_numbers * ymm_len);

            for (size_t i = 0; i < ymm_reg_numbers; ++i) {
                vmovdqu(ptr[rsp + i * ymm_len], Xbyak::Ymm(i));
            }

        } else {
            sub(rsp, xmm_reg_numbers * xmm_len);

            for (size_t i = 0; i < xmm_reg_numbers; ++i) {
                movdqu(ptr[rsp + i * xmm_len], Xbyak::Xmm(i));
            }
        }

        for (size_t i = 0; i < num_common_save_gpr_regs; ++i) {
            push(Xbyak::Reg64(common_save_gpr_regs[i]));
        }
    }

    void restore_common_regs() {
        for (size_t i = 0; i < num_common_save_gpr_regs; ++i) {
            pop(Xbyak::Reg64(common_save_gpr_regs[num_common_save_gpr_regs - 1 - i]));
        }

        if (mayiuse(avx512_core)) {
            for (size_t i = 0; i < zmm_reg_numbers; ++i) {
                vmovdqu32(Xbyak::Zmm(i), ptr[rsp + i * zmm_len]);
            }

            add(rsp, zmm_reg_numbers * zmm_len);
        } else if (mayiuse(avx)) {
            for (size_t i = 0; i < ymm_reg_numbers; ++i) {
                vmovdqu(Xbyak::Ymm(i), ptr[rsp + i * ymm_len]);
            }

            add(rsp, ymm_reg_numbers * ymm_len);
        } else {
            for (size_t i = 0; i < xmm_reg_numbers; ++i) {
                movdqu(Xbyak::Xmm(i), ptr[rsp + i * xmm_len]);
            }

            add(rsp, xmm_reg_numbers * xmm_len);
        }
    }


    void mic_prefetcht0(Xbyak::Address a) {
        if (mayiuse(avx512_mic)) {
            prefetcht0(a);
        }
    }

    void mic_prefetcht1(Xbyak::Address a) {
        if (mayiuse(avx512_mic)) {
            prefetcht1(a);
        }
    }

    void mic_prefetcht2(Xbyak::Address a) {
        if (mayiuse(avx512_mic)) {
            prefetcht2(a);
        }
    }

    void uni_vzeroupper() {
        if (mayiuse(avx) && !mayiuse(avx512_mic)) {
            vzeroupper();
        }
    }

    void postamble() {
        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i) {
            pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
        }

        if (xmm_to_preserve) {
            for (size_t i = 0; i < xmm_to_preserve; ++i) {
                movdqu(Xbyak::Xmm(xmm_to_preserve_start + i), ptr[rsp + i * xmm_len]);
            }

            add(rsp, xmm_to_preserve * xmm_len);
        }

        uni_vzeroupper();
        ret();
    }



    Xbyak::Address make_safe_addr(const Xbyak::Reg64& reg_out, size_t offt,
                                  const Xbyak::Reg64& tmp_reg, bool bcast = false) {
        if (offt > INT_MAX) {
            mov(tmp_reg, offt);
            return bcast ? ptr_b[reg_out + tmp_reg] : ptr[reg_out + tmp_reg];
        } else {
            return bcast ? ptr_b[reg_out + offt] : ptr[reg_out + offt];
        }
    }

    Xbyak::Address EVEX_compress_addr_safe(const Xbyak::Reg64& base,
                                           size_t raw_offt, const Xbyak::Reg64& reg_offt, bool bcast = false) {
        if (raw_offt > INT_MAX) {
            return make_safe_addr(base, raw_offt, reg_offt, bcast);
        } else {
            return EVEX_compress_addr(base, raw_offt, bcast);
        }
    }

    void safe_add(const Xbyak::Reg64& base, size_t raw_offt,
                  const Xbyak::Reg64& reg_offt) {
        if (raw_offt > INT_MAX) {
            mov(reg_offt, raw_offt);
            add(base, reg_offt);
        } else {
            add(base, raw_offt);
        }
    }

    void safe_sub(const Xbyak::Reg64& base, size_t raw_offt,
                  const Xbyak::Reg64& reg_offt) {
        if (raw_offt > INT_MAX) {
            mov(reg_offt, raw_offt);
            sub(base, reg_offt);
        } else {
            sub(base, raw_offt);
        }
    }

    void uni_vpxor(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                   const Xbyak::Operand& op) {
        assert(x1.getIdx() == x2.getIdx());
        pxor(x2, op);
    }

    void uni_vpxor(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2,
                   const Xbyak::Operand& op) {
        if (mayiuse(avx2)) {
            vpxor(x1, x2, op);
        } else {
            vxorps(x1, x2, op);
        }
    }

    void uni_vpxor(const Xbyak::Zmm& x1, const Xbyak::Zmm& x2,
                   const Xbyak::Operand& op) {
        vpxord(x1, x2, op);
    }

    void uni_vmovdqu(const Xbyak::Address& addr, const Xbyak::Xmm& x) {
        movdqu(addr, x);
    }

    void uni_vmovdqu(const Xbyak::Address& addr, const Xbyak::Ymm& x) {
        vmovdqu(addr, x);
    }

    void uni_vmovdqu(const Xbyak::Address& addr, const Xbyak::Zmm& x) {
        vmovdqu32(addr, x);
    }

    void uni_vmovdqu(const Xbyak::Xmm& x, const Xbyak::Address& addr) {
        movdqu(x, addr);
    }

    void uni_vmovdqu(const Xbyak::Ymm& x, const Xbyak::Address& addr) {
        vmovdqu(x, addr);
    }

    void uni_vmovdqu(const Xbyak::Zmm& x, const Xbyak::Address& addr) {
        vmovdqu32(x, addr);
    }

    void uni_vmovups(const Xbyak::Address& addr, const Xbyak::Xmm& x) {
        movups(addr, x);
    }

    void uni_vmovups(const Xbyak::Address& addr, const Xbyak::Ymm& x) {
        vmovups(addr, x);
    }

    void uni_vmovups(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
        movups(x, op);
    }

    void uni_vmovups(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
        vmovups(x, op);
    }

    void uni_vmovntps(const Xbyak::Address& addr, const Xbyak::Xmm& x) {
        movntps(addr, x);
    }

    void uni_vmovntps(const Xbyak::Address& addr, const Xbyak::Ymm& x) {
        vmovntps(addr, x);
    }

    void uni_vbroadcastss(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
        movss(x, op);
        shufps(x, x, 0x0);
    }

    void uni_vbroadcastss(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
        if (mayiuse(avx2)) {
            vbroadcastss(x, op);
        } else {
            Xbyak::Xmm t(x.getIdx());

            if (t.getIdx() != op.getIdx()) {
                movss(t, op);
            }

            vinsertf128(x, x, t, 1);
            vshufps(x, x, x, 0);
        }
    }

    void uni_vpbroadcastd(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
        movsd(x, op);
        pshufd(x, x, 0x0);
    }

    void uni_vpbroadcastd(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
        if (mayiuse(avx2)) {
            vpbroadcastd(x, op);
        } else {
            Xbyak::Xmm t(x.getIdx());

            if (t.getIdx() != op.getIdx()) {
                movsd(t, op);
            }

            vinsertf128(x, x, t, 1);
            vshufps(x, x, x, 0);
        }
    }

    void uni_vdivps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        divps(x, op2);
    }

    void uni_vdivps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        vdivps(x, op1, op2);
    }

    void uni_vdivps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2, const Xbyak::Xmm& buf) {
        movups(buf, op1);
        divps(buf, op2);

        if (x.getIdx() != buf.getIdx()) {
            movups(x, buf);
        }
    }

    void uni_vdivps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2, const Xbyak::Ymm& buf) {
        vdivps(x, op1, op2);
    }

    void uni_vaddps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        addps(x, op2);
    }

    void uni_vaddps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        vaddps(x, op1, op2);
    }

    void uni_vpsignd(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                     const Xbyak::Operand& op) {
        assert(x1.getIdx() == x2.getIdx());
        psignd(x1, op);
    }
    void uni_vpsignd(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2,
                     const Xbyak::Operand& op) {
        vpsignd(x1, x2, op);
    }

    void uni_vsubps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        subps(x, op2);
    }

    void uni_vsubps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        vsubps(x, op1, op2);
    }

    void uni_vsubps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2, const Xbyak::Xmm& buf) {
        movups(buf, op1);
        subps(buf, op2);

        if (x.getIdx() != buf.getIdx()) {
            movups(x, buf);
        }
    }

    void uni_vsubps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2, const Xbyak::Ymm& buf) {
        vsubps(x, op1, op2);
    }

    void uni_vmulps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        mulps(x, op2);
    }

    void uni_vmulps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        vmulps(x, op1, op2);
    }

    void uni_vfmadd213ps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                         const Xbyak::Operand& op) {
        mulps(x1, x2);
        addps(x1, op);
    }

    void uni_vfmadd213ps(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2,
                         const Xbyak::Operand& op) {
        vfmadd213ps(x1, x2, op);
    }

    void uni_vfmadd231ps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                         const Xbyak::Operand& op) {
        mulps(x2, op);
        addps(x1, x2);
    }

    void uni_vfmadd231ps(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2,
                         const Xbyak::Operand& op) {
        vfmadd231ps(x1, x2, op);
    }

    void uni_vfnmadd231ps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                          const Xbyak::Operand& op) {
        mulps(x2, op);
        subps(x1, x2);
    }

    void uni_vfnmadd231ps(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2,
                          const Xbyak::Operand& op) {
        vfnmadd231ps(x1, x2, op);
    }

    void uni_vsqrtps(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
        sqrtps(x, op);
    }

    void uni_vsqrtps(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
        vsqrtps(x, op);
    }

    void uni_vpaddd(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                    const Xbyak::Operand& op) {
        assert(x1.getIdx() == x2.getIdx());
        paddd(x2, op);
    }

    void uni_vpaddd(const Xbyak::Ymm& x1, const Xbyak::Xmm& x2,
                    const Xbyak::Operand& op) {
        vpaddd(x1, x2, op);
    }

    void uni_vandps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        andps(x, op2);
    }

    void uni_vandps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        vandps(x, op1, op2);
    }

    void uni_vorps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                   const Xbyak::Operand& op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        orps(x, op2);
    }

    void uni_vorps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                   const Xbyak::Operand& op2 = Xbyak::Operand()) {
        vorps(x, op1, op2);
    }

    void uni_vpslld(const Xbyak::Xmm& x, const Xbyak::Operand& op,
                    const int imm) {
        assert(x.getIdx() == op.getIdx());
        pslld(x, imm);
    }

    void uni_vpslld(const Xbyak::Ymm& x, const Xbyak::Operand& op,
                    const int imm) {
        vpslld(x, op, imm);
    }

    void uni_vpsrld(const Xbyak::Xmm& x, const Xbyak::Operand& op,
                    const int imm) {
        assert(x.getIdx() == op.getIdx());
        psrld(x, imm);
    }

    void uni_vpsrld(const Xbyak::Ymm& x, const Xbyak::Operand& op,
                    const int imm) {
        vpsrld(x, op, imm);
    }

    void uni_vmaxps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        maxps(x, op2);
    }

    void uni_vmaxps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        vmaxps(x, op1, op2);
    }

    void uni_vminps(const Xbyak::Xmm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        assert(x.getIdx() == op1.getIdx());
        minps(x, op2);
    }

    void uni_vminps(const Xbyak::Ymm& x, const Xbyak::Operand& op1,
                    const Xbyak::Operand& op2 = Xbyak::Operand()) {
        vminps(x, op1, op2);
    }

    void uni_vcmpgtps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                      const Xbyak::Operand& op) {
        assert(x1.getIdx() == x2.getIdx());
        cmpps(x1, op, 0x6);
    }

    void uni_vcmpgtps(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2,
                      const Xbyak::Operand& op) {
        vcmpgtps(x1, x2, op);
    }

    void uni_vblendvps(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2,
                       const Xbyak::Operand& op, const Xbyak::Xmm& msk) {
        assert(x1.getIdx() == x2.getIdx());
        blendvps(x1, op);
    }

    void uni_vblendvps(const Xbyak::Ymm& x1, const Xbyak::Ymm& x2,
                       const Xbyak::Operand& op, const Xbyak::Ymm& msk) {
        vblendvps(x1, x2, op, msk);
    }

    void uni_vroundps(const Xbyak::Xmm& x, const Xbyak::Operand& op,
                      const int imm) {
        roundps(x, op, imm);
    }

    void uni_vroundps(const Xbyak::Ymm& x, const Xbyak::Operand& op,
                      const int imm) {
        vroundps(x, op, imm);
    }

    void uni_vcvtps2dq(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
        cvtps2dq(x, op);
    }

    void uni_vcvtps2dq(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
        vcvtps2dq(x, op);
    }

    void uni_vcvtdq2ps(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
        cvtdq2ps(x, op);
    }

    void uni_vcvtdq2ps(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
        vcvtdq2ps(x, op);
    }

    void uni_vmovmskps(const Xbyak::Reg& x1, const Xbyak::Xmm& x2) {
        movmskps(x1.cvt64(), x2);
    }

    void uni_vmovmskps(const Xbyak::Reg& x1, const Xbyak::Ymm& x2) {
        vmovmskps(x1, x2);
    }
    template <typename T>
    Xbyak::Address EVEX_compress_addr(Xbyak::Reg64 base,
                                      T raw_offt,
                                      bool bcast = false) {
        using Xbyak::Zmm;
        using Xbyak::Reg64;
        using Xbyak::Address;
        using Xbyak::RegExp;

        auto offt = static_cast<int>(raw_offt);

        int scale = 0;

        if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
            offt = offt - 2 * EVEX_max_8b_offt;
            scale = 1;
        } else if (3 * EVEX_max_8b_offt <= offt && offt < 5 * EVEX_max_8b_offt) {
            offt = offt - 4 * EVEX_max_8b_offt;
            scale = 2;
        }

        auto re = RegExp() + base + offt;

        if (scale) {
            re = re + reg_EVEX_max_8b_offt * scale;
        }

        if (bcast) {
            return zword_b[re];
        } else {
            return zword[re];
        }
    }

    void L(const char* label) {
        Xbyak::CodeGenerator::L(label);
    }
    void L(const Xbyak::Label& label) {
        Xbyak::CodeGenerator::L(label);
    }

    void dump_code(const Xbyak::uint8* code) const {
        if (code) {
            static int counter = 0;
#define MAX_FNAME_LEN 256
            char fname[MAX_FNAME_LEN + 1];
            snprintf(fname, MAX_FNAME_LEN, "jit_dump_%s.%d.bin", name(), counter);
            counter++;

            FILE* fp = fopen(fname, "w+");

            // Failure to dump code is not fatal
            if (fp) {
                fwrite(code, getSize(), 1, fp);
                fclose(fp);
            }
        }

#undef MAX_FNAME_LEN
    }

public:
    static constexpr size_t max_code_size = 256 * 4096;

#ifdef USE_SGX
private:
    struct SGXAllocator : Xbyak::Allocator {
        Xbyak::uint8* const jit_start;
        const size_t meta_size;
        std::unique_ptr<bool[]> meta;

        SGXAllocator(Xbyak::uint8* jit_start, Xbyak::uint8* jit_end)
            : Xbyak::Allocator(), jit_start(jit_start),
              meta_size((jit_end - jit_start) / max_code_size),
              meta(new bool[meta_size]) {
            memset(meta.get(), 0, sizeof(bool) * meta_size);
        }

        Xbyak::uint8* alloc(size_t size) override {
            if (size != max_code_size) {
                abort();
            }

            for (int i = 0; i < meta_size; ++i) {
                if (!meta[i]) {
                    meta[i] = true;
                    return jit_start + i * size;
                }
            }

            abort();
            return nullptr;
        }

        void free(Xbyak::uint8* p) {
            size_t dis = p - jit_start;

            if (dis % max_code_size) {
                abort();
            }

            meta[dis / max_code_size] = false;
        }

        bool useProtect() const override {
            return false;
        }
    };

    static Xbyak::Allocator* get_jit_allocator() {
        static SGXAllocator _allocator(&__jit_start, &__jit_end);
        return &_allocator;
    };
#else
#define get_jit_allocator() nullptr
#endif

public:
    jit_generator()
        : Xbyak::CodeGenerator(max_code_size, nullptr, get_jit_allocator()) {}

    virtual const char* name() const = 0;
    virtual const char* source_file() const = 0;

    // XXX: use normal_case name and update all callees (?)
    const Xbyak::uint8* getCode() {
        const Xbyak::uint8* code = CodeGenerator::getCode();

#ifdef WITH_DUMP_CODE

        // only can dump code when cmake option is enabled
        if (util::env::jit_dump_code()) {
            dump_code(code);
        }

#endif

        return code;
    }

    template <typename F>
    const F getCode() {
        // XXX (Roma): Xbyak code probably has a bug here
        return (const F)getCode();
    }
};

}
}
}
