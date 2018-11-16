#ifndef ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AXPY_NCHWC16_KERNEL_H
#define ANAKIN_SABER_FUNCS_IMPL_X86_KERMEL_JIT_AXPY_NCHWC16_KERNEL_H

#include "saber/funcs/impl/x86/kernel/jit_call_conf.h"
#include "jit_generator.h"

namespace anakin {
namespace saber {
namespace jit {

struct jit_axpy_nChwc16_kernel : public jit_generator {
public:
    jit_axpy_nChwc16_kernel(jit_axpy_conf_t ajcp) : jcp_(ajcp) {
        generate();
        jit_ker_ = (void (*)(jit_axpy_call_t*))getCode();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_axpy_nChwc16_kernel);

    jit_axpy_conf_t jcp_;
    void (*jit_ker_)(jit_axpy_call_t*);

private:
    enum {
        USE_ZMM = 512,
        USE_YMM = 256,
        USE_XMM = 128,
    };

    using reg64_t = const Xbyak::Reg64;
    using zmm_t = const Xbyak::Zmm;
    using ymm_t = const Xbyak::Ymm;
    using xmm_t = const Xbyak::Xmm;

    reg64_t param = abi_param1;
    reg64_t reg_ptr_src = r8;
    reg64_t reg_work_amount = r9;
    reg64_t reg_x = r12;
    reg64_t reg_y = r13;
    reg64_t reg_alpha = r14;

    zmm_t zmm_y = zmm_t(15);
    ymm_t ymm_y = ymm_t(15);
    xmm_t xmm_y = xmm_t(15);

    zmm_t zmm_x = zmm_t(14);
    ymm_t ymm_x = ymm_t(14);
    xmm_t xmm_x = xmm_t(14);

    zmm_t zmm_alpha1 = zmm_t(13);
    ymm_t ymm_alpha1 = ymm_t(13);
    xmm_t xmm_alpha1 = xmm_t(13);

    zmm_t zmm_alpha2 = zmm_t(12);
    ymm_t ymm_alpha2 = ymm_t(12);
    xmm_t xmm_alpha2 = xmm_t(12);

    void generate();
};

} // namespace jit
} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_X86_KERNEL_JIT_AXPY_NCHWC16_KERNEL_H