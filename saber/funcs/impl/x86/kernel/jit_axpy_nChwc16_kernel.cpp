#include "jit_axpy_nChwc16_kernel.h"

#define GET_OFF(field) offsetof(jit_axpy_call_t, field)

// @note: do not use any MACRO or #define inside JIT kernel
// it would have some uncertain issue in JIT, need figure out why

namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

void jit_axpy_nChwc16_kernel::generate() {
    preamble();

    mov(reg_ptr_src, ptr[param + GET_OFF(src)]);
    mov(reg_y, ptr[param + GET_OFF(dst)]);
    mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

    mov(reg_alpha, ptr[reg_ptr_src]);
    mov(reg_x, ptr[reg_ptr_src + sizeof(void*)]);

    // move alpha
    vmovups(ymm_alpha1, ptr[reg_alpha]);
    vmovups(ymm_alpha2, ptr[reg_alpha + 32]);

    Label loop_256;
    Label loop_64;
    Label loop_16;
    Label tail;

    if (jcp_.w * jcp_.h >= 64) {
        L(loop_256);
        {
            cmp(reg_work_amount, 64);
            jl(loop_64, T_NEAR);

            vmovups(ymm_x, ptr[reg_x]);
            vmovups(ymm_y, ptr[reg_y]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 32]);
            vmovups(ymm_y, ptr[reg_y + 32]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 32], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 64]);
            vmovups(ymm_y, ptr[reg_y + 64]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 64], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 96]);
            vmovups(ymm_y, ptr[reg_y + 96]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 96], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 128]);
            vmovups(ymm_y, ptr[reg_y + 128]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 128], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 160]);
            vmovups(ymm_y, ptr[reg_y + 160]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 160], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 192]);
            vmovups(ymm_y, ptr[reg_y + 192]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 192], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 224]);
            vmovups(ymm_y, ptr[reg_y + 224]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 224], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 256]);
            vmovups(ymm_y, ptr[reg_y + 256]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 256], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 288]);
            vmovups(ymm_y, ptr[reg_y + 288]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 288], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 320]);
            vmovups(ymm_y, ptr[reg_y + 320]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 320], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 352]);
            vmovups(ymm_y, ptr[reg_y + 352]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 352], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 384]);
            vmovups(ymm_y, ptr[reg_y + 384]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 384], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 416]);
            vmovups(ymm_y, ptr[reg_y + 416]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 416], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 448]);
            vmovups(ymm_y, ptr[reg_y + 448]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 448], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 480]);
            vmovups(ymm_y, ptr[reg_y + 480]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 480], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 512]);
            vmovups(ymm_y, ptr[reg_y + 512]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 512], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 544]);
            vmovups(ymm_y, ptr[reg_y + 544]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 544], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 576]);
            vmovups(ymm_y, ptr[reg_y + 576]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 576], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 608]);
            vmovups(ymm_y, ptr[reg_y + 608]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 608], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 640]);
            vmovups(ymm_y, ptr[reg_y + 640]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 640], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 672]);
            vmovups(ymm_y, ptr[reg_y + 672]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 672], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 704]);
            vmovups(ymm_y, ptr[reg_y + 704]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 704], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 736]);
            vmovups(ymm_y, ptr[reg_y + 736]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 736], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 768]);
            vmovups(ymm_y, ptr[reg_y + 768]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 768], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 800]);
            vmovups(ymm_y, ptr[reg_y + 800]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 800], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 832]);
            vmovups(ymm_y, ptr[reg_y + 832]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 832], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 864]);
            vmovups(ymm_y, ptr[reg_y + 864]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 864], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 896]);
            vmovups(ymm_y, ptr[reg_y + 896]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 896], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 928]);
            vmovups(ymm_y, ptr[reg_y + 928]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 928], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 960]);
            vmovups(ymm_y, ptr[reg_y + 960]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 960], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 992]);
            vmovups(ymm_y, ptr[reg_y + 992]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 992], ymm_y);

            add(reg_x, 1024);
            add(reg_y, 1024);
            sub(reg_work_amount, 256);
            jmp(loop_256);
        }
    }

    if (jcp_.w * jcp_.h >= 4) {
        L(loop_64);
        {
            cmp(reg_work_amount, 64);
            jl(loop_16, T_NEAR);

            vmovups(ymm_x, ptr[reg_x]);
            vmovups(ymm_y, ptr[reg_y]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 32]);
            vmovups(ymm_y, ptr[reg_y + 32]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 32], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 64]);
            vmovups(ymm_y, ptr[reg_y + 64]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 64], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 96]);
            vmovups(ymm_y, ptr[reg_y + 96]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 96], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 128]);
            vmovups(ymm_y, ptr[reg_y + 128]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 128], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 160]);
            vmovups(ymm_y, ptr[reg_y + 160]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 160], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 192]);
            vmovups(ymm_y, ptr[reg_y + 192]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
            vmovups(ptr[reg_y + 192], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 224]);
            vmovups(ymm_y, ptr[reg_y + 224]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
            vmovups(ptr[reg_y + 224], ymm_y);

            add(reg_x, 256);
            add(reg_y, 256);
            sub(reg_work_amount, 64);
            jmp(loop_64);
        }
    }

    L(loop_16);
    {
        cmp(reg_work_amount, 16);
        jl(tail);

        vmovups(ymm_x, ptr[reg_x]);
        vmovups(ymm_y, ptr[reg_y]);
        vfmadd231ps(ymm_y, ymm_x, ymm_alpha1);
        vmovups(ptr[reg_y], ymm_y);

        vmovups(ymm_x, ptr[reg_x + 32]);
        vmovups(ymm_y, ptr[reg_y + 32]);
        vfmadd231ps(ymm_y, ymm_x, ymm_alpha2);
        vmovups(ptr[reg_y + 32], ymm_y);

        add(reg_x, 64);
        add(reg_y, 64);
        sub(reg_work_amount, 16);
        jmp(loop_16);
    }
    L(tail);
    postamble();
}

} // namespace jit
} // namespace saber
} // namespace anakin
