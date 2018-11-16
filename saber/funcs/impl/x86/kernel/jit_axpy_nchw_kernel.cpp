#include "jit_axpy_nchw_kernel.h"

#define GET_OFF(field) offsetof(jit_axpy_call_t, field)

// @note: do not use any MACRO or #define inside JIT kernel
// it would have some uncertain issue in JIT, need figure out why

namespace anakin {
namespace saber {
namespace jit {

using namespace Xbyak;

void jit_axpy_nchw_kernel::generate() {
    preamble();

    mov(reg_ptr_src, ptr[param + GET_OFF(src)]);
    mov(reg_y, ptr[param + GET_OFF(dst)]);
    mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

    mov(reg_alpha, ptr[reg_ptr_src]);
    mov(reg_x, ptr[reg_ptr_src + sizeof(void*)]);

    // broadcast alpha
    vbroadcastss(ymm_alpha, ptr[reg_alpha]);

    Label loop_128;
    Label loop_32;
    Label loop_16;
    Label loop_8;
    Label loop_pst;
    Label tail;

    if (jcp_.w * jcp_.h >= 128) {
        L(loop_128);
        {
            cmp(reg_work_amount, 128);
            jl(loop_32, T_NEAR);

            vmovups(ymm_x, ptr[reg_x]);
            vmovups(ymm_y, ptr[reg_y]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 32]);
            vmovups(ymm_y, ptr[reg_y + 32]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 32], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 64]);
            vmovups(ymm_y, ptr[reg_y + 64]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 64], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 96]);
            vmovups(ymm_y, ptr[reg_y + 96]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 96], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 128]);
            vmovups(ymm_y, ptr[reg_y + 128]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 128], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 160]);
            vmovups(ymm_y, ptr[reg_y + 160]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 160], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 192]);
            vmovups(ymm_y, ptr[reg_y + 192]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 192], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 224]);
            vmovups(ymm_y, ptr[reg_y + 224]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 224], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 256]);
            vmovups(ymm_y, ptr[reg_y + 256]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 256], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 288]);
            vmovups(ymm_y, ptr[reg_y + 288]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 288], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 320]);
            vmovups(ymm_y, ptr[reg_y + 320]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 320], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 352]);
            vmovups(ymm_y, ptr[reg_y + 352]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 352], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 384]);
            vmovups(ymm_y, ptr[reg_y + 384]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 384], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 416]);
            vmovups(ymm_y, ptr[reg_y + 416]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 416], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 448]);
            vmovups(ymm_y, ptr[reg_y + 448]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 448], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 480]);
            vmovups(ymm_y, ptr[reg_y + 480]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 480], ymm_y);

            add(reg_x, 512);
            add(reg_y, 512);
            sub(reg_work_amount, 128);
            jmp(loop_128);
        }
    }

    if (jcp_.w * jcp_.h >= 32) {
        L(loop_32);
        {
            cmp(reg_work_amount, 32);
            jl(loop_8, T_NEAR);

            vmovups(ymm_x, ptr[reg_x]);
            vmovups(ymm_y, ptr[reg_y]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 32]);
            vmovups(ymm_y, ptr[reg_y + 32]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 32], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 64]);
            vmovups(ymm_y, ptr[reg_y + 64]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 64], ymm_y);

            vmovups(ymm_x, ptr[reg_x + 96]);
            vmovups(ymm_y, ptr[reg_y + 96]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y + 96], ymm_y);

            add(reg_x, 128);
            add(reg_y, 128);
            sub(reg_work_amount, 32);
            jmp(loop_32);
        }
    }

    if (jcp_.w * jcp_.h >= 8) {
        L(loop_8);
        {
            cmp(reg_work_amount, 8);
            jl(loop_pst);

            vmovups(ymm_x, ptr[reg_x]);
            vmovups(ymm_y, ptr[reg_y]);
            vfmadd231ps(ymm_y, ymm_x, ymm_alpha);
            vmovups(ptr[reg_y], ymm_y);

            add(reg_x, 32);
            add(reg_y, 32);
            sub(reg_work_amount, 8);
            jmp(loop_8);
        }
    }

    L(loop_pst);
    {
        cmp(reg_work_amount, 0);
        jle(tail);

        vmovss(xmm_x, dword[reg_x]);
        vmovss(xmm_y, dword[reg_y]);
        vfmadd231ps(xmm_y, xmm_x, xmm_alpha);
        vmovss(dword[reg_y], xmm_y);

        add(reg_x, 4);
        add(reg_y, 4);
        sub(reg_work_amount, 1);
        jmp(loop_pst);
    }

    L(tail);
    postamble();
}

} // namespace jit
} // namespace saber
} // namespace anakin
