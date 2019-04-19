#include <sgx_trts_exception.h>
#include "cpuid.h"
#include "stdio.h"
#include "stdlib.h"

#if defined(_M_X64) || defined(__x86_64__)
#define REG(INFO, REG)   ((INFO)->r##REG)
#define RD_REG32(INFO, REG)     static_cast<uint32_t>(0xFFFFFFFFLLU & ((INFO)->r##REG))
#define WR_REG32_O(INFO, REG)   ((INFO)->r##REG)
#else
#define REG(INFO, REG)   ((INFO)->e##REG)
#define RD_REG32(INFO, REG)     ((INFO)->e##REG)
#define WR_REG32_O(INFO, REG)   RD_REG32(INFO, REG)
#endif

static int illegal_inst_handler(sgx_exception_info_t *info) {
    static constexpr uint16_t cpuid_inst = 0xa20f;

    if (info->exception_vector != SGX_EXCEPTION_VECTOR_UD)
        return EXCEPTION_CONTINUE_SEARCH;

    auto *cpu_ctx = &info->cpu_context;
    if (*reinterpret_cast<uint16_t *>(REG(cpu_ctx, ip)) == cpuid_inst) {
        __cpuid_count(RD_REG32(cpu_ctx, ax), RD_REG32(cpu_ctx, cx),
                      REG(cpu_ctx, ax), REG(cpu_ctx, bx),
                      REG(cpu_ctx, cx), REG(cpu_ctx, dx));

        REG(cpu_ctx, ip) += 2;

        return EXCEPTION_CONTINUE_EXECUTION;
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

static int anakin_enclave_init() {
    if (!sgx_register_exception_handler(true, illegal_inst_handler)) {
        abort();
    }

    return 0;
}

extern "C" const int __anakin_enclave_init_status = anakin_enclave_init();
