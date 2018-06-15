#ifndef BMDNN_RUNTIME_H_
#define BMDNN_RUNTIME_H_

#include "bmlib_runtime.h"

#if defined (__cplusplus)
extern "C" {
#endif

bm_status_t bmdnn_init(
    bm_handle_t     *handle);

void bmdnn_deinit(
    bm_handle_t      handle);

#if defined (__cplusplus)
}
#endif

#endif
