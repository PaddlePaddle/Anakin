#ifndef BMRUNTIME_INTERFACE_H_
#define BMRUNTIME_INTERFACE_H_

#include "bmruntime.h"
#include "bmdnn_runtime.h"

bmruntime* create_bmruntime(bm_handle_t* bm_handle);

void destroy_bmruntime(bm_handle_t bm_handle, bmruntime* p_bmrt);

#endif
