#ifndef BM_CONV_H
#define BM_CONV_H

#include <stdio.h>
#include "bm_common.h"
#include "atomic_dma_gen_cmd.h"
#include "atomic_conv_gen_cmd.h"
#include "atomic_md_sum_gen_cmd.h"

#ifdef USING_CMODEL
#include "cmodel_runtime.h"
#include "atomic_dma.h"
#include "atomic_conv.h"
#include "atomic_md_sum.h"
#endif
#include "bmkernel_base.h"
int bm_conv_fwd(bm_api_conv_forward *conv_param);
#endif
