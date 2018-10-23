#include "bmkernel_base.h"
#include "bm_config.h"
#include <stdio.h>
/**
 * bmkernel_func is the user entry to BMKERNEL just like "main" to some applications.
 * 
 * \param args - Pointer to arguments that user sends from host.
 *               op - Flag to determine the operation type.             
 */

int bmkernel_func(void *args)
{
    bmkernel_api_base* param = (bmkernel_api_base *)args;
    switch (param->op) {
        case ACTIVATION: {
            // bm_activation_fwd(param)
            return 0;
        }
        case CONV: {
            // bm_api_conv_forward* api = (bm_api_conv_forward *)param->opParam;
            return 0;
        }
        default: {
            printf("op %d is not supported by BM yet.\n", param->op);
            return -1;
        }
    }
}
