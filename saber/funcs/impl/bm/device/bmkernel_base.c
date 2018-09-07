#include "bmkernel_base.h"
#include "bm_config.h"
#include "core/common.h"
#include "core/tensor.h"
#include <stdio.h>
/**
 * bmkernel_func is the user entry to BMKERNEL just like "main" to some applications.
 * 
 * \param args - Pointer to arguments that user sends from host.
 *               op - Flag to determine which op forward function 
 *                    it should delegate to.             
 */

namespace anakin {
namespace saber {

int bmkernel_func(void *args)
{
    bmkernel_api_base* param = (bmkernel_api_base *)args;
    switch (param->op) {
        case "activation":
            // bm_activation_fwd(param)
            break;
        case "conv":
            // bm_conv_fwd(param)
            break;
        default:
            LOG(FATAL) << "op is not supported by BM yet.";
    }
    return 0;
}

}
}
