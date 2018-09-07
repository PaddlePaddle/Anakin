
#include "saber/funcs/impl/cuda/vender_deconv.h"

namespace anakin {
namespace saber {


template class VenderDeconv2D<NV, AK_FLOAT>;
DEFINE_OP_TEMPLATE(VenderDeconv2D, ConvParam, NV, AK_INT16);
DEFINE_OP_TEMPLATE(VenderDeconv2D, ConvParam, NV, AK_INT8);
}
}