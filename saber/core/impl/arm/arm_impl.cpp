#include "saber/core/tensor.h"
#include "saber/core/env.h"

namespace anakin{

namespace saber{

#ifdef USE_ARM_PLACE
//! target wrapper
template struct TargetWrapper<ARM, __host_target>;

//! ARM Buffer
template class Buffer<ARM>;

//! ARM Tensor
#ifdef ANAKIN_TYPE_FP32
template class Tensor<ARM, AK_FLOAT, NCHW>;
//template class Tensor<ARM, AK_FLOAT, NHWC>;
//template class Tensor<ARM, AK_FLOAT, HW>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class Tensor<ARM, AK_INT8, NCHW>;
template class Tensor<ARM, AK_INT8, NHWC>;
template class Tensor<ARM, AK_INT8, HW>;
#endif

#ifdef ANAKIN_TYPE_FP16
template class Tensor<ARM, AK_HALF, NCHW>;
template class Tensor<ARM, AK_HALF, NHWC>;
template class Tensor<ARM, AK_HALF, HW>;
#endif
template class Env<ARM>;

#endif //USE_ARM_PLACE

} //namespace saber

} //namespace anakin