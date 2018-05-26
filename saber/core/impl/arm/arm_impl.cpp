#include "core/tensor.h"
#include "core/env.h"

namespace anakin{

namespace saber{

#ifdef USE_ARM_PLACE
//! target wrapper
template struct TargetWrapper<ARM, __host_target>;

//! ARM Buffer
template class Buffer<ARM>;

//! ARM Tensor
template class Tensor<ARM, AK_FLOAT, NCHW>;
template class Tensor<ARM, AK_FLOAT, NHWC>;
template class Tensor<ARM, AK_FLOAT, HW>;

template class Tensor<ARM, AK_INT8, NCHW>;
template class Tensor<ARM, AK_INT8, NHWC>;
template class Tensor<ARM, AK_INT8, HW>;

template class Tensor<ARM, AK_HALF, NCHW>;
template class Tensor<ARM, AK_HALF, NHWC>;
template class Tensor<ARM, AK_HALF, HW>;

template class Env<ARM>;

#endif //USE_ARM_PLACE

} //namespace saber

} //namespace anakin