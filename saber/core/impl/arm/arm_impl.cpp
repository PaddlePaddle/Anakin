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
template class Tensor<ARM>;

//! ARM Env
template class Env<ARM>;

#endif //USE_ARM_PLACE

} //namespace saber

} //namespace anakin