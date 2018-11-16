#include "core/tensor.h"
#include "env.h"

namespace anakin{

namespace saber{

//! target wrapper
template struct TargetWrapper<X86, __host_target>;

//! X86 Buffer
template class Buffer<X86>;

//! X86 Tensor
template class Tensor<X86>;

template struct Env<X86>;

} //namespace saber

} //namespace anakin