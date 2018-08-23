#include "saber/core/tensor.h"
#include "saber/core/env.h"

namespace anakin{

namespace saber{

//! target wrapper
template struct TargetWrapper<X86, __host_target>;

//! X86 Buffer
template class Buffer<X86>;

//! X86 Tensor
template class Tensor<X86, AK_FLOAT, NCHW>;
template class Tensor<X86, AK_FLOAT, NHWC>;
template class Tensor<X86, AK_FLOAT, NCHW_C16>;
template class Tensor<X86, AK_FLOAT, HW>;

template class Tensor<X86, AK_INT8, NCHW>;
template class Tensor<X86, AK_INT8, NHWC>;
template class Tensor<X86, AK_INT8, HW>;

template class Tensor<X86, AK_HALF, NCHW>;
template class Tensor<X86, AK_HALF, NHWC>;
template class Tensor<X86, AK_HALF, HW>;

template struct Env<X86>;

} //namespace saber

} //namespace anakin