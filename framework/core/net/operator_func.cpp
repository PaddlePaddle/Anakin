#include "framework/core/net/operator_func.h"

namespace anakin {

template<typename Ttype, Precision Ptype>
void OperatorFunc<Ttype, Ptype>::launch() {
    (*op)(*ctx_p, ins, outs);
}

template<typename Ttype, Precision Ptype>
void OperatorFunc<Ttype, Ptype>::infer_shape() {
    op->_helper->InferShape(ins, outs);
}

#ifdef USE_CUDA
template class OperatorFunc<NV, Precision::FP32>;
template class OperatorFunc<NV, Precision::FP16>;
template class OperatorFunc<NV, Precision::INT8>;
#endif

#ifdef USE_X86_PLACE
template class OperatorFunc<X86, Precision::FP32>;
template class OperatorFunc<X86, Precision::FP16>;
template class OperatorFunc<X86, Precision::INT8>;
#endif

#ifdef AMD_GPU 
template class OperatorFunc<AMD, Precision::FP32>;
template class OperatorFunc<AMD, Precision::FP16>;
template class OperatorFunc<AMD, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
template class OperatorFunc<ARM, Precision::FP32>;
#endif

#ifdef ANAKIN_TYPE_FP16
template class OperatorFunc<ARM, Precision::FP16>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class OperatorFunc<ARM, Precision::INT8>;
#endif

#endif //arm
} /* namespace */

