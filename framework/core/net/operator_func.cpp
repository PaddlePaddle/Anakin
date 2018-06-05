#include "framework/core/net/operator_func.h"

namespace anakin {

template<typename Ttype, DataType Dtype, Precision Ptype>
void OperatorFunc<Ttype, Dtype, Ptype>::launch() {
    (*op)(*ctx_p, ins, outs);
}

template<typename Ttype, DataType Dtype, Precision Ptype>
void OperatorFunc<Ttype, Dtype, Ptype>::infer_shape() {
    op->_helper->InferShape(ins, outs);
}

#ifdef USE_CUDA
template class OperatorFunc<NV, AK_FLOAT, Precision::FP32>;
template class OperatorFunc<NV, AK_FLOAT, Precision::FP16>;
template class OperatorFunc<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_X86_PLACE
template class OperatorFunc<X86, AK_FLOAT, Precision::FP32>;
template class OperatorFunc<X86, AK_FLOAT, Precision::FP16>;
template class OperatorFunc<X86, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class OperatorFunc<ARM, AK_FLOAT, Precision::FP32>;
template class OperatorFunc<ARM, AK_FLOAT, Precision::FP16>;
template class OperatorFunc<ARM, AK_FLOAT, Precision::INT8>;
#endif

} /* namespace */

