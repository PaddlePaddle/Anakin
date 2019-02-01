#include "framework/operators/arithmetic.h"

namespace anakin {

namespace ops {

#define INSTANCE_ARITHMETIC(Ttype, Ptype) \
template<> \
void Arithmetic<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<ArithmeticHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ArithmeticHelper<Ttype, Ptype>*>(this->_helper)->_param_arithmetic; \
    impl->_funcs_arithmetic(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
ArithmeticHelper<Ttype, Ptype>::~ArithmeticHelper() {
}

template<typename Ttype, Precision Ptype>
Status ArithmeticHelper<Ttype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing Arithmetic op parameter.";
    auto type = GET_PARAMETER(int, op_type);
    if (type <= 3) {
        ArithmeticParam<Ttype> param_arithmetic(ArithmeticType(type-1));
        _param_arithmetic = param_arithmetic;
    } else {
        LOG(FATAL) << "Other Arithmetic type" << type << " should be replace by other ops.";
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ArithmeticHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_arithmetic.init(ins, outs, _param_arithmetic, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ArithmeticHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins, 
                                                  std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_arithmetic.compute_output_shape(ins, outs, _param_arithmetic));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ARITHMETIC(NV, Precision::FP32);

template<>
Status ArithmeticHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx, 
                                                   const std::vector< Tensor4dPtr<NV> > & ins, 
                                                   std::vector< Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_arithmetic.init(ins, outs, _param_arithmetic, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(Arithmetic, ArithmeticHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_ARITHMETIC(X86, Precision::FP32);
INSTANCE_ARITHMETIC(X86, Precision::FP16);
INSTANCE_ARITHMETIC(X86, Precision::INT8);
template class ArithmeticHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Arithmetic, ArithmeticHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_ARITHMETIC(ARM, Precision::FP32);
template class ArithmeticHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Arithmetic, ArithmeticHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_ARITHMETIC(AMD, Precision::FP32);
template class ArithmeticHelper<AMD, Precision::FP32>;
template class ArithmeticHelper<AMD, Precision::FP16>;
template class ArithmeticHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Arithmetic, ArithmeticHelper, AMD, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Arithmetic)
.Doc("Arithmetic operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("arithmetic")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("arithmetic")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("arithmetic")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("arithmetic")
#endif
.num_in(2)
.num_out(1)
.Args<std::string>("op_type", " type of Arithmetic ");

} /* namespace ops */

} /* namespace anakin */

