#include "framework/operators/concat.h"

namespace anakin {

namespace ops {

//#ifdef USE_CUDA
//template<>
//void Concat<NV, AK_FLOAT, Precision::FP32>::operator()(
//    OpContext<NV>& ctx,
//    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
//    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
//    auto* impl = static_cast<ConcatHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
//    auto& param = static_cast<ConcatHelper<NV, AK_FLOAT, Precision::FP32>*>
//                  (this->_helper)->_param_concat;
//    impl->_funcs_concat(ins, outs, param, ctx);
//}
//#endif
//#ifdef USE_X86_PLACE
//template<>
//void Concat<X86, AK_FLOAT, Precision::FP32>::operator()(
//        OpContext<X86>& ctx,
//        const std::vector<Tensor4dPtr<X86, AK_FLOAT> >& ins,
//        std::vector<Tensor4dPtr<X86, AK_FLOAT> >& outs) {
//    auto* impl = static_cast<ConcatHelper<X86, AK_FLOAT, Precision::FP32>*>(this->_helper);
//    auto& param = static_cast<ConcatHelper<X86, AK_FLOAT, Precision::FP32>*>
//    (this->_helper)->_param_concat;
//    impl->_funcs_concat(ins, outs, param, ctx);
//}
//#endif

/// TODO ... specialization other type of operator
#define INSTANCE_CONCAT(Ttype, Dtype, Ptype) \
template<> \
void Concat<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
                std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = static_cast<ConcatHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ConcatHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_concat; \
    impl->_funcs_concat(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
ConcatHelper<Ttype, Dtype, Ptype>::~ConcatHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ConcatHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Concat op parameter.";
    auto axis = GET_PARAMETER(int, axis);
    ConcatParam<Tensor4d<Ttype, Dtype>> param_concat(axis);
    _param_concat = param_concat;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ConcatHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_concat.init(ins, outs, _param_concat, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ConcatHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_concat.compute_output_shape(ins, outs, _param_concat));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_CONCAT(NV, AK_FLOAT, Precision::FP32);
template class ConcatHelper<NV, AK_FLOAT, Precision::FP32>;
template class ConcatHelper<NV, AK_FLOAT, Precision::FP16>;
template class ConcatHelper<NV, AK_FLOAT, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE

#ifdef ANAKIN_TYPE_FP32
INSTANCE_CONCAT(ARM, AK_FLOAT, Precision::FP32);
template class ConcatHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef ANAKIN_TYPE_FP16
template class ConcatHelper<ARM, AK_FLOAT, Precision::FP16>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class ConcatHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

#endif //arm

#ifdef USE_X86_PLACE
INSTANCE_CONCAT(X86, AK_FLOAT, Precision::FP32);
template class ConcatHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, X86, AK_FLOAT, Precision::FP32);
template class ConcatHelper<X86, AK_FLOAT, Precision::FP16>;
template class ConcatHelper<X86, AK_FLOAT, Precision::INT8>;
#endif


//! register op
ANAKIN_REGISTER_OP(Concat)
.Doc("Concat operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("concat")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("concat")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("concat")
#endif
.num_in(2)
.num_out(1)
.Args<int>("axis", " axis for concat the input ");

} /* namespace ops */

} /* namespace anakin */


