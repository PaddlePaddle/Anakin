#include "framework/operators/axpy.h"

namespace anakin {

namespace ops {

//#ifdef USE_CUDA
//template<>
//void Axpy<NV, AK_FLOAT, Precision::FP32>::operator()(
//    OpContext<NV>& ctx,
//    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
//    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
//    auto* impl =
//        static_cast<AxpyHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
//    auto& param = impl->_param_axpy;
//    impl->_funcs_axpy(ins, outs, param, ctx);
//}
//#endif

/// TODO ... specialization other type of operator
#define INSTANCE_AXPY(Ttype, Dtype, Ptype) \
template<> \
void Axpy<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<AxpyHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = impl->_param_axpy; \
    impl->_funcs_axpy(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
AxpyHelper<Ttype, Dtype, Ptype>::~AxpyHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status AxpyHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Axpy op parameter.";

    saber::AxpyParam<Tensor4d<Ttype, Dtype>> axpy_param;
    _param_axpy = axpy_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status AxpyHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_axpy.init(ins, outs, _param_axpy, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status AxpyHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_axpy.compute_output_shape(ins, outs, _param_axpy));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_AXPY(NV, AK_FLOAT, Precision::FP32);
template class AxpyHelper<NV, AK_FLOAT, Precision::FP32>;
template class AxpyHelper<NV, AK_FLOAT, Precision::FP16>;
template class AxpyHelper<NV, AK_FLOAT, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Axpy, AxpyHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
INSTANCE_AXPY(X86, AK_FLOAT, Precision::FP32);
template class AxpyHelper<X86, AK_FLOAT, Precision::FP32>;
template class AxpyHelper<X86, AK_FLOAT, Precision::FP16>;
template class AxpyHelper<X86, AK_FLOAT, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Axpy, AxpyHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE

#ifdef ANAKIN_TYPE_FP32
INSTANCE_AXPY(ARM, AK_FLOAT, Precision::FP32);
template class AxpyHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Axpy, AxpyHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef ANAKIN_TYPE_FP16
template class AxpyHelper<ARM, AK_FLOAT, Precision::FP16>;
#endif

#ifdef ANAKIN_TYPE_INT8
template class AxpyHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

#endif//arm

//! register op
ANAKIN_REGISTER_OP(Axpy)
.Doc("Axpy operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("axpy")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("axpy")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, AK_FLOAT, Precision::FP32>("axpy")
#endif
.num_in(3)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


