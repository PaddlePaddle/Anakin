#include "framework/operators/reduce_min.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void ReduceMin<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl =
        static_cast<ReduceMinHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param =
        static_cast<ReduceMinHelper<NV, Precision::FP32>*>(this->_helper)->_param_reduce_min;
    impl->_funcs_reduce_min(ins, outs, param, ctx);
}
#endif

#ifdef USE_X86_PLACE
template<>
void ReduceMin<X86, Precision::FP32>::operator()(
        OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl =
            static_cast<ReduceMinHelper<X86, Precision::FP32>*>(this->_helper);
    auto& param =
            static_cast<ReduceMinHelper<X86, Precision::FP32>*>(this->_helper)->_param_reduce_min;
    impl->_funcs_reduce_min(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
ReduceMinHelper<Ttype, Ptype>::~ReduceMinHelper() {
}

template<typename Ttype, Precision Ptype>
Status ReduceMinHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing ReduceMin op parameter.";
    auto keep_dim = GET_PARAMETER(bool, keep_dim);
    auto reduce_dim = GET_PARAMETER(PTuple<int>, reduce_dim);
    ReduceMinParam<Ttype> param_reduce_min(reduce_dim.vector(), keep_dim);
    _param_reduce_min = param_reduce_min;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ReduceMinHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_reduce_min.init(ins, outs, _param_reduce_min, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ReduceMinHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_reduce_min.compute_output_shape(ins, outs, _param_reduce_min));
    return Status::OK();
}

#ifdef USE_CUDA
template class ReduceMinHelper<NV, Precision::FP32>;
template class ReduceMinHelper<NV, Precision::FP16>;
template class ReduceMinHelper<NV, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class ReduceMinHelper<ARM, Precision::FP32>;
template class ReduceMinHelper<ARM, Precision::FP16>;
template class ReduceMinHelper<ARM, Precision::INT8>;
#endif
#ifdef USE_X86_PLACE
template class ReduceMinHelper<X86, Precision::FP32>;
template class ReduceMinHelper<X86, Precision::FP16>;
template class ReduceMinHelper<X86, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(ReduceMin, ReduceMinHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(ReduceMin, ReduceMinHelper, ARM, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(ReduceMin, ReduceMinHelper, X86, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(ReduceMin)
.Doc("ReduceMin operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("reduce_min")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("reduce_min")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("reduce_min")
#endif
.num_in(1)
.num_out(1)
.Args<int>("groups", " split tensor's channel by size groups. ");

} /* namespace ops */

} /* namespace anakin */


