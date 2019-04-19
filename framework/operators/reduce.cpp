
#include "framework/operators/reduce.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Reduce<NV, Precision::FP32>::operator()(
        OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV> >& ins,
        std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<ReduceHelper<NV, Precision::FP32>*>(
            this->_helper);
    auto& param = static_cast<ReduceHelper<NV, Precision::FP32>*>(
            this->_helper)->_param_reduce;
    impl->_funcs_reduce(ins, outs, param, ctx);
}
#endif

#ifdef USE_X86_PLACE
template<>
void Reduce<X86, Precision::FP32>::operator()(
        OpContext<X86>& ctx,
        const std::vector<Tensor4dPtr<X86> >& ins,
        std::vector<Tensor4dPtr<X86> >& outs) {
    auto* impl = static_cast<ReduceHelper<X86, Precision::FP32>*>(
            this->_helper);
    auto& param = static_cast<ReduceHelper<X86, Precision::FP32>*>(
            this->_helper)->_param_reduce;
    impl->_funcs_reduce(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator

/// set helper
template<typename Ttype, Precision Ptype>
ReduceHelper<Ttype, Ptype>::~ReduceHelper() {
}

template<typename Ttype, Precision Ptype>
Status ReduceHelper<Ttype, Ptype>::InitParam() {
            DLOG(WARNING) << "Parsing Reduce op parameter.";
    auto type_str = GET_PARAMETER(std::string, reduce_type);
    ReduceType type = Reduce_unknow;
    if (type_str == "Reduce_min") {
        type = Reduce_min;
    } else if (type_str == "Reduce_max") {
        type = Reduce_max;
    } else if (type_str == "Reduce_sum") {
        type = Reduce_sum;
    } else if (type_str == "Reduce_avg") {
        type = Reduce_avg;
    } else if (type_str == "Reduce_prod") {
        type = Reduce_prod;
    }
    auto keep_dim = GET_PARAMETER(bool, keep_dim);
    auto reduce_all = GET_PARAMETER(bool, reduce_all);
    auto reduce_dim = GET_PARAMETER(PTuple<int>, reduce_dim);
    auto coeff = GET_PARAMETER(float, coeff);
    ReduceParam<Ttype> param_reduce(reduce_dim.vector(),
            type, keep_dim, reduce_all, coeff);

    _param_reduce = param_reduce;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ReduceHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    SABER_CHECK(_funcs_reduce.init(ins, outs, _param_reduce,
            SPECIFY, SABER_IMPL, ctx));

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ReduceHelper<Ttype, Ptype>::InferShape(
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    SABER_CHECK(_funcs_reduce.compute_output_shape(ins, outs, _param_reduce));
    return Status::OK();
}

#ifdef USE_CUDA
template class ReduceHelper<NV, Precision::FP32>;
template class ReduceHelper<NV, Precision::FP16>;
template class ReduceHelper<NV, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
// template class ReduceHelper<ARM, Precision::FP32>;
// template class ReduceHelper<ARM, Precision::FP16>;
// template class ReduceHelper<ARM, Precision::INT8>;
#endif
#ifdef USE_X86_PLACE
template class ReduceHelper<X86, Precision::FP32>;
template class ReduceHelper<X86, Precision::FP16>;
template class ReduceHelper<X86, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Reduce, ReduceHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Reduce, ReduceHelper, ARM, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(Reduce, ReduceHelper, X86, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Reduce)
.Doc("Reduce operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("reduce")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("reduce")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("reduce")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("reduce_dim", "ratios of gen_anchor_param")
.Args<bool>("keep_dim", "ratios of gen_anchor_param")
.Args<std::string>("reduce_type", "ratios of gen_anchor_param")
.Args<bool>("reduce_all", "ratios of gen_anchor_param")
.Args<float>("coeff", "ratios of gen_anchor_param");

} /* namespace ops */

} /* namespace anakin */


