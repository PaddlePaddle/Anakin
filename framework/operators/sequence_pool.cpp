#include "framework/operators/sequence_pool.h"

namespace anakin {

namespace ops {

#ifdef USE_X86_PLACE
template<>
void SequencePool<X86, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<X86>& ctx,
    const std::vector<Tensor4dPtr<X86, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<X86, AK_FLOAT> >& outs) {
    auto* impl = static_cast<SequencePoolHelper<X86, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<SequencePoolHelper<X86, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_sequence_pool;
    impl->_funcs_sequence_pool(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
SequencePoolHelper<Ttype, Dtype, Ptype>::~SequencePoolHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SequencePoolHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing SequencePool op parameter.";
    auto pooltype = GET_PARAMETER(std::string, pooltype);
    std::unordered_map<std::string, SequencePoolType> type_map;
    type_map.insert(std::make_pair("null", anakin::saber::Sequence_pool_unknow));
    type_map.insert(std::make_pair("AVERAGE", anakin::saber::Sequence_pool_average));
    type_map.insert(std::make_pair("SUM", anakin::saber::Sequence_pool_sum));
    type_map.insert(std::make_pair("SQRT", anakin::saber::Sequence_pool_sqrt));
    type_map.insert(std::make_pair("LAST", anakin::saber::Sequence_pool_last));
    type_map.insert(std::make_pair("FIRST", anakin::saber::Sequence_pool_first));
    type_map.insert(std::make_pair("MAX", anakin::saber::Sequence_pool_max));
    saber::SequencePoolParam<Tensor4d<Ttype, Dtype>> sequence_pool_param(type_map[pooltype]);
    _param_sequence_pool = sequence_pool_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SequencePoolHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_sequence_pool.init(ins, outs, _param_sequence_pool, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SequencePoolHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_sequence_pool.compute_output_shape(ins, outs, _param_sequence_pool));
    return Status::OK();
}

#ifdef USE_CUDA
template class SequencePoolHelper<NV, AK_FLOAT, Precision::FP32>;
template class SequencePoolHelper<NV, AK_FLOAT, Precision::FP16>;
template class SequencePoolHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class SequencePoolHelper<ARM, AK_FLOAT, Precision::FP32>;
template class SequencePoolHelper<ARM, AK_FLOAT, Precision::FP16>;
template class SequencePoolHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_X86_PLACE
template class SequencePoolHelper<X86, AK_FLOAT, Precision::FP32>;
template class SequencePoolHelper<X86, AK_FLOAT, Precision::FP16>;
template class SequencePoolHelper<X86, AK_FLOAT, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(SequencePool, SequencePoolHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(SequencePool, SequencePoolHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
ANAKIN_REGISTER_OP_HELPER(SequencePool, SequencePoolHelper, X86, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(SequencePool)
.Doc("SequencePool operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("SequencePool")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("SequencePool")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("SequencePool")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("pooltype", " pooltype to compute ");

} /* namespace ops */

} /* namespace anakin */


