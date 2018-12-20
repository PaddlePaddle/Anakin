#include "framework/operators/sequence_pool.h"

namespace anakin {

namespace ops {

#define INSTANCE_SEQUENCE_POOL(Ttype, Ptype) \
template<> \
void SequencePool<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<SequencePoolHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SequencePoolHelper<Ttype, Ptype>*>(this->_helper)->_param_sequence_pool; \
    impl->_funcs_sequence_pool(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
SequencePoolHelper<Ttype, Ptype>::~SequencePoolHelper() {
}

template<typename Ttype, Precision Ptype>
Status SequencePoolHelper<Ttype, Ptype>::InitParam() {
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
    saber::SequencePoolParam<Ttype> sequence_pool_param(type_map[pooltype]);
    _param_sequence_pool = sequence_pool_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SequencePoolHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_sequence_pool.init(ins, outs, _param_sequence_pool, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SequencePoolHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_sequence_pool.compute_output_shape(ins, outs, _param_sequence_pool));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SEQUENCE_POOL(NV, Precision::FP32);
template class SequencePoolHelper<NV, Precision::FP32>;
template class SequencePoolHelper<NV, Precision::FP16>;
template class SequencePoolHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(SequencePool, SequencePoolHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_SEQUENCE_POOL(AMD, Precision::FP32);
template class SequencePoolHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequencePool, SequencePoolHelper, AMD, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SEQUENCE_POOL(ARM, Precision::FP32);
template class SequencePoolHelper<ARM, Precision::FP32>;
template class SequencePoolHelper<ARM, Precision::FP16>;
template class SequencePoolHelper<ARM, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(SequencePool, SequencePoolHelper, ARM, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_SEQUENCE_POOL(X86, Precision::FP32);
template class SequencePoolHelper<X86, Precision::FP32>;
template class SequencePoolHelper<X86, Precision::FP16>;
template class SequencePoolHelper<X86, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(SequencePool, SequencePoolHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(SequencePool)
.Doc("SequencePool operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("SequencePool")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("SequencePool")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("SequencePool")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("SequencePool")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("pooltype", " pooltype to compute ");

} /* namespace ops */

} /* namespace anakin */


