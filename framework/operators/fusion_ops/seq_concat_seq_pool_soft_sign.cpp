#include "framework/operators/fusion_ops/seq_concat_seq_pool_soft_sign.h"

namespace anakin {

namespace ops {
#define INSTANCE_SEQ_CONCAT_SEQ_POOL_SOFT_SIGN(Ttype, Ptype) \
template<> \
void SeqConcatSeqPoolSoftSign<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<SeqConcatSeqPoolSoftSignHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SeqConcatSeqPoolSoftSignHelper<Ttype, Ptype>*>(this->_helper)->_param_seq_concat_seq_pool_soft_sign; \
    impl->_funcs_seq_concat_seq_pool_soft_sign(ins, outs, param, ctx); \
}


/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
SeqConcatSeqPoolSoftSignHelper<Ttype, Ptype>::~SeqConcatSeqPoolSoftSignHelper() {
    LOG(INFO) << "Decons permute_cpu_float";
}

template<typename Ttype, Precision Ptype>
Status SeqConcatSeqPoolSoftSignHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing SeqConcatSeqPoolSoftSign op parameter.";
    auto pooltype = GET_PARAMETER(std::string, seq_pool_0_pooltype);
    std::unordered_map<std::string, SequencePoolType> type_map;
    type_map.insert(std::make_pair("null", anakin::saber::Sequence_pool_unknow));
    type_map.insert(std::make_pair("AVERAGE", anakin::saber::Sequence_pool_average));
    type_map.insert(std::make_pair("SUM", anakin::saber::Sequence_pool_sum));
    type_map.insert(std::make_pair("SQRT", anakin::saber::Sequence_pool_sqrt));
    type_map.insert(std::make_pair("LAST", anakin::saber::Sequence_pool_last));
    type_map.insert(std::make_pair("FIRST", anakin::saber::Sequence_pool_first));
    type_map.insert(std::make_pair("MAX", anakin::saber::Sequence_pool_max));

    saber::SequenceConcatParam<Ttype> seq_concat_param;
    saber::SequencePoolParam<Ttype> seq_pool_param(type_map[pooltype]);
    saber::SoftSignParam<Ttype> soft_sign_param;

    saber::SeqConcatSeqPoolSoftSignParam<Ttype> seq_concat_seq_pool_soft_sign_param(seq_concat_param, seq_pool_param, soft_sign_param);
    _param_seq_concat_seq_pool_soft_sign = seq_concat_seq_pool_soft_sign_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SeqConcatSeqPoolSoftSignHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_seq_concat_seq_pool_soft_sign.init(ins, outs, _param_seq_concat_seq_pool_soft_sign, SPECIFY, SABER_IMPL, ctx);
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SeqConcatSeqPoolSoftSignHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_seq_concat_seq_pool_soft_sign.compute_output_shape(ins, outs, _param_seq_concat_seq_pool_soft_sign);
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SEQ_CONCAT_SEQ_POOL_SOFT_SIGN(NV, Precision::FP32);
template class SeqConcatSeqPoolSoftSignHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SeqConcatSeqPoolSoftSign, SeqConcatSeqPoolSoftSignHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_SEQ_CONCAT_SEQ_POOL_SOFT_SIGN(X86, Precision::FP32);
template class SeqConcatSeqPoolSoftSignHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SeqConcatSeqPoolSoftSign, SeqConcatSeqPoolSoftSignHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SEQ_CONCAT_SEQ_POOL_SOFT_SIGN(ARM, Precision::FP32);
template class SeqConcatSeqPoolSoftSignHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SeqConcatSeqPoolSoftSign, SeqConcatSeqPoolSoftSignHelper, ARM, Precision::FP32);
#endif//arm

#ifdef AMD_GPU
INSTANCE_SEQ_CONCAT_SEQ_POOL_SOFT_SIGN(AMD, Precision::FP32);
template class SeqConcatSeqPoolSoftSignHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SeqConcatSeqPoolSoftSign, SeqConcatSeqPoolSoftSignHelper, AMD, Precision::FP32);
#endif


//! register op
ANAKIN_REGISTER_OP(SeqConcatSeqPoolSoftSign)
.Doc("SeqConcatSeqPoolSoftSign fusion operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("seq_concat_seq_pool_soft_sign")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("seq_concat_seq_pool_soft_sign")
#endif
.num_in(1)
.num_out(1)
.Args<float>("pooltype", " sequence pool type");

} /* namespace ops */

} /* namespace anakin */


