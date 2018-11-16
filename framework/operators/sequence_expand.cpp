#include "framework/operators/sequence_expand.h"

namespace anakin {

namespace ops {

#define INSTANCE_SEQUENCE_EXPAND(Ttype, Ptype) \
template<> \
void SequenceExpand<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<SequenceExpandHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SequenceExpandHelper<Ttype, Ptype>*>(this->_helper)->_param_sequence_expand; \
    impl->_funcs_sequence_expand(ins, outs, param, ctx); \
}

/// TODO ... specialization other type of operator

/// set helper
template<typename Ttype, Precision Ptype>
SequenceExpandHelper<Ttype, Ptype>::~SequenceExpandHelper() {
}

template<typename Ttype, Precision Ptype>
Status SequenceExpandHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing SequenceExpand op parameter.";
    auto ref_level = GET_PARAMETER(int, ref_level);
    SequenceExpandParam<Ttype> param_sequence_expand(ref_level);
    _param_sequence_expand = param_sequence_expand;

    return Status::OK();
}

template<typename Ttype,  Precision Ptype>
Status SequenceExpandHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_sequence_expand.init(ins, outs, _param_sequence_expand, SPECIFY, SABER_IMPL,
                                            ctx));
    return Status::OK();
}

template<typename Ttype,  Precision Ptype>
Status SequenceExpandHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_sequence_expand.compute_output_shape(ins, outs, _param_sequence_expand));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SEQUENCE_EXPAND(NV, Precision::FP32);
template<>
Status SequenceExpandHelper<NV, Precision::FP32>::Init(OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV> >& ins,
        std::vector<Tensor4dPtr<NV> >& outs) {
    SABER_CHECK(_funcs_sequence_expand.init(ins, outs, _param_sequence_expand, STATIC, VENDER_IMPL,
                                            ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(SequenceExpand, SequenceExpandHelper, NV, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_SEQUENCE_EXPAND(X86,  Precision::FP32);
INSTANCE_SEQUENCE_EXPAND(X86,  Precision::FP16);
INSTANCE_SEQUENCE_EXPAND(X86,  Precision::INT8);
template class SequenceExpandHelper<X86,  Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequenceExpand, SequenceExpandHelper, X86,  Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SEQUENCE_EXPAND(ARM,  Precision::FP32);
template class SequenceExpandHelper<ARM,  Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequenceExpand, SequenceExpandHelper, ARM,  Precision::FP32);
#endif//arm

//! register op
ANAKIN_REGISTER_OP(SequenceExpand)
.Doc("SequenceExpand operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("sequence_expand")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("sequence_expand")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("sequence_expand")
#endif
.num_in(2)
.num_out(1)
.Args<int>("ref_level", "ref level must be 0");

} /* namespace ops */

} /* namespace anakin */