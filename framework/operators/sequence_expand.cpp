#include "framework/operators/sequence_expand.h"

namespace anakin {

namespace ops {

#define INSTANCE_SEQUENCE_EXPAND(Ttype, Dtype, Ptype) \
template<> \
void SequenceExpand<Ttype, Dtype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<SequenceExpandHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<SequenceExpandHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_sequence_expand; \
    impl->_funcs_sequence_expand(ins, outs, param, ctx); \
}

/// TODO ... specialization other type of operator

/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
SequenceExpandHelper<Ttype, Dtype, Ptype>::~SequenceExpandHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SequenceExpandHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing SequenceExpand op parameter.";
    auto ref_level = GET_PARAMETER(int, ref_level);
    SequenceExpandParam<Tensor4d<Ttype, Dtype>> param_sequence_expand(ref_level);
    _param_sequence_expand = param_sequence_expand;

    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SequenceExpandHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_sequence_expand.init(ins, outs, _param_sequence_expand, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status SequenceExpandHelper<Ttype, Dtype, Ptype>::InferShape(const
                                                         std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                                                         std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_sequence_expand.compute_output_shape(ins, outs, _param_sequence_expand));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SEQUENCE_EXPAND(NV, AK_FLOAT, Precision::FP32);
template<>
Status SequenceExpandHelper<NV, AK_FLOAT, Precision::FP32>::Init(OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
        std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    SABER_CHECK(_funcs_sequence_expand.init(ins, outs, _param_sequence_expand, STATIC, VENDER_IMPL, ctx));
    return Status::OK();
}
ANAKIN_REGISTER_OP_HELPER(SequenceExpand, SequenceExpandHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_SEQUENCE_EXPAND(X86, AK_FLOAT, Precision::FP32);
INSTANCE_SEQUENCE_EXPAND(X86, AK_FLOAT, Precision::FP16);
INSTANCE_SEQUENCE_EXPAND(X86, AK_FLOAT, Precision::INT8);
template class SequenceExpandHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequenceExpand, SequenceExpandHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SEQUENCE_EXPAND(ARM, AK_FLOAT, Precision::FP32);
template class SequenceExpandHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(SequenceExpand, SequenceExpandHelper, ARM, AK_FLOAT, Precision::FP32);
#endif//arm

//! register op
ANAKIN_REGISTER_OP(SequenceExpand)
.Doc("SequenceExpand operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("sequence_expand")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("sequence_expand")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, AK_FLOAT, Precision::FP32>("sequence_expand")
#endif
.num_in(2)
.num_out(1)
.Args<int>("ref_level", "ref level must be 0");

} /* namespace ops */

} /* namespace anakin */

