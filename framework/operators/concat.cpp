#include "framework/operators/concat.h"

namespace anakin {

namespace ops {

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ConcatHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Concat op parameter.";
    auto axis = GET_PARAMETER(int, axis);
    ConcatParam<Tensor4d<Ttype, Dtype>> param_concat(axis);
    _param_concat = param_concat;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ConcatHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype> &ctx,
                                  const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
                                    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs){
    SABER_CHECK(_funcs_concat.init(ins, outs, _param_concat, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ConcatHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype>> &ins,
                                std::vector<Tensor4dPtr<Ttype, Dtype>> &outs) {
    SABER_CHECK(_funcs_concat.compute_output_shape(ins, outs, _param_concat));
    return Status::OK();
}


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

#ifdef USE_CUDA
INSTANCE_CONCAT(NV, AK_FLOAT, Precision::FP32);
template class ConcatHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_CONCAT(ARM, AK_FLOAT, Precision::FP32);
template class ConcatHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_CONCAT(X86, AK_FLOAT, Precision::FP32);
template class ConcatHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Concat, ConcatHelper, X86, AK_FLOAT, Precision::FP32);
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


