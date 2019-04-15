#include "framework/operators/maxout.h"

namespace anakin {

namespace ops {

#define INSTANCE_MAXOUT(Ttype, Ptype) \
template<> \
void MaxOut<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<MaxOutHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<MaxOutHelper<Ttype, Ptype>*> \
                  (this->_helper)->_param_maxout; \
    impl->_funcs_maxout(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
MaxOutHelper<Ttype, Ptype>::~MaxOutHelper() {
}

template<typename Ttype, Precision Ptype>
Status MaxOutHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing MaxOut op parameter.";
    auto groups = GET_PARAMETER(int, groups);
    MaxOutParam<Ttype> param_maxout(groups);
    _param_maxout = param_maxout;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status MaxOutHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_maxout.init(ins, outs, _param_maxout, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status MaxOutHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_maxout.compute_output_shape(ins, outs, _param_maxout));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_MAXOUT(NV, Precision::FP32);
template class MaxOutHelper<NV, Precision::FP32>;
template class MaxOutHelper<NV, Precision::FP16>;
template class MaxOutHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(MaxOut, MaxOutHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
INSTANCE_MAXOUT(ARM, Precision::FP32);
template class MaxOutHelper<ARM, Precision::FP32>;
template class MaxOutHelper<ARM, Precision::FP16>;
template class MaxOutHelper<ARM, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(MaxOut, MaxOutHelper, ARM, Precision::FP32);
#endif
#ifdef USE_X86_PLACE
INSTANCE_MAXOUT(X86, Precision::FP32);
template class MaxOutHelper<X86, Precision::FP32>;
template class MaxOutHelper<X86, Precision::FP16>;
template class MaxOutHelper<X86, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(MaxOut, MaxOutHelper, X86, Precision::FP32);
#endif
#ifdef AMD_GPU
INSTANCE_MAXOUT(AMD, Precision::FP32);
template class MaxOutHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(MaxOut, MaxOutHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(MaxOut)
.Doc("MaxOut operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("maxout")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("maxout")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("maxout")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("maxout")
#endif
.num_in(1)
.num_out(1)
.Args<int>("groups", " split tensor's channel by size groups. ");

} /* namespace ops */

} /* namespace anakin */


