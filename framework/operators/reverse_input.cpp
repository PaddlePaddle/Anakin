#include "framework/operators/reverse_input.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
Status ReverseInputHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing ReverseInput op parameter.";
    _param_reverse_input = EmptyParam<Ttype>();
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ReverseInputHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_reverse_input.init(ins, outs, _param_reverse_input, SPECIFY,
                SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ReverseInputHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>>&
        ins,
        std::vector<Tensor4dPtr<Ttype>>& outs) {
    SABER_CHECK(_funcs_reverse_input.compute_output_shape(ins, outs,
                _param_reverse_input));
    return Status::OK();
}


#define INSTANCE_CONCAT(Ttype, Ptype) \
template<> \
void ReverseInput<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
        const std::vector<Tensor4dPtr<Ttype> >& ins, \
                std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<ReverseInputHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ReverseInputHelper<Ttype, Ptype>*>(this->_helper)->_param_reverse_input; \
    impl->_funcs_reverse_input(ins, outs, param, ctx); \
}

#ifdef USE_CUDA
INSTANCE_CONCAT(NV, Precision::FP32);
template class ReverseInputHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReverseInput, ReverseInputHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_CONCAT(ARM, Precision::FP32);
template class ReverseInputHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReverseInput, ReverseInputHelper, ARM, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_CONCAT(X86, Precision::FP32);
template class ReverseInputHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(ReverseInput, ReverseInputHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(ReverseInput)
.Doc("ReverseInput operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("reverse_input")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("reverse_input")
#endif
#ifdef USE_X86_PLACE
.__alias__<X86, Precision::FP32>("reverse_input")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


