#include "framework/operators/arg_max.h"

namespace anakin {

namespace ops {

/// TODO ... specialization other type of operator
#define INSTANCE_ARGMAX(Ttype, Ptype) \
template<> \
void Argmax<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<ArgmaxHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = impl->_param_argmax; \
    impl->_funcs_argmax(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
ArgmaxHelper<Ttype, Ptype>::~ArgmaxHelper() {
}

template<typename Ttype, Precision Ptype>
Status ArgmaxHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Argmax op parameter.";
    auto out_max_val = GET_PARAMETER(bool, out_max_val);
    auto top_k = GET_PARAMETER(int, top_k);
    auto axis_term = GET_PARAMETER(bool, axis_term);

    if (axis_term == true) {
        auto axis = GET_PARAMETER(int, axis);
        saber::ArgmaxParam <Ttype> argmax_param(out_max_val, top_k, axis);
        _param_argmax = argmax_param;
    } else {
        saber::ArgmaxParam <Ttype> argmax_param(out_max_val, top_k);
        _param_argmax = argmax_param;
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ArgmaxHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_argmax.init(ins, outs, _param_argmax, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ArgmaxHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
                                              std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_argmax.compute_output_shape(ins, outs, _param_argmax));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_ARGMAX(NV, Precision::FP32);
template class ArgmaxHelper<NV, Precision::FP32>;
template class ArgmaxHelper<NV, Precision::FP16>;
template class ArgmaxHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Argmax, ArgmaxHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_ARGMAX(AMD, Precision::FP32);
template class ArgmaxHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Argmax, ArgmaxHelper, AMD, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_ARGMAX(X86, Precision::FP32);
template class ArgmaxHelper<X86, Precision::FP32>;
template class ArgmaxHelper<X86, Precision::FP16>;
template class ArgmaxHelper<X86, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(Argmax, ArgmaxHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE

#ifdef ANAKIN_TYPE_FP32
INSTANCE_ARGMAX(ARM, Precision::FP32);
template class ArgmaxHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Argmax, ArgmaxHelper, ARM, Precision::FP32);
#endif //fp32

#ifdef ANAKIN_TYPE_FP16
template class ArgmaxHelper<ARM, Precision::FP16>;
#endif //fp16

#ifdef ANAKIN_TYPE_INT8
template class ArgmaxHelper<ARM, Precision::INT8>;
#endif //int8

#endif //arm

//! register op
ANAKIN_REGISTER_OP(Argmax)
.Doc("Argmax operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("Argmax")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("Argmax")
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("Argmax")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("Argmax")
#endif
.num_in(1)
.num_out(1)
.Args<bool>("out_max_val", " out_max_val for argmax ")
.Args<unsigned int>("top_k", " top_k for argmax")
.Args<int>("axis", " axis for argmax");
} /* namespace ops */

} /* namespace anakin */


