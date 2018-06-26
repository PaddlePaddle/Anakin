#include "framework/operators/arg_max.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Argmax<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl =
        static_cast<ArgmaxHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = impl->_param_argmax;
    impl->_funcs_argmax(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
ArgmaxHelper<Ttype, Dtype, Ptype>::~ArgmaxHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ArgmaxHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Argmax op parameter.";
    auto out_max_val = GET_PARAMETER(bool, out_max_val);
    auto top_k = GET_PARAMETER(int, top_k);
    auto axis_term = GET_PARAMETER(bool, axis_term);

    if (axis_term == true) {
        auto axis = GET_PARAMETER(int, axis);
        saber::ArgmaxParam <Tensor4d<Ttype, Dtype>> argmax_param(out_max_val, top_k, axis);
        _param_argmax = argmax_param;
    } else {
        saber::ArgmaxParam <Tensor4d<Ttype, Dtype>> argmax_param(out_max_val, top_k);
        _param_argmax = argmax_param;
    }

    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ArgmaxHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_argmax.init(ins, outs, _param_argmax, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ArgmaxHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_argmax.compute_output_shape(ins, outs, _param_argmax));
    return Status::OK();
}

#ifdef USE_CUDA
template class ArgmaxHelper<NV, AK_FLOAT, Precision::FP32>;
template class ArgmaxHelper<NV, AK_FLOAT, Precision::FP16>;
template class ArgmaxHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class ArgmaxHelper<ARM, AK_FLOAT, Precision::FP32>;
template class ArgmaxHelper<ARM, AK_FLOAT, Precision::FP16>;
template class ArgmaxHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Argmax, ArgmaxHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Argmax, ArgmaxHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Argmax)
.Doc("Argmax operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("Argmax")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("Argmax")
#endif
.num_in(1)
.num_out(1)
.Args<bool>("out_max_val", " out_max_val for argmax ")
.Args<unsigned int>("top_k", " top_k for argmax")
.Args<int>("axis", " axis for argmax");
} /* namespace ops */

} /* namespace anakin */


