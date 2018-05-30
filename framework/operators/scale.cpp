#include "framework/operators/scale.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Scale<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl =
        static_cast<ScaleHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param =
        static_cast<ScaleHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper)->_param_scale;
    impl->_funcs_scale(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
ScaleHelper<Ttype, Dtype, Ptype>::~ScaleHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ScaleHelper<Ttype, Dtype, Ptype>::InitParam() {
    LOG(WARNING) << "Parsing Scale op parameter.";
    auto axis = GET_PARAMETER(int, axis);
    auto num_axes = GET_PARAMETER(int, num_axes);
    auto bias_term = GET_PARAMETER(bool, bias_term);
    auto weights = GET_PARAMETER(PTuple<typename DataTypeWarpper<Dtype>::type>, scale_w);
    auto bias = GET_PARAMETER(PTuple<typename DataTypeWarpper<Dtype>::type>, scale_b);
    ScaleParam<Tensor4d<Ttype, Dtype>> param_scale(weights.vector(), bias.vector(), bias_term, axis, num_axes);
    _param_scale = param_scale;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ScaleHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_scale.init(ins, outs, _param_scale, SPECIFY, VENDER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ScaleHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_scale.compute_output_shape(ins, outs, _param_scale));
    return Status::OK();
}

#ifdef USE_CUDA
template class ScaleHelper<NV, AK_FLOAT, Precision::FP32>;
template class ScaleHelper<NV, AK_FLOAT, Precision::FP16>;
template class ScaleHelper<NV, AK_FLOAT, Precision::INT8>;
#endif
#ifdef USE_ARM_PLACE
template class ScaleHelper<ARM, AK_FLOAT, Precision::FP32>;
template class ScaleHelper<ARM, AK_FLOAT, Precision::FP16>;
template class ScaleHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, ARM, AK_FLOAT, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Scale)
.Doc("Scale operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("scale")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("scale")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " type of Scale ");

} /* namespace ops */

} /* namespace anakin */


