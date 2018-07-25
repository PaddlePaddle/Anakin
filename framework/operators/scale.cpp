#include "framework/operators/scale.h"

namespace anakin {

namespace ops {

#define INSTANCE_SCALE(Ttype, Dtype, Ptype) \
template<> \
void Scale<Ttype, Dtype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins, \
    std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) { \
    auto* impl = \
        static_cast<ScaleHelper<Ttype, Dtype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ScaleHelper<Ttype, Dtype, Ptype>*>(this->_helper)->_param_scale; \
    impl->_funcs_scale(ins, outs, param, ctx); \
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ScaleHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Scale op parameter.";
    using pblock_type = PBlock<typename DataTypeWarpper<Dtype>::type, Ttype>;

    auto axis = GET_PARAMETER(int, axis);
    auto num_axes = GET_PARAMETER(int, num_axes);
    auto bias_term = GET_PARAMETER(bool, bias_term);
    auto weights = GET_PARAMETER(pblock_type, weight_1);

    if (bias_term) {
        auto bias = GET_PARAMETER(pblock_type, weight_2);
        ScaleParam <Tensor4d<Ttype, Dtype>> param_scale(weights.vector(), bias.vector(), bias_term, axis, num_axes);
        _param_scale = param_scale;
    } else {
        ScaleParam <Tensor4d<Ttype, Dtype>> param_scale(weights.vector(), bias_term, axis, num_axes);
        _param_scale = param_scale;
    }
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status ScaleHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_scale.init(ins, outs, _param_scale, SPECIFY, SABER_IMPL, ctx));
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
INSTANCE_SCALE(NV, AK_FLOAT, Precision::FP32);
template class ScaleHelper<NV, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, NV, AK_FLOAT, Precision::FP32);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
INSTANCE_SCALE(X86, AK_FLOAT, Precision::FP32);
template class ScaleHelper<X86, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, X86, AK_FLOAT, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SCALE(ARM, AK_FLOAT, Precision::FP32);
template class ScaleHelper<ARM, AK_FLOAT, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, ARM, AK_FLOAT, Precision::FP32);
#endif//arm

//! register op
ANAKIN_REGISTER_OP(Scale)
.Doc("Scale operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("Scale")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("Scale")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
.__alias__<X86, AK_FLOAT, Precision::FP32>("Scale")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " type of Scale ");

} /* namespace ops */

} /* namespace anakin */


