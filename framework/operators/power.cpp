#include "framework/operators/power.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Power<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl =
        static_cast<PowerHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = impl->_param_power;
    impl->_funcs_power(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator
#ifdef USE_ARM_PLACE
template<>
void Power<ARM, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<ARM>& ctx,
    const std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<ARM, AK_FLOAT> >& outs) {
    auto* impl =
        static_cast<PowerHelper<ARM, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = impl->_param_power;
    impl->_funcs_power(ins, outs, param, ctx);
}
#endif

/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
PowerHelper<Ttype, Dtype, Ptype>::~PowerHelper() {
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PowerHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Power op parameter.";
    auto scale = GET_PARAMETER(float, scale);
    auto shift = GET_PARAMETER(float, shift);
    auto power = GET_PARAMETER(float, power);

    saber::PowerParam<Tensor4d<Ttype, Dtype>> power_param(power, scale, shift);
    _param_power = power_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PowerHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_power.init(ins, outs, _param_power, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PowerHelper<Ttype, Dtype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype, Dtype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    SABER_CHECK(_funcs_power.compute_output_shape(ins, outs, _param_power));
    return Status::OK();
}

#ifdef USE_CUDA
template class PowerHelper<NV, AK_FLOAT, Precision::FP32>;
template class PowerHelper<NV, AK_FLOAT, Precision::FP16>;
template class PowerHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
#ifdef ANAKIN_TYPE_FP32
template class PowerHelper<ARM, AK_FLOAT, Precision::FP32>;
#endif
#ifdef ANAKIN_TYPE_FP16
template class PowerHelper<ARM, AK_FLOAT, Precision::FP16>;
#endif
#ifdef ANAKIN_TYPE_INT8
template class PowerHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Power, PowerHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Power, PowerHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Power)
.Doc("Power operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("power")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("power")
#endif
.num_in(1)
.num_out(1)
.Args<float>("scale", " scale of param for pawer")
.Args<float>("shift", " shift of param for power")
.Args<float>("power", " power of param for power");

} /* namespace ops */

} /* namespace anakin */


