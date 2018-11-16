#include "framework/operators/power.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Power<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl =
        static_cast<PowerHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param = impl->_param_power;
    impl->_funcs_power(ins, outs, param, ctx);
}
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
    template<>
    void Power<X86, Precision::FP32>::operator()(
      OpContext<X86>& ctx,
      const std::vector<Tensor4dPtr<X86> >& ins,
      std::vector<Tensor4dPtr<X86> >& outs) {
        auto* impl =
        static_cast<PowerHelper<X86, Precision::FP32>*>(this->_helper);
        auto& param = impl->_param_power;
        impl->_funcs_power(ins, outs, param, ctx);
    }
#endif
/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
PowerHelper<Ttype, Ptype>::~PowerHelper() {
}

template<typename Ttype, Precision Ptype>
Status PowerHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Power op parameter.";
    auto scale = GET_PARAMETER(float, scale);
    auto shift = GET_PARAMETER(float, shift);
    auto power = GET_PARAMETER(float, power);

    saber::PowerParam<Ttype> power_param(power, scale, shift);
    _param_power = power_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PowerHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_power.init(ins, outs, _param_power, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PowerHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_power.compute_output_shape(ins, outs, _param_power));
    return Status::OK();
}

#ifdef USE_CUDA
template class PowerHelper<NV, Precision::FP32>;
template class PowerHelper<NV, Precision::FP16>;
template class PowerHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class PowerHelper<ARM, Precision::FP32>;
template class PowerHelper<ARM, Precision::FP16>;
template class PowerHelper<ARM, Precision::INT8>;
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
template class PowerHelper<X86, Precision::FP32>;
template class PowerHelper<X86, Precision::FP16>;
template class PowerHelper<X86, Precision::INT8>;
#endif


// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Power, PowerHelper, NV, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Power, PowerHelper, ARM, Precision::FP32);
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
ANAKIN_REGISTER_OP_HELPER(Power, PowerHelper, X86, Precision::FP32);
#endif
//! register op
ANAKIN_REGISTER_OP(Power)
.Doc("Power operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("power")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("power")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("power")
#endif
.num_in(1)
.num_out(1)
.Args<float>("scale", " scale of param for pawer")
.Args<float>("shift", " shift of param for power")
.Args<float>("power", " power of param for power");

} /* namespace ops */

} /* namespace anakin */


