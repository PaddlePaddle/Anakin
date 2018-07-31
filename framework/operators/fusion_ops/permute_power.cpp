#include "framework/operators/fusion_ops/permute_power.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void PermutePower<NV, AK_FLOAT, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV, AK_FLOAT> >& ins,
    std::vector<Tensor4dPtr<NV, AK_FLOAT> >& outs) {
    auto* impl = static_cast<PermutePowerHelper<NV, AK_FLOAT, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<PermutePowerHelper<NV, AK_FLOAT, Precision::FP32>*>
                  (this->_helper)->_param_permute_power;
    impl->_funcs_permute_power(ins, outs, param, ctx);
}
#endif

/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, DataType Dtype, Precision Ptype>
PermutePowerHelper<Ttype, Dtype, Ptype>::~PermutePowerHelper() {
    LOG(INFO) << "Decons permute_cpu_float";
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PermutePowerHelper<Ttype, Dtype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing PermutePower op parameter.";
    auto dims = GET_PARAMETER(PTuple<int>, dims);
    auto scale = GET_PARAMETER(float, power_0_scale);
    auto shift = GET_PARAMETER(float, power_0_shift);
    auto power = GET_PARAMETER(float, power_0_power);

    saber::PermuteParam<Tensor4d<Ttype, Dtype>> permute_param(dims.vector());
    saber::PowerParam<Tensor4d<Ttype, Dtype>> power_param(power, scale, shift);
    saber::PermutePowerParam<Tensor4d<Ttype, Dtype>> permute_power_param(permute_param, power_param);
    _param_permute_power = permute_power_param;
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PermutePowerHelper<Ttype, Dtype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    _funcs_permute_power.init(ins, outs, _param_permute_power, SPECIFY, SABER_IMPL, ctx);
    return Status::OK();
}

template<typename Ttype, DataType Dtype, Precision Ptype>
Status PermutePowerHelper<Ttype, Dtype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype, Dtype> >& ins,
        std::vector<Tensor4dPtr<Ttype, Dtype> >& outs) {
    _funcs_permute_power.compute_output_shape(ins, outs, _param_permute_power);
    return Status::OK();
}

#ifdef USE_CUDA
template class PermutePowerHelper<NV, AK_FLOAT, Precision::FP32>;
template class PermutePowerHelper<NV, AK_FLOAT, Precision::FP16>;
template class PermutePowerHelper<NV, AK_FLOAT, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class PermutePowerHelper<ARM, AK_FLOAT, Precision::FP32>;
template class PermutePowerHelper<ARM, AK_FLOAT, Precision::FP16>;
template class PermutePowerHelper<ARM, AK_FLOAT, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(PermutePower, PermutePowerHelper, NV, AK_FLOAT, Precision::FP32);
#endif
#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(PermutePower, PermutePowerHelper, ARM, AK_FLOAT, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(PermutePower)
.Doc("PermutePower fusion operator")
#ifdef USE_CUDA
.__alias__<NV, AK_FLOAT, Precision::FP32>("permute_power")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, AK_FLOAT, Precision::FP32>("permute_power")
#endif
.num_in(1)
.num_out(1)
.Args<float>("power_0_scale", " scale of param for pawer")
.Args<float>("power_0_shift", " shift of param for power")
.Args<float>("power_0_power", " power of param for power")
.Args<PTuple<int>>("dims", " dims for permuting the order of input ");

} /* namespace ops */

} /* namespace anakin */


