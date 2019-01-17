/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "framework/operators/power.h"

namespace anakin {

namespace ops {

#define INSTANCE_POWER(Ttype, Ptype) \
template<> \
void Power<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<PowerHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<PowerHelper<Ttype, Ptype>*> \
                  (this->_helper)->_param_power; \
    impl->_funcs_power(ins, outs, param, ctx); \
}

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
INSTANCE_POWER(NV, Precision::FP32);
template class PowerHelper<NV, Precision::FP32>;
template class PowerHelper<NV, Precision::FP16>;
template class PowerHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
INSTANCE_POWER(ARM, Precision::FP32);
template class PowerHelper<ARM, Precision::FP32>;
template class PowerHelper<ARM, Precision::FP16>;
template class PowerHelper<ARM, Precision::INT8>;
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_POWER(X86, Precision::FP32);
template class PowerHelper<X86, Precision::FP32>;
template class PowerHelper<X86, Precision::FP16>;
template class PowerHelper<X86, Precision::INT8>;
#endif

#ifdef AMD_GPU
INSTANCE_POWER(AMD, Precision::FP32);
template class PowerHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Power, PowerHelper, AMD, Precision::FP32);
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
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("power")
#endif
.num_in(1)
.num_out(1)
.Args<float>("scale", " scale of param for pawer")
.Args<float>("shift", " shift of param for power")
.Args<float>("power", " power of param for power");

} /* namespace ops */

} /* namespace anakin */


