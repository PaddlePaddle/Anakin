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
#include "framework/operators/fusion_ops/permute_power.h"

namespace anakin {

namespace ops {

#define INSTANCE_PERMUTE_POWER(Ttype, Ptype) \
template<> \
void PermutePower<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<PermutePowerHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<PermutePowerHelper<Ttype, Ptype>*> \
                  (this->_helper)->_param_permute_power; \
    impl->_funcs_permute_power(ins, outs, param, ctx); \
}

/// set helper
template<typename Ttype, Precision Ptype>
PermutePowerHelper<Ttype, Ptype>::~PermutePowerHelper() {
    LOG(INFO) << "Decons permute_cpu_float";
}

template<typename Ttype, Precision Ptype>
Status PermutePowerHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing PermutePower op parameter.";
    auto dims = GET_PARAMETER(PTuple<int>, dims);
    auto scale = GET_PARAMETER(float, power_0_scale);
    auto shift = GET_PARAMETER(float, power_0_shift);
    auto power = GET_PARAMETER(float, power_0_power);

    saber::PermuteParam<Ttype> permute_param(dims.vector());
    saber::PowerParam<Ttype> power_param(power, scale, shift);
    saber::PermutePowerParam<Ttype> permute_power_param(permute_param, power_param);
    _param_permute_power = permute_power_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PermutePowerHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_permute_power.init(ins, outs, _param_permute_power, SPECIFY, SABER_IMPL, ctx);
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status PermutePowerHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    _funcs_permute_power.compute_output_shape(ins, outs, _param_permute_power);
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_PERMUTE_POWER(NV, Precision::FP32);
template class PermutePowerHelper<NV, Precision::FP32>;
template class PermutePowerHelper<NV, Precision::FP16>;
template class PermutePowerHelper<NV, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(PermutePower, PermutePowerHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_PERMUTE_POWER(ARM, Precision::FP32);

ANAKIN_REGISTER_OP_HELPER(PermutePower, PermutePowerHelper, ARM, Precision::FP32);
#endif

#ifdef USE_X86_PLACE
INSTANCE_PERMUTE_POWER(X86, Precision::FP32);
template class PermutePowerHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(PermutePower, PermutePowerHelper, X86, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_PERMUTE_POWER(AMD, Precision::FP32);
template class PermutePowerHelper<AMD, Precision::FP32>;
template class PermutePowerHelper<AMD, Precision::FP16>;
template class PermutePowerHelper<AMD, Precision::INT8>;
ANAKIN_REGISTER_OP_HELPER(PermutePower, PermutePowerHelper, AMD, Precision::FP32);
#endif


//! register op
ANAKIN_REGISTER_OP(PermutePower)
.Doc("PermutePower fusion operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("permute_power")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("permute_power")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("permute_power")
#endif
.num_in(1)
.num_out(1)
.Args<float>("power_0_scale", " scale of param for pawer")
.Args<float>("power_0_shift", " shift of param for power")
.Args<float>("power_0_power", " power of param for power")
.Args<PTuple<int>>("dims", " dims for permuting the order of input ");

} /* namespace ops */

} /* namespace anakin */


