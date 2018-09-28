/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *     
*/

#include "framework/operators/scale.h"

namespace anakin {

namespace ops {

#define INSTANCE_SCALE(Ttype, Ptype) \
template<> \
void Scale<Ttype, Ptype>::operator()( \
    OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<ScaleHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ScaleHelper<Ttype, Ptype>*>(this->_helper)->_param_scale; \
    impl->_funcs_scale(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status ScaleHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Scale op parameter.";
    using pblock_type = PBlock<Ttype>;

    auto axis = GET_PARAMETER(int, axis);
    auto num_axes = GET_PARAMETER(int, num_axes);
    auto bias_term = GET_PARAMETER(bool, bias_term);
    auto weights = GET_PARAMETER(pblock_type, weight_1);

    if (bias_term) {
        auto bias = GET_PARAMETER(pblock_type, weight_2);
        ScaleParam<Ttype> param_scale(weights.vector(), bias.vector(), bias_term, axis, num_axes);
        _param_scale = param_scale;
    } else {
        ScaleParam<Ttype> param_scale(weights.vector(), bias_term, axis, num_axes);
        _param_scale = param_scale;
    }
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ScaleHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_scale.init(ins, outs, _param_scale, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ScaleHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_scale.compute_output_shape(ins, outs, _param_scale));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_SCALE(NV, Precision::FP32);
template class ScaleHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_SCALE(AMD, Precision::FP32);
template class ScaleHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, AMD, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_SCALE(X86, Precision::FP32);
template class ScaleHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SCALE(ARM, Precision::FP32);
template class ScaleHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Scale, ScaleHelper, ARM, Precision::FP32);
#endif//arm

//! register op
ANAKIN_REGISTER_OP(Scale)
.Doc("Scale operator")

#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("Scale")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("Scale")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("Scale")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("Scale")
#endif
.num_in(1)
.num_out(1)
.Args<std::string>("type", " type of Scale ");

} /* namespace ops */

} /* namespace anakin */


