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
#include "framework/operators/normalize.h"

namespace anakin {

namespace ops {

#ifdef USE_CUDA
template<>
void Normalize<NV, Precision::FP32>::operator()(
    OpContext<NV>& ctx,
    const std::vector<Tensor4dPtr<NV> >& ins,
    std::vector<Tensor4dPtr<NV> >& outs) {
    auto* impl = static_cast<NormalizeHelper<NV, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<NormalizeHelper<NV, Precision::FP32>*>(this->_helper)->_param_normalize;
    impl->_funcs_normalize(ins, outs, param, ctx);
}
#endif

#ifdef AMD_GPU
template<>
void Normalize<AMD, Precision::FP32>::operator()(
    OpContext<AMD>& ctx,
    const std::vector<Tensor4dPtr<AMD> >& ins,
    std::vector<Tensor4dPtr<AMD> >& outs) {
    auto* impl = static_cast<NormalizeHelper<AMD, Precision::FP32>*>(this->_helper);
    auto& param = static_cast<NormalizeHelper<AMD, Precision::FP32>*>(this->_helper)->_param_normalize;
    impl->_funcs_normalize(ins, outs, param, ctx);
}
#endif
/// TODO ... specialization other type of operator


/// set helper
template<typename Ttype, Precision Ptype>
NormalizeHelper<Ttype, Ptype>::~NormalizeHelper() {
}

template<typename Ttype, Precision Ptype>
Status NormalizeHelper<Ttype, Ptype>::InitParam() {
    //DLOG(WARNING) << "Parsing Normalize op parameter.";
    auto is_across_spatial = GET_PARAMETER(bool, is_across_spatial);
    auto is_shared_channel = GET_PARAMETER(bool, is_shared_channel);
    auto eps = GET_PARAMETER(float, eps);
    auto p = GET_PARAMETER(int, p);

    using pblock_type = PBlock<Ttype>;
    auto input_scale = GET_PARAMETER(pblock_type, weight_1);

    saber::NormalizeParam<Ttype> normalize_param(is_across_spatial, is_shared_channel, \
            & (input_scale.d_tensor()), eps, p);
    _param_normalize = normalize_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status NormalizeHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_normalize.init(ins, outs, _param_normalize, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status NormalizeHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_normalize.compute_output_shape(ins, outs, _param_normalize));
    return Status::OK();
}

#ifdef USE_CUDA
template class NormalizeHelper<NV, Precision::FP32>;
template class NormalizeHelper<NV, Precision::FP16>;
template class NormalizeHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class NormalizeHelper<ARM, Precision::FP32>;
template class NormalizeHelper<ARM, Precision::FP16>;
template class NormalizeHelper<ARM, Precision::INT8>;
#endif

#ifdef AMD_GPU
template class NormalizeHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Normalize, NormalizeHelper, AMD, Precision::FP32);
#endif
// register helper
#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(Normalize, NormalizeHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(Normalize, NormalizeHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Normalize)
.Doc("Normalize operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("normalize")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("normalize")
#endif
.num_in(1)
.num_out(1)
.Args<bool>("is_across_spatial", "")
.Args<bool>("is_shared_channel", "")
.Args<float>("eps", "")
.Args<int>("p", "");

} /* namespace ops */

} /* namespace anakin */


