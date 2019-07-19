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
#include "framework/operators/group_norm.h"

namespace anakin {

namespace ops {

#define INSTANCE_GROUP_NORMAL(Ttype, Ptype) \
template<> \
void GroupNormal<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = static_cast<GroupNormalHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = static_cast<GroupNormalHelper<Ttype, Ptype>*> \
                  (this->_helper)->_param_group_normal; \
    impl->_funcs_group_normal(ins, outs, param, ctx); \
}

/// TODO ... specialization other type of operator
/// set helper
template<typename Ttype, Precision Ptype>
GroupNormalHelper<Ttype, Ptype>::~GroupNormalHelper() {
}

template<typename Ttype, Precision Ptype>
Status GroupNormalHelper<Ttype, Ptype>::InitParam() {
    //DLOG(WARNING) << "Parsing GroupNormal op parameter.";
    auto eps = GET_PARAMETER(float, eps);
    auto p = GET_PARAMETER_WITH_DEFAULT(int, p, 1);
    auto group = GET_PARAMETER_WITH_DEFAULT(int, group, 0);
    auto has_bias = GET_PARAMETER_WITH_DEFAULT(bool, has_bias, false);
    auto has_scale = GET_PARAMETER_WITH_DEFAULT(bool, has_scale, false);
    CHECK_GE(group, 1) << "group normal group must > 1";
    PBlock<Ttype> bias;
    PBlock<Ttype> scale;
    if (has_scale){
      scale = GET_PARAMETER(PBlock<Ttype>, scale);
    }
    if (has_bias){
      bias = GET_PARAMETER(PBlock<Ttype>, bias);
    }
    saber::NormalizeParam<Ttype> group_normal_param(has_scale, &(scale.d_tensor()), 
      has_bias, &(bias.d_tensor()), group, eps);
    _param_group_normal = group_normal_param;


    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status GroupNormalHelper<Ttype, Ptype>::Init(OpContext<Ttype> &ctx,
                                                const std::vector<Tensor4dPtr<Ttype> >& ins,
                                                std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_group_normal.init(ins, outs, _param_group_normal, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status GroupNormalHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
                                                      std::vector<Tensor4dPtr<Ttype> >& outs) {
   SABER_CHECK(_funcs_group_normal.compute_output_shape(ins, outs, _param_group_normal));
   return Status::OK();
}

#ifdef AMD_GPU
INSTANCE_GROUP_NORMAL(AMD, Precision::FP32);
template class GroupNormalHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(GroupNormal, GroupNormalHelper, AMD, Precision::FP32);
#endif

#ifdef USE_CUDA
INSTANCE_GROUP_NORMAL(NV, Precision::FP32);
template class GroupNormalHelper<NV, Precision::FP32>;
template class GroupNormalHelper<NV, Precision::FP16>;
template class GroupNormalHelper<NV, Precision::INT8>;
#endif

#ifdef USE_X86_PLACE
INSTANCE_GROUP_NORMAL(X86, Precision::FP32);
template class GroupNormalHelper<X86, Precision::FP32>;
#endif

#ifdef USE_ARM_PLACE
INSTANCE_GROUP_NORMAL(ARM, Precision::FP32);
template class GroupNormalHelper<ARM, Precision::FP32>;
template class GroupNormalHelper<ARM, Precision::FP16>;
template class GroupNormalHelper<ARM, Precision::INT8>;
#endif

#ifdef USE_CUDA
ANAKIN_REGISTER_OP_HELPER(GroupNormal, GroupNormalHelper, NV, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
ANAKIN_REGISTER_OP_HELPER(GroupNormal, GroupNormalHelper, ARM, Precision::FP32);
#endif

#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
ANAKIN_REGISTER_OP_HELPER(GroupNormal, GroupNormalHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(GroupNormal)
    .Doc("GroupNormal operator")
#ifdef USE_CUDA
    .__alias__<NV, Precision::FP32>("group_normal")
#endif
#if defined(USE_X86_PLACE) || defined(BUILD_LITE)
    .__alias__<X86, Precision::FP32>("group_normal")
#endif
#ifdef USE_ARM_PLACE
    .__alias__<ARM, Precision::FP32>("group_normal")
#endif
#ifdef AMD_GPU
    .__alias__<AMD, Precision::FP32>("group_normal")
#endif
    .num_in(1)
    .num_out(1)
    .Args<bool>("is_across_spatial", "")
    .Args<bool>("is_shared_channel", "")
    .Args<float>("eps", "")
    .Args<int>("p", "");

} /* namespace ops */

} /* namespace anakin */


