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
#include "framework/operators/batch_norm.h"

namespace anakin {

namespace ops {

#define INSTANCE_BATCH_NORM(Ttype, Ptype) \
template<> \
void BatchNorm<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
        auto* impl = static_cast<BatchNormHelper<Ttype, Ptype>*>(this->_helper); \
        auto& param = static_cast<BatchNormHelper<Ttype, Ptype>*>(this->_helper)->_param_scale; \
        impl->_funcs_scale(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status BatchNormHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Scale op parameter.";
    using pblock_type = PBlock<Ttype>;

    auto eps = GET_PARAMETER(float, epsilon);
    auto mean = GET_PARAMETER(pblock_type, weight_1);
    auto var = GET_PARAMETER(pblock_type, weight_2);
    auto scale_factor = GET_PARAMETER(pblock_type, weight_3);
    auto mean_vec = mean.vector();
    auto var_vec = var.vector();
    auto scale_factor_vec = scale_factor.vector();
    std::vector<typename PrecisionWrapper<Ptype>::type> scale;
    std::vector<typename PrecisionWrapper<Ptype>::type> bias;
    scale.resize(mean.count());
    bias.resize(mean.count());
    auto scale_val = scale_factor_vec[0] == 0 ? 0 : 1 / scale_factor_vec[0];
    for (int i = 0; i < mean.count(); i++) {
        scale[i] = 1.0f / std::sqrt(var_vec[i] * scale_val + eps);
        bias[i] = - mean_vec[i] * scale_val / std::sqrt(var_vec[i] * scale_val + eps); 
    }

    saber::ScaleParam<Ttype> param_scale(scale, bias, true, 1, 1);
    _param_scale = param_scale;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status BatchNormHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_scale.init(ins, outs, _param_scale, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status BatchNormHelper<Ttype, Ptype>::InferShape(const
        std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_scale.compute_output_shape(ins, outs, _param_scale));
    return Status::OK();
}

// register helper
#ifdef USE_CUDA
INSTANCE_BATCH_NORM(NV, Precision::FP32);
template class BatchNormHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(BatchNorm, BatchNormHelper, NV, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_BATCH_NORM(X86, Precision::FP32);
template class BatchNormHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(BatchNorm, BatchNormHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_BATCH_NORM(ARM, Precision::FP32);
template class BatchNormHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(BatchNorm, BatchNormHelper, ARM, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_BATCH_NORM(AMD, Precision::FP32);
template class BatchNormHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(BatchNorm, BatchNormHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(BatchNorm)
.Doc("BatchNorm operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("eps")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("eps")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("eps")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("eps")
#endif
.num_in(1)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


