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

#ifndef ANAKIN_SABER_FUNCS_TYPE_TRANS_H
#define ANAKIN_SABER_FUNCS_TYPE_TRANS_H

#include "saber/core/tensor.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/saturate.h"
#include "saber/saber_types.h"

namespace anakin {
namespace saber {

typedef enum{
    CONV_TYPE = 0,
    DECONV_TYPE = 1,
    FC_TYPE = 2
} TRANS_TYPE;

template<typename TargetType>
SaberStatus trans_weights_dtype(Tensor<TargetType>& weights, DataType type, float scale_factor, TRANS_TYPE op_type, int group){
    LOG(ERROR) << "trans_weights_dtype has no impl";
    return SaberUnImplError;
}
template<typename TargetType, DataType T1, DataType T2>
SaberStatus trans_tensor_dtype(Tensor<TargetType>& tin, Tensor<TargetType>& tout, \
    float input_scale, float output_scale, std::vector<float> weights_scale){
    LOG(ERROR) << "trans_tensor_dtype has no impl";
    return SaberUnImplError;
}
template<typename TargetType>
SaberStatus trans_fp32_bias_to_int32(Tensor<TargetType>& tin, Tensor<TargetType>& tout, \
    float in_scale, std::vector<float> vector_weight_scale){
    LOG(ERROR) << "trans_fp32_bias_to_int32 has no impl";
    return SaberUnImplError;
}

#ifdef USE_ARM_PLACE

template<>
SaberStatus trans_weights_dtype<ARM>(Tensor<ARM>& weights, DataType type, float scale_factor, \
     TRANS_TYPE op_type, int group);

template<>
SaberStatus trans_fp32_bias_to_int32<ARM>(Tensor<ARM>& tin, Tensor<ARM>& tout, \
    float in_scale, std::vector<float> vector_weight_scale);

template<>
SaberStatus trans_tensor_dtype<ARM, AK_FLOAT, AK_INT8>(Tensor<ARM>& tin, Tensor<ARM>& tout, \
    float input_scale, float output_scale, std::vector<float> weights_scale);

template<>
SaberStatus trans_tensor_dtype<ARM, AK_INT8, AK_FLOAT>(Tensor<ARM>& tin, Tensor<ARM>& tout, \
    float input_scale, float output_scale, std::vector<float> weights_scale);

template<>
SaberStatus trans_tensor_dtype<ARM, AK_INT32, AK_FLOAT>(Tensor<ARM>& tin, Tensor<ARM>& tout, \
    float input_scale, float output_scale, std::vector<float> weights_scale);

template<>
SaberStatus trans_tensor_dtype<ARM, AK_INT32, AK_INT8>(Tensor<ARM>& tin, Tensor<ARM>& tout, \
    float input_scale, float output_scale, std::vector<float> weights_scale);

SaberStatus get_tensor_scale(const Tensor<ARM>& tin, std::vector<float>& scale_out, \
    int axis, float scale_factor);

template <typename dtype>
void int32_to_dtype(const int* din, dtype* dout, const float* scale,
    int axis_size, long long outer_size, long long inner_size);
#endif

} // namespace saber
} // namespace anakin
#endif // ANAKIN_SABER_FUNCS_TYPE_TRANS_H