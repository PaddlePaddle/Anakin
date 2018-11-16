/* Copyright (c) 2018 Baidu, Inc. All Rights Reserved.

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

#ifndef ANAKIN_SABER_LITE_FUNCS_CALIBRATE_H
#define ANAKIN_SABER_LITE_FUNCS_CALIBRATE_H

#include "saber/lite/core/tensor_lite.h"
#include "saber/lite/core/context_lite.h"
namespace anakin{
namespace saber{
namespace lite{

SaberStatus get_tensor_scale(const Tensor<CPU>& tin, std::vector<float>& scale_out, \
    int axis, float scale_factor);

SaberStatus get_tensor_scale_inplace(Tensor<CPU>& tin, int axis, float scale_factor);

SaberStatus trans_fp32_weights_to_int8(const Tensor<CPU>& tin, Tensor<CPU>& tout, \
    float scale_factor, int axis, Context* ctx);

SaberStatus trans_fp32_weights_to_int8_gemm(const Tensor<CPU>& tin, Tensor<CPU>& tout, \
    float scale_factor, bool is_trans, int group, Context* ctx);

SaberStatus trans_fp32_weights_to_int8_inplace(Tensor<CPU>& tin, float scale_factor, \
    int axis, Context* ctx);

SaberStatus trans_fp32_weights_to_int8_inplace_gemm(Tensor<CPU>& tin, float scale_factor, \
    bool is_trans, int group, Context* ctx);

SaberStatus trans_tensor_fp32_to_int8(const Tensor<CPU>& tin, Tensor<CPU>& tout, Context* ctx);

SaberStatus trans_tensor_fp32_to_int8_inplace(Tensor<CPU>& tin, Context* ctx);

SaberStatus trans_tensor_int32_to_fp32(const Tensor<CPU>& tin, Tensor<CPU>& tout, \
	float input_scale, std::vector<float>& weights_scale, Context* ctx);

SaberStatus trans_tensor_int32_to_fp32_inplace(Tensor<CPU>& tin, float input_scale, \
	std::vector<float>& weights_scale, Context* ctx);

SaberStatus trans_tensor_int32_to_int8(Tensor<CPU>& tin, Tensor<CPU>& tout, \
	float input_scale, std::vector<float>& weights_scale, Context* ctx);

SaberStatus trans_tensor_int32_to_int8_inplace(Tensor<CPU>& tin, float input_scale,\
	std::vector<float>& weights_scale, Context* ctx);

SaberStatus trans_tensor_int8_to_fp32(Tensor<CPU>& tin, Tensor<CPU>& tout, \
	float input_scale, Context* ctx);

SaberStatus trans_tensor_int8_to_fp32_inplace(Tensor<CPU>& tin, float input_scale, Context* ctx);

SaberStatus trans_fp32_bias_to_int32(const Tensor<CPU>& tin, Tensor<CPU>& tout, \
    float in_scale, std::vector<float> vector_weight_scale, Context* ctx);

SaberStatus trans_fp32_bias_to_int32_inplace(Tensor<CPU>& tin, \
    float in_scale, std::vector<float> vector_weight_scale, Context* ctx);

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_FUNCS_CALIBRATE_H
