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

#ifndef ANAKIN_mobile_v2_H 
#define ANAKIN_mobile_v2_H 

#include <stdio.h>
#include <stdlib.h>

#include <saber/lite/core/common_lite.h>
#include <saber/lite/funcs/detection_lite.h>
#include <saber/lite/funcs/saber_activation.h>
#include <saber/lite/funcs/saber_concat.h>
#include <saber/lite/funcs/saber_detection_output.h>
#include <saber/lite/funcs/saber_eltwise.h>
#include <saber/lite/funcs/saber_permute.h>
#include <saber/lite/funcs/saber_prelu.h>
#include <saber/lite/funcs/saber_priorbox.h>
#include <saber/lite/funcs/saber_slice.h>
#include <saber/lite/funcs/timer_lite.h>
#include <saber/lite/funcs/utils_arm.h>
#include <saber/lite/funcs/saber_conv.h>
#include <saber/lite/funcs/saber_conv_act.h>
#include <saber/lite/funcs/saber_conv_batchnorm_scale.h>
#include <saber/lite/funcs/saber_conv_batchnorm_scale_relu.h>
#include <saber/lite/funcs/saber_fc.h>
#include <saber/lite/funcs/saber_pooling.h>
#include <saber/lite/funcs/saber_softmax.h>

using namespace anakin;
using namespace anakin::saber;
using namespace anakin::saber::lite;

namespace anakin { 

/// Model mobile_v2 have  1 inputs.
///  |-- input name : input_0  -- Shape(1,3,224,224)
LITE_EXPORT Tensor<CPU, AK_FLOAT>* get_in(const char* in_name);

/// Model mobile_v2 have  1 outputs.
///  |-- output name : prob_out  -- Shape(1,1000,1,1)
LITE_EXPORT Tensor<CPU, AK_FLOAT>* get_out(const char* out_name);

LITE_EXPORT bool mobile_v2_load_param(const char* param_path);

/// mobile_v2_init should only be invoked once when input shape changes.
LITE_EXPORT void mobile_v2_init(Context& ctx);

/// Running prediction for model mobile_v2.
LITE_EXPORT void mobile_v2_prediction();

/// Release all resource used by model mobile_v2.
LITE_EXPORT void mobile_v2_release_resource();

} /* namespace anakin */

#endif

