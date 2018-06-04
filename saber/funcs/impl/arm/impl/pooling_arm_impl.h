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
#ifndef ANAKIN_SABER_FUNCS_ARM_IMPL_POOLING_ARM_IMPL_H
#define ANAKIN_SABER_FUNCS_ARM_IMPL_POOLING_ARM_IMPL_H

#include "saber/core/tensor.h"
#include "saber/core/tensor_op.h"

#ifdef USE_ARM_PLACE

namespace anakin {

namespace saber{

void pooling_basic(Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, \
    Tensor<ARM, AK_FLOAT, NCHW>& tensor_in, PoolingType type, bool global, \
    int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h);

void pooling_global(Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, \
    Tensor<ARM, AK_FLOAT, NCHW>& tensor_in, PoolingType type, bool global, \
    int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h);

void pooling2x2s2_max(Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, \
    Tensor<ARM, AK_FLOAT, NCHW>& tensor_in, PoolingType type, bool global, \
    int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h);

void pooling2x2s2_ave(Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, \
    Tensor<ARM, AK_FLOAT, NCHW>& tensor_in, PoolingType type, bool global, \
    int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h);

void pooling3x3s2_max(Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, \
    Tensor<ARM, AK_FLOAT, NCHW>& tensor_in, PoolingType type, bool global, \
    int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h);

void pooling3x3s2_ave(Tensor<ARM, AK_FLOAT, NCHW>& tensor_out, \
    Tensor<ARM, AK_FLOAT, NCHW>& tensor_in, PoolingType type, bool global, \
    int kernel_w, int kernel_h, int stride_w, int stride_h, int pad_w, int pad_h);

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE

#endif //ANAKIN_SABER_FUNCS_ARM_IMPL_POOLING_ARM_IMPL_H
