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
#ifndef ANAKIN_SABER_LITE_FUNCS_NEON_IMPL_POOLING_ARM_IMPL_H
#define ANAKIN_SABER_LITE_FUNCS_NEON_IMPL_POOLING_ARM_IMPL_H

#include "saber/lite/core/tensor_lite.h"

#ifdef USE_ARM_PLACE

namespace anakin{

namespace saber{

namespace lite{

void pooling_basic(const float* din, float* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          PoolingType type, bool global, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int pad_w, int pad_h);

void pooling_global(const float* din, float* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          PoolingType type, bool global, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int pad_w, int pad_h);

void pooling2x2s2_max(const float* din, float* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          PoolingType type, bool global, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int pad_w, int pad_h);

void pooling2x2s2_ave(const float* din, float* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          PoolingType type, bool global, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int pad_w, int pad_h);

void pooling3x3s2_max(const float* din, float* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          PoolingType type, bool global, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int pad_w, int pad_h);

void pooling3x3s2_ave(const float* din, float* dout, \
                          int num, int chout, int hout, int wout, \
                          int chin, int hin, int win, \
                          PoolingType type, bool global, int kernel_w, int kernel_h, \
                          int stride_w, int stride_h, int pad_w, int pad_h);

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //USE_ARM_PLACE

#endif //ANAKIN_SABER_LITE_FUNCS_NEON_IMPL_POOLING_ARM_IMPL_H
