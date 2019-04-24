/* Copyright (c) 2016 Anakin Authors All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
#ifndef ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_CONV_ARM_DEPTHWISE_H
#define ANAKIN_SABER_FUNCS_IMPL_ARM_NEON_IMPL_CONV_ARM_DEPTHWISE_H

#include "saber/core/tensor.h"
#include "saber/core/context.h"

namespace anakin{

namespace saber{

void conv_depthwise_3x3p1(const float* din, float* dout, \
                      int num, int ch_out, int h_out, int w_out, \
                      int ch_in, int h_in, int w_in, \
                      const float* weights, const float* bias, \
                      int stride, bool flag_bias, bool flag_relu, Context<ARM>* ctx);

void conv_depthwise_3x3p0(const float* din, float* dout, \
                      int num, int ch_out, int h_out, int w_out, \
                      int ch_in, int h_in, int w_in, \
                      const float* weights, const float* bias, \
                      int stride, bool flag_bias, bool flag_relu, Context<ARM>* ctx);

void conv_depthwise_5x5s2(const float* din, float* dout, \
                      int num, int chout, int hout, int wout, \
                      int chin, int hin, int win, \
                      const float* weights, const float* bias, \
                      int pad, bool flag_bias, bool flag_relu, Context<ARM>* ctx);

void conv_depthwise_5x5s1(const float* din,float* dout, \
                      int num, int chout, int hout, int wout, \
                      int chin, int hin, int win, \
                      const float* weights, const float* bias, \
                      int pad, bool flag_bias, bool flag_relu, Context<ARM>* ctx);
}
}
#endif
