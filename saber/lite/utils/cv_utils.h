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
#ifndef ANAKIN_SABER_LITE_UTILS_CV_UTILS_H
#define ANAKIN_SABER_LITE_UTILS_CV_UTILS_H

#include "saber/lite/core/common_lite.h"
#include "saber/lite/core/tensor_lite.h"
namespace anakin{

namespace saber{

namespace lite{
typedef Tensor<CPU> TensorHf;

void rotate(const unsigned char* src, unsigned char* dst, int srcw, int srch, int dstw, int dsth, int angle);

//x: flip_num = 1 y: flip_num = -1 xy: flip_num = 0;
void flip(const unsigned char* src, unsigned char* dst, int srcw, int srch, int dstw, int dsth, int flip_num);

//y_w = srcw, y_h = 2/3 * srch uv_w = srcw uv_h = 1/3 * srch
void resize(const unsigned char* src, unsigned char* dst, int srcw, int srch, int dstw, int dsth);

//nv21(yvu)  to BGR: bgr store bbbgggrrrbbb dsth * dstw = (3 * srch) * (srcw) y_w = srcw, y_h = 2/3 * srch uv_w = srcw uv_h = 1/3 * srch
void nv21_to_bgr(const unsigned char* src, unsigned char* dst, int srcw, int srch, int dstw, int dsth);

//nv12(yuv)  to BGR: bgr store bbbgggrrrbbb dsth * dstw = (3 * srch) * (srcw) y_w = srcw, y_h = 2/3 * srch uv_w = srcw uv_h = 1/3 * srch
void nv12_to_bgr(const unsigned char* src, unsigned char* dst, int srcw, int srch, int dstw, int dsth);

//bgr output.w == width output.h == height/3
void bgr_to_tensor(const unsigned char* bgr, TensorHf& output, int width, int height, float* means, float* scales);

//yvu   y_w = width, y_h = height uv_w = width uv_h = 1/2 * height
void nv21_to_tensor(const unsigned char* nv21, TensorHf& output, int width, int height, float* means, float* scales);

//yuv  y_w = width, y_h = height uv_w = width uv_h = 1/2 * height
void nv12_to_tensor(const unsigned char* nv12, TensorHf& output, int width, int height, float* means, float* scales);

} //namespace lite

} //namespace saber

} //namespace anakin

#endif //ANAKIN_SABER_LITE_FUNCS_ARM_UTILS_H
