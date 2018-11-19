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
#ifndef ANAKIN_SABER_FUNC_IMPL_AMD_UTILS_H
#define ANAKIN_SABER_FUNC_IMPL_AMD_UTILS_H

#include <CL/cl.h>
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"

#define MLO_POOLING_OP_AVE 0
#define MLO_POOLING_OP_MAX 1

namespace anakin {

namespace saber {
// so that MIOpen works whether or not recent MIOpenGEMM changes pulled:
// convert size_t and ulong kernel function parameters to unsigned.
namespace tempfix {
void add_bias_relu(std::string& clstr);
void add_relu(std::string& clstr);
void set_offsets_to_uint(std::string& clstr, int times);
void set_offsets_to_uint(std::string& clstr);
} // namespace tempfix

void Im2ColGPU(
    AMDKernelPtr& kptr,
    int device_id,
    int c,
    int h,
    int w,
    int wei_h,
    int wei_w,
    int out_h,
    int out_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w);

void transpose_NCHW2CNHW(
    AMDKernelPtr& kptr,
    int device_id,
    int n,
    int c,
    int h_in,
    int w_in,
    int h_out,
    int w_out,
    int in_offset,
    int out_offset,
    int h_stride,
    int w_stride);

void transpose_CNHW2NCHW(
    AMDKernelPtr& kptr,
    int device_id,
    int n,
    int c,
    int h_out,
    int w_out,
    int h_in,
    int w_in,
    int in_offset,
    int out_offset,
    int h_stride,
    int w_stride,
    bool isBias);

void BiasReluPool(
    std::vector<AMDKernelPtr>& vkptr,
    int device_id,
    int bt_size,
    int n_wei,
    int in_h,
    int in_w,
    int in_c,
    int out_h,
    int out_w,
    int out_c,
    int pooling_w_h,
    int pooling_w_w,
    int pooling_s_h,
    int pooling_s_w,
    int pooling_p_h,
    int pooling_p_w,
    int pooling_type,
    bool isBias,
    bool isActive);

} // namespace saber
} // namespace anakin
#endif
