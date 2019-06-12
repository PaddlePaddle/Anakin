/* Copyright (c) 2019 Anakin Authors, Inc. All Rights Reserved.

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
// row vector = row vector x A Matrix
// 1 * N      = 1 * K      x K * N
// Current kernel is for row by row, so that A matrix must do transpose in advance
// eq: SetKernelArgs(
//        point of input row vector,
//        point of A matrix,
//        point of output row vector,
//        (int)K,
//        (int)N);

#include "saber/funcs/impl/amd/include/amd_sgemv.h"

namespace anakin {
namespace saber {
bool find_sgemv(int device_id, AMDKernelPtr& kptr, bool trans_A, int k, int n) {
    //support row by row only
    if (trans_A == false) {
        return false;
    }

    KernelInfo kernelInfo;

    kernelInfo.kernel_file = "sgemv.cl";
    kernelInfo.kernel_name = "InnerProduct";
    kernelInfo.wk_dim      = 1;
    kernelInfo.kernel_type = SABER;

    if (k == 128 && n >= 65536) {
        kernelInfo.l_wk        = {64, 1, 1};
        kernelInfo.g_wk        = {64 * 64 * 32, 1, 1};
        kernelInfo.comp_options += std::string(" -DMETHOD=") + std::to_string(2);
    } else {
        kernelInfo.l_wk        = {64, 1, 1};
        kernelInfo.g_wk        = {64 * ((n + 1023) / 1024 * 1024), 1, 1};
        kernelInfo.comp_options += std::string(" -DMETHOD=") + std::to_string(1);
    }

    kptr =  CreateKernel(device_id, &kernelInfo);

    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to create kernel";
        return false;
    }

    return true;
}
}
}
