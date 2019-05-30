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
#ifndef ANAKIN_SABER_FUNC_IMPL_AMD_SGEMV_H
#define ANAKIN_SABER_FUNC_IMPL_AMD_SGEMV_H

#include <CL/cl.h>
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"
#include "saber/funcs/base.h"

namespace anakin {
namespace saber {
bool find_sgemv(int device_id, AMDKernelPtr& kptr, bool trans_A, int k, int n);
}
}
#endif
