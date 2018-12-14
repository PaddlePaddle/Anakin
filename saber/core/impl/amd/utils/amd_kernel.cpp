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
#include "amd_kernel.h"
#include "saber/core/env.h"
#include "saber/core/device.h"

namespace anakin {
namespace saber {

#ifdef USE_OPENCL
AMDKernelPtr CreateKernel(int device_id, KernelInfo* ki) {
    cl_context context  = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[device_id]; // anakin device id to AMD device
    device          = dev.get_device();
    context         = dev.get_context();

    return gen_shared_ocl(new OCLKernel(context, device, ki));
}

bool LaunchKernel(AMDStream_t stream, amd_kernel_list kernels, bool sync) {
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << __func__;

    float exec_time_ms = 0;

    for (amd_kernel_list::iterator it = kernels.begin(); it != kernels.end(); it++) {
        cl_event event;
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << __func__ << " E";

        if (!it->get()->Invoke(stream, 0, NULL, (sync ? &event : NULL))) {
            LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << __func__ << " Failed";
            return false;
        }

        if (sync) {
            TargetWrapper<AMD>::sync_event(event);
#ifdef ENABLE_AMD_PROFILING
            cl_ulong start, end;
            clGetEventProfilingInfo(
                event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            exec_time_ms = (end - start) * 1e-6;
#endif
        }

        if (sync) {
            TargetWrapper<AMD>::destroy_event(event);
        }

        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << __func__ << " X : " << exec_time_ms << " ms";
    }

    return true;
}
#endif

} // namespace saber
} // namespace anakin
