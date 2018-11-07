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
#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_UTILS_OCLKERNEL_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_UTILS_OCLKERNEL_H

#include "shared_pointer.h"
#include "amd_base.h"
#include <CL/cl.h>
#include <string.h>

namespace anakin {
namespace saber {

using ClProgramPtr = SHARED_OBJ(cl_program);
#define gen_shared_cl_program(t) GEN_SHARED_OBJ_WITH_DELETER(cl_program, clReleaseProgram, t)

using ClKernelPtr = SHARED_OBJ(cl_kernel);
#define gen_shared_cl_kernel(t) GEN_SHARED_OBJ_WITH_DELETER(cl_kernel, clReleaseKernel, t)

class OCLKernel {
public:
    OCLKernel(cl_context ctx, cl_device_id id, anakin::saber::KernelInfo* ki) :
            _context(ctx),
            _device_id(id) {
        if (ki == nullptr)
            throw "KernelInfo can't be null";
        kernel_info = ki;
        kernel_info.print();

        CreateProgram();
        CreateKernel();
        AMD_LOGD("create kernel complete");
    }

    bool isInit() {
        return _kernel == NULL ? false : true;
    }

    cl_program getProgram() {
        return _program.get();
    };
    cl_kernel getKernel() {
        return _kernel.get();
    };
    template <class... Ts>
    bool SetKernelArgs(const Ts&... xs) {
        AMD_LOGD(__func__);
        if (!isInit())
            return false;
        AMD_LOGD("Kernel is init");

        if (!setKernelArgs(0, xs...)) {
            return false;
        }
        AMD_LOGD("Set Kernel Args complete");
        return true;
    }

    bool
    Invoke(cl_command_queue cm, int wait_events_num, const cl_event* wait_events, cl_event* event) {
        AMD_LOGD(__func__);
        if (!isInit())
            return false;
        AMD_LOGD("Kernel is init");
        return run(cm, wait_events_num, wait_events, event);
    }

    template <class... Ts>
    bool
    Invoke(cl_command_queue cm,
           int wait_events_num,
           const cl_event* wait_events,
           cl_event* event,
           const Ts&... xs) {
        AMD_LOGD(__func__);
        if (!isInit())
            return false;
        AMD_LOGD("Kernel is init");

        if (!setKernelArgs(0, xs...)) {
            return false;
        }
        AMD_LOGD("Set Kernel Args complete");

        return run(cm, wait_events_num, wait_events, event);
    }

    std::string GetName();

    ~OCLKernel() {}

private:
    template <class T>
    bool setKernelArgs(int index, const T& x) {

        cl_int errNum = clSetKernelArg(_kernel.get(), index, sizeof(T), &x);
        if (errNum != CL_SUCCESS) {

            if (errNum == CL_INVALID_ARG_INDEX) // workaround for miopengemm kenrel
            {
                AMD_LOGE("set kernel args[" << index << "] err = " << errNum);
                return true;
            }
            AMD_LOGE("set kernel args[" << index << "] err = " << errNum);
            return false;
        }
        AMD_LOGD("Set Kernel Args[" << index << "]");
        return true;
    }
    template <class T, class... Ts>
    bool setKernelArgs(int index, const T& x, const Ts&... xs) {
        if (!setKernelArgs(index, x))
            return false;
        return setKernelArgs(index + 1, xs...);
    }

    bool
    run(cl_command_queue cm, int wait_events_num, const cl_event* wait_events, cl_event* event);

    void CreateProgram();
    void CreateKernel();

    ClProgramPtr _program;
    ClKernelPtr _kernel;
    cl_context _context;
    cl_device_id _device_id;
    KernelInfo kernel_info;
};

using OCLKernelPtr = SHARED_OBJ(OCLKernel);
#define gen_shared_ocl(t) GEN_SHARED_OBJ(OCLKernel, t)

} // namespace saber
} // namespace anakin
#endif
