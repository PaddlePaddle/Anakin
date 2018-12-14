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
#include "ocl_kernel.h"
#include "amd_cache.h"
#include "amd_file_utils.h"
#include "amd_logger.h"

#include <miopen/gcn_asm_utils.hpp>
#include <miopen/kernel.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/kernel_warnings.hpp>

namespace anakin {
namespace saber {
//#define ENABLE_LOG
#define SAVE_TO_FILE
#define MAX_LOG_LENGTH 65535
extern std::string GetKernelSrc(std::string name);

std::string GetDeviceName(cl_device_id device_id) {

    char deviceName[100];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "Device Name: " << deviceName;
    return std::string(deviceName);
}

#if 1

void SaveProgramBinary(const cl_program program, const std::string& name) {
    size_t binary_size;
    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, nullptr);

    std::vector<char> binary(binary_size);
    char* src[1] = {binary.data()};
    clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(src), &src, nullptr);

    std::ofstream fout(name.c_str(), std::ios::out | std::ios::binary);
    fout.write(binary.data(), binary.size());

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "save program to cache file: " << name;
}

void WriteProgramToFile(cl_program cl_prg, cl_device_id device_id, KernelInfo* ki) {
#ifndef SAVE_TO_FILE

    if (true) {
        return;
    }

#endif

    std::string kernelKey;

    if (ki->kernel_type == SOURCE) {
        return;
    } else {
        kernelKey = ki->kernel_file;
    }

    std::string deviceName = GetDeviceName(device_id);
    auto path              = GetCachePath() + unique_path();
    SaveProgramBinary(cl_prg, path);
    SaveBinary(path, deviceName, kernelKey, ki->comp_options);
}

cl_program LoadBinaryProgram(
    cl_context context,
    cl_device_id device_id,
    const char* source_data,
    size_t size) {
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << __func__;
    cl_int errNum;
    cl_program program = clCreateProgramWithBinary(
                             context, 1, &device_id, &size, (const unsigned char**)&source_data, NULL, &errNum);

    if (errNum != CL_SUCCESS) {
        LOG(ERROR) << __func__ << " error(" << errNum << ")";
    }

    return program;
}

cl_program LoadProgramFromFileCache(cl_context context, cl_device_id device_id, KernelInfo* ki) {
#ifndef SAVE_TO_FILE

    if (true) {
        return NULL;
    }

#endif

    std::string kernelKey;

    if (ki->kernel_type == SOURCE) {
        return NULL;
    } else {
        kernelKey = ki->kernel_file;
    }

    std::string deviceName = GetDeviceName(device_id);

    std::string cacheFilePath = LoadBinaryPath(deviceName.c_str(), kernelKey, ki->comp_options);

    if (!cacheFilePath.empty()) {
        std::string source = LoadFile(cacheFilePath);

        cl_program program = LoadBinaryProgram(context, device_id, source.data(), source.size());
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "Create CL program from cache file: " << kernelKey;
        return program;
    }

    return NULL;
}

ClProgramPtr LoadProgramFromMemCache(cl_context context, KernelInfo* ki) {
    ProgramCache* programCache = ProgramCache::getInstance();

    std::string kernelKey;

    if (ki->kernel_type == SOURCE) {
        return NULL;
    } else {
        kernelKey.assign(ki->kernel_file + ki->comp_options + ":" + ki->kernel_name);
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << __func__ << " " << kernelKey;

    std::string progKey(kernelKey);
    // Consider different compile options for single program
    progKey += ki->comp_options;

    // get program from program cache
    auto program_ptr = programCache->lookup(std::pair<cl_context, std::string>(context, progKey));

    if (program_ptr != NULL && program_ptr.get() != NULL) {
        return program_ptr;
    }

    return NULL;
}

void WriteProgramIntoMemCache(cl_context context, ClProgramPtr program, KernelInfo* ki) {
    ProgramCache* programCache = ProgramCache::getInstance();

    std::string kernelKey;

    if (ki->kernel_type == SOURCE) {
        return NULL;
    } else {
        kernelKey.assign(ki->kernel_file + ki->comp_options + ":" + ki->kernel_name);
    }

    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << __func__ << " " << kernelKey;

    std::string progKey(kernelKey);
    // Consider different compile options for single program
    progKey += ki->comp_options;

    // Save to Program Cache
    if (programCache->getSize() < programCache->MAX_CACHE_SIZE) {
        programCache->add(std::pair<cl_context, std::string>(context, progKey), program);
    } else {
        LOG(INFO) << "Warning: program code cache has been full.\n";
    }
}
bool BuildProgram(cl_program program, cl_device_id device_id, KernelInfo* ki) {

    if (program == NULL) {
        LOG(ERROR) << "Failed to Build Program, cl_program is not initialized.";
        return false;
    }

    cl_int errNum;
    auto is_asm = miopen::EndsWith(ki->kernel_file, ".s");

    if (is_asm) {
        clBuildProgram(program, 1, &device_id, "", NULL, NULL);
    } else {
        errNum = clBuildProgram(program, 1, &device_id, ki->comp_options.c_str(), NULL, NULL);
    }

    if (errNum != CL_SUCCESS) {
        char buildErrLog[MAX_LOG_LENGTH];
        clGetProgramBuildInfo(
            program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buildErrLog), buildErrLog, NULL);

        LOG(ERROR) << "CL program build error log in kernel: " << buildErrLog;
        return true;
    }

    return true;
}

cl_program CreateProgramFromSource(cl_context context, cl_device_id device_id, KernelInfo* ki) {

    cl_program program = NULL;
    bool is_binary     = false;
    std::string source;

    if (ki->kernel_type == SOURCE) {
        source = ki->kernel_file;
    } else {
        try {
            if (ki->kernel_type == MIOPEN) {
                source = miopen::GetKernelSrc(ki->kernel_file);
            } else {
                source = GetKernelSrc(ki->kernel_file);
            }
        } catch (...) {
            LOG(ERROR) << "Can't Load CL Program";
            return NULL;
        }

        if (miopen::EndsWith(ki->kernel_file, ".s") || miopen::EndsWith(ki->kernel_file, ".so")) {
            is_binary = true;
        }

        if (!source.empty()) {
            auto is_asm = miopen::EndsWith(ki->kernel_file, ".s");

            if (is_asm) {
                std::string deviceName = GetDeviceName(device_id);
                AmdgcnAssemble(
                    source, std::string(" -mcpu=") + deviceName + " " + ki->comp_options);
            }
        } else {
            std::ifstream kFile(ki->kernel_file, std::ios::in);

            if (!kFile.is_open()) {
                LOG(ERROR) << "Failed to open file for reading: " << ki->kernel_file;
                return NULL;
            }

            source = std::string(
                         (std::istreambuf_iterator<char>(kFile)), std::istreambuf_iterator<char>());
            kFile.close();
        }
    }

    if (source.empty()) {
        return NULL;
    }

    if (is_binary) {
        program = LoadBinaryProgram(context, device_id, source.data(), source.size());
    } else {
        LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << "createPrograWithSource";
        std::string params = ki->comp_options;
#if defined(ENABLE_DEBUG) || defined(ENABLE_LOG)
        params += " -Werror";
#ifdef __linux__
        params += miopen::KernelWarningsString();
#endif
#endif
        params += " -cl-std=CL1.2";

        // load program from header
        const char* srcStr = source.data();
        size_t size        = source.size();

        program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, &size, NULL);

        if (program == NULL) {
            LOG(ERROR) << "Failed to create CL program from header: " << ki->kernel_file;
            return NULL;
        }
    }

    return program;
}

#if 0
ClKernelPtr LoadKernelFromMemCache(cl_program program, KernelInfo* ki) {
    std::string kernelKey;

    if (ki->kernel_type == SOURCE) {
        return NULL;
    } else {
        kernelKey.assign(ki->kernel_file + ki->comp_options + ":" + ki->kernel_name);
    }

    KernelCache* kernelCache = KernelCache::getInstance();
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << __func__ << " " << kernelKey;
    auto kernel_ptr = kernelCache->lookup(std::pair<cl_program, std::string>(program, kernelKey));

    if (kernel_ptr != NULL && kernel_ptr.get() != NULL) {
        return kernel_ptr;
    }

    return NULL;
}

void WriteKernelIntoMemCache(cl_program program, ClKernelPtr kernel, KernelInfo* ki) {
    std::string kernelKey;

    if (ki->kernel_type == SOURCE) {
        return NULL;
    } else {
        kernelKey.assign(ki->kernel_file + ki->comp_options + ":" + ki->kernel_name);
    }

    KernelCache* kernelCache = KernelCache::getInstance();
    LOG_IF_S(INFO, ENABLE_AMD_DEBUG_LOG) << __func__ << " " << kernelKey;

    if (kernelCache->getSize() < kernelCache->MAX_CACHE_SIZE) {
        kernelCache->add(std::pair<cl_program, std::string>(program, kernelKey), kernel);
    } else {
        LOG(INFO) << "Warning: kernel cache has been full.\n";
    }
}
#endif

cl_kernel CreateKernelFromSource(cl_program program, KernelInfo* ki) {
    cl_kernel kernel = NULL;

    kernel = clCreateKernel(program, ki->kernel_name.c_str(), NULL);

    if (kernel == NULL) {
        LOG(ERROR) << "error: failed to create CL kernel.\n";
        return NULL;
    }

    return kernel;
}

ClProgramPtr CreateProgram(cl_context context, cl_device_id device_id, KernelInfo* kernel_info) {

    ClProgramPtr program_ptr = LoadProgramFromMemCache(context, kernel_info);

    if (program_ptr != NULL) {
        return program_ptr;
    }

    bool is_load_from_file = true;
    cl_program program     = LoadProgramFromFileCache(context, device_id, kernel_info);

    if (program == NULL) {
        program           = CreateProgramFromSource(context, device_id, kernel_info);
        is_load_from_file = false;
    }

    if (!BuildProgram(program, device_id, kernel_info)) {
        program = NULL;
        return NULL;
    }

    program_ptr = gen_shared_cl_program(program);
    WriteProgramIntoMemCache(context, program_ptr, kernel_info);

    if (!is_load_from_file) {
        WriteProgramToFile(program, device_id, kernel_info);
    }

    return program_ptr;
}

ClKernelPtr CreateKernel(cl_program program, KernelInfo* kernel_info) {

#if 0
    //Kernel cache cannot work at multi-thread scenario
    ClKernelPtr kernel_ptr = LoadKernelFromMemCache(program, kernel_info);

    if (kernel_ptr != NULL) {
        return kernel_ptr;
    }

#endif

    cl_kernel kernel = CreateKernelFromSource(program, kernel_info);

    if (kernel == NULL) {
        return NULL;
    }

    ClKernelPtr kernel_ptr = gen_shared_cl_kernel(kernel);
    //WriteKernelIntoMemCache(program, kernel_ptr, kernel_info);

    return kernel_ptr;
}

#endif
#if 1
void OCLKernel::CreateProgram() {
    _program = anakin::saber::CreateProgram(_context, _device_id, &kernel_info);
}

void OCLKernel::CreateKernel() {
    _kernel = anakin::saber::CreateKernel(_program.get(), &kernel_info);
}

bool OCLKernel::run(
    cl_command_queue cm,
    int wait_events_num,
    const cl_event* wait_events,
    cl_event* event) {

    cl_int errNum = clEnqueueNDRangeKernel(
                        cm,
                        _kernel.get(),
                        kernel_info.wk_dim,
                        (kernel_info.g_wk_offset.size() > 0 ? kernel_info.g_wk_offset.data() : NULL),
                        kernel_info.g_wk.data(),
                        kernel_info.l_wk.data(),
                        wait_events_num,
                        wait_events,
                        event);

    if (errNum != CL_SUCCESS) {
        LOG(ERROR) << "Fail to set execution: " << errNum;
        kernel_info.printE();
        return false;
    }

    return true;
}

std::string OCLKernel::GetName() {
    return kernel_info.kernel_name;
}

#endif

} // namespace saber
} // namespace anakin
