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
#include "saber/funcs/impl/amd/amd_utils.h"
#include "utils/logger/logger.h"

namespace anakin {
namespace saber {
#define MAX_LOG_LENGTH 65535
cl_program CreateCLProgram(cl_context context, cl_device_id device, const char* fileName,
                           KernelInfo* ki) {
    cl_int errNum;
    cl_program program;

    std::ifstream kFile(fileName, std::ios::in);

    if (!kFile.is_open()) {
        LOG(ERROR) << "Failed to open file for reading: " << fileName;
        return NULL;
    }

    std::string src(
        (std::istreambuf_iterator<char>(kFile)),
        std::istreambuf_iterator<char>()
    );
    char* srcStr = src.c_str();
    program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, NULL);

    kFile.close();

    if (program == NULL) {
        LOG(ERROR) << "Failed to create CL program with source file.";
        return NULL;
    }

    char* comp_options = NULL;

    if (ki != NULL) {
        comp_options = ki->comp_options.c_str();
    }

    errNum = clBuildProgram(program, 1, &device, comp_options, NULL, NULL);

    if (errNum != CL_SUCCESS) {
        char buildErrLog[MAX_LOG_LENGTH];
        clGetProgramBuildInfo(program,
                              device,
                              CL_PROGRAM_BUILD_LOG,
                              sizeof(buildErrLog),
                              buildErrLog,
                              NULL);

        LOG(ERROR) << "CL program build error log in kernel: " << buildErrLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
};

cl_program CreatProgramFromBinaryFile(cl_context context, cl_device_id device,
                                      const char* binFile) {
    cl_program program;
    cl_int errNum;

    FILE* fp = fopen(binFile, "rb");

    if (fp == NULL) {
        LOG(ERROR) << "Can't open bin file: " <<  std::string(binFile);
        return NULL;
    }

    size_t binSize;
    fseek(fp, 0, SEEK_END);
    binSize = ftell(fp);
    rewind(fp);

    unsigned char* binProgram = new unsigned char[binSize];
    fread(binProgram, 1, binSize, fp);
    fclose(fp);

    program = clCreateProgramWithBinary(context, 1, &device, &binSize,
                                        (const unsigned char**)&binProgram, NULL, &errNum);
    errNum = clBuildProgram(program, 1, &device, "", NULL, NULL);

    delete[] binProgram;

    if (errNum != CL_SUCCESS) {
        char buildErrLog[MAX_LOG_LENGTH];
        clGetProgramBuildInfo(program,
                              device,
                              CL_PROGRAM_BUILD_LOG,
                              sizeof(buildErrLog),
                              buildErrLog,
                              NULL);

        LOG(ERROR) << "CL program build error log in kernel: " << buildErrLog;
        clReleaseProgram(program);

        return NULL;
    }

    return program;
};


}
}
