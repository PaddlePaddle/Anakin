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

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#define MLO_POOLING_OP_MAX 0
#define MLO_POOLING_OP_AVE 1

namespace anakin {
namespace saber {

typedef struct ExtSolutionConfigTpye
{
    int in_tile0, in_tile1;
    int grp_tile0, grp_tile1;
    int out_pix_tile0, out_pix_tile1;
    int n_stacks;
    int n_out_pix_tiles;
    int n_out_tiles_perstack;
    int n_in_data_tiles;
    int n_read_procs;
    int alu_tile0, alu_tile1;
    int horiz_out_pix; 
    int vert_out_pix; 
}T_ExtSolutionConfig;

struct KernelInfo
{
    std::string comp_options;
    std::vector<size_t> l_wk;
    std::vector<size_t> g_wk;
    std::string kernel_file;
    std::string kernel_name;
    friend std::ostream& operator<<(std::ostream& os, const KernelInfo& k);
};

extern cl_program CreateCLProgram(cl_context context, cl_device_id device, const char* fileName, KernelInfo* ki=NULL);
extern cl_program CreatProgramFromBinaryFile(cl_context context, cl_device_id device, const char* binFile);


inline cl_int _setKernelArgs(cl_kernel &k,int i){ return CL_SUCCESS;}

template<typename T, typename... Args>
inline cl_int _setKernelArgs(cl_kernel &kernel,int i, const T &firstParameter, const Args& ...restOfParameters){
    return clSetKernelArg(kernel, i, sizeof(firstParameter), &firstParameter) | \
           _setKernelArgs(kernel,i+1,restOfParameters...);
}

template<typename... Args>
inline cl_int setKernelArgs(cl_kernel &kernel, const Args& ...args){
    return _setKernelArgs(kernel, 0, args...);
}
}
}
#endif
